'''
Docstring for ml.train

This module implements the training loop for the permeability and dispersion models.
It is independent of the data loading and model definition modules.
'''
import torch
import torch.nn as nn
import math
import os
import zarr
import numpy as np
from torch.amp import autocast, GradScaler
import time
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # Added for numerical stability
    
    def forward(self, x, y):
        # Calculate MSE and take the square root
        loss = torch.sqrt(self.mse(x, y) + self.eps)
        return loss
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, config,criterion=nn.MSELoss()):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.config = config

        self.a = torch.sinh(torch.ones(1))/9299.419921875
        
        # training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'R2_train': [],
            'R2_val': [],
            'R2_test': [],
            'grad_norm': [],
            'grad_norm_clipped': [],  # NEW: track how often clipping occurs
            # 'test_pred': []
        }

        # gradient clipping
        self.clip_grad = config.get('clip_grad', True)
        self.max_grad_norm = config.get('max_grad_norm', 5.0)

        # lr scheduler
        warmup_steps = config.get('warmup_steps', 0)
        total_steps = config.get('total_steps', config.get('num_epochs', 10) * len(train_loader))
        decay = config.get('decay', '')  # 'linear' or 'cosine'
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1)/ warmup_steps
            else:
                # decay_epochs = total_steps - warmup_steps
                decay_progress = (step - warmup_steps) / total_steps
                if decay=="linear":
                    return max(0.0, 1.0 - decay_progress)
                elif decay=="cosine":
                    return 0.5 * (1 + math.cos(math.pi * decay_progress))
                else:
                    return 1
                
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Scaler to Half precision:
        use_amp = config.get('use_amp', True)
        self.scaler = GradScaler(enabled=use_amp,device='cuda' if torch.cuda.is_available() else 'cpu')  


        # Deal with output directories:
        save_path = config.get('save_model_path', 'results')
        os.makedirs(save_path, exist_ok=True)
        self.model_name = config['model'].get('name', 'model')

        # Pe values used by the dispersion model. Ensure proper dtype/device.
        self.Pes = torch.tensor([0.1, 10, 50, 100, 500], dtype=torch.float32, device=self.device)

        task = config.get('task','permeability')
        if task == 'dispersion':
            self.train_epoch = self.train_epoch_dispersion
            self.validate_epoch = self.validate_dispersion
            self.test_epoch = self.test_dispersion
        else:
            self.train_epoch = self.train_epoch_permeability
            self.validate_epoch = self.validate_permeability
            self.test_epoch = self.test_permeability
    
    # '''
    # Defining training methods:
    # '''
    def train_epoch_permeability(self):
        self.model.train()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        gradient_norm = 0.0
        num_clipped = 0  # NEW: count how many batches were clipped
        count = 0
        # preds = []
        # trues = []
        for inputs, targets in self.train_loader:
            # Move data to device and handle pin_memory if specified
            if self.config.get('pin_memory', False):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            # Move data to device without pin_memory
            else:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision context
            with autocast(device_type= 'cuda',enabled=self.scaler.is_enabled()):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

            self.scaler.scale(loss).backward()

            if self.clip_grad:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                gradient_norm += grad_norm.item()  # FIXED: convert to item()
                if grad_norm > self.max_grad_norm:  # NEW: track when clipping occurs
                    num_clipped += 1

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # preds.append(outputs.detach().cpu())
            # trues.append(targets.detach().cpu())

            # Accumulate R2 components (on CPU to save GPU memory)
            with torch.no_grad():
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                
                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

            self.scheduler.step()
            
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        
        # preds = torch.cat(preds, dim=0)
        # trues = torch.cat(trues, dim=0)
        # r2 = self.R2_score(trues, preds)
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )

        avg_grad_norm = gradient_norm / len(self.train_loader)
        clip_percentage = 100 * num_clipped / len(self.train_loader) if len(self.train_loader) > 0 else 0
        
        self.metrics['R2_train'].append(r2)
        self.metrics['train_loss'].append(epoch_loss)
        self.metrics['grad_norm'].append(avg_grad_norm)
        self.metrics['grad_norm_clipped'].append(clip_percentage)  # NEW: percentage of batches clipped
        
        return epoch_loss, r2, avg_grad_norm
    
    def scale(self,x):
        return torch.asinh(x)
        # return torch.sign(x)*torch.log(torch.abs(x)+1)
    
    def inverse_scale(self,y):
        # return torch.sign(y) * (torch.exp(torch.abs(y)) - 1)
        return torch.sinh(y)

    def train_epoch_dispersion(self):
        self.model.train()

        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        gradient_norm = 0.0
        num_clipped = 0  # NEW: count how many batches were clipped

        total_samples = 0
        count = 0
        grad_steps = 0

        use_amp = self.scaler.is_enabled()
        # i = 0
        for Batch in self.train_loader:
            # i+=1
            # print(f'{i}/{len(self.train_loader)}')
            if self.config['pe']['include_direction']:
                inputs, D, Pe, Direction = Batch
            else:
                inputs, D, Pe = Batch


            B = inputs.shape[0]

            if self.config.get('pin_memory', False):
                inputs = inputs.to(self.device, non_blocking=True)
                D = D.to(self.device, non_blocking=True)
                Pe = Pe.to(self.device, non_blocking=True)
                if self.config['pe']['include_direction']:
                    Direction = Direction.to(self.device, non_blocking=True)
            else:
                inputs = inputs.to(self.device)
                D = D.to(self.device)
                Pe = Pe.to(self.device)
                # Direction = Direction.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (only the model under autocast)
            if self.config['pe']['include_direction']:
                with autocast(device_type='cuda', enabled=use_amp):
                    outputs = self.model(inputs, Pe, Direction)
            else:
                with autocast(device_type='cuda', enabled=use_amp):
                    outputs = self.model(inputs, Pe)

            # Compute loss in FP32 (outside autocast) to ensure stable scaling
            outputs_fp32 = outputs.float()
            outputs = self.inverse_scale(outputs)
            D_fp32 = D.float()
            # a = self.a.to(self.device)
            # scaled_outputs = torch.arcsinh(outputs_fp32*a)
            # scaled_D = torch.arcsinh(D_fp32*a)

            # scaled_outputs = torch.sign(outputs_fp32) * torch.sqrt(torch.abs(outputs_fp32) + 1e-8)
            # scaled_D = torch.sign(D_fp32) * torch.sqrt(torch.abs(D_fp32) + 1e-8)

            # scaled_outputs = self.scale(outputs_fp32)
            # scaled_outputs = self.scale(scaled_outputs)
            scaled_D = self.scale(D_fp32)
            # scaled_D = self.scale(scaled_D)
            # scaled_outputs = torch.sign(outputs_fp32) * torch.log1p(torch.abs(outputs_fp32)/100)
            # scaled_D = torch.sign(D_fp32) * torch.log1p(torch.abs(D_fp32)/100)

            loss = self.criterion(outputs_fp32, scaled_D)
            # loss = self.criterion(outputs_fp32, D_fp32)
            running_loss += loss.item() * B
            total_samples += B

            # Backward
            if use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient clipping
            if self.clip_grad:
                if use_amp:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                gradient_norm += grad_norm.item()  # FIXED: convert to item()
                if grad_norm > self.max_grad_norm:  # NEW: track when clipping occurs
                    num_clipped += 1

            # Optimizer step
            if use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            grad_steps += 1

            # Metrics (detach safely after backward)
            with torch.no_grad():
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = D.detach().cpu()

                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

            self.scheduler.step()

        epoch_loss = running_loss / len(self.train_loader.dataset)

        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )

        avg_grad_norm = gradient_norm / len(self.train_loader)
        clip_percentage = 100 * num_clipped / len(self.train_loader) if len(self.train_loader) > 0 else 0

        self.metrics['train_loss'].append(epoch_loss)
        self.metrics['R2_train'].append(r2)
        self.metrics['grad_norm'].append(avg_grad_norm)
        self.metrics['grad_norm_clipped'].append(clip_percentage)  # NEW: percentage of batches clipped

        return epoch_loss, r2, avg_grad_norm
    
    def validate_permeability(self):
        self.model.eval()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                # outputs = torch.arcsinh(0.2*outputs)
                # D = torch.arcsinh(0.2*D)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                
                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

        epoch_loss = running_loss / len(self.val_loader.dataset)
        r2 = self._compute_r2_from_accumulators(sum_squared_error, sum_targets, sum_targets_squared, count)

        self.metrics['R2_val'].append(r2)
        self.metrics['val_loss'].append(epoch_loss)

        return epoch_loss,r2
    
    def validate_dispersion(self):
        """
        Validation for dispersion task.

        Returns:
            epoch_loss (float): mean validation loss
            r2 (float): R^2 score over validation set
        """
        self.model.eval()

        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        total_samples = 0

        with torch.no_grad():
            for Batch in self.val_loader:
                # Support datasets that yield either (inputs, D) or (inputs, D, Pe)
                if self.config['pe']['include_direction']:
                    inputs, D, Pe, Direction = Batch
                    inputs = inputs.to(self.device, non_blocking=True)
                    D = D.to(self.device, non_blocking=True)
                    Pe = Pe.to(self.device, non_blocking=True)
                    Direction = Direction.to(self.device, non_blocking=True)
                    outputs = self.model(inputs, Pe, Direction)
                else:
                    inputs, D, Pe = Batch
                    inputs = inputs.to(self.device, non_blocking=True)
                    D = D.to(self.device, non_blocking=True)
                    Pe = Pe.to(self.device, non_blocking=True)
                    outputs = self.model(inputs, Pe)

                # Apply arcsinh transform consistently with training
                # a = self.a.to(self.device)
                # scaled_outputs = torch.arcsinh(a*outputs.float())
                # scaled_D = torch.arcsinh(a*D.float())
                # scaled_outputs = torch.sign(outputs)*torch.log1p(torch.abs(outputs)/100)
                # scaled_D = torch.sign(D) * torch.log1p(torch.abs(D)/100)
                # scaled_outputs = self.scale(outputs)
                # scaled_outputs = self.scale(scaled_outputs)

                scaled_D = self.scale(D)
                # scaled_D = self.scale(scaled_D)
                loss = self.criterion(outputs, scaled_D)
                # loss = self.criterion(outputs, D)
                outputs = self.inverse_scale(outputs)

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Metrics in original space
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = D.detach().cpu()

                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )

        self.metrics['val_loss'].append(epoch_loss)
        self.metrics['R2_val'].append(r2)

        return epoch_loss, r2
    
    def test_permeability(self):
        self.model.eval()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()
        epoch_loss = running_loss / len(self.test_loader.dataset)
        
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )
        self.metrics['R2_test'].append(r2)
        self.metrics['test_loss'].append(epoch_loss)
        
        return epoch_loss, r2
    
    def test_dispersion(self):
        self.model.eval()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                B, Pe, _ = targets.shape
                for i in range(Pe):
                    D = targets[:, i]
                    inputs, D = inputs.to(self.device), D.to(self.device)
                    outputs = self.model(inputs, self.Pes[i])
                    outputs_scaled = torch.arcsinh(outputs)
                    outputs_scaled = torch.arcsinh(outputs.float())
                    D_scaled = torch.arcsinh(D.float())
                    loss = self.criterion(outputs_scaled, D_scaled)
                    running_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)

                    outputs_cpu = outputs.detach().cpu()
                    targets_cpu = D.detach().cpu()
                    sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                    sum_targets += torch.sum(targets_cpu).item()
                    sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                    count += targets_cpu.numel()
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )
        self.metrics['R2_test'].append(r2)
        self.metrics['test_loss'].append(epoch_loss)
        
        return epoch_loss, r2
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        save_path = os.path.join(self.config.get('save_model_path', 'results'), f"{self.config['model']['name']}_lr-{self.config['learning_rate']}_wd-{self.config['weight_decay']}_bs-{self.config['batch_size']}_epochs-{num_epochs}_{self.config.get('decay','no-decay')}_warmup-{self.config.get('warmup_steps',0)}_clipgrad-{self.config.get('clip_grad',False)}_pe-encoder-{self.config.get('pe_encoder',None)}_pe-{self.config.get('Pe',None)}_{self.config.get('loss_function','mse')}")
        
        print(f'Saving state-dicts to: {save_path}.pth and {save_path}_last_model.pth')
        print(f'Saving metrics to: {save_path}_metrics.zarr')
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} started")

            # ---- TRAIN ----
            train_loss, train_r2, grad_norm = self.train_epoch()
            clip_pct = self.metrics['grad_norm_clipped'][-1]  # Get the last recorded clip percentage
            print(
                f"  Train done | "
                f"loss: {train_loss:.5f} | "
                f"R2: {train_r2:.5f} | "
                f"grad‖: {grad_norm:.5e} | "
                f"clipped: {clip_pct:.1f}%"  # NEW: show clipping percentage
            )

            # ---- VALIDATION ----
            val_loss, val_r2 = self.validate_epoch()
            print(
                f"  Val   done | "
                f"loss: {val_loss:.5f} | "
                f"R2: {val_r2:.5f}"
            )

            # ---- LR ----
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"  LR: {current_lr:.6e}")

            # ---- CHECKPOINT ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path + ".pth")
                print(f"    New best model saved (val loss = {val_loss:.5f})")
        #save last model
        self.save_model(save_path+"_last_model.pth")
        self.save_metrics(save_path+'_metrics.zarr')
        # test_loss, test_r2 = self.test()
        # print(f"Final Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def save_metrics(self, path):
        '''
        Methods for saving training metrics as zarr files.
        
        :param self: 
        :param path: Path to save the metrics.
        :return: None
        '''
        root = zarr.open(path, mode='w')
        for key, values in self.metrics.items():
            root.create_dataset(name=key, data=np.array(values), dtype='f4')

    def R2_score(self, targets, outputs):
        '''
        Docstring for R2_score
        
        :param self:
        :param targets: Ground truth values.
        :param outputs: Model predictions.
        :return: R² score as a float tensor.
        '''
        ss_res = torch.sum((targets - outputs) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def _compute_r2_from_accumulators(self, sum_squared_error, sum_targets, sum_targets_squared, count):
        '''
        Compute R2 score from accumulated statistics.
        
        :param sum_squared_error: Sum of squared errors (residuals)
        :param sum_targets: Sum of target values
        :param sum_targets_squared: Sum of squared target values
        :param count: Total number of elements
        :return: R2 score as a float
        '''
        mean_target = sum_targets / count
        ss_res = sum_squared_error
        ss_tot = sum_targets_squared - count * (mean_target ** 2)
        
        # Handle edge case where variance is zero
        if ss_tot == 0:
            return 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
if __name__ == '__main__':
    # dummy test code with example data and model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    from torch.utils.data import DataLoader, TensorDataset
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    trainer = Trainer(model, train_loader, None, None, optimizer, torch.device('cpu'), {'pin_memory': False})
    trainer.train(num_epochs=5)
    trainer.save_model('model.pth')
    trainer.save_metrics('metrics.zarr')