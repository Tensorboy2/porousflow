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

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer
        self.device = device

        self.config = config

        
        # training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'R2_train': [],
            'R2_val': [],
            'R2_test': [],
            'grad_norm': [],
            # 'test_pred': []
        }

        # gradient clipping
        self.clip_grad = config.get('clip_grad', False)
        self.max_grad_norm = config.get('max_grad_norm', 10.0)

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
        use_amp = config.get('use_amp', False)
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
                gradient_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

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

        grad_norm = gradient_norm / len(self.train_loader)
        self.metrics['R2_train'].append(r2)
        self.metrics['train_loss'].append(epoch_loss)
        self.metrics['grad_norm'].append(grad_norm)
        
        return epoch_loss, r2, grad_norm
    
    def train_epoch_dispersion(self):
        self.model.train()

        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        gradient_norm = 0.0

        total_samples = 0
        count = 0
        grad_steps = 0

        use_amp = self.scaler.is_enabled()
        # i = 0
        for inputs, D, Pe in self.train_loader:
            # i+=1
            # print(f'{i}/{len(self.train_loader)}')
            B = inputs.shape[0]

            if self.config.get('pin_memory', False):
                inputs = inputs.to(self.device, non_blocking=True)
                D = D.to(self.device, non_blocking=True)
                Pe = Pe.to(self.device, non_blocking=True)
            else:
                inputs = inputs.to(self.device)
                D = D.to(self.device)
                Pe = Pe.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (only the model under autocast)
            with autocast(device_type='cuda', enabled=use_amp):
                outputs = self.model(inputs, Pe)

            # Always do scaling + loss in FP32
            scaled_outputs = torch.arcsinh(outputs)
            scaled_D = torch.arcsinh(D)
            loss = self.criterion(scaled_outputs, scaled_D)

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
                gradient_norm += torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

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

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0

        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )

        grad_norm = gradient_norm / grad_steps if grad_steps!=torch.inf else self.max_grad_norm

        self.metrics['train_loss'].append(epoch_loss)
        self.metrics['R2_train'].append(r2)
        self.metrics['grad_norm'].append(grad_norm)

        return epoch_loss, r2, grad_norm
    
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
            for batch in self.val_loader:
                # Support datasets that yield either (inputs, D) or (inputs, D, Pe)
                if len(batch) == 3:
                    inputs, D, Pe = batch
                    inputs = inputs.to(self.device, non_blocking=True)
                    D = D.to(self.device, non_blocking=True)
                    Pe = Pe.to(self.device, non_blocking=True)
                    outputs = self.model(inputs, Pe)
                else:
                    inputs, D = batch
                    inputs = inputs.to(self.device)
                    D = D.to(self.device)
                    outputs = self.model(inputs)

                # Apply arcsinh transform consistently with training
                scaled_outputs = torch.arcsinh(outputs)
                scaled_D = torch.arcsinh(D)
                loss = self.criterion(scaled_outputs, scaled_D)

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
                    D_scaled = torch.arcsinh(D)
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
        save_path = os.path.join(self.config.get('save_model_path', 'results'), f"{self.config['model']['name']}_lr-{self.config['learning_rate']}_wd-{self.config['weight_decay']}_bs-{self.config['batch_size']}_epochs-{num_epochs}_{self.config.get('decay','no-decay')}_warmup-{self.config.get('warmup_steps',0)}_clipgrad-{self.config.get('clip_grad',False)}_pe-encoder-{self.config.get('pe_encoder',None)}")
        
        print(f'Saving state-dicts to: {save_path}.pth and {save_path}_last_model.pth')
        print(f'Saving metrics to: {save_path}_metrics.zarr')
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} started")

            # ---- TRAIN ----
            train_loss, train_r2, grad_norm = self.train_epoch()
            print(
                f"  Train done | "
                f"loss: {train_loss:.5f} | "
                f"R2: {train_r2:.5f} | "
                f"grad‖: {grad_norm:.5e}"
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