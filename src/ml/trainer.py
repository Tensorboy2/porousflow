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
            'R2_test': []
        }

        # gradient clipping
        self.clip_grad = config.get('clip_grad', False)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # lr scheduler
        warmup_steps = config.get('warmup_steps', 0)
        total_steps = config.get('total_steps', config.get('num_epochs', 10) * len(train_loader))
        decay = config.get('decay', 'linear')  # 'linear' or 'cosine'
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1)/ warmup_steps
            else:
                decay_epochs = total_steps - warmup_steps
                decay_progress = (step - warmup_steps) / decay_epochs
                if decay=="linear":
                    return max(0.0, 1.0 - decay_progress)
                elif decay=="cosine":
                    return 0.5 * (1 + math.cos(math.pi * decay_progress))
                else:
                    return 1
                
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Scaler to Half precision:
        use_amp = config.get('use_amp', False)
        self.scaler = GradScaler(enabled=use_amp)    

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        # preds = []
        # trues = []
        for inputs, targets in self.train_loader:
            # Move data to device and handle pin_memory if specified
            if self.config['pin_memory']:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            # Move data to device without pin_memory
            else:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision context
            with autocast(device_type= self.device,enabled=self.scaler.is_enabled()):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

            self.scaler.scale(loss).backward()

            if self.clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

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

        self.metrics['R2_train'].append(r2)
        self.metrics['train_loss'].append(epoch_loss)
        
        return epoch_loss
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        # preds = []
        # trues = []
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                # preds.append(outputs.detach().cpu())
                # trues.append(targets.detach().cpu())
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                
                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

        epoch_loss = running_loss / len(self.val_loader.dataset)
        
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        r2 = self.R2_score(trues, preds)

        self.metrics['R2_val'].append(r2)
        self.metrics['val_loss'].append(epoch_loss)
        
        return epoch_loss
    
    def test(self):
        self.model.eval()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        # preds = []
        # trues = []
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                preds.append(outputs.detach().cpu())
                trues.append(targets.detach().cpu())
        
        epoch_loss = running_loss / len(self.test_loader.dataset)
        
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        r2 = self.R2_score(trues, preds)

        self.metrics['R2_test'].append(r2)
        self.metrics['test_loss'].append(epoch_loss)
        
        return epoch_loss
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(self.config.get('save_model_path', '.'), 'best_model.pth'))
        #save last model
        self.save_model(os.path.join(self.config.get('save_model_path', '.'), 'last_model.pth'))
        self.save_metrics(os.path.join(self.config.get('save_model_path', '.'), 'metrics.zarr'))
        # test_loss = self.test()
        # print(f"Final Test Loss: {test_loss:.4f}")

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