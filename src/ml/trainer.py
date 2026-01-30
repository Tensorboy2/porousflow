'''
Docstring for ml.train

This module implements the training loop for the permeability and dispersion models.
It is independent of the data loading and model definition modules.
Enhanced with comprehensive metric tracking.
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

        self.a = torch.sinh(torch.ones(1))/9299.419921875
        
        # Enhanced training metrics
        self.metrics = {
            # Loss metrics
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            
            # R² metrics
            'R2_train': [],
            'R2_val': [],
            'R2_test': [],
            
            # Gradient metrics
            'grad_norm': [],
            'grad_norm_clipped': [],  # percentage of batches clipped
            'grad_norm_max': [],  # maximum gradient norm in epoch
            'grad_norm_min': [],  # minimum gradient norm in epoch
            
            # Learning rate
            'learning_rate': [],
            
            # Timing
            'epoch_time': [],
            'train_time': [],
            'val_time': [],
            
            # Additional error metrics
            'train_mae': [],  # mean absolute error
            'val_mae': [],
            'test_mae': [],
            'train_rmse': [],  # root mean squared error
            'val_rmse': [],
            'test_rmse': [],
            'train_mape': [],  # mean absolute percentage error
            'val_mape': [],
            'test_mape': [],
            'train_max_error': [],  # maximum absolute error
            'val_max_error': [],
            'test_max_error': [],
            'train_median_error': [],  # median absolute error
            'val_median_error': [],
            'test_median_error': [],
            
            # Model statistics
            'model_param_norm': [],  # L2 norm of model parameters
            'model_param_mean': [],  # mean of model parameters
            'model_param_std': [],  # std of model parameters
            
            # Training dynamics
            'scaler_scale': [],  # GradScaler scale factor (for AMP)
            'samples_per_second': [],  # throughput metric
            
            # Per-layer gradient norms (if enabled)
            'layer_grad_norms': [],  # stores dict per epoch
        }

        # gradient clipping
        self.clip_grad = config.get('clip_grad', True)
        self.max_grad_norm = config.get('max_grad_norm', 5.0)
        self.track_layer_grads = config.get('track_layer_grads', False)  # Can be expensive

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
    
    def _compute_additional_metrics(self, outputs, targets):
        """
        Compute additional error metrics beyond MSE.
        
        Returns dict with: mae, rmse, mape, max_error, median_error
        """
        with torch.no_grad():
            abs_error = torch.abs(outputs - targets)
            
            mae = torch.mean(abs_error).item()
            rmse = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
            
            # MAPE - handle division by zero
            epsilon = 1e-8
            mape = torch.mean(abs_error / (torch.abs(targets) + epsilon)).item() * 100
            
            max_error = torch.max(abs_error).item()
            median_error = torch.median(abs_error).item()
            
            return {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'max_error': max_error,
                'median_error': median_error
            }
    
    def _compute_model_stats(self):
        """Compute statistics about model parameters."""
        with torch.no_grad():
            all_params = torch.cat([p.flatten() for p in self.model.parameters()])
            return {
                'norm': torch.norm(all_params, p=2).item(),
                'mean': torch.mean(all_params).item(),
                'std': torch.std(all_params).item()
            }
    
    def _compute_layer_grad_norms(self):
        """Compute gradient norms for each layer (can be expensive)."""
        layer_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                layer_norms[name] = torch.norm(param.grad, p=2).item()
        return layer_norms
    
    # '''
    # Defining training methods:
    # '''
    def train_epoch_permeability(self):
        self.model.train()
        epoch_start = time.time()
        
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        gradient_norm = 0.0
        num_clipped = 0
        count = 0
        
        # For additional metrics
        all_outputs = []
        all_targets = []
        grad_norms_list = []
        
        for inputs, targets in self.train_loader:
            # Move data to device and handle pin_memory if specified
            if self.config.get('pin_memory', False):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
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
                gradient_norm += grad_norm.item()
                grad_norms_list.append(grad_norm.item())
                if grad_norm > self.max_grad_norm:
                    num_clipped += 1

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate for metrics
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

            # Accumulate R2 components (on CPU to save GPU memory)
            with torch.no_grad():
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                
                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

            self.scheduler.step()
            
        train_time = time.time() - epoch_start
        
        # Concatenate all predictions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        epoch_loss = running_loss / len(self.train_loader.dataset)
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )
        
        # Additional metrics
        additional_metrics = self._compute_additional_metrics(all_outputs, all_targets)
        
        # Gradient metrics
        avg_grad_norm = gradient_norm / len(self.train_loader)
        clip_percentage = 100 * num_clipped / len(self.train_loader) if len(self.train_loader) > 0 else 0
        max_grad_norm = max(grad_norms_list) if grad_norms_list else 0.0
        min_grad_norm = min(grad_norms_list) if grad_norms_list else 0.0
        
        # Model statistics
        model_stats = self._compute_model_stats()
        
        # Throughput
        samples_per_second = len(self.train_loader.dataset) / train_time if train_time > 0 else 0
        
        # Store metrics
        self.metrics['R2_train'].append(r2)
        self.metrics['train_loss'].append(epoch_loss)
        self.metrics['grad_norm'].append(avg_grad_norm)
        self.metrics['grad_norm_clipped'].append(clip_percentage)
        self.metrics['grad_norm_max'].append(max_grad_norm)
        self.metrics['grad_norm_min'].append(min_grad_norm)
        self.metrics['train_mae'].append(additional_metrics['mae'])
        self.metrics['train_rmse'].append(additional_metrics['rmse'])
        self.metrics['train_mape'].append(additional_metrics['mape'])
        self.metrics['train_max_error'].append(additional_metrics['max_error'])
        self.metrics['train_median_error'].append(additional_metrics['median_error'])
        self.metrics['model_param_norm'].append(model_stats['norm'])
        self.metrics['model_param_mean'].append(model_stats['mean'])
        self.metrics['model_param_std'].append(model_stats['std'])
        self.metrics['scaler_scale'].append(self.scaler.get_scale())
        self.metrics['train_time'].append(train_time)
        self.metrics['samples_per_second'].append(samples_per_second)
        self.metrics['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        # Optional: layer-wise gradient norms
        if self.track_layer_grads:
            self.metrics['layer_grad_norms'].append(self._compute_layer_grad_norms())
        
        return epoch_loss, r2, avg_grad_norm, additional_metrics
    
    def scale(self,x):
        return torch.asinh(x)
        # return torch.sign(x)*torch.log(torch.abs(x)+1)
    
    def inverse_scale(self,y):
        # return torch.sign(y) * (torch.exp(torch.abs(y)) - 1)
        return torch.sinh(y)

    def train_epoch_dispersion(self):
        self.model.train()
        epoch_start = time.time()

        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        gradient_norm = 0.0
        num_clipped = 0

        total_samples = 0
        count = 0
        grad_steps = 0

        # For additional metrics
        all_outputs = []
        all_targets = []
        grad_norms_list = []

        use_amp = self.scaler.is_enabled()
        
        for Batch in self.train_loader:
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

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (only the model under autocast)
            if self.config['pe']['include_direction']:
                with autocast(device_type='cuda', enabled=use_amp):
                    outputs = self.model(inputs, Pe, Direction)
            else:
                with autocast(device_type='cuda', enabled=use_amp):
                    outputs = self.model(inputs, Pe)

            # Compute loss in FP32
            outputs_fp32 = outputs.float()
            outputs = self.inverse_scale(outputs)
            D_fp32 = D.float()
            
            scaled_D = self.scale(D_fp32)
            loss = self.criterion(outputs_fp32, scaled_D)
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
                gradient_norm += grad_norm.item()
                grad_norms_list.append(grad_norm.item())
                if grad_norm > self.max_grad_norm:
                    num_clipped += 1

            # Optimizer step
            if use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            grad_steps += 1

            # Accumulate for metrics
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(D.detach().cpu())

            # Metrics (detach safely after backward)
            with torch.no_grad():
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = D.detach().cpu()

                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

            self.scheduler.step()

        train_time = time.time() - epoch_start
        
        # Concatenate all predictions and targets
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )

        # Additional metrics
        additional_metrics = self._compute_additional_metrics(all_outputs, all_targets)
        
        # Gradient metrics
        avg_grad_norm = gradient_norm / len(self.train_loader)
        clip_percentage = 100 * num_clipped / len(self.train_loader) if len(self.train_loader) > 0 else 0
        max_grad_norm = max(grad_norms_list) if grad_norms_list else 0.0
        min_grad_norm = min(grad_norms_list) if grad_norms_list else 0.0
        
        # Model statistics
        model_stats = self._compute_model_stats()
        
        # Throughput
        samples_per_second = len(self.train_loader.dataset) / train_time if train_time > 0 else 0

        # Store metrics
        self.metrics['train_loss'].append(epoch_loss)
        self.metrics['R2_train'].append(r2)
        self.metrics['grad_norm'].append(avg_grad_norm)
        self.metrics['grad_norm_clipped'].append(clip_percentage)
        self.metrics['grad_norm_max'].append(max_grad_norm)
        self.metrics['grad_norm_min'].append(min_grad_norm)
        self.metrics['train_mae'].append(additional_metrics['mae'])
        self.metrics['train_rmse'].append(additional_metrics['rmse'])
        self.metrics['train_mape'].append(additional_metrics['mape'])
        self.metrics['train_max_error'].append(additional_metrics['max_error'])
        self.metrics['train_median_error'].append(additional_metrics['median_error'])
        self.metrics['model_param_norm'].append(model_stats['norm'])
        self.metrics['model_param_mean'].append(model_stats['mean'])
        self.metrics['model_param_std'].append(model_stats['std'])
        self.metrics['scaler_scale'].append(self.scaler.get_scale())
        self.metrics['train_time'].append(train_time)
        self.metrics['samples_per_second'].append(samples_per_second)
        self.metrics['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        if self.track_layer_grads:
            self.metrics['layer_grad_norms'].append(self._compute_layer_grad_norms())

        return epoch_loss, r2, avg_grad_norm, additional_metrics
    
    def validate_permeability(self):
        self.model.eval()
        val_start = time.time()
        
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                
                all_outputs.append(outputs_cpu)
                all_targets.append(targets_cpu)
                
                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

        val_time = time.time() - val_start
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        r2 = self._compute_r2_from_accumulators(sum_squared_error, sum_targets, sum_targets_squared, count)
        
        # Additional metrics
        additional_metrics = self._compute_additional_metrics(all_outputs, all_targets)

        self.metrics['R2_val'].append(r2)
        self.metrics['val_loss'].append(epoch_loss)
        self.metrics['val_mae'].append(additional_metrics['mae'])
        self.metrics['val_rmse'].append(additional_metrics['rmse'])
        self.metrics['val_mape'].append(additional_metrics['mape'])
        self.metrics['val_max_error'].append(additional_metrics['max_error'])
        self.metrics['val_median_error'].append(additional_metrics['median_error'])
        self.metrics['val_time'].append(val_time)

        return epoch_loss, r2, additional_metrics
    
    def validate_dispersion(self):
        """
        Validation for dispersion task.

        Returns:
            epoch_loss (float): mean validation loss
            r2 (float): R^2 score over validation set
            additional_metrics (dict): additional error metrics
        """
        self.model.eval()
        val_start = time.time()

        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        total_samples = 0
        
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for Batch in self.val_loader:
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

                scaled_D = self.scale(D)
                loss = self.criterion(outputs, scaled_D)
                outputs = self.inverse_scale(outputs)

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Metrics in original space
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = D.detach().cpu()
                
                all_outputs.append(outputs_cpu)
                all_targets.append(targets_cpu)

                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()

        val_time = time.time() - val_start
        
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )
        
        # Additional metrics
        additional_metrics = self._compute_additional_metrics(all_outputs, all_targets)

        self.metrics['val_loss'].append(epoch_loss)
        self.metrics['R2_val'].append(r2)
        self.metrics['val_mae'].append(additional_metrics['mae'])
        self.metrics['val_rmse'].append(additional_metrics['rmse'])
        self.metrics['val_mape'].append(additional_metrics['mape'])
        self.metrics['val_max_error'].append(additional_metrics['max_error'])
        self.metrics['val_median_error'].append(additional_metrics['median_error'])
        self.metrics['val_time'].append(val_time)

        return epoch_loss, r2, additional_metrics
    
    def test_permeability(self):
        self.model.eval()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
                outputs_cpu = outputs.detach().cpu()
                targets_cpu = targets.detach().cpu()
                
                all_outputs.append(outputs_cpu)
                all_targets.append(targets_cpu)
                
                sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                sum_targets += torch.sum(targets_cpu).item()
                sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                count += targets_cpu.numel()
                
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        epoch_loss = running_loss / len(self.test_loader.dataset)
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )
        
        # Additional metrics
        additional_metrics = self._compute_additional_metrics(all_outputs, all_targets)
        
        self.metrics['R2_test'].append(r2)
        self.metrics['test_loss'].append(epoch_loss)
        self.metrics['test_mae'].append(additional_metrics['mae'])
        self.metrics['test_rmse'].append(additional_metrics['rmse'])
        self.metrics['test_mape'].append(additional_metrics['mape'])
        self.metrics['test_max_error'].append(additional_metrics['max_error'])
        self.metrics['test_median_error'].append(additional_metrics['median_error'])
        
        return epoch_loss, r2, additional_metrics
    
    def test_dispersion(self):
        self.model.eval()
        running_loss = 0.0
        sum_squared_error = 0.0
        sum_targets = 0.0
        sum_targets_squared = 0.0
        count = 0
        total_samples = 0
        
        all_outputs = []
        all_targets = []
        
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
                    
                    all_outputs.append(outputs_cpu)
                    all_targets.append(targets_cpu)
                    
                    sum_squared_error += torch.sum((targets_cpu - outputs_cpu) ** 2).item()
                    sum_targets += torch.sum(targets_cpu).item()
                    sum_targets_squared += torch.sum(targets_cpu ** 2).item()
                    count += targets_cpu.numel()
                    
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        r2 = self._compute_r2_from_accumulators(
            sum_squared_error, sum_targets, sum_targets_squared, count
        )
        
        # Additional metrics
        additional_metrics = self._compute_additional_metrics(all_outputs, all_targets)
        
        self.metrics['R2_test'].append(r2)
        self.metrics['test_loss'].append(epoch_loss)
        self.metrics['test_mae'].append(additional_metrics['mae'])
        self.metrics['test_rmse'].append(additional_metrics['rmse'])
        self.metrics['test_mape'].append(additional_metrics['mape'])
        self.metrics['test_max_error'].append(additional_metrics['max_error'])
        self.metrics['test_median_error'].append(additional_metrics['median_error'])
        
        return epoch_loss, r2, additional_metrics
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        save_path = os.path.join(self.config.get('save_model_path', 'results'), f"{self.config['model']['name']}_lr-{self.config['learning_rate']}_wd-{self.config['weight_decay']}_bs-{self.config['batch_size']}_epochs-{num_epochs}_{self.config.get('decay','no-decay')}_warmup-{self.config.get('warmup_steps',0)}_clipgrad-{self.config.get('clip_grad',False)}_pe-encoder-{self.config.get('pe_encoder',None)}_pe-{self.config.get('Pe',None)}")
        
        print(f'Saving state-dicts to: {save_path}.pth and {save_path}_last_model.pth')
        print(f'Saving metrics to: {save_path}_metrics.zarr')
        print(f"\n{'='*100}")
        print(f"Training Configuration:")
        print(f"  Model: {self.model_name}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning Rate: {self.config.get('learning_rate', 'N/A')}")
        print(f"  Batch Size: {self.config.get('batch_size', 'N/A')}")
        print(f"  Gradient Clipping: {self.clip_grad} (max norm: {self.max_grad_norm})")
        print(f"  Mixed Precision: {self.scaler.is_enabled()}")
        print(f"{'='*100}\n")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            print(f"\n{'='*100}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*100}")

            # ---- TRAIN ----
            train_results = self.train_epoch()
            train_loss = train_results[0]
            train_r2 = train_results[1]
            avg_grad_norm = train_results[2]
            train_metrics = train_results[3] if len(train_results) > 3 else {}
            
            clip_pct = self.metrics['grad_norm_clipped'][-1]
            grad_max = self.metrics['grad_norm_max'][-1]
            grad_min = self.metrics['grad_norm_min'][-1]
            
            print(f"\n TRAINING METRICS:")
            print(f"  Loss:        {train_loss:.6f}")
            print(f"  R²:          {train_r2:.6f}")
            print(f"  MAE:         {train_metrics.get('mae', 0):.6f}")
            print(f"  RMSE:        {train_metrics.get('rmse', 0):.6f}")
            print(f"  MAPE:        {train_metrics.get('mape', 0):.2f}%")
            print(f"  Max Error:   {train_metrics.get('max_error', 0):.6f}")
            print(f"  Median Err:  {train_metrics.get('median_error', 0):.6f}")
            print(f"\n GRADIENT METRICS:")
            print(f"  Avg Norm:    {avg_grad_norm:.5e}")
            print(f"  Max Norm:    {grad_max:.5e}")
            print(f"  Min Norm:    {grad_min:.5e}")
            print(f"  Clipped:     {clip_pct:.1f}% of batches")

            # ---- VALIDATION ----
            val_results = self.validate_epoch()
            val_loss = val_results[0]
            val_r2 = val_results[1]
            val_metrics = val_results[2] if len(val_results) > 2 else {}
            
            print(f"\n VALIDATION METRICS:")
            print(f"  Loss:        {val_loss:.6f}")
            print(f"  R²:          {val_r2:.6f}")
            print(f"  MAE:         {val_metrics.get('mae', 0):.6f}")
            print(f"  RMSE:        {val_metrics.get('rmse', 0):.6f}")
            print(f"  MAPE:        {val_metrics.get('mape', 0):.2f}%")
            print(f"  Max Error:   {val_metrics.get('max_error', 0):.6f}")
            print(f"  Median Err:  {val_metrics.get('median_error', 0):.6f}")

            # ---- OPTIMIZATION STATS ----
            current_lr = self.scheduler.get_last_lr()[0]
            scaler_scale = self.metrics['scaler_scale'][-1]
            param_norm = self.metrics['model_param_norm'][-1]
            
            print(f"\n  OPTIMIZATION:")
            print(f"  Learning Rate:    {current_lr:.6e}")
            print(f"  Scaler Scale:     {scaler_scale:.1f}")
            print(f"  Param L2 Norm:    {param_norm:.6e}")

            # ---- TIMING ----
            epoch_time = time.time() - epoch_start
            train_time = self.metrics['train_time'][-1]
            val_time = self.metrics['val_time'][-1]
            throughput = self.metrics['samples_per_second'][-1]
            
            self.metrics['epoch_time'].append(epoch_time)
            
            print(f"\n  TIMING:")
            print(f"  Epoch Time:       {epoch_time:.2f}s")
            print(f"  Train Time:       {train_time:.2f}s")
            print(f"  Val Time:         {val_time:.2f}s")
            print(f"  Throughput:       {throughput:.1f} samples/sec")

            # ---- CHECKPOINT ----
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                self.save_model(save_path + ".pth")
                print(f"\nNEW BEST MODEL! (improved by {improvement:.6f})")
            
            print(f"{'='*100}")
            
        # Save final model and metrics
        self.save_model(save_path+"_last_model.pth")
        self.save_metrics(save_path+'_metrics.zarr')
        
        print(f"\n{'='*100}")
        print(f"Training Complete!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Models saved to: {save_path}.pth and {save_path}_last_model.pth")
        print(f"  Metrics saved to: {save_path}_metrics.zarr")
        print(f"{'='*100}\n")

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
            # Skip layer_grad_norms if it contains dicts
            if key == 'layer_grad_norms' and len(values) > 0 and isinstance(values[0], dict):
                # Could save as JSON or nested zarr groups if needed
                continue
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
    
    config = {
        'pin_memory': False,
        'model': {'name': 'test_model'},
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'batch_size': 16,
    }
    trainer = Trainer(model, train_loader, train_loader, train_loader, optimizer, torch.device('cpu'), config)
    trainer.train(num_epochs=5)
    trainer.save_model('model.pth')
    trainer.save_metrics('metrics.zarr')