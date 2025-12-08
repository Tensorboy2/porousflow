'''
Docstring for ml.train

This module implements the training loop for the permeability and dispersion models.
It is independent of the data loading and model definition modules.
'''
import torch
import torch.nn as nn
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(self.val_loader.dataset)
        return epoch_loss
    
    def test(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(self.test_loader.dataset)
        return epoch_loss
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        test_loss = self.test()
        print(f'Test Loss: {test_loss:.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    @torch.jit
    def R2_score(self, loader):
        self.model.eval()
        total_r2 = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                ss_res = torch.sum((targets - outputs) ** 2)
                ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
                r2 = 1 - ss_res / ss_tot
                total_r2 += r2.item() * inputs.size(0)
        overall_r2 = total_r2 / len(loader.dataset)
        return overall_r2