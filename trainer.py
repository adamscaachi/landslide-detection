import time
import torch
import numpy as np
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, model, criterion, optimiser, device, train_loader, val_loader, max_epochs, patience, min_delta):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.early_stop = False
        self.best_model_state = None
        self.wait = 0
        self.best_epoch = 0

    def train(self):
        train_losses = []
        val_losses = []
        for epoch in range(self.max_epochs):
            if self.early_stop:
                print(f'Training stopped early... best model was from epoch {self.best_epoch+1}.')
                break
            self.model.train()
            running_loss = 0.0
            start_time = time.time() 
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimiser.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimiser.step()
                running_loss += loss.item() * inputs.size(0)
            train_losses.append(running_loss / len(self.train_loader.dataset))
            val_losses.append(self.validate())
            self.check_early_stopping(val_losses[epoch], epoch)
            print(
                f'Epoch {epoch+1}/{self.max_epochs}, '
                f'Training Loss: {train_losses[epoch]:.4f}, '
                f'Validation Loss: {val_losses[epoch]:.4f}, '
                f'Duration: {(time.time()-start_time):.2f} seconds'
            )
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        self.plot_losses(train_losses, val_losses)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(self.val_loader.dataset)
        return val_loss

    def check_early_stopping(self, val_loss, epoch):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.wait = 0
            self.best_model_state = self.model.state_dict()
        else: 
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True

    def plot_losses(self, train_losses, val_losses):
        x = np.linspace(1, len(train_losses), len(train_losses))
        plt.plot(x[1:], train_losses[1:], 'b-', label='Training')
        plt.plot(x[1:], val_losses[1:], 'r-', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_curves.png', dpi=400, bbox_inches='tight')