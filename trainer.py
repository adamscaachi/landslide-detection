import time
import torch
import numpy as np
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, model, criterion, optimiser, device, train_loader, val_loader):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, num_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
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
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[epoch]:.4f}, Validation Loss: {val_losses[epoch]:.4f}, Duration: {(time.time()-start_time):.2f} seconds')
        self.plot_losses(num_epochs, train_losses, val_losses)

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

    def plot_losses(self, num_epochs, train_losses, val_losses):
        x = np.linspace(1, num_epochs, num_epochs)
        plt.plot(x[1:], train_losses[1:], 'b-', label='Training')
        plt.plot(x[1:], val_losses[1:], 'r-', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_curves.png', dpi=400, bbox_inches='tight')