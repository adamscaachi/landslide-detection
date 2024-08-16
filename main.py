import torch
import torch.nn as nn
import torch.optim as optim
from data import Data
from unet import UNet
from trainer import Trainer
from evaluator import Evaluator

def train_model(file_name, epochs):
    bands = ["B4", "B3", "B2", "NDVI", "B13", "B14"]	
    model = UNet(in_channels=len(bands), out_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Data(bands)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loader = data.train_loader
    val_loader = data.val_loader
    trainer = Trainer(model, criterion, optimizer, device, train_loader, val_loader)
    trainer.train(epochs)
    torch.save(model.state_dict(), file_name + '.pth' )

def eval_model(file_name):
    bands = ["B4", "B3", "B2", "NDVI", "B13", "B14"]	
    model = UNet(in_channels=len(bands), out_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Data(bands)
    val_loader = data.val_loader
    test_loader = data.test_loader
    state_dict = file_name + '.pth'
    evaluator = Evaluator(model, state_dict, device, val_loader, test_loader)

if __name__ == "__main__":
    mode = input("Do you want to train (type 'train') or evaluate (type 'eval') a model? ")
    if mode == 'train':
        file_name = input("Please specify a name for the model: ")
        epochs = int(input("How many epochs do you want to train the model for? "))
        train_model(file_name, epochs)
        print("Training of model " + file_name + " complete!")
    elif mode == 'eval':
        file_name = input("Please specify the name of the model to be evaluated: ")
        eval_model(file_name)