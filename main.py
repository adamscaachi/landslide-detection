import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import Data
from unet import UNet
from trainer import Trainer
from evaluator import Evaluator

def train_model(name, data, model, criterion, optimizer, device, epochs):
    train_loader = data.train_loader
    val_loader = data.val_loader
    trainer = Trainer(model, criterion, optimizer, device, train_loader, val_loader)
    trainer.train(epochs)
    torch.save(model.state_dict(), name + '.pth' )

def eval_model(name, data, model, device):
    state_dict = name + '.pth'
    val_loader = data.val_loader
    test_loader = data.test_loader
    evaluator = Evaluator(model, state_dict, device, val_loader, test_loader)

# Baseline Approach
def a(mode):
    name = 'a'
    bands = ["B4", "B3", "B2", "NDVI", "B13", "B14"]  
    data = Data(bands)
    model = UNet(in_channels=len(bands), out_channels=1)  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mode == 'train':
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        epochs = 25
        train_model(name, data, model, criterion, optimizer, device, epochs)
    elif mode == 'eval':
        eval_model(name, data, model, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('approach', type=str, help='Approach abbreviation: e.g. [a] if using the baseline approach.')
    parser.add_argument('mode', type=str, help='Mode to run the script: [train] or [eval].')
    args = parser.parse_args()
    approaches = {"a": a}
    approach = approaches.get(args.approach)
    approach(args.mode)