import os.path
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import random
import shap
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

script_path = Path(__file__).parent
os.chdir(script_path)


class Net10(nn.Module):
    def __init__(self, num_in=30, num_out=10, num_hidden_layer=100):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(num_in, num_hidden_layer),
            nn.ReLU(),

            nn.Linear(num_hidden_layer, num_hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(num_hidden_layer, num_hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(num_hidden_layer, num_hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(num_hidden_layer, num_hidden_layer),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(num_hidden_layer, num_out),
        )
        self.softmax_final = nn.Softmax(dim=1)
        self.log_softmax_final = nn.LogSoftmax(dim=1)

    def activations(self, x):
        x = torch.Tensor(x)
        return self.fc_layers(x)

    def forward(self, x, train=False):
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        if train:
            x = self.log_softmax_final(x)
        else:
            x = self.softmax_final(x)
        return x

    def compute_class(self, x):
        shape = x.shape
        x = torch.Tensor(x)
        x = self.fc_layers(x)
        if shape[0] > 1:
            x = self.softmax_final(x)
            x = np.array([torch.argmax(x[n], dim=0).item() for n in range(shape[0])])
        else:
            x = self.softmax_final(x)
            x = torch.argmax(x, dim=1).detach().numpy()
        return x

    def compute_confidence(self, x, binary=False):
        shape = x.shape
        x = torch.Tensor(x)
        x = self.fc_layers(x)
        if not binary:
            if shape[0] > 1:
                x = self.softmax_final(x)
                x = np.array([torch.max(x[n], dim=0)[0].detach().numpy() for n in range(shape[0])])
            else:
                x = self.softmax_final(x)
                x = torch.max(x, dim=1)[0].detach().numpy()
        else:
            x = F.sigmoid(x).detach().numpy()
            print(x.flatten())
        return x
        # return torch.max(x, dim=1)[0].detach().numpy()


def train_net_10(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, train=True)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100.0 * batch_idx / len(train_loader):.3f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )
    print(f'\nTrain set: Average loss: {train_loss / len(train_loader.dataset):.4f}')
    return train_loss / len(train_loader.dataset)

def test_net_10(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, train=True)
            test_loss += F.nll_loss(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f},"
        f" Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100.0 * correct / len(test_loader.dataset):.3f}%)\n"
    )
    return test_loss