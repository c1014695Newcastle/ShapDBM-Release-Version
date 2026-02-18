import os.path
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import random

from matplotlib import pyplot as plt
from tqdm import tqdm

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

class CNN(nn.Module):

    def __init__(self, in_channels=3, linear_neurons=4096):
        super().__init__()
        self.conv_filters = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added batch normalization for better training stability
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # Output: 8x8x64
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(linear_neurons, linear_neurons),
            nn.BatchNorm1d(linear_neurons),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(linear_neurons, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self.softmax_final = nn.Softmax(dim=1)
        self.log_softmax_final = nn.LogSoftmax(dim=1)

    def forward(self, x, train=False):
        x = self.conv_filters(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        if train:
            x = self.log_softmax_final(x)
        else:
            x = self.softmax_final(x)
        return x

    def activations(self, x):
        x = torch.Tensor(x)
        x = self.conv_filters(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

    def _batch_run(self, batch):
        shape = batch.shape
        batch = self.conv_filters(batch)
        batch = batch.view(shape[0], -1)
        batch = self.fc_layers(batch)
        batch = self.softmax_final(batch)
        batch = batch.cpu()
        if shape[0] > 1:
            batch_class = np.array([torch.argmax(batch[n], dim=0).item() for n in range(shape[0])])
            batch_conf = np.array([torch.max(batch[n], dim=0)[0].detach().numpy() for n in range(shape[0])])
        else:
            batch_class = torch.argmax(batch, dim=1).detach().cpu().numpy()
            batch_conf = torch.max(batch, dim=1)[0].detach().numpy()
        return batch_class, batch_conf

    def compute_class(self, x, split=False):
        x = torch.Tensor(x)
        if split:
            classes = []
            confidences = []
            batches = torch.tensor_split(x, 50)
            for batch in tqdm(batches, desc="Processing batches (map)", unit="batches"):
                batch_class, batch_conf = self._batch_run(batch)
                classes.extend(batch_class)
                confidences.extend(batch_conf)
        else:
            classes, confidences = self._batch_run(x)

        return np.array(classes), np.array(confidences)

def __train_conv(model, device, train_loader, optimizer, epoch, report_interval):
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
        if batch_idx % report_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100.0 * batch_idx / len(train_loader):.3f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )
    print(f'\nTrain set: Average loss: {train_loss / len(train_loader.dataset):.4f}')
    return train_loss / len(train_loader.dataset)

def __test_conv(model, device, test_loader):
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

def load_or_train_cnn(train_loader, test_loader, device, cnn_path, num_epochs=100, in_channels=3, report_interval=500, linear_neurons=3136):
    classifier = CNN(linear_neurons=linear_neurons, in_channels=in_channels).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.5)
    if os.path.exists(cnn_path):
        print('Pre-trained Classifier Exists!')
        classifier.load_state_dict(torch.load(cnn_path))
        __test_conv(classifier, device, test_loader)
    else:
        train_loss = []
        test_loss = []
        for epoch in range(1, num_epochs + 1):
            train_loss.append(__train_conv(classifier, device, train_loader, optimizer, epoch, report_interval))
            test_loss.append(__test_conv(classifier, device, test_loader))
        torch.save(classifier.state_dict(), cnn_path)
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, train_loss, 'r', epochs, test_loss, 'b')
        plt.title('Model Test/Train Loss')
        plt.show()
    classifier.eval()
    classifier.to('cpu')
    return classifier