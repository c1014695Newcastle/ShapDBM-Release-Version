import argparse
import statistics

import torchvision
from sklearn.manifold import TSNE
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler

from torchvision import datasets
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn

from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from Util.cnn import load_or_train_cnn
from pathlib import Path

from Util.data_util import load_or_compute_data_projections
from Util.nninv import NNInv, train_model, load_or_train_nninv

from Util.shap_util import load_or_compute_shap_values, load_2d_shap
from Util.vis_util import make_grid_points, compute_decision_boundary_map, make_single_boundary_map, make_scatter, \
    make_single_boundary_map_with_points, plot_original_vs_inverse_grid, make_illustrated_fig, \
    extract_samples_from_bounding_box
from Util.metric_util import calculate_accuracy, calculate_boundary_map_precision_recall, map_points_to_grid, \
    map_to_witness_grid

script_path = Path(__file__).parent
os.chdir(script_path)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

encoder_epochs = 100
batch_size = 32
num_epochs = 100
pixels = 28 # (2352 features)

@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)

class SobelSVHN(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CNN_PATH = f'Models/SVHN/CNN.pth'
    SOBEL_CNN_PATH = f'Models/SVHN/SobelCNN.pth'

    processing = transforms.Compose([
        transforms.Resize(pixels),
        transforms.ToTensor(),
    ])

    trainset = datasets.SVHN(
        "Data/SVHN",
        split='train',
        download=True,
        transform=processing
    )

    testset = datasets.SVHN(
        "Data/SVHN",
        split='test',
        download=True,
        transform=processing
    )

    sobel_classes = [3, 7]

    X_train = []
    y_train = []
    for data, target in tqdm(trainset, desc='Transforming trainset', unit='samples'):
        if target not in sobel_classes:
            y_train.append(target)
            X_train.append(data)
        else:
            data = data.permute(1, 2, 0).numpy()
            sobel_image = torch.tensor(sobel_each(data)).permute(2, 0, 1)
            X_train.append(sobel_image)
            y_train.append(target)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []
    for data, target in tqdm(testset, desc='Transforming testset', unit='samples'):
        if target not in sobel_classes:
            y_test.append(target)
            X_test.append(data)
        else:
            data = data.permute(1, 2, 0).numpy()
            sobel_image = torch.tensor(sobel_each(data)).permute(2, 0, 1)
            X_test.append(sobel_image)
            y_test.append(target)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    sobel_trainset = SobelSVHN(X_train, y_train)
    sobel_testset = SobelSVHN(X_test, y_test)

    sobel_train_loader = DataLoader(
        sobel_trainset,
        batch_size=batch_size,
        shuffle=False
        # sampler=train_sampler
    )

    sobel_test_loader = DataLoader(
        sobel_testset,
        batch_size=batch_size,
        shuffle=False
        # sampler=test_sampler
    )

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    batch = next(iter(sobel_train_loader))[0]
    plt.figure(figsize=(20, 10))
    images2 = torchvision.utils.make_grid(batch, nrow=8)
    plt.imshow(images2.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    #plt.title(f'Resized to {pixels} pixels with Sobel')
    plt.savefig('Results/sobelSVHN.png', dpi=300, bbox_inches='tight')
    plt.show()

    print('========== SOBEL ==========\n')

    sobel_classifier = load_or_train_cnn(sobel_train_loader, sobel_test_loader, device, SOBEL_CNN_PATH)
    y_pred, _ = sobel_classifier.compute_class(X_test, split=True, device=device)

    r_score = recall_score(y_test, y_pred, average='macro')
    p_score = precision_score(y_test, y_pred, average='macro')
    print(f'Model precision: {p_score:.3f}, Model recall: {r_score:.3f}')

    print('========== NORMAL==========\n')

    X_test = []
    y_test = []
    for data, target in tqdm(testset, desc='Transforming testset', unit='samples'):
        y_test.append(target)
        X_test.append(data)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False
        # sampler=train_sampler
    )

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
        # sampler=train_sampler
    )

    batch = next(iter(train_loader))[0]
    plt.figure(figsize=(20, 10))
    images2 = torchvision.utils.make_grid(batch, nrow=8)
    plt.imshow(images2.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    #plt.title(f'Resized to {pixels} pixels with Sobel')
    plt.savefig('Results/SVHN.png', dpi=300, bbox_inches='tight')
    plt.show()


    norm_classifier = load_or_train_cnn(train_loader, test_loader, device, CNN_PATH)
    y_pred, _ = norm_classifier.compute_class(X_test, split=True, device=device)

    r_score = recall_score(y_test, y_pred, average='macro')
    p_score = precision_score(y_test, y_pred, average='macro')
    print(f'Model precision: {p_score:.3f}, Model recall: {r_score:.3f}')
