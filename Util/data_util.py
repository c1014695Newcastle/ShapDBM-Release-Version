import os

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
from umap import UMAP


def compile_dataset(train_loader, test_loader, encoder, device):
    encoder.to(device)
    encoder.eval()

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    with torch.no_grad():
        for data, label in tqdm(train_loader, desc="Processing training data", unit="data"):
            # Get the encoded features (latent space representation)
            encoded = encoder.encode(data.to(device))
            train_features.append(encoded.to('cpu'))
            train_labels.append(label)

        for data, label in tqdm(test_loader, desc="Processing training data", unit="data"):
            # Get the encoded features (latent space representation)
            encoded = encoder.encode(data.to(device))
            test_features.append(encoded.to('cpu'))
            test_labels.append(label)

    train_features = torch.cat(train_features, dim=0).numpy()
    test_features = torch.cat(test_features, dim=0).numpy()
    train_labels = torch.cat(train_labels, dim=0).numpy()
    test_labels = torch.cat(test_labels, dim=0).numpy()

    full = np.concat([train_features, test_features])
    min_val_X = full.min(axis=0)
    max_val_X = full.max(axis=0)

    train_scaled = (train_features - min_val_X) / (max_val_X - min_val_X)
    test_scaled = (test_features - min_val_X) / (max_val_X - min_val_X)
    return torch.Tensor(train_scaled), torch.Tensor(test_scaled), torch.Tensor(train_labels).long(), torch.Tensor(test_labels).long()

def load_or_compute_data_projections(data, classes, path, seed, use_umap):
    if os.path.exists(path):
        print('Loading 2D dataset...')
        reduced_data_df = pd.read_csv(path)
    else:
        print('Making 2D dataset...')
        scaler = MinMaxScaler()
        if use_umap:
            reducer = UMAP(n_components=2, random_state=seed)
        else:
            reducer = TSNE(n_components=2, random_state=seed)
        reduced_dataset = scaler.fit_transform(reducer.fit_transform(data))
        reduced_data_df = pd.DataFrame(reduced_dataset, columns=['x', 'y']).join(pd.DataFrame(classes, columns=['class']))
        reduced_data_df.to_csv(path, index=False)
    return reduced_data_df