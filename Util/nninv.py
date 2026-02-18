import os
import statistics
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.cluster import KMeans
from tqdm import tqdm

from Util.vis_util import make_scatter

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NNInv(nn.Module):
    def __init__(self, n_features=30, in_features=2):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.m(x)

    def inverse(self, X_2d, batch_data=False):
        self.eval()
        X_2d_tensor = torch.tensor(X_2d, dtype=torch.float32)  # Convert input to Tensor
        with torch.no_grad():
            predictions = []
            if batch_data:
                batches = torch.tensor_split(X_2d_tensor, 20)
                for batch in tqdm(batches, desc="Processing batches (inverse)", unit="batches"):
                    inv_batch = self(batch)
                    predictions.extend(inv_batch)
                predictions = np.array(predictions)
            else:
                predictions = self(X_2d_tensor).numpy()
        return predictions

def train_model(model, X, X_2d, device, optimiser, epochs, loss_fn):
    """
    Trains the neural network model to learn the inverse mapping from 2D dimensionally reduced points
    to the original high-dimensional data. Early stopping is applied.

    :param model: An instance of the NNInv class to be trained.
    :param X: The original high-dimensional points.
    :param X_2d: The 2-dimensional points to learn the inverse mapping from.
    :param device: The device on which to train the model, cuda or cpu.
    :param optimiser: The optimiser for training
    :param epochs: The number of epochs to train the model.
    :param loss_fn: The loss function to use.
    """
    best_loss = float('inf')
    best_model_weights = None
    patience = 20

    print('This function has been called')
    print(X_2d.shape, X.shape)
    print(min(X_2d.flatten()), max(X_2d.flatten()))

    if X_2d.shape[1] == 2:
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.scatter(x=X_2d[:, 0], y=X_2d[:, 1], s=5)
        plt.title('Scaled Points')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.show()

    # Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X_2d, X, test_size=0.2, random_state=SEED)

    # Create a DataLoader
    trainset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False)

    testset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    print('Training Model...\n')
    training_loss = []
    testing_loss = []
    final_epoch = 0
    for epoch in range(epochs):
        running_loss = 0.0

        # Training the model
        model.train()
        for inputs, labels in trainloader:
            optimiser.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        print(f'TRAINING: Epoch: {epoch + 1:2}/{epochs}... Loss:{running_loss / len(trainloader):10.7f}')
        training_loss.append(running_loss / len(trainloader))

        # Test the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
        avg_loss = val_loss / len(testloader)
        testing_loss.append(avg_loss)
        print(f'TESTING: Epoch: {epoch + 1:2}/{epochs}... Loss: {avg_loss:10.7f}... Patience: {patience}')

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_weights = model.state_dict()
            patience = 20
        else:
            patience -= 1
            if patience == 0 or (epoch + 1) == epochs:
                final_epoch = epoch
                break

    epochs = range(1, final_epoch + 2)
    plt.plot(epochs, training_loss, 'r', epochs, testing_loss, 'b')
    plt.title('NNInv Test/Train Loss')
    plt.show()
    if final_epoch + 1 != epochs:
        model.load_state_dict(best_model_weights)


def train_model_clusters(model, X, X_2d, X_labels, device, optimiser, epochs, loss_fn, clusters=1000):
    """
    Trains the neural network model to learn the inverse mapping from 2D dimensionally reduced points
    to the original high-dimensional data. Early stopping is applied.

    :param model: An instance of the NNInv class to be trained.
    :param X: The original high-dimensional points.
    :param X_2d: The 2-dimensional points to learn the inverse mapping from.
    :param device: The device on which to train the model, cuda or cpu.
    :param optimiser: The optimiser for training
    :param epochs: The number of epochs to train the model.
    :param loss_fn: The loss function to use.
    """
    best_loss = float('inf')
    best_model_weights = None
    patience = 20
    batch_size = 32

    print(min(X_2d.flatten()), max(X_2d.flatten()), X_2d.shape, X.shape)
    # Cluster the 2-dimensional points using K-means
    clustered_X_2d = KMeans(n_clusters=clusters, random_state=SEED).fit(X_2d)
    cluster_centres = clustered_X_2d.cluster_centers_
    X_2d_cluster_labels = clustered_X_2d.labels_

    cluster_classes = []
    for x in range(clusters):
        mask = X_2d_cluster_labels == x
        labels = X_labels[mask]
        cluster_classes.append(statistics.mode(labels))

    # Split the clusters into test/train
    cluster_ids = set(range(0, clusters))
    train_clusters = random.sample(cluster_ids, int(len(cluster_ids) * 0.8))
    test_clusters = list(set(cluster_ids) - set(train_clusters))

    # Quick plots
    split = [True if x in train_clusters else False for x in cluster_ids]

    plt.rcParams["figure.figsize"] = [10, 10]
    plt.scatter(cluster_centres[:,0], cluster_centres[:,1], c=split, s=5)
    plt.title('NNInv Test/Train Cluster Centres')
    plt.show()

    make_scatter(cluster_centres[:, 0], cluster_centres[:, 1], cluster_classes, no_axis=True, s=10)

    plt.rcParams["figure.figsize"] = [10, 10]
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=X_2d_cluster_labels, s=5)
    plt.title('NNInv Test/Train Clusters')
    plt.show()

    print('Training Model...\n')
    training_loss = []
    testing_loss = []
    final_epoch = 0
    for epoch in range(epochs):
        epoch_train_indexes = train_clusters.copy()
        epoch_test_indexes = test_clusters.copy()
        running_loss = 0.0
        # Training the model
        model.train()

        while epoch_train_indexes: # In place of for inputs, labels in trainloader
            optimiser.zero_grad()
            # Sample our indexes to create a batch to feed into the network.
            size = batch_size if batch_size <= len(epoch_train_indexes) else len(epoch_train_indexes)
            batch_indexes = set()
            inputs = []
            labels = []

            # Create the train batch
            for x in range(size):
                choice = random.choice(tuple(epoch_train_indexes))
                batch_indexes.add(choice)
                epoch_train_indexes.remove(choice)
                mask = X_2d_cluster_labels == choice
                cluster_x2d = X_2d[mask]
                cluster_x = X[mask]

                cluster_choice = random.randint(0, len(cluster_x) - 1)
                inputs.append(cluster_x2d[cluster_choice])
                labels.append(cluster_x[cluster_choice])

            inputs, labels = torch.Tensor(np.array(inputs)).to(device), torch.Tensor(np.array(labels)).to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        print(f'TRAINING: Epoch: {epoch + 1:2}/{epochs}... Loss:{running_loss / len(train_clusters):10.7f}')
        training_loss.append(running_loss / len(train_clusters))

        # Testing the Model
        model.eval()
        val_loss = 0.0
        while epoch_test_indexes:
            size = batch_size if batch_size <= len(epoch_test_indexes) else len(epoch_test_indexes)
            batch_indexes = set()
            inputs = []
            labels = []

            for x in range(size):
                choice = random.choice(tuple(epoch_test_indexes))
                batch_indexes.add(choice)
                epoch_test_indexes.remove(choice)
                mask = X_2d_cluster_labels == choice
                cluster_x2d = X_2d[mask]
                cluster_x = X[mask]

                cluster_choice = random.randint(0, len(cluster_x) - 1)
                inputs.append(cluster_x2d[cluster_choice])
                labels.append(cluster_x[cluster_choice])

            inputs, labels = torch.Tensor(inputs).to(device), torch.Tensor(labels).to(device)

            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        avg_loss = val_loss / len(test_clusters)
        testing_loss.append(avg_loss)
        print(f'TESTING: Epoch: {epoch + 1:2}/{epochs}... Loss: {avg_loss:10.7f}... Patience: {patience}')

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_weights = model.state_dict()
            patience = 20
        else:
            patience -= 1
            if patience == 0:
                final_epoch = epoch
                break
    epochs = range(1, final_epoch + 2)
    plt.plot(epochs, training_loss, 'r', epochs, testing_loss, 'b')
    plt.title('NNInv Test/Train Loss')
    plt.show()
    model.load_state_dict(best_model_weights)

def load_or_train_nninv(X_flat, X_2d, labels, path, device, num_epochs=300, clusters = False):
    inverse_function = NNInv(n_features=X_flat.shape[1]).to(device)
    optimiser = optim.Adam(inverse_function.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    if os.path.exists(path):
        print('Pre-trained NNInv Exists!\n')
        inverse_function.load_state_dict(torch.load(path))
    else:
        train_model(inverse_function,
            X=X_flat,
            X_2d=X_2d,
            device=device,
            optimiser=optimiser,
            epochs=num_epochs,
            loss_fn=loss_fn
        )
        torch.save(inverse_function.state_dict(), path)
    inverse_function.eval()
    inverse_function.to('cpu')
    return inverse_function


if __name__ == '__main__':
    print('=== TESTING NNInv ===')
    X = np.load('../data/encoded_mnist_full.npy')
    X_2d = np.load('../data/dimensionally_reduced_points_full.npy')

    plt.scatter(X_2d[:,0], X_2d[:,1])
    plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test = NNInv().to(device)
    optimiser = optim.Adam(test.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_model_clusters(test, X, X_2d, device, optimiser, epochs=100, loss_fn=loss_fn, clusters=1000)
