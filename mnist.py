import statistics
import time
import torchvision
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score
from torchvision import datasets
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from umap import UMAP

from pathlib import Path

from Util.data_util import load_or_compute_data_projections
from Util.nninv import NNInv, train_model
from Util.shap_util import load_or_compute_shap_values, load_2d_shap
from Util.vis_util import make_grid_points, compute_decision_boundary_map, make_single_boundary_map, make_scatter, make_single_boundary_map_with_points, plot_original_vs_inverse_grid
from Util.metric_util import map_points_to_grid, calculate_accuracy, calculate_boundary_map_precision_recall
from Util.cnn import load_or_train_cnn

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
grid_size = 500
n_points_per_square = 1
shapley = True
use_umap = False

SUFFIX = ''

CNN_PATH = f'Models/MNIST/CNN.pth'
RESULTS_FOLDER = 'Results/MNIST/'
SUFFIX += '_data_space' if not shapley else ''
SUFFIX += '_umap' if use_umap else ''
NNINV_PATH = f'Models/MNIST/NNInv{SUFFIX}.pth'
RES_SUFFIX = '_umap' if use_umap else ''

if __name__ == '__main__':
    script_path = Path(__file__).parent
    os.chdir(script_path)
    print(NNINV_PATH)
    print(f'Working in directory {script_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    if shapley:
        print('Working in Shapley space!')
        RESULTS_FOLDER += 'Shapley'
    else:
        print('Working in Data space!')
        RESULTS_FOLDER += 'Data'
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    if use_umap:
        print('We are using UMAP for projections!')
    else:
        print('We are using t-SNE for projections!')

    processing = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.MNIST(
        "../data/Dataset",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    testset = datasets.MNIST(
        "../data/Dataset",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

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
        # sampler=test_sampler
    )

    batch = next(iter(train_loader))[0]
    plt.figure(figsize=(8, 4))
    images2 = torchvision.utils.make_grid(batch, nrow=8)
    plt.imshow(images2.permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    flattened_samples = []
    point_classes = []
    for data, target in trainset:
        flattened_samples.append(data)
        point_classes.append(target)
    flattened_samples = np.array(flattened_samples).reshape(len(flattened_samples), -1)
    point_classes = np.array(point_classes)

    classifier = load_or_train_cnn(train_loader, test_loader, device, CNN_PATH, in_channels=1, report_interval=100)
    X_train = []
    X_flat = []
    y_train = []
    for data, target in trainset:
        y_train.append(target)
        X_flat.append(data.flatten())
        X_train.append(data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_flat = np.array(X_flat)

    X_test = []
    y_test = []
    for data, target in testset:
        y_test.append(target)
        X_test.append(data)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred, _ = classifier.compute_class(X_test)
    recall_score = recall_score(y_test, y_pred, average='macro')
    precision_score = precision_score(y_test, y_pred, average='macro')

    print(f'Model precision: {precision_score:.3f}, Model recall: {recall_score:.3f}')

    if shapley:
        SHAP_DIR = 'ShapleyValues/MNIST/'
        TRAIN_CSV_PATH = f'{SHAP_DIR}MNIST_CNN_train_dataset.csv'
        REDUCED_POINTS_PATH = f'{SHAP_DIR}MNIST_CNN_train_dataset_2d{RES_SUFFIX}.csv'
        BATCH_DIR = f'{SHAP_DIR}Batches/'

        shap_values_train_df = load_or_compute_shap_values(TRAIN_CSV_PATH, BATCH_DIR, classifier, test_loader, trainset, 28, in_channels=1)
        shap_values_train = shap_values_train_df.drop(['ind', 'class'], axis=1).to_numpy()
        reduced_data_df = load_2d_shap(REDUCED_POINTS_PATH, shap_values_train_df, SEED, use_umap)
        make_scatter(reduced_data_df['x'], reduced_data_df['y'], point_classes, filename=f'{RESULTS_FOLDER}points{RES_SUFFIX}.png', s=10, no_axis=True)
    else:
        REDUCED_POINTS_PATH = f'Data/DataProjections/MNIST/MNIST_2d{RES_SUFFIX}.csv'
        reduced_data_df = load_or_compute_data_projections(X_flat, y_train, REDUCED_POINTS_PATH, SEED, use_umap)
    make_scatter(reduced_data_df['x'], reduced_data_df['y'], y_train, filename=f'{RESULTS_FOLDER}points{RES_SUFFIX}.png', s=10, no_axis=True, num_classes=len(np.unique(y_train)), no_cbar=True)
    reduced_dataset = reduced_data_df.drop(['class'], axis=1).to_numpy()

    X_nninv = flattened_samples
    X_df = reduced_data_df
    X2d_nninv = reduced_dataset
    sample_classes = X_df['class'].to_numpy()

    inverse_function = NNInv(n_features=flattened_samples.shape[1]).to(device)
    optimiser = optim.Adam(inverse_function.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    if os.path.exists(NNINV_PATH):
        print('Pre-trained NNInv Exists!')
        inverse_function.load_state_dict(torch.load(NNINV_PATH))
    else:
        print(len(X_nninv))
        print(len(X2d_nninv))
        train_model(inverse_function,
                    X=X_nninv,
                    X_2d=X2d_nninv,
                    device=device,
                    optimiser=optimiser,
                    epochs=300,
                    loss_fn=loss_fn
                    )
        torch.save(inverse_function.state_dict(), NNINV_PATH)
    inverse_function.eval()
    inverse_function.to('cpu')

    points_to_invert = make_grid_points(grid_size, n_points_per_square)
    flattened_points = points_to_invert.reshape((grid_size * grid_size * n_points_per_square, 2))

    inverted_points = inverse_function.inverse(flattened_points, batch_data=True)
    inverted_points = inverted_points.reshape((inverted_points.shape[0], 1, 28, 28))
    classification_grid, confidence_grid = compute_decision_boundary_map(classifier.to('cpu'), inverted_points, grid_size, n_points_per_square, binary=False)

    make_single_boundary_map(classification_grid, filename=f'{RESULTS_FOLDER}/map{RES_SUFFIX}.png', num_classes=10)
    make_single_boundary_map(confidence_grid, filename=f'{RESULTS_FOLDER}/confidence{RES_SUFFIX}.png', num_classes=10,
                             confidence=True)

    num_of_rand_points = 5_000
    shuffled_point_indecies = np.arange(len(reduced_dataset))
    np.random.shuffle(shuffled_point_indecies)
    shuffled_point_indecies = shuffled_point_indecies[:num_of_rand_points]

    random_points = reduced_dataset[shuffled_point_indecies]
    random_point_classes = point_classes[shuffled_point_indecies]

    make_single_boundary_map_with_points(classification_grid, random_points, random_point_classes,
                                         filename=f'{RESULTS_FOLDER}/map_with_points{RES_SUFFIX}.png', num_classes=10,
                                         grid_size=grid_size)

    _, prediction_grid = map_points_to_grid(reduced_dataset, point_classes)
    accuracy = (calculate_accuracy(classification_grid, prediction_grid) / len(reduced_dataset)) * 100
    print(f'Default Map Accuracy (predicted labels): {accuracy:.2f}%')

    precision_scores = []
    recall_scores = []
    for x in range(10):
        map_precision, map_recall = calculate_boundary_map_precision_recall(classification_grid, prediction_grid, c=x)
        precision_scores.append(map_precision)
        recall_scores.append(map_recall)
        print(f'Class {x} map precision: {map_precision:5f}')
        print(f'Class {x} map recall: {map_recall:5f}')

    avg_precision = statistics.mean(precision_scores)
    average_recall = statistics.mean(recall_scores)

    print(f'Avg map precision: {avg_precision:5f}')
    print(f'Avg map recall: {average_recall:5f}')
    print(f'Most common class; {statistics.mode(classification_grid.flatten())}')
    #map_to_witness_grid(classification_grid, prediction_grid)

    samples_to_invert = 8

    original_images = X_nninv[:samples_to_invert]
    points_to_invert = X2d_nninv[:samples_to_invert]
    sample_labels = sample_classes[:samples_to_invert]

    original_images = original_images.reshape((samples_to_invert, 28, 28))
    inverted_images = inverse_function.inverse(points_to_invert).reshape((samples_to_invert, 28, 28))

    plot_original_vs_inverse_grid(original_images, inverted_images, sample_labels, colour=False, filename=f'{RESULTS_FOLDER}/original_vs_inverse_grid{RES_SUFFIX}.png')
    plt.show()
