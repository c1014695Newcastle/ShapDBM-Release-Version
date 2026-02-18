import statistics
import time
from pathlib import Path

from sklearn.metrics import recall_score, precision_score
from torchvision import datasets
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import os

from Util.cnn import load_or_train_cnn
from Util.data_util import load_or_compute_data_projections
from Util.nninv import load_or_train_nninv
from Util.shap_util import load_or_compute_shap_values, load_2d_shap
from Util.vis_util import make_grid_points, compute_decision_boundary_map, make_single_boundary_map, make_scatter, make_single_boundary_map_with_points, plot_original_vs_inverse_grid
from Util.metric_util import map_points_to_grid, calculate_accuracy, calculate_boundary_map_precision_recall, map_to_witness_grid

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cifar10_label_names = {
    0: "plane",    # class 0
    3: "cat",      # class 1
    4: "deer",     # class 2
    8: "ship",     # class 3
}

BATCH_SIZE = 32
EPOCHS = 100
GRID_SIZE = 500
POINTS_PER_PIXEL = 1
SHAPLEY = False
PIXELS = 28 # (2352 features)
SUFFIX = ''
USE_UMAP = True

CNN_PATH = f'Models/CIFAR4/CNN.pth'
RESULTS_FOLDER = 'Results/CIFAR4/'
PREFIX = 'data_' if not SHAPLEY else 'shap_'
SUFFIX += '_data_space' if not SHAPLEY else ''
SUFFIX += '_umap' if USE_UMAP else ''
NNINV_PATH = f'Models/CIFAR4/NNInv{SUFFIX}.pth'
RES_SUFFIX = '_umap' if USE_UMAP else ''

if __name__ == '__main__':
    script_path = Path(__file__).parent
    os.chdir(script_path)
    print(f'Working in directory {script_path}')
    print('GPU Detected!' if torch.cuda.is_available() else 'CPU not Detected!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    if SHAPLEY:
        print('Working in Shapley space!')
        RESULTS_FOLDER += 'Shapley'
    else:
        print('Working in Data space!')
        RESULTS_FOLDER += 'Data'
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    if USE_UMAP:
        print('We are using UMAP for projections!')
    else:
        print('We are using t-SNE for projections!')
    start_time = time.time()

    processing = transforms.Compose([
        transforms.Resize(PIXELS),
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(
        root='Data/CIFAR',
        train=True,
        download=True,
        transform=processing
    )

    testset = datasets.CIFAR10(
        root='Data/CIFAR',
       train=False,
       download=True,
       transform=processing
    )

    X_train = []
    X_flat = []
    y_train = []
    for data, target in trainset:
        if target in cifar10_label_names:
            y_train.append(target)
            X_flat.append(data.flatten())
            X_train.append(data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_flat = np.array(X_flat)

    X_test = []
    y_test = []
    for data, target in testset:
        if target in cifar10_label_names:
            y_test.append(target)
            X_test.append(data)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Rework labels to be sequential
    y_train[y_train == 3] = 1
    y_test[y_test == 3] = 1
    y_train[y_train == 4] = 2
    y_test[y_test == 4] = 2
    y_train[y_train == 8] = 3
    y_test[y_test == 8] = 3

    trainset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    testset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    classifier = load_or_train_cnn(train_loader, test_loader, device, CNN_PATH, report_interval=100)
    y_pred, _ = classifier.compute_class(X_test)
    recall_score = recall_score(y_test, y_pred, average='macro')
    precision_score = precision_score(y_test, y_pred, average='macro')
    print(f'Model precision: {precision_score:.3f}, Model recall: {recall_score:.3f}\n')
    X_train_pred, _ = classifier.compute_class(X_train)

    if SHAPLEY:
        SHAP_DIR = 'ShapleyValues/CIFAR4/'
        BATCH_DIR = f'{SHAP_DIR}Batches/'
        if not os.path.exists(SHAP_DIR):
            os.mkdir(SHAP_DIR)
        if not os.path.exists(BATCH_DIR):
            os.mkdir(BATCH_DIR)
        TRAIN_CSV_PATH = f'{SHAP_DIR}/CIFAR-4_train_dataset.csv'
        REDUCED_POINTS_CSV_PATH = f'{SHAP_DIR}/CIFAR-4_train_dataset_2d{RES_SUFFIX}.csv'

        shap_values_train_df = load_or_compute_shap_values(TRAIN_CSV_PATH, BATCH_DIR, classifier, test_loader, trainset, 28)
        to_drop = ['ind', 'class'] if 'ind' in shap_values_train_df.columns else ['class']
        shap_values_train = shap_values_train_df.drop(['ind', 'class'], axis=1).to_numpy()
        reduced_data_df = load_2d_shap(REDUCED_POINTS_CSV_PATH, shap_values_train_df, SEED, USE_UMAP)
    else:
        REDUCED_POINTS_PATH = f'Data/DataProjections/CIFAR-4_data_2d{RES_SUFFIX}.csv'
        reduced_data_df = load_or_compute_data_projections(X_flat, y_train, REDUCED_POINTS_PATH, SEED, USE_UMAP)
    reduced_dataset = reduced_data_df.drop(['class'], axis=1).to_numpy()

    make_scatter(reduced_data_df['x'], reduced_data_df['y'], y_train, filename=f'{RESULTS_FOLDER}points{RES_SUFFIX}.png', s=10, no_axis=True, num_classes=len(np.unique(y_train)), tick_labels = cifar10_label_names.values())
    make_scatter(reduced_data_df['x'], reduced_data_df['y'], y_train, filename=f'{RESULTS_FOLDER}points_no_cbar{RES_SUFFIX}.png', s=10,
                 no_axis=True, num_classes=len(np.unique(y_train)), tick_labels=cifar10_label_names.values(),
                 no_cbar=True)

    X_nninv = X_flat
    X_df = reduced_data_df
    X2d_nninv = reduced_dataset
    sample_classes = X_df['class'].to_numpy()
    inverse_function = load_or_train_nninv(X_nninv, X2d_nninv, sample_classes, NNINV_PATH, device)

    points_to_invert = make_grid_points(GRID_SIZE, POINTS_PER_PIXEL)
    flattened_points = points_to_invert.reshape((GRID_SIZE * GRID_SIZE * POINTS_PER_PIXEL, 2))
    inverted_points = inverse_function.inverse(flattened_points).reshape(len(flattened_points), 3, PIXELS, PIXELS)

    classification_grid, confidence_grid = compute_decision_boundary_map(classifier.to('cpu'), inverted_points, GRID_SIZE, n_points=POINTS_PER_PIXEL, binary=False)

    make_single_boundary_map(classification_grid, filename=f'{RESULTS_FOLDER}/map{RES_SUFFIX}.png', num_classes=len(np.unique(y_train)))
    make_single_boundary_map(confidence_grid, filename=f'{RESULTS_FOLDER}/confidence{RES_SUFFIX}.png', num_classes=len(np.unique(y_train)), confidence=True)
    make_single_boundary_map_with_points(classification_grid, reduced_dataset[:10000], y_train[:10000], filename=f'{RESULTS_FOLDER}/map_with_points{RES_SUFFIX}.png', num_classes = len(np.unique(y_train)), grid_size=GRID_SIZE, tick_labels = cifar10_label_names.values())

    end_time = time.time()
    print(f'\nThis process took {round((end_time - start_time)/60, 1)} minutes!\n')

    _, label_grid = map_points_to_grid(reduced_dataset, y_train)
    accuracy = (calculate_accuracy(classification_grid, label_grid) / len(reduced_dataset)) * 100
    print(f'Default Map Accuracy (true labels): {accuracy:.2f}%')

    _, prediction_grid = map_points_to_grid(reduced_dataset, X_train_pred)
    accuracy = (calculate_accuracy(classification_grid, prediction_grid) / len(reduced_dataset)) * 100
    print(f'Default Map Accuracy (pred labels): {accuracy:.2f}%')

    precision_scores = []
    recall_scores = []
    for x in range(len(cifar10_label_names.keys())):
        map_precision, map_recall = calculate_boundary_map_precision_recall(classification_grid, prediction_grid, c=x)
        precision_scores.append(map_precision)
        recall_scores.append(map_recall)
        print(f'Class {x} map precision: {map_precision:2f}')
        print(f'Class {x} map recall: {map_recall:2f}')
    print(f'Avg map precision: {statistics.mean(precision_scores):2f}')
    print(f'Avg map recall: {statistics.mean(recall_scores):2f}')

    samples_to_invert = 8

    original_images = X_train[:samples_to_invert]
    points_to_invert = reduced_dataset[:samples_to_invert]
    sample_labels = y_train[:samples_to_invert]
    sample_labels = [cifar10_label_names.get(s) for s in sample_labels]

    inverted_images = inverse_function.inverse(points_to_invert).reshape((samples_to_invert, 3, PIXELS, PIXELS))

    plot_original_vs_inverse_grid(original_images, inverted_images, sample_labels, filename=f'{RESULTS_FOLDER}/original_vs_inverse_grid{RES_SUFFIX}.png')
