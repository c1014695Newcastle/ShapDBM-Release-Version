import torchvision
from torchvision import datasets
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn

import random
import numpy as np
import os
import matplotlib.pyplot as plt


from Util.cnn import load_or_train_cnn
from pathlib import Path

from Util.data_util import load_or_compute_data_projections
from Util.nninv import NNInv, train_model
from Util.shap_util import load_or_compute_shap_values, load_2d_shap
from Util.vis_util import make_grid_points, compute_decision_boundary_map, make_single_boundary_map, make_scatter, make_single_boundary_map_with_points
from Util.metric_util import map_points_to_grid, calculate_accuracy

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
nninv_points = 0
shapley = False

classifier_path = 'Models/SampledSVHN/CNN.pth'
SUFFIX = '_data_space' if not shapley else ''
nninv_path = f'Models/SampledSVHN/NNInv{SUFFIX}.pth'
RESULTS_FOLDER = 'Results/SampledSVHN/'

if __name__ == '__main__':
    script_path = Path(__file__).parent
    os.chdir(script_path)
    print(f'Working in directory {script_path}')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = datasets.SVHN(
        "Data/Dataset",
        split='train',
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    testset = datasets.SVHN(
        "Data/Dataset",
        split='test',
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    image_shape = testset.data.shape[1:]
    print(image_shape)

    train_samples = 8_000
    shuffled_train_indecies = np.arange(len(trainset))
    np.random.shuffle(shuffled_train_indecies)
    downsized_train_indices = shuffled_train_indecies[:train_samples]
    train_subset = Subset(trainset, downsized_train_indices)

    test_samples = 2_000
    shuffled_test_indices = np.arange(len(testset))
    np.random.shuffle(shuffled_test_indices)
    downsized_test_indices = shuffled_test_indices[:test_samples]
    test_subset = Subset(testset, downsized_test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False
        # sampler=train_sampler
    )

    X_train = []
    X_flat = []
    y_train = []
    for data, target in train_subset:
        y_train.append(target)
        X_flat.append(data.flatten())
        X_train.append(data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_flat = np.array(X_flat)

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f'Train dataset size: {len(train_subset)}')
    print(f'Test dataset size: {len(test_subset)}')

    batch = next(iter(test_loader))[0]

    plt.figure(figsize=(8, 4))
    images2 = torchvision.utils.make_grid(batch, nrow=8)
    plt.imshow(images2.permute(1, 2, 0))
    plt.show()

    classifier = load_or_train_cnn(train_loader, test_loader, device, classifier_path, report_interval=100, linear_neurons=4096)
    X_train_pred, _ = classifier.compute_class(X_train, split=True)

    if shapley:
        SHAP_DIR = 'ShapleyValues/SampledSVHN/'
        BATCH_DIR = f'{SHAP_DIR}Batches/'
        if not os.path.exists(SHAP_DIR):
            os.mkdir(SHAP_DIR)
        if not os.path.exists(BATCH_DIR):
            os.mkdir(BATCH_DIR)
        TRAIN_CSV_PATH = f'{SHAP_DIR}/shap_values_svhn_unencoded_train_dataset.csv'
        REDUCED_POINTS_CSV_PATH = f'{SHAP_DIR}/svhn_unencoded_train_dataset_2d.csv'

        shap_values_train_df = load_or_compute_shap_values(TRAIN_CSV_PATH, BATCH_DIR, classifier, test_loader, train_subset,32)
        to_drop = ['ind', 'class'] if 'ind' in shap_values_train_df.columns else ['class']
        shap_values_train = shap_values_train_df.drop(['ind', 'class'], axis=1).to_numpy()
        reduced_data_df = load_2d_shap(REDUCED_POINTS_CSV_PATH, shap_values_train_df, SEED)
    else:
        REDUCED_POINTS_PATH = 'Data/DataProjections/sampled_SVHN_data_2d.csv'
        reduced_data_df = load_or_compute_data_projections(X_flat, y_train, REDUCED_POINTS_PATH, SEED, False)

    make_scatter(reduced_data_df['x'], reduced_data_df['y'], y_train, filename=f'{RESULTS_FOLDER}/svhn_train_2d.png', no_axis=True, s=10)
    make_scatter(reduced_data_df['x'], reduced_data_df['y'], y_train, filename=f'{RESULTS_FOLDER}/svhn_train_2d_no_cbar.png', no_axis=True, s=10, no_cbar=True)
    reduced_dataset = reduced_data_df.drop(['class'], axis=1).to_numpy()

    inverse_function = NNInv(n_features=X_flat.shape[1]).to(device)
    optimiser = optim.Adam(inverse_function.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    if os.path.exists(nninv_path):
        print('Pre-trained NNInv Exists!')
        inverse_function.load_state_dict(torch.load(nninv_path))
    else:
        train_model(inverse_function,
            X=X_flat,
            X_2d=reduced_dataset,
            device=device,
            optimiser=optimiser,
            epochs=300,
            loss_fn=loss_fn
        )
        torch.save(inverse_function.state_dict(), nninv_path)
    inverse_function.eval()
    inverse_function.to('cpu')

    points_to_invert = make_grid_points(grid_size, n_points_per_square)
    flattened_points = points_to_invert.reshape((grid_size * grid_size * n_points_per_square, 2))

    inverted_points = inverse_function.inverse(flattened_points)
    inverted_points = inverted_points.reshape((inverted_points.shape[0], image_shape[0], image_shape[1], image_shape[2]))
    classification_grid, confidence_grid = compute_decision_boundary_map(classifier.to('cpu'), inverted_points, grid_size, n_points=n_points_per_square, binary=False)

    make_single_boundary_map(classification_grid, filename=f'{RESULTS_FOLDER}/map.png', num_classes=len(np.unique(y_train)))
    make_single_boundary_map(confidence_grid, filename=f'{RESULTS_FOLDER}/confidence.png', num_classes=len(np.unique(y_train)), confidence=True)
    make_single_boundary_map_with_points(classification_grid, reduced_dataset[:10000], y_train[:10000], filename=f'{RESULTS_FOLDER}/map_with_points.png', num_classes=len(np.unique(y_train)), grid_size=grid_size)

    _, prediction_grid = map_points_to_grid(reduced_dataset, y_train)
    accuracy = (calculate_accuracy(classification_grid, prediction_grid) / len(reduced_dataset)) * 100
    print(f'Default Map Accuracy (predicted labels): {accuracy:.2f}%')

    _, prediction_grid = map_points_to_grid(reduced_dataset, X_train_pred)
    accuracy = (calculate_accuracy(classification_grid, prediction_grid) / len(reduced_dataset)) * 100
    print(f'Default Map Accuracy (pred labels): {accuracy:.2f}%')

