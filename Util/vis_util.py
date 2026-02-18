import statistics
import random

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

from tqdm import tqdm

class_colors = [
    [0.00, 1.00, 0.85],  # Red
    [0.10, 1.00, 0.85],  # Orange
    [0.90, 1.00, 0.80],  # Yellow
    [0.33, 1.00, 0.80],  # Green
    [0.50, 1.00, 0.85],  # Cyan
    [0.67, 1.00, 0.80],  # Blue
    [0.76, 1.00, 0.70],  # Purple
    [0.16, 1.00, 0.80],  # Magenta
    [0.08, 0.65, 0.50],  # More brown
    [0.56, 0.50, 0.55],  # Teal
]

binary_colors = [
    [0.00, 1.00, 0.85],  # Red
    [0.67, 1.00, 0.80],  # Blue
    [0.16, 1.00, 0.80],  # Magenta
    [0.10, 1.00, 0.85],  # Orange
]

def make_cmap(num_classes):
    if num_classes <= 4:
        sc_cmap = colors.ListedColormap([colors.hsv_to_rgb(hsv) for hsv in binary_colors[:num_classes]])
    else:
        sc_cmap = colors.ListedColormap([colors.hsv_to_rgb(hsv) for hsv in class_colors[:num_classes]])
    norm = colors.Normalize(vmin=0, vmax=num_classes - 1)
    ticks = np.arange(num_classes)
    return sc_cmap, norm, ticks

def make_scatter(x, y, labels, title=None, filename=None, no_axis=False, s=3, num_classes=10, tick_labels=None, no_cbar=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    sc_cmap, norm, ticks = make_cmap(num_classes)

    scatter = ax.scatter(x, y, c=labels, cmap=sc_cmap, norm=norm, s=s)
    if not no_cbar:
        cbar = plt.colorbar(scatter, ax=ax,  fraction=0.046, pad=0.04)
        cbar.set_ticks(ticks)
        if tick_labels:
            cbar.set_ticklabels(tick_labels)

    ax.set_aspect('equal', adjustable='box')
    if no_axis:
        plt.xlim(left=min(x), right=max(x))  # Set x-axis limits to the data range
        plt.ylim(bottom=min(y), top=max(y))
        plt.xticks([])
        plt.yticks([])
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

def make_grid_points(grid_size, n_points_per_square):
    """
    Create a grid to the specified size and fill it with a set number of points per square

    Args:
        grid_size: The size of the grid to make
        n_points_per_square: The number of points per square

    Returns:

    """
    x_vals = np.linspace(0, 1, grid_size + 1)
    y_vals = np.linspace(0, 1, grid_size + 1)

    x_min_vals = x_vals[:-1]
    y_min_vals = y_vals[:-1]
    x_min_grid, y_min_grid = np.meshgrid(x_min_vals, y_min_vals)

    # 3. Flatten and repeat for multiple points
    x_min_flat = np.repeat(x_min_grid.ravel(), n_points_per_square)
    y_min_flat = np.repeat(y_min_grid.ravel(), n_points_per_square)

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    offsets_x = np.random.rand(len(x_min_flat)) * dx
    offsets_y = np.random.rand(len(y_min_flat)) * dy

    rand_x = x_min_flat + offsets_x
    rand_y = y_min_flat + offsets_y

    rand_points_reshaped = np.empty((grid_size, grid_size, n_points_per_square, 2))
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            rand_points_reshaped[i, j] = np.column_stack((
                rand_x[count:count + n_points_per_square],
                rand_y[count:count + n_points_per_square]
            ))
            count += n_points_per_square
    return rand_points_reshaped

def compute_decision_boundary_map(classifier, inverted_points, grid_size,n_points, binary=True):
    """
    Method to compute the decision boundary map for a given classifier as well as the classifier's confidence in the prediction.

    :param classifier: A PyTorch neural network to be used to generate the boundary maps (NOTE: This classifier must have methods 'compute_class()' and 'compute_confidence()' to be compatible
    :param inverted_points: A numpy array of inverted points with dimensions compatible with the input of the classifier.
    :param grid_size: The size of the image to be produced.
    :return: A classification grid and a confidence grid, both as numpy arrays.
    """
    if binary:
        confidence = (classifier.compute_confidence(inverted_points)).reshape(grid_size, grid_size, n_points)
        classifications = (confidence > 0.5).astype(int).reshape(grid_size, grid_size, n_points)
    else:
        classifications, confidence = classifier.compute_class(inverted_points, split=True)
        classifications = classifications.astype(int).reshape(grid_size, grid_size, n_points)
        confidence = confidence.reshape(grid_size, grid_size, n_points)
    classifications = np.array([[statistics.mode(x) for x in y] for y in tqdm(classifications, desc='Making final grid', unit='rows')])
    confidence = np.array([[statistics.mean(x) for x in y] for y in tqdm(confidence, desc='Making final confidence grid', unit='rows')])
    return classifications, confidence

def make_single_boundary_map(bm, filename, num_classes, confidence=False):
    if confidence:
        norm = colors.Normalize(vmin=0, vmax=1)
        colormap = 'binary'
    else:
        colormap, norm, bar_ticks = make_cmap(num_classes)
        colormap.set_bad(color='black')

    plt.figure(figsize=(10,10))
    plt.imshow(bm, cmap=colormap, norm=norm, origin='lower')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def make_single_boundary_map_with_points(bm, points, point_labels, filename, num_classes, grid_size, tick_labels=None):
    colormap, norm, ticks = make_cmap(num_classes)

    colormap.set_bad(color='black')
    plt.figure(figsize=(10,10))
    plt.imshow(bm, cmap=colormap, norm=norm, origin='lower')
    plt.scatter(points[:, 0]*grid_size, points[:, 1]*grid_size, c=point_labels, cmap=colormap, edgecolors='black', s=20)
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, grid_size])
    plt.ylim([0, grid_size])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_original_vs_inverse_grid(original_images, inverted_images, labels, colour=True, filename=''):
    fig, axs = plt.subplots(ncols=len(original_images), nrows=2, figsize=(len(original_images) * 2.5, 5))
    for x in range(len(original_images)):
        if colour:
            axs[0, x].imshow(np.transpose(original_images[x], (1, 2, 0)))
        else:
            axs[0, x].imshow(original_images[x], cmap='binary_r')
        axs[0, x].axis('off')

        if colour:
            axs[1, x].imshow(np.transpose(inverted_images[x], (1, 2, 0)))
        else:
            axs[1, x].imshow(inverted_images[x], cmap='binary_r')
        axs[1, x].axis('off')
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    if filename != '':
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()