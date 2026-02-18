import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from Util.vis_util import class_colors

def map_points_to_grid(points, labels, grid_size=500):
    """
    Map the data points and labels to a grid of the same size and shape as the boundary map, e.g: if a boundary map is (500,500), it will create a (500,500,N,2) grid, of 2d points (where N is the largest number of points per pixel).
    Args:
        points: The 2d points to be sorted
        labels: The corresponding labels of the 2d points to be sorted
        grid_size: The grid size of the grid, default is 500

    Returns: A grid of 2d points and a grid of corresponding labels.
    """
    new_grid = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    label_grid = [[[] for _ in range(grid_size)] for _ in range(grid_size)]

    ix = np.floor(points[:, 0] * grid_size).astype(int)
    iy = np.floor(points[:, 1] * grid_size).astype(int)

    ix = np.clip(ix, 0, grid_size - 1)
    iy = np.clip(iy, 0, grid_size - 1)

    for point, label, i, j in zip(points, labels, ix, iy):
        new_grid[j][i].append(point)
        label_grid[j][i].append(label)

    return new_grid, label_grid

def calculate_boundary_map_precision_recall(boundary_map, label_grid, c=0):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    grid_size = boundary_map.shape[0]
    for x in range(grid_size):
        for y in range(grid_size):
            pixel = boundary_map[x, y]
            grid_labels = label_grid[x][y]
            if not grid_labels:
                pass
            else:
                for l in grid_labels:
                    if (l == c) and (pixel == c): # e.g: if a yellow coloured point is behind a yellow pixel
                        true_positives += 1
                    elif (pixel == c) and (l != c): # e.g: if a non-yellow point is behind a yellow pixel
                        false_positives += 1
                    elif (pixel != c) and (l == c): # e.g: If a yellow point is behind a non-yellow pixel
                        false_negatives += 1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    return precision, recall

def calculate_accuracy(boundary_map, label_grid):
    """
    Calculates the accuracy of the boundary map based on how many real data point labels match their background pixel
    Args:
        boundary_map: The boundary map as a numpy array of size (grid_size, grid_size)
        label_grid: A numpy array of size (grid_size, grid_size, N), where N is the largest number of samples per point

    Returns: The number of data points which match their background pixel

    """
    matching = 0
    grid_size = boundary_map.shape[0]
    for x in range(grid_size):
        for y in range(grid_size):
            pixel = boundary_map[x, y]
            grid_labels = label_grid[x][y]
            if not grid_labels:
                pass
            else:
                for l in grid_labels:
                    if l == pixel:
                        matching += 1
    return matching

def map_to_witness_grid(boundary_map, points_grid, patch_size=100):
    grid_size = boundary_map.shape[0]
    points_patches = []
    map_patches = []

    for x in range(0, grid_size, patch_size):
        for y in range(0, grid_size, patch_size):
            p_patch = points_grid[x:x + patch_size][y:y + patch_size]
            m_patch = boundary_map[x:x + patch_size, y:y + patch_size].astype(float)
            if m_patch.shape != (patch_size, patch_size):
                pad_x = patch_size - m_patch.shape[0]
                pad_y = patch_size - m_patch.shape[1]
                m_patch = np.pad(
                    m_patch,
                    ((0, pad_x), (0, pad_y)),
                    mode='constant',
                    constant_values=np.nan
                )
            map_patches.append(m_patch)
            points_patches.append(p_patch)

    patch_grid_size = int(np.ceil(grid_size / patch_size))
    map_patches = np.array(map_patches).reshape(patch_grid_size, patch_grid_size, patch_size, patch_size)
    #points_patches = np.array(points_patches).reshape(patch_grid_size, patch_grid_size,patch_size, patch_size)
    print(map_patches.shape)

    sc_cmap = colors.ListedColormap([colors.hsv_to_rgb(hsv) for hsv in class_colors])
    sc_cmap.set_bad(color='black')
    norm = colors.Normalize(vmin=0, vmax=9)

    fig, axs = plt.subplots(nrows=patch_grid_size, ncols=patch_grid_size, figsize=(10, 10))
    for i in range(patch_grid_size):
        for j in range(patch_grid_size):
            axs[i,j].imshow(map_patches[i,j],cmap=sc_cmap, norm=norm)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
    plt.show()