import numpy as np
import torch
import copy
from scipy.io import loadmat
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

sns.set_style('darkgrid')

# Load the data
im_mat1 = loadmat('Datasets/IndianPines/Indian_pines_corrected.mat')  # 原数据
image1 = im_mat1['indian_pines_corrected']

# im_mat1 = loadmat('Datasets/regression data.mat')  # 回归数据
# image1 = im_mat1['regs']

im_mat2 = loadmat('Datasets/IndianPines/Indian_pines_gt.mat')
image2 = im_mat2["indian_pines_gt"]

# Create a mask where the ground truth is greater than 0
d = np.zeros_like(image2)
d[image2 > 0] = True
d = torch.from_numpy(d)
d = d.numpy()
mask1 = copy.deepcopy(d)
mask1 = mask1.astype(bool)
mask = copy.deepcopy(d)
mask = mask.astype(bool)
image1 = image1.transpose(2, 0, 1)

# Prepare the data for T-SNE
for i in range(200):  # Change to the bands of the dataset e.g. 200,103,144
    img = image1[i, :, :]
    img = img.astype(np.float32)
    img = torch.from_numpy(img).squeeze(0)
    img = img.numpy()
    b = img[mask]
    b = b.reshape(1, -1)
    if i == 0:
        out = b
        continue
    out = np.concatenate((out, b), 0)

out = out.transpose(1, 0)
y = image2[mask1]

# 2D T-SNE function
def t_sne_2d(latent_vecs, y):
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    print(latent_vecs_reduced.shape)

    # Define color mapping for each class
    colors = ['red', 'green', 'yellow', 'maroon', 'black', 'cyan', 'blue', 'gray', 'tan',
              'navy', 'bisque', 'magenta', 'orange', 'darkviolet', 'khaki', 'lightgreen']

    # Ensure you have enough colors for all labels
    unique_labels = np.unique(y)
    if len(colors) < len(unique_labels):
        raise ValueError("Not enough colors specified for the number of classes")

    plt.figure(figsize=(10, 10))
    # Plot each class with a unique color
    for idx, cls in enumerate(unique_labels):
        indices = y == cls
        plt.scatter(latent_vecs_reduced[indices, 0], latent_vecs_reduced[indices, 1],
                    color=colors[idx], label=f'Class {cls}')

    plt.title('t-SNE visualization with test samples', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('t-SNE component 1', fontsize=20)
    plt.ylabel('t-SNE component 2', fontsize=20)
    plt.legend(fontsize='x-large')
    plt.show()

t_sne_2d(out, y)
