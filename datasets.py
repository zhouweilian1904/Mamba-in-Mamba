# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
import cv2
from utils import tsne
from tqdm import tqdm
import visdom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import seaborn as sns

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file

DATASETS_CONFIG = {
    "PaviaC": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat",
        ],
        "img": "Pavia.mat",
        "gt": "Pavia_gt.mat",
    },
    "Salinas": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
        ],
        "img": "Salinas_corrected.mat",
        "gt": "Salinas_gt.mat",
    },
    "PaviaU": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
        ],
        "img": "PaviaU.mat",
        "gt": "PaviaU_gt.mat",
    },
    "KSC": {
        "urls": [
            "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
            "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat",
        ],
        "img": "KSC.mat",
        "gt": "KSC_gt.mat",
    },
    "IndianPines": {
        "urls": [
            "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
        ],
        "img": "Indian_pines_corrected.mat",
        "gt": "Indian_pines_gt.mat",
    },
    "Botswana": {
        "urls": [
            "http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat",
            "http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat",
        ],
        "img": "Botswana.mat",
        "gt": "Botswana_gt.mat",
    },
    'Houston2018': {
        'urls': [
        ],
        'img': '2018_IEEE_GRSS_DFC_HSI_TR.HDR',
        'gt': '2018_IEEE_GRSS_DFC_GT_TR.tif'
    },
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get("folder", dataset_name + "/")
    if dataset.get("download", True):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.makedirs(folder)
        for url in datasets[dataset_name]["urls"]:
            # download the files
            filename = url.split("/")[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(
                        unit="B",
                        unit_scale=True,
                        miniters=1,
                        desc="Downloading {}".format(filename),
                ) as t:
                    urlretrieve(url, filename=folder + filename, reporthook=t.update_to)
    elif not os.path.isdir(folder):
        print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == "PaviaC":
        # Load the image
        img = open_file(folder + "Pavia.mat")["pavia"]

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + "Pavia_gt.mat")["pavia_gt"]

        label_values = [
            "Undefined",
            "Water",
            "Trees",
            "Asphalt",
            "Self-Blocking Bricks",
            "Bitumen",
            "Tiles",
            "Shadows",
            "Meadows",
            "Bare Soil",
        ]

        ignored_labels = [0]

    elif dataset_name == "PaviaU":
        # Load the image
        img = open_file(folder + "PaviaU.mat")["paviaU"]

        rgb_bands = (25, 21, 12)

        gt = open_file(folder + "PaviaU_gt.mat")["paviaU_gt"]

        label_values = [
            "0.Undefined",
            "1.Asphalt",
            "2.Meadows",
            "3.Gravel",
            "4.Trees",
            "5.Painted metal sheets",
            "6.Bare Soil",
            "7.Bitumen",
            "8.Self-Blocking Bricks",
            "9.Shadows",
        ]

        ignored_labels = [0]
        palette = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 0),  # Dark Red
            8: (0, 128, 0),  # Dark Green
            9: (0, 0, 128),  # Dark Blue
        }


    elif dataset_name == "Salinas":
        img = open_file(folder + "Salinas_corrected.mat")["salinas_corrected"]

        rgb_bands = (23, 13, 3)  # AVIRIS sensor

        gt = open_file(folder + "Salinas_gt.mat")["salinas_gt"]

        label_values = [
            "0.Undefined",
            "1.Brocoli_green_weeds_1",
            "2.Brocoli_green_weeds_2",
            "3.Fallow",
            "4.Fallow_rough_plow",
            "5.Fallow_smooth",
            "6.Stubble",
            "7.Celery",
            "8.Grapes_untrained",
            "9.Soil_vinyard_develop",
            "10.Corn_senesced_green_weeds",
            "11.Lettuce_romaine_4wk",
            "12.Lettuce_romaine_5wk",
            "13.Lettuce_romaine_6wk",
            "14.Lettuce_romaine_7wk",
            "15.Vinyard_untrained",
            "16.Vinyard_vertical_trellis",
        ]

        ignored_labels = [0]
        palette = {0: (0, 0, 0),
                   1: (31, 119, 180),
                   2: (174, 199, 232),
                   3: (255, 127, 14),
                   4: (255, 187, 120),
                   5: (44, 160, 44),
                   6: (152, 223, 138),
                   7: (214, 39, 40),
                   8: (255, 152, 150),
                   9: (148, 103, 189),
                   10: (197, 176, 213),
                   11: (140, 86, 75),
                   12: (196, 156, 148),
                   13: (227, 119, 194),
                   14: (247, 182, 210),
                   15: (127, 127, 127),
                   16: (199, 199, 199),
                   }

    elif dataset_name == "IndianPines":
        # Load the image
        img = open_file(folder + "Indian_pines_corrected.mat")
        img = img["indian_pines_corrected"]

        rgb_bands = (3, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "Indian_pines_gt.mat")["indian_pines_gt"]
        label_values = [
            "0.Undefined",
            "1.Alfalfa",
            "2.Corn-notill",
            "3.Corn-mintill",
            "4.Corn",
            "5.Grass-pasture",
            "6.Grass-trees",
            "7.Grass-pasture-mowed",
            "8.Hay-windrowed",
            "9.Oats",
            "10.Soybean-notill",
            "11.Soybean-mintill",
            "12.Soybean-clean",
            "13.Wheat",
            "14.Woods",
            "15.Buildings-Grass-Trees-Drives",
            "16.Stone-Steel-Towers",
        ]

        ignored_labels = [0]
        palette = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 0),  # Dark Red
            8: (0, 128, 0),  # Dark Green
            9: (0, 0, 128),  # Dark Blue
            10: (128, 128, 0),  # Olive
            11: (128, 0, 128),  # Purple
            12: (0, 128, 128),  # Teal
            13: (192, 192, 192),  # Light Grey
            14: (64, 64, 64),  # Dark Grey
            15: (255, 128, 0),  # Orange
            16: (128, 128, 255)  # Light Blue
        }

    elif dataset_name == "Botswana":
        # Load the image
        img = open_file(folder + "Botswana.mat")["Botswana"]

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + "Botswana_gt.mat")["Botswana_gt"]
        label_values = [
            "Undefined",
            "Water",
            "Hippo grass",
            "Floodplain grasses 1",
            "Floodplain grasses 2",
            "Reeds",
            "Riparian",
            "Firescar",
            "Island interior",
            "Acacia woodlands",
            "Acacia shrublands",
            "Acacia grasslands",
            "Short mopane",
            "Mixed mopane",
            "Exposed soils",
        ]

        ignored_labels = [0]

    elif dataset_name == "KSC":
        # Load the image
        img = open_file(folder + "KSC.mat")["KSC"]

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + "KSC_gt.mat")["KSC_gt"]
        label_values = [
            "Undefined",
            "Scrub",
            "Willow swamp",
            "Cabbage palm hammock",
            "Cabbage palm/oak hammock",
            "Slash pine",
            "Oak/broadleaf hammock",
            "Hardwood swamp",
            "Graminoid marsh",
            "Spartina marsh",
            "Cattail marsh",
            "Salt marsh",
            "Mud flats",
            "Wate",
        ]

        ignored_labels = [0]

    elif dataset_name == 'Houston2013':
        # Load the image
        img = open_file(folder + 'Houston.mat')['Houston']

        rgb_bands = (59, 40, 23)
        # rgb_bands = (60, 40, 20)
        gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']

        label_values = ['0.Undefined', '1.Healthy grass', '2.Stressed grass', '3.Synthetic grass', '4.Trees',
                        '5.Soil', '6.Water', '7.Residential', '8.Commercial', '9.Road', '10.Highway',
                        '11.Railway', '12.Parking Lot1', '13.Parking Lot2', '14.Tennis court', '15.Running track']

        ignored_labels = [0]
        palette = {
            0: (0, 0, 0),  # Black
            1: (255, 0, 0),  # Red
            2: (0, 255, 0),  # Green
            3: (0, 0, 255),  # Blue
            4: (255, 255, 0),  # Yellow
            5: (255, 0, 255),  # Magenta
            6: (0, 255, 255),  # Cyan
            7: (128, 0, 0),  # Dark Red
            8: (0, 128, 0),  # Dark Green
            9: (0, 0, 128),  # Dark Blue
            10: (128, 128, 0),  # Olive
            11: (128, 0, 128),  # Purple
            12: (0, 128, 128),  # Teal
            13: (192, 192, 192),  # Light Grey
            14: (64, 64, 64),  # Dark Grey
            15: (255, 128, 0),  # Orange
        }

    elif dataset_name == 'Houston2018':
        # Load the image
        # img = open_file(folder + 'Houston.mat')['Houston']

        img = open_file(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')
        img = img[:, :, :48]
        print('img:', img.shape)
        gt = open_file(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
        print('gt', gt.shape)
        gt = cv2.resize(gt, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        print('gt', gt.shape)

        # rgb_bands = (39, 20, 3)
        rgb_bands = (29, 20, 3)

        # gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']

        label_values = ["0.Unclassified",
                        "1.Healthy grass",
                        "2.Stressed grass",
                        "3.Artificial turf",
                        "4.Evergreen trees",
                        "5.Deciduous trees",
                        "6.Bare earth",
                        "7.Water",
                        "8.Residential buildings",
                        "9.Non-residential buildings",
                        "10.Roads",
                        "11.Sidewalks",
                        "12.Crosswalks",
                        "13.Major thoroughfares",
                        "14.Highways",
                        "15.Railways",
                        "16.Paved parking lots",
                        "17.Unpaved parking lots",
                        "18.Cars",
                        "19.Trains",
                        "20.Stadium seats"]

        ignored_labels = [0]
        palette = {0: (0, 0, 0),
                   1: (141, 211, 199),
                   2: (255, 255, 179),
                   3: (190, 186, 218),
                   4: (251, 128, 114),
                   5: (128, 177, 211),
                   6: (253, 180, 98),
                   7: (179, 222, 105),
                   8: (252, 205, 229),
                   9: (217, 217, 217),
                   10: (188, 128, 189),
                   11: (204, 235, 197),
                   12: (255, 237, 111),
                   13: (251, 180, 174),
                   14: (179, 205, 227),
                   15: (204, 235, 197),
                   16: (222, 203, 228),
                   17: (254, 217, 166),
                   18: (255, 255, 204),
                   19: (229, 216, 189),
                   20: (253, 218, 236),
                   }

    else:
        # Custom dataset
        (
            img,
            gt,
            rgb_bands,
            ignored_labels,
            label_values,
            palette,
        ) = CUSTOM_DATASETS_CONFIG[dataset_name]["loader"](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN "
            "data is disabled."
        )
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype="float32")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    # num_groups, groups, group_lengths = adaptive_subband_groups(img, devide=5, min_bands=3, use_share=True, ovl_num=0)
    # print(f"Number of groups: {num_groups}")
    # for i, group in enumerate(groups):
    #     print(f"Group {i + 1}: {group}, Length: {group_lengths[i]}")

    # PCA need?
    img = applyPCA(img, 60)

    # tsne_on_image(img, gt, label_values, ignored_labels)
    return img, gt, label_values, ignored_labels, rgb_bands, palette


def tsne_on_image(img, gt, label_values, ignored_labels=None):
    # Create a mask where the ground truth is not ignored
    mask = ~np.isin(gt, ignored_labels)
    filtered_img = img[mask]
    filtered_gt = gt[mask]

    # Reshape img array for t-SNE
    X = filtered_img.reshape(-1, filtered_img.shape[-1])

    # Apply t-SNE
    tsne = TSNE(n_components=2, verbose=True)
    tsne_results = tsne.fit_transform(X)

    # Plotting with colors
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(filtered_gt)
    for label in unique_labels:
        indices = filtered_gt == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label_values[label])

    plt.title('t-SNE visualization with samples', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('t-SNE component 1', fontsize=20)
    plt.ylabel('t-SNE component 2', fontsize=20)
    plt.legend(fontsize='x-large')
    plt.show()


def applyPCA(X, numComponents):
    print('======>>> PCA ======>>>')
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams["dataset"]
        self.patch_size = hyperparams["patch_size"]
        self.ignored_labels = set(hyperparams["ignored_labels"])
        self.flip_augmentation = hyperparams["flip_augmentation"]
        self.radiation_augmentation = hyperparams["radiation_augmentation"]
        self.mixture_augmentation = hyperparams["mixture_augmentation"]
        self.center_pixel = hyperparams["center_pixel"]
        self.supervision = hyperparams["supervision"]
        # Fully supervised : use all pixels with label not ignored
        if self.supervision == "full":
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif self.supervision == "semi":
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array(
            [
                (x, y)
                for x, y in zip(x_pos, y_pos)
                if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
            ]
        )
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label
