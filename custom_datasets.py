from utils import open_file
import numpy as np

CUSTOM_DATASETS_CONFIG = {
    "Houston2018": {
        "img": "2018_IEEE_GRSS_DFC_HSI_TR.HDR",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
        "download": False,
        "loader": lambda folder: dfc2018_loader(folder),
    }
}


def dfc2018_loader(folder):
    img = open_file(folder + "2018_IEEE_GRSS_DFC_HSI_TR.HDR")[:, :, :-2]
    gt = open_file(folder + "2018_IEEE_GRSS_DFC_GT_TR.tif")
    gt = gt.astype("uint8")

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unclassified",
        "Healthy grass",
        "Stressed grass",
        "Artificial turf",
        "Evergreen trees",
        "Deciduous trees",
        "Bare earth",
        "Water",
        "Residential buildings",
        "Non-residential buildings",
        "Roads",
        "Sidewalks",
        "Crosswalks",
        "Major thoroughfares",
        "Highways",
        "Railways",
        "Paved parking lots",
        "Unpaved parking lots",
        "Cars",
        "Trains",
        "Stadium seats",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette


CUSTOM_DATASETS_CONFIG = {
         'Houston2013': {
            'img': 'Houston.mat',
            'gt': 'Houston_gt.mat',
            'download': False,
            'loader': lambda folder: Houston_loader(folder)
            }
    }

def Houston_loader(folder):
        img = open_file(folder + 'Houston.mat')[:,:,:-2]
        gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']
        gt = gt.astype('uint8')

        rgb_bands = (59, 40, 23)

        label_values = ['Undefined','Healthy grass','Stressed grass','Synthetic grass','Trees',
         'Soil','Water','Residential','Commercial','Road','Highway',
         'Railway','Parking Lot1','Parking Lot2','Tennis court','Running track']

        ignored_labels = [0]
        return img, gt, rgb_bands, ignored_labels, label_values, palette