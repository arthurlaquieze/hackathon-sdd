from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class SegmentationTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, labels):
        # Apply the same transform to both the image and the label(s)

        seed = torch.randint(2147483647, ())
        random_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.random.set_rng_state(random_state)
        torch.manual_seed(seed)
        labels = self.transform(labels)
        torch.random.set_rng_state(random_state)

        return image, labels


class OSSE_Dataset(Dataset):
    def __init__(self, OSSE_tensor, eddie_tensor, segmentation_transforms):
        self.OSSE_tensor = OSSE_tensor
        self.eddie_tensor = eddie_tensor
        self.segmentation_transforms = segmentation_transforms

    def __len__(self):
        return self.OSSE_tensor.shape[0]

    def __getitem__(self, idx):
        features = self.OSSE_tensor[idx, :, :, :]
        labels = self.eddie_tensor[idx, :, :, :]

        if self.segmentation_transforms is not None:
            # apply augmentations transforms
            features, labels = self.segmentation_transforms(features, labels)

        return features, labels


def get_data_loaders(
    batch_size,
    osse_xarray,
    eddies_xarray,
    osse_nan_value=None,
    eddies_nan_value=None,
    shuffle=True,
    augmentations=None,
):
    SegmentationTransforms = SegmentationTransform(augmentations)

    selected_var = ["vomecrtyT", "vozocrtxT", "sossheig", "votemper"]

    X_full = torch.tensor(
        osse_xarray.get(selected_var).to_array().to_numpy(), dtype=torch.float32
    )
    X_full = X_full.permute(1, 0, 2, 3)

    y_full = torch.tensor(eddies_xarray.to_array().to_numpy(), dtype=torch.float32)
    y_full = y_full.permute(1, 0, 2, 3)

    if osse_nan_value is not None:
        X_full = X_full.nan_to_num(osse_nan_value)

    if eddies_nan_value is not None:
        y_full = y_full.nan_to_num(eddies_nan_value)

    nb_val = X_full.shape[0]
    idx_split = int(0.8 * nb_val)
    X_train = X_full[:idx_split, :, :, :].clone().detach()
    y_train = y_full[:idx_split, :, :, :].clone().detach()
    X_val = X_full[idx_split:, :, :, :].clone().detach()
    y_val = y_full[idx_split:, :, :, :].clone().detach()

    ds_train = OSSE_Dataset(
        X_train, y_train, segmentation_transforms=SegmentationTransforms
    )
    train_dataloader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )
    ds_val = OSSE_Dataset(X_val, y_val, segmentation_transforms=SegmentationTransforms)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size, num_workers=4)

    return train_dataloader, val_dataloader


def get_xarray(parent_dir="./data"):
    eddies_train = xr.open_dataset(Path.joinpath(Path(parent_dir), "eddies_train.nc"))
    OSSE_test = xr.open_dataset(
        Path.joinpath(Path(parent_dir), "OSSE_U_V_SLA_SST_test.nc")
    )
    OSSE_train = xr.open_dataset(
        Path.joinpath(Path(parent_dir), "OSSE_U_V_SLA_SST_train.nc")
    )
    OSSE_train = OSSE_train.rename({"time_counter": "time"})

    return OSSE_train, eddies_train, OSSE_test


def normalize_osse(OSSE_train, OSSE_test):
    # Normalize OSSE data
    OSSE_train_mean = OSSE_train.mean()
    OSSE_train_std = OSSE_train.std()
    OSSE_train_norm = (OSSE_train - OSSE_train_mean) / OSSE_train_std

    OSSE_test_norm = (OSSE_test - OSSE_train_mean) / OSSE_train_std

    return OSSE_train_norm, OSSE_test_norm, OSSE_train_mean, OSSE_train_std
