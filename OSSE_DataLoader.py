from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class OSSE_Dataset(Dataset):
    def __init__(self, OSSE_tensor, eddie_tensor):
        self.OSSE_tensor = OSSE_tensor
        self.eddie_tensor = eddie_tensor

    def __len__(self):
        return self.OSSE_tensor.shape[0]

    def __getitem__(self, idx):
        features = self.OSSE_tensor[idx, :, :, :]
        label = self.eddie_tensor[idx, :, :, :]
        return features, label


def get_data_loaders(
    batch_size, osse_xarray, eddies_xarray, osse_nan_value=None, eddies_nan_value=None
):
    selected_var = ["vomecrtyT", "vozocrtxT", "sossheig", "votemper"]

    X_full = torch.tensor(osse_xarray.get(selected_var).to_array().to_numpy())
    X_full = X_full.permute(1, 0, 2, 3)

    y_full = torch.tensor(eddies_xarray.to_array().to_numpy())
    y_full = y_full.permute(1, 0, 2, 3)

    if osse_nan_value is not None:
        X_full = X_full.nan_to_num(osse_nan_value)

    if eddies_nan_value is not None:
        y_full = y_full.nan_to_num(eddies_nan_value)

    nb_val = X_full.shape[0]
    idx_split = int(0.8 * nb_val)
    X_train = torch.tensor(X_full[:idx_split, :, :, :])
    y_train = torch.tensor(y_full[:idx_split, :, :, :])
    X_val = torch.tensor(X_full[idx_split:, :, :, :])
    y_val = torch.tensor(y_full[idx_split:, :, :, :])

    ds_train = OSSE_Dataset(X_train, y_train)
    train_dataloader = DataLoader(ds_train, batch_size=batch_size)
    ds_val = OSSE_Dataset(X_val, y_val)
    val_dataloader = DataLoader(ds_val, batch_size=batch_size)

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
