import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# import Dataset and DataLoader
from torch.utils.data import Dataset, DataLoader
import torch

# # import dataset from ./data/eddies_train.nc with xarray
# eddies_train = xr.open_dataset("./data/eddies_train.nc")

# # import the OSSE_U_V_SLA_SST_train.nc dataset
# eddies_train = xr.open_dataset("./data/eddies_train.nc")
# OSSE_test = xr.open_dataset("./data/OSSE_U_V_SLA_SST_test.nc")
# OSSE_train = xr.open_dataset("./data/OSSE_U_V_SLA_SST_train.nc")
# OSSE_train = OSSE_train.rename({"time_counter": "time"})

from torch.utils.data import Dataset, DataLoader
import xarray as xr


class OSSE_Dataset(Dataset):
    def __init__(self, OSSE_tensor, eddie_tensor):
        self.OSSE_tensor = OSSE_tensor
        self.eddie_tensor = eddie_tensor

    def __len__(self):
        return len(self.OSSE_tensor.shape[0])

    def __getitem__(self, idx):
        features = self.OSSE_tensor[idx, :, :, :]
        label = self.eddie_tensor[idx, :, :, :]
        return features, label


class OSSE_DataLoader:
    def __init__(self, files, selected_var):
        self.files = files
        self.selected_var = selected_var

    def get_data(self):
        eddies_train = xr.open_dataset(self.files[0])
        OSSE_test = xr.open_dataset(self.files[1])
        OSSE_train = xr.open_dataset(self.files[2])
        OSSE_train = OSSE_train.rename({"time_counter": "time"})
        selected_var = ["vomecrtyT", "vozocrtxT", "sossheig", "votemper"]
        X_full = torch.tensor(OSSE_train.get(self.selected_var).to_array().to_numpy())
        X_full = X_full.permute(1, 0, 2, 3)
        na_value = 999.0
        # eddies_train = eddies_train.fillna(na_value) # fill coast with 999.
        y_full = torch.tensor(eddies_train.to_array().to_numpy())
        y_full = X_full.permute(1, 0, 2, 3)

        nb_val = X_full.shape[0]
        idx_split = int(0.8 * nb_val)
        X_train = torch.tensor(X_full[:idx_split, :, :, :])
        y_train = torch.tensor(y_full[:idx_split, :, :, :])
        X_val = torch.tensor(X_full[idx_split:, :, :, :])
        y_val = torch.tensor(y_full[idx_split:, :, :, :])

        ds_train = OSSE_Dataset(X_train, y_train)
        train_dataloader = DataLoader(ds_train, batch_size=4)
        ds_val = OSSE_Dataset(X_val, y_val)
        val_dataloader = DataLoader(ds_val, batch_size=4)

        return train_dataloader, val_dataloader
