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


class XarrayDataset(Dataset):
    """A simple torch Dataset wrapping xr.DataArray"""

    def __init__(self, ar, batch_dim):
        self.ar = ar
        self.batch_dim = batch_dim

    def __len__(self):
        return len(self.ar[self.batch_dim])

    def __getitem__(self, idx):
        return self.ar[{self.batch_dim: idx}].values


class XarrayDataLoader(DataLoader):
    """A simple torch DataLoader wrapping xr.DataArray"""

    def __init__(self, ar, batch_dim, **kwargs):
        ar = XarrayDataset(ar, batch_dim)
        super().__init__(ar, **kwargs)
