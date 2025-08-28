# utils/data_loader.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import config


class RentalDataset(Dataset):
    """
    PyTorch Dataset for rental data.
    """

    def __init__(self, dataframe, feature_cols, target_col):
        self.data = dataframe
        self.feature_cols = feature_cols
        self.target_col = target_col

        self.X = torch.tensor(self.data[self.feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(self.data[self.target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_raw_data(path=config.RAW_DATA_PATH):
    """
    Load raw CSV data into a Pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_csv(path)


def load_processed_data(path=config.PROCESSED_DATA_PATH):
    """
    Load processed CSV data into a Pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at {path}")
    return pd.read_csv(path)


def create_dataloaders(df, feature_cols, target_col, batch_size=config.BATCH_SIZE, test_size=config.TEST_SIZE):
    """
    Split data into train/test and create PyTorch DataLoaders.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=config.RANDOM_SEED)

    train_dataset = RentalDataset(train_df, feature_cols, target_col)
    test_dataset = RentalDataset(test_df, feature_cols, target_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
