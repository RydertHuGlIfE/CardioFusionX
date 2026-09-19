import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import pandas as pd
import numpy as np


class ECGDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        self.label_columns = [
            col for col in self.df.columns
            if col.startswith("label_")
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # Load ECG signal
        mat_data = loadmat(row["mat_path"])
        signal = mat_data["val"].astype(np.float32)

        # Normalize each lead
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True) + 1e-8

        signal = (signal - mean) / std

        # Labels (94)
        labels = row[self.label_columns].values.astype(np.float32)

        signal = torch.tensor(signal, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return signal, labels


if __name__ == "__main__":

    dataset = ECGDataset("train_split.csv")

    signal, labels = dataset[0]

    print("Dataset size:", len(dataset))
    print("Signal shape:", signal.shape)
    print("Labels shape:", labels.shape)
    