import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    12-lead ECG Dataset loader for PTB-XL / Chapman-Shaoxing datasets.
    
    Expected signal dimensions: (12, 5000) at 500 Hz (10 seconds).
    Labels: 94 binary multi-label targets.
    """

    def __init__(self, csv_file, augment=False, root_dir=None):
        self.csv_path = Path(csv_file)
        self.df = pd.read_csv(self.csv_path)
        self.augment = augment
        self.root_dir = Path(root_dir) if root_dir else None

        # Extract all label columns in stable CSV order
        self.label_columns = [
            col for col in self.df.columns
            if col.startswith("label_")
        ]

        if len(self.label_columns) == 0:
            raise ValueError(f"No columns starting with 'label_' found in {csv_file}")

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, raw_path):
        p = Path(raw_path)
        if p.is_file():
            return p
        if self.root_dir:
            candidate = self.root_dir / raw_path
            if candidate.is_file():
                return candidate
            candidate_name = self.root_dir / p.name
            if candidate_name.is_file():
                return candidate_name
        # Check relative to CSV directory
        csv_dir = self.csv_path.parent
        candidate = csv_dir / raw_path
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"ECG record file not found: {raw_path}")

    def _apply_augmentations(self, signal):
        """
        Applies medically conservative, morphology-preserving perturbations:
        1. Low-level Gaussian noise (simulates electrode interface noise)
        2. Subtle amplitude scaling (simulates slight skin impedance changes)
        3. Low-frequency baseline wander (simulates respiration drift)
        """
        # 1. Subtle Gaussian noise (amplitude < 1% std)
        if np.random.rand() < 0.5:
            noise_std = np.random.uniform(0.002, 0.008)
            signal = signal + np.random.normal(0.0, noise_std, size=signal.shape).astype(np.float32)

        # 2. Mild amplitude scaling [0.97, 1.03]
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.97, 1.03)
            signal = signal * scale

        # 3. Baseline wander: low-frequency sinusoid (<0.4 Hz, where 5000 samples = 10s)
        if np.random.rand() < 0.3:
            num_samples = signal.shape[1]
            time = np.linspace(0, 10, num_samples, dtype=np.float32)
            freq = np.random.uniform(0.1, 0.35)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.02, 0.06)
            wander = amp * np.sin(2 * np.pi * freq * time + phase)
            signal = signal + wander[np.newaxis, :]

        return signal

    def __getitem__(self, index):
        row = self.df.iloc[index]
        mat_path = self._resolve_path(row["mat_path"])

        # Load ECG signal from MATLAB .mat
        mat_data = loadmat(str(mat_path))
        signal = mat_data["val"].astype(np.float32)

        # Lead-wise Z-score normalization: (x - mean) / (std + 1e-8)
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True) + 1e-8
        signal = (signal - mean) / std

        # Apply augmentation if enabled (training mode only)
        if self.augment:
            signal = self._apply_augmentations(signal)

        # Multi-label vector (94 floats)
        labels = row[self.label_columns].values.astype(np.float32)

        signal_tensor = torch.from_numpy(signal)
        labels_tensor = torch.from_numpy(labels)

        return signal_tensor, labels_tensor


if __name__ == "__main__":
    dataset = ECGDataset("train_split.csv", augment=False)
    signal, labels = dataset[0]
    print(f"Loaded dataset of length: {len(dataset):,}")
    print(f"Signal shape: {signal.shape}, Labels shape: {labels.shape}")
    print(f"Signal finite: {torch.isfinite(signal).all().item()}, Labels finite: {torch.isfinite(labels).all().item()}")