"""
Stage 2: PyTorch Dataset definitions.

Two datasets are provided:
  - MFCCDataset: returns MFCC sequences, used by the baseline MLP and CNN-LSTM.
  - MelSpecDataset: returns mel-spectrograms reshaped as single-channel images,
    used by the "spectrogram-as-image" CNN (advanced stage).
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class MFCCDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, label_encoder: LabelEncoder, flatten: bool = False):
        self.manifest = manifest.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.flatten = flatten

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        mfcc = np.load(row["mfcc_path"])  # shape: (n_mfcc, time)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        if self.flatten:
            x = torch.tensor(mfcc.flatten(), dtype=torch.float32)
        else:
            # (channels=1, n_mfcc, time) — treat as a 1-channel "image" for CNN/CNN-LSTM
            x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        y = torch.tensor(self.label_encoder.transform([row["emotion"]])[0], dtype=torch.long)
        return x, y


class MelSpecDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, label_encoder: LabelEncoder):
        self.manifest = manifest.reset_index(drop=True)
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        mel = np.load(row["mel_path"])  # shape: (n_mels, time)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, time)
        y = torch.tensor(self.label_encoder.transform([row["emotion"]])[0], dtype=torch.long)
        return x, y


def get_splits(manifest_path: str, test_size: float, val_size: float, seed: int):
    """Stratified, actor-independent-ish split (simple stratify on emotion for clarity;
    see README for a note on actor-disjoint splitting for a stricter evaluation)."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(manifest_path)
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["emotion"], random_state=seed
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size, stratify=train_val["emotion"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
