"""
Exploratory Data Analysis for the RAVDESS dataset.

python notebooks/01_eda.py
"""

import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import build_manifest, load_config  # noqa: E402


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    records = build_manifest(cfg["data"]["raw_dir"], cfg["labels"])
    df = pd.DataFrame(records)

    print(f"Total clips: {len(df)}")
    print(df["emotion"].value_counts())
    print(df["gender"].value_counts())

    # 1. Class balance
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="emotion", order=df["emotion"].value_counts().index)
    plt.title("Emotion class distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.close()

    # 2. Waveform + spectrogram for one example per emotion
    fig, axes = plt.subplots(len(df["emotion"].unique()), 2, figsize=(10, 2.2 * df["emotion"].nunique()))
    for i, emotion in enumerate(sorted(df["emotion"].unique())):
        sample_path = df[df["emotion"] == emotion].iloc[0]["filepath"]
        y, sr = librosa.load(sample_path, sr=22050)

        librosa.display.waveshow(y, sr=sr, ax=axes[i, 0])
        axes[i, 0].set_title(f"{emotion} - waveform")

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=1.0)
        img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[i, 1])
        axes[i, 1].set_title(f"{emotion} - mel-spectrogram")

    plt.tight_layout()
    plt.savefig("emotion_examples.png", dpi=150)
    plt.close()

    print("Saved class_distribution.png and emotion_examples.png")


if __name__ == "__main__":
    main()
