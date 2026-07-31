"""
Stage 1: Preprocessing.

Walks the raw RAVDESS audio folder, extracts fixed-length MFCC and mel-spectrogram
features for every clip, and caches everything to disk as .npy arrays + a manifest
CSV so later stages (training) never need to touch librosa again.

Usage:
    python src/preprocess.py --config config.yaml
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import build_manifest, load_config, set_seed


def load_fixed_length_audio(path: str, sr: int, duration: float) -> np.ndarray:
    """Load audio, resample to `sr`, and pad/truncate to exactly `duration` seconds."""
    y, _ = librosa.load(path, sr=sr)
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y


def extract_features(y: np.ndarray, sr: int, n_mfcc: int, n_mels: int) -> dict:
    """Extract MFCCs (for classic CNN/LSTM models) and a mel-spectrogram
    (for the spectrogram-as-image CNN in the advanced stage)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return {"mfcc": mfcc.astype(np.float32), "mel": mel_db.astype(np.float32)}


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    os.makedirs(os.path.join(processed_dir, "mfcc"), exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "mel"), exist_ok=True)

    records = build_manifest(raw_dir, cfg["labels"])
    print(f"Found {len(records)} audio clips.")

    manifest_rows = []
    for i, rec in enumerate(tqdm(records, desc="Extracting features")):
        y = load_fixed_length_audio(rec["filepath"], cfg["data"]["sample_rate"], cfg["data"]["duration"])
        feats = extract_features(
            y, cfg["data"]["sample_rate"], cfg["data"]["n_mfcc"], cfg["data"]["n_mels"]
        )

        clip_id = f"clip_{i:05d}"
        mfcc_path = os.path.join(processed_dir, "mfcc", f"{clip_id}.npy")
        mel_path = os.path.join(processed_dir, "mel", f"{clip_id}.npy")
        np.save(mfcc_path, feats["mfcc"])
        np.save(mel_path, feats["mel"])

        manifest_rows.append({
            "clip_id": clip_id,
            "mfcc_path": mfcc_path,
            "mel_path": mel_path,
            "emotion": rec["emotion"],
            "intensity": rec["intensity"],
            "gender": rec["gender"],
            "actor": rec["actor"],
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(processed_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest with {len(manifest)} rows to {manifest_path}")
    print(manifest["emotion"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
