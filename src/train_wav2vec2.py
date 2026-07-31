"""
Stage 5 (Advanced): Fine-tune a pretrained wav2vec2 encoder on raw waveforms.

This is the model that will give you the best accuracy and the best portfolio
"talking point" - it shows you can go beyond hand-crafted features (MFCCs) and
fine-tune a transformer-based speech model, which is current industry practice.

Requires raw audio, so this script reads .wav files directly rather than using
the cached MFCC/mel features from preprocess.py.

Usage:
    python src/train_wav2vec2.py --config config.yaml
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor

from models import build_model
from utils import build_manifest, load_config, set_seed


class RawAudioDataset(Dataset):
    def __init__(self, records, label_encoder, feature_extractor, sr, duration):
        self.records = records
        self.le = label_encoder
        self.fe = feature_extractor
        self.sr = sr
        self.target_len = int(sr * duration)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        y, _ = librosa.load(rec["filepath"], sr=self.sr)
        if len(y) > self.target_len:
            y = y[: self.target_len]
        else:
            y = np.pad(y, (0, self.target_len - len(y)))

        inputs = self.fe(y, sampling_rate=self.sr, return_tensors="pt", padding=False)
        x = inputs.input_values.squeeze(0)
        label = torch.tensor(self.le.transform([rec["emotion"]])[0], dtype=torch.long)
        return x, label


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = build_manifest(cfg["data"]["raw_dir"], cfg["labels"])
    all_emotions = sorted({r["emotion"] for r in records})
    le = LabelEncoder().fit(all_emotions)

    from sklearn.model_selection import train_test_split
    train_recs, temp_recs = train_test_split(
        records, test_size=cfg["data"]["test_size"] + cfg["data"]["val_size"],
        stratify=[r["emotion"] for r in records], random_state=cfg["seed"],
    )
    val_recs, test_recs = train_test_split(
        temp_recs, test_size=0.5,
        stratify=[r["emotion"] for r in temp_recs], random_state=cfg["seed"],
    )

    pretrained_name = "facebook/wav2vec2-base"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_name)

    train_ds = RawAudioDataset(train_recs, le, feature_extractor, cfg["data"]["sample_rate"], cfg["data"]["duration"])
    val_ds = RawAudioDataset(val_recs, le, feature_extractor, cfg["data"]["sample_rate"], cfg["data"]["duration"])

    # Smaller batch size than the MFCC models - raw waveforms + wav2vec2 are far more memory hungry
    batch_size = max(4, cfg["train"]["batch_size"] // 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model("wav2vec2", num_classes=len(le.classes_),
                         freeze_feature_extractor=True, pretrained_name=pretrained_name).to(device)

    criterion = nn.CrossEntropyLoss()
    # Small LR since we're fine-tuning a pretrained encoder, not training from scratch
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    best_val_loss = float("inf")
    os.makedirs(cfg["models_dir"], exist_ok=True)
    best_path = os.path.join(cfg["models_dir"], "wav2vec2_best.pt")

    epochs = min(15, cfg["train"]["epochs"])  # fine-tuning converges much faster than training from scratch
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss, train_acc = total_loss / total, correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += x.size(0)
        val_loss, val_acc = v_loss / v_total, v_correct / v_total

        print(f"Epoch {epoch+1:02d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "label_classes": le.classes_.tolist(),
                        "pretrained_name": pretrained_name}, best_path)
            print(f"  -> saved new best model to {best_path}")

    pd.DataFrame(test_recs).to_csv(
        os.path.join(cfg["data"]["processed_dir"], "test_split_raw.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
