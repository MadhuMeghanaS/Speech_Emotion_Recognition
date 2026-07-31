"""
Stage 4: Training loop for the baseline MLP, CNN2D, and CNN-LSTM models.
(For the wav2vec2 fine-tune, see train_wav2vec2.py - it needs a different data
pipeline since it consumes raw waveforms instead of cached MFCC/mel features.)

Usage:
    python src/train.py --config config.yaml --model cnn_lstm
    python src/train.py --config config.yaml --model baseline_mlp
    python src/train.py --config config.yaml --model cnn2d
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MFCCDataset, get_splits
from models import build_model
from utils import load_config, set_seed


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def main(config_path: str, model_name: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest_path = os.path.join(cfg["data"]["processed_dir"], "manifest.csv")
    train_df, val_df, test_df = get_splits(
        manifest_path, cfg["data"]["test_size"], cfg["data"]["val_size"], cfg["seed"]
    )

    le = LabelEncoder()
    le.fit(train_df["emotion"])
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}")

    flatten = model_name == "baseline_mlp"
    train_ds = MFCCDataset(train_df, le, flatten=flatten)
    val_ds = MFCCDataset(val_df, le, flatten=flatten)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model_kwargs = {}
    if model_name == "baseline_mlp":
        sample_x, _ = train_ds[0]
        model_kwargs["input_dim"] = sample_x.shape[0]
    else:
        model_kwargs["n_mfcc"] = cfg["data"]["n_mfcc"]

    model = build_model(model_name, num_classes, **model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    os.makedirs(cfg["models_dir"], exist_ok=True)
    best_path = os.path.join(cfg["models_dir"], f"{model_name}_best.pt")

    for epoch in range(cfg["train"]["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "label_classes": le.classes_.tolist(),
                        "model_name": model_name, "model_kwargs": model_kwargs}, best_path)
            print(f"  -> saved new best model to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["patience"]:
                print(f"Early stopping at epoch {epoch+1} (no val improvement for "
                      f"{cfg['train']['patience']} epochs)")
                break

    history_path = os.path.join(cfg["models_dir"], f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Also persist the test split so evaluate.py uses the exact same held-out set
    test_df.to_csv(os.path.join(cfg["data"]["processed_dir"], "test_split.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="cnn_lstm",
                         choices=["baseline_mlp", "cnn2d", "cnn_lstm"])
    args = parser.parse_args()
    main(args.config, args.model)
