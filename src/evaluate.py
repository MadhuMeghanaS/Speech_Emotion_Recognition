"""
Stage 6: Evaluate a trained model on the held-out test split.
Produces accuracy, a full classification report, and a confusion matrix plot.

Usage:
    python src/evaluate.py --config config.yaml --model cnn_lstm
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from dataset import MFCCDataset
from models import build_model
from utils import load_config


def main(config_path: str, model_name: str):
    cfg = load_config(config_path)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(cfg["models_dir"], f"{model_name}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    le = LabelEncoder()
    le.classes_ = pd.Index(ckpt["label_classes"])

    model = build_model(model_name, num_classes=len(le.classes_), **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_df = pd.read_csv(os.path.join(cfg["data"]["processed_dir"], "test_split.csv"))
    flatten = model_name == "baseline_mlp"
    test_ds = MFCCDataset(test_df, le, flatten=flatten)
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    report = classification_report(all_labels, all_preds, target_names=le.classes_, digits=3)
    print(report)

    report_path = os.path.join(cfg["models_dir"], f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    fig_path = os.path.join(cfg["models_dir"], f"{model_name}_confusion_matrix.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved confusion matrix to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="cnn_lstm",
                         choices=["baseline_mlp", "cnn2d", "cnn_lstm"])
    args = parser.parse_args()
    main(args.config, args.model)
