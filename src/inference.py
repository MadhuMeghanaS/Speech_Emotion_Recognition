"""
Stage 7: Run inference on a single new .wav file with a trained model.
This is the script you'd wrap in a small API/demo (e.g. Gradio, FastAPI).

Usage:
    python src/inference.py --audio path/to/clip.wav --model cnn_lstm
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from models import build_model
from utils import load_config


def extract_mfcc(path: str, sr: int, duration: float, n_mfcc: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr)
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).astype(np.float32)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc


def predict(audio_path: str, model_name: str, config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(cfg["models_dir"], f"{model_name}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    le = LabelEncoder()
    le.classes_ = pd.Index(ckpt["label_classes"])

    model = build_model(model_name, num_classes=len(le.classes_), **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mfcc = extract_mfcc(audio_path, cfg["data"]["sample_rate"], cfg["data"]["duration"], cfg["data"]["n_mfcc"])
    if model_name == "baseline_mlp":
        x = torch.tensor(mfcc.flatten(), dtype=torch.float32).unsqueeze(0)
    else:
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = le.classes_[pred_idx]
    prob_dict = {label: float(p) for label, p in zip(le.classes_, probs)}

    return {"predicted_emotion": pred_label, "probabilities": prob_dict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to a .wav file")
    parser.add_argument("--model", type=str, default="cnn_lstm",
                         choices=["baseline_mlp", "cnn2d", "cnn_lstm"])
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    result = predict(args.audio, args.model, args.config)
    print(f"\nPredicted emotion: {result['predicted_emotion']}\n")
    print("Class probabilities:")
    for label, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:10s}: {p:.3f}")
