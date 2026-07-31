"""
Shared helper utilities: config loading, seeding, and RAVDESS filename parsing.

RAVDESS filenames encode metadata in their name, e.g.:
    03-01-06-01-02-01-12.wav
Positions (dash-separated):
    1: Modality        (01=full-AV, 02=video-only, 03=audio-only)
    2: Vocal channel    (01=speech, 02=song)
    3: Emotion          (01=neutral ... 08=surprised)
    4: Emotional intensity (01=normal, 02=strong)
    5: Statement
    6: Repetition
    7: Actor            (odd=male, even=female)
"""

import os
import random
import yaml
import numpy as np


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_ravdess_filename(filename: str, label_map: dict) -> dict:
    """Parse a RAVDESS filename into its metadata fields."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename format: {filename}")

    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = parts

    return {
        "filepath": filename,
        "modality": modality,
        "vocal_channel": "speech" if vocal_channel == "01" else "song",
        "emotion_code": emotion_code,
        "emotion": label_map[emotion_code],
        "intensity": "normal" if intensity == "01" else "strong",
        "statement": statement,
        "repetition": repetition,
        "actor": int(actor),
        "gender": "male" if int(actor) % 2 == 1 else "female",
    }


def build_manifest(raw_dir: str, label_map: dict):
    """Walk the raw RAVDESS directory and build a list of parsed file records.
    RAVDESS ships as Actor_01 ... Actor_24 subfolders, each full of .wav files.
    """
    records = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                full_path = os.path.join(root, f)
                try:
                    records.append(parse_ravdess_filename(full_path, label_map))
                except ValueError:
                    continue
    if not records:
        raise FileNotFoundError(
            f"No .wav files found under '{raw_dir}'. Did you download and unzip the "
            f"RAVDESS dataset there? See README.md 'Getting the data' section."
        )
    return records
