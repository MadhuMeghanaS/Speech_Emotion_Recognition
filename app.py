"""
Gradio demo app - the "portfolio preview" layer.

Lets a visitor upload or record a short audio clip and see the predicted
emotion plus a confidence breakdown across all 8 classes, with waveform and
mel-spectrogram visualizations. This is what you screen-record / screenshot
for your README and LinkedIn post, and what you deploy to a free Hugging
Face Space to link from your resume.

Usage:
    python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

Note: requires a trained checkpoint at models/<model_name>_best.pt - run
src/preprocess.py and src/train.py first (see README).
"""

import os
import sys

import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models import build_model  # noqa: E402
from utils import load_config  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
CFG = load_config(CONFIG_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😮",
}
EMOTION_COLOR = {
    "neutral": "#9CA3AF", "calm": "#60A5FA", "happy": "#FBBF24", "sad": "#3B82F6",
    "angry": "#EF4444", "fearful": "#8B5CF6", "disgust": "#10B981", "surprised": "#F472B6",
}

_model_cache = {}


def load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]

    ckpt_path = os.path.join(CFG["models_dir"], f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        raise gr.Error(
            f"No trained checkpoint found at '{ckpt_path}'. Train a model first: "
            f"python src/train.py --model {model_name}"
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    le = LabelEncoder()
    le.classes_ = pd.Index(ckpt["label_classes"])

    model = build_model(model_name, num_classes=len(le.classes_), **ckpt["model_kwargs"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _model_cache[model_name] = (model, le)
    return model, le


def extract_mfcc(y: np.ndarray, sr: int, duration: float, n_mfcc: int) -> np.ndarray:
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).astype(np.float32)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc, y


def make_waveform_and_spectrogram_plot(y: np.ndarray, sr: int):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), facecolor="#0f172a")
    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#38bdf8")
    axes[0].set_title("Waveform", color="#e2e8f0", fontsize=10)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set_title("Mel-spectrogram", color="#e2e8f0", fontsize=10)

    plt.tight_layout()
    return fig


def make_probability_chart(prob_dict: dict):
    sorted_items = sorted(prob_dict.items(), key=lambda kv: kv[1])
    labels = [f"{EMOTION_EMOJI.get(k, '')}  {k}" for k, _ in sorted_items]
    values = [v * 100 for _, v in sorted_items]
    colors = [EMOTION_COLOR.get(k, "#38bdf8") for k, _ in sorted_items]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        margin=dict(l=10, r=40, t=10, b=10),
        height=320,
        xaxis=dict(range=[0, 100], title="Confidence (%)"),
        font=dict(size=13),
    )
    return fig


def predict(audio, model_name):
    if audio is None:
        raise gr.Error("Please upload or record an audio clip first.")

    sr_in, y_in = audio
    y_in = y_in.astype(np.float32)
    if y_in.ndim > 1:
        y_in = y_in.mean(axis=1)
    y_in = y_in / (np.max(np.abs(y_in)) + 1e-8)

    sr = CFG["data"]["sample_rate"]
    if sr_in != sr:
        y_in = librosa.resample(y_in, orig_sr=sr_in, target_sr=sr)

    model, le = load_model(model_name)
    mfcc, y_fixed = extract_mfcc(y_in, sr, CFG["data"]["duration"], CFG["data"]["n_mfcc"])

    if model_name == "baseline_mlp":
        x = torch.tensor(mfcc.flatten(), dtype=torch.float32).unsqueeze(0)
    else:
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = x.to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_dict = {label: float(p) for label, p in zip(le.classes_, probs)}
    pred_label = max(prob_dict, key=prob_dict.get)
    headline = f"{EMOTION_EMOJI.get(pred_label, '')}  **{pred_label.upper()}**  ({prob_dict[pred_label]*100:.1f}% confidence)"

    fig_wave = make_waveform_and_spectrogram_plot(y_fixed, sr)
    fig_probs = make_probability_chart(prob_dict)

    return headline, fig_probs, fig_wave


CUSTOM_CSS = """
.gradio-container {max-width: 980px !important; margin: auto;}
#title {text-align: center; margin-bottom: 0;}
#subtitle {text-align: center; color: #94a3b8; margin-top: 0; margin-bottom: 24px;}
#headline {font-size: 1.4rem; text-align: center; padding: 16px; border-radius: 12px;
           background: #1e293b; border: 1px solid #334155;}
footer {display: none !important;}
"""

with gr.Blocks(theme=gr.themes.Base(primary_hue="sky", neutral_hue="slate"), css=CUSTOM_CSS,
                title="Speech Emotion Recognition") as demo:
    gr.Markdown("# 🎙️ Speech Emotion Recognition", elem_id="title")
    gr.Markdown(
        "Deep learning model trained on the RAVDESS dataset · Upload or record a short "
        "clip and see the predicted emotion with a full confidence breakdown.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio input")
            model_dropdown = gr.Dropdown(
                choices=["cnn_lstm", "cnn2d", "baseline_mlp"],
                value="cnn_lstm",
                label="Model",
                info="cnn_lstm = best accuracy · baseline_mlp = fastest / simplest",
            )
            predict_btn = gr.Button("Analyze Emotion", variant="primary")

        with gr.Column(scale=1):
            headline_out = gr.Markdown(elem_id="headline")
            probs_plot = gr.Plot(label="Confidence by emotion")

    waveform_plot = gr.Plot(label="Waveform & Mel-spectrogram")

    predict_btn.click(
        fn=predict,
        inputs=[audio_input, model_dropdown],
        outputs=[headline_out, probs_plot, waveform_plot],
    )

    gr.Markdown(
        "---\n"
        "Built with PyTorch · MFCC / CNN-LSTM / wav2vec2 · "
        "[View source on GitHub](#) &nbsp;·&nbsp; "
        "Dataset: [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggleuwrfkaggle/ravdess-emotional-speech-audio)"
    )

if __name__ == "__main__":
    demo.launch()
