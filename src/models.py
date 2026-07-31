"""
Stage 3: Model definitions, ordered by increasing sophistication.

  1. BaselineMLP     - flattened MFCCs -> dense layers. The "does this even work" baseline.
  2. CNN2D           - MFCCs treated as a single-channel image, 2D convolutions.
  3. CNN_LSTM        - CNN feature extractor over time steps, fed into an LSTM to model
                       temporal dynamics explicitly.
  4. Wav2Vec2Classifier - fine-tunes a pretrained HuggingFace wav2vec2 encoder with a
                       classification head on top. This is the "advanced" model and will
                       outperform everything above with far less feature engineering.
"""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN2D(nn.Module):
    """Treats the (n_mfcc x time) MFCC matrix as a 1-channel image."""

    def __init__(self, num_classes: int, n_mfcc: int = 40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)


class CNN_LSTM(nn.Module):
    """CNN extracts local spectral patterns per time-frame, LSTM models how those
    patterns evolve over the utterance."""

    def __init__(self, num_classes: int, n_mfcc: int = 40, lstm_hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=lstm_hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, n_mfcc, time) -> (batch, n_mfcc, time)
        x = x.squeeze(1)
        x = self.conv(x)                    # (batch, channels, time')
        x = x.permute(0, 2, 1)              # (batch, time', channels) for LSTM
        out, (h_n, _) = self.lstm(x)
        # concat final forward/backward hidden states
        final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(final)


class Wav2Vec2Classifier(nn.Module):
    """Advanced model: fine-tunes facebook/wav2vec2-base on raw waveforms.
    Requires raw audio input (not MFCCs) - see src/train_wav2vec2.py for the
    matching data pipeline, which is kept separate since it needs raw waveforms
    rather than the cached MFCC/mel features.
    """

    def __init__(self, num_classes: int, freeze_feature_extractor: bool = True,
                 pretrained_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_name)
        if freeze_feature_extractor:
            self.encoder.feature_extractor._freeze_parameters()

        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)  # mean-pool over time
        return self.classifier(pooled)


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    registry = {
        "baseline_mlp": BaselineMLP,
        "cnn2d": CNN2D,
        "cnn_lstm": CNN_LSTM,
        "wav2vec2": Wav2Vec2Classifier,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry.keys())}")
    return registry[name](num_classes=num_classes, **kwargs)
