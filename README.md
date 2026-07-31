# Speech Emotion Recognition (RAVDESS)

Deep learning project that classifies the emotion expressed in a short speech
clip — neutral, calm, happy, sad, angry, fearful, disgust, or surprised —
directly from raw audio. Built as a progression from a simple baseline to a
fine-tuned transformer, with every stage benchmarked on the same held-out test set.


## Dataset

**[RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://www.kaggle.com/datasets/uwrfkaggleuwrfkaggle/ravdess-emotional-speech-audio)**

- 24 professional actors (12 male, 12 female), North American accent
- 1,440 speech audio files (plus a song subset, not used here)
- 8 emotion classes, 2 intensity levels (normal / strong)
- ~1 GB, WAV format, 48kHz (resampled to 22.05kHz in preprocessing)

### Getting the data

```bash
# Option A: Kaggle CLI (needs ~/.kaggle/kaggle.json with your API token)
kaggle datasets download -d uwrfkaggleuwrfkaggle/ravdess-emotional-speech-audio
unzip ravdess-emotional-speech-audio.zip -d data/raw


## Project structure

```
speech-emotion-recognition/
├── config.yaml                  # all hyperparameters and paths in one place
├── requirements.txt
├── data/
│   ├── raw/                     # downloaded RAVDESS files go here (gitignored)
│   └── processed/                # cached MFCC/mel features + manifest (gitignored)
├── notebooks/
│   └── 01_eda.py                 # exploratory data analysis, class balance, waveform plots
├── src/
│   ├── utils.py                  # config loading, seeding, RAVDESS filename parsing
│   ├── preprocess.py             # extracts & caches MFCC + mel-spectrogram features
│   ├── dataset.py                 # PyTorch Dataset classes + stratified train/val/test split
│   ├── models.py                  # BaselineMLP, CNN2D, CNN_LSTM, Wav2Vec2Classifier
│   ├── train.py                   # training loop for MLP / CNN2D / CNN-LSTM
│   ├── train_wav2vec2.py          # fine-tuning loop for the transformer model
│   ├── evaluate.py                # test-set metrics + confusion matrix
│   └── inference.py                # predict emotion on a single new .wav file
└── models/                        # saved checkpoints, metrics, plots (gitignored)
```

## Pipeline / how to run

```bash
pip install -r requirements.txt

# 1. Explore the data
python notebooks/01_eda.py

# 2. Extract and cache MFCC + mel-spectrogram features (one-time, ~5-10 min)
python src/preprocess.py --config config.yaml

# 3. Train a model (pick one)
python src/train.py --config config.yaml --model baseline_mlp
python src/train.py --config config.yaml --model cnn2d
python src/train.py --config config.yaml --model cnn_lstm

# 4. Fine-tune the advanced transformer model
python src/train_wav2vec2.py --config config.yaml

# 5. Evaluate on the held-out test set
python src/evaluate.py --config config.yaml --model cnn_lstm

# 6. Run inference on a new audio clip
python src/inference.py --audio path/to/clip.wav --model cnn_lstm

# 7. Launch the interactive demo (Gradio UI, for portfolio preview)
python app.py
```


## Model progression

| Stage | Model | Input | Idea |
|---|---|---|---|
| 1. Baseline | `BaselineMLP` | Flattened MFCCs | Sanity-check baseline — no spatial/temporal structure used |
| 2. Intermediate | `CNN2D` | MFCC as image | Learns local time-frequency patterns via 2D convolutions |
| 3. Intermediate+ | `CNN_LSTM` | MFCC sequence | CNN extracts per-frame features, LSTM models how they evolve over the utterance |
| 4. Advanced | `Wav2Vec2Classifier` | Raw waveform | Fine-tunes a pretrained self-supervised speech transformer (`facebook/wav2vec2-base`) |


## Future improvements

- Actor-disjoint train/test split (see above) for a stricter generalization estimate
- Data augmentation: pitch shift, time stretch, added background noise
- Attention pooling instead of mean pooling for the wav2vec2 head
- Multi-modal fusion with the RAVDESS video/facial-expression data
- Small Gradio demo for live microphone inference

## Tech stack

Python, PyTorch, librosa, HuggingFace Transformers (wav2vec2), scikit-learn,
matplotlib/seaborn.


