# Egyptian-Arabic-Speech-Emotion-Classifier

Predict the emotion in a therapy session recording using acoustic features and a trained Random Forest classifier.  
Give it a video or audio file — get back an emotion label per chunk and an overall majority prediction.

---

## How it works

1. Convert the input file to WAV (handles video and any audio format)
2. Resample to 16 kHz
3. Split into 15-second chunks with a 0.6 s silence offset
4. Extract acoustic features per chunk: ZCR · RMSE · MFCC · Mel Spectrogram
5. Predict emotion with a pre-trained Random Forest (majority vote across chunks)

See [docs/pipeline_flow.md](docs/pipeline_flow.md) for the full visual diagram.

---

## Project structure

```
speech-emotion-recognition-project/
├── config/
│   └── settings.py          # constants and env-var defaults
├── services/
│   ├── media_processor.py   # video → audio, WAV conversion
│   ├── feature_extractor.py # ZCR · RMSE · MFCC · Mel + augmentations
│   └── emotion_predictor.py # RF + scaler loader and prediction
├── pipeline/
│   └── speech_emotion_pipeline.py  # entry point
├── utils/
│   └── file_utils.py        # file type helpers
├── docs/
│   └── pipeline_flow.md     # Mermaid flow diagram
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/speech-emotion-recognition-project.git
cd speech-emotion-recognition-project
pip install -r requirements.txt
cp .env.example .env   # then fill in the paths
```

| Variable | Description |
|---|---|
| `FFMPEG_PATH` | Path to the FFmpeg `bin/` directory |
| `RF_MODEL_PATH` | Path to `best_random_forest_model.pkl` |
| `SCALER_PATH` | Path to `standard_scaler.pkl` |

---

## Usage

```python
from pipeline.speech_emotion_pipeline import predict_emotions_from_file

result = predict_emotions_from_file("session.mp4")

print(result["overall"])   # e.g. "sad"
print(result["chunks"])
# [
#   {"chunk": 1, "emotion": "sad"},
#   {"chunk": 2, "emotion": "neutral"},
#   {"chunk": 3, "emotion": "sad"},
# ]
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Pre-trained models

The `.pkl` files are **not** included (listed in `.gitignore`).  
Set their paths in `.env` or update `RF_MODEL_PATH` / `SCALER_PATH` in `config/settings.py`.

---

## Tech stack

| Layer | Library |
|---|---|
| Audio / video processing | librosa · pydub · MoviePy · soundfile |
| Emotion classification | scikit-learn (Random Forest · StandardScaler) |
| Deep learning runtime | PyTorch (device detection) |