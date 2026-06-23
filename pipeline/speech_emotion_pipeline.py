import os
import shutil
import tempfile

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

# ── FFmpeg setup ─────────────────────────────────────────────────────────────
from config.settings import FFMPEG_DIR, CHUNK_LENGTH_SEC, OFFSET_SEC, TARGET_SAMPLE_RATE

if FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

if not shutil.which("ffmpeg"):
    raise EnvironmentError("FFmpeg not found — set FFMPEG_PATH in your .env file.")

from services.media_processor import prepare_audio
from services.feature_extractor import extract_features
from services.emotion_predictor import predict_emotion


def predict_emotions_from_file(file_path: str) -> dict:
    """
    Running the full emotion recognition pipeline on a single media file.

    Parameters
    ----------
    file_path : str
        Path to a video (.mp4, .avi, …) or audio (.wav, .mp3, …) file.

    Returns
    -------
    list of {"chunk": int, "emotion": str} — one entry per 15-sec window
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with tempfile.TemporaryDirectory() as tmpdir:

        # 1 ── Convert to WAV ─────────────────────────────────────────────────
        wav_path = prepare_audio(file_path)

        # 2 ── Resample to 16 kHz ─────────────────────────────────────────────
        y16, _ = librosa.load(wav_path, sr=TARGET_SAMPLE_RATE)
        resampled_path = os.path.join(tmpdir, "resampled.wav")
        sf.write(resampled_path, y16, TARGET_SAMPLE_RATE)

        # 3 ── Load audio for feature extraction ─────────────────────────────
        data, sr = librosa.load(resampled_path, sr=TARGET_SAMPLE_RATE)

        # 4 ── Split into 15-sec chunks ───────────────────────────────────────
        segment_length = CHUNK_LENGTH_SEC * sr
        offset_samples = int(OFFSET_SEC * sr)

        audio_chunks = [
            data[i + offset_samples : i + segment_length]
            for i in range(0, len(data), segment_length)
            if len(data[i + offset_samples : i + segment_length]) > 0
        ]

        if not audio_chunks:
            raise RuntimeError("No audio chunks could be extracted from the file.")

        # 5 ── Predict emotion per chunk ──────────────────────────────────────
        predictions = []
        for idx, chunk in enumerate(audio_chunks):
            if len(chunk) == 0 or np.max(np.abs(chunk)) == 0:
                predictions.append("unknown")
                continue
            try:
                chunk = chunk / np.max(np.abs(chunk))
                signal = np.zeros(int(sr * 6))
                signal[: min(len(signal), len(chunk))] = chunk[: min(len(signal), len(chunk))]
                features = extract_features(signal, sr).reshape(1, -1)
                predictions.append(predict_emotion(features))
            except Exception as e:
                print(f"⚠️  Chunk {idx + 1} failed: {e}")
                predictions.append("unknown")

        return [
            {"chunk": i + 1, "emotion": emotion}
            for i, emotion in enumerate(predictions)
        ]