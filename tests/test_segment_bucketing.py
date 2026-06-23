import numpy as np
import pytest
from pipeline.speech_emotion_pipeline import predict_emotions_from_file


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        predict_emotions_from_file("nonexistent_file.wav")