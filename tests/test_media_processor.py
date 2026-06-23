import os
import tempfile
import numpy as np
import soundfile as sf
import pytest
from services.media_processor import convert_to_wav


def _make_wav(path: str, sr: int = 16000, duration: float = 1.0):
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(path, samples, sr)


def test_convert_to_wav_identity():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "test.wav")
        _make_wav(src)
        out = convert_to_wav(src)
        assert os.path.exists(out)
        assert out.endswith(".wav")


def test_unsupported_raises():
    from utils.file_utils import is_video_file, is_audio_file
    assert not is_video_file("file.txt")
    assert not is_audio_file("file.txt")
