import numpy as np
import pytest
from services.feature_extractor import (
    zcr, rmse, mfcc, mel_spectrogram,
    extract_features, build_augmented_feature_matrix,
    noise, pitch, time_masking, frequency_masking,
)

SR = 16000
DURATION = 6  # seconds
SIGNAL = np.random.randn(SR * DURATION).astype(np.float32) * 0.1


def test_zcr_shape():
    out = zcr(SIGNAL)
    assert out.ndim == 1 and len(out) > 0


def test_rmse_shape():
    out = rmse(SIGNAL)
    assert out.ndim == 1 and len(out) > 0


def test_mfcc_shape():
    out = mfcc(SIGNAL, SR)
    assert out.ndim == 1 and len(out) > 0


def test_mel_spectrogram_shape():
    out = mel_spectrogram(SIGNAL, SR)
    assert out.ndim == 1 and len(out) > 0


def test_extract_features_1d():
    out = extract_features(SIGNAL, SR)
    assert out.ndim == 1 and len(out) > 0


def test_augmented_matrix_rows():
    matrix = build_augmented_feature_matrix(SIGNAL, SR)
    # 6 augmentation variants
    assert matrix.shape[0] == 6
    assert matrix.shape[1] > 0


def test_noise_augmentation():
    noisy = noise(SIGNAL)
    assert noisy.shape == SIGNAL.shape
    assert not np.allclose(noisy, SIGNAL)


def test_pitch_augmentation():
    pitched = pitch(SIGNAL, SR)
    assert len(pitched) > 0


def test_time_masking_length():
    masked = time_masking(SIGNAL, SR)
    assert len(masked) == len(SIGNAL)


def test_frequency_masking_length():
    masked = frequency_masking(SIGNAL, SR)
    assert len(masked) == len(SIGNAL)
