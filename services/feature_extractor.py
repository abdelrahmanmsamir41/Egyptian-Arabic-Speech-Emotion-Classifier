import numpy as np
import librosa
from config.settings import (
    FRAME_LENGTH, HOP_LENGTH, N_MFCC, N_MELS, N_FFT, WIN_LENGTH, MEL_HOP_LENGTH, MEL_FMAX,
    SNR_LOW, SNR_HIGH, PITCH_FACTOR,
    TIME_MASK_WIDTH, TIME_MASK_COUNT, FREQ_MASK_WIDTH, FREQ_MASK_COUNT,
)

# ── Augmentations ────────────────────────────────────────────────────────────

def noise(data: np.ndarray, snr_low: int = SNR_LOW, snr_high: int = SNR_HIGH) -> np.ndarray:
    """Add random Gaussian noise at a random SNR level."""
    noise_signal = np.random.normal(size=data.shape)
    norm_constant = 2.0 ** 15
    signal_power = np.mean((data / norm_constant) ** 2)
    noise_power = np.mean((noise_signal / norm_constant) ** 2)
    target_snr = np.random.randint(snr_low, snr_high)
    scale = np.sqrt(signal_power / (noise_power * 10 ** (target_snr / 10)))
    return data + noise_signal * scale


def stretch(data: np.ndarray, rate: float = 0.8) -> np.ndarray:
    return librosa.effects.time_stretch(y=data, rate=rate)


def shift(data: np.ndarray) -> np.ndarray:
    shift_range = int(np.random.uniform(-5, 5) * 1000)
    return np.roll(data, shift_range)


def pitch(data: np.ndarray, sr: int, pitch_factor: float = PITCH_FACTOR) -> np.ndarray:
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)


def _apply_stft_mask(data: np.ndarray, axis: int, mask_width: int, mask_count: int) -> np.ndarray:
    original_length = len(data)
    stft = librosa.stft(data)
    magnitude, phase = librosa.magphase(stft)
    dim = magnitude.shape[axis]
    for _ in range(mask_count):
        start = np.random.randint(0, max(1, dim - mask_width))
        if axis == 1:
            magnitude[:, start : start + mask_width] = 0
        else:
            magnitude[start : start + mask_width, :] = 0
    masked_audio = librosa.istft(magnitude * phase)
    if len(masked_audio) > original_length:
        return masked_audio[:original_length]
    return np.pad(masked_audio, (0, original_length - len(masked_audio)))


def time_masking(
    data: np.ndarray,
    sr: int,
    mask_width: int = TIME_MASK_WIDTH,
    mask_count: int = TIME_MASK_COUNT,
) -> np.ndarray:
    return _apply_stft_mask(data, axis=1, mask_width=mask_width, mask_count=mask_count)


def frequency_masking(
    data: np.ndarray,
    sr: int,
    mask_width: int = FREQ_MASK_WIDTH,
    mask_count: int = FREQ_MASK_COUNT,
) -> np.ndarray:
    return _apply_stft_mask(data, axis=0, mask_width=mask_width, mask_count=mask_count)


# ── Feature extraction ───────────────────────────────────────────────────────

def zcr(data: np.ndarray) -> np.ndarray:
    return np.squeeze(
        librosa.feature.zero_crossing_rate(y=data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    )


def rmse(data: np.ndarray) -> np.ndarray:
    return np.squeeze(
        librosa.feature.rms(y=data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    )


def mfcc(data: np.ndarray, sr: int) -> np.ndarray:
    feat = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=N_MFCC)
    return np.ravel(feat.T)


def mel_spectrogram(data: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=data,
        sr=sr,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        window="hamming",
        hop_length=MEL_HOP_LENGTH,
        n_mels=N_MELS,
        fmax=MEL_FMAX,
    )
    return np.ravel(librosa.amplitude_to_db(mel, ref=np.max).T)


def extract_features(data: np.ndarray, sr: int) -> np.ndarray:
    """Concatenate all features for a single audio segment."""
    return np.hstack([zcr(data), rmse(data), mfcc(data, sr), mel_spectrogram(data, sr)])


# def build_augmented_feature_matrix(segment: np.ndarray, sr: int) -> np.ndarray:

#     variants = [
#         segment,
#         noise(segment),
#         pitch(segment, sr),
#         noise(pitch(segment, sr)),
#         time_masking(segment, sr),
#         frequency_masking(segment, sr),
#     ]
#     return np.vstack([extract_features(v, sr) for v in variants])
