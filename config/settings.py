import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

#  FFmpeg 
FFMPEG_DIR = os.getenv(
    "FFMPEG_PATH",
    r"\ffmpeg-7.1.1-essentials_build\bin",
)

#  Pre-trained model paths 
RF_MODEL_PATH = os.getenv(
    "RF_MODEL_PATH",
    r"\best_random_forest_model.pkl",
)
SCALER_PATH = os.getenv(
    "SCALER_PATH",
    r"\standard_scaler.pkl",
)

#  Audio segmentation 
CHUNK_LENGTH_SEC = 15
OFFSET_SEC = 0.6
TARGET_SAMPLE_RATE = 16000

#  Feature extraction 
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13
N_MELS = 100
N_FFT = 1024
WIN_LENGTH = 512
MEL_HOP_LENGTH = 256
MEL_FMAX = 22050 / 2          

#  Augmentation (training only) 
SNR_LOW = 10
SNR_HIGH = 25
PITCH_FACTOR = 2.7
TIME_MASK_WIDTH = 20
TIME_MASK_COUNT = 2
FREQ_MASK_WIDTH = 10
FREQ_MASK_COUNT = 2