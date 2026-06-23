import numpy as np
import joblib
from collections import Counter
from config.settings import RF_MODEL_PATH, SCALER_PATH

_rf_model = None
_scaler = None


def _load_models():
    global _rf_model, _scaler
    if _rf_model is None or _scaler is None:
        print("📥 Loading emotion classification models...")
        _rf_model = joblib.load(RF_MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        print("✅ Emotion models loaded")
    return _rf_model, _scaler


def predict_emotion(feature_matrix: np.ndarray) -> str:
    """Return the predicted emotion label for a (1, N_features) feature matrix."""
    rf, scaler = _load_models()
    X_scaled = scaler.transform(feature_matrix.astype(np.float32))
    return rf.predict(X_scaled)[0]
