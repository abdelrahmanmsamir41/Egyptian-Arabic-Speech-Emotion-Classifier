# Speech Emotion Recognition — Pipeline Flow

```mermaid
flowchart TD

    A(["`**INPUT**
    file_path  _(video or audio)_`"])
    A --> B[predict_emotions_from_file]

    %% ── Step 1 · Media conversion ────────────────────────────────────────────
    B --> C{File type?}
    C -- video\n.mp4 .avi .mov … --> D["convert_video_to_audio()\nMoviePy → pcm_s16le WAV"]
    C -- audio non-WAV\n.mp3 .m4a … --> E["convert_to_wav()\npydub → WAV"]
    C -- already WAV --> F[Pass through]
    D & E & F --> G[wav_path]

    %% ── Step 2 · Resample ────────────────────────────────────────────────────
    G --> H["librosa.load  +  soundfile.write\nResample → 16 000 Hz WAV"]

    %% ── Step 3 · Chunking ────────────────────────────────────────────────────
    H --> I["Split into 15-sec chunks\nwith 0.6 s silence offset"]
    I --> J["audio_chunks\n[chunk₁, chunk₂, …]"]

    %% ── Step 4 · Per-chunk loop ──────────────────────────────────────────────
    J --> K{For each chunk}
    K --> L["Normalise  ÷ max(|signal|)\nPad / trim → exactly 6 sec"]

    %% ── Feature extraction ────────────────────────────────────────────────────
    L --> M["extract_features(signal, sr)"]
    M --> M1[ZCR]
    M --> M2[RMSE]
    M --> M3["MFCC  ×13"]
    M --> M4["Mel Spectrogram\n100 mels"]
    M1 & M2 & M3 & M4 --> M5["np.hstack → flat vector\nreshape(1, -1)"]

    %% ── Prediction ───────────────────────────────────────────────────────────
    M5 --> N["predict_emotion(features)"]
    N --> N1[StandardScaler.transform]
    N1 --> N2["RandomForest.predict"]
    N2 --> O["emotion label\ne.g. sad · happy · angry"]

    O --> K

    %% ── Aggregate ────────────────────────────────────────────────────────────
    O --> P["Collect all chunk predictions"]
    P --> Q["Counter.most_common(1)\nMajority vote across chunks"]

    %% ── Output ───────────────────────────────────────────────────────────────
    Q --> R(["`**OUTPUT**
    {
      chunks:  [{ chunk: 1, emotion: 'sad' }, …],
      overall: 'sad'
    }`"])

    %% ── Styles ───────────────────────────────────────────────────────────────
    style A  fill:#4CAF50,color:#fff,stroke:none
    style R  fill:#4CAF50,color:#fff,stroke:none
    style C  fill:#FF9800,color:#fff,stroke:none
    style K  fill:#FF9800,color:#fff,stroke:none
    style M  fill:#2196F3,color:#fff,stroke:none
    style N  fill:#2196F3,color:#fff,stroke:none
```

---

## Module responsible for each step

| Step | What happens | Module |
|---|---|---|
| ① | Detect file type, convert video → WAV or re-encode audio → WAV | `services/media_processor.py` |
| ② | Resample to 16 000 Hz | `pipeline/speech_emotion_pipeline.py` |
| ③ | Split raw audio into 15-sec chunks (0.6 s offset) | `pipeline/speech_emotion_pipeline.py` |
| ④ | Normalise + pad/trim each chunk to 6 sec | `pipeline/speech_emotion_pipeline.py` |
| ⑤ | Extract ZCR · RMSE · MFCC · Mel Spectrogram → flat feature vector | `services/feature_extractor.py` |
| ⑥ | Scale features → Random Forest predict → emotion label | `services/emotion_predictor.py` |
| ⑦ | Majority vote across all chunk predictions → overall emotion | `pipeline/speech_emotion_pipeline.py` |