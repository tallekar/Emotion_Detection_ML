# Emotion Detector — OpenCV + DeepFace

Live webcam emotion detection with a styled HUD overlay.

## Setup

```bash
pip install -r requirements.txt
```

> First run downloads DeepFace models automatically (~500MB total).

## Run

```bash
python emotion_detector.py
```

## Controls

| Key | Action |
|-----|--------|
| Q   | Quit |
| S   | Save screenshot to `screenshots/` folder |
| P   | Pause / resume |

## What it detects

- 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
- Estimated age
- Gender
- Live FPS (camera + AI analysis)

## Config (top of script)

| Variable | Default | Description |
|----------|---------|-------------|
| `ANALYZE_FPS` | 5 | How many times/sec DeepFace runs |
| `SMOOTH_N` | 6 | Smoothing window for emotion bars |
| `BACKEND` | `opencv` | Face detector: opencv / retinaface / mtcnn |

## Tips

- Use `retinaface` backend for more accurate detection (slower)
- Use `mtcnn` for good balance of speed + accuracy
- `opencv` is fastest, works well in good lighting
- Run on GPU for real-time AI FPS (CUDA/Metal)
