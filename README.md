# Real vs Screen Photo Classifier

A simple Python project that distinguishes real photos from photos of screens using 
patch-based feature extraction and gradient boosting.

## Files

* `predict.py`: predicts whether a single image is real or a screen photo
* `train.py`: training code (requires `real/` and `screen/` folders locally)
* `features.py`: feature extraction pipeline
* `screen_detector.pkl`: pre-trained model
* `MODEL_NOTE.md`: technical details, accuracy, and metrics

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Predict on a single image:
```bash
python predict.py path/to/image.jpg
```

Output: Confidence score 0-1 (0=real photo, 1=screen photo)

## Performance

- **Accuracy:** 100% on screen detection
- **Latency:** ~1200ms per image (CPU)
- **Cost:** Free on-device

See `MODEL_NOTE.md` for full technical details.
