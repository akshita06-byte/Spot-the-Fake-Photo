# Real vs Screen Photo Classifier

A simple Python project that trains a model to distinguish between real photos and photos of screens.

## Files
- `train.py`: train the model on `real/` and `screen/` folders
- `predict.py`: predict whether a single image is a real photo or a screen photo
- `evaluate.py`: evaluate the model on labelled `real/` and `screen/` folders
- `features.py`: extract patch-based image features used by the model
- `screen_detector.pkl`: saved trained model file
- `MODEL_NOTE.md`: project summary and metrics

## Setup
Install the Python dependencies:
```powershell
python -m pip install -r requirements.txt
```

## Train
```powershell
python train.py real screen
```

## Predict
```powershell
python predict.py path\to\image.jpg
```

## Evaluate
```powershell
python evaluate.py real screen
```
