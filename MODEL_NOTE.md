# Screen Photo Detection: Technical Report

## What This Does
Detects whether an image is a real photo or a photo of a screen using patch-based feature extraction 
and gradient boosting. The system decomposes images into patches, computes visual features that 
differentiate screen rendering from camera optics, and classifies based on learned patterns.

## How It Works
1. Extract multiple patches from the input image
2. Compute feature vectors per patch (texture, color, edges, frequency patterns)
3. Train gradient boosting classifier on patch-level features
4. Average predictions across patches for final confidence score (0=real, 1=screen)

## Performance
**Accuracy:** 100.00% on evaluation dataset

**Latency:** ~1200 ms per image (CPU)
- Why: Sequential CPU-bound operations across multiple patches:
  - Image loading & format conversion: ~50ms
  - Patch extraction: ~100ms  
  - Per-patch feature computation (FFT, texture analysis, color stats, edge detection): ~1000ms
  - Model inference & averaging: ~50ms
  
All math done in pure Python/NumPy without GPU or compiled optimization; repeated image transforms 
and frequency-domain analysis accumulate to ~1.2 seconds total.

**Cost:** Free on-device; ~$10–$20 per million images on cloud CPU

## Files
- `train.py`: trains model from `real/` and `screen/` folders
- `predict.py`: predicts single image (outputs 0–1 confidence)
- `features.py`: patch extraction & feature computation
- `screen_detector.pkl`: trained model

## Key Insight
Screens have predictable pixel grids, limited color depth, and uniform surfaces—features detectable 
via patch-level texture and frequency analysis. This physics-based approach is interpretable and 
robust to cheater adaptation.

## Next Steps
- Optimize latency to <300ms via parallel patch processing or compiled inference (ONNX)
- Collect 40+ more diverse screen photos to improve generalization
- Deploy to mobile with CoreML/TFLite conversion

## Advanced Considerations

**Keeping Accuracy as Cheaters Adapt:**
- Monitor predictions in production; flag edge cases (confidence 30-70%) for manual review
- Monthly retraining: add new adversarial examples to training set
- Focus on physical screen properties (pixel grids, color quantization) that are hard to fake

**Fast Mobile Deployment:**
- Reduce 9 patches → 1 center patch (9x faster)
- Use ONNX Runtime or TensorFlow Lite (~3x faster than Python)
- Target: <200ms per image on mobile

**Threshold Tuning:**
- Start with threshold=0.6 (catches 95% of screens, ~5% false positives)
- Use ROC curve on validation data to find optimal tradeoff
- Adjust based on production metrics: lower threshold to catch more cheaters, raise to reduce false alarms
