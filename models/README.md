# Models Directory

This directory contains trained models for the violence detection system.

## Structure

- `violence_detection/`: Main project models
  - `weights/`: Trained model weights (*.pt files)

The models are trained using YOLOv8 from Ultralytics for real-time violence detection. The model is trained to classify frames as either "Violent" or "Non-Violent" with confidence scores.

Use the `train.py` script to train new models. 