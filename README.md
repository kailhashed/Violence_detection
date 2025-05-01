# Real-Time Violence Detection System

A YOLOv8-based system for detecting violence in videos and real-time video streams.

## Project Structure
```
violence_detection/
├── data/
│   ├── raw/                 # Raw video dataset
│   ├── processed/           # Extracted frames
│   └── annotations/         # YOLO format annotations
├── src/
│   ├── extract_frames.py    # Script to extract frames from videos
│   ├── prepare_dataset.py   # Script to prepare YOLO dataset
│   ├── train.py             # Script to train YOLOv8 model
│   └── realtime_inference.py # Real-time detection script
├── utils/
│   └── visualization.py     # Visualization utilities
├── models/                  # Trained models will be saved here
├── README.md
└── requirements.txt         # Project dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/violence_detection.git
cd violence_detection
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
# source venv/bin/activate
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python src/download_dataset.py
```

4. Extract frames from videos:
```bash
python src/extract_frames.py
```

5. Prepare YOLO dataset:
```bash
python src/prepare_dataset.py
```

6. Train the model:
```bash
python src/train.py
```

7. Run real-time inference:
```bash
python src/realtime_inference.py --source 0  # For webcam
# OR
python src/realtime_inference.py --source path/to/video.mp4  # For video file
```

## Dataset
The project uses the [A Dataset for Automatic Violence Detection in Videos](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos) which contains labeled violent and non-violent video clips.

## Model
The system uses YOLOv8 from Ultralytics for real-time violence detection. The model is trained to classify frames as either "Violent" or "Non-Violent" with confidence scores. 