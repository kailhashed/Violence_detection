# Data Directory

This directory contains data files for the violence detection system.

## Structure

- `raw/`: Raw video dataset
- `processed/`: Extracted frames
- `annotations/`: YOLO format annotations

The dataset used is the [A Dataset for Automatic Violence Detection in Videos](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos) which contains labeled violent and non-violent video clips.

Use the scripts in the `src` directory to download and process the dataset. 