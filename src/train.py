#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path

def create_yolo_cls_dataset(annotations_dir, output_dir):
    """
    Create a YOLO classification dataset structure from our existing annotations
    
    Args:
        annotations_dir: Path to the annotations directory
        output_dir: Path to the output directory for training
    
    Returns:
        Path to the created dataset directory
    """
    # Create output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train and val directories for classes
    for split in ['train', 'val']:
        images_dir = os.path.join(annotations_dir, 'images', split)
        
        # Check if images directory exists
        if not os.path.exists(images_dir):
            print(f"Error: {images_dir} not found")
            return None
        
        # Get list of classes (subdirectories)
        classes = ["0-Violent", "1-NonViolent"]
        
        # Create class directories in output directory
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    
    # Copy images to class directories
    for split in ['train', 'val']:
        images_dir = os.path.join(annotations_dir, 'images', split)
        labels_dir = os.path.join(annotations_dir, 'labels', split)
        
        # Get all image files
        image_files = list(Path(images_dir).glob("**/*.jpg"))
        print(f"Found {len(image_files)} images in {split} split")
        
        # Process each image
        for img_path in image_files:
            # Get image basename
            img_name = os.path.basename(img_path)
            img_stem = os.path.splitext(img_name)[0]
            
            # Find corresponding label file
            label_path = os.path.join(labels_dir, f"{img_stem}.txt")
            
            if os.path.exists(label_path):
                # Read label to determine class
                with open(label_path, 'r') as f:
                    label_content = f.read().strip()
                    class_id = int(label_content.split()[0])
                    
                # Set class directory name based on class ID
                if class_id == 0:
                    class_dir = "0-Violent"
                else:
                    class_dir = "1-NonViolent"
                
                # Copy image to appropriate class directory
                dest_path = os.path.join(output_dir, split, class_dir, img_name)
                shutil.copy2(img_path, dest_path)
    
    print(f"YOLO classification dataset created at {output_dir}")
    return output_dir


def train_yolo_model(data_yaml_path, model_type="yolov8n-cls.pt", epochs=50, 
                     batch_size=16, image_size=224, device=None):
    """
    Train YOLOv8 model for violence detection
    
    Args:
        data_yaml_path: Path to the data.yaml file
        model_type: YOLOv8 model type to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image size for the model
        device: Device to use for training (None for auto-selection)
    
    Returns:
        Path to the trained model
    """
    # Create classification dataset
    annotations_dir = os.path.dirname(data_yaml_path)
    cls_dataset_dir = os.path.join(os.path.dirname(annotations_dir), "annotations_cls")
    
    dataset_path = create_yolo_cls_dataset(annotations_dir, cls_dataset_dir)
    if dataset_path is None:
        print("Error creating classification dataset")
        return None
    
    print(f"Training YOLOv8 model on {dataset_path}")
    print(f"Model: {model_type}, Epochs: {epochs}, Batch size: {batch_size}, Image size: {image_size}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if device is None:
        device = 0 if cuda_available else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create a new model or load a pre-trained one
    if model_type.endswith('.pt') and os.path.exists(model_type):
        # Load custom trained model
        model = YOLO(model_type)
        print(f"Loaded custom model from {model_type}")
    else:
        # Load pre-trained YOLO model - use classification model
        if not model_type.endswith('-cls.pt'):
            # If not a classification model, convert to classification model
            model_type = model_type.replace('.pt', '-cls.pt')
            print(f"Converting to classification model: {model_type}")
        model = YOLO(model_type)
        print(f"Loaded pre-trained {model_type} model")
    
    # Train the model
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project="models",
        name="violence_detection",
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    # Get the path to the best model
    best_model_path = results.best
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for violence detection")
    parser.add_argument("--data_yaml", default="data/annotations/data.yaml", 
                      help="Path to the data.yaml file")
    parser.add_argument("--model", default="yolov8n-cls.pt",
                      help="YOLOv8 model to use")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=224,
                      help="Input image size for the model")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use for training (None for auto-selection)")
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train the model
    best_model_path = train_yolo_model(
        args.data_yaml,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device
    )
    
    if best_model_path:
        print(f"Training completed. Best model saved to {best_model_path}")
        print("Next step: Run realtime_inference.py for real-time violence detection.")
    else:
        print("Training failed.")


if __name__ == "__main__":
    main() 