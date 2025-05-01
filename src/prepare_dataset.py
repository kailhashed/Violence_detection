#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import shutil
from pathlib import Path
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

def create_dir_structure(base_dir):
    """
    Create YOLO dataset directory structure
    
    Args:
        base_dir: Base directory for the YOLO dataset
    """
    # Create main directories
    os.makedirs(os.path.join(base_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "val"), exist_ok=True)
    
    # Create data.yaml file
    yaml_content = """
# YOLO Violence Detection Dataset

# Train/val/test sets
train: images/train
val: images/val

# Classes
names:
  0: Violent
  1: Non-Violent
"""
    
    with open(os.path.join(base_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)
    
    print(f"Created YOLO dataset structure in {base_dir}")


def generate_annotation(image_path, class_id):
    """
    Generate YOLO annotation for an image
    
    Args:
        image_path: Path to the image file
        class_id: Class ID (0 for violent, 1 for non-violent)
        
    Returns:
        annotation_line: YOLO format annotation line
    """
    # For this task, we're treating the entire frame as the object
    # In YOLO format, this means the bounding box covers the entire image
    # Format: <class_id> <x_center> <y_center> <width> <height>
    
    # Get image dimensions to calculate bounding box
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # For the entire image, the center is (0.5, 0.5) and width/height are 1.0
    x_center, y_center = 0.5, 0.5
    width, height = 1.0, 1.0
    
    # Generate annotation line
    annotation_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
    
    return annotation_line


def prepare_dataset(processed_dir, output_dir, test_size=0.2, random_seed=42):
    """
    Prepare YOLO dataset from extracted frames
    
    Args:
        processed_dir: Directory containing processed frames
        output_dir: Output directory for YOLO dataset
        test_size: Proportion of the dataset to include in the validation split
        random_seed: Random seed for reproducibility
    """
    # Create directory structure
    create_dir_structure(output_dir)
    
    # Get paths to violent and non-violent frames
    violent_dir = os.path.join(processed_dir, "violent")
    nonviolent_dir = os.path.join(processed_dir, "non-violent")
    
    violent_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        violent_images.extend(glob(os.path.join(violent_dir, f"**/*{ext}"), recursive=True))
    
    nonviolent_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        nonviolent_images.extend(glob(os.path.join(nonviolent_dir, f"**/*{ext}"), recursive=True))
    
    print(f"Found {len(violent_images)} violent frames and {len(nonviolent_images)} non-violent frames")
    
    # Balance the dataset if needed
    if len(violent_images) > 0 and len(nonviolent_images) > 0:
        # Limit to minimum count if there's a large imbalance
        min_count = min(len(violent_images), len(nonviolent_images))
        max_count = max(len(violent_images), len(nonviolent_images))
        
        # If imbalance is significant (more than 2x), balance the dataset
        if max_count > min_count * 2:
            print(f"Balancing dataset: limiting to {min_count} images per class")
            if len(violent_images) > min_count:
                violent_images = random.sample(violent_images, min_count)
            if len(nonviolent_images) > min_count:
                nonviolent_images = random.sample(nonviolent_images, min_count)
    
    # Split into train and validation sets
    v_train, v_val = train_test_split(violent_images, test_size=test_size, random_state=random_seed)
    nv_train, nv_val = train_test_split(nonviolent_images, test_size=test_size, random_state=random_seed)
    
    print(f"Train: {len(v_train)} violent, {len(nv_train)} non-violent")
    print(f"Val: {len(v_val)} violent, {len(nv_val)} non-violent")
    
    # Copy images and create annotations
    print("Processing training set...")
    process_images(v_train, os.path.join(output_dir, "images", "train"), 
                  os.path.join(output_dir, "labels", "train"), class_id=0)
    process_images(nv_train, os.path.join(output_dir, "images", "train"), 
                  os.path.join(output_dir, "labels", "train"), class_id=1)
    
    print("Processing validation set...")
    process_images(v_val, os.path.join(output_dir, "images", "val"), 
                  os.path.join(output_dir, "labels", "val"), class_id=0)
    process_images(nv_val, os.path.join(output_dir, "images", "val"), 
                  os.path.join(output_dir, "labels", "val"), class_id=1)
    
    print(f"Dataset preparation completed. YOLO dataset saved to {output_dir}")


def process_images(image_paths, image_out_dir, label_out_dir, class_id):
    """
    Process a list of images: copy to output directory and create annotations
    
    Args:
        image_paths: List of image paths
        image_out_dir: Output directory for images
        label_out_dir: Output directory for labels
        class_id: Class ID (0 for violent, 1 for non-violent)
    """
    for img_path in tqdm(image_paths, desc=f"Processing class {class_id}"):
        # Get image filename
        img_filename = os.path.basename(img_path)
        img_name = os.path.splitext(img_filename)[0]
        
        # Copy image to output directory
        dst_img_path = os.path.join(image_out_dir, img_filename)
        try:
            shutil.copy2(img_path, dst_img_path)
            
            # Generate annotation
            annotation = generate_annotation(img_path, class_id)
            
            if annotation:
                # Save annotation file
                label_filename = f"{img_name}.txt"
                with open(os.path.join(label_out_dir, label_filename), 'w') as f:
                    f.write(annotation)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset for violence detection")
    parser.add_argument("--input_dir", default="data/processed", 
                      help="Input directory containing processed frames")
    parser.add_argument("--output_dir", default="data/annotations",
                      help="Output directory for YOLO dataset")
    parser.add_argument("--test_size", type=float, default=0.2,
                      help="Proportion of the dataset to include in the validation split")
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    prepare_dataset(args.input_dir, args.output_dir, 
                   test_size=args.test_size, random_seed=args.random_seed)
    
    print("YOLO dataset preparation completed.")
    print("Next step: Run train.py to train the YOLO model.")


if __name__ == "__main__":
    main() 