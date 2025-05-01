#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

def extract_frames(video_path, output_dir, sampling_rate=5, max_frames=100):
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        sampling_rate: Extract 1 frame every N frames
        max_frames: Maximum number of frames to extract
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing {video_name}: {total_frames} frames, {fps:.2f} fps")
    
    # Calculate actual sampling rate based on max_frames
    if total_frames > max_frames * sampling_rate:
        sampling_rate = total_frames // max_frames
    
    # Initialize counters
    frame_count = 0
    saved_count = 0
    
    # Loop through video frames
    with tqdm(total=total_frames, desc=f"Extracting frames from {video_name}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save frame at the specified sampling rate
            if frame_count % sampling_rate == 0:
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{saved_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
            
            # Exit if we've extracted enough frames
            if saved_count >= max_frames:
                break
    
    # Release resources
    cap.release()
    
    print(f"Extracted {saved_count} frames from {video_name}")
    return saved_count

def process_video_folder(input_folder, output_folder, sampling_rate=5, max_frames=100):
    """
    Process all videos in a folder
    
    Args:
        input_folder: Folder containing video files
        output_folder: Folder to save extracted frames
        sampling_rate: Extract 1 frame every N frames
        max_frames: Maximum number of frames to extract per video
    """
    # Get list of video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(Path(input_folder).glob(f'**/*{ext}')))
    
    # Check if any videos were found
    if not video_files:
        print(f"No video files found in {input_folder}")
        return
    
    print(f"Found {len(video_files)} videos in {input_folder}")
    
    # Process each video
    total_frames = 0
    for video_file in video_files:
        # Create subfolder structure in output directory
        rel_path = os.path.relpath(os.path.dirname(video_file), input_folder)
        output_subdir = os.path.join(output_folder, rel_path)
        
        # Extract frames
        num_frames = extract_frames(
            str(video_file), 
            output_subdir, 
            sampling_rate=sampling_rate, 
            max_frames=max_frames
        )
        
        total_frames += num_frames
    
    print(f"Total frames extracted: {total_frames}")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos for violence detection")
    parser.add_argument("--input_dir", default="A-Dataset-for-Automatic-Violence-Detection-in-Videos/violence-detection-dataset", 
                      help="Input directory containing violent and non-violent videos")
    parser.add_argument("--output_dir", default="data/processed",
                      help="Output directory for extracted frames")
    parser.add_argument("--sampling_rate", type=int, default=5,
                      help="Extract 1 frame every N frames")
    parser.add_argument("--max_frames", type=int, default=100,
                      help="Maximum number of frames to extract per video")
    
    args = parser.parse_args()
    
    # Process violent videos
    violent_input = os.path.join(args.input_dir, "violent")
    violent_output = os.path.join(args.output_dir, "violent")
    
    if os.path.exists(violent_input):
        print(f"Processing violent videos from {violent_input}")
        process_video_folder(
            violent_input, 
            violent_output, 
            sampling_rate=args.sampling_rate, 
            max_frames=args.max_frames
        )
    else:
        print(f"Violent video folder not found: {violent_input}")
    
    # Process non-violent videos
    nonviolent_input = os.path.join(args.input_dir, "non-violent")
    nonviolent_output = os.path.join(args.output_dir, "non-violent")
    
    if os.path.exists(nonviolent_input):
        print(f"Processing non-violent videos from {nonviolent_input}")
        process_video_folder(
            nonviolent_input, 
            nonviolent_output, 
            sampling_rate=args.sampling_rate, 
            max_frames=args.max_frames
        )
    else:
        print(f"Non-violent video folder not found: {nonviolent_input}")
    
    print("Frame extraction completed.")
    print("Next step: Run prepare_dataset.py to prepare the YOLO dataset.")

if __name__ == "__main__":
    main() 