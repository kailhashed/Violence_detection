#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import draw_prediction, overlay_confidence_bar


def run_inference(model_path, source="0", confidence_threshold=0.5, fps_target=30, display_scale=1.0):
    """
    Run real-time violence detection inference
    
    Args:
        model_path: Path to the trained YOLOv8 model
        source: 0 for webcam, or path to video file
        confidence_threshold: Confidence threshold for detection
        fps_target: Target FPS (frames per second)
        display_scale: Scale factor for display
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
        print(f"Model task type: {model.task}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set up video capture
    try:
        if source.isdigit():
            # If source is numeric, it's a webcam index
            cap = cv2.VideoCapture(int(source))
            source_name = "Webcam"
        else:
            # Otherwise, it's a video file
            cap = cv2.VideoCapture(source)
            source_name = os.path.basename(source)
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set display dimensions
    display_width = int(width * display_scale)
    display_height = int(height * display_scale)
    
    print(f"Video source: {source_name}, Resolution: {width}x{height}, FPS: {fps:.2f}")
    print(f"Display resolution: {display_width}x{display_height}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Class names
    class_names = {0: "Violent", 1: "Non-Violent"}
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    # Calculate target frame interval
    frame_interval = 1.0 / fps_target if fps_target > 0 else 0
    
    # Main loop
    while True:
        loop_start = time.time()
        
        # Read a frame
        ret, frame = cap.read()
        
        # Break the loop if video ended
        if not ret:
            if source.isdigit():
                # For webcam, try to reconnect
                print("Lost connection to webcam. Attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(int(source))
                if not cap.isOpened():
                    break
                continue
            else:
                # For video file, exit loop
                print("End of video file reached.")
                break
        
        # Increment frame counter
        frame_count += 1
        
        # Perform inference
        results = model(frame, verbose=False)[0]
        
        # Process results for classification task
        if model.task == 'classify':
            # Get the predictions
            probs = results.probs
            # Get the class with highest confidence
            if probs is not None:
                class_id = int(probs.top1)
                confidence = float(probs.top1conf)
                
                # Map class ID to label
                label = class_names.get(class_id, f"Class {class_id}")
                violent = class_id == 0
                
                # Draw prediction on frame
                frame = draw_prediction(frame, label, confidence, violent, threshold=confidence_threshold)
                
                # Overlay confidence bar
                frame = overlay_confidence_bar(frame, confidence, violent)
            else:
                # No detection
                label = "No detection"
                confidence = 0.0
                
                # Draw default label
                frame = draw_prediction(frame, label, confidence, violent=False)
        else:
            # For detection task (standard YOLO behavior)
            if len(results.boxes.cls) > 0:
                # Get the class with highest confidence
                confidence = float(results.boxes.conf[0])
                class_id = int(results.boxes.cls[0])
                
                # Map class ID to label
                label = class_names.get(class_id, f"Class {class_id}")
                violent = class_id == 0
                
                # Draw prediction on frame
                frame = draw_prediction(frame, label, confidence, violent, threshold=confidence_threshold)
                
                # Overlay confidence bar
                frame = overlay_confidence_bar(frame, confidence, violent)
            else:
                # No detection
                label = "No detection"
                confidence = 0.0
                
                # Draw default label
                frame = draw_prediction(frame, label, confidence, violent=False)
        
        # Calculate and display FPS
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= 1.0:
            fps_display = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (display_width - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Resize frame for display
        if display_scale != 1.0:
            frame = cv2.resize(frame, (display_width, display_height))
        
        # Display the frame
        cv2.imshow("Violence Detection", frame)
        
        # Wait for key press (1ms) and check for exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or Esc key
            break
        
        # Control frame rate
        loop_time = time.time() - loop_start
        if loop_time < frame_interval:
            time.sleep(frame_interval - loop_time)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Inference completed.")


def main():
    parser = argparse.ArgumentParser(description="Real-time violence detection")
    parser.add_argument("--model", default="models/violence_detection/weights/best.pt", 
                      help="Path to the trained YOLOv8 model")
    parser.add_argument("--source", default="0",
                      help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                      help="Confidence threshold for detection")
    parser.add_argument("--fps_target", type=int, default=30,
                      help="Target FPS (frames per second)")
    parser.add_argument("--display_scale", type=float, default=1.0,
                      help="Scale factor for display")
    
    args = parser.parse_args()
    
    # Use fallback model if specified model doesn't exist
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}")
        # Look for any .pt file in the models directory
        model_files = list(Path("models").glob("**/*.pt"))
        
        if model_files:
            args.model = str(model_files[0])
            print(f"Using alternative model: {args.model}")
        else:
            print("No model file found. Please train the model first.")
            return
    
    # Run inference
    run_inference(
        args.model,
        source=args.source,
        confidence_threshold=args.conf_threshold,
        fps_target=args.fps_target,
        display_scale=args.display_scale
    )


if __name__ == "__main__":
    main() 