#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import cv2
import numpy as np
import time
from pathlib import Path
import torch

# Import YOLO only if available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics package not available. Using fallback inference method.")

# Class names
CLASS_NAMES = {0: "Violent", 1: "Non-Violent"}

def draw_prediction(frame, label, confidence, violent=False, threshold=0.5):
    """
    Draw prediction on frame
    """
    height, width = frame.shape[:2]
    
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    
    # Format the label and confidence
    text = f"{label}: {confidence:.2f}"
    
    # Calculate text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Draw background rectangle for text
    if violent and confidence >= threshold:
        # Red background for violent
        bg_color = (0, 0, 255)  # Red (BGR)
        
        # Draw alert on top of the frame
        alert_text = "VIOLENCE DETECTED!"
        cv2.putText(annotated_frame, alert_text, (width // 2 - 150, 70), 
                   font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, timestamp, (10, height - 10),
                   font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Add alert border
        border_size = 10
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), border_size)
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
    else:
        # Green background for non-violent
        bg_color = (0, 128, 0)  # Green (BGR)
    
    # Draw label background
    label_background_pos = (10, 10, text_width + 20, text_height + 20)
    cv2.rectangle(annotated_frame, 
                 (label_background_pos[0], label_background_pos[1]), 
                 (label_background_pos[0] + label_background_pos[2], label_background_pos[1] + label_background_pos[3]), 
                 bg_color, -1)
    
    # Draw label text
    cv2.putText(annotated_frame, text, (20, 30), 
               font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return annotated_frame

def overlay_confidence_bar(frame, confidence, violent=False, pos=(20, 60)):
    """
    Overlay confidence bar on frame
    """
    # Create blank image for the bar
    height, width = 20, 150
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate filled width
    filled_width = int(confidence * width)
    
    # Choose color based on class
    if violent:
        color = (0, 0, 255)  # Red (BGR)
    else:
        color = (0, 255, 0)  # Green (BGR)
    
    # Draw the filled portion
    cv2.rectangle(bar, (0, 0), (filled_width, height), color, -1)
    
    # Draw the empty portion
    cv2.rectangle(bar, (filled_width, 0), (width, height), (100, 100, 100), -1)
    
    # Draw border
    cv2.rectangle(bar, (0, 0), (width, height), (200, 200, 200), 1)
    
    # Get dimensions
    bar_height, bar_width = bar.shape[:2]
    
    # Create region of interest
    roi = frame[pos[1]:pos[1]+bar_height, pos[0]:pos[0]+bar_width]
    
    # Overlay confidence bar on ROI
    result = cv2.addWeighted(roi, 0.7, bar, 0.9, 0)
    
    # Replace ROI with result
    frame_with_bar = frame.copy()
    frame_with_bar[pos[1]:pos[1]+bar_height, pos[0]:pos[0]+bar_width] = result
    
    return frame_with_bar

def run_inference(model_path, source="0", confidence_threshold=0.5, fps_target=30, display_scale=1.0):
    """
    Run real-time violence detection inference
    """
    # Load the trained model if ultralytics is available
    if ULTRALYTICS_AVAILABLE:
        print(f"Loading model from {model_path}...")
        try:
            model = YOLO(model_path)
            print("Model loaded successfully")
            print(f"Model task type: {model.task}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Using fallback inference mode (random predictions for demo)")
        model = None
    
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
        if ULTRALYTICS_AVAILABLE and model is not None:
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
                    label = CLASS_NAMES.get(class_id, f"Class {class_id}")
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
                    label = CLASS_NAMES.get(class_id, f"Class {class_id}")
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
            # Fallback inference (random prediction)
            if frame_count % 30 < 15:  # Change prediction every 15 frames
                class_id = 0  # Violent
                confidence = 0.7 + 0.2 * np.sin(time.time())  # Varying confidence
            else:
                class_id = 1  # Non-violent
                confidence = 0.6 + 0.3 * np.sin(time.time())  # Varying confidence
            
            label = CLASS_NAMES.get(class_id, f"Class {class_id}")
            violent = class_id == 0
            
            # Draw prediction on frame
            frame = draw_prediction(frame, label, confidence, violent, threshold=confidence_threshold)
            
            # Overlay confidence bar
            frame = overlay_confidence_bar(frame, confidence, violent)
        
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
            print("No model file found. Using fallback demo mode without a model.")
            if ULTRALYTICS_AVAILABLE:
                args.model = "yolov8n-cls.pt"  # Use a default model if ultralytics is available
            else:
                args.model = None
    
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