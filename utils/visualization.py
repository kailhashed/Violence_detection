#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def draw_prediction(frame, label, confidence, violent=False, threshold=0.5):
    """
    Draw prediction on frame
    
    Args:
        frame: Input frame
        label: Prediction label
        confidence: Prediction confidence
        violent: Whether the frame is classified as violent
        threshold: Confidence threshold for displaying alert
    
    Returns:
        Annotated frame
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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


def create_confidence_bar(confidence, violent=False, width=150, height=20):
    """
    Create a confidence bar visualization
    
    Args:
        confidence: Prediction confidence (0-1)
        violent: Whether the frame is classified as violent
        width: Width of the confidence bar
        height: Height of the confidence bar
    
    Returns:
        Confidence bar image
    """
    # Create a blank image
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
    
    return bar


def overlay_confidence_bar(frame, confidence, violent=False, pos=(20, 60)):
    """
    Overlay confidence bar on frame
    
    Args:
        frame: Input frame
        confidence: Prediction confidence
        violent: Whether the frame is classified as violent
        pos: Position of the confidence bar (x, y)
    
    Returns:
        Frame with overlaid confidence bar
    """
    # Create the confidence bar
    bar = create_confidence_bar(confidence, violent)
    
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


def plot_training_results(results_file):
    """
    Plot training results
    
    Args:
        results_file: Path to the training results CSV file
    """
    # Load results
    results = np.loadtxt(results_file, delimiter=',', skiprows=1)
    
    # Check if results is empty
    if results.size == 0:
        print("No training results found")
        return
    
    # Extract data
    epochs = results[:, 0]
    loss = results[:, 1]
    val_loss = results[:, 2]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if results.shape[1] > 3:
        # Extract accuracy data if available
        accuracy = results[:, 3]
        val_accuracy = results[:, 4]
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label='Training Accuracy')
        plt.plot(epochs, val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_results.png')
    plt.close()
    
    print(f"Training results plot saved to models/training_results.png") 