#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import requests
import zipfile
from tqdm import tqdm
import shutil
import subprocess
import time
import stat
import errno

def download_file(url, destination):
    """
    Download a file from a URL to a destination with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    print(f"Downloading from {url} to {destination}")
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)
            
    return destination

def handle_remove_readonly(func, path, exc):
    """
    Error handler for shutil.rmtree to handle read-only files
    """
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        # Change file to be writeable
        os.chmod(path, stat.S_IWUSR)
        # Retry
        func(path)
    else:
        raise

def safe_rmtree(path):
    """
    Safely remove a directory tree, handling Windows permission errors
    """
    if not os.path.exists(path):
        return
    
    print(f"Cleaning up {path}...")
    try:
        # First attempt: standard removal
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        print(f"Standard cleanup failed: {e}")
        try:
            # Second attempt: use custom error handler
            shutil.rmtree(path, onerror=handle_remove_readonly)
        except Exception as e:
            print(f"Enhanced cleanup failed: {e}")
            # Last resort: use subprocess to call system commands
            try:
                if os.name == 'nt':  # Windows
                    subprocess.run(['rd', '/s', '/q', path], shell=True, check=False)
                else:  # Unix/Linux
                    subprocess.run(['rm', '-rf', path], check=False)
            except Exception as e:
                print(f"System command cleanup failed: {e}")
                print(f"Warning: Could not remove {path}. Please delete it manually.")

def download_violence_dataset():
    """
    Download the violence detection dataset from GitHub
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), "data")
    raw_dir = os.path.join(data_dir, "raw")
    
    os.makedirs(raw_dir, exist_ok=True)
    
    # GitHub repository URL
    repo_url = "https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git"
    temp_repo_dir = "temp_dataset_repo"
    
    # Clone the repository
    print(f"Cloning the dataset repository from {repo_url}")
    try:
        # Ensure the temp directory doesn't exist
        if os.path.exists(temp_repo_dir):
            safe_rmtree(temp_repo_dir)
            
        # Clone with depth 1 to speed up
        subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_repo_dir], check=True)
        
        # Move the dataset files to our data directory
        dataset_path = os.path.join(os.getcwd(), temp_repo_dir, "dataset")
        if os.path.exists(dataset_path):
            print(f"Moving dataset files to {raw_dir}")
            
            # Move violent and non-violent folders
            violent_src = os.path.join(dataset_path, "violent")
            nonviolent_src = os.path.join(dataset_path, "non-violent")
            
            if os.path.exists(violent_src):
                violent_dest = os.path.join(raw_dir, "violent")
                # Use copy instead of move to avoid permission issues
                if not os.path.exists(violent_dest):
                    print(f"Copying violent videos to {violent_dest}")
                    shutil.copytree(violent_src, violent_dest)
            
            if os.path.exists(nonviolent_src):
                nonviolent_dest = os.path.join(raw_dir, "non-violent")
                if not os.path.exists(nonviolent_dest):
                    print(f"Copying non-violent videos to {nonviolent_dest}")
                    shutil.copytree(nonviolent_src, nonviolent_dest)
                
            print(f"Dataset successfully copied to {raw_dir}")
        else:
            # Alternative approach - download from dataset releases
            print("Dataset not found in repository. Attempting to download from releases...")
            # Add fallback URL if available
            fallback_download()
        
        # Clean up temporary repository
        print("Cleaning up temporary files...")
        time.sleep(1)  # Brief pause to ensure all file handles are closed
        safe_rmtree(temp_repo_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning the repository: {e}")
        # Fallback method for downloading dataset
        fallback_download()
    except Exception as e:
        print(f"Unexpected error: {e}")
        fallback_download()

def fallback_download():
    """
    Fallback method to download dataset if git clone doesn't work
    """
    print("Using fallback method to download dataset...")
    print("Please download the dataset manually from https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos")
    print("and place the violent and non-violent folders in the data/raw directory.")
    
    # Create placeholder directories to maintain structure
    raw_dir = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(os.path.join(raw_dir, "violent"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "non-violent"), exist_ok=True)
    
    # Create dummy files for testing
    print("Creating dummy files for testing purposes...")
    os.makedirs(os.path.join(raw_dir, "violent", "dummy"), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, "non-violent", "dummy"), exist_ok=True)
    
    # Create dummy video files (small text files with video extensions)
    for i in range(5):
        with open(os.path.join(raw_dir, "violent", "dummy", f"violent_sample_{i}.mp4"), 'w') as f:
            f.write("This is a dummy video file for testing purposes.")
        with open(os.path.join(raw_dir, "non-violent", "dummy", f"nonviolent_sample_{i}.mp4"), 'w') as f:
            f.write("This is a dummy video file for testing purposes.")

def main():
    parser = argparse.ArgumentParser(description="Download Violence Detection Dataset")
    args = parser.parse_args()
    
    download_violence_dataset()
    
    print("Dataset download process completed.")
    print("Next step: Run extract_frames.py to extract frames from the videos.")

if __name__ == "__main__":
    main() 