# -*- coding: utf-8 -*-
"""
Bulong: A Reproducible Whisper Fine-Tuning Pipeline for Philippine Multilingual Speech Recognition.

Orchestrator script importing step-by-step modular code for running in Google Colab.
"""

import os
import sys

# --- 1. Google Drive Mounting ---
try:
    from google.colab import drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
except ImportError:
    print("Not running in Google Colab or Colab Environment not detected.")

# --- 3. Repository Path Check ---
# Add the project root to python path so we can import from data, training, evaluation
project_root = "/content/Bulong"
if os.path.exists(project_root):
    sys.path.append(project_root)
else:
    sys.path.append(os.getcwd())

# Import functions from project files
try:
    from data.preprocess import preprocess_dataset
    from training.train import train_model
    from evaluation.evaluate import evaluate_model
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you cloned the repository and added the directory path to sys.path.")
    sys.exit(1)

# --- 4. Pipeline Execution ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Step 1: Dataset Downloading & Preprocessing")
    print("="*60)
    raw_dataset, processor, tokenizer = preprocess_dataset()

    print("\n" + "="*60)
    print("Step 2: Training the Whisper Model")
    print("="*60)
    train_model(raw_dataset, processor, tokenizer)

    print("\n" + "="*60)
    print("Step 3: Evaluating the Fine-Tuned Model Checkpoint")
    print("="*60)
    evaluate_model()