# -*- coding: utf-8 -*-
"""
Bulong: A Reproducible Whisper Fine-Tuning Pipeline for Philippine Multilingual Speech Recognition.

Orchestrator script importing step-by-step modular code for running in Google Colab.
"""

import os
import sys

# --- 1. Google Drive Check ---
if not os.path.exists('/content/drive'):
    print("Warning: Google Drive not detected at /content/drive.")
    print("If you are running in Colab, please run 'from google.colab import drive; drive.mount(\"/content/drive\")' in a notebook cell first.")
else:
    print("Google Drive is mounted and accessible.")

# --- 2. Repository Path Check ---
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

# --- 3. Pipeline Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a quick 1-minute sanity check with a tiny dataset and 2 steps.")
    parser.add_argument("--data-dir", type=str, default="/content", help="Directory where the dataset folder or archive is located.")
    parser.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/whisper-tiny-philippine-dialects", help="Directory to save model checkpoints.")
    parser.add_argument("--test-json", type=str, default="/content/test.json", help="Path to the test.json split file.")
    args = parser.parse_args()

    dry_run = args.dry_run
    if dry_run:
        print("\n" + "="*60)
        print("RUNNING IN DRY RUN MODE (SANITY CHECK)")
        print("="*60)

    print("\n" + "="*60)
    print("Step 1: Dataset Preprocessing")
    print("="*60)
    raw_dataset, processor, tokenizer = preprocess_dataset(destination=args.data_dir, dry_run=dry_run)

    print("\n" + "="*60)
    print("Step 2: Training the Whisper Model")
    print("="*60)
    train_model(raw_dataset, processor, tokenizer, output_dir=args.output_dir, dry_run=dry_run)

    print("\n" + "="*60)
    print("Step 3: Evaluating the Fine-Tuned Model Checkpoint")
    print("="*60)
    
    # Configure path to checkpoint folder
    checkpoint_path = os.path.join(args.output_dir, "checkpoint-4000")
    if dry_run:
        checkpoint_path = os.path.join(args.output_dir, "checkpoint-2")
        # If the output folder doesn't exist, fall back to base model just to verify code flow
        if not os.path.exists(checkpoint_path):
            print("Warning: Dry run checkpoint not found. Evaluating base model instead.")
            checkpoint_path = "openai/whisper-tiny"

    evaluate_model(test_json_path=args.test_json, model_checkpoint_path=checkpoint_path, dry_run=dry_run)