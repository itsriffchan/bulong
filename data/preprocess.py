# -*- coding: utf-8 -*-
"""
Data preprocessing script for UP-DSP Philippine Languages Database (PLD).
"""

import os
import zipfile
import json
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

def load_split(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        # Normalize paths by converting backslashes and mapping to the Colab environment
        wav_path = item["wav_file"].replace("\\", "/")
        item["wav_file"] = "/content/" + wav_path.split("PLD/", 1)[1] if "PLD/" in wav_path else wav_path
        orig_path = item["original_file"].replace("\\", "/")
        item["original_file"] = "/content/" + orig_path.split("PLD/", 1)[1] if "PLD/" in orig_path else orig_path
    return data

def to_hf_dataset(split_data):
    hf_dict = {
        "audio": [item["wav_file"] for item in split_data],
        "sentence": [item["normalized_transcript"] for item in split_data],
        "speaker_id": [item["speaker_id"] for item in split_data],
        "dialect": [item["dialect"] for item in split_data]
    }
    dataset = Dataset.from_dict(hf_dict)
    # Cast audio column to Audio format (automatically handles resampling to 16kHz)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def preprocess_dataset(destination="/content", dry_run=False):
    # Check if the extracted PLD dataset directory exists
    pld_dir = os.path.join(destination, "PLD")

    # Fallback to local path if not in Colab environment
    if not os.path.exists(pld_dir) and not destination.startswith("/content"):
        pld_dir = "./data/PLD"
        destination = "./data"

    if not os.path.exists(pld_dir):
        # Scan the directory for any ZIP or TAR archive containing "pld" or "up-dsp" (case-insensitive)
        archive_files = [
            f for f in os.listdir(destination)
            if (f.lower().endswith('.zip') or f.lower().endswith('.tar.gz') or f.lower().endswith('.tgz'))
            and ('pld' in f.lower() or 'up-dsp' in f.lower())
        ]
        
        if archive_files:
            matched_archive = archive_files[0]
            local_archive_path = os.path.join(destination, matched_archive)
            print(f"Found dataset archive: '{matched_archive}'. Extracting...")
            
            if matched_archive.lower().endswith('.zip'):
                with zipfile.ZipFile(local_archive_path, 'r') as zip_ref:
                    zip_ref.extractall(destination)
            elif matched_archive.lower().endswith('.tar.gz') or matched_archive.lower().endswith('.tgz'):
                import tarfile
                with tarfile.open(local_archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(destination)
                    
            os.remove(local_archive_path)
            print("Extraction complete.")
        else:
            print("\n" + "="*60)
            print("ERROR: UP-DSP-PLD dataset not found!")
            print("="*60)
            print("The UP-DSP Philippine Languages Database is licensed and cannot be distributed publicly.")
            print("Please obtain the dataset officially and place it in your workspace.")
            print(f"\nExpected directory path: {os.path.abspath(pld_dir)}")
            print("If running in Google Colab:")
            print("  1. Mount your Google Drive.")
            print("  2. Upload 'PLD.zip' to Drive and copy it to '/content/PLD.zip', or extract 'PLD' directly to '/content/PLD'.")
            print("="*60 + "\n")
            raise FileNotFoundError(f"Dataset directory '{pld_dir}' not found.")

    # --- 2. Load Splits & Adjust Paths ---
    train_path = os.path.join(destination, "train.json")
    val_path = os.path.join(destination, "val.json")

    # Fallback to check inside PLD directory
    if not os.path.exists(train_path):
        train_path = os.path.join(pld_dir, "train.json")
        val_path = os.path.join(pld_dir, "val.json")

    if not os.path.exists(train_path):
        print("\n" + "="*60)
        print("ERROR: train.json or val.json not found!")
        print("="*60)
        print(f"Looked in: '{os.path.abspath(destination)}' and '{os.path.abspath(pld_dir)}'")
        print("Please ensure your dataset contains the split files ('train.json' and 'val.json').")
        print("="*60 + "\n")
        raise FileNotFoundError(f"Could not find 'train.json' or 'val.json'.")

    print(f"Loading dataset splits from {train_path} and {val_path}...")
    train_data = load_split(train_path)
    val_data = load_split(val_path)

    if dry_run:
        print("Dry run mode: Slicing dataset splits (5 train, 2 validation)...")
        train_data = train_data[:5]
        val_data = val_data[:2]

    # --- 3. Convert to Hugging Face Format ---
    print("Preparing Hugging Face datasets...")
    raw_dataset = DatasetDict({
        "train": to_hf_dataset(train_data),
        "validation": to_hf_dataset(val_data)
    })
    print("Dataset structure:")
    print(raw_dataset)

    # --- 4. Load Whisper Feature Extractor & Tokenizer ---
    model_name = "openai/whisper-tiny"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Tagalog", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language="Tagalog", task="transcribe")
    
    return raw_dataset, processor, tokenizer

if __name__ == "__main__":
    preprocess_dataset()
