# -*- coding: utf-8 -*-
"""
Data preprocessing script for UP-DSP Philippine Languages Database (PLD).
"""

import os
import zipfile
import json
import re
import random
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

def generate_splits_from_logs(dataset_dir, output_dir):
    print(f"Generating dataset split files from logs in {dataset_dir}...")
    all_samples = []
    
    # Recursively find all log files
    log_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.log'):
                log_files.append(os.path.join(root, f))
                
    print(f"Found {len(log_files)} session log files.")
    
    for log_path in log_files:
        log_dir = os.path.dirname(log_path)
        
        # Read the log file
        metadata = {}
        data_lines = []
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if '=' in line and not ('.wav' in line.lower()):
                        parts = line.split('=', 1)
                        key = parts[0].strip()
                        val = parts[1].strip().strip('"')
                        metadata[key] = val
                    elif '.wav' in line.lower():
                        data_lines.append(line)
        except Exception as e:
            print(f"Error reading log file {log_path}: {e}")
            continue
            
        speaker_id = metadata.get('SpeakerID') or metadata.get('SpeakerName') or 'unknown'
        dialect = metadata.get('SpekaerDialect') or metadata.get('SpeakerDialect') or 'unknown'
        
        for line in data_lines:
            match = re.match(r'^([^\s]+\.wav)\s+"([^"]*)"\s+"([^"]*)"', line)
            if match:
                wav_name = match.group(1)
                orig_file = match.group(2)
                transcript = match.group(3).strip()
            else:
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    wav_name = parts[0]
                    orig_file = parts[1].strip('"')
                    transcript = parts[2].strip().strip('"').strip()
                else:
                    continue
            
            wav_path = os.path.join(log_dir, wav_name)
            
            if os.path.exists(wav_path):
                sample = {
                    "wav_file": os.path.abspath(wav_path),
                    "original_file": orig_file,
                    "normalized_transcript": transcript,
                    "speaker_id": speaker_id,
                    "dialect": dialect
                }
                all_samples.append(sample)
                
    print(f"Total valid samples found: {len(all_samples)}")
    if not all_samples:
        print("Warning: No valid speech samples found.")
        return False
        
    random.seed(42)
    random.shuffle(all_samples)
    
    total = len(all_samples)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    train_split = all_samples[:train_end]
    val_split = all_samples[train_end:val_end]
    test_split = all_samples[val_end:]
    
    print(f"Splits generated: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test.")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_split, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=2, ensure_ascii=False)
        
    print(f"Splits saved to {output_dir}")
    return True

def resolve_audio_path(wav_path, train_dir):
    clean_path = wav_path.replace("\\", "/")
    if "PLD/" in clean_path:
        sub_path = clean_path.split("PLD/", 1)[1]
    else:
        sub_path = clean_path

    candidates = [
        os.path.join(train_dir, sub_path),
        os.path.join(train_dir, "PLD", sub_path),
        os.path.join(os.path.dirname(train_dir), sub_path),
        os.path.join(os.path.dirname(train_dir), "PLD", sub_path),
        os.path.join(train_dir, clean_path),
        os.path.join(os.path.dirname(train_dir), clean_path),
    ]

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    if train_dir.startswith("/content"):
        return os.path.join("/content", sub_path)
    return os.path.abspath(os.path.join(train_dir, sub_path))

def load_split(json_path):
    train_dir = os.path.dirname(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        item["wav_file"] = resolve_audio_path(item["wav_file"], train_dir)
        if "original_file" in item:
            item["original_file"] = resolve_audio_path(item["original_file"], train_dir)
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

def find_pld_dir(search_root):
    if not os.path.exists(search_root):
        return None
    for root_dir, dirs, files in os.walk(search_root):
        # Skip hidden and system/venv/drive folders to avoid slow recursive search
        if any(part.startswith('.') or part == 'node_modules' or part == 'venv' or part == '.venv' or part == 'drive' for part in root_dir.split(os.sep)):
            continue
        log_files = [f for f in files if f.endswith('.log')]
        if log_files:
            log_path = os.path.join(root_dir, log_files[0])
            speaker_dir = os.path.dirname(log_path)
            lang_dir = os.path.dirname(speaker_dir)
            pld_dir = os.path.dirname(lang_dir)
            return pld_dir
    return None

def preprocess_dataset(destination="/content", dry_run=False):
    # Scan destination directory to find any existing folder containing log files
    pld_dir = find_pld_dir(destination)

    # If not found and not in Colab, fallback to checking local "./data" directory or current directory
    if not pld_dir and not destination.startswith("/content"):
        pld_dir = find_pld_dir("./data") or find_pld_dir(".")

    # If still not found, try to locate and extract an archive
    if not pld_dir or not os.path.exists(pld_dir):
        # Scan the directory for any ZIP or TAR archive containing "pld" or "up-dsp" (case-insensitive)
        archive_files = [
            f for f in os.listdir(destination)
            if (f.lower().endswith('.zip') or f.lower().endswith('.tar.gz') or f.lower().endswith('.tgz'))
            and ('pld' in f.lower() or 'up-dsp' in f.lower())
        ]
        # Prioritize .tar.gz and .tgz over .zip
        archive_files.sort(key=lambda f: 0 if f.lower().endswith(('.tar.gz', '.tgz')) else 1)
        
        if not archive_files and destination == "/content":
            drive_path = "/content/drive/MyDrive"
            if os.path.exists(drive_path):
                print(f"Scanning Google Drive ({drive_path}) for dataset archives...")
                drive_archives = [
                    f for f in os.listdir(drive_path)
                    if (f.lower().endswith('.zip') or f.lower().endswith('.tar.gz') or f.lower().endswith('.tgz'))
                    and ('pld' in f.lower() or 'up-dsp' in f.lower())
                ]
                # Prioritize .tar.gz and .tgz over .zip
                drive_archives.sort(key=lambda f: 0 if f.lower().endswith(('.tar.gz', '.tgz')) else 1)
                
                if drive_archives:
                    drive_file = drive_archives[0]
                    drive_src = os.path.join(drive_path, drive_file)
                    local_dest = os.path.join(destination, drive_file)
                    print(f"Found archive on Google Drive: '{drive_file}'. Copying to local Colab storage...")
                    import shutil
                    shutil.copy(drive_src, local_dest)
                    archive_files = [drive_file]

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

            # Scan again to locate the extracted folder name
            pld_dir = find_pld_dir(destination)
            if not pld_dir:
                pld_dir = os.path.join(destination, "PLD")
        else:
            print("\n" + "="*60)
            print("ERROR: UP-DSP-PLD dataset not found!")
            print("="*60)
            print("The UP-DSP Philippine Languages Database is licensed and cannot be distributed publicly.")
            print("Please obtain the dataset officially and place it in your workspace.")
            print(f"\nExpected destination directory: {os.path.abspath(destination)}")
            print("If running in Google Colab:")
            print("  1. Mount your Google Drive.")
            print("  2. Copy the zip/tarball to '/content/' before executing the pipeline.")
            print("="*60 + "\n")
            raise FileNotFoundError("Dataset archive/folder not found.")

    # --- 2. Load Splits & Adjust Paths ---
    train_path = None
    val_path = None

    # Walk the destination directory to find train.json and val.json recursively
    for root, dirs, files in os.walk(destination):
        if "train.json" in files:
            train_path = os.path.join(root, "train.json")
        if "val.json" in files:
            val_path = os.path.join(root, "val.json")

    # If still not found, try walking from the current directory
    if not train_path or not val_path:
        for root, dirs, files in os.walk("."):
            if any(part.startswith('.') or part == 'node_modules' or part == 'venv' or part == '.venv' for part in root.split(os.sep)):
                continue
            if "train.json" in files:
                train_path = os.path.join(root, "train.json")
            if "val.json" in files:
                val_path = os.path.join(root, "val.json")

    if not train_path or not val_path:
        print("JSON split files not found. Attempting to generate splits from log files...")
        if pld_dir and os.path.exists(pld_dir):
            success = generate_splits_from_logs(pld_dir, destination)
            if success:
                train_path = os.path.join(destination, "train.json")
                val_path = os.path.join(destination, "val.json")
            else:
                raise FileNotFoundError("Could not generate splits: no valid log/wav samples found in dataset directory.")
        else:
            raise FileNotFoundError(f"Could not locate dataset folder to generate splits. Please ensure dataset is extracted.")

    print(f"Found train.json at: {train_path}")
    print(f"Found val.json at: {val_path}")

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
