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

def preprocess_dataset(file_id="17gyqQrLWpdse2iceosEZxTVwn0CDUNKT", destination="/content"):
    # Download and extract the PLD dataset
    local_zip_path = os.path.join(destination, "PLD.zip")
    pld_dir = os.path.join(destination, "PLD")

    if not os.path.exists(pld_dir):
        print(f"Downloading PLD.zip to {local_zip_path}...")
        os.system(f'gdown --id "{file_id}" -O "{local_zip_path}"')
        
        if os.path.exists(local_zip_path):
            print("Extracting PLD.zip...")
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(destination)
            os.remove(local_zip_path)
            print("Extraction complete.")
        else:
            print("Dataset ZIP download failed. Please ensure Drive folder paths or IDs are correct.")

    # --- 2. Load Splits & Adjust Paths ---
    print("Loading dataset splits...")
    train_data = load_split(os.path.join(destination, "train.json"))
    val_data = load_split(os.path.join(destination, "val.json"))

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
