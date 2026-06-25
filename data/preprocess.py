import os
import sys
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

# Suppress Hugging Face download warning logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configuration
MODEL_NAME = "openai/whisper-tiny"
LANGUAGE = "tagalog"
TASK = "transcribe"
OUTPUT_DIR = "./data/processed_dataset"

def generate_synthetic_samples(count):
    """Generates synthetic audio arrays and dummy transcripts for offline testing."""
    samples = []
    transcripts = [
        "naimbag a bigat kadakayo amin",
        "marhay na aga sa saindo gabos",
        "maupay nga aga ha iyo ngatanan",
        "kumusta kayo amin a kakailian",
        "damo nga salamat ha pagbulig niyo"
    ]
    
    # Standard 2 seconds of silence/sine wave at 16kHz
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    for i in range(count):
        # Create a simple 440Hz sine wave as dummy audio data
        audio_array = 0.5 * np.sin(2 * np.pi * 440 * t)
        samples.append({
            "audio": {
                "path": None,
                "array": audio_array.astype(np.float32),
                "sampling_rate": sr
            },
            "transcription": transcripts[i % len(transcripts)]
        })
    return samples

def main():
    print("=" * 60)
    print("PH-Whisper: Data Preprocessing Pipeline")
    print("=" * 60)
    
    # 1. Load Feature Extractor and Tokenizer
    print(f"Loading feature extractor and tokenizer for '{MODEL_NAME}'...")
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    except Exception as e:
        print(f"Error loading tokenizer/feature extractor: {e}")
        sys.exit(1)

    # 2. Load Dataset (Attempt HF Streaming with local fallback)
    print("\nAttempting to load Google FLEURS Filipino ('fil_ph') dataset via streaming...")
    try:
        # Stream the training split of Filipino FLEURS
        train_stream = load_dataset("google/fleurs", "fil_ph", split="train", streaming=True)
        train_samples = list(train_stream.take(10))
        
        # Stream the validation split of Filipino FLEURS
        val_stream = load_dataset("google/fleurs", "fil_ph", split="validation", streaming=True)
        val_samples = list(val_stream.take(5))
        
        print(f"Successfully loaded {len(train_samples)} training and {len(val_samples)} validation samples via streaming!")
    except Exception as e:
        print(f"\n[Warning] Could not stream from Hugging Face: {e}")
        print("Falling back to generating local synthetic audio dataset for offline testing...")
        
        train_samples = generate_synthetic_samples(10)
        val_samples = generate_synthetic_samples(5)
        print(f"Generated {len(train_samples)} synthetic training samples and {len(val_samples)} validation samples.")

    # Convert the list back to Hugging Face Datasets
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    
    raw_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    # 3. Preprocessing function
    def prepare_dataset(batch):
        audio = batch["audio"]
        # Compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Encode target text to label ids
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    # 4. Apply Preprocessing
    print("\nProcessing audio and text data (resampling to 16kHz & tokenizing transcripts)...")
    processed_dataset = raw_dataset.map(
        prepare_dataset,
        remove_columns=raw_dataset["train"].column_names
    )

    # 5. Save processed dataset to disk
    print(f"\nSaving preprocessed dataset to disk at '{OUTPUT_DIR}'...")
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    processed_dataset.save_to_disk(OUTPUT_DIR)
    
    print("\nPreprocessing completed successfully!")
    print(f"Processed dataset size: {len(processed_dataset['train'])} train samples, {len(processed_dataset['validation'])} validation samples.")
    print("=" * 60)

if __name__ == "__main__":
    main()
