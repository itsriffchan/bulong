# -*- coding: utf-8 -*-
"""
Model checkpoint evaluation script (Word Error Rate).
"""

import os
import json
import torch
import soundfile as sf
import librosa
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

def resolve_audio_path(wav_path, test_dir):
    clean_path = wav_path.replace("\\", "/")
    if "PLD/" in clean_path:
        sub_path = clean_path.split("PLD/", 1)[1]
    else:
        sub_path = clean_path

    candidates = [
        os.path.join(test_dir, sub_path),
        os.path.join(test_dir, "PLD", sub_path),
        os.path.join(os.path.dirname(test_dir), sub_path),
        os.path.join(os.path.dirname(test_dir), "PLD", sub_path),
        os.path.join(test_dir, clean_path),
        os.path.join(os.path.dirname(test_dir), clean_path),
    ]

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    if test_dir.startswith("/content"):
        return os.path.join("/content", sub_path)
    return os.path.abspath(os.path.join(test_dir, sub_path))

def load_split(json_path):
    test_dir = os.path.dirname(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        item["wav_file"] = resolve_audio_path(item["wav_file"], test_dir)
    return data

def evaluate_model(test_json_path="/content/test.json", model_checkpoint_path="/content/drive/MyDrive/whisper-tiny-philippine-dialects/checkpoint-4000", base_model_name="openai/whisper-tiny", dry_run=False):
    # 1. Load model and tokenizer
    print(f"Loading checkpoint from {model_checkpoint_path}...")
    
    # Handle CPU fallback during dry run if CUDA is unavailable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint_path).to(device)
    processor = WhisperProcessor.from_pretrained(base_model_name, language="Tagalog", task="transcribe")
    metric = evaluate.load("wer")
    model.eval()

    # 2. Load test split
    if not os.path.exists(test_json_path):
        # Recursively search for test.json
        found_test_path = None
        search_roots = ["/content", "."] if os.path.exists("/content") else ["."]
        for s_root in search_roots:
            for root, dirs, files in os.walk(s_root):
                if any(part.startswith('.') or part == 'node_modules' or part == 'venv' or part == '.venv' for part in root.split(os.sep)):
                    continue
                if "test.json" in files:
                    found_test_path = os.path.join(root, "test.json")
                    break
            if found_test_path:
                break
        if found_test_path:
            test_json_path = found_test_path

    print(f"Loading test split from {test_json_path}...")
    test_data = load_split(test_json_path)
    
    if dry_run:
        print("Dry run mode: Slicing test set (2 files)...")
        test_subset = test_data[:2]
    else:
        test_subset = test_data[:100]

    predictions = []
    references = []

    print(f"Evaluating model on {len(test_subset)} test files...")
    for item in tqdm(test_subset):
        audio_path = item["wav_file"]
        reference_text = item["normalized_transcript"]

        # Read and resample audio
        speech, rate = sf.read(audio_path)
        if rate != 16000:
            speech = librosa.resample(speech, orig_sr=rate, target_sr=16000)

        # Extract Mel-features & run model generation
        input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        predictions.append(prediction)
        references.append(reference_text)

    # Compute Word Error Rate
    wer_score = 100 * metric.compute(predictions=predictions, references=references)
    print("\n" + "="*40)
    print(f"Evaluation Complete!")
    print(f"Average Word Error Rate (WER) on Test Set: {wer_score:.2f}%")
    print("="*40)

    # Print a few samples side-by-side
    print("\nSample Comparisons:")
    for i in range(min(5, len(test_subset))):
        print(f"\nAudio: {test_subset[i]['wav_file']}")
        print(f"  Target:     '{references[i]}'")
        print(f"  Prediction: '{predictions[i]}'")

if __name__ == "__main__":
    evaluate_model()
