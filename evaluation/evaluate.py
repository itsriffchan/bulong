# -*- coding: utf-8 -*-
"""
Model checkpoint evaluation script (Word Error Rate).
"""

import json
import torch
import soundfile as sf
import librosa
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

def load_split(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        # Normalize paths by converting backslashes and mapping to the Colab environment
        wav_path = item["wav_file"].replace("\\", "/")
        item["wav_file"] = "/content/" + wav_path.split("PLD/", 1)[1] if "PLD/" in wav_path else wav_path
    return data

def evaluate_model(test_json_path="/content/test.json", model_checkpoint_path="/content/drive/MyDrive/whisper-tiny-philippine-dialects/checkpoint-4000", base_model_name="openai/whisper-tiny"):
    # 1. Load model and tokenizer
    print(f"Loading checkpoint from {model_checkpoint_path}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint_path).to("cuda")
    processor = WhisperProcessor.from_pretrained(base_model_name, language="Tagalog", task="transcribe")
    metric = evaluate.load("wer")
    model.eval()

    # 2. Load test split
    print(f"Loading test split from {test_json_path}...")
    test_data = load_split(test_json_path)
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
        input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
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
