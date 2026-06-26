# -*- coding: utf-8 -*-
"""
Bulong: A Reproducible Whisper Fine-Tuning Pipeline for Philippine Multilingual Speech Recognition.

Cleaned, consolidated end-to-end pipeline for Google Colab.
"""

import os
import zipfile
import json
import torch
import dataclasses
import soundfile as sf
import librosa
import evaluate
from typing import Any, Dict, List
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# --- 1. Environment & Setup ---
try:
    import datacollective
except ImportError:
    print("Installing required libraries...")
    os.system("pip install datacollective datasets transformers[torch] accelerate evaluate soundfile librosa jiwer gdown")

# Mount Google Drive
from google.colab import drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# --- 2. Dataset Download & Extraction ---
FILE_ID = "17gyqQrLWpdse2iceosEZxTVwn0CDUNKT"
drive_zip_path = "/content/drive/MyDrive/PLD.zip"
local_zip_path = "/content/PLD.zip"

if os.path.exists(drive_zip_path):
    print("Copying PLD.zip from Drive to Colab local storage...")
    os.system(f'cp "{drive_zip_path}" "{local_zip_path}"')
else:
    print("PLD.zip not found in Drive. Downloading directly via gdown...")
    os.system(f'gdown --id "{FILE_ID}" -O "{local_zip_path}"')

if os.path.exists(local_zip_path):
    print("Extracting PLD.zip...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall("/content/")
    os.remove(local_zip_path)
    print("Extraction complete. Local zip file cleaned up.")
else:
    raise FileNotFoundError("Dataset ZIP file could not be retrieved.")

# --- 3. Load & Format Dataset Splits ---
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

print("Loading dataset splits...")
train_data = load_split('/content/train.json')
val_data = load_split('/content/val.json')
test_data = load_split('/content/test.json')

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

raw_dataset = DatasetDict({
    "train": to_hf_dataset(train_data),
    "validation": to_hf_dataset(val_data)
})
print("Dataset loaded successfully:")
print(raw_dataset)

# --- 4. Load Whisper Model & Processor ---
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name, language="Tagalog", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Tagalog", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="Tagalog", task="transcribe")
model.config.suppress_tokens = []

# --- 5. Data Collator ---
@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [feature["audio"]["array"] for feature in features]
        sampling_rates = [feature["audio"]["sampling_rate"] for feature in features]

        input_features = [
            self.processor.feature_extractor(arr, sampling_rate=sr).input_features[0]
            for arr, sr in zip(audio_arrays, sampling_rates)
        ]

        batch = self.processor.feature_extractor.pad(
            [{"input_features": feat} for feat in input_features],
            return_tensors="pt"
        )

        sentences = [feature["sentence"] for feature in features]
        labels_batch = self.processor.tokenizer(sentences, return_tensors="pt", padding=True)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 6. Metrics & Training Setup ---
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/whisper-tiny-philippine-dialects",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=50,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_num_workers=2,
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor
)

print("Starting training...")
trainer.train()
print("Training complete! The best model checkpoint has been saved to Google Drive.")

# --- 7. Evaluation ---
print("Loading best model checkpoint for evaluation...")
model_path = "/content/drive/MyDrive/whisper-tiny-philippine-dialects/checkpoint-4000"
model = WhisperForConditionalGeneration.from_pretrained(model_path).to("cuda")
model.eval()

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