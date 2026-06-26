import os
import json
import torch
import dataclasses
from typing import Any, Dict, List, Union
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate

# --- 1. Load splits and prepare Hugging Face datasets ---
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

def to_hf_dataset(split_data):
    hf_dict = {
        "audio": [item["wav_file"] for item in split_data],
        "sentence": [item["normalized_transcript"] for item in split_data],
        "speaker_id": [item["speaker_id"] for item in split_data],
        "dialect": [item["dialect"] for item in split_data]
    }
    dataset = Dataset.from_dict(hf_dict)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

raw_dataset = DatasetDict({
    "train": to_hf_dataset(train_data),
    "validation": to_hf_dataset(val_data)
})

# --- 2. Load model, tokenizer, and processor ---
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name, language="Tagalog", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Tagalog", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="Tagalog", task="transcribe")
model.config.suppress_tokens = []

# --- 3. Data Collator ---
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

# --- 4. Training Arguments & Trainer ---
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
print("Training complete! The best model checkpoint has been saved to your Google Drive.")
