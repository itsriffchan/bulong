# -*- coding: utf-8 -*-
"""
Model training script for Whisper.
"""

import os
import json
import torch
import dataclasses
from typing import Any, Dict, List
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate

# Data Collator class
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

def train_model(raw_dataset, processor, tokenizer, model_name="openai/whisper-tiny", output_dir="/content/drive/MyDrive/whisper-tiny-philippine-dialects"):
    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="Tagalog", task="transcribe")
    model.config.suppress_tokens = []

    # Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Metrics Setup
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
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

    # Initialize Trainer
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
    print("Training complete! The best model checkpoint has been saved.")

if __name__ == "__main__":
    # If run as standalone, load preprocess module dynamically
    import sys
    # Add root folder if needed
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.preprocess import preprocess_dataset
    raw_dataset, processor, tokenizer = preprocess_dataset()
    train_model(raw_dataset, processor, tokenizer)
