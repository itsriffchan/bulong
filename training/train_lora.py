import os
import sys
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
MODEL_NAME = "openai/whisper-tiny"
LANGUAGE = "tagalog"
TASK = "transcribe"
DATASET_PATH = "./data/processed_dataset"
OUTPUT_DIR = "./training/whisper-tiny-lora-fil"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have different padding strategies
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad label sequences to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id's of target tokenizer with -100 to ignore in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in labels, strip it
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def main():
    print("=" * 60)
    print("PH-Whisper: Whisper LoRA Fine-Tuning Pipeline")
    print("=" * 60)

    # 1. Check if preprocessed dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Preprocessed dataset not found at '{DATASET_PATH}'.")
        print("Please run 'python data/preprocess.py' first.")
        sys.exit(1)

    print(f"Loading preprocessed dataset from '{DATASET_PATH}'...")
    dataset = load_from_disk(DATASET_PATH)

    # 2. Load model and processor
    print(f"Loading model and processor for '{MODEL_NAME}'...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Configure model generation settings
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # 3. PEFT (LoRA) Setup
    print("Configuring LoRA adapters...")
    # Prepare model for PEFT training (handles cast to fp32, gradient checkpointing)
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        decoder_trainable_only=True
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Data Collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 5. Training Arguments
    # We choose conservative settings for CPU/Low GPU capabilities suitable for an MVP run
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=2,
        max_steps=10,             # Keep steps minimal for quick verification
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=5,
        save_steps=5,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=1,
        report_to="none",
        fp16=torch.cuda.is_available()  # Enable fp16 only if CUDA is available
    )

    # 6. Trainer Initialization
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    # 7. Start Training
    print("\nStarting Whisper LoRA fine-tuning...")
    trainer.train()
    
    # Save the adapter model
    print(f"\nSaving fine-tuned LoRA adapters to '{OUTPUT_DIR}'...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    
    print("\nFine-tuning completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
