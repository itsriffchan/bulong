import os
import sys
import torch
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig

try:
    import jiwer
except ImportError:
    print("Error: 'jiwer' library is required for evaluation. Run 'pip install jiwer'.")
    sys.exit(1)

# Configuration
MODEL_NAME = "openai/whisper-tiny"
LORA_MODEL_PATH = "./training/whisper-tiny-lora-fil"
DATASET_PATH = "./data/processed_dataset"
LANGUAGE = "tagalog"
TASK = "transcribe"

def main():
    print("=" * 60)
    print("PH-Whisper: Evaluation & Metrics Pipeline")
    print("=" * 60)

    # 1. Load validation dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Processed dataset not found at '{DATASET_PATH}'.")
        print("Please run 'python data/preprocess.py' first.")
        sys.exit(1)

    print(f"Loading evaluation dataset from '{DATASET_PATH}'...")
    dataset = load_from_disk(DATASET_PATH)["validation"]

    # 2. Determine which model to load
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    
    if os.path.exists(LORA_MODEL_PATH) and os.path.exists(os.path.join(LORA_MODEL_PATH, "adapter_config.json")):
        print(f"\nLoading fine-tuned LoRA adapters from '{LORA_MODEL_PATH}'...")
        # Load base model
        base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    else:
        print(f"\n[Warning] Fine-tuned LoRA model not found at '{LORA_MODEL_PATH}'.")
        print(f"Evaluating the base pre-trained model '{MODEL_NAME}' instead.")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # 3. Transcribe and evaluate
    print("\nTranscribing audio samples...")
    references = []
    hypotheses = []

    # Reload raw dataset for reference transcripts if removing columns erased them
    from datasets import load_dataset
    raw_stream = load_dataset("google/fleurs", "fil_ph", split="validation", streaming=True)
    raw_samples = list(raw_stream.take(len(dataset)))
    
    for idx, batch in enumerate(dataset):
        # Extract features
        input_features = torch.tensor([batch["input_features"]])
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")

        # Generate token predictions
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode predicted text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Get ground truth
        reference = raw_samples[idx]["transcription"]
        
        references.append(reference)
        hypotheses.append(transcription)

        print(f"\nSample {idx+1}:")
        print(f"  Reference:  {reference}")
        print(f"  Prediction: {transcription}")

    # 4. Compute metrics
    print("\nCalculating metrics...")
    try:
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        
        print("\n" + "=" * 40)
        print(f"Evaluation Results:")
        print(f"  Word Error Rate (WER):      {wer:.2%}")
        print(f"  Character Error Rate (CER):   {cer:.2%}")
        print("=" * 40)
    except Exception as e:
        print(f"Error computing metrics: {e}")

if __name__ == "__main__":
    main()
