import json
import torch
import soundfile as sf
import librosa
import evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

# 1. Load the model and tokenizer
model_path = "/content/drive/MyDrive/whisper-tiny-philippine-dialects/checkpoint-4000"
model = WhisperForConditionalGeneration.from_pretrained(model_path).to("cuda") # Run on GPU
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="Tagalog", task="transcribe")
metric = evaluate.load("wer")

# 2. Load test split and adjust paths
with open('/content/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

for item in test_data:
    # Normalize paths by converting backslashes and mapping to the Colab environment
    wav_path = item["wav_file"].replace("\\", "/")
    item["wav_file"] = "/content/" + wav_path.split("PLD/", 1)[1] if "PLD/" in wav_path else wav_path

# 3. Evaluate on a subset of the test set (e.g., first 100 files for quick results)
test_subset = test_data[:100]
predictions = []
references = []

print(f"Evaluating model on {len(test_subset)} test files...")
for item in tqdm(test_subset):
    audio_path = item["wav_file"]
    reference_text = item["normalized_transcript"]

    # Read audio
    speech, rate = sf.read(audio_path)
    if rate != 16000:
        speech = librosa.resample(speech, orig_sr=rate, target_sr=16000)

    # Extract features
    input_features = processor(speech, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")

    # Generate transcript
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    predictions.append(prediction)
    references.append(reference_text)

# 4. Compute Word Error Rate (WER)
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
