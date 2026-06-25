# Bulong: Fine-Tuning Whisper for Philippine Multilingual Speech Recognition

# Group Name & Members ↓
## def \_\_init__():
- self.member1 = **"Reese Chan"**
- self.member2 = **"Alfred Nathaniel Embuscado"**
- self.member3 = **"Noel Ocampo"**
- self.member4 = **"John Christopher Umali"**

---

## Project Overview

Bulong is an open-source research project designed to build a reproducible pipeline for fine-tuning OpenAI's Whisper model on Philippine languages using the **UP-DSP Philippine Languages Database (UP-DSP-PLD)**.

The project's goal is to improve automatic speech recognition (ASR) for underrepresented Philippine languages by building foundational infrastructure, including:

* A reproducible data preprocessing pipeline
* A parameter-efficient Whisper fine-tuning setup (using LoRA adapters)
* A standardized evaluation benchmark comparing fine-tuned models against base models
* Open and accessible training and evaluation pipelines for other researchers to extend

Rather than creating a standalone end-user application, Bulong focuses on providing the open-source engineering blueprint to make multilingual speech models accessible in the local research community.

---

## AI Disclosure
This project utilizes AI-assisted tools for development: ChatGPT was used for explaining technical procedures and conceptual mapping of the architecture, while Google DeepMind's Antigravity AI assistant supported the initial structural design, pipeline scaffolding, and codebase optimization. The core machine learning training pipeline is designed to use OpenAI's Whisper model fine-tuned via Low-Rank Adaptation (LoRA) to adapt to low-resource and code-switched Philippine languages.

---

## Motivation & Research Scope

Despite the Philippines being home to more than 130 languages, many remain severely underrepresented in modern speech recognition systems.

General-purpose ASR models such as Whisper perform well on high-resource languages but struggle with:

* Regional Philippine languages (e.g. Cebuano, Kapampangan, Hiligaynon)
* Code-switched speech (Taglish)
* Local accents and pronunciations
* Domain-specific vocabulary

This project will investigate whether parameter-efficient fine-tuning can systematically improve multilingual ASR performance across these low-resource languages while keeping training reproducible and accessible on consumer-grade hardware.

---

## Features We Will Build

* **LoRA-Based Fine-Tuning:** Parameter-efficient adaptation of Whisper models using Hugging Face PEFT to reduce GPU memory requirements.
* **Multilingual Support:** Scaffolding to process and train on multiple Philippine languages concurrently.
* **Automated Data Preprocessing:** Scripts to automatically resample audio to 16kHz, extract log-Mel spectrograms, normalize transcript text, and handle tokenization. Includes a local synthetic fallback to test code pipelines offline or under Hugging Face rate limits.
* **Standardized Evaluation:** Automating WER (Word Error Rate) and CER (Character Error Rate) evaluations across base and fine-tuned models.

---

## Planned Supported Languages

The project is designed to train and evaluate on the [UP-DSP Philippine Languages Database](https://mozilladatacollective.com/datasets/cmmxhw46c00tqnw07xyr94zjk), which contains recordings in:

* Filipino / Tagalog
* English
* Cebuano
* Kapampangan
* Hiligaynon
* Ilokano
* Bikolano
* Waray
* Tausug

---

## Project Structure & Architecture

We will organize the codebase as follows:

```text
Bulong/
│
├── data/
│   ├── preprocess.py      # Resamples audio, extracts spectrograms, tokenizes transcripts
│   └── ...
│
├── training/
│   ├── train_lora.py      # Core script to apply LoRA and run Seq2SeqTrainer
│   └── config.yaml        # Configuration file for model and training hyperparameters
│
├── evaluation/
│   └── evaluate.py        # Generates test transcriptions and computes WER/CER
│
├── .gitignore             # Exclusions (virtual environments, data caches, credentials)
├── README.md
└── requirements.txt       # Project python dependencies
```

---

## Target Tech Stack

* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Model Pipeline & Tokenizer:** Hugging Face Transformers
* **Dataset Management:** Hugging Face Datasets
* **Parameter-Efficient Adapters:** PEFT (LoRA)
* **Training Acceleration:** Accelerate
* **Audio Decoding:** torchcodec, soundfile, librosa
* **Evaluation Metrics:** Evaluate, JiWER

---

## Pipeline Architecture

```text
UP-DSP-PLD (Raw Audio & Text)
        │
        ▼
Data Validation & Normalization
        │
        ▼
Audio Preprocessing (16kHz Resampling)
        │
        ▼
Feature Extraction (Log-Mel Spectrograms)
        │
        ▼
Whisper LoRA Fine-Tuning (Seq2Seq Training)
        │
        ▼
ASR Evaluation (WER / CER Metrics calculation)
        │
        ▼
Baseline Comparisons & Benchmark Results
```

---

## Target Dataset

This project is built around the **UP-DSP Philippine Languages Database (UP-DSP-PLD)**.

The raw dataset is **not redistributed** with this project. Users must obtain the dataset directly from its official source and comply with its licensing terms. The repository will contain only the code required to preprocess, train, and evaluate models using the dataset.

---

## Setup & Running the Pipeline

### 1. Installation & Environment Setup
Activate the virtual environment and install the required libraries:

* **PowerShell**:
  ```powershell
  .venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```
* **Command Prompt (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  pip install -r requirements.txt
  ```

### 2. Running Data Preprocessing
This script will preprocess local files, or run in a streaming demo mode (fetching samples from Hugging Face FLEURS) to verify the pipeline. If rate-limited or offline, it automatically falls back to generating local synthetic files:

```bash
python data/preprocess.py
```

### 3. Executing Whisper LoRA Fine-Tuning
Run the training loop based on the configurations inside `training/config.yaml`. This wraps the model in LoRA adapters and trains it on the preprocessed dataset:

```bash
python training/train_lora.py
```

### 4. Running Evaluation
Test the base or fine-tuned model on the validation/test subsets and calculate ASR accuracy metrics (WER and CER):

```bash
python evaluation/evaluate.py
```

---

## Future Goals

* Support additional Philippine regional languages
* Build code-switching benchmark splits
* Integrate speaker adaptation methods
* Compile quantized models for mobile deployment
* Construct a public evaluation benchmark leaderboard

---

## Acknowledgements

* University of the Philippines – Department of Speech Processing
* OpenAI Whisper Team
* Hugging Face PEFT & Transformers Teams

---

## License

This repository is released under the MIT License.

**Note:** The UP-DSP-PLD dataset is distributed under its own license. Users are responsible for obtaining the dataset legally and complying with its terms.
