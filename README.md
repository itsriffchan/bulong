# Bulong: A Reproducible Whisper Fine-Tuning Pipeline for Philippine Multilingual Speech Recognition

## def __init__():

* self.member1 = **"Reese Chan"**
* self.member2 = **"Alfred Nathaniel Embuscado"**
* self.member3 = **"Noel Ocampo"**
* self.member4 = **"John Christopher Umali"**

---

## AI Disclosure
This project utilizes AI-assisted tools (ChatGPT and Google DeepMind's Antigravity AI) for conceptual explanation, architecture planning, and engineering implementation guidance. All evaluation and engineering decisions are made by the project team.

---

# Overview & Objectives

**Bulong** provides a reproducible research pipeline for fine-tuning OpenAI's Whisper model on regional Philippine languages using the **UP-DSP Philippine Languages Database (UP-DSP-PLD)**.

### Challenges Addressed
Out of over 130 Philippine languages, very few have resources for Automatic Speech Recognition (ASR). Baseline Whisper models lack specific optimization for regional dialects and accents. Bulong addresses these issues by providing a structured, parameter-flexible pipeline that reduces data preprocessing overhead and creates a standardized benchmark.

### Core Features
* **Reproducible Data Pipeline**: Automated transcript normalization, audio resampling, and conversion to Hugging Face datasets.
* **Modular Fine-Tuning**: Whisper training using the Hugging Face `Trainer` API (fully supports full parameter tuning, and can easily extend to parameter-efficient methods like LoRA).
* **Standardized Evaluation**: Automated WER (Word Error Rate) benchmark evaluations.
* **Multilingual Support**: Initial focus on regional languages: **Ilokano, Bikolano, and Waray**.

---

# Supported Dataset

This project uses the **UP-DSP Philippine Languages Database (UP-DSP-PLD)**[Link](https://mozilladatacollective.com/datasets/cmmxhw46c00tqnw07xyr94zjk).
> [!IMPORTANT]
> The dataset is **not redistributed** with this repository. Users must obtain access through the official UP-DSP-PLD distribution channels and comply with its licensing terms.

---

# Repository Structure

```text
Bulong/
│
├── run_pipeline.py        # End-to-end training and evaluation pipeline
│
├── data/
│   └── preprocess.py      # Preprocesses and sets up dataset splits
│
├── training/
│   ├── train.py           # Model fine-tuning logic
│
├── evaluation/
│   └── evaluate.py        # Model checkpoint evaluation (WER)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Setup & Usage

### 1. Installation
Create a virtual environment and install the required dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows: .venv\Scripts\Activate.ps1
# On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Flow

#### Running the Full Pipeline (Recommended)
You can run the entire dataset downloading, preprocessing, training, and evaluation pipeline with a single orchestrator command:
```bash
python run_pipeline.py
```

#### Running Modular Steps Individually
Alternatively, you can run individual stages of the pipeline separately:

* **Preprocessing**:
  ```bash
  python data/preprocess.py
  ```
* **Training**:
  ```bash
  python training/train.py
  ```
* **Evaluation**:
  ```bash
  python evaluation/evaluate.py
  ```

---

# Tech Stack & Libraries
* **Core**: Python, PyTorch
* **ML Ecosystem**: Hugging Face (Transformers, Datasets, Accelerate)
* **Audio & Metrics**: Librosa, Soundfile, JiWER, Evaluate

---

# Future Work
* Parameter-efficient fine-tuning (LoRA) integration.
* Support for additional Philippine languages and code-switched speech recognition.
* Quantized weights for edge-device deployment.
* Public multilingual ASR benchmark leaderboard.

---

# Acknowledgements & License
* UP-DSP Philippine Languages Database team.
* OpenAI Whisper and Hugging Face developers.
* Mozilla Data Collective.

The source code in this repository is released under the **MIT License**.
