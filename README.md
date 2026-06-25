# Bulong: A Reproducible Whisper Fine-Tuning Pipeline for Philippine Multilingual Speech Recognition

# Group Name & Members ↓

## def __init__():

* self.member1 = **"Reese Chan"**
* self.member2 = **"Alfred Nathaniel Embuscado"**
* self.member3 = **"Noel Ocampo"**
* self.member4 = **"John Christopher Umali"**

---

# Project Overview

**Bulong** is an open-source research project that provides a reproducible pipeline for adapting OpenAI's Whisper model to Philippine languages using the **UP-DSP Philippine Languages Database (UP-DSP-PLD)**.

Rather than building an end-user speech recognition application, Bulong focuses on creating reusable infrastructure that allows researchers and developers to preprocess data, fine-tune Whisper using parameter-efficient techniques, and evaluate multilingual speech recognition performance.

The project aims to lower the barrier to Philippine ASR research by providing a complete, reproducible workflow from raw speech recordings to a fine-tuned model.

---

# AI Disclosure

This project utilizes AI-assisted tools during development.

* **ChatGPT** was used to explain machine learning concepts, assist with architecture planning, and provide technical guidance throughout development.
* **Google DeepMind's Antigravity AI** was used during the initial project planning, repository scaffolding, and implementation discussions.

All engineering decisions, implementation, experimentation, and evaluation are performed by the project team.

---

# Motivation

Despite the Philippines being home to more than **130 languages**, only a handful have sufficient resources for modern speech recognition research.

Although models such as OpenAI Whisper support multilingual speech recognition, they are not specifically optimized for Philippine languages, regional accents, or code-switched speech.

Researchers wishing to improve these models often face several challenges:

* Limited reproducible training pipelines
* Dataset preprocessing complexity
* Lack of standardized evaluation workflows
* High hardware requirements for full model fine-tuning

Bulong addresses these problems by providing an accessible, reproducible pipeline for adapting Whisper to Philippine speech using parameter-efficient fine-tuning.

---

# Objectives

The project's primary objective is to build a reusable research toolkit for Philippine multilingual ASR.

The initial MVP includes:

* Automated preprocessing for UP-DSP-PLD
* Whisper Small fine-tuning using LoRA adapters
* Reproducible training scripts
* Evaluation using Word Error Rate (WER) and Character Error Rate (CER)
* Benchmark comparison against the original Whisper model

The project is designed so that future researchers can easily extend it to additional Philippine languages and datasets.

---

# Planned Features

## Reproducible Data Pipeline

* Dataset validation
* Transcript normalization
* Audio preprocessing
* Dataset conversion into Hugging Face-compatible format

---

## Parameter-Efficient Fine-Tuning

Fine-tune Whisper using **Low-Rank Adaptation (LoRA)** through Hugging Face PEFT.

Instead of updating every model parameter, LoRA trains lightweight adapter weights, significantly reducing GPU memory requirements while maintaining strong performance.

---

## Multilingual Support

The pipeline is designed to support multiple Philippine languages contained within the UP-DSP-PLD corpus.

The experiments will focus specifically on **Ilokano, Bikolano, and Waray**, showing the model's capacity to adapt to regional languages.

---

## Standardized Evaluation

Automatically compare:

* Original Whisper
* Fine-tuned Whisper

using:

* Word Error Rate (WER)
* Character Error Rate (CER)

Evaluation scripts will be reusable for future experiments.

---

# Supported Dataset

This project is built around the **UP-DSP Philippine Languages Database (UP-DSP-PLD)**.

The project focuses on speech recordings in:

* Ilokano
* Bikolano
* Waray

The dataset itself is **not redistributed** with this repository.

Users must obtain access through the official UP-DSP-PLD distribution channel and comply with its licensing terms.

---

# Project Structure

```text
Bulong/
│
├── data/
│   ├── preprocess.py
│   ├── validate.py
│   └── create_splits.py
│
├── training/
│   ├── train_lora.py
│   └── config.yaml
│
├── evaluation/
│   ├── evaluate.py
│   ├── wer.py
│   └── cer.py
│
├── notebooks/
│   └── explore_dataset.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Pipeline

```text
UP-DSP-PLD
      │
      ▼
Dataset Validation
      │
      ▼
Transcript Normalization
      │
      ▼
Audio Preprocessing
      │
      ▼
Whisper Feature Extraction
      │
      ▼
LoRA Fine-Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Benchmark Report
```

---

# Tech Stack

**Programming Language**

* Python

**Machine Learning**

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets

**Parameter-Efficient Training**

* PEFT (LoRA)

**Training**

* Accelerate

**Audio Processing**

* librosa
* soundfile
* torchcodec

**Evaluation**

* Evaluate
* JiWER

---

# Benchmark

The project compares the original Whisper model against the fine-tuned model using:

* Word Error Rate (WER)
* Character Error Rate (CER)

Results will be reported per experiment to quantify improvements in Philippine multilingual speech recognition.

---

# Installation

## Create a virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows (CMD)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Usage

## 1. Obtain the Dataset

Request or download the UP-DSP Philippine Languages Database through its official distribution source.

---

## 2. Preprocess the Dataset

```bash
python data/preprocess.py
```

---

## 3. Fine-Tune Whisper

```bash
python training/train_lora.py
```

---

## 4. Evaluate the Model

```bash
python evaluation/evaluate.py
```

---

# Future Work

The project is intentionally modular and can be extended to include:

* Additional Philippine languages
* Code-switched speech recognition
* Quantized models for edge devices
* Public multilingual ASR benchmark leaderboard
* Additional ASR model comparisons beyond Whisper

---

# Acknowledgements

* University of the Philippines – Department of Speech Processing
* UP-DSP Philippine Languages Database
* OpenAI Whisper
* Hugging Face (Transformers, Datasets, PEFT, Accelerate)
* Mozilla Data Collective

---

# License

The source code in this repository is released under the **MIT License**.

The **UP-DSP Philippine Languages Database (UP-DSP-PLD)** is distributed under its own license and **is not included** in this repository.

Users are responsible for obtaining the dataset legally and complying with all applicable licensing and research-use restrictions.
