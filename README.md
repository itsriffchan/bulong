# Bulong: Fine-Tuning Whisper for Philippine Multilingual Speech Recognition

## Overview

Bulong is an open-source research project that provides a reproducible pipeline for fine-tuning OpenAI's Whisper model on Philippine languages using the **UP-DSP Philippine Languages Database (UP-DSP-PLD)**.

The project aims to improve automatic speech recognition (ASR) for underrepresented Philippine languages while providing reusable tools that future researchers and developers can build upon.

Rather than creating another speech application, Bulong focuses on building foundational infrastructure:

* Reproducible data preprocessing
* Parameter-efficient Whisper fine-tuning (LoRA)
* Standardized evaluation benchmarks
* Open training and evaluation pipelines

---

## Motivation

Despite the Philippines being home to more than 130 languages, many remain underrepresented in modern speech recognition systems.

General-purpose ASR models such as Whisper perform well on high-resource languages but often struggle with:

* Regional Philippine languages
* Code-switched speech
* Local accents and pronunciations
* Domain-specific vocabulary

This project investigates whether parameter-efficient fine-tuning can improve multilingual ASR performance while keeping training reproducible and accessible.

---

## Features

* Fine-tune Whisper using LoRA (Low-Rank Adaptation)
* Support for multiple Philippine languages
* Automated dataset preprocessing
* Hugging Face-compatible dataset conversion
* Standardized evaluation using WER and CER

---

## Supported Languages

The project is designed around the UP-DSP Philippine Languages Database, which contains recordings in:

* Filipino
* English
* Cebuano
* Kapampangan
* Hiligaynon
* Ilokano
* Bikolano
* Waray
* Tausug

---

## Project Structure

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
│   ├── inference.py
│   └── config.yaml
│
├── evaluation/
│   ├── evaluate.py
│   ├── wer.py
│   └── cer.py
│
├── notebooks/
│   └── dataset_exploration.ipynb
│
├── README.md
└── requirements.txt
```

---

## Pipeline

```text
UP-DSP-PLD
        │
        ▼
Data Validation
        │
        ▼
Audio Preprocessing
        │
        ▼
Transcript Normalization
        │
        ▼
Dataset Conversion
        │
        ▼
Whisper LoRA Fine-Tuning
        │
        ▼
Evaluation (WER / CER)
        │
        ▼
Model & Benchmark Results
```

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT (LoRA)
* Accelerate
* Evaluate
* JiWER

---

## Evaluation

Models are evaluated using:

* Word Error Rate (WER)
* Character Error Rate (CER)

Performance is compared against the original Whisper model to measure improvements across Philippine languages.

---

## Dataset

This project uses the **UP-DSP Philippine Languages Database (UP-DSP-PLD)**.

The dataset is **not included** in this repository.

Users must request or obtain the dataset directly from its official source and comply with its licensing terms.

This repository contains only the code required to preprocess, train, and evaluate models using the dataset.

---

## Installation

```bash
git clone https://github.com/your-org/Bulong.git
cd Bulong

pip install -r requirements.txt
```

---

## Usage

### 1. Download the dataset

Obtain the UP-DSP-PLD dataset through its official distribution channel.

### 2. Preprocess

```bash
python data/preprocess.py
```

### 3. Train

```bash
python training/train_lora.py
```

### 4. Evaluate

```bash
python evaluation/evaluate.py
```

---

## Future Work

* Support additional Philippine languages
* Code-switching benchmarks
* Speaker adaptation
* Quantized models for mobile deployment
* Public evaluation leaderboard
* Additional ASR model comparisons

---

## Acknowledgements

* University of the Philippines – Department of Speech Processing
* OpenAI Whisper
* Hugging Face
* Mozilla Data Collective

---

## License

This repository is released under the MIT License.

**Note:** The UP-DSP-PLD dataset is distributed under its own license and is **not redistributed** with this project. Users are responsible for obtaining the dataset legally and complying with its terms.
