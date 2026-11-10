<p align="center">
  <img src="Care2ClaimZA-LLM_logo_final.png" alt="Care2ClaimZA-LLM Logo" width="220" />
</p>

<h1 align="center">Care2ClaimZA-LLM</h1>
<p align="center"><b>Fine-Tuned Gemma 3 1B for South African Healthcare Claims Adjudication & Triage</b></p>

---

## 📌 Executive Overview

Navigating private health insurance and medical schemes in South Africa presents significant administrative complexity. **Care2ClaimLLM-ZA** is an open-source, dual-pathway AI initiative designed to translate everyday patient descriptions into administrative guidance and categorize medical claims against scheme policy rules.

Built on **Google's Gemma 3 1B** foundation model using **Parameter-Efficient Fine-Tuning (LoRA)** on a **JAX backend**, the system operates across two core operational pathways:
1. **Task 1 (Multi-Turn Conversational Triage)**: A 3-turn interactive dialogue agent tailored with South African medical terminology, local slang, and demographic context to conduct patient intake and non-diagnostic administrative triage.
2. **Task 2 (Single-Task Claim Policy Classification)**: An automated instruction-tuned classifier that categorizes clinical claim descriptions into three policy decision categories: `In-Cover`, `Out-of-Cover`, and `Requires Pre-Auth`.

---

## 📂 Recommended GitHub Repository Structure

To ensure full reproducibility and clean modularization, the repository is organized into distinct data, script, notebook, and prediction directories:

```text
Care2ClaimLLM-ZA/
├── README.md                                  # Comprehensive Project Overview & Setup
├── LICENSE                                    # MIT License
├── requirements.txt                           # Dependency specifications
├── imports.py                                 # Centralized imports & JAX/Keras environment setup
│
├── data/
│   ├── raw/
│   │   ├── task1_conversation_data_balanced.jsonl   # Raw synthesized 3-turn dialogues
│   │   └── task2_classification_data_balanced.jsonl # Raw algorithmically labelled claim records
│   └── splits/
│       ├── conversation_training_data.jsonl         # Task 1 Training Split (80%)
│       ├── conversation_validation_data.jsonl       # Task 1 Validation Split (10%)
│       ├── conversation_test_data.jsonl             # Task 1 Test Split (10%)
│       ├── classification_training_data.jsonl       # Task 2 Training Split (80%)
│       ├── classification_validation_data.jsonl     # Task 2 Validation Split (10%)
│       └── classification_test_data.jsonl           # Task 2 Test Split (10%)
│
├── predictions/
│   ├── predictions_classification_train.jsonl       # Task 2 Train set prediction outputs
│   ├── predictions_classification_val.jsonl         # Task 2 Validation set prediction outputs
│   └── predictions_classification_test.jsonl        # Task 2 Test set prediction outputs
│
├── scripts/
│   ├── task1/
│   │   ├── task1_data_preparation.py          # Multilingual dialogue generation & balancing
│   │   ├── task1_data_cleaning.py             # Normalization & 80/10/10 stratified splitting
│   │   └── task1_training_and_evaluation.py   # Multi-turn LoRA fine-tuning & evaluation
│   └── task2/
│       ├── task2_data_preparation.py          # Rule engine labelling & class balancing
│       ├── task2_data_cleaning.py             # HTML/Unicode cleaning & partition export
│       └── task2_training_and_evaluation.py   # Single-task LoRA fine-tuning & evaluation
│
└── notebooks/
    ├── Task1_Conversational_Triage.ipynb       # Colab execution notebook for Task 1
    └── Task2_Policy_Classification.ipynb      # Colab execution notebook for Task 2
```

---

## 📊 Dataset Specifications & Files to Upload

All dataset files follow a **100% JSONL-native pipeline**, eliminating CSV escaping corruptions and preserving multiline text integrity.

| Category | File Name | Description | Format | Record Count |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Master Data** | `task1_conversation_data_balanced.jsonl` | Complete multi-turn dialogue dataset | JSONL | Balanced |
| | `task2_classification_data_balanced.jsonl` | Complete medical claim classification dataset | JSONL | 294 records |
| **Task 1 Splits** | `conversation_training_data.jsonl` | 80% Stratified Training Partition | JSONL | 80% |
| | `conversation_validation_data.jsonl` | 10% Stratified Validation Partition | JSONL | 10% |
| | `conversation_test_data.jsonl` | 10% Stratified Test Partition | JSONL | 10% |
| **Task 2 Splits** | `classification_training_data.jsonl` | 80% Stratified Training Partition | JSONL | 234 records |
| | `classification_validation_data.jsonl` | 10% Stratified Validation Partition | JSONL | 30 records |
| | `classification_test_data.jsonl` | 10% Stratified Test Partition | JSONL | 30 records |
| **Predictions** | `predictions_classification_train.jsonl` | Model generated predictions (Train Sample) | JSONL | 30 records |
| | `predictions_classification_val.jsonl` | Model generated predictions (Val Set) | JSONL | 30 records |
| | `predictions_classification_test.jsonl` | Model generated predictions (Test Set) | JSONL | 30 records |

---

## 🛠️ Environment & System Setup

### Prerequisites & Dependencies

The project is optimized for execution in Python 3.10+ environments (including Google Colab GPUs) leveraging **Keras 3** with the **JAX execution backend**.

Install required packages:

```bash
pip install -q keras-nlp keras>=3.0 jax jaxlib datasets rapidfuzz scikit-learn pandas numpy tqdm
```

### Centralized Configuration (`imports.py`)

All top-level imports and JAX/Keras hardware optimization flags are consolidated inside `imports.py` to ensure GPU memory stability:

```python
import os

# Set JAX Backend & VRAM Headroom Allocations
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

import keras
import keras_nlp
import pandas as pd
import numpy as np
import json
import re
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from tqdm import tqdm
```

---

## ⚙️ Model Architecture & Training Hyperparameters

The model architecture utilizes Parameter-Efficient Fine-Tuning (LoRA) applied to Google's **Gemma 3 1B** base architecture:

```python
# Enable mixed_bfloat16 precision prior to preset instantiation
keras.config.set_dtype_policy("mixed_bfloat16")
keras.utils.set_random_seed(812)

# Load Gemma 3 1B Base Model
model = keras_nlp.models.Gemma3CausalLM.from_preset("gemma3_1b")

# Configure LoRA Adaptation
lora_rank = 6
learning_rate = 5e-5
num_epochs = 3
batch_size = 1
sequence_length = 256

model.backbone.enable_lora(rank=lora_rank)
model.preprocessor.sequence_length = sequence_length

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)
```

---

## 📈 Benchmark Results: Individual Task Performance

### Task 2: Single-Task Claim Policy Classification Results

Fine-tuning Gemma 3 1B via LoRA transformed the base model from an un-tuned text repeater into a highly precise medical claim decision engine:

#### Pre-Training Zero-Shot Baseline
* **Behavior**: Echoed input prompt text, repeated clinical descriptions in loops, or generated unformatted tokens.
* **Classification Accuracy**: **0.00%** (Failed to generate formatted target category tokens).

#### Fine-Tuned Performance Metrics (Post-LoRA)

Evaluating across 80% training (sampled), 10% validation, and 10% test splits:

| Metric | Train Partition (Seen) | Validation Partition | Test Partition (Unseen) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **100.00%** | **93.33%** | **90.00%** |
| **Precision (Weighted)** | **100.00%** | **93.94%** | **89.93%** |
| **Recall (Weighted)** | **100.00%** | **93.33%** | **90.00%** |

#### Detailed Test Set Classification Report

| Policy Decision Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **In-Cover** | 0.91 | **1.00** | 0.95 | 10 |
| **Out-of-Cover** | 0.89 | 0.80 | 0.84 | 10 |
| **Requires Pre-Auth** | 0.90 | 0.90 | 0.90 | 10 |
| **Overall Accuracy** | | | **0.90** | 30 |

* **In-Cover Claims**: Achieved **100% Recall** (0 false negatives), ensuring legitimate covered claims are never incorrectly rejected.
* **Requires Pre-Auth**: Achieved a **0.90 F1-score**, reliably flagging high-cost procedures for prior authorization.
* **Inference Speed**: Batched GPU generation (`batch_size=8`) processed evaluation partitions in under **2.5 minutes per split** (~8 minutes total across all splits).

---

### Task 1: Multi-Turn Conversational Triage Results
*(Detailed individual metrics for the 3-turn conversational intake model will be detailed here prior to multi-task integration).*

---

### Unified Multi-Task Pathway Results
*(Results for the combined multi-task model integrating both Conversational Triage and Claim Classification will be populated here).*

---

## 🔒 Safety, Ethics & Non-Diagnostic Guardrails

Care2ClaimLLM-ZA is designed strictly as an **administrative navigation tool**. To ensure clinical safety and ethical compliance:
1. **No Autonomous Diagnosis**: The system is explicitly constrained from diagnosing medical conditions or prescribing clinical treatments.
2. **De-identified Data**: All datasets consist of synthetic, non-identifiable medical scenarios and scheme rules.
3. **Regulatory Alignment**: Operational recommendations align with South African healthcare frameworks, emphasizing human-in-the-loop scheme verification.

---

## 📜 License

Distributed under the **MIT License**.
