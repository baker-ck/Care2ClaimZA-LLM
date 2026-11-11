<p align="center">
<img src="Care2ClaimZA-LLM_logo_final.png" alt="Care2ClaimZA-LLM Logo" width="220" />
</p>

<h1 align="center">Care2ClaimZA-LLM</h1>
<p align="center"><b>Fine-Tuned Gemma 3 1B for South African Healthcare Claims Adjudication & Triage</b></p>

---

#### 📌 Executive Overview
Navigating private health insurance and medical schemes in South Africa presents significant administrative complexity. **Care2ClaimLLM-ZA** is an open-source, dual-pathway AI initiative designed to translate everyday patient descriptions into administrative guidance and categorize medical claims against scheme policy rules.

Built on **Google's Gemma 3 1B** foundation model using **Parameter-Efficient Fine-Tuning (LoRA)** on a **JAX backend**, the system operates across two core operational pathways:
1. **Task 1 (Multi-Turn Conversational Triage)**: A 3-turn interactive dialogue agent tailored with South African medical terminology, local slang, and demographic context to conduct patient intake and non-diagnostic administrative triage.
2. **Task 2 (Single-Task Claim Policy Classification)**: An automated instruction-tuned classifier that categorizes clinical claim descriptions into three policy decision categories: `In-Cover`, `Out-of-Cover`, and `Requires Pre-Auth`.

---

#### 📂 Recommended GitHub Repository Structure
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

#### 📊 Dataset Specifications & Files to Upload
All dataset files follow a **100% JSONL-native pipeline**, eliminating CSV escaping corruptions and preserving multiline text integrity.

| Category | File Name | Description | Format | Record Count |
| :--- | :--- | :--- | :--- | :--- |
| **Raw Master Data** | `task1_conversation_data_balanced.jsonl` | Complete multi-turn dialogue dataset | JSONL | Balanced (240) |
| | `task2_classification_data_balanced.jsonl` | Complete medical claim classification dataset | JSONL | 294 records |
| **Task 1 Splits** | `conversation_training_data.jsonl` | 80% Stratified Training Partition | JSONL | 192 records |
| | `conversation_validation_data.jsonl` | 10% Stratified Validation Partition | JSONL | 24 records |
| | `conversation_test_data.jsonl` | 10% Stratified Test Partition | JSONL | 24 records |
| **Task 2 Splits** | `classification_training_data.jsonl` | 80% Stratified Training Partition | JSONL | 235 records |
| | `classification_validation_data.jsonl` | 10% Stratified Validation Partition | JSONL | 29 records |
| | `classification_test_data.jsonl` | 10% Stratified Test Partition | JSONL | 30 records |
| **Predictions** | `predictions_classification_train.jsonl` | Model generated predictions (Train Sample) | JSONL | 30 records |
| | `predictions_classification_val.jsonl` | Model generated predictions (Val Set) | JSONL | 29 records |
| | `predictions_classification_test.jsonl` | Model generated predictions (Test Set) | JSONL | 30 records |

---

#### 🛠️ Environment & System Setup

##### 💻 Hardware & Runtime Specifications (Google Colab)
All data processing, LoRA fine-tuning, and benchmark evaluations were executed on **Google Colab** under the following hardware environment and resource ceilings:

| Resource Component | Ceiling / Specification | Operational Details |
| :--- | :--- | :--- |
| **Platform Environment** | **Google Colab** | Python 3.10+ / Linux Environment |
| **Accelerator GPU** | **NVIDIA T4 GPU** | CUDA / XLA Tensor Core Acceleration |
| **GPU VRAM Ceiling** | **15.0 GB GPU RAM** | Peak VRAM consumption: ~6.5 GB (`mixed_bfloat16`) |
| **System RAM Ceiling** | **12.7 GB System RAM** | Memory-mapped JSONL streaming |
| **Execution Engine** | **JAX Backend + Keras 3** | `XLA_PYTHON_CLIENT_MEM_FRACTION = 0.75` |

##### Prerequisites & Dependencies
Install required packages:
```bash
pip install -q keras-nlp keras>=3.0 jax jaxlib datasets rapidfuzz scikit-learn pandas numpy tqdm
```

##### Centralized Configuration (`imports.py`)
All top-level imports and JAX/Keras hardware optimization flags are consolidated inside `imports.py` to ensure GPU memory stability on the T4 GPU:
```python
import os

# Set JAX Backend & VRAM Headroom Allocations
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

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

#### ⚙️ Model Architecture & Training Hyperparameters
The model architecture utilizes Parameter-Efficient Fine-Tuning (LoRA) applied to Google's **Gemma 3 1B** base architecture:

```python
# Enable mixed_bfloat16 precision prior to preset instantiation
keras.config.set_dtype_policy("mixed_bfloat16")
keras.utils.set_random_seed(812)

# Load Gemma 3 1B Base Model
model = keras_nlp.models.Gemma3CausalLM.from_preset("gemma3_1b")

# Configure LoRA Adaptation & Preprocessor
lora_rank = 8
learning_rate = 1e-4
num_epochs = 3
batch_size = 1  # Optimized for T4 GPU memory allocation stability
sequence_length = 512  # Accommodates full SA scheme context without truncation

model.backbone.enable_lora(rank=lora_rank)
model.preprocessor.sequence_length = sequence_length

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    jit_compile=True
)
```

---

#### 📈 Benchmark Results: Performance Overview

##### Execution & Runtime Benchmarks on NVIDIA T4 GPU
| Operational Phase | Total Items | Batch Size | Epochs / Runtime | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1 Pre-processing & Split** | 240 dialogues | N/A | ~15 seconds | < 1.0 GB |
| **Task 1 LoRA Fine-Tuning** | 192 dialogues | 2 | 3 Epochs (~8 mins) | ~6.2 GB |
| **Task 2 Pre-processing & Split** | 294 claims | N/A | ~12 seconds | < 1.0 GB |
| **Task 2 LoRA Fine-Tuning** | 235 claims | 1 | 3 Epochs (~9 mins) | ~6.5 GB |
| **Task 2 Test Inference Evaluation**| 30 test claims | 8 (Batched) | ~2.5 mins | ~5.8 GB |

---

##### Task 2: Single-Task Claim Policy Classification Results
Fine-tuning Gemma 3 1B via LoRA transformed the base model into an instruction-tuned decision engine capable of adjudicating complex South African medical scheme rules:

###### Pre-Training Zero-Shot Baseline
* **Behavior**: Echoed input prompt text, repeated clinical descriptions in loops, or generated unformatted tokens.
* **Classification Accuracy**: **0.00%** (Failed to generate formatted target category tokens).

###### Fine-Tuned Performance Metrics (Post-LoRA)
Evaluating across 80% training (sampled 30 records), 10% validation (29 records), and 10% test splits (30 records):

| Metric | Train Partition (Seen Sample) | Validation Partition | Test Partition (Unseen) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **86.67%** | **72.41%** | **73.33%** |
| **Precision (Weighted)** | **90.48%** | **84.67%** | **77.08%** |
| **Recall (Weighted)** | **86.67%** | **72.41%** | **73.33%** |

###### Detailed Test Set Classification Report
| Policy Decision Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **In-Cover** | 0.56 | **0.90** | 0.69 | 10 |
| **Out-of-Cover** | **1.00** | **1.00** | **1.00** | 10 |
| **Requires Pre-Auth** | 0.75 | 0.30 | 0.43 | 10 |
| **Overall Accuracy** | | | **0.73** | 30 |

* **Out-of-Cover Claims**: Achieved **100% Precision, 100% Recall, and 1.00 F1-score**, flawlessly detecting uninsured/cash-only claims and excluded cosmetic/elective procedures.
* **In-Cover Claims**: Achieved **90% Recall** on unseen test claims (and **100%** on train/val), providing a strong safety net for covered primary care and chronic conditions.
* **Inference Speed**: Batched GPU generation (`batch_size=8`) processed evaluation partitions on the NVIDIA T4 GPU in under **2.5 minutes per split** (~8 minutes total across all splits).

---

##### Task 1: Multi-Turn Conversational Triage Results
*(Detailed individual metrics for the 3-turn conversational intake model will be populated during the Task 1 parallel training and evaluation loop).*

---

##### Unified Multi-Task Pathway Results
*(Results for the combined multi-task model integrating both Conversational Triage and Claim Classification will be populated upon final pipeline assembly).*

---

#### 🔒 Safety, Ethics & Non-Diagnostic Guardrails
Care2ClaimLLM-ZA is designed strictly as an **administrative navigation tool**. To ensure clinical safety and ethical compliance:
1. **No Autonomous Diagnosis**: The system is explicitly constrained from diagnosing medical conditions or prescribing clinical treatments.
2. **De-identified Data**: All datasets consist of synthetic, non-identifiable medical scenarios and scheme rules.
3. **Regulatory Alignment**: Operational recommendations align with South African healthcare frameworks, emphasizing human-in-the-loop scheme verification.

---

#### 📜 License
Distributed under the **MIT License**.
