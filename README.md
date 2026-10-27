<p align="center">
  <img src="Care2ClaimZA-LLM_logo_final.png" alt="Care2ClaimZA-LLM Logo" width="220" />
</p>

<h1 align="center">Care2ClaimZA-LLM</h1>
<p align="center"><b>Fine-Tuned Gemma 3 1B for South African Healthcare Claims Adjudication & Triage</b></p>

---

## 📌 Project Overview

**Care2ClaimZA-LLM** is a domain-adapted, parameter-efficient fine-tuned large language model based on **Google's Gemma 3 1B**. It is engineered to assist with healthcare administrative workflows within the South African medical scheme regulatory context—balancing structured scheme policy adjudication with empathetic conversational patient intake and triage.

Navigating South African private health insurance (medical aid) involves strict statutory regulations established by the **Council for Medical Schemes (CMS)**, including **Prescribed Minimum Benefits (PMBs)**, **Chronic Disease Lists (CDLs)**, and **Designated Service Provider (DSP)** network rules.

**Care2ClaimZA-LLM** addresses a dual operational workload:
1. **Task 1: Scheme Policy Adjudication (Closed-Set Classification)**  
   Classifies incoming medical claims based on specialty and clinical descriptions into three policy decision categories:
   - `In-Cover` (Routine covered benefit / statutory PMB)
   - `Out-of-Cover` (Excluded cosmetic / out-of-scheme treatment)
   - `Requires Pre-Auth` (Specialized procedure requiring scheme authorization)
2. **Task 2: Conversational Intake & Triage (Open-Ended Dialogue Generation)**  
   Provides plain-language, empathetic guidance to medical scheme members regarding chronic registration, co-payments, emergency PMB coverage, and referral pathways.

---

## ⚙️ Model Architecture & Hyperparameters

The model was fine-tuned on an **NVIDIA T4 GPU (15 GB VRAM)** using **LoRA (Low-Rank Adaptation)** via `KerasNLP`:

| Parameter | Value | Description / Rationale |
| :--- | :---: | :--- |
| **Base Model** | `gemma_3_1b_en` | Lightweight 1B parameter causal language model |
| **Fine-Tuning Method** | LoRA | Parameter-efficient adapter tuning on backbone weights |
| **LoRA Rank (`lora_rank`)** | `6` | Balances conversational capacity with T4 memory constraints |
| **Sequence Length** | `512` (Train) / `600` (Eval) | Prevents GPU OOM while accommodating long clinical descriptions |
| **Learning Rate** | `5e-5` | Stable convergence using AdamW optimizer |
| **Batch Size** | `2` | Fits comfortably within T4 VRAM |
| **Epochs** | `4` | Ensures dialogue convergence while preventing overfitting |
| **Optimizer** | `AdamW` | Weight decay = 0.01 (excluded for LN/bias parameters) |

---

## 📁 Repository Directory Structure

To ensure complete reproducibility across Google Colab environments, the repository is structured as follows:

```text
data/
├── raw/
│   ├── task1_classification_data.csv      # Original policy classification dataset
│   └── task2_conversational_data.json     # Original conversational intake & triage dataset
└── processed/
    └── splits/
        ├── train_data.jsonl               # Training set (seen during LoRA fitting)
        ├── val_data.jsonl                 # Validation set (used for epoch checkpointing)
        ├── test_classification_data.csv   # Unseen Task 1 test split (30 items)
        └── test_conversational_data.csv   # Unseen Task 2 test split (30 items)
```

---

## 📊 Performance & Evaluation Results

### 1. Task 1: Scheme Policy Adjudication (Classification)
Task 1 evaluates exact-match multi-class adjudication accuracy across 3 balanced target classes (`In-Cover`, `Out-of-Cover`, `Requires Pre-Auth`).

| Dataset Split | Samples | Exact Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 234 | 24.79% | 61.81% | 24.79% | 17.50% |
| **Validation Split** | 30 | 26.67% | 12.12% | 26.67% | 16.67% |
| **Test Split (Unseen)** | 30 | **20.00%** | **11.11%** | **20.00%** | **14.29%** |

> **Key Diagnostic Finding:** The close alignment between Train (24.79%), Val (26.67%), and Test (20.00%) confirms the model did **not overfit**. However, unmasked causal loss and long dialogue tokens from Task 2 caused **underfitting and majority class collapse** onto `Out-of-Cover` (60% recall on test set).

### 2. Task 2: Conversational Intake & Triage (Generation)
Task 2 evaluates open-ended generation against reference South African scheme triage guidance across automated lexical and semantic metrics.

| Evaluation Metric | Score | Metric Description |
| :--- | :---: | :--- |
| **BLEU-2** | **0.5448** | N-gram precision on paired domain terms |
| **ROUGE-1** | **0.7021** | Unigram recall of SA healthcare terminology |
| **ROUGE-2** | **0.6222** | Bigram overlap on statutory phrases (*Pre-Auth*, *CDL List*) |
| **ROUGE-L** | **0.6856** | Longest common sequence structural alignment |
| **BERTScore (Cosine Similarity)** | **0.7043** | Contextual semantic embedding equivalence |

---

## 🚀 Quickstart & Usage

### Installation

```bash
pip install -q keras-nlp keras tensorflow nltk rouge-score sentence-transformers scikit-learn
```

### Fine-Tuning Script Snippet

```python
import keras
import keras_nlp

# Load base Gemma 3 1B model and preprocessor
model = keras_nlp.models.Gemma3CausalLM.from_preset("gemma_3_1b_en")
model.preprocessor.sequence_length = 512

# Enable LoRA
model.backbone.enable_lora(rank=6)

# Compile model
optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
optimizer.exclude_from_weight_decay(var_names=["reciprocal_scale", "layer_norm", "bias"])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Train model
model.fit(train_dataset, validation_data=val_dataset, epochs=4, batch_size=2)
```

### Inference & Prompt Formatting

```python
def ask_cliniclaim(prompt_text: str, max_length: int = 600) -> str:
    formatted_prompt = (
        f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
    )
    raw_output = model.generate(formatted_prompt, max_length=max_length)
    completion = raw_output.replace(formatted_prompt, "").replace("<end_of_turn>", "").strip()
    return completion

# Example Policy Query
print(ask_cliniclaim(
    "Classify the following medical claim into one of the policy decision categories: "
    "'In-Cover', 'Out-of-Cover', or 'Requires Pre-Auth'.\n\n"
    "Medical Specialty: Cardiology\n"
    "Clinical Description: Diagnostic coronary catheterization and evaluation."
))
```

---

## 🛠️ Roadmap & Future Model Improvements

To resolve the classification underfitting observed in Task 1 while preserving strong Task 2 conversational performance, the following architecture updates are planned:

1. **Target-Only Loss Masking (`train_on_responses_only`)**:  
   Mask prompt tokens out of cross-entropy loss computation so 100% of gradient updates target response/label tokens.
2. **Multi-Task Rebalancing & Task Prefixes**:  
   Prepend explicit task identifiers (e.g., `[TASK: ADJUDICATION]`) and oversample Task 1 samples to balance loss contribution against long dialogue turns.
3. **Constrained Decoding & Logit Biasing**:  
   Enforce logit masking at inference time to restrict candidate tokens strictly to valid label strings (`['In-Cover', 'Out-of-Cover', 'Requires Pre-Auth']`) with `temperature=0.0`.
4. **Expanded LoRA Capacity**:  
   Increase LoRA rank to `16` or `32` and expand adapter target modules across all linear attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
5. **Structured JSON Output Constraints**:  
   Format Task 1 targets as JSON objects (e.g., `{"decision": "Requires Pre-Auth"}`) to eliminate preamble conversational text and enable deterministic parsing.

---

## 📜 License

This repository is distributed under the **MIT License**.
