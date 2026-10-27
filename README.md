# CliniClaimZA-LLM: Fine-Tuned Gemma 3 1B for South African Healthcare Claims Adjudication & Triage

**CliniClaimZA-LLM** is a domain-adapted, parameter-efficient fine-tuned large language model based on **Google's Gemma 3 1B**. It is engineered to assist with healthcare administrative workflows within the South African medical scheme regulatory context—balancing structured scheme policy adjudication with empathetic conversational patient intake and triage.

---

## 📌 Project Overview

Navigating South African private health insurance (medical aid) involves strict statutory regulations established by the **Council for Medical Schemes (CMS)**, including **Prescribed Minimum Benefits (PMBs)**, **Chronic Disease Lists (CDLs)**, and **Designated Service Provider (DSP)** network rules.

**CliniClaimZA-LLM** addresses a dual operational workload:
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

## 📊 Performance & Evaluation Results

### 1. Task 1: Scheme Policy Adjudication (Classification)
Task 1 evaluates exact-match multi-class adjudication accuracy across 3 balanced target classes (`In-Cover`, `Out-of-Cover`, `Requires Pre-Auth`).

| Dataset Split | Samples | Exact Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | 234 | 24.79% | 61.81% | 24.79% | 17.50% |
| **Validation Split** | 30 | 26.67% | 12.12% | 26.67% | 16.67% |
| **Test Split (Unseen)** | 30 | **20.00%** | **11.11%** | **20.00%** | **14.29%** |

> **Key Diagnostic Finding:** The close alignment between Train (24.79%), Val (26.67%), and Test (20.00%) confirms the model did **not overfit**. However, unmasked causal loss and long dialogue tokens from Task 2 caused **underfitting and majority class collapse** onto `Out-of-Cover` (60% recall on test set, with 0% recall on `In-Cover` and `Requires Pre-Auth`).

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
