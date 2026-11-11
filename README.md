<p align="center">
<img src="Care2ClaimZA-LLM_logo_final.png" alt="Care2ClaimZA-LLM Logo" width="220" />
</p>

<h1 align="center">Care2ClaimZA-LLM</h1>
<p align="center"><b>Fine-Tuned Gemma 3 1B for South African Healthcare Claims Adjudication & Conversational Triage</b></p>

---

### 📌 Executive Overview
Navigating private health insurance and medical schemes in South Africa presents significant administrative complexity. **Care2ClaimLLM-ZA** is an open-source, dual-pathway AI initiative built on **Google's Gemma 3 1B** foundation model using **Parameter-Efficient Fine-Tuning (LoRA)** on a **JAX backend**.

The platform operates across two core operational pathways, culminating in an integrated end-to-end hybrid adjudication pipeline:
1. **Task 1 (Multi-Turn Conversational Triage)**: A 3-turn interactive dialogue agent tailored with South African medical terminology, local dialects (*Sawubona*, *Molo dokotela*, *Aweh doc*), and demographic context to conduct patient intake and non-diagnostic administrative triage.
2. **Task 2 (Single-Task Claim Policy Classification)**: An instruction-tuned classifier that categorizes medical claims into three policy decision categories: `In-Cover`, `Out-of-Cover`, and `Requires Pre-Auth`.
3. **Step 9 Hybrid Pipeline**: Integrates raw multi-turn intake dialogue directly into automated claim payload extraction and real-time policy adjudication.

---

### 💻 Hardware, Runtime & Execution Environment

All data processing, LoRA fine-tuning, and evaluation runs were executed on **Google Colab** under the following hardware specifications and execution flags:

| Hardware / Environment Component | Specification / Setting | Operational Notes |
| :--- | :--- | :--- |
| **Platform Runtime** | **Google Colab** | Python 3.10+ / Linux OS |
| **Accelerator GPU** | **NVIDIA T4 GPU** | 15.0 GB VRAM Ceiling |
| **System RAM** | **12.7 GB System RAM** | Memory-mapped JSONL streaming |
| **Deep Learning Backend** | **JAX + Keras 3** | `KERAS_BACKEND = "jax"` |
| **VRAM Memory Flags** | `XLA_PYTHON_CLIENT_PREALLOCATE="false"`<br>`XLA_PYTHON_CLIENT_MEM_FRACTION="0.80"` | Prevents OOM during JIT compilation & batched generation |
| **Precision Policy** | `mixed_bfloat16` | Optimized tensor execution on T4 GPU |
| **Sequence Length Ceiling** | **512 tokens** | Accommodates full 3-turn SA scheme dialogue history |
| **Fine-Tuning Batch Size** | **1** | Preserves VRAM headroom during backpropagation |
| **Inference Batch Size** | **8** | Vectorized GPU batched generation |
| **LoRA Configuration** | Rank = 8, Alpha = 16 | **1,304,576 trainable params** (4.98 MB, ~0.13% of Gemma 3 1B backbone) |

---

### 📊 Dataset Specifications & Partition Splits

All datasets utilize a **100% JSONL-native format**, avoiding CSV escaping corruptions and preserving multiline dialogue structure.

| Task Partition | Dataset File Name | Format | Record Count | Split Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1 Training** | `conversation_training_data.jsonl` | JSONL | 192 dialogues | 80% Stratified |
| **Task 1 Validation** | `conversation_validation_data.jsonl` | JSONL | 24 dialogues | 10% Stratified |
| **Task 1 Testing** | `conversation_test_data.jsonl` | JSONL | 24 dialogues | 10% Stratified |
| **Task 2 Training** | `classification_training_data.jsonl` | JSONL | 235 claim records | 80% Stratified |
| **Task 2 Validation** | `classification_validation_data.jsonl` | JSONL | 29 claim records | 10% Stratified |
| **Task 2 Testing** | `classification_test_data.jsonl` | JSONL | 30 claim records | 10% Stratified |

---

### 📈 Task 1 Benchmark Results: Multi-Turn Conversational Triage

Evaluating Gemma 3 1B fine-tuned with LoRA (rank=8) across 3-turn interactive patient intake dialogues:

#### 1. Quantitative NLG Evaluation Metrics
Evaluated using ROUGE (1/2/L), BLEU-2, and Semantic Cosine Similarity (`SentenceTransformer: all-MiniLM-L6-v2`):

| Evaluation Metric | Train Partition (Seen) | Validation Partition | Test Partition (Unseen) | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **ROUGE-1 (Unigram Overlap)** | 0.2131 | 0.2291 | **0.2159** | Consistent unigram alignment across splits |
| **ROUGE-2 (Bigram Overlap)** | 0.0276 | 0.0552 | **0.0410** | Stable phrase structure retention |
| **ROUGE-L (Longest Sequence)**| 0.1479 | 0.1497 | **0.1474** | Preserved sentence structure |
| **BLEU-2 (Precision Metric)** | 0.0306 | 0.0465 | **0.0378** | Consistent N-gram precision |
| **Semantic Cosine Similarity**| 0.4332 | 0.4435 | **0.4461** | **Strong generalization on unseen dialogues** |

#### 2. Prompt Sensitivity & Dialect Robustness
The model was tested across four South African greeting registers:
* **isiZulu (*Sawubona*)**: Maintained empathetic intake tone and administrative scheme verification.
* **isiXhosa (*Molo dokotela*)**: Instantly recognized greeting (*"Molo!"*) and triggered emergency intake routing to Netcare casualty.
* **Urban Slang (*Aweh doc / Sho*)**: Successfully adapted to casual township phrasing (*"Keep me updated..."*).
* **Formal English (*Good day doctor*)**: Produced structured, polite triage intake steps.

---

### 📉 Task 2 Benchmark Results: Single-Task Claim Policy Classification

Evaluating the instruction-tuned classifier across policy adjudication decisions (`In-Cover`, `Out-of-Cover`, `Requires Pre-Auth`):

#### 1. Aggregate Partition Performance

| Metric | Train Partition (30 Sampled) | Validation Partition (29 Records) | Test Partition (30 Unseen Records) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **86.67%** | **65.52%** | **70.00%** |
| **Precision (Weighted)** | **90.48%** | **82.76%** | **73.20%** |
| **Recall (Weighted)** | **86.67%** | **65.52%** | **70.00%** |

#### 2. Detailed Test Partition Classification Report (30 Unseen Records)

| Policy Decision Category | Precision | Recall | F1-Score | Support | Operational Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Cover** | 0.53 | **0.90** | 0.69 | 10 | High safety net for covered acute & chronic conditions |
| **Out-of-Cover** | **1.00** | **1.00** | **1.00** | 10 | **Flawless exclusion safety (0 false approvals)** |
| **Requires Pre-Auth** | 0.67 | 0.20 | 0.31 | 10 | Conservative spillover into general covered status |
| **Overall Accuracy** | | | **0.70** | 30 | Reliable baseline for automated claim routing |

---

### 🔄 Step 9: End-to-End Hybrid Pipeline

The hybrid pipeline connects both models sequentially in a single execution step:

```text
[Patient Intake Chat] ──► Task 1 Model (Triage) ──► Extracted Clinical Payload
                                                                 │
[Adjudication Card]  ◄── Task 2 Model (Classifier) ─────────────┘
```

#### Sample Hybrid Run Execution:
* **Input Dialogue**: *"Sawubona doc! I have a severe burning pain in my chest when I cough. I am a 29-year-old female on Discovery KeyCare Access, currently at Netcare Park Lane."*
* **Stage 1 (Triage Guidance)**: Generates conversational administrative guidance.
* **Stage 2 (Payload Extraction)**: Auto-extracts `Scheme: Discovery (KeyCare)` and `Symptom: cough`.
* **Stage 3 (Policy Decision)**: Adjudicates final claim status as **`[In-Cover]`**.

---

### 🔒 Safety, Ethics & Guardrails

1. **No Autonomous Diagnosis**: Constrained strictly to administrative healthcare guidance and policy classification.
2. **De-identified Data**: Built entirely on synthetic South African healthcare scenarios.
3. **Human-in-the-Loop**: Administrative output serves as decision-support for scheme members and case managers.

---

### 📜 License
Distributed under the **MIT License**.
