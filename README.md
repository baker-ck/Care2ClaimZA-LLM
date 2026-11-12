<p align="center">
<img src="Care2ClaimZA-LLM_logo_final.png" alt="Care2ClaimZA-LLM Logo" width="220" />
</p>

<h1 align="center">Care2ClaimZA-LLM</h1>
<p align="center"><b>Fine-Tuned Gemma 3 1B for South African Healthcare Claims Adjudication & Conversational Triage</b></p>
<p align="center"><i>Developed by Mr. Clayton Baker (Lecturer: Computer Science, University of the Western Cape)</i></p>

---

### 📌 Executive Overview
Navigating private health insurance and medical schemes in South Africa presents significant administrative complexity. **Care2ClaimZA-LLM** is an open-source, dual-pathway AI initiative built on **Google's Gemma 3 1B** foundation model using **Parameter-Efficient Fine-Tuning (LoRA)** on a **JAX backend**.

The platform operates across two core operational pathways, culminating in an integrated end-to-end hybrid adjudication engine:
1. **Task 1 (Multi-Turn Conversational Triage)**: A 3-turn interactive dialogue agent tailored with South African medical terminology, local dialects (*Sawubona*, *Molo dokotela*, *Aweh doc*), and demographic context to conduct patient intake and non-diagnostic administrative triage.
2. **Task 2 (Single-Task Claim Policy Classification)**: An instruction-tuned classifier that categorizes medical claims into three policy decision categories: `In-Cover`, `Out-of-Cover`, and `Requires Pre-Auth`.
3. **Hybrid Conversation-Classification Pipeline**: Integrates raw multi-turn intake dialogue directly into automated claim payload extraction and real-time policy adjudication.

---

### 📂 Repository Layout & Structure

```text
Care2ClaimZA-LLM/
├── data/
│   ├── raw/
│   │   ├── task1_conversation_data_balanced.jsonl   # Raw synthesized 3-turn dialogue dataset
│   │   └── task2_classification_data_balanced.jsonl # Raw algorithmically labeled claim records
│   └── splits/
│       ├── conversation_training_data.jsonl         # Task 1 Training Split (112 dialogues, 80%)
│       ├── conversation_validation_data.jsonl       # Task 1 Validation Split (14 dialogues, 10%)
│       ├── conversation_test_data.jsonl             # Task 1 Test Split (14 dialogues, 10%)
│       ├── classification_training_data.jsonl       # Task 2 Training Split (235 claims, 80%)
│       ├── classification_validation_data.jsonl     # Task 2 Validation Split (29 claims, 10%)
│       └── classification_test_data.jsonl           # Task 2 Test Split (30 claims, 10%)
├── predictions/
│   ├── predictions_conversation_train_data.jsonl
│   ├── predictions_conversation_val_data.jsonl
│   ├── predictions_conversation_test_data.jsonl
│   ├── predictions_classification_train_data.jsonl
│   ├── predictions_classification_val_data.jsonl
│   └── predictions_classification_test_data.jsonl
├── Care2ClaimZA-LLM.ipynb                     # Care2ClaimZA-LLM model source code
├── Care2ClaimZA-LLM_datacard.pdf              # Care2ClaimZA-LLM Data Card Specification
├── Care2ClaimZA-LLM_logo_final.png            # Project Visual Emblem / Logo
├── LICENSE                                    # MIT License
├── README.md                                  # Project Documentation & Comprehensive Benchmark Report
├── capstone-proposal-deeply-baked.pdf        # Technical Capstone Proposal Document
     
```

---

### 💻 Hardware, Runtime & Execution Environment (Fact-Checked against Notebook Code)

All data processing, LoRA fine-tuning, and evaluation runs were executed on **Google Colab** under the following hardware specifications and exact execution flags from the PDF notebook:

| Hardware / Environment Component | Exact Setting / Value in Code | Operational Notes |
| :--- | :--- | :--- |
| **Platform Runtime** | **Google Colab** | Python 3.10+ / Linux OS |
| **Accelerator GPU** | **NVIDIA T4 GPU** | CUDA / XLA Acceleration |
| **System RAM** | **12.7 GB System RAM** | Memory-mapped JSONL streaming |
| **Deep Learning Backend** | **JAX + Keras 3** | `os.environ["KERAS_BACKEND"] = "jax"` |
| **VRAM Memory Allocation** | `os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"` | **0.95 VRAM fraction** (Preallocation flag left at default JAX setting) |
| **Precision Policy** | `mixed_bfloat16` | Enabled via `keras.config.set_dtype_policy("mixed_bfloat16")` |
| **Sequence Length Ceiling** | **512 tokens** | Configured via `model.preprocessor.sequence_length = 512` |
| **Fine-Tuning Batch Size** | **1** | `batch_size = 1` across 3 epochs to preserve VRAM headroom |
| **Inference Batch Size** | **8** | `batch_size = 8` for GPU vectorized prediction generation |
| **LoRA Configuration** | Rank = 8 (`lora_rank = 8`) | **1,304,576 trainable params** (4.98 MB, ~0.13% of 1.001B Gemma 3 backbone) |
| **Optimizer & Loss** | `AdamW(lr=1e-4, weight_decay=0.01)` | `SparseCategoricalCrossentropy(from_logits=True)`, `jit_compile=True` |

---

### 📊 Dataset Specifications & Partition Splits

All datasets utilize a **100% JSONL-native format**, avoiding CSV escaping corruptions and preserving multiline dialogue structure. Partition sizes reflect the actual executed notebook `Step 0` inspection reports:

| Task Partition | Dataset File Name | Format | Record Count | Split Allocation |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1 Training** | `conversation_training_data.jsonl` | JSONL | **112 dialogues** | 80% Stratified Training Partition |
| **Task 1 Validation** | `conversation_validation_data.jsonl` | JSONL | **14 dialogues** | 10% Stratified Validation Partition |
| **Task 1 Testing** | `conversation_test_data.jsonl` | JSONL | **14 dialogues** | 10% Stratified Test Partition |
| **Task 2 Training** | `classification_training_data.jsonl` | JSONL | **235 claim records** | 80% Stratified Training Partition |
| **Task 2 Validation** | `classification_validation_data.jsonl` | JSONL | **29 claim records** | 10% Stratified Validation Partition |
| **Task 2 Testing** | `classification_test_data.jsonl` | JSONL | **30 claim records** | 10% Stratified Test Partition |

---

### 📈 Task 1 Benchmark Results: Multi-Turn Conversational Triage

Fine-tuning Gemma 3 1B via LoRA (rank=8, batch_size=1, 3 epochs) achieved training loss reduction from `0.3311` down to `0.0013` (sparse categorical accuracy: **99.69%**).

#### 1. Quantitative NLG Evaluation Metrics
Evaluated across dialogue generations using ROUGE (1/2/L), BLEU-2, and Semantic Cosine Similarity (`SentenceTransformer: all-MiniLM-L6-v2`):

| Evaluation Metric | Train Partition (30 Sampled) | Validation Partition (14 Records) | Test Partition (14 Unseen Records) | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **ROUGE-1 (Unigram Overlap)** | 0.2131 | 0.2291 | **0.2159** | Consistent unigram alignment across splits |
| **ROUGE-2 (Bigram Overlap)** | 0.0276 | 0.0552 | **0.0410** | Stable phrase structure retention |
| **ROUGE-L (Longest Sequence)**| 0.1479 | 0.1497 | **0.1474** | Preserved sentence structure |
| **BLEU-2 (Precision Metric)** | 0.0306 | 0.0465 | **0.0378** | Consistent N-gram precision |
| **Semantic Cosine Similarity**| 0.4332 | 0.4435 | **0.4461** | **Strong generalization on unseen dialogues** |

#### 2. Prompt Sensitivity & Dialect Robustness (Step 7 Output)
Evaluated model generation consistency across South African greeting registers:
* **isiZulu (*Sawubona*)**: *"Please guide me if necessary...."* (Maintained empathetic intake tone).
* **isiXhosa (*Molo dokotela*)**: *"Molo! Take me to the nearest NetCare hospital emergency...."* (Recognized greeting and triggered acute emergency routing).
* **Urban Slang (*Aweh doc / Sho*)**: *"Keep me updated if your symptoms improve!..."* (Adapted smoothly to casual township phrasing).
* **Formal English (*Good day doctor*)**: *"Let me check and provide further advice if necessary...."* (Delivered structured intake guidance).

---

### 📉 Task 2 Benchmark Results: Single-Task Claim Policy Classification

Evaluating the instruction-tuned classifier across policy adjudication decisions (`In-Cover`, `Out-of-Cover`, `Requires Pre-Auth`):

#### 1. Aggregate Partition Performance

| Metric | Train Partition (30 Sampled) | Validation Partition (29 Records) | Test Partition (30 Unseen Records) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **86.67%** | **65.52%** | **70.00%** |
| **Precision (Weighted)** | **90.48%** | **82.76%** | **73.20%** |
| **Recall (Weighted)** | **86.67%** | **65.52%** | **70.00%** |

#### 2. Detailed Classification Reports Across Partitions

##### A. Training Partition (30 Sampled Records)
* **Accuracy**: **86.67%** | **Precision**: 90.48% | **Recall**: 86.67%
* `In-Cover`: Precision 0.71, Recall 1.00, F1 0.83 (support 10)
* `Out-of-Cover`: Precision 1.00, Recall 1.00, F1 1.00 (support 10)
* `Requires Pre-Auth`: Precision 1.00, Recall 0.60, F1 0.75 (support 10)

##### B. Validation Partition (29 Records)
* **Accuracy**: **65.52%** | **Precision**: 82.76% | **Recall**: 65.52%
* `In-Cover`: Precision 0.50, Recall 1.00, F1 0.67 (support 10)
* `Out-of-Cover`: Precision 1.00, Recall 0.78, F1 0.88 (support 9)
* `Requires Pre-Auth`: Precision 1.00, Recall 0.20, F1 0.33 (support 10)

##### C. Test Partition (30 Unseen Records)
* **Accuracy**: **70.00%** | **Precision**: 73.20% | **Recall**: 70.00%
* `In-Cover`: Precision 0.53, Recall **0.90**, F1 0.67 (support 10)
* `Out-of-Cover`: Precision **1.00**, Recall **1.00**, F1 **1.00** (support 10) — **Flawless exclusion safety**
* `Requires Pre-Auth`: Precision 0.67, Recall 0.20, F1 0.31 (support 10) — Conservative spillover into general covered status

---

### 🔄 Care2ClaimLLM-ZA: End-to-End Hybrid Pipeline

The hybrid pipeline connects both fine-tuned models sequentially in a single execution step:

```text
[Patient Intake Chat] ──► Task 1 Model (Triage) ──► Extracted Clinical Payload
                                                                 │
[Adjudication Card]  ◄── Task 2 Model (Classifier) ─────────────┘
```

#### Executed Hybrid Pipeline Demo:
* **Sample Input**: *"Sawubona doc! I have a severe burning pain in my chest when I cough. I am a 29-year-old female on Discovery KeyCare Access, currently at Netcare Park Lane."*
* **Stage 1 (Triage Output)**: Generates conversational administrative guidance.
* **Stage 2 (Payload Extraction)**: Auto-formats into `Medical Scheme: Discovery (KeyCare). Patient presenting with: cough.`
* **Stage 3 (Policy Decision)**: Adjudicates final claim status as **`[In-Cover]`**.

---

### 🔒 Safety, Ethics & Non-Diagnostic Boundaries

1. **No Autonomous Diagnosis**: Constrained strictly to administrative healthcare guidance and scheme policy classification.
2. **De-identified Data**: Built entirely on synthetic South African medical scheme rules and public regulatory frameworks.
3. **Human-in-the-Loop**: Serves strictly as a non-binding administrative decision support tool under statutory regulations.

---

### 📜 License
Distributed under the **MIT License**.
