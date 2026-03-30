# Relation Extraction: BiLSTM with RPE vs BERT + SVM

A comparative study of two approaches to **Relation Extraction (RE)** on the SemEval 2010 Task 8 dataset — a BiLSTM model with custom Relative Positional Encoding (RPE) and a hybrid BERT + SVM pipeline.

> **Group Project** — Joshwin Sundarraj, Habiba Ahmad, Dion Recai, Manan Mal

---

## Overview

Relation Extraction is the task of identifying semantic relationships between entity pairs in text — a core component of knowledge base construction, biomedical text mining, and automated Q&A systems. This project implements and compares two contrasting paradigms:

| Aspect | BiLSTM + RPE | BERT + SVM |
|--------|-------------|------------|
| Feature learning | End-to-end (learned) | Pretrained embeddings + classical classifier |
| Positional awareness | Custom relative positional encoding | Entity tags preserved in input |
| Sequence modelling | Bidirectional LSTM | None (mean-pooled embeddings) |
| Hyperparameter tuning | Bayesian Optimisation | GridSearchCV (5-fold CV) |
| Test Accuracy | **75.27%** | 64.26% |
| Macro F1 | **70.42%** | 59.0% |

---

## Dataset

**SemEval 2010 Task 8** — sentence-level relation extraction benchmark.

- 19 directional relation types (including an *Other* category)
- Human-annotated entity pairs with positional tags (`<e1>`, `<e2>`)
- Pre-defined train/test split

Entity tags are **retained** in the input for both approaches, consistent with findings by Baldini Soares et al. (2019) and Tao et al. (2019) that syntactic indicators improve relation learning.

---

## Approach 1: BiLSTM with Relative Positional Encoding

### Preprocessing
- Entity tags (`<e1>`, `</e1>`, `<e2>`, `</e2>`) are space-separated for clean tokenisation
- Punctuation is filtered (except `<`, `>`, `\`) to reduce OOV tokens
- Sequences are padded to uniform length
- Relation labels are one-hot encoded to vectors of size 19

### Relative Positional Encoding
Two position encodings are computed per sample — one relative to each tagged entity. For example:

```
"The <e1>bottle</e1> was filled with <e2>coloured water</e2> and placed on the table."

RPE w.r.t. <e1>bottle</e1>  → [-2, -1, 0, +1, ..., +13]
RPE w.r.t. <e2>water</e2>   → [-8, ..., -1, 0, 0, +1, ..., +6]
```

Tagged entities are masked with `0`. Both encodings are padded to match token sequence length.

### Architecture
- **Embedding layer:** GloVe pretrained embeddings (dim = 300, tuned via Bayesian Optimisation)
- **Position embeddings:** learned from RPE inputs
- **Concatenation:** word + position₁ + position₂ embeddings → 1D Spatial Dropout
- **BiLSTM layer:** single layer (research by Peters et al. (2018) shows lower layers capture significant syntactic information)
- **Output:** Dense layer with Softmax over 19 relation classes

### Optimal Hyperparameters (Bayesian Optimisation)

| Hyperparameter | Value |
|---------------|-------|
| Embedding dim | 300 |
| LSTM units | 96 |
| Dropout rate | 0.8 |
| Learning rate | 0.001 |
| BiLSTM dropout | 0.4 |
| Kernel regulariser (L2) | 0.0001 |

Training: 30 epochs, batch size 10, full training set with optimal hyperparameters.

---

## Approach 2: BERT + SVM

### BERT Embeddings
- **Model:** `bert-base-uncased` (768-dimensional output embeddings)
- Base model chosen over BERT-large due to comparable performance at significantly lower compute cost (Shi and Lin, 2019)
- **Sentence representation:** mean of all token embeddings (outperforms [CLS] token pooling in our experiments; consistent with Reimers and Gurevych, 2019)
- PCA dimensionality reduction was tested but produced significantly worse results and was dropped

### SVM Classifier
- Kernel: **linear** (recommended for text classification, Hsu et al., 2009)
- Hyperparameter tuning via **GridSearchCV** (5-fold cross-validation) over `C` and `class_weight`
- Chosen for robustness to overfitting and computational efficiency

---

## Results

### Overall Performance

| Model | Test Accuracy | Macro F1 |
|-------|-------------|----------|
| BiLSTM + RPE | **75.27%** | **70.42%** |
| BERT + SVM | 64.26% | 59.0% |

### Per-Relation Results

| Relation | BiLSTM P | BiLSTM R | BiLSTM F1 | BERT+SVM P | BERT+SVM R | BERT+SVM F1 |
|----------|----------|----------|-----------|------------|------------|-------------|
| Cause-Effect(e1,e2) | 0.89 | 0.89 | 0.89 | 0.85 | 0.63 | 0.72 |
| Cause-Effect(e2,e1) | 0.89 | 0.87 | 0.88 | 0.63 | 0.88 | 0.74 |
| Component-Whole(e1,e2) | 0.80 | 0.78 | 0.79 | 0.68 | 0.70 | 0.69 |
| Component-Whole(e2,e1) | 0.62 | 0.64 | 0.63 | 0.51 | 0.65 | 0.57 |
| Content-Container(e1,e2) | 0.72 | 0.95 | 0.82 | 0.79 | 0.83 | 0.81 |
| Content-Container(e2,e1) | 0.78 | 0.74 | 0.76 | 0.56 | 0.56 | 0.56 |
| Entity-Destination(e1,e2) | 0.85 | 0.90 | 0.87 | 0.73 | 0.85 | 0.78 |
| Entity-Destination(e2,e1) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Entity-Origin(e1,e2) | 0.79 | 0.87 | 0.83 | 0.59 | 0.78 | 0.67 |
| Entity-Origin(e2,e1) | 0.80 | 0.87 | 0.84 | 0.65 | 0.77 | 0.71 |
| Instrument-Agency(e1,e2) | 0.52 | 0.55 | 0.53 | 0.33 | 0.27 | 0.30 |
| Instrument-Agency(e2,e1) | 0.69 | 0.73 | 0.71 | 0.69 | 0.48 | 0.56 |
| Member-Collection(e1,e2) | 0.43 | 0.66 | 0.52 | 0.50 | 0.41 | 0.45 |
| Member-Collection(e2,e1) | 0.80 | 0.91 | 0.85 | 0.63 | 0.88 | 0.73 |
| Message-Topic(e1,e2) | 0.76 | 0.90 | 0.82 | 0.71 | 0.80 | 0.75 |
| Message-Topic(e2,e1) | 0.78 | 0.75 | 0.76 | 0.78 | 0.57 | 0.66 |
| Product-Producer(e1,e2) | 0.76 | 0.78 | 0.77 | 0.60 | 0.58 | 0.59 |
| Product-Producer(e2,e1) | 0.68 | 0.69 | 0.69 | 0.55 | 0.53 | 0.54 |
| Other | 0.54 | 0.33 | 0.41 | 0.49 | 0.22 | 0.30 |
| **Macro Avg** | **0.69** | **0.73** | **0.70** | **0.59** | **0.60** | **0.59** |
| **Weighted Avg** | **0.74** | **0.75** | **0.74** | **0.63** | **0.64** | **0.62** |

### Key Observations

- **BiLSTM + RPE outperforms BERT + SVM on all overall metrics**, reinforcing the advantage of end-to-end sequential feature learning over a pipeline approach.
- **Entity-Destination(e2,e1) scored 0% for both models** — this relation had only one training sample, making it impossible to learn. Potential fixes: SMOTE oversampling, converting reverse-relation sentences, or removing the class.
- **BERT + SVM showed competitive precision on some individual relations** (e.g. Cause-Effect, Content-Container) where BERT's contextual embeddings captured sufficient signal.
- Both models underperform the SOTA BiLSTM+Attention model (Zhou et al., 2016) which achieves 85% accuracy — attributable to its attention mechanism, which we did not implement.
- The BERT + SVM model's precision (59%) falls below specialist SVM systems (63–84%) that use syntactic features like POS tags, WordNet, and NomLex — showing that BERT embeddings alone do not substitute for feature engineering in SVMs.

---

## Future Work

- Fine-tuning BERT end-to-end for relation classification rather than using it as a frozen feature extractor
- Adding an attention mechanism to the BiLSTM (as in Zhou et al., 2016) to target the performance gap with SOTA
- Addressing class imbalance via SMOTE or data augmentation for rare relation types
- Exploring graph-based approaches (e.g. GCN over dependency parse trees) for richer structural representations

---

## References

- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL*.
- Zhou et al. (2016). Attention-based BiLSTM for Relation Classification. *ACL*.
- Baldini Soares et al. (2019). Matching the Blanks: Distributional Similarity for Relation Learning. *ACL*.
- Pennington et al. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
- Reimers and Gurevych (2019). Sentence-BERT. *EMNLP*.
- Hendrickx et al. (2010). SemEval-2010 Task 8. *SemEval Workshop*.
- Rink and Harabagiu (2010). UTD: Classifying Semantic Relations. *SemEval Workshop*.
