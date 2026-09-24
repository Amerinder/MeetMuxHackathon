# English → Bhojpuri Low-Resource Neural Machine Translation (NMT)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: NLLB-200](https://img.shields.io/badge/Model-NLLB--200--Distilled--600M-orange)](https://huggingface.co/facebook/nllb-200-distilled-600M)
[![Held-out BLEU](https://img.shields.io/badge/Held--out%20BLEU-7.30-brightgreen)](results/bleu.txt)

A reproducible, low-resource Neural Machine Translation system adapting a pretrained multilingual Seq2Seq Transformer (`facebook/nllb-200-distilled-600M`) to translate English sentences into Bhojpuri (`eng_Latn` $\rightarrow$ `bho_Deva`).

---

## 1. Project Architecture

```
English Source Text (Latin Script)
                │
                ▼
SentencePiece BPE Tokenization (eng_Latn)
                │
                ▼
Pretrained Seq2Seq Transformer (NLLB-200 Distilled 600M)
                │
         [Cross-Attention]
                │
                ▼
Autoregressive Decoder & Beam Search (forced_bos_token = bho_Deva)
                │
                ▼
Bhojpuri Output (Devanagari Script)
```

---

## 2. Dataset & Preprocessing

- **Dataset Source**: [`nilayshenai/English-Bhojpuri_Translation_Dataset`](https://huggingface.co/datasets/nilayshenai/English-Bhojpuri_Translation_Dataset)
- **Raw Sentences**: 28,999 parallel pairs
- **Cleaned Dataset**: 28,571 verified pairs (after deduplication, whitespace normalization, and script validation)
- **Splits**:
  - `data/train.csv` (22,856 pairs — 80%)
  - `data/val.csv` (2,857 pairs — 10%)
  - `data/test.csv` (2,858 pairs — 10% held-out)
  - `data/train_2k.csv` (2,000 pairs — fast low-resource benchmark)

---

## 3. Benchmark Results

Evaluated on 200 held-out test sentence pairs using SacreBLEU:

| Metric | Score | Configuration |
| :--- | :--- | :--- |
| **Corpus BLEU** | **7.30** | SacreBLEU (Beam Search, `num_beams=4`) |
| **Training Pairs** | 2,000 | Low-resource benchmark subset |
| **Epochs** | 3 | AdamW (lr=3e-5, FP16 AMP) |

### Sample Translations
| English Source | Reference Bhojpuri | Model Prediction |
| :--- | :--- | :--- |
| *Hello, how are you?* | हैलो, रउआ कइसन बानी? | **हैलो, तू कइसे हव?** |
| *Where are you going today?* | आजु कहवाँ जा रहल बानी? | **तू आज कहवाँ जात हउअ?** |
| *May I sit on my bench?* | बेंच पे हम बइठ सकी ल। | **का हम आपन बेंच पर बइठ सकत हईं?** |
| *The foreman says you have to work tonight!* | परबंधक क कहल हव की आपको काम करय के हव। | **नौकमेन कहत हव कि तोहके आज रात काम करे के हव।** |
| *I want to learn Bhojpuri language.* | हमके भोजपुरी भाषा सीखे के बा। | **हम भोजपुरी भाषा सीखे चाहत हईं।** |

---

## 4. Repository Structure

```
MeetMux/
├── data/
│   ├── README.md              # Dataset stats & cleaning details
│   ├── cleaned.csv            # Cleaned, aligned dataset (28,571 pairs)
│   ├── train.csv              # 80% train split
│   ├── val.csv                # 10% validation split
│   ├── test.csv               # 10% held-out test split
│   └── train_2k.csv           # 2,000 pair benchmark subset
├── notebooks/
│   ├── 01_data_check.ipynb    # Step 1 & 2 inspection notebook
│   └── 02_train.ipynb         # Step 4-8 Google Colab training notebook
├── src/
│   ├── data_preprocessing.py  # Cleaning & alignment script
│   ├── split_data.py          # Reproducible split generator
│   ├── tokenizer_utils.py     # SentencePiece BPE tokenizer inspection
│   ├── train.py               # Standalone fine-tuning script
│   ├── translate.py           # translate(text) -> str inference hook
│   └── evaluate.py            # SacreBLEU test-set evaluator
├── models/
│   └── README.md              # Architecture documentation & judge Q&A
├── results/
│   ├── bleu.txt               # Recorded BLEU benchmark (7.30)
│   └── sample_translations.csv# Reference vs. Predicted test outputs
├── requirements.txt           # Project dependencies
└── README.md                  # Main documentation
```

---

## 5. Quickstart

### Installation
```bash
pip install -r requirements.txt
```

### Run Inference
```python
from src.translate import translate

print(translate("Hello, how are you?"))
# Output: हैलो, तू कइसे हव?
```

### Run Evaluation
```bash
python src/evaluate.py --test_path data/test.csv --sample_size 200
```
