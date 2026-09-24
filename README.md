# MeetMux — English → Bhojpuri AI Translator

> **Low-Resource Neural Machine Translation with Subword Tokenization and Attention**

MeetMux is an end-to-end **English-to-Bhojpuri Neural Machine Translation (NMT)** system built to explore how pretrained multilingual Transformer models can be adapted to a **low-resource regional language**.

The project combines a cleaned English–Bhojpuri parallel corpus, fine-tuning of Meta's **NLLB-200 Distilled 600M** model, quantitative evaluation with **SacreBLEU**, and an interactive **Gradio** web application.

## 🚀 Project at a Glance

| Component | Details |
|---|---|
| Translation Direction | English → Bhojpuri |
| Source Language Code | `eng_Latn` |
| Target Language Code | `bho_Deva` |
| Base Model | `facebook/nllb-200-distilled-600M` |
| Cleaned Parallel Pairs | **28,571** |
| Data Split | 80% Train / 10% Validation / 10% Test |
| Evaluation Set | 200 held-out sentence pairs |
| Benchmark | **SacreBLEU** |
| Reported BLEU | **7.30** |
| Interface | **Gradio** |
| Core Frameworks | Python, PyTorch, Hugging Face Transformers & Datasets |

## 🎯 Problem Statement

Many regional and dialectal languages have significantly less digitized training data than high-resource languages such as English.

This creates challenges for Neural Machine Translation:
- Less parallel training data is available.
- Vocabulary coverage can be limited.
- Dialectal variation can make exact-match evaluation difficult.
- Training a large translation model from scratch is computationally expensive.

### Our approach

Instead of building a large Transformer from scratch, we **fine-tune an existing multilingual NMT model** on a focused English–Bhojpuri parallel corpus.

This makes the translation pipeline practical while directly addressing the low-resource setting.

## 🧠 System Architecture

```text
English Input
     │
     ▼
Subword Tokenization
eng_Latn
     │
     ▼
NLLB-200 Distilled 600M
Seq2Seq Transformer
     │
Cross-Attention
     │
     ▼
Autoregressive Decoder
+ Beam Search
     │
     ▼
Bhojpuri Output
bho_Deva
     │
     ▼
SacreBLEU Evaluation
```

### Why NLLB-200?

**NLLB — No Language Left Behind** is a multilingual translation model supporting a large number of languages, including Bhojpuri.

We use the distilled 600M variant as a practical starting point for a hackathon-scale low-resource fine-tuning experiment.

### Why subword tokenization?

The model uses a SentencePiece-based subword tokenizer. Subword units allow the model to represent rare or unseen words using smaller pieces instead of requiring every complete word to appear in the vocabulary.

### Where does attention fit?

The Transformer uses **cross-attention** to connect the decoder to relevant information from the encoded English input while generating the Bhojpuri translation.

## 📊 Dataset

We use the **English–Bhojpuri Translation Dataset** containing approximately 29K original parallel sentence pairs.

### Preprocessing

The dataset was processed to:
- Remove empty entries
- Normalize whitespace
- Remove duplicate pairs
- Check source/target alignment
- Validate expected script/content
- Create reproducible train, validation, and test splits

### Final dataset

| Split | Pairs |
|---|---:|
| Training | 22,856 |
| Validation | 2,857 |
| Test | 2,858 |
| **Total cleaned** | **28,571** |

A separate **2,000-pair subset** is included for a fast low-resource benchmark.

## 📈 Evaluation

We evaluate translation quality using **SacreBLEU**.

Recorded low-resource benchmark:
- Model: `facebook/nllb-200-distilled-600M`
- Training subset: **2,000 pairs**
- Validation subset: **500 pairs**
- Training epochs: **3**
- Optimizer: **AdamW**
- Learning rate: `3e-5`
- Mixed precision: **FP16 AMP**
- Evaluation: **200 held-out sentence pairs**
- Decoding: **Beam Search (`num_beams=4`)**
- Reported Corpus BLEU: **7.30**

### Important interpretation

BLEU measures n-gram overlap with reference translations. It is useful for benchmarking, but it does not capture every valid translation, dialectal variation, synonym, or grammatical alternative.

The reported **7.30 BLEU is a low-resource benchmark result**, not a claim of production-level translation quality.

Detailed results are available in:
```text
results/bleu.txt
results/sample_translations.csv
```

## 💻 Web Application

The project includes a Gradio interface for live demonstration.

### Features

- English sentence input
- Bhojpuri translation output
- One-click translation
- Example sentence buttons
- Model and language-pair information
- Evaluation information
- Sample translation analysis
- Clean interface for demonstration

### Example

**Input**
```text
What are you doing here?
```

**Output**
```text
तू इहाँ का करत हउआ?
```

> Output can vary depending on the active checkpoint, decoding configuration, and sentence context.

## 🛠️ Technology Stack

### Machine Learning
- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Hugging Face Datasets**
- **NLLB-200 Distilled 600M**

### NLP
- SentencePiece / subword tokenization
- Seq2Seq Transformer
- Cross-attention
- Autoregressive decoding
- Beam search

### Evaluation
- **SacreBLEU**
- Held-out test examples
- Reference vs. prediction analysis

### Application
- **Gradio**

### Development
- Jupyter / Google Colab
- Git & GitHub

## 📁 Repository Structure

```text
MeetMuxHackathon/
│
├── data/
│   ├── README.md
│   ├── cleaned.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── train_2k.csv
│
├── models/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_check.ipynb
│   └── 02_train.ipynb
│
├── results/
│   ├── bleu.txt
│   └── sample_translations.csv
│
├── src/
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── split_data.py
│   ├── tokenizer_utils.py
│   ├── train.py
│   ├── translate.py
│   └── evaluate.py
│
├── tests/
│
├── requirements.txt
├── requirements-training.txt
└── README.md
```

## ⚙️ Run the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/Amerinder/MeetMuxHackathon.git
cd MeetMuxHackathon
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Gradio application

```bash
python src/app.py
```

Open:
```text
http://127.0.0.1:7860
```

For training/evaluation dependencies:
```bash
pip install -r requirements-training.txt
```

## 🧪 Training & Evaluation

### Data inspection
```text
notebooks/01_data_check.ipynb
```

### Model fine-tuning
```text
notebooks/02_train.ipynb
```

### Training pipeline
```text
src/train.py
```

### Evaluation
```bash
python src/evaluate.py --test_path data/test.csv --sample_size 200
```

### Translation module

```python
from src.translate import translate

result = translate("Hello, how are you?")
print(result)
```

## 🔬 Reproducibility

The project is organized into clear stages:

```text
Data
  ↓
Cleaning & Alignment
  ↓
Train / Validation / Test Split
  ↓
Tokenization
  ↓
NLLB-200 Fine-Tuning
  ↓
Inference
  ↓
SacreBLEU Evaluation
  ↓
Gradio Demo
```

Data splits and evaluation artifacts are stored in the repository so the experiment can be inspected independently of the web interface.

Model weights are intentionally not committed because of their size. A compatible fine-tuned checkpoint can be placed at:

```text
models/bhojpuri-nmt-best
```

for local neural inference.

## ⚠️ Current Limitations

This is a **hackathon-scale low-resource NMT prototype**, not a production translation service.

1. The parallel corpus is relatively small compared with high-resource translation datasets.
2. Bhojpuri contains dialectal and stylistic variation that may not be fully represented.
3. BLEU can penalize valid translations that differ from the reference wording.
4. The reported 7.30 BLEU benchmark used a 2,000-pair training subset and 200-sentence held-out evaluation.
5. Model checkpoints are not stored directly in this Git repository because of their size.

## 🚀 Future Improvements

- Larger and more diverse English–Bhojpuri parallel corpora
- Better alignment and linguistic filtering
- Hyperparameter optimization
- Longer fine-tuning runs
- Data augmentation and back-translation
- Additional metrics such as **chrF++**
- Human evaluation by native Bhojpuri speakers
- Model quantization for faster inference
- Improved deployment and scalable inference

## 🌍 Why This Project Matters

MeetMux demonstrates a practical approach to **low-resource NLP**:

> A multilingual pretrained Transformer can be adapted to a regional language using a relatively small parallel corpus and a focused fine-tuning pipeline.

The project explores how existing multilingual AI technology can be adapted to increase language coverage and accessibility for regional languages.

## 👥 Team

| Role | Responsibility |
|---|---|
| **Member 1 — ML / NLP** | Data preparation, preprocessing, NLLB fine-tuning, inference, and evaluation |
| **Member 2 — Application / Integration** | Gradio interface, sample translations, evaluation display, and deployment integration |

## 📚 References

- **NLLB-200:** Meta's No Language Left Behind multilingual translation model
- **Hugging Face Transformers:** Model and tokenizer implementation
- **Hugging Face Datasets:** Dataset loading and processing
- **SacreBLEU:** Machine translation evaluation
- **English–Bhojpuri Translation Dataset:** `nilayshenai/English-Bhojpuri_Translation_Dataset`

---

## ⭐ Project Summary

**MeetMux** combines:

**Pretrained multilingual Transformer + Subword Tokenization + Attention + Low-Resource Fine-Tuning + BLEU Evaluation + Gradio**

to build an end-to-end **English → Bhojpuri Neural Machine Translation system**.

**English → NLLB-200 → Attention → Bhojpuri → BLEU → Gradio**
