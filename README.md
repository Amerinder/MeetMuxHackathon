---
title: MeetMux Bhojpuri Translator
emoji: 🌐
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.10"
app_file: src/app.py
pinned: false
---

# English → Bhojpuri Low-Resource Neural Machine Translation

A lightweight, reproducible neural machine translation (NMT) system that translates English sentences into Bhojpuri using subword tokenization, a pretrained multilingual sequence-to-sequence Transformer, held-out BLEU evaluation, and an interactive Gradio user interface.

---

## 1. Problem Statement

Bhojpuri is a regional Indo-Aryan language spoken by over 50 million people, primarily in northern India and the Terai region of Nepal. Despite having a substantial speaker base, Bhojpuri is digitally low-resource:
- Parallel English–Bhojpuri digital corpora are scarce and noisy compared to high-resource languages like Hindi, French, or German.
- Existing general-purpose translation services often lack support for or perform poorly on regional dialects and colloquial Bhojpuri expressions.
- Training large neural machine translation models from scratch on low-resource language pairs often leads to overfitting and poor generalization.

---

## 2. Project Objective

The objective of this project is to develop an efficient, reproducible English-to-Bhojpuri neural machine translation pipeline by:
1. Adapting a pretrained multilingual Seq2Seq Transformer (e.g., IndicBART / mBART / NLLB) rather than training from scratch.
2. Utilizing subword tokenization (SentencePiece/BPE) to represent rare and agglutinated dialectal terms.
3. Evaluating translation quality rigorously on a fixed, held-out test split using standard corpus-level BLEU.
4. Exposing the system via an interactive, lightweight Gradio web interface for direct user demonstration.

---

## 3. System Architecture

```text
       English Input Sentence
                 │
                 ▼
      Subword Tokenization (BPE/SentencePiece)
                 │
                 ▼
 Pretrained Multilingual Seq2Seq Transformer Encoder
                 │
                 ▼
 Cross-Attention Mechanism & Autoregressive Decoder
                 │
                 ▼
        Bhojpuri Target Text
          ┌──────┴──────┐
          ▼             ▼
   Corpus BLEU      Gradio UI
   Evaluation       Interactive Demo
```

### Key Technical Mechanisms
- **Subword Tokenization**: Breaks words into morphological subwords and characters, drastically alleviating out-of-vocabulary (OOV) issues when handling rare Bhojpuri terms.
- **Multilingual Knowledge Transfer**: Leverages cross-lingual representations from related Indo-Aryan languages (such as Hindi and Bengali) encoded in pretrained multilingual weights.
- **Attention Mechanism**: Allows the decoder to dynamically attend to relevant source words during sequence generation.

---

## 4. Technology Stack

| Component | Tool / Library |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch |
| **Model & Tokenizer** | Hugging Face Transformers |
| **Dataset Processing** | Hugging Face Datasets / pandas |
| **Evaluation Metric** | SacreBLEU / Hugging Face Evaluate |
| **User Interface** | Gradio |
| **Version Control** | Git + GitHub |

---

## 5. Dataset

- **Language Pair**: English (`en`) → Bhojpuri (`bho`)
- **Corpus Source**: Curated English–Bhojpuri parallel text corpora (e.g., AI4Bharat / Samanantar / Tatoeba / PMIndia subsets).
- **Preprocessing**: Whitespace normalization, duplicate and empty row removal, sentence alignment verification.
- **Data Splits**: Fixed, reproducible random split (recommended 80% Train, 10% Validation, 10% Held-out Test).
- **License**: Refer to the upstream dataset distribution guidelines (e.g., CC-BY-4.0 / Open Data Commons).

---

## 6. Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <REPO_URL>
   cd MeetMux
   ```

2. **Create and activate a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

   # For training and evaluation tools, install the additional dependencies:
   pip install -r requirements-training.txt
   ```

---

## 7. Running the Application

Launch the interactive Gradio translation interface:
```bash
python src/app.py
```

The application will start locally at:
```text
http://127.0.0.1:7860/
```

### Application Features:
- **Clean Input/Output**: English text input box, single-click "Translate" button, and read-only Bhojpuri translation display.
- **Model Info & Status**: Real-time display of model architecture status and checkpoint availability.
- **BLEU Score Panel**: Dynamically displays the held-out evaluation BLEU score from `results/bleu.txt`.
- **Preloaded Test Sentences**: Quick-access examples to test unseen sentences instantly.
- **Decoupled Architecture**: Model loading and inference (`translate_text`) are completely decoupled from UI layout code.

---

## 8. Training & Evaluation Pipeline

### Model Fine-Tuning (Member 1)
- Managed by `src/train.py` and exploratory notebooks `notebooks/01_data_check.ipynb` and `notebooks/02_train.ipynb`.
- Fine-tunes the multilingual Seq2Seq model on the cleaned English–Bhojpuri parallel training set.
- Checkpoints and tokenizer configurations are saved to `models/`.

### Inference (`src/translate.py`)
- Provides a clean `translate(text: str) -> str` interface called directly by the Gradio application.

### Evaluation (`src/evaluate.py`)
```text
Held-Out Test Set ──> Model Inference ──> SacreBLEU ──> results/bleu.txt
```
- Evaluated on a strictly held-out test split never seen during training or validation.
- Output metrics are saved to `results/bleu.txt`.

---

## 9. Evaluation Results & Sample Translations

### BLEU Score
Evaluated on 200 held-out test sentences using SacreBLEU:
- **Model**: `facebook/nllb-200-distilled-600M`
- **Language Direction**: English (`eng_Latn`) → Bhojpuri (`bho_Deva`)
- **Corpus BLEU**: **`7.30`**
- **Test Set Size**: 200 sentence pairs
- **Metric File**: [`results/bleu.txt`](results/bleu.txt)

---

### Sample Translations (10+ High-Quality Examples)
Evaluated directly from the held-out test split ([`results/sample_translations.csv`](results/sample_translations.csv)):

| # | English Source | Human Reference (Bhojpuri) | Model Prediction (`nllb-200`) |
| :-: | :--- | :--- | :--- |
| 1 | *What do you mean?* | का मतलब | का मतलब हव? |
| 2 | *How can I help you?* | हम कइसे तोहार मदद कई सकीऽल। | हम तोहार कइसे मदद कइ सकत हईं? |
| 3 | *You're right.* | आप सही हवँ। | तु सही हउअ। |
| 4 | *You're doing good.* | तूँ अच्छा करत हवा। | तु अच्छा करत हउअ। |
| 5 | *Teach me swimming.* | हमके तैरब सिखाऽव। | हमके तैरना सिखावा। |
| 6 | *Will you start a school?* | का तु एक स्कूल शूरु करब ? | का तु स्कूल शुरू कईब? |
| 7 | *Father, do not do it, I beseech you!* | पिताजी अइसन जानि करा, हम तुहसे बिनती करत हईं | पिताजी, अइसन मत करा, हम तोहसे बिनती करत हईं। |
| 8 | *I bought a new car.* | हम नया कार खरीद लेहलीं। | हम एगो नई कार खरीदलीं। |
| 9 | *I want to spend my life with her...* | हम चाहत हईं ओकरे संहे आपन जीवन बितावल.... | हम ओकरे साथ आपन जीवन बितावे चाहत हईं। |
| 10 | *My father is a university professor.* | हमार बाउजी विश्वविद्यालय में प्रोफेसर हवें। | हमार पिताजी विश्वविद्यालय के प्रोफेसर हऽ। |
| 11 | *Because the soup was too cold, I warmed it in the microwave.* | काहे की सूप बहुत ठंडा रहल हम माइक्रोबेब मे यहके हम गरम कइनी । | चूंकि सूप बहुत ठंडा रहे, हम माइक्रोवेव में गरम कइलीं। |
| 12 | *The book is very interesting.* | किताब बहुत रोचक ह । | किताब बहुते रोचक हऽ। |

---

### Error Analysis (Imperfect Examples)
Critical examination of translation failures on the held-out test set:

| # | English Source | Human Reference | Model Prediction | Linguistic Issue |
| :-: | :--- | :--- | :--- | :--- |
| 1 | *Kick that old man's ass!* | ओ बूढ़ मनई क लात! | ओ बुढ़ऊ के गधे में टक्कर द। | **Literal Idiom Translation**: The vulgar/slang idiom was translated literally into "गधे" (donkey). |
| 2 | *You switched off the lights.* | तु बिजली बुता द् । | तु लाइट बंद कइ दिहला। | **Dialect vs Loanword**: Employs borrowed English loanwords ("लाइट बंद") rather than dialectal idiom ("बुता देहल"). |
| 3 | *When does the cock crow?* | जब कौउवा काव काव करत ह । | मुर्गा कब मुरगावेला? | **Morphological Neologism**: The model derived an anomalous verb form ("मुरगावेला") instead of standard crowing terms. |
| 4 | *All he needed was a suit and tie and a jiggly hula gal...* | अउर एंडी के बस जरूरत रहल सूटअउट टाई क... | उनकर जरूरत बस एगो सूट और टाई... अगर तु प्लीज। | **Complex Multi-Clause Awkwardness**: Degrades on long literary clauses, leaving phrases like "if you please" transliterated verbatim. |

---

## 10. Limitations & Future Work

1. **Data Scarcity**: Low-resource parallel datasets limit domain diversity; vocabulary outside common conversational patterns may yield sub-optimal outputs.
2. **Spelling & Dialectal Variation**: Bhojpuri possesses regional colloquial variants (e.g., Western vs. Eastern Bhojpuri) that lack standardized digital orthography.
3. **Metric Limitations**: BLEU measures surface n-gram overlap and may penalize semantically valid synonymous expressions common in regional languages.
4. **Future Extensions**: Back-translation from monolingual Bhojpuri corpora, integration of human-in-the-loop evaluation, and expanding to Bhojpuri speech synthesis.

---

## 11. Team Contributions

- **Member 1 (ML / NLP)**:
  - Dataset acquisition, cleaning, filtering, and splitting.
  - Pretrained Seq2Seq Transformer setup and tokenizer configuration.
  - Model fine-tuning, hyperparameter optimization, and checkpoint persistence.
  - Inference function implementation (`src/translate.py`) and held-out BLEU evaluation (`src/evaluate.py`).

- **Member 2 (Gradio UI / Evaluation / Integration / Documentation)**:
  - Interactive Gradio translation interface (`src/app.py`).
  - Decoupled model loading, fallback handling, and UI test-sentence integration.
  - Evaluation results presentation and dynamic BLEU display.
  - Results structure (`results/bleu.txt`, `results/sample_translations.csv`).
  - Comprehensive documentation, requirements specification, and system verification.
