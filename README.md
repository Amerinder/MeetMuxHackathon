---
title: MeetMux Bhojpuri Translator
emoji: 🌐
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.10"
app_file: src/app.py
pinned: false
---

# MeetMux: English–Bhojpuri Translator

MeetMux is a low-resource machine translation project for English and Bhojpuri. It brings together a cleaned parallel corpus, an NLLB fine-tuning and evaluation workflow, and a Gradio web app.

[Open the live demo](https://amerinder-meetmux-bhojpuri-translator.hf.space) · [View the GitHub repository](https://github.com/Amerinder/MeetMuxHackathon)

## At a glance

| | |
| --- | --- |
| Translation direction | English (`eng_Latn`) → Bhojpuri (`bho_Deva`) |
| Cleaned parallel pairs | 28,571 |
| Evaluated model | Fine-tuned `facebook/nllb-200-distilled-600M` |
| Held-out evaluation | 200 sentence pairs |
| Corpus BLEU | **7.30** |
| Interface | Gradio |

## What the app does

- Accepts an English sentence and returns a Bhojpuri translation.
- Provides example prompts and a table of verified sample translations.
- Shows the project’s held-out BLEU result and model status.
- Uses the fine-tuned checkpoint when model weights are available; otherwise, it falls back to a fast search over the cleaned parallel corpus.

> **Current demo behavior:** model checkpoints are excluded from this repository, so the hosted demo currently uses the corpus-search fallback. The reported BLEU score is from the separately evaluated fine-tuned NLLB checkpoint; it is not a score for the fallback currently running in the demo.

## Method

The project starts with an English–Bhojpuri parallel dataset and removes empty, duplicate, punctuation-only, and misaligned pairs. The cleaned data contains 28,571 sentence pairs. Reproducible 80/10/10 splits are provided for training, validation, and testing.

The training workflow fine-tunes Meta’s multilingual NLLB-200 distilled 600M sequence-to-sequence model. The tokenizer uses the language codes `eng_Latn` and `bho_Deva`; inference uses beam search. The small-data benchmark recorded in `results/bleu.txt` used 2,000 training pairs, 500 validation pairs, three epochs, and a 200-pair held-out evaluation.

### Evaluation

| Metric | Result |
| --- | ---: |
| Model | `facebook/nllb-200-distilled-600M` (fine-tuned) |
| Test examples | 200 |
| SacreBLEU | **7.30** |

BLEU measures n-gram overlap with reference translations; it is useful for comparison but does not capture every valid dialectal or synonymous translation. See [`results/bleu.txt`](results/bleu.txt) and [`results/sample_translations.csv`](results/sample_translations.csv) for the recorded result and examples.

## Run locally

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Amerinder/MeetMuxHackathon.git
cd MeetMuxHackathon
python -m venv .venv
```

Activate the environment in your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the app dependencies and start Gradio:

```bash
pip install -r requirements.txt
python src/app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). To install the additional training and evaluation dependencies, run:

```bash
pip install -r requirements-training.txt
```

## Training and evaluation files

- `notebooks/01_data_check.ipynb` explores and checks the data.
- `notebooks/02_train.ipynb` documents the fine-tuning workflow.
- `src/train.py` contains the training pipeline.
- `src/evaluate.py` runs held-out evaluation and records the BLEU result.
- `src/translate.py` loads a local checkpoint when present and otherwise uses corpus search.

Model weights are intentionally not committed. To run neural inference locally, place a compatible fine-tuned checkpoint at `models/bhojpuri-nmt-best`.

## Repository layout

```text
data/       cleaned corpus and train/validation/test splits
models/     checkpoint instructions; model weights are not included
notebooks/  data exploration and training notebooks
results/    BLEU result and sample translations
src/        preprocessing, training, evaluation, inference, and Gradio app
tests/      app and sample analysis checks
```

## Dataset and model references

- Dataset: [English–Bhojpuri Translation Dataset](https://huggingface.co/datasets/nilayshenai/English-Bhojpuri_Translation_Dataset). The dataset documentation identifies the source license as MIT; review upstream terms when redistributing data.
- Base model: [Meta NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M).
- Dataset cleaning and split details: [`data/README.md`](data/README.md).
- Model and tokenization notes: [`models/README.md`](models/README.md).

## Team contributions

- **Member 1 — ML and NLP:** data preparation, model fine-tuning workflow, inference module, and held-out evaluation.
- **Member 2 — Application and integration:** Gradio interface, sample translations, evaluation display, and deployment integration.
