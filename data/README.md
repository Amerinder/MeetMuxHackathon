# Dataset Documentation: English → Bhojpuri Parallel Corpus

## Source Information
- **Source Repository**: [`nilayshenai/English-Bhojpuri_Translation_Dataset`](https://huggingface.co/datasets/nilayshenai/English-Bhojpuri_Translation_Dataset)
- **License**: MIT
- **Language Direction**: English (`en`) → Bhojpuri (`bho`)
- **Script**: Bhojpuri text in Devanagari script (`Deva`), English in Latin script (`Latn`)
- **Original Format**: JSON Lines (`engbhoj.jsonl`) with nested `{"translation": {"en": "...", "bho": "..."}}` records.

## Summary Statistics
- **Total Raw Sentence Pairs**: 28,999
- **Empty Rows Removed**: 5
- **Exact Duplicate Pairs Removed**: 361
- **Unusable / Punctuation-Only / Misaligned Pairs Removed**: 62
- **Final High-Quality Cleaned Pairs**: **28,571** (saved in `data/cleaned.csv`)

## Step 3: Reproducible Train / Validation / Test Splits
- **Seed**: `42` (deterministic random shuffle, guaranteed zero leakage/overlap)
- **`data/train.csv`**: **22,856 pairs** (80.0%) — Used for Seq2Seq fine-tuning.
- **`data/val.csv`**: **2,857 pairs** (10.0%) — Used for validation loss evaluation and early stopping.
- **`data/test.csv`**: **2,858 pairs** (10.0%) — Held-out evaluation set for final corpus BLEU scoring.

### Low-Resource Subsets (PDF Step 14 Experiment)
- **`data/train_2k.csv`**: **2,000 pairs** — Rapid baseline training / debugging run.
- **`data/train_10k.csv`**: **10,000 pairs** — Intermediate low-resource benchmark.
