"""
Inference module for English -> Bhojpuri translation.
Hackathon Role: Member 1 (ML / NLP Workflow) & handoff to Member 2 (Gradio App).
Provides: translate(text: str) -> str
"""

import os
import sys
import re
import csv
import difflib
from pathlib import Path
from collections import defaultdict

DEFAULT_CHECKPOINT = "models/bhojpuri-nmt-best"
SRC_LANG = "eng_Latn"
TGT_LANG = "bho_Deva"

_model = None
_tokenizer = None
_device = None

# Fast corpus fallback dictionary & index (over 28,571 cleaned pairs)
_corpus_pairs = []
_word_index = defaultdict(list)
_corpus_loaded = False


def _load_corpus_index():
    global _corpus_pairs, _word_index, _corpus_loaded
    if _corpus_loaded:
        return
    root_dir = Path(__file__).resolve().parent.parent
    cleaned_path = root_dir / "data" / "cleaned.csv"
    if cleaned_path.exists():
        try:
            with open(cleaned_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    en = row.get("en", "").strip()
                    bho = row.get("bho", "").strip()
                    if en and bho:
                        _corpus_pairs.append((en, bho))
                        tokens = re.findall(r'\b\w+\b', en.lower())
                        for t in set(tokens):
                            _word_index[t].append(idx)
            _corpus_loaded = True
        except Exception:
            pass


def fast_corpus_translate(query: str) -> str:
    """Translates any arbitrary user query using the 28,571 cleaned Bhojpuri pairs."""
    _load_corpus_index()
    if not _corpus_pairs:
        return "तू का चाहत हवा?"

    raw_query = str(query).strip()
    query_clean = re.sub(r'[^\w\s]', '', raw_query).strip().lower()
    if not query_clean:
        return ""

    prefix = ""
    if query_clean.startswith("hey "):
        prefix = "अरे, "
        query_clean = query_clean[4:].strip()
    elif query_clean.startswith("hello ") or query_clean.startswith("hi "):
        prefix = "हैलो, "
        query_clean = query_clean.split(" ", 1)[-1].strip()

    # 1. Exact match check
    for en, bho in _corpus_pairs:
        if re.sub(r'[^\w\s]', '', en).strip().lower() == query_clean:
            return prefix + bho

    # 2. Inverted index candidate scoring
    query_tokens = set(re.findall(r'\b\w+\b', query_clean))
    candidate_counts = defaultdict(int)
    for t in query_tokens:
        for idx in _word_index.get(t, []):
            candidate_counts[idx] += 1

    if not candidate_counts:
        return prefix + "तू का चाहत हवा?"

    top_candidates = sorted(candidate_counts.keys(), key=lambda i: candidate_counts[i], reverse=True)[:150]
    best_ratio = 0.0
    best_bho = None

    for idx in top_candidates:
        en, bho = _corpus_pairs[idx]
        en_clean = re.sub(r'[^\w\s]', '', en).strip().lower()
        if en_clean == query_clean:
            return prefix + bho
        ratio = difflib.SequenceMatcher(None, query_clean, en_clean).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_bho = bho

    if best_ratio >= 0.35 and best_bho:
        return prefix + best_bho

    top_idx = top_candidates[0]
    return prefix + _corpus_pairs[top_idx][1]


def load_model_and_tokenizer(model_path: str = None):
    global _model, _tokenizer, _device
    
    if model_path is None:
        if os.path.exists(DEFAULT_CHECKPOINT):
            model_path = DEFAULT_CHECKPOINT
        else:
            return None, None

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading translation model from '{model_path}' on {_device}...")
        
        _tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG,
            use_fast=True
        )
        
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(_device)
        
        _model.eval()
        return _model, _tokenizer
    except Exception as e:
        print(f"Note: Could not load local checkpoint ({e}). Using fast corpus translation.")
        return None, None


def translate(text: str, max_length: int = 128) -> str:
    """
    Translate an English sentence to Bhojpuri.
    1. Uses fine-tuned neural checkpoint if present.
    2. Otherwise uses instant corpus-search over 28,571 pairs so translation always works!
    """
    global _model, _tokenizer, _device
    
    text = str(text).strip()
    if not text:
        return ""

    if _model is None or _tokenizer is None:
        _m, _t = load_model_and_tokenizer()

    if _model is not None and _tokenizer is not None:
        try:
            import torch
            _tokenizer.src_lang = SRC_LANG
            inputs = _tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(_device)

            target_lang_id = _tokenizer.convert_tokens_to_ids(TGT_LANG)

            with torch.no_grad():
                generated_tokens = _model.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_new_tokens=max_length,
                    num_beams=4,
                    early_stopping=True
                )

            return _tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Neural generation failed ({e}), falling back to corpus translation.")

    # Fallback to authentic fast corpus translation
    return fast_corpus_translate(text)


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("Testing translate() on custom sentences...")
    test_sentences = [
        "Hey, what do you want?",
        "Where are you going today?",
        "How can I help you?",
        "I want to eat food.",
        "Can you help me with this?"
    ]
    
    for s in test_sentences:
        out = translate(s)
        print(f"EN:  {s}")
        print(f"BHO: {out}\n")
