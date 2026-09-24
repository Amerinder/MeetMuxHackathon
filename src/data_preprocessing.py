"""
Data preprocessing and verification module for English -> Bhojpuri NMT.
Hackathon Role: Member 1 (ML / NLP Workflow)
Handles Step 1 (Dataset inspection & deduplication) and Step 2 (Cleaning & alignment validation).
"""

import os
import json
import re
import csv
from typing import List, Dict, Tuple


# Regex patterns for script and character validation
BHO_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
EN_LATIN_RE = re.compile(r'[a-zA-Z]')
PUNCT_ONLY_RE = re.compile(r'^[\s\W\d_]+$')


def normalize_text(text: str) -> str:
    """Normalize whitespace and clean unwanted control characters."""
    if not text:
        return ""
    # Standardize spaces and newlines
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove leading dashes/bullet markers often left over from subtitle dialogues
    text = re.sub(r'^[-—–•]\s*', '', text)
    return text.strip()


def is_valid_pair(en: str, bho: str, min_words: int = 1, max_words: int = 80) -> bool:
    """
    Check if the sentence pair is valid and strictly aligned:
    - English must contain actual Latin characters (not just numbers/punctuation).
    - Bhojpuri must contain actual Devanagari characters (not just numbers/punctuation).
    - Neither side can be punctuation/symbol-only.
    - Word counts must stay within reasonable bounds [min_words, max_words].
    - Extreme length ratio is rejected to avoid misaligned text.
    """
    if not en or not bho:
        return False
    
    # Must contain genuine linguistic content
    if not EN_LATIN_RE.search(en):
        return False
    if not BHO_DEVANAGARI_RE.search(bho):
        return False
        
    # Reject if either side is purely punctuation/numbers
    if PUNCT_ONLY_RE.match(en) or PUNCT_ONLY_RE.match(bho):
        return False
        
    en_words = len(en.split())
    bho_words = len(bho.split())
    
    # Length boundaries
    if en_words < min_words or bho_words < min_words:
        return False
    if en_words > max_words or bho_words > max_words:
        return False
        
    # Check length ratio to prevent severe misalignments
    ratio = max(en_words, bho_words) / max(min(en_words, bho_words), 1)
    if ratio > 4.5 and max(en_words, bho_words) > 8:
        return False
        
    return True


def load_raw_dataset(raw_path: str) -> List[Dict[str, str]]:
    """Load JSONL dataset containing {"translation": {"en": "...", "bho": "..."}}."""
    pairs = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                trans = data.get('translation', {})
                en = trans.get('en', '')
                bho = trans.get('bho', '')
                pairs.append({'en': en, 'bho': bho})
            except Exception as e:
                print(f"[Warning] Skipped malformed JSON at line {idx}: {e}")
    return pairs


def clean_and_inspect_dataset(raw_path: str, output_csv_path: str = None) -> Tuple[List[Dict[str, str]], Dict]:
    """Execute Step 1 inspection and Step 2 cleaning."""
    print(f"Loading raw dataset from: {raw_path}")
    raw_pairs = load_raw_dataset(raw_path)
    total_raw = len(raw_pairs)
    
    seen = set()
    cleaned_pairs = []
    num_empty = 0
    num_duplicates = 0
    num_unusable = 0
    
    for item in raw_pairs:
        en = normalize_text(item.get('en', ''))
        bho = normalize_text(item.get('bho', ''))
        
        if not en or not bho:
            num_empty += 1
            continue
            
        pair_key = (en, bho)
        if pair_key in seen:
            num_duplicates += 1
            continue
        seen.add(pair_key)
        
        if not is_valid_pair(en, bho):
            num_unusable += 1
            continue
            
        cleaned_pairs.append({'en': en, 'bho': bho})
        
    stats = {
        'total_raw': total_raw,
        'empty_rows_removed': num_empty,
        'exact_duplicates_removed': num_duplicates,
        'unusable_or_misaligned_removed': num_unusable,
        'total_cleaned': len(cleaned_pairs)
    }
    
    print("=== Step 1 & 2 Dataset Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
        
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['en', 'bho'])
            writer.writeheader()
            writer.writerows(cleaned_pairs)
        print(f"Cleaned dataset successfully saved to: {output_csv_path}")
        
    return cleaned_pairs, stats


if __name__ == '__main__':
    raw_file = os.path.join('data', 'raw', 'engbhoj.jsonl')
    out_file = os.path.join('data', 'cleaned.csv')
    clean_and_inspect_dataset(raw_file, out_file)
