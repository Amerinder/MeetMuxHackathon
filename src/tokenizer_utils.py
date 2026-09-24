"""
Tokenizer utilities and Subword/BPE demonstration for English -> Bhojpuri NMT.
Hackathon Role: Member 1 (ML / NLP Workflow)
Handles Step 4 (Tokenizer setup & subword tokenization documentation).

Model: facebook/nllb-200-distilled-600M
- Source Language: eng_Latn (English in Latin script)
- Target Language: bho_Deva (Bhojpuri in Devanagari script)
- Tokenization Mechanism: SentencePiece with Byte-Pair Encoding (BPE)
"""

import sys
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from transformers import AutoTokenizer, PreTrainedTokenizerFast


DEFAULT_MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "bho_Deva"


def get_nllb_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedTokenizerFast:
    """
    Load NLLB-200 fast tokenizer configured for English -> Bhojpuri translation.
    SentencePiece BPE tokenizer handles low-resource morphological variants by
    decomposing unfamiliar Bhojpuri words into frequent subword units.
    """
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        use_fast=True
    )
    return tokenizer


def inspect_subwords(text: str, tokenizer: PreTrainedTokenizerFast, is_target: bool = False) -> Dict[str, Any]:
    """
    Demonstrate how SentencePiece breaks words into subwords.
    This explains how the model handles rare and unseen Bhojpuri words.
    """
    if is_target:
        tokenizer.src_lang = TGT_LANG
    else:
        tokenizer.src_lang = SRC_LANG

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return {
        "text": text,
        "tokens": tokens,
        "token_ids": token_ids,
        "num_subwords": len(tokens)
    }


def preprocess_batch(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerFast,
    max_src_length: int = 128,
    max_tgt_length: int = 128
) -> Dict[str, Any]:
    """
    Preprocess and tokenize a batch of English-Bhojpuri sentence pairs.
    Handles source encoding and target label masking (-100 for loss computation).
    """
    inputs = examples["en"]
    targets = examples["bho"]

    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    model_inputs = tokenizer(
        inputs,
        max_length=max_src_length,
        truncation=True,
        padding=False  # Dynamic padding will be handled by DataCollatorForSeq2Seq
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_tgt_length,
        truncation=True,
        padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    print("Initializing English -> Bhojpuri Tokenizer...")
    tok = get_nllb_tokenizer()
    
    sample_en = "Since childhood, she has never asked me anything till now."
    sample_bho = "बचपनअ से, ऊ कबों हमसे आज ले कुछ ना मंगलअ।"
    
    en_analysis = inspect_subwords(sample_en, tok, is_target=False)
    bho_analysis = inspect_subwords(sample_bho, tok, is_target=True)
    
    print("\n--- English Tokenization ---")
    print("Text:   ", en_analysis["text"])
    print("Tokens: ", en_analysis["tokens"])
    print(f"Count:   {en_analysis['num_subwords']} subwords")
    
    print("\n--- Bhojpuri Tokenization (SentencePiece Subwords) ---")
    print("Text:   ", bho_analysis["text"])
    print("Tokens: ", bho_analysis["tokens"])
    print(f"Count:   {bho_analysis['num_subwords']} subwords")
