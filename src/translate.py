"""
Inference module for English -> Bhojpuri translation.
Hackathon Role: Member 1 (ML / NLP Workflow) & handoff to Member 2 (Gradio App).
Provides: translate(text: str) -> str
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_CHECKPOINT = "models/bhojpuri-nmt-best"
FALLBACK_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "bho_Deva"

_model = None
_tokenizer = None
_device = None


def load_model_and_tokenizer(model_path: str = None):
    global _model, _tokenizer, _device
    
    if model_path is None:
        if os.path.exists(DEFAULT_CHECKPOINT):
            model_path = DEFAULT_CHECKPOINT
        else:
            model_path = FALLBACK_MODEL

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


def translate(text: str, max_length: int = 128) -> str:
    """
    Translate an English sentence to Bhojpuri.
    This function is directly imported by Member 2 into the Gradio UI.
    """
    global _model, _tokenizer, _device
    
    if _model is None or _tokenizer is None:
        load_model_and_tokenizer()

    text = str(text).strip()
    if not text:
        return ""

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


if __name__ == "__main__":
    print("Testing translate() on 10 sample sentences...")
    test_sentences = [
        "Hello, how are you?",
        "Where are you going today?",
        "Since childhood, she has never asked me anything till now.",
        "The foreman says you have to work tonight!",
        "May I sit on my bench?",
        "Indeed he deserved to live.",
        "What do you mean, he just wasn't here?",
        "I want to learn Bhojpuri language.",
        "We built this translator during the hackathon.",
        "Good morning, my friend!"
    ]
    
    for s in test_sentences:
        out = translate(s)
        print(f"EN:  {s}")
        print(f"BHO: {out}\n")
