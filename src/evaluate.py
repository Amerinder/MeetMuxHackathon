"""
Evaluation module for English -> Bhojpuri NMT.
Hackathon Role: Member 1 (ML / NLP Workflow).
Handles Step 8 (Corpus BLEU evaluation on held-out test set).
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

DEFAULT_MODEL = "models/bhojpuri-nmt-best"
FALLBACK_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "bho_Deva"


def evaluate_test_set(
    test_csv_path: str = "data/test.csv",
    model_path: str = None,
    results_dir: str = "results",
    sample_size: int = 200
):
    if model_path is None:
        model_path = DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else FALLBACK_MODEL

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model '{model_path}' on {device}...")

    print(f"Loading test set from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    if sample_size and sample_size < len(test_df):
        print(f"Subsampling {sample_size} test pairs for evaluation...")
        eval_df = test_df.head(sample_size).copy()
    else:
        eval_df = test_df.copy()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        use_fast=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()

    target_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    predictions = []
    references = [str(ref).strip() for ref in eval_df["bho"]]

    print("Generating predictions on held-out test set...")
    for text in tqdm(eval_df["en"]):
        tokenizer.src_lang = SRC_LANG
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True
            )
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        predictions.append(pred)

    os.makedirs(results_dir, exist_ok=True)

    # Compute BLEU
    if sacrebleu:
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        score = bleu.score
        print(f"\n==========================================")
        print(f"Corpus BLEU Score: {score:.2f}")
        print(f"Details: {bleu}")
        print(f"==========================================\n")
    else:
        score = 0.0
        print("\nsacrebleu not installed; saving raw predictions without score calculation.")

    # Save bleu.txt
    bleu_file = os.path.join(results_dir, "bleu.txt")
    with open(bleu_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Samples: {len(eval_df)}\n")
        f.write(f"Corpus BLEU: {score:.2f}\n")
    print(f"Summary written to {bleu_file}")

    # Save sample_translations.csv
    eval_df["Predicted_Bhojpuri"] = predictions
    sample_file = os.path.join(results_dir, "sample_translations.csv")
    eval_df.to_csv(sample_file, index=False, encoding="utf-8")
    print(f"Translations saved to {sample_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate English -> Bhojpuri NMT on held-out test set")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--sample_size", type=int, default=200)
    args = parser.parse_args()

    evaluate_test_set(
        test_csv_path=args.test_path,
        model_path=args.model_path,
        results_dir=args.results_dir,
        sample_size=args.sample_size
    )
