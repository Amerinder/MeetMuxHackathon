"""
Training script for English -> Bhojpuri NMT using Seq2SeqTrainer.
Hackathon Role: Member 1 (ML / NLP Workflow)
Handles Step 4 (Tokenizer/Model setup), Step 5 (Baseline check), and Step 6 (Fine-tuning).
"""

import os
import inspect
import argparse
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "bho_Deva"


def load_dataset_splits(train_path: str, val_path: str):
    """Load train and validation CSV files into Hugging Face DatasetDict."""
    print(f"Loading train split from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading val split from {val_path}...")
    val_df = pd.read_csv(val_path)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })


def train_model(
    train_path: str = "data/train_2k.csv",
    val_path: str = "data/val.csv",
    output_dir: str = "models/bhojpuri-nmt-best",
    model_name: str = MODEL_NAME,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    max_src_length: int = 128,
    max_tgt_length: int = 128,
    fp16: bool = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | CUDA Available: {torch.cuda.is_available()}")

    # 1. Load Tokenizer (Step 4)
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        use_fast=True
    )

    # 2. Load Datasets
    raw_datasets = load_dataset_splits(train_path, val_path)

    # 3. Preprocessing & dynamic tokenization
    def preprocess_fn(examples):
        tokenizer.src_lang = SRC_LANG
        model_inputs = tokenizer(
            [str(x) for x in examples["en"]],
            max_length=max_src_length,
            truncation=True
        )
        tokenizer.src_lang = TGT_LANG
        labels = tokenizer(
            text_target=[str(x) for x in examples["bho"]],
            max_length=max_tgt_length,
            truncation=True
        )
        tokenizer.src_lang = SRC_LANG
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=["en", "bho"]
    )

    # 4. Load Model
    print(f"Loading pretrained model {model_name}...")
    torch_dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype
    ).to(device)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    # Step 5: Baseline Check (Verify forward pass & loss)
    print("Running baseline sanity check...")
    sample_batch = data_collator([tokenized_datasets["train"][0]])
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
    with torch.no_grad():
        test_out = model(**sample_batch)
    print(f"Baseline sanity loss: {test_out.loss.item():.4f}")

    # 5. Training Arguments (Step 6)
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    strategy_key = "eval_strategy" if hasattr(Seq2SeqTrainingArguments(output_dir="tmp"), "eval_strategy") else "evaluation_strategy"

    args_dict = {
        "output_dir": checkpoint_dir,
        strategy_key: "epoch",
        "save_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 2,
        "weight_decay": 0.01,
        "save_total_limit": 1,
        "num_train_epochs": num_epochs,
        "predict_with_generate": True,
        "fp16": fp16 and torch.cuda.is_available(),
        "logging_steps": 25,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": "none"
    }

    training_args = Seq2SeqTrainingArguments(**args_dict)

    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "data_collator": data_collator
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # 6. Fine-Tuning Execution
    print(f"Starting fine-tuning for {num_epochs} epochs...")
    trainer.train()

    # 7. Save Best Checkpoint & Tokenizer
    print(f"Saving best model and tokenizer to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune English -> Bhojpuri NMT model")
    parser.add_argument("--train_path", type=str, default="data/train_2k.csv", help="Path to training CSV")
    parser.add_argument("--val_path", type=str, default="data/val.csv", help="Path to validation CSV")
    parser.add_argument("--output_dir", type=str, default="models/bhojpuri-nmt-best", help="Directory to save final model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    args = parser.parse_args()

    train_model(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
