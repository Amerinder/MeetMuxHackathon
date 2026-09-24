"""
Dataset splitting module for English -> Bhojpuri NMT.
Hackathon Role: Member 1 (ML / NLP Workflow)
Handles Step 3 (Reproducible 80/10/10 Train-Val-Test split with zero data leakage).
"""

import os
import csv
import random
from typing import Tuple, List, Dict


def load_csv(filepath: str) -> List[Dict[str, str]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_csv(filepath: str, data: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['en', 'bho'])
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data):,} pairs to {filepath}")


def split_dataset(
    cleaned_csv_path: str = os.path.join('data', 'cleaned.csv'),
    data_dir: str = 'data',
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split the cleaned dataset into reproducible train, validation, and test sets.
    Strictly verifies zero data leakage between splits.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    print(f"Loading cleaned dataset from {cleaned_csv_path}...")
    records = load_csv(cleaned_csv_path)
    total_len = len(records)
    print(f"Total dataset pairs: {total_len:,}")

    # Set deterministic random seed
    rng = random.Random(seed)
    shuffled_indices = list(range(total_len))
    rng.shuffle(shuffled_indices)

    # Compute split boundaries
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)
    # Remaining goes to test to account for rounding
    n_test = total_len - n_train - n_val

    train_indices = set(shuffled_indices[:n_train])
    val_indices = set(shuffled_indices[n_train:n_train + n_val])
    test_indices = set(shuffled_indices[n_train + n_val:])

    # Verification: Disjoint sets
    assert len(train_indices.intersection(val_indices)) == 0, "Leakage detected between train and val!"
    assert len(train_indices.intersection(test_indices)) == 0, "Leakage detected between train and test!"
    assert len(val_indices.intersection(test_indices)) == 0, "Leakage detected between val and test!"

    train_data = [records[i] for i in shuffled_indices[:n_train]]
    val_data = [records[i] for i in shuffled_indices[n_train:n_train + n_val]]
    test_data = [records[i] for i in shuffled_indices[n_train + n_val:]]

    print("\n=== Step 3 Split Summary ===")
    print(f"Random Seed: {seed}")
    print(f"Train set:      {len(train_data):,} pairs ({len(train_data)/total_len*100:.1f}%)")
    print(f"Validation set: {len(val_data):,} pairs ({len(val_data)/total_len*100:.1f}%)")
    print(f"Test set:       {len(test_data):,} pairs ({len(test_data)/total_len*100:.1f}%)")

    # Save standard splits
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    save_csv(train_path, train_data)
    save_csv(val_path, val_data)
    save_csv(test_path, test_data)

    # Optional Low-Resource subsets for rapid iteration & low-resource experiments (PDF Step 14)
    train_2k_path = os.path.join(data_dir, 'train_2k.csv')
    train_10k_path = os.path.join(data_dir, 'train_10k.csv')
    save_csv(train_2k_path, train_data[:2000])
    save_csv(train_10k_path, train_data[:10000])

    return train_data, val_data, test_data


if __name__ == '__main__':
    split_dataset()
