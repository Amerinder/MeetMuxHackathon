"""
Verification and Test Suite for Member 2 Work
"""

import os
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_app():
    print("=== 1. Gradio App Build & Import Test ===")
    from src.app import build_app, translate_text, get_bleu_score, init_model, SAMPLE_EXAMPLES
    app = build_app()
    assert app is not None, "Gradio app failed to initialize"
    print("SUCCESS: Gradio App built successfully.")

    print("\n=== 2. BLEU Display Test ===")
    bleu_info = get_bleu_score()
    print("Retrieved BLEU Info:", bleu_info)
    assert len(bleu_info) > 0, "BLEU info should not be empty"
    print("SUCCESS: BLEU retrieval logic functions properly.")

    print("\n=== 3. Empty & Whitespace Input Handling ===")
    res_empty = translate_text("")
    res_spaces = translate_text("   ")
    print("Empty response:", res_empty)
    print("Whitespace response:", res_spaces)
    assert "Please enter" in res_empty
    assert "Please enter" in res_spaces
    print("SUCCESS: Input validation handles blank inputs correctly.")

    print("\n=== 4. Testing 10+ Unseen Sentences ===")
    unseen_sentences = [
        "Where are you going?",
        "How are you?",
        "What is your name?",
        "I am going to the market.",
        "The weather is very hot today.",
        "Can you help me with this work?",
        "She is reading a book under the tree.",
        "We have completed the project on time.",
        "They will arrive tomorrow morning.",
        "Please give me a glass of cold water.",
        "This village is very peaceful and green.",
    ]
    for idx, sentence in enumerate(unseen_sentences, 1):
        out = translate_text(sentence)
        print(f"[{idx}] '{sentence}' -> '{out}'")
        assert out is not None and len(out) > 0, f"Translation failed for '{sentence}'"
    print(f"SUCCESS: Successfully tested {len(unseen_sentences)} unseen sentences without crash.")

    print("\n=== 5. Security & Sensitive Data Check ===")
    member2_files = [
        ROOT / "src" / "app.py",
        ROOT / "requirements.txt",
        ROOT / "results" / "bleu.txt",
        ROOT / "results" / "sample_translations.csv",
    ]
    secret_pattern = re.compile(r"(ghp_[a-zA-Z0-9]+|sk-[a-zA-Z0-9]+|bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE)
    for file_path in member2_files:
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            match = secret_pattern.search(content)
            assert match is None, f"Potential secret found in {file_path.name}: {match.group(0)}"
    print("SUCCESS: No secrets or credentials found in Member 2 files.")

    print("\n=== ALL MEMBER 2 TESTS PASSED! ===")

if __name__ == "__main__":
    test_app()
