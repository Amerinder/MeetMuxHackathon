# Model & Tokenization Architecture: English → Bhojpuri NMT

## 1. Selected Model: `facebook/nllb-200-distilled-600M`
We selected Meta's **NLLB-200 (No Language Left Behind)** distilled 600M parameter model for fine-tuning.

### Key Specifications:
- **Architecture**: Sequence-to-Sequence (Encoder-Decoder) Transformer.
- **Source Language Tag**: `eng_Latn` (English in Latin script)
- **Target Language Tag**: `bho_Deva` (Bhojpuri in Devanagari script)
- **Parameters**: 600 Million (distilled from 54B), optimized for fast fine-tuning on standard GPUs (e.g. Google Colab T4 with 15GB VRAM).

---

## 2. Tokenization & Subword (BPE / SentencePiece) Mechanism

### Why Subwords / BPE?
In low-resource languages like Bhojpuri, vocabulary coverage is often sparse. If we use word-level tokenization:
1. Rare, dialectal, and inflected forms (e.g., `मंगलअ`, `परबंधक`, `बचपनअ`) become Out-Of-Vocabulary (`<unk>`).
2. The vocabulary table explodes in size, requiring massive parameters that overfit easily.

### Practical Mechanism Used:
- **SentencePiece with BPE (Byte-Pair Encoding)**:
  - Treats text as a raw stream of bytes/characters without language-specific whitespace assumptions.
  - Decomposes rare/compound words into frequent subword units (morphemes and prefixes/suffixes).
  - Uses byte-fallback when characters are completely novel, guaranteeing zero `<unk>` token crashes.
  - Shares subwords across related Indo-Aryan languages (Hindi, Maithili, Magahi, Bhojpuri), enabling positive cross-lingual transfer from high-resource representations.

---

## 3. Q&A for Presentation / Judges (Hackathon Guide)

| Question | Explanation |
| :--- | :--- |
| **Why Bhojpuri?** | Bhojpuri is spoken by over 50 million people but has limited digital MT parallel resources, making it a classic low-resource NLP challenge. |
| **Why a pretrained model?** | Pretrained multilingual models transfer knowledge of syntactic structure and semantic representations, enabling convergence on limited parallel datasets without training billions of weights from scratch. |
| **Why BPE / SentencePiece?** | Reduces the impact of rare and unseen words, keeps vocabulary size compact (~256k shared multilingual tokens), and eliminates Out-Of-Vocabulary errors. |
| **Why Transformer Encoder-Decoder?** | Cross-attention allows the decoder to condition on the full contextualized representation of the English source sentence while autoregressively generating Bhojpuri tokens. |
| **Why BLEU score?** | Industry-standard corpus-level n-gram overlap metric comparing generated translations with reference target sentences. |
