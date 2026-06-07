# PyTorch Python Code Summarization Tool

This repository contains a PyTorch-based Python code summarization tool that implements, trains, and evaluates two Sequence-to-Sequence (Seq2Seq) model architectures from scratch on the CodeXGLUE Python dataset to generate natural language summaries (docstrings) for Python functions.

---

## Model Architectures

1. **LSTM Seq2Seq with Bahdanau Attention (`lstm_seq2seq` - Baseline):**
   - **Encoder:** A bidirectional multi-layer LSTM that processes code sequences.
   - **Attention:** A custom Bahdanau (additive) attention mechanism that aligns target words with code source tokens.
   - **Decoder:** A unidirectional LSTM cell incorporating attention context vectors.
   - **Teacher Forcing:** Implements a scheduled teacher forcing ratio decaying as: `teacher_forcing_ratio = max(0.5, 1.0 - (epoch * 0.05))`.

2. **Transformer Seq2Seq from Scratch (`transformer_seq2seq` - Main Model):**
   - **Custom Multi-Head Self-Attention:** Developed entirely from mathematical principles, avoiding standard high-level libraries.
   - **Sinusoidal Positional Encodings:** Implemented from scratch to represent sequence positions.
   - **Encoder & Decoder Layers:** Modular multi-head self-attention, cross-attention, layer normalization, residual connections, and Feedforward layers.
   - **Causal Masking:** Natively parallelizes training using causal masks for target decoder inputs.

---

## Directory Structure

```
codesummer/
├── data/                  # Local dataset cache and training corpus (auto-created)
├── models/                # Saved PyTorch checkpoints and tokenizer (auto-created)
├── results/               # Training curves, CSV metrics, and comparisons (auto-created)
│   ├── metrics.csv        # Quantitative test metrics for both models
│   ├── comparison_table.md # Summary comparison table of both models
│   ├── training_curves.png # Comparative training loss/perplexity curves
│   └── sample_predictions.txt # Qualitative predictions on test samples
├── src/                   # Core Python package
│   ├── __init__.py
│   ├── config.py          # Shared hyperparameter configurations
│   ├── dataset.py         # PyTorch Dataset and DataLoader collation logic
│   ├── model.py           # Model definitions built from scratch
│   ├── tokenizer.py       # direct wrapper for ByteLevelBPETokenizer
│   └── utils.py           # Evaluation metrics (BLEU, ROUGE) and plotting
├── train_tokenizer.py     # Script to train custom BPE tokenizer on our dataset
├── train.py               # Main model training and validation script
├── evaluate.py            # Benchmark evaluation on the test split
├── summarize.py           # CLI inference script to generate summaries
├── test_model.py          # Unit tests to verify tensor shapes and gradients
├── requirements.txt       # Dependencies
└── README.md              # Documentation and execution instructions
```

---

## Setup Instructions

### 1. Install Dependencies
Execute the following to install all necessary requirements (without the `transformers` library):
```bash
pip install -r requirements.txt
```

### 2. Train the Custom Tokenizer
Train the Byte-level BPE tokenizer on the python corpus with a locked vocabulary size of `16000`:
```bash
python train_tokenizer.py
```
This writes the configuration file `models/tokenizer.json`.

### 3. Run Verification Tests
Verify the neural layer dimensions and backpropagation flow:
```bash
python test_model.py
```

---

## Training & Evaluation Guidelines

### Phase 1: Pipeline Verification (100 samples)
Run a quick, 1-epoch pipeline check with a small dataset to verify the train, validation, and evaluation paths:
```bash
# Verify LSTM Seq2Seq pipeline
python train.py --model_type lstm_seq2seq --train_size 100 --val_size 100 --epochs 1

# Verify Transformer Seq2Seq pipeline
python train.py --model_type transformer_seq2seq --train_size 100 --val_size 100 --epochs 1
```

### Phase 2: Final Comparative Experiments
Train both models on the complete experimental subset (**10,000 train samples**, **1,000 validation samples**, for a maximum of **10 epochs** with **early stopping patience of 3** epochs based on the lowest validation loss):
```bash
# Train baseline LSTM model
python train.py --model_type lstm_seq2seq --train_size 10000 --val_size 1000 --epochs 10

# Train main Transformer model
python train.py --model_type transformer_seq2seq --train_size 10000 --val_size 1000 --epochs 10
```

### Phase 3: Benchmark Evaluation
Evaluate both trained model checkpoints on the test split (**1,000 test samples**), compute loss, perplexity, BLEU-4, and ROUGE-1/2/L scores, and write predictions to the `results/` folder:
```bash
python evaluate.py --test_size 1000
```
This auto-generates:
- `results/metrics.csv`
- `results/comparison_table.md`
- `results/sample_predictions.txt`

---

## Run CLI Inference

You can summarize custom Python code snippets or files using either model via greedy decoding:

```bash
# Summarize a text snippet using the Transformer
python summarize.py --model_type transformer_seq2seq --code "def add(x, y): return x + y"

# Summarize a local python file using the LSTM
python summarize.py --model_type lstm_seq2seq --file "src/config.py"
```