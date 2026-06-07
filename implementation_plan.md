# Python Code Summarization Tool in PyTorch

This project builds a custom Sequence-to-Sequence (Seq2Seq) Transformer-based model from scratch in PyTorch to generate natural language summaries (docstrings) for Python code snippets. It includes preprocessing, tokenization, model definition, training, validation, evaluation, and CLI inference.

## User Review Required

Please review the proposed architecture, dependencies, and validation strategy before we proceed with the implementation.

> [!IMPORTANT]
> - **Custom BPE Tokenizer Trained from Scratch:** We will train a custom Byte-Level BPE (Byte-Pair Encoding) tokenizer directly on the Python code and docstrings from the training dataset. We will save this configuration as `models/tokenizer.json`. This makes the vocabulary fully custom and tailored specifically to our Python source code and docstring distribution.
> - **Pretrained Language Models (like CodeBERT/GPT) are NOT used** for the main project, in accordance with the guidelines. The tokenizer will be trained entirely from scratch.
> - **Data Selection & Split Sizes**: The full CodeXGLUE python dataset has ~250k training samples. To guarantee a fair comparison and match hardware constraints:
>   - **Verification Runs:** We will use a small subset of **100 samples** for all splits (train, validation, test) to test our pipelines.
>   - **Final Comparative Experiments:** We will use **10,000 training samples**, **1,000 validation samples**, and **1,000 test samples** to train and compare our two models.
>   - Configurable flags (`--train_size`, `--val_size`, `--test_size`) will be added to the scripts to enforce these splits.
> - **Additional Packages**: We will require `nltk` and `rouge-score` to calculate BLEU and ROUGE scores. We will install these via a user-approved command.

## Proposed Changes

We will organize the code into the following structure:
```
codesummer/
├── data/                  # Local dataset cache
├── models/                # Saved PyTorch model checkpoints (.pt) and custom tokenizer.json (auto-created)
├── results/               # Training and evaluation results (auto-created)
│   ├── metrics.csv        # Quantitative test metrics for both models
│   ├── comparison_table.md # Summary comparison table of the two models
│   ├── training_curves.png # Comparative loss/perplexity plots
│   └── sample_predictions.txt # Qualitative predictions on test samples
├── src/                   # Core Python package
│   ├── __init__.py
│   ├── config.py          # Shared hyperparameter configurations
│   ├── dataset.py         # PyTorch Dataset and batching logic
│   ├── model.py           # Model implementations (lstm_seq2seq, transformer_seq2seq)
│   ├── tokenizer.py       # Helper functions to train and load custom tokenizer
│   └── utils.py           # Evaluation metrics (BLEU, ROUGE, PPL) and plotting
├── train_tokenizer.py     # Script to train BPE tokenizer on our dataset
├── train.py               # Main model training and validation script
├── evaluate.py            # Quantitative evaluation script (BLEU, ROUGE, PPL)
├── summarize.py           # Inference script to summarize custom Python code
├── test_model.py          # Verification script for model shapes and training
├── requirements.txt       # Dependencies
└── README.md              # Documentation and execution instructions
```

---

### Phase 1: Foundation & Dependencies

#### [NEW] [requirements.txt](file:///c:/Users/USER/Documents/codesummer/requirements.txt)
Define external Python dependencies:
- `torch>=2.0.0`
- `numpy>=1.24.0`
- `datasets>=2.12.0`
- `tokenizers>=0.13.0`
- `nltk>=3.8.1`
- `rouge-score>=0.1.2`
- `matplotlib>=3.7.1`
- `tqdm>=4.65.0`

#### [NEW] [config.py](file:///c:/Users/USER/Documents/codesummer/src/config.py)
A central config file containing training and model hyperparameters:
- Embedding and model dimensions: `d_model = 256`
- Number of heads: `nhead = 4`
- Encoder/Decoder layers: `num_encoder_layers = 3`, `num_decoder_layers = 3`
- Feedforward dimension: `dim_feedforward = 512`
- Dropout: `0.1`
- Learning rate: `1e-4` (with AdamW)
- Max code sequence length (strictly locked): `256`
- Max summary sequence length (strictly locked): `64`
- Max training epochs (strictly locked): `10`
- Early stopping patience (strictly locked): `3` epochs (monitored on lowest validation loss)
- Vocab size: `16000` (locked vocabulary size for custom tokenizer)

---

### Phase 2: Custom BPE Tokenizer & Dataset Preprocessing

#### [NEW] [train_tokenizer.py](file:///c:/Users/USER/Documents/codesummer/train_tokenizer.py)
A script that:
- Automatically creates the `models/` and `results/` directories if they do not exist.
- Loads the python code-docstring dataset.
- Writes a text file corpus of all python code and target docstrings from the training split.
- Uses `tokenizers.ByteLevelBPETokenizer` to train a custom BPE tokenizer on the corpus with a vocab size of `16000` and special tokens `<s>`, `<pad>`, `</s>`, `<unk>`, and `<sep>` (the `<mask>` token is removed).
- Saves the trained tokenizer to `models/tokenizer.json`.

#### [NEW] [tokenizer.py](file:///c:/Users/USER/Documents/codesummer/src/tokenizer.py)
Utility functions to load and wrap our custom tokenizer directly from the `tokenizers` library:
- Uses `tokenizers.Tokenizer.from_file("models/tokenizer.json")` to load the tokenizer.
- Provides helper methods and properties for token mapping, padding, and truncation, retrieving special token IDs (`<s>`: bos, `<pad>`: pad, `</s>`: eos, `<sep>`: sep, `<unk>`: unk).
- Implements encoding and decoding functions that clean up special tokens.

#### [NEW] [dataset.py](file:///c:/Users/USER/Documents/codesummer/src/dataset.py)
Implements:
- Data download and extraction using `datasets.load_dataset("google/code_x_glue_ct_code_to_text", "python")`.
- Loading the custom trained tokenizer via `src/tokenizer.py`.
- Padding and truncation logic to format sequences to fixed/variable lengths.
- PyTorch `Dataset` and `DataLoader` routines to yield batches of code-tokens, target-tokens, and corresponding padding/attention masks.

---

### Phase 3: Model Architecture

#### [NEW] [model.py](file:///c:/Users/USER/Documents/codesummer/src/model.py)
Implements two separate PyTorch architectures using custom building blocks written from scratch:

* **Custom Mathematical Components (Built from Scratch):**
  1. **Sinusoidal Positional Encoding:** Positional encodings computed using sine and cosine waves of different frequencies to inject word order information.
  2. **Multi-Head Self-Attention (`MultiHeadAttention`):** Multi-head scaled dot-product attention built from scratch (Linear projections of Q, K, V -> split heads -> scaled dot-product score computation with optional attention mask -> softmax -> weighted values sum -> linear project out).
  3. **Bahdanau Attention (`BahdanauAttention`):** Additive alignment score calculator for the LSTM decoder to query encoder hidden states.

* **Model Implementations:**
  1. **LSTM Seq2Seq with Attention (`lstm_seq2seq`):**
     - **Encoder:** Bidirectional multi-layer LSTM encoding Python code token sequences.
     - **Decoder:** Unidirectional LSTM decoder querying encoder states via the custom `BahdanauAttention`.
     - **Teacher Forcing:** Implemented step-by-step during training using a scheduled teacher forcing ratio `teacher_forcing_ratio = max(0.5, 1.0 - (epoch * 0.05))` (where epoch is 0-indexed), which starts at 1.0 and gradually decays to a minimum of 0.5 to smooth the transition to inference.
  2. **Transformer Seq2Seq (`transformer_seq2seq`):**
     - **Encoder:** Sinusoidal Positional Encoding + Stack of Transformer Encoder layers constructed using our custom `MultiHeadAttention` block.
     - **Decoder:** Sinusoidal Positional Encoding + Stack of Decoder layers utilizing custom self-attention and cross-attention blocks.
     - **Teacher Forcing:** Parallel teacher forcing is employed natively during training via a causal target attention mask (triangular matrix).

---

### Phase 4: Training & Validation Loop

#### [NEW] [train.py](file:///c:/Users/USER/Documents/codesummer/train.py)
Routines to:
- Automatically create the `models/` and `results/` directories if they do not exist.
- Instantiate the dataset, tokenizer, and chosen model (`--model_type` can be `lstm_seq2seq` or `transformer_seq2seq`).
- Run the training loop for up to `max_epochs=10` with early stopping patience of `3` epochs based on validation loss.
- Run validation at the end of each epoch to compute validation loss, validation perplexity, and small-sample validation BLEU (evaluated on a fast subset of 50 samples strictly for progress monitoring/logging, not model selection).
- Save the best model checkpoints to `models/best_model_lstm_seq2seq.pt` or `models/best_model_transformer_seq2seq.pt` based strictly on the **lowest validation loss** (early stopping/checkpoint selection).
- Save training log data and generate comparative plots of loss and perplexity curves for both models inside `results/training_curves.png`.

---

### Phase 5: Evaluation & Inference

#### [NEW] [utils.py](file:///c:/Users/USER/Documents/codesummer/src/utils.py)
Contains helpers for evaluation:
- BLEU-4 computation using `nltk.translate.bleu_score.corpus_bleu`.
- ROUGE-1/2/L computation using `rouge_score.rouge_scorer.RougeScorer`.
- Perplexity calculator.
- Plotting functions to compare loss and perplexity trends of both models on the same canvas inside `results/training_curves.png` using a premium, dark-themed theme.

#### [NEW] [evaluate.py](file:///c:/Users/USER/Documents/codesummer/evaluate.py)
- Evaluates both models (`lstm_seq2seq` and `transformer_seq2seq`) on the test split.
- Computes test loss, test perplexity, BLEU-4, and ROUGE-1/2/L scores for each over the test split.
- Writes the quantitative metrics to `results/metrics.csv` and a clean Markdown table comparing both models side-by-side to `results/comparison_table.md`.
- Generates qualitative sample generated summaries for a selection of test code snippets and writes them to `results/sample_predictions.txt`.

#### [NEW] [summarize.py](file:///c:/Users/USER/Documents/codesummer/summarize.py)
- Command-line inference script.
- Takes a raw Python code snippet or file, tokenizes it, loads the specified model (`--model_type lstm_seq2seq` or `transformer_seq2seq`), and runs greedy decoding to generate the output natural language summary. Beam search is documented as an optional future extension.

---

### Phase 6: Documentation

#### [MODIFY] [README.md](file:///c:/Users/USER/Documents/codesummer/README.md)
Update the documentation to provide clear setup and execution guidelines.

---

## Verification Plan

### Automated Tests
We will write a verification test in `test_model.py` to ensure that:
1. The dataset tokenizes and yields correctly shaped tensors.
2. Both architectures (LSTM Seq2Seq and Transformer Seq2Seq) execute a single forward pass without shape mismatch errors.
3. The loss computation and masking (including causal and padding masking) behave as expected.

### Manual Verification
1. Run a dummy training run for both models with `--train_size 100 --val_size 100 --epochs 1` to verify all pipelines run end-to-end:
   - `python train.py --model_type lstm_seq2seq --train_size 100 --val_size 100 --epochs 1`
   - `python train.py --model_type transformer_seq2seq --train_size 100 --val_size 100 --epochs 1`
2. Run inference for both model types to verify summaries are generated:
   - `python summarize.py --model_type lstm_seq2seq --code "def add(a, b): return a + b"`
   - `python summarize.py --model_type transformer_seq2seq --code "def add(a, b): return a + b"`
3. Execute evaluation on a subset for both models:
   - `python evaluate.py --test_size 100`

