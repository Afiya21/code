# src/utils.py

import math
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def compute_bleu(references, hypotheses):
    """
    Computes corpus-level BLEU-4 score.
    references: list of lists of tokens (e.g. [['the', 'docstring', 'here'], ...])
    hypotheses: list of lists of tokens (e.g. [['the', 'docstring', 'generated'], ...])
    """
    # NLTK corpus_bleu expects references in shape [num_examples, num_references_per_example, seq_len]
    # Here we have 1 reference per example
    ref_formatted = [[ref] for ref in references]
    
    # Use method1 smoothing to avoid 0 scores for short sequences
    smoothing = SmoothingFunction().method1
    
    bleu4 = corpus_bleu(ref_formatted, hypotheses, smoothing_function=smoothing)
    return bleu4 * 100 # scale to percentage

def compute_rouge(references, hypotheses):
    """
    Computes average ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
    references: list of strings (raw reference summaries)
    hypotheses: list of strings (raw generated summaries)
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    r1_scores = []
    r2_scores = []
    rl_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        # Handle empty outputs gracefully
        if len(ref.strip()) == 0 or len(hyp.strip()) == 0:
            r1_scores.append(0.0)
            r2_scores.append(0.0)
            rl_scores.append(0.0)
            continue
            
        scores = scorer.score(ref, hyp)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)
        
    return {
        "rouge1": np.mean(r1_scores) * 100,
        "rouge2": np.mean(r2_scores) * 100,
        "rougeL": np.mean(rl_scores) * 100
    }

def compute_perplexity(loss_value):
    """
    Computes perplexity from cross-entropy loss value.
    Safely handles extreme loss values to prevent overflow.
    """
    try:
        return math.exp(min(loss_value, 20.0))
    except OverflowError:
        return float('inf')

def plot_training_curves(lstm_log, transformer_log, save_path):
    """
    Generates premium, dark-themed comparative plots for training and validation losses/perplexities.
    lstm_log, transformer_log: dict containing 'train_loss', 'val_loss', 'train_ppl', 'val_ppl' per epoch.
    """
    # Use dark theme style
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Custom color palette
    c_lstm_train = '#e0aaff' # soft purple
    c_lstm_val = '#9d4edd'   # medium purple
    c_trans_train = '#c7f9cc' # soft green
    c_trans_val = '#38b000'   # bright green
    
    epochs_lstm = range(1, len(lstm_log['train_loss']) + 1)
    epochs_trans = range(1, len(transformer_log['train_loss']) + 1)
    
    # Plot Losses
    ax1.plot(epochs_lstm, lstm_log['train_loss'], label='LSTM Train', color=c_lstm_train, linestyle='--', marker='o')
    ax1.plot(epochs_lstm, lstm_log['val_loss'], label='LSTM Val', color=c_lstm_val, linestyle='-', marker='s')
    ax1.plot(epochs_trans, transformer_log['train_loss'], label='Transformer Train', color=c_trans_train, linestyle='--', marker='o')
    ax1.plot(epochs_trans, transformer_log['val_loss'], label='Transformer Val', color=c_trans_val, linestyle='-', marker='s')
    
    ax1.set_title("Cross-Entropy Loss Comparison", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Epochs", fontsize=11, labelpad=10)
    ax1.set_ylabel("Loss", fontsize=11, labelpad=10)
    ax1.grid(color='#444444', linestyle=':', linewidth=0.5)
    ax1.legend(frameon=True, facecolor='#222222', edgecolor='#444444')
    
    # Plot Perplexities
    ax2.plot(epochs_lstm, lstm_log['train_ppl'], label='LSTM Train', color=c_lstm_train, linestyle='--', marker='o')
    ax2.plot(epochs_lstm, lstm_log['val_ppl'], label='LSTM Val', color=c_lstm_val, linestyle='-', marker='s')
    ax2.plot(epochs_trans, transformer_log['train_ppl'], label='Transformer Train', color=c_trans_train, linestyle='--', marker='o')
    ax2.plot(epochs_trans, transformer_log['val_ppl'], label='Transformer Val', color=c_trans_val, linestyle='-', marker='s')
    
    ax2.set_title("Perplexity (PPL) Comparison", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Epochs", fontsize=11, labelpad=10)
    ax2.set_ylabel("PPL", fontsize=11, labelpad=10)
    ax2.grid(color='#444444', linestyle=':', linewidth=0.5)
    ax2.legend(frameon=True, facecolor='#222222', edgecolor='#444444')
    
    # Premium layout spacing
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, facecolor='#121212')
    plt.close()
    print(f"Comparative training curves saved to {save_path}")
