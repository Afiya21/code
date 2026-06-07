# train.py

import os
import argparse
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import src.config as config
from src.tokenizer import CustomTokenizer
from src.dataset import get_dataloader
from src.model import LSTMSeq2Seq, TransformerSeq2Seq
from src.utils import compute_bleu, compute_perplexity, plot_training_curves

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==============================================================================
# Greedy Decoding Helpers (Fast Batch Processing for Validation BLEU Monitor)
# ==============================================================================

@torch.no_grad()
def greedy_decode_lstm(model, src, src_padding_mask, tokenizer, device, max_len=config.MAX_SUMMARY_LEN):
    model.eval()
    batch_size = src.size(0)
    encoder_outputs, (hidden, cell) = model.encoder(src)
    
    # Precompute encoder projection for Bahdanau Attention to save substantial computation time in loop
    precomputed_U = model.decoder.attention.U(encoder_outputs)
    
    decoder_input = torch.full((batch_size,), tokenizer.bos_id, dtype=torch.long, device=device)
    decoded_ids = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for t in range(1, max_len):
        output, hidden, cell, _ = model.decoder(
            decoder_input, hidden, cell, encoder_outputs, src_padding_mask, precomputed_U
        )
        top1 = output.argmax(dim=1)
        
        for i in range(batch_size):
            if not finished[i]:
                if top1[i].item() == tokenizer.eos_id:
                    finished[i] = True
                else:
                    decoded_ids[i].append(top1[i].item())
                    
        decoder_input = top1
        if finished.all():
            break
            
    return [tokenizer.decode(ids) for ids in decoded_ids]


@torch.no_grad()
def greedy_decode_transformer(model, src, src_padding_mask, tokenizer, device, max_len=config.MAX_SUMMARY_LEN):
    model.eval()
    batch_size = src.size(0)
    
    # 1. Run custom encoder stack
    src_emb = model.pos_encoder(model.src_embedding(src) * math.sqrt(model.d_model))
    src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)
    memory = src_emb
    for layer in model.encoder_layers:
        memory = layer(memory, src_mask)
        
    # 2. Iteratively predict target tokens
    tgt = torch.full((batch_size, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len - 1):
        tgt_pad_mask = torch.zeros_like(tgt, dtype=torch.bool, device=device)  # no padding inside decoded sequence
        tgt_emb = model.pos_decoder(model.tgt_embedding(tgt) * math.sqrt(model.d_model))
        
        causal_mask = model.generate_causal_mask(tgt.size(1), device)
        tgt_mask = causal_mask | tgt_pad_mask.unsqueeze(1).unsqueeze(2)
        memory_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)
        
        out = tgt_emb
        for layer in model.decoder_layers:
            out = layer(out, memory, tgt_mask, memory_mask)
            
        logits = model.fc_out(out[:, -1, :])
        next_tokens = logits.argmax(dim=-1)
        
        tgt = torch.cat((tgt, next_tokens.unsqueeze(1)), dim=1)
        
        for i in range(batch_size):
            if next_tokens[i].item() == tokenizer.eos_id:
                finished[i] = True
                
        if finished.all():
            break
            
    decoded_texts = []
    for i in range(batch_size):
        ids = tgt[i].tolist()
        if len(ids) > 0 and ids[0] == tokenizer.bos_id:
            ids = ids[1:]
        if tokenizer.eos_id in ids:
            ids = ids[:ids.index(tokenizer.eos_id)]
        decoded_texts.append(tokenizer.decode(ids))
        
    return decoded_texts

# ==============================================================================
# Main Training loop
# ==============================================================================

def train_model(args):
    # Ensure directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = CustomTokenizer(config.TOKENIZER_PATH)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # Load Dataloaders
    train_loader = get_dataloader(
        split="train",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        size_limit=args.train_size,
        shuffle=True
    )
    val_loader = get_dataloader(
        split="validation",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        size_limit=args.val_size,
        shuffle=False
    )
    
    # Create a small subset of 50 validation samples for fast BLEU monitoring
    val_subset_dataset = []
    val_subset_count = 50
    for batch in val_loader:
        for i in range(len(batch["raw_code"])):
            if len(val_subset_dataset) < val_subset_count:
                val_subset_dataset.append({
                    "code_ids": batch["code_ids"][i].unsqueeze(0).to(device),
                    "code_padding_mask": batch["code_padding_mask"][i].unsqueeze(0).to(device),
                    "raw_docstring": batch["raw_docstring"][i]
                })
            else:
                break
        if len(val_subset_dataset) >= val_subset_count:
            break
            
    print(f"Created validation monitor subset containing {len(val_subset_dataset)} samples.")
    
    # Model instantiation
    if args.model_type == "lstm_seq2seq":
        model = LSTMSeq2Seq(
            vocab_size=tokenizer.vocab_size,
            d_model=config.D_MODEL,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            dropout=config.DROPOUT
        )
        model_name = "LSTM Seq2Seq with Attention"
        checkpoint_path = os.path.join(config.MODELS_DIR, "best_model_lstm_seq2seq.pt")
    elif args.model_type == "transformer_seq2seq":
        model = TransformerSeq2Seq(
            vocab_size=tokenizer.vocab_size,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            pad_id=tokenizer.pad_id,
            dropout=config.DROPOUT
        )
        model_name = "Transformer Seq2Seq from Scratch"
        checkpoint_path = os.path.join(config.MODELS_DIR, "best_model_transformer_seq2seq.pt")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    model = model.to(device)
    print(f"Initialized {model_name} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Metrics logs
    history = {
        "train_loss": [], "train_ppl": [],
        "val_loss": [], "val_ppl": [], "val_bleu": []
    }
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        # ----------------------------------------------------------------------
        # Training Split
        # ----------------------------------------------------------------------
        model.train()
        total_train_loss = 0.0
        train_steps = 0
        
        # Scheduled Teacher Forcing Ratio for LSTM (0-indexed epoch)
        tf_ratio = max(0.5, 1.0 - (epoch * 0.05)) if args.model_type == "lstm_seq2seq" else 1.0
        if args.model_type == "lstm_seq2seq":
            print(f"Teacher Forcing Ratio: {tf_ratio:.2f}")
            
        train_pbar = tqdm(train_loader, desc="Training")
        for batch in train_pbar:
            code_ids = batch["code_ids"].to(device)
            code_padding_mask = batch["code_padding_mask"].to(device)
            summary_ids = batch["summary_ids"].to(device)
            summary_padding_mask = batch["summary_padding_mask"].to(device)
            
            optimizer.zero_grad()
            
            if args.model_type == "lstm_seq2seq":
                # LSTM forward
                outputs = model(code_ids, code_padding_mask, summary_ids, teacher_forcing_ratio=tf_ratio)
                # outputs: [batch_size, seq_len_tgt, vocab_size]
                # CrossEntropy expects inputs [batch_size * seq_len, vocab_size], targets [batch_size * seq_len]
                # Skip the <bos> token in outputs and target
                loss = criterion(
                    outputs[:, 1:].reshape(-1, tokenizer.vocab_size),
                    summary_ids[:, 1:].reshape(-1)
                )
            else:
                # Transformer forward
                # Decoder input: all target tokens except <eos>
                dec_input = summary_ids[:, :-1]
                dec_padding_mask = summary_padding_mask[:, :-1]
                
                # Target output (ground truth): all target tokens except <bos>
                targets = summary_ids[:, 1:]
                
                outputs = model(code_ids, code_padding_mask, dec_input, dec_padding_mask)
                loss = criterion(
                    outputs.reshape(-1, tokenizer.vocab_size),
                    targets.reshape(-1)
                )
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            train_steps += 1
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        epoch_train_loss = total_train_loss / train_steps
        epoch_train_ppl = compute_perplexity(epoch_train_loss)
        
        # ----------------------------------------------------------------------
        # Validation Split (Loss & Perplexity)
        # ----------------------------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                code_ids = batch["code_ids"].to(device)
                code_padding_mask = batch["code_padding_mask"].to(device)
                summary_ids = batch["summary_ids"].to(device)
                summary_padding_mask = batch["summary_padding_mask"].to(device)
                
                if args.model_type == "lstm_seq2seq":
                    # Disable teacher forcing entirely during validation
                    outputs = model(code_ids, code_padding_mask, summary_ids, teacher_forcing_ratio=0.0)
                    loss = criterion(
                        outputs[:, 1:].reshape(-1, tokenizer.vocab_size),
                        summary_ids[:, 1:].reshape(-1)
                    )
                else:
                    dec_input = summary_ids[:, :-1]
                    dec_padding_mask = summary_padding_mask[:, :-1]
                    targets = summary_ids[:, 1:]
                    
                    outputs = model(code_ids, code_padding_mask, dec_input, dec_padding_mask)
                    loss = criterion(
                        outputs.reshape(-1, tokenizer.vocab_size),
                        targets.reshape(-1)
                    )
                    
                total_val_loss += loss.item()
                val_steps += 1
                
        epoch_val_loss = total_val_loss / val_steps
        epoch_val_ppl = compute_perplexity(epoch_val_loss)
        
        # ----------------------------------------------------------------------
        # Validation BLEU Monitor (Evaluated on the fast 50-sample subset)
        # ----------------------------------------------------------------------
        refs = []
        hyps = []
        for sample in val_subset_dataset:
            code_ids = sample["code_ids"]
            code_padding_mask = sample["code_padding_mask"]
            ref_str = sample["raw_docstring"]
            
            if args.model_type == "lstm_seq2seq":
                pred_texts = greedy_decode_lstm(model, code_ids, code_padding_mask, tokenizer, device)
            else:
                pred_texts = greedy_decode_transformer(model, code_ids, code_padding_mask, tokenizer, device)
                
            pred_str = pred_texts[0]
            
            # Tokenize strings for BLEU computation
            refs.append(ref_str.lower().split())
            hyps.append(pred_str.lower().split())
            
        epoch_val_bleu = compute_bleu(refs, hyps)
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train PPL: {epoch_train_ppl:.2f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val PPL:   {epoch_val_ppl:.2f}")
        print(f"Val BLEU (50-sample monitor): {epoch_val_bleu:.2f}")
        
        # Log stats
        history["train_loss"].append(epoch_train_loss)
        history["train_ppl"].append(epoch_train_ppl)
        history["val_loss"].append(epoch_val_loss)
        history["val_ppl"].append(epoch_val_ppl)
        history["val_bleu"].append(epoch_val_bleu)
        
        # Checkpoint selection strictly based on lowest validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved to {checkpoint_path}!")
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve. Early stopping counter: {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}")
            
        # Early stopping check
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered. Training terminated.")
            break
            
    # Save training log JSON file
    log_path = os.path.join(config.RESULTS_DIR, f"history_{args.model_type}.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"History logs written to {log_path}")
    
    # Try to plot comparative curves if both history files are now present
    other_model = "transformer_seq2seq" if args.model_type == "lstm_seq2seq" else "lstm_seq2seq"
    other_log_path = os.path.join(config.RESULTS_DIR, f"history_{other_model}.json")
    
    if os.path.exists(other_log_path):
        with open(other_log_path, "r") as f:
            other_history = json.load(f)
            
        curves_path = os.path.join(config.RESULTS_DIR, "training_curves.png")
        if args.model_type == "lstm_seq2seq":
            plot_training_curves(history, other_history, curves_path)
        else:
            plot_training_curves(other_history, history, curves_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch Seq2Seq Code Summarizers.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["lstm_seq2seq", "transformer_seq2seq"],
        help="Model architecture to train: 'lstm_seq2seq' or 'transformer_seq2seq'"
    )
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=type(config.LEARNING_RATE), default=config.LEARNING_RATE, help="Learning rate")
    
    args = parser.parse_args()
    train_model(args)
