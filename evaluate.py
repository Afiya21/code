# evaluate.py

import os
import argparse
import csv
import math
import torch
import torch.nn as nn
from tqdm import tqdm

import src.config as config
from src.tokenizer import CustomTokenizer
from src.dataset import get_dataloader
from src.model import LSTMSeq2Seq, TransformerSeq2Seq
from src.utils import compute_bleu, compute_rouge, compute_perplexity
from train import greedy_decode_lstm, greedy_decode_transformer

def evaluate_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = CustomTokenizer(config.TOKENIZER_PATH)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    
    # Load test dataloader
    test_loader = get_dataloader(
        split="test",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        size_limit=args.test_size,
        shuffle=False
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    # Dictionary to collect results for comparison table
    results = {}
    
    # Models to evaluate
    model_configs = [
        {
            "type": "lstm_seq2seq",
            "name": "LSTM Seq2Seq with Attention (Baseline)",
            "checkpoint": os.path.join(config.MODELS_DIR, "best_model_lstm_seq2seq.pt")
        },
        {
            "type": "transformer_seq2seq",
            "name": "Transformer Seq2Seq from Scratch (Main)",
            "checkpoint": os.path.join(config.MODELS_DIR, "best_model_transformer_seq2seq.pt")
        }
    ]
    
    # We will collect sample predictions for qualitative comparison
    samples_to_log = []
    num_samples_to_log = 5
    
    for m_cfg in model_configs:
        m_type = m_cfg["type"]
        checkpoint_path = m_cfg["checkpoint"]
        
        if not os.path.exists(checkpoint_path):
            print(f"\n[Warning] Checkpoint not found for {m_cfg['name']} at {checkpoint_path}. Skipping evaluation.")
            continue
            
        print(f"\nEvaluating model: {m_cfg['name']}...")
        
        # Instantiate model
        if m_type == "lstm_seq2seq":
            model = LSTMSeq2Seq(
                vocab_size=tokenizer.vocab_size,
                d_model=config.D_MODEL,
                pad_id=tokenizer.pad_id,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                dropout=0.0 # No dropout during evaluation
            )
        else:
            model = TransformerSeq2Seq(
                vocab_size=tokenizer.vocab_size,
                d_model=config.D_MODEL,
                nhead=config.NHEAD,
                num_encoder_layers=config.NUM_ENCODER_LAYERS,
                num_decoder_layers=config.NUM_DECODER_LAYERS,
                dim_feedforward=config.DIM_FEEDFORWARD,
                pad_id=tokenizer.pad_id,
                dropout=0.0
            )
            
        # Load weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        total_loss = 0.0
        steps = 0
        
        references_bleu = []
        hypotheses_bleu = []
        references_text = []
        hypotheses_text = []
        
        # Track predictions for qualitative reporting
        m_samples = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                code_ids = batch["code_ids"].to(device)
                code_padding_mask = batch["code_padding_mask"].to(device)
                summary_ids = batch["summary_ids"].to(device)
                summary_padding_mask = batch["summary_padding_mask"].to(device)
                
                raw_code = batch["raw_code"]
                raw_docstring = batch["raw_docstring"]
                
                # Compute Loss
                if m_type == "lstm_seq2seq":
                    outputs = model(code_ids, code_padding_mask, summary_ids, teacher_forcing_ratio=0.0)
                    loss = criterion(
                        outputs[:, 1:].reshape(-1, tokenizer.vocab_size),
                        summary_ids[:, 1:].reshape(-1)
                    )
                    
                    # Generate predictions
                    preds = greedy_decode_lstm(model, code_ids, code_padding_mask, tokenizer, device)
                else:
                    dec_input = summary_ids[:, :-1]
                    dec_padding_mask = summary_padding_mask[:, :-1]
                    targets = summary_ids[:, 1:]
                    
                    outputs = model(code_ids, code_padding_mask, dec_input, dec_padding_mask)
                    loss = criterion(
                        outputs.reshape(-1, tokenizer.vocab_size),
                        targets.reshape(-1)
                    )
                    
                    # Generate predictions
                    preds = greedy_decode_transformer(model, code_ids, code_padding_mask, tokenizer, device)
                    
                total_loss += loss.item()
                steps += 1
                
                # Process lists
                for i in range(len(raw_docstring)):
                    ref_str = raw_docstring[i]
                    pred_str = preds[i]
                    
                    references_text.append(ref_str)
                    hypotheses_text.append(pred_str)
                    
                    # split to tokens for BLEU
                    references_bleu.append(ref_str.lower().split())
                    hypotheses_bleu.append(pred_str.lower().split())
                    
                    # Keep track of a few samples
                    if len(m_samples) < num_samples_to_log:
                        m_samples.append({
                            "code": raw_code[i],
                            "ref": ref_str,
                            "pred": pred_str
                        })
                        
        test_loss = total_loss / steps
        test_ppl = compute_perplexity(test_loss)
        
        print("Computing BLEU score...")
        test_bleu = compute_bleu(references_bleu, hypotheses_bleu)
        
        print("Computing ROUGE score...")
        rouge_scores = compute_rouge(references_text, hypotheses_text)
        
        results[m_type] = {
            "name": m_cfg["name"],
            "loss": test_loss,
            "ppl": test_ppl,
            "bleu4": test_bleu,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "samples": m_samples
        }
        
        print(f"Results for {m_cfg['name']}:")
        print(f"  Loss: {test_loss:.4f} | Perplexity: {test_ppl:.2f}")
        print(f"  BLEU-4: {test_bleu:.2f}")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.2f} | ROUGE-2: {rouge_scores['rouge2']:.2f} | ROUGE-L: {rouge_scores['rougeL']:.2f}")
        
    if not results:
        print("No models evaluated. Exiting.")
        return
        
    # Write to results/metrics.csv
    csv_path = os.path.join(config.RESULTS_DIR, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "loss", "perplexity", "bleu4", "rouge1", "rouge2", "rougeL"])
        for m_type, metrics in results.items():
            writer.writerow([
                m_type,
                f"{metrics['loss']:.4f}",
                f"{metrics['ppl']:.2f}",
                f"{metrics['bleu4']:.2f}",
                f"{metrics['rouge1']:.2f}",
                f"{metrics['rouge2']:.2f}",
                f"{metrics['rougeL']:.2f}"
            ])
    print(f"Metrics CSV saved to {csv_path}")
    
    # Write to results/comparison_table.md
    md_path = os.path.join(config.RESULTS_DIR, "comparison_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Performance Comparison\n\n")
        f.write("| Model Architecture | Test Loss | Test Perplexity (PPL) | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m_type, m in results.items():
            f.write(f"| {m['name']} | {m['loss']:.4f} | {m['ppl']:.2f} | {m['bleu4']:.2f}% | {m['rouge1']:.2f}% | {m['rouge2']:.2f}% | {m['rougeL']:.2f}% |\n")
    print(f"Comparison table Markdown saved to {md_path}")
    
    # Write sample predictions to results/sample_predictions.txt
    txt_path = os.path.join(config.RESULTS_DIR, "sample_predictions.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("QUALITATIVE COMPARISON: EXAMPLE CODE SUMMARIES GENERATED BY MODELS\n")
        f.write("========================================================================\n\n")
        
        # Align samples
        for i in range(num_samples_to_log):
            f.write(f"--- Example {i+1} ---\n")
            
            # Print code (just first 15 lines if too long)
            code_lines = ""
            if "lstm_seq2seq" in results:
                code_lines = results["lstm_seq2seq"]["samples"][i]["code"]
            elif "transformer_seq2seq" in results:
                code_lines = results["transformer_seq2seq"]["samples"][i]["code"]
                
            truncated_code = "\n".join(code_lines.split("\n")[:15])
            if len(code_lines.split("\n")) > 15:
                truncated_code += "\n# ... [code truncated for length]"
                
            f.write("Code Snippet:\n")
            f.write("```python\n")
            f.write(truncated_code + "\n")
            f.write("```\n\n")
            
            # Print reference docstring
            ref_str = ""
            if "lstm_seq2seq" in results:
                ref_str = results["lstm_seq2seq"]["samples"][i]["ref"]
            elif "transformer_seq2seq" in results:
                ref_str = results["transformer_seq2seq"]["samples"][i]["ref"]
            f.write(f"Ground-Truth Docstring:\n  {ref_str}\n\n")
            
            # Print model predictions
            if "lstm_seq2seq" in results:
                pred = results["lstm_seq2seq"]["samples"][i]["pred"]
                f.write(f"LSTM Seq2Seq prediction:\n  {pred}\n")
            if "transformer_seq2seq" in results:
                pred = results["transformer_seq2seq"]["samples"][i]["pred"]
                f.write(f"Transformer Seq2Seq prediction:\n  {pred}\n")
                
            f.write("\n" + "="*80 + "\n\n")
            
    print(f"Sample qualitative predictions saved to {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PyTorch Code Summarizers.")
    parser.add_argument("--test_size", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    args = parser.parse_args()
    evaluate_models(args)
