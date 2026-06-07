# summarize.py

import os
import argparse
import torch
import math

import src.config as config
from src.tokenizer import CustomTokenizer
from src.model import LSTMSeq2Seq, TransformerSeq2Seq
from train import greedy_decode_lstm, greedy_decode_transformer

def summarize_code(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Fetch code snippet
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: Code file not found at {args.file}")
            return
        with open(args.file, "r", encoding="utf-8") as f:
            code_str = f.read()
    elif args.code:
        code_str = args.code
    else:
        # Provide a default fallback snippet to show it works
        code_str = (
            "def add_integers(x, y):\n"
            "    \"\"\"Calculates sum\"\"\"\n"
            "    return int(x) + int(y)"
        )
        print("No source code input provided. Running on default sample snippet:")
        print(code_str)
        print("-" * 50)

    # 2. Load custom tokenizer
    tokenizer = CustomTokenizer(config.TOKENIZER_PATH)
    
    # 3. Load model checkpoint
    if args.model_type == "lstm_seq2seq":
        checkpoint_path = os.path.join(config.MODELS_DIR, "best_model_lstm_seq2seq.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Model checkpoint not found at {checkpoint_path}. Please train the model first.")
            return
            
        model = LSTMSeq2Seq(
            vocab_size=tokenizer.vocab_size,
            d_model=config.D_MODEL,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            dropout=0.0
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        
        # Preprocess input
        encoded_ids = tokenizer.encode_code(code_str)
        src = torch.tensor([encoded_ids], dtype=torch.long, device=device)
        src_padding_mask = torch.zeros(1, len(encoded_ids), dtype=torch.bool, device=device)
        
        # Generate summary
        pred_texts = greedy_decode_lstm(model, src, src_padding_mask, tokenizer, device)
        summary = pred_texts[0]
        
    elif args.model_type == "transformer_seq2seq":
        checkpoint_path = os.path.join(config.MODELS_DIR, "best_model_transformer_seq2seq.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Model checkpoint not found at {checkpoint_path}. Please train the model first.")
            return
            
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
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        
        # Preprocess input
        encoded_ids = tokenizer.encode_code(code_str)
        src = torch.tensor([encoded_ids], dtype=torch.long, device=device)
        src_padding_mask = torch.zeros(1, len(encoded_ids), dtype=torch.bool, device=device)
        
        # Generate summary
        pred_texts = greedy_decode_transformer(model, src, src_padding_mask, tokenizer, device)
        summary = pred_texts[0]
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    # Output formatting
    print("\n" + "=" * 50)
    print("Source Code Input:")
    print("=" * 50)
    print(code_str)
    print("=" * 50)
    print(f"Generated Summary ({args.model_type}):")
    print("=" * 50)
    print(summary if summary.strip() else "[No summary generated / empty sequence]")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Python code summaries using trained models.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["lstm_seq2seq", "transformer_seq2seq"],
        help="Trained model architecture to use: 'lstm_seq2seq' or 'transformer_seq2seq'"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--code", type=str, help="Python code snippet as a string")
    group.add_argument("--file", type=str, help="Path to Python code file")
    
    args = parser.parse_args()
    summarize_code(args)
