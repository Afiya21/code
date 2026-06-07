# test_model.py

import torch
from src.model import LSTMSeq2Seq, TransformerSeq2Seq

def test_models():
    print("==================================================")
    print("Starting Model Architecture Verification Tests")
    print("==================================================")
    
    # Mock parameters
    vocab_size = 16000
    d_model = 256
    nhead = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 512
    pad_id = 1
    bos_id = 2
    eos_id = 3
    
    batch_size = 4
    seq_len_src = 20
    seq_len_tgt = 15
    
    # Generate dummy input tensors
    # Token IDs between 0 and vocab_size-1
    src = torch.randint(0, vocab_size, (batch_size, seq_len_src))
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len_tgt))
    
    # Generate random padding masks (True for padded, False for active)
    src_padding_mask = torch.zeros(batch_size, seq_len_src, dtype=torch.bool)
    src_padding_mask[:, -3:] = True # Mock last 3 tokens as pad
    
    tgt_padding_mask = torch.zeros(batch_size, seq_len_tgt, dtype=torch.bool)
    tgt_padding_mask[:, -2:] = True # Mock last 2 tokens as pad
    
    print("\n[Test 1] Verifying LSTM Seq2Seq Model...")
    lstm_model = LSTMSeq2Seq(
        vocab_size=vocab_size,
        d_model=d_model,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        dropout=0.1
    )
    
    # Forward pass
    lstm_outputs = lstm_model(src, src_padding_mask, tgt, teacher_forcing_ratio=1.0)
    print(f"LSTM Output shape: {lstm_outputs.shape} (Expected: [{batch_size}, {seq_len_tgt}, {vocab_size}])")
    assert lstm_outputs.shape == (batch_size, seq_len_tgt, vocab_size), "LSTM output shape mismatch!"
    
    # Test gradients / backward pass
    loss = lstm_outputs.sum()
    loss.backward()
    print("LSTM backward pass completed. Gradients checked successfully.")
    
    print("\n[Test 2] Verifying Transformer Seq2Seq Model...")
    transformer_model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        pad_id=pad_id,
        dropout=0.1
    )
    
    # Forward pass
    transformer_outputs = transformer_model(src, src_padding_mask, tgt, tgt_padding_mask)
    print(f"Transformer Output shape: {transformer_outputs.shape} (Expected: [{batch_size}, {seq_len_tgt}, {vocab_size}])")
    assert transformer_outputs.shape == (batch_size, seq_len_tgt, vocab_size), "Transformer output shape mismatch!"
    
    # Test gradients / backward pass
    loss_transformer = transformer_outputs.sum()
    loss_transformer.backward()
    print("Transformer backward pass completed. Gradients checked successfully.")
    
    print("\n==================================================")
    print("All Model Architecture Verification Tests Passed!")
    print("==================================================")

if __name__ == "__main__":
    test_models()
