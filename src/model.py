# src/model.py

import torch
import torch.nn as nn
import math
import random

# ==============================================================================
# 1. Custom Mathematical Components (Built from Scratch)
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Q, K, V linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        # Input shapes: [batch_size, seq_len, d_model]
        batch_size = q.size(0)
        
        # 1. Project Q, K, V
        # Shape: [batch_size, seq_len, nhead, d_k] -> transpose to [batch_size, nhead, seq_len, d_k]
        q = self.q_linear(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Score
        # scores shape: [batch_size, nhead, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask should broadcast to [batch_size, nhead, seq_len_q, seq_len_k]
            # True values represent positions to mask out, filling them with large negative value
            scores = scores.masked_fill(mask, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 3. Multiply by values
        # output shape: [batch_size, nhead, seq_len_q, d_k]
        output = torch.matmul(attn_weights, v)
        
        # 4. Concat heads and project out
        # Shape: [batch_size, seq_len_q, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)  # Decoder hidden state projection
        self.U = nn.Linear(hidden_dim, hidden_dim)  # Encoder outputs projection
        self.v = nn.Linear(hidden_dim, 1, bias=False)  # Score projection
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None, precomputed_U=None):
        # decoder_hidden shape: [batch_size, hidden_dim]
        # encoder_outputs shape: [batch_size, seq_len_src, hidden_dim]
        # mask shape: [batch_size, seq_len_src] (True for pad tokens, False for real)
        
        dec_h = decoder_hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Calculate alignment energy: [batch_size, seq_len_src, hidden_dim]
        # Use precomputed encoder projections if available to avoid redundant computations inside target loop
        U_proj = precomputed_U if precomputed_U is not None else self.U(encoder_outputs)
        energy = torch.tanh(self.W(dec_h) + U_proj)
        
        # Compute scores: [batch_size, seq_len_src]
        scores = self.v(energy).squeeze(2)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Calculate context vector: [batch_size, hidden_dim]
        # weights: [batch_size, 1, seq_len_src]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attn_weights

# ==============================================================================
# 2. LSTM Sequence-to-Sequence with Attention (lstm_seq2seq)
# ==============================================================================

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Bidirectional LSTM (outputs bidirectional representations)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))
        # outputs shape: [batch_size, seq_len, hidden_dim]
        # hidden/cell shapes: [2, batch_size, hidden_dim // 2]
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate final hidden and cell states of forward and backward passes
        # hidden: [batch_size, hidden_dim]
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        cell = torch.cat((cell[0], cell[1]), dim=1)
        
        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(hidden_dim)
        
        # Takes embedded target token + attention context vector
        self.lstm = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(embedding_dim + hidden_dim + hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, target_input, hidden, cell, encoder_outputs, src_padding_mask=None, precomputed_U=None):
        # target_input shape: [batch_size]
        # hidden, cell shape: [batch_size, hidden_dim]
        # encoder_outputs shape: [batch_size, seq_len_src, hidden_dim]
        
        embedded = self.dropout(self.embedding(target_input))  # [batch_size, embedding_dim]
        
        # Calculate attention context vector, using precomputed encoder projections if available
        context, attn_weights = self.attention(hidden, encoder_outputs, src_padding_mask, precomputed_U)
        
        # LSTM input: concat target input and context
        lstm_input = torch.cat((embedded, context), dim=1)
        
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        
        # Project output using embedding, context, and hidden state
        # Output prediction shape: [batch_size, vocab_size]
        output = self.fc_out(torch.cat((hidden, context, embedded), dim=1))
        
        return output, hidden, cell, attn_weights


class LSTMSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, pad_id, bos_id, eos_id, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        self.encoder = LSTMEncoder(vocab_size, d_model, d_model, dropout)
        self.decoder = LSTMDecoder(vocab_size, d_model, d_model, dropout)
        
    def forward(self, src, src_padding_mask, tgt, teacher_forcing_ratio=1.0):
        # src: [batch_size, seq_len_src]
        # src_padding_mask: [batch_size, seq_len_src]
        # tgt: [batch_size, seq_len_tgt]
        batch_size = src.size(0)
        max_len = tgt.size(1)
        
        # Run Encoder
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Precompute decoder projection of encoder outputs once to avoid recalculating in step loop
        precomputed_U = self.decoder.attention.U(encoder_outputs)
        
        # Initialize output predictions tensor
        outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=src.device)
        
        # The first input to the decoder is the <bos> token
        decoder_input = tgt[:, 0]
        
        for t in range(1, max_len):
            output, hidden, cell, _ = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, src_padding_mask, precomputed_U
            )
            outputs[:, t] = output
            
            # Scheduled Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            decoder_input = tgt[:, t] if teacher_force else top1
            
        return outputs

# ==============================================================================
# 3. Transformer Sequence-to-Sequence (transformer_seq2seq)
# ==============================================================================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # Self-attention with residual
        attn_out = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_out))
        
        # FFN with residual
        ff_out = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_out))
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Target self-attention (causal masking)
        self_attn_out = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(self_attn_out))
        
        # Cross-attention (query tgt, key/value memory)
        cross_attn_out = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))
        
        # FFN
        ff_out = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff_out))
        return tgt


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        
        # Word Embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
        # Sinusoidal Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len=1000, dropout=dropout)
        
        # Stack of custom encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Stack of custom decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection head
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_causal_mask(self, sz, device):
        # True means masked out (future tokens), False means attendable
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        # Shape: [1, 1, sz, sz] for broadcasting
        return mask.unsqueeze(0).unsqueeze(1)
        
    def forward(self, src, src_padding_mask, tgt, tgt_padding_mask):
        # src: [batch_size, seq_len_src]
        # src_padding_mask: [batch_size, seq_len_src] (True for pad, False for real)
        # tgt: [batch_size, seq_len_tgt]
        # tgt_padding_mask: [batch_size, seq_len_tgt]
        
        batch_size = src.size(0)
        seq_len_src = src.size(1)
        seq_len_tgt = tgt.size(1)
        device = src.device
        
        # Embed and add Positional Encodings
        # Scale embeddings by sqrt(d_model) as in standard Transformer paper
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Create attention masks:
        # Encoder self-attn padding mask: [batch_size, 1, 1, seq_len_src]
        src_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)
        
        # Decoder self-attn causal + padding mask:
        # Causal mask: [1, 1, seq_len_tgt, seq_len_tgt]
        causal_mask = self.generate_causal_mask(seq_len_tgt, device)
        # Padding mask: [batch_size, 1, 1, seq_len_tgt]
        tgt_pad_mask = tgt_padding_mask.unsqueeze(1).unsqueeze(2)
        # Combine (logical OR: if either pad OR future, mask it)
        tgt_mask = causal_mask | tgt_pad_mask
        
        # Decoder cross-attn key padding mask (mask out encoder pads): [batch_size, 1, 1, seq_len_src]
        memory_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)
        
        # Run custom encoder stack
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)
            
        # Run custom decoder stack
        out = tgt_emb
        for layer in self.decoder_layers:
            out = layer(out, memory, tgt_mask, memory_mask)
            
        # Project outputs
        return self.fc_out(out)
