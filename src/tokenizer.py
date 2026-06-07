# src/tokenizer.py

import os
from tokenizers import Tokenizer
import src.config as config

class CustomTokenizer:
    def __init__(self, tokenizer_path=config.TOKENIZER_PATH):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer file not found at {tokenizer_path}. "
                "Please run train_tokenizer.py first."
            )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Get special token IDs
        self.bos_id = self.tokenizer.token_to_id(config.BOS_TOKEN)
        self.pad_id = self.tokenizer.token_to_id(config.PAD_TOKEN)
        self.eos_id = self.tokenizer.token_to_id(config.EOS_TOKEN)
        self.unk_id = self.tokenizer.token_to_id(config.UNK_TOKEN)
        self.sep_id = self.tokenizer.token_to_id(config.SEP_TOKEN)
        
        self.special_ids = {self.bos_id, self.pad_id, self.eos_id, self.unk_id, self.sep_id}
        
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def encode_code(self, code_str, max_len=config.MAX_CODE_LEN):
        """
        Encodes code snippet.
        Adds <s> at start and </s> at end. Truncates to max_len.
        """
        encoded = self.tokenizer.encode(code_str)
        ids = encoded.ids
        
        # Truncate to leave room for <s> and </s>
        if len(ids) > max_len - 2:
            ids = ids[:max_len - 2]
            
        # Add special tokens
        return [self.bos_id] + ids + [self.eos_id]
        
    def encode_summary(self, summary_str, max_len=config.MAX_SUMMARY_LEN):
        """
        Encodes natural language summary.
        Adds <s> at start and </s> at end. Truncates to max_len.
        """
        encoded = self.tokenizer.encode(summary_str)
        ids = encoded.ids
        
        if len(ids) > max_len - 2:
            ids = ids[:max_len - 2]
            
        return [self.bos_id] + ids + [self.eos_id]

    def decode(self, ids, skip_special_tokens=True):
        """
        Decodes a list of token IDs back to a string.
        """
        if skip_special_tokens:
            filtered_ids = [token_id for token_id in ids if token_id not in self.special_ids]
        else:
            filtered_ids = ids
            
        return self.tokenizer.decode(filtered_ids)

    def id_to_token(self, token_id):
        return self.tokenizer.id_to_token(token_id)

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)
