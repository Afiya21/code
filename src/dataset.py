# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import src.config as config
from src.tokenizer import CustomTokenizer

class CodeSummarizationDataset(Dataset):
    def __init__(self, split, tokenizer, size_limit=None):
        """
        split: "train", "validation", or "test"
        tokenizer: CustomTokenizer instance
        size_limit: integer or None, limit dataset size for experiments
        """
        self.tokenizer = tokenizer
        
        print(f"Loading split '{split}' from CodeXGLUE python dataset...")
        # Note: CodeXGLUE dataset name on HF is "google/code_x_glue_ct_code_to_text"
        self.dataset = load_dataset("google/code_x_glue_ct_code_to_text", "python", split=split)
        
        # Enforce split sizes if specified
        if size_limit is not None:
            self.dataset = self.dataset.select(range(min(size_limit, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} examples for split '{split}'.")
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        code = item["code"]
        docstring = item["docstring"]
        
        # Encode source (code) and target (summary)
        code_ids = self.tokenizer.encode_code(code)
        summary_ids = self.tokenizer.encode_summary(docstring)
        
        return {
            "code_ids": code_ids,
            "summary_ids": summary_ids,
            "raw_code": code,
            "raw_docstring": docstring
        }

def make_collate_fn(pad_id):
    def collate_fn(batch):
        code_ids_list = [item["code_ids"] for item in batch]
        summary_ids_list = [item["summary_ids"] for item in batch]
        raw_code = [item["raw_code"] for item in batch]
        raw_docstring = [item["raw_docstring"] for item in batch]
        
        # Determine maximum sequence length in this batch
        max_code_len = max(len(x) for x in code_ids_list)
        max_summary_len = max(len(x) for x in summary_ids_list)
        
        padded_code = []
        code_padding_mask = []
        for ids in code_ids_list:
            pad_len = max_code_len - len(ids)
            padded_code.append(ids + [pad_id] * pad_len)
            # Mask is True for padding elements, False for real tokens (PyTorch standard)
            code_padding_mask.append([False] * len(ids) + [True] * pad_len)
            
        padded_summary = []
        summary_padding_mask = []
        for ids in summary_ids_list:
            pad_len = max_summary_len - len(ids)
            padded_summary.append(ids + [pad_id] * pad_len)
            summary_padding_mask.append([False] * len(ids) + [True] * pad_len)
            
        return {
            "code_ids": torch.tensor(padded_code, dtype=torch.long),
            "code_padding_mask": torch.tensor(code_padding_mask, dtype=torch.bool),
            "summary_ids": torch.tensor(padded_summary, dtype=torch.long),
            "summary_padding_mask": torch.tensor(summary_padding_mask, dtype=torch.bool),
            "raw_code": raw_code,
            "raw_docstring": raw_docstring
        }
    return collate_fn

def get_dataloader(split, tokenizer, batch_size=config.BATCH_SIZE, size_limit=None, shuffle=False):
    dataset = CodeSummarizationDataset(split, tokenizer, size_limit=size_limit)
    collate_fn = make_collate_fn(tokenizer.pad_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return dataloader
