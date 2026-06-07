# train_tokenizer.py

import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import src.config as config

def train_custom_tokenizer():
    print("Loading CodeXGLUE python dataset split 'train'...")
    # Load dataset
    dataset = load_dataset("google/code_x_glue_ct_code_to_text", "python", split="train")
    
    # We will write the training data corpus to a file
    corpus_path = os.path.join(config.DATA_DIR, "corpus.txt")
    print(f"Extracting code and docstrings to build training corpus at {corpus_path}...")
    
    # Write code and summaries to file
    with open(corpus_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(dataset):
            code = item["code"]
            docstring = item["docstring"]
            # Write both to corpus so BPE learns representations for both code tokens and summary words
            f.write(code + "\n")
            f.write(docstring + "\n")
            
            # Print progress every 50k items
            if (idx + 1) % 50000 == 0:
                print(f"Written {idx + 1} items to corpus file...")

    print("Corpus file written. Training ByteLevelBPETokenizer...")
    
    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Train the tokenizer
    tokenizer.train(
        files=[corpus_path],
        vocab_size=config.VOCAB_SIZE,
        min_frequency=2,
        special_tokens=config.SPECIAL_TOKENS
    )
    
    # Save the tokenizer JSON config to models/tokenizer.json
    print(f"Saving trained tokenizer to {config.TOKENIZER_PATH}...")
    tokenizer.save(config.TOKENIZER_PATH)
    
    # Print some examples
    print("Tokenizer trained successfully! Testing encoding/decoding:")
    test_code = "def calculate_sum(a, b):\n    return a + b"
    encoded = tokenizer.encode(test_code)
    print("Original text:", test_code)
    print("Encoded IDs:", encoded.ids)
    print("Decoded text:", tokenizer.decode(encoded.ids))
    print(f"Vocab size check: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    train_custom_tokenizer()
