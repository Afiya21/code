# src/config.py

import os

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.json")

# Ensure folders exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Vocabulary & Tokenizer Parameters
VOCAB_SIZE = 16000
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"

SPECIAL_TOKENS = [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN]

# Sequence lengths (Strictly locked constraints)
MAX_CODE_LEN = 256
MAX_SUMMARY_LEN = 64

# Training hyperparameters (Strictly locked constraints)
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Model Architecture Dimensions
D_MODEL = 64
NHEAD = 2
NUM_ENCODER_LAYERS = 1
NUM_DECODER_LAYERS = 1
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
