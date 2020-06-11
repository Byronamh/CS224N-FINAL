import os

CSV_KEYS = {
    'WORD': 'word',
    'TAG': 'tag',
    'POS': 'pos',
    'ID': 'id'
}

VERBOSE = True

VERBOSE_TABLE_FORMAT_STRING = "{:10} {:5}: ({})"

RANDOM_SEED = 2018
MAX_LENGTH = 50
BATCH_SIZE = 16
PADDING = "__PAD__"
OUTPUT_PATH = './outputs'
CHECKPOINT_PATH = './checkpoints'
checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
