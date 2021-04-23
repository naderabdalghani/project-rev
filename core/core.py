import os

import pandas as pd
from transformers import AutoTokenizer

from preprocessing import ConversationDataset
from utilities.config import TOKENIZER_NAME, CACHE_DIR, SPECIAL_TOKENS_DICT, OUTPUT_DIR


def get_response(user_utterance):
    return "Hey!"


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=CACHE_DIR)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    dataset = ConversationDataset(tokenizer)



