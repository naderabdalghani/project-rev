import pandas as pd
from transformers import AutoTokenizer

from preprocessing import ConversationDataset
from utilities.config import TOKENIZER_NAME, CACHE_DIR


def get_response(user_utterance):
    return "Hey!"


if __name__ == '__main__':
    df = pd.read_csv('../output/final_es_conv.csv')
    df = df.dropna()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=CACHE_DIR)
    dataset = ConversationDataset(tokenizer, df)
    x = 5
