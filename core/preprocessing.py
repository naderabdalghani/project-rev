import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset

from utilities.config import CACHE_DIR, MODEL_TYPE, OVERWRITE_CACHE


def construct_dialogue(row, tokenizer):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv


class ConversationDataset(Dataset):
    def __init__(self, tokenizer, df):
        block_size = tokenizer.model_max_length
        cached_features_file = os.path.join(CACHE_DIR, MODEL_TYPE + "_cached_lm_" + str(block_size))
        if os.path.exists(cached_features_file) and not OVERWRITE_CACHE:
            print("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as cached_file:
                self.dialogues = pickle.load(cached_file)
        else:
            print("Creating features from dataset file at %s", CACHE_DIR)
            self.dialogues = []
            for _, row in df.iterrows():
                dialogue = construct_dialogue(row, tokenizer)
                if len(dialogue) > block_size:
                    continue
                self.dialogues.append(dialogue)

            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.dialogues, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):
        return torch.tensor(self.dialogues[item], dtype=torch.long)
