import logging
import os
import pickle
import torch
from torch.utils.data import Dataset
from utilities.config import CACHE_DIR, MODEL_TYPE, OVERWRITE_CACHE, DATASET_FILENAME, OUTPUT_DIR, DIALOGUE_SIZE


logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    def __init__(self, tokenizer):
        block_size = tokenizer.model_max_length
        cached_features_file = os.path.join(CACHE_DIR, MODEL_TYPE + "_cached_features_" + str(block_size))
        if os.path.exists(cached_features_file) and not OVERWRITE_CACHE:
            logger.info("Loading features from cached file {}".format(cached_features_file))
            with open(cached_features_file, "rb") as cached_file:
                self.dialogues = pickle.load(cached_file)
        else:
            logger.info("Creating features from dataset file at {}".format(CACHE_DIR))
            self.dialogues = []
            with open(os.path.join(OUTPUT_DIR, DATASET_FILENAME)) as f:
                file_content = f.readlines()
                for i in range(0, len(file_content), DIALOGUE_SIZE):
                    dialogue = file_content[i:min(i+DIALOGUE_SIZE, len(file_content))]
                    tokenized_dialogue = [tokenizer.encode(line) + [tokenizer.eos_token_id] for line in dialogue]
                    if len(tokenized_dialogue) > block_size:
                        continue
                    self.dialogues.append([token for line in tokenized_dialogue for token in line])
            logger.info("Saving features into cached file {}".format(cached_features_file))
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.dialogues, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, item):
        return torch.tensor(self.dialogues[item], dtype=torch.long)
