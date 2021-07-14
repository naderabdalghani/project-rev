import logging
import os
import pickle
import torch
from torch.utils.data import Dataset
from utilities.config import CACHE_DIR, MODEL_TYPE, OVERWRITE_CACHE, DATASET_FILENAME, OUTPUT_DIR, BOT_TOKEN

logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    def __init__(self, tokenizer):
        block_size = tokenizer.model_max_length
        cached_features_file = os.path.join(CACHE_DIR, MODEL_TYPE + "_cached_features_" + str(block_size))
        if os.path.exists(cached_features_file) and not OVERWRITE_CACHE:
            logger.info("Loading features from cached file {}".format(cached_features_file))
            with open(cached_features_file, "rb") as cached_file:
                self.utterances, self.responses = pickle.load(cached_file)
        else:
            logger.info("Creating features from dataset file at {}".format(CACHE_DIR))
            self.utterances = []
            self.responses = []
            with open(os.path.join(OUTPUT_DIR, DATASET_FILENAME)) as f:
                file_content = f.readlines()
                i = 0
                while i < len(file_content):
                    if file_content[i].startswith(tokenizer.bos_token):
                        j = i
                        while j < len(file_content) and file_content[j].startswith(tokenizer.bos_token):
                            j += 1
                        if j < len(file_content):
                            bot_response = file_content[j].removeprefix(BOT_TOKEN).strip()
                            k = j + 1
                            while k < len(file_content) and file_content[k].startswith(BOT_TOKEN):
                                bot_response += ' ' + file_content[k].removeprefix(BOT_TOKEN).strip()
                                k += 1
                            self.utterances.append(tokenizer.encode(file_content[j - 1]
                                                                    .removeprefix(tokenizer.bos_token).strip(),
                                                                    truncation=True))
                            self.responses.append(tokenizer.encode(bot_response, truncation=True))
                            i = k
                            continue
                    i += 1
            logger.info("Saving features into cached file {}".format(cached_features_file))
            with open(cached_features_file, "wb") as handle:
                pickle.dump((self.utterances, self.responses), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        return torch.tensor(self.utterances[index], dtype=torch.long), \
               torch.tensor(self.responses[index], dtype=torch.long)
