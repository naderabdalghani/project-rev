import glob
import logging
import os
import re

import torch
from torch.utils.data import random_split
from transformers import BlenderbotTokenizer, BlenderbotConfig, BlenderbotForConditionalGeneration
from .train import train, evaluate
from .preprocessing import ConversationDataset
from utilities.config import TOKENIZER_NAME, CACHE_DIR, OUTPUT_DIR, MODEL_NAME, \
    MODEL_CONFIG_NAME, DEVICE, LOCAL_RANK, N_GPUS, FP16, DO_TRAIN, EVAL_DATA_SPLIT_RATIO, DO_EVAL, \
    SAVED_INSTANCE_PREFIX, BAD_WORDS

logger = logging.getLogger(__name__)
saved_instance_path = None
loaded_model = None
loaded_tokenizer = None
chat_history = []
bad_words_ids = []


class ModelNotTrained(Exception):
    def __init__(self, msg="No saved instances of the model and the tokenizer were found. Please make sure"
                           "the model is trained and saved.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


def update_chat_history(tokenizer, new_input_ids, from_bot=False):
    global chat_history
    if len(chat_history) != 0 and not from_bot:
        chat_history.append(torch.cat([torch.tensor(tokenizer.encode(tokenizer.bos_token, add_special_tokens=False))
                                      .unsqueeze(1).T.to(DEVICE), new_input_ids], dim=1))
        flattened_chat_history = torch.cat(chat_history, dim=1).to(DEVICE)
        while flattened_chat_history.shape[1] > tokenizer.model_max_length:
            chat_history.pop(0)
            flattened_chat_history = torch.cat(chat_history, dim=1).to(DEVICE)
        return flattened_chat_history
    else:
        chat_history.append(new_input_ids)
        return new_input_ids


def get_bot_response_as_text(user_utterance):
    global saved_instance_path, loaded_model, loaded_tokenizer, chat_history
    if saved_instance_path is None:
        saved_instance_path = get_saved_instance_path()
        if saved_instance_path is None:
            raise ModelNotTrained()
    if loaded_model is None or loaded_tokenizer is None:
        loaded_model, loaded_tokenizer = load_saved_instance(saved_instance_path)

    new_user_input_ids = loaded_tokenizer.encode(user_utterance, truncation=True, return_tensors='pt').to(DEVICE)

    flattened_chat_history = update_chat_history(loaded_tokenizer, new_user_input_ids)

    # bot_response_ids = loaded_model.generate(flattened_chat_history, bad_words_ids=bad_words_ids).to(DEVICE)
    bot_response_ids = loaded_model.generate(flattened_chat_history).to(DEVICE)
    update_chat_history(loaded_tokenizer, bot_response_ids, from_bot=True)
    return loaded_tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)


def load_saved_instance(path):
    global bad_words_ids
    model_config = BlenderbotConfig.from_pretrained(path)
    model = BlenderbotForConditionalGeneration.from_pretrained(path, config=model_config).to(DEVICE)
    tokenizer = BlenderbotTokenizer.from_pretrained(path)
    bad_words_ids = [tokenizer(bad_word, add_prefix_space=True, add_special_tokens=False).input_ids
                     for bad_word in BAD_WORDS]
    return model, tokenizer


def get_saved_instance_path(use_mtime=False):
    saved_instances = []

    saved_instances_paths = glob.glob(os.path.join(OUTPUT_DIR, "{}-*".format(SAVED_INSTANCE_PREFIX)))
    if len(saved_instances_paths) < 1:
        return None

    for path in saved_instances_paths:
        if use_mtime:
            saved_instances.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)-([0-9]+)".format(SAVED_INSTANCE_PREFIX), path)
            if regex_match and regex_match.groups():
                saved_instances.append((int(regex_match.groups()[1]), path))

    if use_mtime:
        return sorted(saved_instances, reverse=True)[0][1]
    return sorted(saved_instances)[0][1]


def main():
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bit training: %s",
                   LOCAL_RANK, DEVICE, N_GPUS, bool(LOCAL_RANK != -1), FP16)

    if LOCAL_RANK not in [-1, 0]:
        torch.distributed.barrier()

    global saved_instance_path
    saved_instance_path = get_saved_instance_path()
    if DO_TRAIN or (DO_EVAL and saved_instance_path is None):
        tokenizer = BlenderbotTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=CACHE_DIR)

        model_config = BlenderbotConfig.from_pretrained(MODEL_CONFIG_NAME, cache_dir=CACHE_DIR)

        model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME, from_tf=False, config=model_config,
                                                                   cache_dir=CACHE_DIR)
        model.to(DEVICE)

        dataset = ConversationDataset(tokenizer)
        datasets_lengths = [len(dataset) - int(len(dataset) * EVAL_DATA_SPLIT_RATIO),
                            int(len(dataset) * EVAL_DATA_SPLIT_RATIO)]
        train_dataset, eval_dataset = random_split(dataset, datasets_lengths)

    if LOCAL_RANK == 0:
        torch.distributed.barrier()

    if DO_TRAIN:
        total_number_of_steps, training_loss, model = train(train_dataset, eval_dataset, model, tokenizer)
        logger.info("Number of steps = %s, Average training loss = %s", total_number_of_steps, training_loss)

        if LOCAL_RANK == -1 or torch.distributed.get_rank() == 0:
            saved_instance_output_dir = os.path.join(OUTPUT_DIR, "{}-{}-{}".format(SAVED_INSTANCE_PREFIX,
                                                                                   total_number_of_steps,
                                                                                   training_loss))
            os.makedirs(saved_instance_output_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, "module") else model
            logger.info("Saving model instance to %s", saved_instance_output_dir)
            model_to_save.save_pretrained(saved_instance_output_dir)
            logger.info("Saving tokenizer instance to %s", saved_instance_output_dir)
            tokenizer.save_pretrained(saved_instance_output_dir)

    if DO_EVAL and LOCAL_RANK in [-1, 0]:
        if saved_instance_path is not None:
            model, tokenizer = load_saved_instance(saved_instance_path)
        result = evaluate(eval_dataset, model, tokenizer)
        output_eval_file = os.path.join(OUTPUT_DIR, "evaluation_result.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Evaluation Result *****")
            for key in result.keys():
                logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == '__main__':
    main()
