import glob
import logging
import os
import re
from functools import partial

from comet_ml import Experiment
import torch
from torch.utils.data import random_split
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import BasicVariantGenerator

from .config import MODEL_NAME, DO_TRAIN, VALID_DATA_SPLIT_RATIO, DO_VALID, BAD_WORDS, SAVED_INSTANCE_PREFIX, \
    HYPER_PARAMS, NUM_SAMPLES, MAX_NUM_STEPS, MIN_NUM_STEPS, DEFAULT_HYPER_PARAMS, HYPER_PARAMS_TUNING, \
    AVOID_BAD_WORDS, VALIDATE_WHILE_TRAINING
from exceptions import CoreModelNotTrained
from app_config import MODELS_DIR, CACHE_DIR, DEVICE, CUDA
from .preprocessing import ConversationDataset
from .train import train, validate

logger = logging.getLogger(__name__)
saved_instance_path = None
loaded_core_model = None
loaded_tokenizer = None
chat_history = []
bad_words_ids = []


def load_core_model():
    global saved_instance_path, loaded_core_model, loaded_tokenizer
    if loaded_core_model is None or loaded_tokenizer is None:
        if saved_instance_path is None:
            saved_instance_path = get_saved_instance_path()
            if saved_instance_path is None:
                raise CoreModelNotTrained()
        logger.info("Loading core model from {}".format(saved_instance_path))
        loaded_core_model, loaded_tokenizer = load_saved_instance(saved_instance_path)
        loaded_core_model.eval()


@torch.no_grad()
def get_bot_response_as_text(user_utterance):
    load_core_model()
    global loaded_core_model, loaded_tokenizer
    new_user_input_ids = loaded_tokenizer.encode(user_utterance, truncation=True, return_tensors='pt').to(DEVICE)
    flattened_chat_history = update_chat_history(loaded_tokenizer, new_user_input_ids)
    bot_response_ids = loaded_core_model.generate(flattened_chat_history, bad_words_ids=bad_words_ids).to(DEVICE) \
        if AVOID_BAD_WORDS else loaded_core_model.generate(flattened_chat_history).to(DEVICE)
    update_chat_history(loaded_tokenizer, bot_response_ids, from_bot=True)
    return loaded_tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)


def update_chat_history(tokenizer, new_input_ids, from_bot=False):
    global chat_history
    if len(chat_history) != 0 and not from_bot:
        chat_history.append(torch.cat([tokenizer.encode(tokenizer.bos_token, add_special_tokens=False,
                                                        return_tensors='pt').to(DEVICE), new_input_ids], dim=1))
        flattened_chat_history = torch.cat(chat_history, dim=1).to(DEVICE)
        while flattened_chat_history.shape[1] > tokenizer.model_max_length:
            chat_history.pop(0)
            flattened_chat_history = torch.cat(chat_history, dim=1).to(DEVICE)
        return flattened_chat_history
    else:
        chat_history.append(new_input_ids)
        return new_input_ids


def load_saved_instance(path):
    global bad_words_ids
    model = BlenderbotForConditionalGeneration.from_pretrained(path).to(DEVICE)
    tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if AVOID_BAD_WORDS:
        bad_words_ids = [tokenizer(bad_word, add_prefix_space=True, add_special_tokens=False).input_ids
                         for bad_word in BAD_WORDS]
    return model, tokenizer


def get_saved_instance_path(use_mtime=False):
    saved_instances = []

    saved_instances_paths = glob.glob(os.path.join(MODELS_DIR, "{}-*".format(SAVED_INSTANCE_PREFIX)))
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


def load_datasets(tokenizer, force_load_valid=False):
    dataset = ConversationDataset(tokenizer)
    if VALIDATE_WHILE_TRAINING or force_load_valid:
        dataset_len = len(dataset)
        datasets_lengths = [dataset_len - int(dataset_len * VALID_DATA_SPLIT_RATIO),
                            int(dataset_len * VALID_DATA_SPLIT_RATIO)]
        return random_split(dataset, datasets_lengths)
    return dataset, None


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger.info("Running on GPU" if CUDA else "Running on CPU")

    tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    train_dataset, valid_dataset = load_datasets(tokenizer)

    hyper_params = DEFAULT_HYPER_PARAMS

    if DO_TRAIN:
        if HYPER_PARAMS_TUNING:
            ray.init(include_dashboard=False)
            metric = "perplexity" if VALIDATE_WHILE_TRAINING else "training_loss"
            scheduler = ASHAScheduler(
                metric=metric,
                mode="min",
                max_t=MAX_NUM_STEPS,
                grace_period=MIN_NUM_STEPS
            )
            result = ray.tune.run(
                partial(train, train_dataset=train_dataset, valid_dataset=valid_dataset, tokenizer=tokenizer),
                resources_per_trial={"cpu": 4, "gpu": 1},
                config=HYPER_PARAMS,
                num_samples=NUM_SAMPLES,
                scheduler=scheduler,
                search_alg=BasicVariantGenerator(max_concurrent=1),
                verbose=0
            )
            best_trial = result.get_best_trial(metric, "min", "last")
            logger.info("Best trial config: {}".format(best_trial.config))
            logger.info("Best trial final result: {}".format(best_trial.last_result))

            hyper_params = best_trial.config

        total_number_of_steps, model, results = train(hyper_params, train_dataset, valid_dataset, tokenizer,
                                                      hyper_params_tuning=False)

        saved_instance_output_dir = os.path.join(MODELS_DIR, "{}-{}-{}".format(SAVED_INSTANCE_PREFIX,
                                                                               total_number_of_steps,
                                                                               results['average_training_loss']
                                                                               ))
        os.makedirs(saved_instance_output_dir, exist_ok=True)
        logger.info("Saving trained model instance to %s", saved_instance_output_dir)
        model.save_pretrained(saved_instance_output_dir)

        logger.info("***** Training Result *****")
        for key in results.keys():
            logger.info("%s = %s", key, str(results[key]))

    if DO_VALID:
        global saved_instance_path
        saved_instance_path = get_saved_instance_path()
        if saved_instance_path is None:
            model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME, from_tf=False, cache_dir=CACHE_DIR) \
                .to(DEVICE)
        else:
            model, tokenizer = load_saved_instance(saved_instance_path)
        if valid_dataset is None:
            train_dataset, valid_dataset = load_datasets(tokenizer, force_load_valid=True)
        result = validate(hyper_params, valid_dataset, model, tokenizer, silent=False)
        logger.info("***** Validation Result *****")
        for key in result.keys():
            logger.info("%s = %s", key, str(result[key]))


if __name__ == '__main__':
    main()
