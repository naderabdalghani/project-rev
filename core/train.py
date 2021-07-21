import glob
import logging
import os
import re
import shutil

from comet_ml import Experiment
import torch
from ray import tune
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup, AdamW, BlenderbotForConditionalGeneration

from config import MAX_STEPS, NO_DECAY_PARAMS_NAMES, DEVICE, LOGGING_STEPS, MODELS_DIR, SAVE_STEPS, \
    MAX_CHECKPOINTS, CHECKPOINT_PREFIX, LOSS_FN_IGNORE_INDEX, RESUME_TRAINING, MAX_GRAD_NORM, MODEL_NAME, CACHE_DIR, \
    TRIAL_NAME, VALIDATE_WHILE_TRAINING
from keys import COMET_API_KEY

logger = logging.getLogger(__name__)


def get_checkpoint_path(use_mtime=False):
    sorted_checkpoints = sort_checkpoints(use_mtime)
    if len(sorted_checkpoints) < 1:
        return None
    return sorted_checkpoints[len(sorted_checkpoints) - 1]


def sort_checkpoints(use_mtime=False):
    checkpoints = []

    checkpoints_paths = glob.glob(os.path.join(MODELS_DIR, "{}-*".format(CHECKPOINT_PREFIX)))

    for path in checkpoints_paths:
        if use_mtime:
            checkpoints.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(CHECKPOINT_PREFIX), path)
            if regex_match and regex_match.groups():
                checkpoints.append((int(regex_match.groups()[0]), path))

    sorted_checkpoints = sorted(checkpoints)
    sorted_checkpoints = [checkpoint[1] for checkpoint in sorted_checkpoints]
    return sorted_checkpoints


def rotate_checkpoints(use_mtime=False):
    if not MAX_CHECKPOINTS or MAX_CHECKPOINTS <= 0:
        return
    checkpoints_sorted = sort_checkpoints(use_mtime)
    if len(checkpoints_sorted) <= MAX_CHECKPOINTS:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - MAX_CHECKPOINTS)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to exceeding SAVE_TOTAL_LIMIT: {}".format(checkpoint,
                                                                                                  MAX_CHECKPOINTS))
        shutil.rmtree(checkpoint)


def validate(config, dataset, model, tokenizer, experiment=Experiment(api_key='dummy_key', disabled=True),
             global_step=-1, silent=True):
    def collate(dialogues):
        if tokenizer.pad_token is None:
            return pad_sequence([x for x, _ in dialogues], batch_first=True), \
                   pad_sequence([y for _, y in dialogues], batch_first=True)
        return pad_sequence([x for x, _ in dialogues], batch_first=True, padding_value=tokenizer.pad_token_id), \
            pad_sequence([y for _, y in dialogues], batch_first=True, padding_value=tokenizer.pad_token_id)

    sampler = SequentialSampler(dataset)
    valid_dataloader = DataLoader(dataset, sampler=sampler, batch_size=config["BATCH_SIZE"], collate_fn=collate)

    if not silent:
        logger.info("***** Running Validation *****")
        logger.info("Number of dialogues = %d", len(dataset))
        logger.info("Batch size = %d", config["BATCH_SIZE"])

    model.eval()

    perplexities = []
    valid_loss = 0
    valid_steps = 0

    with torch.no_grad():
        for inputs, labels in tqdm(valid_dataloader, desc="Validation", unit="batch", leave=True, position=2,
                                   disable=silent):
            inputs = inputs.detach().clone().to(DEVICE)
            labels = labels.detach().clone().to(DEVICE)

            output = model(inputs, labels=labels)
            valid_loss += output.loss.item()
            valid_steps += 1
            logits = output.logits
            scores = torch.softmax(logits, dim=2)
            batch_prob = torch.gather(scores, index=labels.unsqueeze(dim=2), dim=2).squeeze(dim=2)
            batch_prob[labels == 0] = 1
            batch_prob = batch_prob.type(torch.float64)
            batch_prob = batch_prob.prod(dim=1)
            perplexity = torch.pow(batch_prob, -1 / labels.count_nonzero(dim=1))

            avg_perplexity = torch.mean(perplexity[perplexity.isfinite()])
            if not avg_perplexity.isnan():
                perplexities.append(avg_perplexity.item())
        results = {"perplexity": (sum(perplexities) / len(perplexities)),
                   'validation_loss': valid_loss / valid_steps}
        for key, value in results.items():
            experiment.log_metric("{}".format(key), value, global_step)

    model.train()

    return results


def train(config, train_dataset, valid_dataset, tokenizer, hyper_params_tuning=True):
    if hyper_params_tuning:
        tune.utils.wait_for_gpu()
    if COMET_API_KEY:
        experiment = Experiment(api_key=COMET_API_KEY, project_name="Core Module", parse_args=False)
        if hyper_params_tuning and TRIAL_NAME:
            experiment.add_tag(TRIAL_NAME)
    else:
        experiment = Experiment(api_key='dummy_key', disabled=True)

    experiment.log_parameters(config)

    model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME, from_tf=False, cache_dir=CACHE_DIR) \
        .to(DEVICE)

    def collate(dialogues):
        if tokenizer.pad_token is None:
            return pad_sequence([x for x, _ in dialogues], batch_first=True), \
                   pad_sequence([y for _, y in dialogues], batch_first=True)
        return pad_sequence([x for x, _ in dialogues], batch_first=True, padding_value=tokenizer.pad_token_id), \
            pad_sequence([y for _, y in dialogues], batch_first=True, padding_value=tokenizer.pad_token_id)

    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=config["BATCH_SIZE"], collate_fn=collate)

    num_of_epochs = config["NUM_TRAIN_EPOCHS"]
    if MAX_STEPS > 0:
        total_optimization_steps = MAX_STEPS
        num_of_epochs = \
            total_optimization_steps // (len(train_dataloader)) + 1
    else:
        total_optimization_steps = (len(train_dataloader)) * num_of_epochs

    optim_grouped_params = [
        {
            "params": [param for name, param in model.named_parameters()
                       if not any(no_decay_param_name in name for no_decay_param_name in NO_DECAY_PARAMS_NAMES)],
            "weight_decay": config["WEIGHT_DECAY"]
        },
        {
            "params": [param for name, param in model.named_parameters()
                       if any(no_decay_param_name in name for no_decay_param_name in NO_DECAY_PARAMS_NAMES)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optim_grouped_params, lr=config["LEARNING_RATE"], eps=config["ADAM_EPSILON"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["WARMUP_STEPS"],
                                                num_training_steps=total_optimization_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    checkpoint_path = get_checkpoint_path()
    if checkpoint_path is not None and RESUME_TRAINING and not hyper_params_tuning:
        try:
            checkpoint_directory_suffix = checkpoint_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_directory_suffix)
            epochs_trained = global_step // len(train_dataloader)
            steps_trained_in_current_epoch = global_step % len(train_dataloader)
            logger.info("Loading saved optimizer and scheduler states from checkpoint path %s", checkpoint_path)
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scheduler.pt")))
            logger.info("Loading saved model from checkpoint path %s", checkpoint_path)
            model = BlenderbotForConditionalGeneration.from_pretrained(checkpoint_path)
            logger.info("Continuing training from checkpoint, will skip to saved global_step")
            logger.info("Continuing training from epoch %d", epochs_trained)
            logger.info("Continuing training from global step %d", global_step)
            logger.info("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("Starting fine-tuning...")

    logger.info("***** Training Model *****")
    logger.info("Number of dialogues = {}".format(len(train_dataset)))
    logger.info("Batch size = {}".format(config["BATCH_SIZE"]))
    logger.info("Learning rate = {}".format(config["LEARNING_RATE"]))
    logger.info("Weight decay = {}".format(config["WEIGHT_DECAY"]))
    logger.info("Warmup steps = {}".format(config["WARMUP_STEPS"]))
    logger.info("Number of epochs = {}".format(num_of_epochs))
    logger.info("Total optimization steps = {}".format(total_optimization_steps))

    training_loss = 0.0
    last_recorded_loss = 0.0
    model.zero_grad()
    epochs = trange(epochs_trained, int(num_of_epochs), desc="Epochs", unit="epoch", leave=True, position=0,
                    disable=hyper_params_tuning)

    for _ in epochs:
        data_iterator = tqdm(train_dataloader, desc="Training epoch", unit="batch", leave=True, position=1,
                             disable=hyper_params_tuning)
        for step, (inputs, labels) in enumerate(data_iterator):

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            if inputs.shape[1] > tokenizer.model_max_length or labels.shape[1] > tokenizer.model_max_length:
                continue

            inputs = inputs.detach().clone().to(DEVICE)
            labels = labels.detach().clone().to(DEVICE)

            labels[labels == tokenizer.pad_token_id] = LOSS_FN_IGNORE_INDEX
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            loss.backward()

            last_recorded_loss = loss.item()
            training_loss += last_recorded_loss
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            experiment.log_metric("learning_rate", scheduler.get_last_lr()[0], global_step)
            experiment.log_metric("training_loss", last_recorded_loss, global_step)

            if LOGGING_STEPS > 0 and global_step % LOGGING_STEPS == 0:
                if VALIDATE_WHILE_TRAINING:
                    results = validate(config, valid_dataset, model, tokenizer, experiment, global_step)
                    if hyper_params_tuning:
                        tune.report(validation_loss=results['validation_loss'], perplexity=results['perplexity'])
                else:
                    if hyper_params_tuning:
                        tune.report(training_loss=last_recorded_loss)

            if SAVE_STEPS > 0 and global_step % SAVE_STEPS == 0 and not hyper_params_tuning:
                checkpoint_output_dir = os.path.join(MODELS_DIR, "{}-{}".format(CHECKPOINT_PREFIX, global_step))
                os.makedirs(checkpoint_output_dir, exist_ok=True)
                logger.info("Saving model checkpoint to %s", checkpoint_output_dir)
                model.save_pretrained(checkpoint_output_dir)
                logger.info("Saving optimizer and scheduler states to %s", checkpoint_output_dir)
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_output_dir, "scheduler.pt"))
                rotate_checkpoints()

            if 0 < MAX_STEPS < global_step:
                data_iterator.close()
                break

    experiment.end()

    if hyper_params_tuning:
        if VALIDATE_WHILE_TRAINING:
            return validate(config, valid_dataset, model, tokenizer, experiment, global_step)
        return {'training_loss': last_recorded_loss}
    else:
        if VALIDATE_WHILE_TRAINING:
            valid_results = validate(config, valid_dataset, model, tokenizer, experiment, global_step)
            valid_results["average_training_loss"] = (training_loss / global_step)
            return global_step, model, valid_results
        return global_step, model, {'average_training_loss': (training_loss / global_step)}
