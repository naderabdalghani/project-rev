import glob
import logging
import os
import re
import shutil

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup, AdamW, BlenderbotConfig, BlenderbotForConditionalGeneration

from utilities.config import TRAIN_BATCH_SIZE, MAX_STEPS, NUM_BATCHES_TILL_GRADIENT_ACCUMULATION, NUM_TRAIN_EPOCHS, \
    LEARNING_RATE, ADAM_EPSILON, WARMUP_STEPS, NO_DECAY_PARAMS_NAMES, FP16, N_GPUS, LOCAL_RANK, \
    PER_GPU_TRAIN_BATCH_SIZE, FP16_OPT_LEVEL, DEVICE, MAX_GRAD_NORM, LOGGING_STEPS, EVALUATE_DURING_TRAINING, \
    OUTPUT_DIR, SAVE_STEPS, MAX_CHECKPOINTS, CHECKPOINT_PREFIX, LOSS_FN_IGNORE_INDEX, WEIGHT_DECAY, EVAL_BATCH_SIZE

logger = logging.getLogger(__name__)


def get_most_recent_checkpoint_path(use_mtime=False):
    sorted_checkpoints = sort_checkpoints(use_mtime)
    if len(sorted_checkpoints) < 1:
        return None
    return sorted_checkpoints[len(sorted_checkpoints) - 1]


def sort_checkpoints(use_mtime=False):
    checkpoints = []

    checkpoints_paths = glob.glob(os.path.join(OUTPUT_DIR, "{}-*".format(CHECKPOINT_PREFIX)))

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
        logger.info("Deleting older checkpoint [{}] due to SAVE_TOTAL_LIMIT".format(checkpoint))
        shutil.rmtree(checkpoint)


def evaluate(dataset, model, tokenizer, prefix=""):
    if not os.path.exists(OUTPUT_DIR) and LOCAL_RANK in [-1, 0]:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def collate(dialogues):
        if tokenizer.pad_token is None:
            return pad_sequence(dialogues, batch_first=True)
        return pad_sequence(dialogues, batch_first=True, padding_value=tokenizer.pad_token_id)

    sampler = SequentialSampler(dataset) if LOCAL_RANK == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=sampler, batch_size=EVAL_BATCH_SIZE, collate_fn=collate)

    if N_GPUS > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Evaluation *****")
    logger.info("Number of dialogues = %d", len(dataset))
    logger.info("Batch size = %d", EVAL_BATCH_SIZE)

    eval_loss = 0.0
    eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluation", leave=True, position=0):
        inputs = batch.detach().clone().to(DEVICE)
        labels = batch.detach().clone().to(DEVICE)

        labels[labels == tokenizer.pad_token_id] = LOSS_FN_IGNORE_INDEX

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            eval_loss += loss.mean().item()
        eval_steps += 1

    eval_loss = eval_loss / eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    return {"perplexity": perplexity.item()}


def train(train_dataset, eval_dataset, model, tokenizer):
    tb_writer = None
    if LOCAL_RANK in [-1, 0]:
        tb_writer = SummaryWriter()

    def collate(dialogues):
        if tokenizer.pad_token is None:
            return pad_sequence(dialogues, batch_first=True)
        return pad_sequence(dialogues, batch_first=True, padding_value=tokenizer.pad_token_id)

    sampler = RandomSampler(train_dataset) if LOCAL_RANK == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate)

    num_of_epochs = NUM_TRAIN_EPOCHS
    if MAX_STEPS > 0:
        total_optimization_steps = MAX_STEPS
        num_of_epochs = \
            total_optimization_steps // (len(train_dataloader) // NUM_BATCHES_TILL_GRADIENT_ACCUMULATION) + 1
    else:
        total_optimization_steps = (len(train_dataloader) // NUM_BATCHES_TILL_GRADIENT_ACCUMULATION) * num_of_epochs

    model = model.module if hasattr(model, "module") else model  # In case of distributed training

    optim_grouped_params = [
        {
            "params": [param for name, param in model.named_parameters()
                       if not any(no_decay_param_name in name for no_decay_param_name in NO_DECAY_PARAMS_NAMES)],
            "weight_decay": WEIGHT_DECAY
        },
        {
            "params": [param for name, param in model.named_parameters()
                       if any(no_decay_param_name in name for no_decay_param_name in NO_DECAY_PARAMS_NAMES)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optim_grouped_params, lr=LEARNING_RATE, eps=ADAM_EPSILON)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                                num_training_steps=total_optimization_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    checkpoint_path = get_most_recent_checkpoint_path()
    if checkpoint_path is not None:
        try:
            checkpoint_directory_suffix = checkpoint_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_directory_suffix)
            epochs_trained = global_step // (len(train_dataloader) // NUM_BATCHES_TILL_GRADIENT_ACCUMULATION)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // NUM_BATCHES_TILL_GRADIENT_ACCUMULATION)
            logger.info("Loading saved optimizer and scheduler states from checkpoint path %s",
                        checkpoint_path)
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scheduler.pt")))
            logger.info("Loading saved model and its config from checkpoint path %s", checkpoint_path)
            model_config = BlenderbotConfig.from_pretrained(checkpoint_path)
            model = BlenderbotForConditionalGeneration.from_pretrained(checkpoint_path, config=model_config)
            logger.info("Continuing training from checkpoint, will skip to saved global_step")
            logger.info("Continuing training from epoch %d", epochs_trained)
            logger.info("Continuing training from global step %d", global_step)
            logger.info("Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("Starting fine-tuning...")

    if FP16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=FP16_OPT_LEVEL)

    # Multi-gpu training (should be after apex fp16 initialization)
    if N_GPUS > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if LOCAL_RANK != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True
        )

    logger.info("***** Training Model *****")
    logger.info("Number of dialogues = %d", len(train_dataset))
    logger.info("Number of epochs = %d", num_of_epochs)
    logger.info("Batch size per GPU = %d", PER_GPU_TRAIN_BATCH_SIZE)
    logger.info(
        "Total train batch size (w. parallel, distributed & accumulation) = %d",
        TRAIN_BATCH_SIZE
        * NUM_BATCHES_TILL_GRADIENT_ACCUMULATION
        * (torch.distributed.get_world_size() if LOCAL_RANK != -1 else 1),
    )
    logger.info("Number of batches till gradient accumulation = %d", NUM_BATCHES_TILL_GRADIENT_ACCUMULATION)
    logger.info("Total optimization steps = %d", total_optimization_steps)

    training_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    epochs = trange(epochs_trained, int(num_of_epochs), desc="Epoch", disable=LOCAL_RANK not in [-1, 0], leave=True,
                    position=0)

    for _ in epochs:
        data_iterator = tqdm(train_dataloader, desc="Iteration", disable=LOCAL_RANK not in [-1, 0], leave=True,
                             position=1)
        for step, batch in enumerate(data_iterator):

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            if batch.shape[1] > tokenizer.model_max_length:
                continue

            inputs = batch.detach().clone().to(DEVICE)
            labels = batch.detach().clone().to(DEVICE)

            labels[labels == tokenizer.pad_token_id] = LOSS_FN_IGNORE_INDEX
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

            if N_GPUS > 1:
                loss = loss.mean()
            if NUM_BATCHES_TILL_GRADIENT_ACCUMULATION > 1:
                loss = loss / NUM_BATCHES_TILL_GRADIENT_ACCUMULATION

            if FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            training_loss += loss.item()
            if (step + 1) % NUM_BATCHES_TILL_GRADIENT_ACCUMULATION == 0:
                if FP16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), MAX_GRAD_NORM)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if LOCAL_RANK in [-1, 0] and LOGGING_STEPS > 0 and global_step % LOGGING_STEPS == 0:
                    if LOCAL_RANK == -1 and EVALUATE_DURING_TRAINING:
                        results = evaluate(eval_dataset, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("{}".format(key.capitalize()), value, global_step)
                    tb_writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("Training_Loss", (training_loss - logging_loss) / LOGGING_STEPS, global_step)
                    logging_loss = training_loss

                if LOCAL_RANK in [-1, 0] and SAVE_STEPS > 0 and global_step % SAVE_STEPS == 0:
                    checkpoint_output_dir = os.path.join(OUTPUT_DIR, "{}-{}".format(CHECKPOINT_PREFIX, global_step))
                    os.makedirs(checkpoint_output_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    logger.info("Saving model checkpoint to %s", checkpoint_output_dir)
                    model_to_save.save_pretrained(checkpoint_output_dir)
                    logger.info("Saving optimizer and scheduler states to %s", checkpoint_output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(checkpoint_output_dir, "scheduler.pt"))
                    rotate_checkpoints()

            if 0 < MAX_STEPS < global_step:
                data_iterator.close()
                break
        if 0 < MAX_STEPS < global_step:
            epochs.close()
            break

    if LOCAL_RANK in [-1, 0]:
        tb_writer.close()

    return global_step, training_loss / global_step, model
