import logging

from tqdm import tqdm
import torch

from .config import DEVICE, LOGGING_STEPS
from .utils import greedy_decode, calculate_character_error_rate, calculate_word_error_rate
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def train(model, train_loader, criterion, optimizer, scheduler, epoch, step, experiment):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, data in enumerate(tqdm(train_loader, desc="Training", leave=True, position=1)):
        spectrograms, labels, input_lengths, label_lengths = data
        spectrograms, labels = spectrograms.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        experiment.log_metric('training_loss', loss.item(), step=step)
        experiment.log_metric('learning_rate', scheduler.get_last_lr(), step=step)

        optimizer.step()
        scheduler.step()
        step += 1
        if batch_idx % LOGGING_STEPS == 0 or batch_idx == data_len:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len, 100. * batch_idx / len(train_loader), loss.item()))


def validate(model, valid_loader, criterion, iter_meter, experiment):
    logger.info('\nValidating...')
    model.eval()
    valid_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_loader, desc="Validating", leave=True, position=2)):
            spectrograms, labels, input_lengths, label_lengths = data
            spectrograms, labels = spectrograms.to(DEVICE), labels.to(DEVICE)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            valid_loss += loss.item() / len(valid_loader)

            decoded_predictions, decoded_targets = greedy_decode(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_predictions)):
                test_cer.append(calculate_character_error_rate(decoded_targets[j], decoded_predictions[j]))
                test_wer.append(calculate_word_error_rate(decoded_targets[j], decoded_predictions[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    experiment.log_metric('validation_loss', valid_loss, step=iter_meter.get())
    experiment.log_metric('character_error_rate', avg_cer, step=iter_meter.get())
    experiment.log_metric('word_error_rate', avg_wer, step=iter_meter.get())

    logger.info(
        'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(valid_loss, avg_cer, avg_wer))

    return valid_loss
