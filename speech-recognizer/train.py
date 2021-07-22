from tqdm import tqdm

from config import DEVICE, LOGGING_STEPS
import torch.nn.functional as F


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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len, 100. * batch_idx / len(train_loader), loss.item()))


