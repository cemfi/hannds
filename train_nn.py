import datetime as dt
import math
import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from hannds_data import train_test_data, convert

DEBUG_MODE = True
USE_CUDA = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
print(f"Using {DEVICE}")


def main():
    parser = argparse.ArgumentParser(description='Learn hannds neural net')
    parser.add_argument('--layers', metavar='N', type=int, required=True, help='numbers of layers')
    parser.add_argument('--length', metavar='N', type=int, required=True, help='sequence length')
    parser.add_argument('--net', metavar='TYPE', type=str, required=True, help='type of the network: LSTM or RNN')
    args = parser.parse_args()

    train_dataset, validate_dataset = train_test_data('data/', args.length, args.length, debug=DEBUG_MODE)
    model = train(args, train_dataset, validate_dataset)
    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(model, 'models/' + make_filename(args))


def make_filename(args):
    if DEBUG_MODE:
        return "debug.pt"
    else:
        return f"{args.net}_layers{args.layers}_len{args.length}.pt"


class FFNetwork(nn.Module):
    def __init__(self, len_sequence):
        super(FFNetwork, self).__init__()
        hidden_size = 200
        self.network = nn.Sequential(
            nn.Linear(88 * len_sequence, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 88),
            nn.Tanh())
        self.len_sequence = len_sequence

    def forward(self, input):
        """
        :param input: (batch_size, len_sequence, 88)
        :return: (batch_size, 1, 88)
        """
        transformed_input = input.view(-1, self.len_sequence * 88)
        output = self.network.forward(transformed_input)
        transformed_output = output.view(-1, 1, 88)
        return transformed_output


class RNN(nn.Module):
    def __init__(self, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=88, hidden_size=88, num_layers=num_layers, batch_first=True)

    def forward(self, input):
        """
        :param input: (batch_size, len_sequence, 88)
        :return: (batch_size, len_sequence, 88)
        """
        output, hidden = self.rnn.forward(input)
        return output, hidden


class LSTM(nn.Module):
    def __init__(self, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=88, hidden_size=88, num_layers=num_layers, batch_first=True)

    def forward(self, input):
        output, hidden = self.lstm.forward(input)
        return output, hidden


def train(args, train_dataset, validate_dataset):
    n_epochs = 3
    batch_size = 100

    # Prepare
    time = dt.datetime.now().strftime('%m-%d-%H%M-')
    writer = SummaryWriter('runs/' + time + make_filename(args))
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if args.net.lower() == 'rnn':
        model = RNN(num_layers=args.layers).to(DEVICE)
    elif args.net.lower() == 'lstm':
        model = LSTM(num_layers=args.layers).to(DEVICE)
    else:
        raise Exception('Invalid argument')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    batches_per_epoch = len(train_data.dataset) // batch_size
    scale_epoch_fractional = 10 ** math.ceil(math.log10(batches_per_epoch))
    total_iterations = 0

    # Train
    for epoch in range(n_epochs):
        start = dt.datetime.now()
        epoch_train_loss = (0.0, 0)
        epoch_test_loss = (0.0, 0)
        for batch_idx, (X_batch, Y_batch) in enumerate(train_data):  # Gradient descent
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            optimizer.zero_grad()
            output, *_ = model(X_batch)
            train_loss = criterion(output[:, -1], Y_batch[:, -1])
            train_loss.backward()
            epoch_train_loss = (epoch_train_loss[0] + train_loss, epoch_train_loss[1] + 1)
            optimizer.step()
            total_iterations += 1

            if batch_idx % 100 == 0:
                cur_epoch = total_iterations / batches_per_epoch
                test_loss = compute_test_loss(model, validate_dataset, max_batches=20)
                epoch_test_loss = (epoch_test_loss[0] + test_loss, epoch_test_loss[1] + 1)

                scaled_epoch = cur_epoch * scale_epoch_fractional
                writer.add_scalar('loss/training', math.log(train_loss), scaled_epoch)
                writer.add_scalar('loss/validation', math.log(test_loss), scaled_epoch)

                for name, param in model.named_parameters():
                    writer.add_histogram(name + '/data', param.data.clone().cpu().numpy(), scaled_epoch)
                    writer.add_histogram(name + '/grad', param.grad.clone().cpu().numpy(), scaled_epoch)

        end = dt.datetime.now()
        epoch_train_loss = epoch_train_loss[0] / epoch_train_loss[1]
        epoch_test_loss = epoch_test_loss[0] / epoch_test_loss[1]

        t_total = (end - start).total_seconds()

        print(f"Epoch: {epoch}({total_iterations}), duration {t_total:.1f}s, "
              f"log train loss = {math.log(epoch_train_loss):.2f}, log test loss = {math.log(epoch_test_loss):.2f}")

    return model


def compute_test_loss(model, validate_dataset, max_batches=-1):
    """
    computes test loss. If max_batches == -1 it will compute the loss
    over the entire data set.
    """
    validate_data = DataLoader(validate_dataset, batch_size=100, shuffle=True)
    with torch.no_grad():  # Compute test error
        total_loss = 0.0
        n = 0
        for X_batch, Y_batch in validate_data:
            if n == max_batches:
                break
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            output, *_ = model(X_batch)
            criterion = nn.MSELoss()
            test_loss = criterion(output[:, -1], Y_batch[:, -1])
            total_loss += test_loss
            n += 1

        total_loss /= n
        return total_loss


if __name__ == '__main__':
    convert('data/', overwrite=False)
    main()
