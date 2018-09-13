import argparse
import datetime as dt
import math
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from hannds_data import train_valid_test, convert, ContinuitySampler

_debug = None

def main():
    global _debug

    parser = argparse.ArgumentParser(description='Learn hannds neural net')
    parser.add_argument('--hidden_size', metavar='N', type=int, required=True, help='number of hidden units per layer')
    parser.add_argument('--layers', metavar='N', type=int, required=True, help='numbers of layers')
    parser.add_argument('--length', metavar='N', type=int, required=True, help='sequence length used in training')
    parser.add_argument('--cuda', action='store_true', required=False, help='use CUDA')
    parser.add_argument('--bidirectional', action='store_true', required=False, help='use a bi-directional LSTM')
    parser.add_argument('--debug', action='store_true', required=False, help='run with minimal data')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using {device}", flush=True)
    _debug = args.debug

    convert('data/', overwrite=False)
    train_data, valid_data, _ = train_valid_test('data/', args.length, debug=args.debug)
    trainer = Trainer(train_data, valid_data, args.hidden_size, args.layers, args.bidirectional, device)
    trainer.run()
    model = trainer.model
    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(model, 'models/' + _make_filename(args.hidden_size, args.layers, args.bidirectional))


def _make_filename(hidden_size, layers, bidirectional):
    if _debug:
        return "debug.pt"
    else:
        bi_str = '-bidirectional' if bidirectional else ''
        return f"hidden{hidden_size}_layers{layers}{bi_str}.pt"


class LSTMTransformOut(nn.Module):
    def __init__(self, hidden_size, n_layers, bidirectional):
        super(LSTMTransformOut, self).__init__()
        self.lstm = nn.LSTM(input_size=88, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=0.5, bidirectional=bidirectional)
        self.n_directions = 2 if bidirectional else 1
        self.out_linear = nn.Linear(hidden_size * self.n_directions, 88)
        self.n_layers = n_layers

    def forward(self, input, h_prev, c_prev):
        lstm_output, (h_n, c_n) = self.lstm.forward(input, (h_prev, c_prev))
        output = self.out_linear(lstm_output)
        output = torch.tanh(output)
        output = output * input
        return output, h_n, c_n


class Trainer(object):
    def __init__(self, train_data, valid_data, hidden_size, layers, bidirectional, device):
        self.n_epochs = 30
        self.batch_size_train = 10
        self.layers = layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        sampler_train = ContinuitySampler(len(train_data), self.batch_size_train)
        self.data = {
            'train': DataLoader(train_data, batch_size=self.batch_size_train, sampler=sampler_train, drop_last=True),
            'valid': DataLoader(valid_data, batch_size=self.batch_size_train)
        }
        self.model = LSTMTransformOut(hidden_size, layers, bidirectional).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        self.device = device

        time = dt.datetime.now().strftime('%m-%d-%H%M-')
        self.writer = SummaryWriter('runs/' + time + _make_filename(hidden_size, layers, bidirectional))

    def zero_state(self, phase):
        n_directions = self.model.n_directions
        if phase == 'train':
            h_0 = torch.zeros((self.layers * n_directions, self.batch_size_train, self.hidden_size)).to(self.device)
            c_0 = torch.zeros((self.layers * n_directions, self.batch_size_train, self.hidden_size)).to(self.device)
        else:
            h_0 = torch.zeros((self.layers * n_directions, 1, self.hidden_size)).to(self.device)
            c_0 = torch.zeros((self.layers * n_directions, 1, self.hidden_size)).to(self.device)
        return h_0, c_0

    def run(self):
        for epoch in range(self.n_epochs):
            start_ts = dt.datetime.now()
            avg_loss = {'train': 0.0, 'test': 0.0}

            for phase in ['train', 'valid']:
                h_n, c_n = self.zero_state(phase)
                phase_loss = (0.0, 0)
                valid_accuracy = (0.0, 0)
                self.model = self.model.train(phase == 'train')

                for batch_idx, (X_batch, Y_batch) in enumerate(self.data[phase]):
                    with torch.set_grad_enabled(phase == 'train'):
                        X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                        self.optimizer.zero_grad()
                        output, h_n, c_n = self.model(X_batch, h_n, c_n)
                        train_loss = self.criterion(output, Y_batch)
                        if phase == 'train':
                            train_loss.backward()
                            self.optimizer.step()
                        else:
                            va1 = valid_accuracy[0] + compute_accuracy(X_batch, Y_batch, output)
                            va2 = valid_accuracy[1] + 1
                            valid_accuracy = (va1, va2)

                        h_n = h_n.detach()
                        c_n = c_n.detach()
                        if self.bidirectional:
                            h_n[1::2] = 0  # ignore backward state as we are stepping forward from batch to batch
                            c_n[1::2] = 0
                        phase_loss = (phase_loss[0] + train_loss, phase_loss[1] + 1)

                avg_loss[phase] = phase_loss[0] / phase_loss[1]
                self._save_summaries(avg_loss, epoch, phase, valid_accuracy)

            end_ts = dt.datetime.now()
            t_total = (end_ts - start_ts).total_seconds()
            self._print_epoch_stats(epoch, t_total, avg_loss, valid_accuracy)

        plot_output(output[0])

    def _save_summaries(self, avg_loss, epoch, phase, valid_accuracy):
        self.writer.add_scalar('loss/' + phase, avg_loss[phase], epoch)
        if phase == 'train':
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name + '/data', param.data.clone().cpu().numpy(), epoch)
                self.writer.add_histogram(name + '/grad', param.grad.clone().cpu().numpy(), epoch)
        else:
            self.writer.add_scalar('accuracy', valid_accuracy[0] / valid_accuracy[1], epoch)

    def _print_epoch_stats(self, epoch, t_total, avg_loss, valid_accuracy):
        print(f'Epoch: {epoch}/{self.n_epochs - 1}, {t_total:.1f}sec')
        print('-' * 21)
        print(f'Train loss = {avg_loss["train"]:.6f}')
        print(f'Valid loss = {avg_loss["valid"]:.6f}')
        print(f'Accuracy   = {valid_accuracy[0] / valid_accuracy[1]:.1f}%', flush=True)
        print()
        print()


def compute_accuracy(X, Y, prediction):
    num_notes = torch.sum(X)
    left_hand = (prediction < 0.0).float()
    right_hand = (prediction > 0.0).float()
    prediction = left_hand * -1.0 + right_hand * 1.0
    diff = (prediction != Y).float()
    errors_percent = torch.sum(diff) / num_notes * 100.0
    return 100.0 - errors_percent


def plot_output(output, max_pages=1):
    with PdfPages('results.pdf') as pdf:
        for i in reversed(range(max_pages)):
            if (i + 1) * 100 <= output.shape[0]:
                fig, ax = plt.subplots()
                ax.imshow(output[i * 100: (i + 1) * 100], cmap='bwr', origin='lower', vmin=-1, vmax=1)
                pdf.savefig(fig)
                plt.close()


if __name__ == '__main__':
    main()
