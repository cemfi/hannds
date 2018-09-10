import argparse
import datetime as dt
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from hannds_data import train_valid_test, convert, ContinuitySampler

DEBUG = False


def main():
    parser = argparse.ArgumentParser(description='Learn hannds neural net')
    parser.add_argument('--hidden_size', metavar='N', type=int, required=True, help='number of hidden units per layer')
    parser.add_argument('--layers', metavar='N', type=int, required=True, help='numbers of layers')
    parser.add_argument('--length', metavar='N', type=int, required=True, help='sequence length used in training')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using {device}", flush=True)

    convert('data/', overwrite=False)
    train_data, valid_data, _ = train_valid_test('data/', args.length, debug=DEBUG)
    trainer = Trainer(train_data, valid_data, args.hidden_size, args.layers, device)
    trainer.run()
    model = trainer.model
    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(model, 'models/' + _make_filename(args.hidden_size, args.layers))


def _make_filename(hidden_size, layers):
    if DEBUG:
        return "debug.pt"
    else:
        return f"hidden{hidden_size}_layers{layers}.pt"


class LSTMTransformOut(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(LSTMTransformOut, self).__init__()
        self.lstm = nn.LSTM(input_size=88, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.5)
        self.out_linear = nn.Linear(hidden_size, 88)

    def forward(self, input, h_prev, c_prev):
        lstm_output, (h_n, c_n) = self.lstm.forward(input, (h_prev, c_prev))
        output = self.out_linear(lstm_output)
        output = torch.tanh(output)
        output = output * input
        return output, h_n, c_n


class Trainer(object):
    def __init__(self, train_data, valid_data, hidden_size, layers, device):
        self.n_epochs = 20
        self.batch_size = 10
        self.layers = layers
        self.hidden_size = hidden_size

        sampler_train = ContinuitySampler(len(train_data), self.batch_size)
        sampler_valid = ContinuitySampler(len(valid_data), self.batch_size)
        self.data = {
            'train': DataLoader(train_data, batch_size=self.batch_size, sampler=sampler_train),
            'valid': DataLoader(valid_data, batch_size=self.batch_size, sampler=sampler_valid)
        }
        self.model = LSTMTransformOut(hidden_size, layers).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        self.device = device

        time = dt.datetime.now().strftime('%m-%d-%H%M-')
        self.writer = SummaryWriter('runs/' + time + _make_filename(hidden_size, layers))

    def initial_state(self):
        h_0 = torch.zeros((self.layers, self.batch_size, self.hidden_size)).to(self.device)
        c_0 = torch.zeros((self.layers, self.batch_size, self.hidden_size)).to(self.device)
        return h_0, c_0

    def run(self):
        for epoch in range(self.n_epochs):
            start_ts = dt.datetime.now()
            avg_loss = {'train': 0.0, 'test': 0.0}

            for phase in ['train', 'valid']:
                h_n, c_n = self.initial_state()
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
