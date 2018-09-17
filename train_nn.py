import argparse
from collections import Counter
import copy
import datetime as dt
import json
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader

import hannds_data as hd

# global vars
g_debug = None
g_time = dt.datetime.now().strftime('%m-%d-%H%M-%S')


def main():
    global g_debug

    parser = argparse.ArgumentParser(description='Learn hannds neural net')
    parser.add_argument('--hidden_size', metavar='N', type=int, required=True, help='number of hidden units per layer')
    parser.add_argument('--layers', metavar='N', type=int, required=True, help='numbers of layers')
    parser.add_argument('--length', metavar='N', type=int, required=True, help='sequence length used in training')
    parser.add_argument('--cuda', action='store_true', required=False, help='use CUDA')
    parser.add_argument('--bidirectional', action='store_true', required=False, help='use a bi-directional LSTM')
    parser.add_argument('--debug', action='store_true', required=False, help='run with minimal data')
    parser.add_argument('--cv_partition', metavar='N', type=int, required=False, default=1,
                        help='the partition index (from 1 to 10) for 10-fold cross validation')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Using {device}", flush=True)
    g_debug = args.debug

    data = hd.AllData(debug=args.debug)
    data.initialize_from_dir(args.length, cv_partition=args.cv_partition)
    train_data = data.train_data
    valid_data = data.valid_data
    trainer = Trainer(train_data, valid_data, args.hidden_size, args.layers, args.bidirectional, device)
    trainer.run()
    model = trainer.model
    if not os.path.exists('models'):
        os.mkdir('models')

    os.mkdir('models/' + g_time)
    torch.save(model, f'models/{g_time}/model.pt')
    desc = {
        'args': vars(args),
        'train': data.train_files,
        'valid': data.valid_files,
        'test': data.test_files
    }
    with open(f'models/{g_time}/desc.json', 'w') as file:
        json.dump(desc, file, indent=4)


class Network(nn.Module):
    def __init__(self, hidden_size, n_layers, bidirectional):
        super(Network, self).__init__()
        self.lstm = nn.LSTM(input_size=88, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=0.5, bidirectional=bidirectional)
        self.n_directions = 2 if bidirectional else 1
        self.out_linear = nn.Linear(hidden_size * self.n_directions, 88 * 3)
        self.n_layers = n_layers

    def forward(self, input, h_prev=None, c_prev=None):
        hidden_in = (h_prev, c_prev) if h_prev is not None else None
        lstm_output, (h_n, c_n) = self.lstm.forward(input, hidden_in)
        output = self.out_linear(lstm_output)
        output = output.view(-1, output.shape[1], 88, 3)
        # Residual connection
        # output[:, :, :, 0] += 1.0 - input
        # output[:, :, :, 1] += input
        # output[:, :, :, 2] += input
        return output, h_n, c_n


class Trainer(object):
    def __init__(self, train_data, valid_data, hidden_size, layers, bidirectional, device):
        self.n_epochs = 50
        self.batch_size_train = 10
        self.layers = layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        sampler_train = hd.ContinuationSampler(len(train_data), self.batch_size_train)
        self.data = {
            'train': DataLoader(train_data, batch_size=self.batch_size_train, sampler=sampler_train, drop_last=True),
            'valid': DataLoader(valid_data, batch_size=self.batch_size_train)
        }
        self.model = Network(hidden_size, layers, bidirectional).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        self.device = device

        bi_str = '-bidirectional' if bidirectional else ''
        desc = f"hidden{hidden_size}_layers{layers}{bi_str}"
        self.writer = SummaryWriter('runs/' + g_time + '-' + desc)

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
        best_accuracy = 0.0
        final_model = None

        for epoch in range(self.n_epochs):
            start_ts = dt.datetime.now()
            avg_loss = {'train': 0.0, 'test': 0.0}

            for phase in ['train', 'valid']:
                h_n, c_n = self.zero_state(phase)
                phase_loss = (0.0, 0)
                valid_accuracy = (0.0, 0)
                valid_accuracy_plain = (0.0, 0)
                self.model = self.model.train(phase == 'train')

                for batch_idx, (X_batch, Y_batch) in enumerate(self.data[phase]):
                    with torch.set_grad_enabled(phase == 'train'):
                        X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                        self.optimizer.zero_grad()
                        output, h_n, c_n = self.model(X_batch, h_n, c_n)
                        train_loss = self.criterion(output.view((-1, 3)), Y_batch.view(-1))
                        if phase == 'train':
                            train_loss.backward()
                            self.optimizer.step()
                        else:
                            predicted_classes = torch.argmax(output, dim=3)
                            filter_func = majority_filter if self.bidirectional else causal_filter
                            X_numpy = X_batch.squeeze().cpu().numpy()
                            Y_numpy = Y_batch.squeeze().cpu().numpy()
                            classes_numpy = predicted_classes.squeeze().cpu().numpy()

                            va1 = valid_accuracy[0] + compute_accuracy(X_numpy, Y_numpy, classes_numpy, filter_func)
                            va2 = valid_accuracy[1] + 1
                            valid_accuracy = (va1, va2)

                            va1 = valid_accuracy_plain[0] + compute_accuracy(X_numpy, Y_numpy, classes_numpy)
                            va2 = valid_accuracy_plain[1] + 1
                            valid_accuracy_plain = (va1, va2)

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
            if valid_accuracy[0] / valid_accuracy[1] > best_accuracy:
                best_accuracy = valid_accuracy[0] / valid_accuracy[1]
                final_model = copy.deepcopy(self.model)
                print('*')
            self._print_epoch_stats(epoch, t_total, avg_loss, valid_accuracy, valid_accuracy_plain)

        self.model = final_model
        plot_output(torch.softmax(output[0], dim=-1))

    def _save_summaries(self, avg_loss, epoch, phase, valid_accuracy):
        self.writer.add_scalar('loss/' + phase, avg_loss[phase], epoch)
        if phase == 'train':
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name + '/data', param.data.clone().cpu().numpy(), epoch)
                self.writer.add_histogram(name + '/grad', param.grad.clone().cpu().numpy(), epoch)
        else:
            self.writer.add_scalar('accuracy', valid_accuracy[0] / valid_accuracy[1], epoch)

    def _print_epoch_stats(self, epoch, t_total, avg_loss, valid_accuracy, valid_accuracy_plain):
        filtered_acc = valid_accuracy[0] / valid_accuracy[1]
        unfiltered_acc = valid_accuracy_plain[0] / valid_accuracy_plain[1]
        print(f'Epoch: {epoch}/{self.n_epochs - 1}, {t_total:.1f}sec')
        print('-' * 21)
        print(f'Train loss = {avg_loss["train"]:.6f}')
        print(f'Valid loss = {avg_loss["valid"]:.6f}')
        print(f'Accuracy   = {filtered_acc:.1f}({unfiltered_acc:.1f})%', flush=True)
        print()
        print()


def compute_accuracy(X, Y, predicted_classes, filter_func=lambda x: x):
    assert len(X.shape) == len(Y.shape) == len(predicted_classes.shape) == 2
    n_notes = X.sum()
    predicted_classes = filter_func(predicted_classes)
    pred_lh = predicted_classes == hd.LEFT_HAND_LABEL
    pred_rh = predicted_classes == hd.RIGHT_HAND_LABEL
    label_lh = Y == hd.LEFT_HAND_LABEL
    label_rh = Y == hd.RIGHT_HAND_LABEL
    n_lh_correct = (pred_lh * label_lh).sum()
    n_rh_correct = (pred_rh * label_rh).sum()
    assert n_lh_correct <= n_notes
    assert n_rh_correct <= n_notes
    n_correct = n_lh_correct + n_rh_correct
    assert n_correct <= n_notes
    return n_correct / n_notes * 100.0


def plot_output(output, max_pages=32):
    with PdfPages('results.pdf') as pdf:
        for i in reversed(range(max_pages)):
            if (i + 1) * 100 <= output.shape[0]:
                region = output[i * 100: (i + 1) * 100]
                image = region[:, :, 1] * -1.0 + region[:, :, 2] * 1.0
                fig, ax = plt.subplots()
                ax.imshow(image, cmap='bwr', origin='lower', vmin=-1, vmax=1)
                pdf.savefig(fig)
                plt.close()


def causal_filter(predicted_classes):
    """
    Filters the predicted classes: the assignment is decided at the
    note-on event. Changes the content of predicted_classes.
    """
    predicted_classes = predicted_classes.copy()
    for row in range(1, predicted_classes.shape[0]):
        last_line = predicted_classes[row - 1]
        current_line = predicted_classes[row]
        both_note_on = np.logical_and(current_line != hd.NOT_PLAYED_LABEL, last_line != hd.NOT_PLAYED_LABEL)
        predicted_classes[row, both_note_on] = last_line[both_note_on]

    return predicted_classes


def majority_filter(predicted_classes):  # Needs to be profiled
    """
    Filters the predicted classes: the assignment is decided at the
    note-off event. Changes the content of predicted_classes.
    """
    predicted_classes = predicted_classes.copy()
    for column in range(predicted_classes.shape[1]):
        row = 0
        while row < predicted_classes.shape[0]:
            last_note = predicted_classes[row - 1, column] if row > 0 else 0
            current_note = predicted_classes[row, column]
            start = row
            if last_note == hd.NOT_PLAYED_LABEL and current_note != hd.NOT_PLAYED_LABEL:
                row += 1
                while row < predicted_classes.shape[0]:
                    current_note = predicted_classes[row, column]
                    if current_note == hd.NOT_PLAYED_LABEL or row == predicted_classes.shape[0] - 1:
                        end = row
                        c = Counter(predicted_classes[start: end, column])
                        majority_value, _ = c.most_common()[0]
                        predicted_classes[start: end, column] = majority_value
                        row += 1
                        break
                    else:
                        row += 1
            else:
                row += 1

    return predicted_classes


if __name__ == '__main__':
    main()
