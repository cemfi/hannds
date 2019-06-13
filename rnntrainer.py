import copy
import datetime as dt
import os

from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

import hannds_data as hd

# global vars
g_time = dt.datetime.now().strftime('%m-%d-%H%M')


class RNNTrainer(object):
    """Trains a neural network.

    Args:
        model: the neural network model
        train_data: the training data
        valid_data: the validation data
        args: the command line arguments
        device: the torch.device where the training is executed
    """

    def __init__(self, model, train_data, valid_data, args, device):
        self.n_epochs = 50
        self.batch_size_train = 10
        self.layers = args['layers']
        self.hidden_size = args['hidden_size']
        self.bidirectional = args['bidirectional']

        sampler_train = hd.ContinuationSampler(len(train_data), self.batch_size_train)
        self.data = {
            'train': DataLoader(train_data, batch_size=self.batch_size_train, sampler=sampler_train, drop_last=True),
            'valid': DataLoader(valid_data, batch_size=self.batch_size_train)
        }
        self.model = model
        self.device = device
        self.rnn_type = args['rnn_type'].upper()

        bi_str = '_bidirectional' if self.bidirectional else ''
        self.descriptive_filename = f"network={args['network']}({args['rnn_type']})_hidden={args['hidden_size']}" \
            f"_layers={args['layers']}{bi_str}_cv={args['cv_partition']}"
        logdir = f'runs/t={g_time}_pid={os.getpid()}_{self.descriptive_filename}'
        print('Logging to ' + logdir)
        self.writer = SummaryWriter(logdir)

    def zero_state_lstm(self, phase):
        n_directions = self.model.n_directions
        if phase == 'train':
            h_0 = torch.zeros((self.layers * n_directions, self.batch_size_train, self.hidden_size)).to(self.device)
            c_0 = torch.zeros((self.layers * n_directions, self.batch_size_train, self.hidden_size)).to(self.device)
        else:
            h_0 = torch.zeros((self.layers * n_directions, 1, self.hidden_size)).to(self.device)
            c_0 = torch.zeros((self.layers * n_directions, 1, self.hidden_size)).to(self.device)
        return h_0, c_0

    def zero_state_gru(self, phase):
        n_directions = self.model.n_directions
        if phase == 'train':
            hidden = torch.zeros((self.layers * n_directions, self.batch_size_train, self.hidden_size)).to(self.device)
        else:
            hidden = torch.zeros((self.layers * n_directions, 1, self.hidden_size)).to(self.device)
        return hidden

    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        best_accuracy = 0.0
        final_model = None

        for epoch in range(self.n_epochs):
            start_ts = dt.datetime.now()
            avg_loss = {'train': 0.0, 'test': 0.0}

            for phase in ['train', 'valid']:
                if self.rnn_type == 'LSTM':
                    h_n, c_n = self.zero_state_lstm(phase)
                else:
                    h_gru = self.zero_state_gru(phase)
                phase_loss = (0.0, 0)
                valid_accuracy = (0.0, 0)
                self.model = self.model.train(phase == 'train')

                for batch_idx, (X_batch, Y_batch) in enumerate(self.data[phase]):
                    with torch.set_grad_enabled(phase == 'train'):
                        X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                        optimizer.zero_grad()
                        if self.rnn_type == 'LSTM':
                            output, (h_n, c_n) = self.model(X_batch, (h_n, c_n))
                        else:
                            output, h_gru = self.model(X_batch, h_gru)
                        train_loss = self.model.compute_loss(output, Y_batch)
                        if phase == 'train':
                            train_loss.backward()
                            optimizer.step()
                        else:
                            va0 = valid_accuracy[0] + self.model.compute_accuracy(X_batch, Y_batch, output)
                            va1 = valid_accuracy[1] + 1
                            valid_accuracy = (va0, va1)

                        if self.rnn_type == 'LSTM':
                            h_n = h_n.detach()
                            c_n = c_n.detach()
                            if self.bidirectional:
                                h_n[1::2] = 0  # Ignore backward state as we are stepping forward from batch to batch.
                                c_n[1::2] = 0
                        else:
                            h_gru = h_gru.detach()
                            if self.bidirectional:
                                h_gru[1::2] = 0  # Ignore backward state as we are stepping forward from batch to batch.
                        phase_loss = (phase_loss[0] + train_loss, phase_loss[1] + 1)

                avg_loss[phase] = phase_loss[0] / phase_loss[1]
                self._save_summaries(avg_loss, epoch, phase, valid_accuracy)

            end_ts = dt.datetime.now()
            t_total = (end_ts - start_ts).total_seconds()
            if valid_accuracy[0] / valid_accuracy[1] > best_accuracy:
                best_accuracy = valid_accuracy[0] / valid_accuracy[1]
                final_model = copy.deepcopy(self.model)
                print('*')
            self._print_epoch_stats(epoch, t_total, avg_loss, valid_accuracy)

        self.model = final_model
        # plot_output(torch.softmax(output[0], dim=-1))

    def _save_summaries(self, avg_loss, epoch, phase, valid_accuracy):
        self.writer.add_scalar('loss/' + phase, avg_loss[phase], epoch)
        if phase == 'train':
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name + '/data', param.data.clone().cpu().numpy(), epoch)
                self.writer.add_histogram(name + '/grad', param.grad.clone().cpu().numpy(), epoch)
        else:
            self.writer.add_scalar('accuracy', valid_accuracy[0] / valid_accuracy[1], epoch)

    def _print_epoch_stats(self, epoch, t_total, avg_loss, valid_accuracy):
        filtered_acc = valid_accuracy[0] / valid_accuracy[1]
        print(f'Epoch: {epoch}/{self.n_epochs - 1}, {t_total:.1f} sec')
        print('-' * 22)
        print(f'Train loss = {avg_loss["train"]:.7f}')
        print(f'Valid loss = {avg_loss["valid"]:.7f}')
        print(f'Accuracy   = {filtered_acc:.1f}%', flush=True)
        print()
        print(flush=True)
