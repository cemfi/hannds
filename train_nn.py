"""Trains a neural network for hand mapping."""

import argparse
import copy
import datetime as dt
import json
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader

import hannds_data as hd
from network_zoo import Network88, Network88Tanh

# global vars
g_time = dt.datetime.now().strftime('%m-%d-%H%M')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args['cuda'] else 'cpu')
    print(f"Using {device}", flush=True)

    if args['network'] == '88Tanh':
        train_data, valid_data, _ = \
            hd.train_valid_test_data_windowed_tanh(len_train_sequence=100, cv_partition=args['cv_partition'],
                                                   debug=args['debug'])
        num_features = train_data.len_features()
        model = Network88Tanh(args['hidden_size'], args['layers'], args['bidirectional'], num_features).to(device)
    elif args['network'] == '88':
        train_data, valid_data, _ = \
            hd.train_valid_test_data_windowed(len_train_sequence=100, cv_partition=args['cv_partition'],
                                              debug=args['debug'])
        num_features = train_data.len_features()
        num_categories = train_data.num_categories()
        model = Network88(args['hidden_size'], args['layers'], args['bidirectional'],
                          num_features, num_categories).to(device)
    else:
        raise Exception('Invalid --network argument')

    trainer = Trainer(model, train_data, valid_data, args, device)

    trainer.run()
    model = trainer.model
    if not os.path.exists('models'):
        os.mkdir('models')

    directory = f'models/{g_time}'
    os.mkdir(directory)
    torch.save(model, os.path.join(directory, 'model.pt'))
    desc = {
        'args': args
    }
    with open(directory + '/desc.json', 'w') as file:
        json.dump(desc, file, indent=4)


class Trainer(object):
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

        bi_str = '-bidirectional' if self.bidirectional else ''
        desc = f"hidden{args['hidden_size']}_layers{args['layers']}{bi_str}"
        self.writer = SummaryWriter(f'runs/{g_time}-p{os.getpid()}-{desc}')

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        best_accuracy = 0.0
        final_model = None

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
                        optimizer.zero_grad()
                        output, h_n, c_n = self.model(X_batch, h_n, c_n)
                        train_loss = self.model.compute_loss(output, Y_batch)
                        if phase == 'train':
                            train_loss.backward()
                            optimizer.step()
                        else:
                            va0 = valid_accuracy[0] + self.model.compute_accuracy(X_batch, Y_batch, output)
                            va1 = valid_accuracy[1] + 1
                            valid_accuracy = (va0, va1)

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
        print(f'Epoch: {epoch}/{self.n_epochs - 1}, {t_total:.1f}sec')
        print('-' * 21)
        print(f'Train loss = {avg_loss["train"]:.6f}')
        print(f'Valid loss = {avg_loss["valid"]:.6f}')
        print(f'Accuracy   = {filtered_acc:.1f}%', flush=True)
        print()
        print(flush=True)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn hannds neural net')
    parser.add_argument('--hidden_size', metavar='N', type=int, required=True, help='number of hidden units per layer')
    parser.add_argument('--layers', metavar='N', type=int, required=True, help='numbers of layers')
    parser.add_argument('--length', metavar='N', type=int, required=True, help='sequence length used in training')
    parser.add_argument('--cuda', action='store_true', required=False, help='use CUDA')
    parser.add_argument('--bidirectional', action='store_true', required=False, help='use a bi-directional LSTM')
    parser.add_argument('--debug', action='store_true', required=False, help='run with minimal data')
    parser.add_argument('--cv_partition', metavar='N', type=int, required=False, default=1,
                        help='the partition index (from 1 to 10) for 10-fold cross validation')
    parser.add_argument('--network', metavar='NET', type=str, default='88',
                        help='which network to train. Use "88" or "88Tanh"')
    args = parser.parse_args()
    main(vars(args))
