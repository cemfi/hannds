"""Trains a neural network for hand mapping."""

import argparse
import datetime as dt
import json
import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch

import hannds_data as hd
from network_zoo import Network88, Network88Tanh, NetworkMidi, NetworkMagenta
import rnntrainer


g_time = dt.datetime.now().strftime('%m-%d-%H%M')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args['cuda'] else 'cpu')
    print(f"Using {device}", flush=True)

    if args['network'] == '88Tanh':
        train_data, valid_data, _ = \
            hd.train_valid_test_data_windowed_tanh(len_train_sequence=100, cv_partition=args['cv_partition'],
                                                   debug=args['debug'])
        num_features = train_data.len_features()
        model = Network88Tanh(args['hidden_size'], args['layers'], args['bidirectional'], num_features,
                              args['rnn_type']).to(device)
    elif args['network'] == '88':
        train_data, valid_data, _ = \
            hd.train_valid_test_data_windowed(len_train_sequence=100, cv_partition=args['cv_partition'],
                                              debug=args['debug'])
        num_features = train_data.len_features()
        num_categories = train_data.num_categories()
        model = Network88(args['hidden_size'], args['layers'], args['bidirectional'],
                          num_features, num_categories, args['rnn_type']).to(device)
    elif args['network'] == 'MIDI':
        train_data, valid_data, _ = \
            hd.train_valid_test_data_event(len_train_sequence=100, cv_partition=args['cv_partition'],
                                           debug=args['debug'])
        model = NetworkMidi(args['hidden_size'], args['layers'], args['rnn_type'], args['bidirectional']).to(device)
    elif args['network'] == 'Magenta':
        train_data, valid_data, _ = \
            hd.train_valid_test_data_magenta(len_train_sequence=100, cv_partition=args['cv_partition'],
                                             debug=args['debug'])
        model = NetworkMagenta(args['hidden_size'], args['layers'], args['rnn_type'], args['bidirectional']).to(device)

    else:
        raise Exception('Invalid --network argument')

    t = rnntrainer.RNNTrainer(model, train_data, valid_data, args, device)

    t.run()
    model = t.model
    if not os.path.exists('models'):
        os.mkdir('models')

    directory = f'models/{g_time}-{t.descriptive_filename}'
    os.mkdir(directory)
    torch.save(model, os.path.join(directory, 'model.pt'))
    desc = {
        'args': args
    }
    with open(directory + '/args.json', 'w') as file:
        json.dump(desc, file, indent=4)


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
    parser.add_argument('--network', metavar='NET', type=str,
                        help='which network to train. Use "88", "88Tanh", Magenta or "MIDI"')
    parser.add_argument('--rnn_type', metavar='RNN_TYPE', type=str,
                        help='which type of RNN to use. Use "GRU" or "LSTM".')
    args = parser.parse_args()
    main(vars(args))
