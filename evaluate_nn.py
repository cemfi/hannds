import argparse
import os
import json

import torch
import numpy as np

import hannds_data


def evaluate(model, training_args):
    device = torch.device('cuda' if torch.cuda.is_available() and training_args['cuda'] else 'cpu')
    model.to(device)

    x, y = _get_test(training_args, cv_partition=training_args['cv_partition'])
    x = torch.tensor(x)
    y = torch.tensor(y)
    hidden = _get_hidden(training_args, device)
    output, _ = model(x, hidden)

    n_notes = x.shape[1]
    accuracy = model.compute_accuracy(x, y, output)
    result = {
        'correct': accuracy,
        'n_notes': n_notes
    }
    return result


def _get_hidden(training_args, device):
    n_directions = 2 if training_args['bidirectional'] else 1
    hidden_size = int(training_args['hidden_size'])
    layers = int(training_args['layers'])
    if training_args['rnn_type'] == 'LSTM':
        h_0 = torch.zeros((layers * n_directions, 1, hidden_size)).to(device)
        c_0 = torch.zeros((layers * n_directions, 1, hidden_size)).to(device)
        return h_0, c_0
    if training_args['rnn_type'] == 'GRU':
        hidden = torch.zeros((layers * n_directions, 1, hidden_size)).to(device)
        return hidden

    raise ValueError('rnn_rype not supported' + training_args['rnn_type'])


def _get_test(training_args, cv_partition):
    if training_args['network'] == '88':
        _, _, test_data = hannds_data.train_valid_test_data_windowed(len_train_sequence=100, cv_partition=cv_partition)
    if training_args['network'] == '88Tanh':
        _, _, test_data = hannds_data.train_valid_test_data_windowed_tanh(len_train_sequence=100,
                                                                          cv_partition=cv_partition)
    if training_args['network'] == 'MIDI':
        _, _, test_data = hannds_data.train_valid_test_data_event(len_train_sequence=100, cv_partition=cv_partition)
    if training_args['network'] == 'Magenta':
        _, _, test_data = hannds_data.train_valid_test_data_magenta(len_train_sequence=100, cv_partition=cv_partition)

    x, y = test_data[0]
    return x[np.newaxis], y[np.newaxis]


def models_in_dir(path):
    models = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    models.sort()
    return models


def main(args):
    idx = 0
    dirs = models_in_dir(args['models_path'])
    all_results = []
    for main_folder in dirs:
        print(main_folder)
        model_path = os.path.join(main_folder, 'model.pt')
        args_path = os.path.join(main_folder, 'args.json')
        with open(args_path, 'r') as f:
            training_args = json.load(f)['args']

        model = torch.load(model_path, map_location='cpu')
        result = evaluate(model, training_args)
        print(result)
        all_results.append({
            'train': training_args,
            'result': result
        })
        print('')

    result_dict = {}
    for r in all_results:
        key = f"{r['train']['network']} ({r['train']['rnn_type']}) bidirectional={r['train']['bidirectional']}"
        if key not in result_dict:
            result_dict[key] = [r['result']]
        else:
            result_dict[key].append(r['result'])

    for key, values in result_dict.items():
        total_notes = 0
        total_correct = 0.0
        for v in values:
            total_correct += v['correct'] / 100.0 * v['n_notes']
            total_notes += v['n_notes']

        accuracy = total_correct / total_notes * 100.0
        print(f'{key};{accuracy:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn hannds neural net')
    parser.add_argument('--models_path', metavar='PATH', type=str, required=True, help='path to the model')
    parser.add_argument('--cuda', action='store_true', required=False, help='use CUDA')
    args = parser.parse_args()
    main(vars(args))
