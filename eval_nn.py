import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import hannds_data
from train_nn import Network
import train_nn


def main():
    parser = argparse.ArgumentParser(description='Evaluate hannds neural net')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to base directory where the .pt-file is contained.')
    args = parser.parse_args()

    desc_path = os.path.join(args.model, 'desc.json')
    model_path = os.path.join(args.model, 'model.pt')

    with open(desc_path) as file:
        desc = json.load(file)

    data = hannds_data.AllData()
    data.initialize_from_lists(desc['train'], desc['valid'], desc['test'], len_train_sequence=1)
    test_data = data.test_data

    model = torch.load(model_path, map_location='cpu')
    model.eval()

    loader = DataLoader(test_data)
    X, Y = next(iter(loader))
    output, _, _ = model.forward(X)
    predicted_classes = torch.argmax(output, dim=3)
    accuracy = train_nn.compute_accuracy(X, Y, predicted_classes)
    print(f'Test accuracy = {accuracy:.1f}')


if __name__ == '__main__':
    main()
