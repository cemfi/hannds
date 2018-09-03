import math
import sys
import argparse

import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from hannds_data import train_test_data
import train_nn
from train_nn import FFNetwork, RNN, LSTM

DEBUG_MODE = False


def main():
    parser = argparse.ArgumentParser(description='Learn hannds neural net')
    parser.add_argument('--model', metavar='Path', type=str, required=True, help='path to .pt file')
    args = parser.parse_args()

    _, test_dataset = train_test_data('data/', len_sequence_train=-1, debug=DEBUG_MODE)
    print(f"Test set length = {len(test_dataset)}")
    validate_data = DataLoader(test_dataset, batch_size=100_000)
    model = torch.load(args.model)
    input, output, labels, loss = compute_result(model, validate_data)
    print(f"Log validation loss = {math.log(loss):.2f}")
    evaluate_model(input, output, labels)


def compute_result(model, validate_data):
    with torch.no_grad():
        inputs = None
        labels = None
        outputs = None
        total_loss = 0.0
        n = 0.0
        for X_batch, Y_batch in validate_data:
            sys.stdout.write('.')
            if n % 80 == 79:
                sys.stdout.write('\n')
            sys.stdout.flush()
            output, *_ = model(X_batch)
            if inputs is None:
                inputs = X_batch
                labels = Y_batch[:, -1]
                outputs = output
            else:
                inputs = torch.cat((inputs, X_batch), dim=0)
                labels = torch.cat((labels, Y_batch[:, -1]), dim=0)
                outputs = torch.cat((outputs, output), dim=0)

            criterion = nn.MSELoss()
            test_loss = criterion(output[:, -1], Y_batch[:, -1])
            total_loss += test_loss
            n += 1
        total_loss /= n
        print("")
    return inputs.numpy(), outputs.numpy(), labels.numpy(), total_loss


def evaluate_model(input, output, labels):
    output = output[:, -1, :]
    input = input[:, -1, :]
    output = output * input

    with PdfPages('results.pdf') as pdf:
        for i in reversed(range(32)):
            fig, ax = plt.subplots()
            ax.imshow(output[i * 100:(i + 1) * 100], cmap='bwr', origin='lower', vmin=-1, vmax=1)
            pdf.savefig(fig)
            plt.close()

    num_notes = np.sum(input)
    left_hand = output < 0.0
    right_hand = output > 0.0
    output = left_hand * -1.0 + right_hand * 1.0
    diff = (output != labels)
    errors_percent = np.sum(diff) / num_notes * 100.0
    print(f"{errors_percent:.2f}% errors")


if __name__ == '__main__':
    main()
