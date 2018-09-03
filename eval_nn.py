import math
import argparse

import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from hannds_data import train_test_data
from train_nn import FFNetwork, RNN, LSTM

DEBUG_MODE = True


def main():
    parser = argparse.ArgumentParser(description='Learn hannds neural net')
    parser.add_argument('--model', metavar='Path', type=str, required=True, help='path to .pt file')
    args = parser.parse_args()

    _, test_dataset = train_test_data('data/', len_sequence_train=-1, len_sequence_test=-1, debug=DEBUG_MODE)
    validate_data = DataLoader(test_dataset, batch_size=100_000)
    model = torch.load(args.model).cpu()
    input, output, labels, loss = compute_result(model, validate_data)
    print(f"Log validation loss = {math.log(loss):.2f}")
    evaluate_model(input, output, labels)


def compute_result(model, validate_data):
    with torch.no_grad():
        X_batch, Y_batch = iter(validate_data).next()
        output, *_ = model(X_batch)
        output = output.squeeze()
        X_batch = X_batch.squeeze()
        Y_batch = Y_batch.squeeze()
        criterion = nn.MSELoss()
        test_loss = criterion(output, Y_batch)
        return X_batch.numpy(), output.numpy(), Y_batch.numpy(), test_loss


def evaluate_model(input, output, labels):
    output = output * input

    with PdfPages('results.pdf') as pdf:
        for i in reversed(range(32)):
            if (i + 1) * 100 < output.shape[0]:
                fig, ax = plt.subplots()
                ax.imshow(output[i * 100: (i + 1) * 100], cmap='bwr', origin='lower', vmin=-1, vmax=1)
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
