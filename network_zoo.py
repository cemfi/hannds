"""
Collection of different neural network architectures together with
architecture-specific helper functions.
"""

from collections import Counter

from torch import nn
import torch
import numpy as np

import hannds_data as hd


class Network88(nn.Module):
    def __init__(self, hidden_size, n_layers, bidirectional, n_features, n_categories):
        super(Network88, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=0.5, bidirectional=bidirectional)
        self.n_directions = 2 if bidirectional else 1
        self.out_linear = nn.Linear(hidden_size * self.n_directions, n_features * n_categories)
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_categories = n_categories
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, h_prev=None, c_prev=None):
        hidden_in = (h_prev, c_prev) if h_prev is not None else None
        lstm_output, (h_n, c_n) = self.lstm.forward(input, hidden_in)
        output = self.out_linear(lstm_output)
        output = output.view(-1, output.shape[1], self.n_features, self.n_categories)
        # Residual connection
        # output[:, :, :, 0] += 1.0 - input
        # output[:, :, :, 1] += input
        # output[:, :, :, 2] += input
        return output, h_n, c_n

    def compute_loss(self, output, labels):
        return self.criterion(output.view((-1, self.n_categories)), labels.view(-1))

    def compute_accuracy(self, X_batch, Y_batch, prediction):
        predicted_classes = torch.argmax(prediction, dim=3)
        filter_func = causal_filter
        X_numpy = X_batch.squeeze().cpu().numpy()
        Y_numpy = Y_batch.squeeze().cpu().numpy()
        classes_numpy = predicted_classes.squeeze().cpu().numpy()
        return compute_accuracy88(X_numpy, Y_numpy, classes_numpy, filter_func)
        # va1 = valid_accuracy_plain[0] + compute_accuracy88(X_numpy, Y_numpy, classes_numpy)
        # va2 = valid_accuracy_plain[1] + 1
        # valid_accuracy_plain = (va1, va2)


class Network88Tanh(nn.Module):
    def __init__(self, hidden_size, n_layers, bidirectional, n_features):
        super(Network88Tanh, self).__init__()
        self.lstm = nn.LSTM(input_size=88, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=0.5, bidirectional=bidirectional)
        self.n_directions = 2 if bidirectional else 1
        self.out_linear = nn.Linear(hidden_size * self.n_directions, n_features)
        self.n_layers = n_layers
        self.criterion = nn.MSELoss()

    def forward(self, input, h_prev, c_prev):
        lstm_output, (h_n, c_n) = self.lstm.forward(input, (h_prev, c_prev))
        output = self.out_linear(lstm_output)
        output = torch.tanh(output)
        output = output * input
        return output, h_n, c_n

    def compute_loss(self, output, labels):
        return self.criterion(output, labels)

    def compute_accuracy(self, X_batch, Y_batch, prediction):
        num_notes = torch.sum(X_batch)
        left_hand = (prediction < 0.0).float()
        right_hand = (prediction > 0.0).float()
        prediction = left_hand * -1.0 + right_hand * 1.0
        prediction = causal_filter(prediction)
        diff = (prediction != Y_batch).float()
        errors_percent = torch.sum(diff) / num_notes * 100.0
        return 100.0 - errors_percent


class NetworkMidi(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(NetworkMidi, self).__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_features=1)
        self.n_directions = 1
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input, hidden):
        out, hidden_tmp = self.gru(input, hidden)
        out = self.linear(out)
        return out, hidden_tmp

    def compute_loss(self, output, labels):
        return self.criterion(output.view(-1), labels.view(-1))

    def compute_accuracy(self, X_batch, Y_batch, prediction):
        labels_flat = Y_batch.view(-1)
        prediction_flat = (torch.sigmoid(prediction.view(-1)) >= 0.5).float()
        num_events = float(labels_flat.shape[0])
        num_correct = float(torch.sum(labels_flat == prediction_flat))
        return num_correct / num_events * 100.0


def causal_filter(predicted_classes, label_not_played=0):
    """
    Filters the predicted classes: the assignment is decided at the
    note-on event. Changes the content of predicted_classes.
    """
    for row in range(1, predicted_classes.shape[0]):
        last_line = predicted_classes[row - 1]
        current_line = predicted_classes[row]
        both_note_on = np.logical_and(current_line != label_not_played, last_line != label_not_played)
        predicted_classes[row, both_note_on] = last_line[both_note_on]

    return predicted_classes


def majority_filter(predicted_classes):  # Needs to be profiled
    """
    Filters the predicted classes: the assignment is decided at the
    note-off event. Changes the content of predicted_classes.
    """
    for column in range(predicted_classes.shape[1]):
        row = 0
        while row < predicted_classes.shape[0]:
            last_note = predicted_classes[row - 1, column] if row > 0 else 0
            current_note = predicted_classes[row, column]
            start = row
            if last_note == hd.WINDOWED_NOT_PLAYED and current_note != hd.WINDOWED_NOT_PLAYED:
                row += 1
                while row < predicted_classes.shape[0]:
                    current_note = predicted_classes[row, column]
                    if current_note == hd.WINDOWED_NOT_PLAYED or row == predicted_classes.shape[0] - 1:
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


def compute_accuracy88(X, Y, predicted_classes, filter_func=lambda x: x):
    assert len(X.shape) == len(Y.shape) == len(predicted_classes.shape) == 2
    n_notes = X.sum()
    predicted_classes = filter_func(predicted_classes)
    pred_lh = predicted_classes == hd.WINDOWED_LEFT_HAND
    pred_rh = predicted_classes == hd.WINDOWED_RIGHT_HAND
    label_lh = Y == hd.WINDOWED_LEFT_HAND
    label_rh = Y == hd.WINDOWED_RIGHT_HAND
    n_lh_correct = (pred_lh * label_lh).sum()
    n_rh_correct = (pred_rh * label_rh).sum()
    assert n_lh_correct <= n_notes
    assert n_rh_correct <= n_notes
    n_correct = n_lh_correct + n_rh_correct
    assert n_correct <= n_notes
    return n_correct / n_notes * 100.0
