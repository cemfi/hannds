import glob
import logging
import math
import os
from collections import namedtuple

import pretty_midi
import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(level=logging.DEBUG)


def get_files_from_path(path, extensions):
    if os.path.isfile(path):  # Load single file
        files = [path]
    else:  # Get list of all files with correct extensions in path
        files = []
        for file_type in extensions:
            files.extend(glob.glob(os.path.join(path, file_type)))

        if len(files) == 0:
            raise FileNotFoundError('No files found with correct extensions ' + str(extensions))

    return files


def convert(path, ms_window=20, overwrite=True):
    midi_files = get_files_from_path(path, ['*.mid', '*.midi'])

    samples_per_sec = 1000 // ms_window
    for midi_file in midi_files:
        npy_file = midi_file + '_' + str(ms_window) + 'ms' + '.npy'

        if overwrite or not os.path.exists(npy_file):
            midi = pretty_midi.PrettyMIDI(midi_file)
            logging.debug("Converting file '" + midi_file + "'")
            midi_data = midi.instruments[0], midi.instruments[1]

            # Generate empty numpy arrays
            n_windows = math.ceil(midi.get_end_time() * samples_per_sec)
            hands = np.zeros((
                n_windows,  # Number of windows to calculate
                2,  # Left and right hand = 2 hands
                88  # 88 keys on a piano
            ), dtype=np.bool)

            # Fill array with data
            for hand, midi_hand in enumerate(midi_data):
                for note in midi_hand.notes:
                    start = int(math.floor(note.start * samples_per_sec))
                    end = int(math.ceil(note.end * samples_per_sec))
                    hands[start:end, hand, note.pitch - 21] = True

            # Save array to disk
            np.save(npy_file, hands)


def train_test_data(path, len_sequence_train, debug=False):
    """
    returns training and test data. Test data will always be of the
    shape (1, -1, 88), which is a sequence of maximal length.
    """
    npy_files = get_files_from_path(path, ['*.npy'])
    if debug:
        npy_data = np.load(npy_files[0])
    else:
        npy_data = np.concatenate([np.load(npy_file) for npy_file in npy_files], axis=0)
    split = math.floor(npy_data.shape[0] * 0.7)
    data_train, data_test = npy_data[:split], npy_data[split:]
    train = HanndsDataset(data_train, len_sequence_train)
    test = HanndsDataset(data_test, -1)
    return train, test


XY = namedtuple('XY', ['X', 'Y'])


class HanndsDataset(Dataset):
    """
    provides the Hannds dataset as (overlapping) sequences of size
    len_sequence. If len_sequenc == -1, it provides a single sequence
    of maximal length.
    """

    def __init__(self, npy_data, len_sequence):
        self.len_sequence = len_sequence
        self.data = XY(*self._compute_X_Y(npy_data))

    def _compute_X_Y(self, data):
        data = data.astype(np.bool)

        batch_size = data.shape[0]
        # Merge both hands in a single array
        X = np.logical_or(
            data[:, 0, :],
            data[:, 1, :]
        )

        # Mark if both hands are played simultaneously
        both = np.logical_and(
            data[:, 0, :],
            data[:, 1, :]
        )

        # Return last played window of every sample:
        #    -1 => left hand
        #    +1 => right hand
        #     0 => both hands / hands
        no_hand_value = 0.0  # nan if you want to distinguish both hands / no hand
        Y = np.full((batch_size, 88), 0.0)
        Y[data[:, 0, :]] = +1
        Y[data[:, 1, :]] = -1
        Y[both] = 0
        return X.astype(np.float32), Y.astype(np.float32)

    def __len__(self):
        if self.len_sequence == -1:
            return 1
        else:
            return (math.floor(self.data.X.shape[0] / self.len_sequence) - 1) * self.len_sequence

    def __getitem__(self, idx):
        if self.len_sequence == -1:
            return self.data.X, self.data.Y
        else:
            return self.data.X[idx: idx + self.len_sequence], self.data.Y[idx: idx + self.len_sequence]


def main():
    import matplotlib.pyplot as plt

    convert(path='../data/hannds', ms_window=20, overwrite=True)
    data, _ = train_test_data('../data/hannds', 100)

    batchX, batchY = data[0]
    print(batchX.shape)
    print(batchY.shape)

    # for i in range(5):
    #     X, Y = data[i]
    #     img = np.full((X.shape[0] + 2, X.shape[1]), -0.2)
    #     img[:-2] = X
    #     img[-1] = Y[-1]
    #
    #     plt.imshow(img, cmap='bwr', origin='lower', vmin=-1, vmax=1)
    #     plt.show()


if __name__ == '__main__':
    convert('data/', overwrite=False)
    main()
