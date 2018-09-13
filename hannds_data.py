import glob
import logging
import math
import os
from collections import namedtuple

import numpy as np
import pretty_midi
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

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


def train_valid_test(path, len_sequence, debug=False):
    """
    returns training and test data. If len_sequence == 1 sequences of
    shape (1, -1, 88), which is a sequence of maximal length, will be
    returned. This is always the shape of the valid and test data.
    """
    npy_files = get_files_from_path(path, ['*.npy'])
    if debug:
        npy_data = np.concatenate([np.load(npy_file) for npy_file in npy_files[:2]], axis=0)
    else:
        npy_data = np.concatenate([np.load(npy_file) for npy_file in npy_files], axis=0)

    split_1 = math.floor(npy_data.shape[0] * 0.7)
    split_2 = math.floor(npy_data.shape[0] * 0.9)
    train_npy, valid_npy, test_npy = npy_data[:split_1], npy_data[split_1: split_2], npy_data[split_2:]
    train_data = HanndsDataset(train_npy, len_sequence)
    valid_data = HanndsDataset(valid_npy, -1)
    test_data = HanndsDataset(test_npy, -1)

    return train_data, valid_data, test_data


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
        Y = np.full((batch_size, 88), no_hand_value)
        Y[data[:, 0, :]] = +1
        Y[data[:, 1, :]] = -1
        Y[both] = 0
        return X.astype(np.float32), Y.astype(np.float32)

    def __len__(self):
        if self.len_sequence == -1:
            return 1
        else:
            return self.data.X.shape[0] // self.len_sequence - 1

    def __getitem__(self, idx):
        if self.len_sequence == -1:
            return self.data.X, self.data.Y
        else:
            start = idx * self.len_sequence
            end = start + self.len_sequence
            res1 = self.data.X[start: end]
            res2 = self.data.Y[start: end]
            assert res1.shape[0] == res2.shape[0] == self.len_sequence
            return res1, res2


class ContinuitySampler(Sampler):

    def __init__(self, len_dataset, batch_size):
        Sampler.__init__(self, None)
        self.len_dataset = len_dataset
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self._generate_indices())

    def __len__(self):
        num_batches = self.len_dataset // self.batch_size
        return num_batches * self.batch_size

    def _generate_indices(self):
        num_batches = step = self.len_dataset // self.batch_size
        for i in range(num_batches):
            index = i
            for j in range(self.batch_size):
                yield index
                index += step

        raise StopIteration


def main():
    convert('data/', overwrite=False)

    import matplotlib.pyplot as plt

    data, _, _ = train_valid_test('data/', len_sequence=100, debug=True)
    batchX, batchY = data[0]
    print(batchX.shape)
    print(batchY.shape)

    batch_size = 20
    continuity = ContinuitySampler(len(data), batch_size)
    loader = DataLoader(data, batch_size, sampler=continuity)

    for idx, (X_batch, Y_batch) in enumerate(loader):
        X = X_batch[10]
        Y = Y_batch[10]
        img = np.full((X.shape[0] + 2, X.shape[1]), -0.2)
        img[:-2] = X
        img[-1] = Y[-1]

        plt.imshow(img, cmap='bwr', origin='lower', vmin=-1, vmax=1)
        plt.show()


if __name__ == '__main__':
    main()
