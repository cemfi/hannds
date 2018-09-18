import math
from collections import namedtuple
import os

import numpy as np
import pretty_midi
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import hannds_files


def train_valid_test_data_windowed(len_train_sequence, cv_partition=1, debug=False):
    make_npz_files(overwrite=False, subdir='windowed', convert_func=convert_windowed)
    all_files = hannds_files.TrainValidTestFiles()
    all_files.read_files_from_dir(cv_partition)
    train_data = HanndsDataset(all_files.train_files, 'windowed', len_sequence=len_train_sequence, debug=debug)
    valid_data = HanndsDataset(all_files.valid_files, 'windowed', len_sequence=-1, debug=debug)
    test_data = HanndsDataset(all_files.test_files, 'windowed', len_sequence=-1, debug=debug)
    return train_data, valid_data, test_data


def train_valid_test_data_event(len_train_sequence, cv_partition=1, debug=False):
    make_npz_files(overwrite=False, subdir='event', convert_func=convert_event)
    all_files = hannds_files.TrainValidTestFiles()
    all_files.read_files_from_dir(cv_partition)
    train_data = HanndsDataset(all_files.train_files, 'event', len_sequence=len_train_sequence, debug=debug)
    valid_data = HanndsDataset(all_files.valid_files, 'event', len_sequence=-1, debug=debug)
    test_data = HanndsDataset(all_files.test_files, 'event', len_sequence=-1, debug=debug)
    return train_data, valid_data, test_data


WINDOWED_NOT_PLAYED_LABEL = 0
WINDOWED_LEFT_HAND_LABEL = 1
WINDOWED_RIGHT_HAND_LABEL = 2


def make_npz_files(overwrite, subdir, convert_func):
    midi_files = hannds_files.all_midi_files(absolute_path=True)
    npy_paths = hannds_files.npz_files_for_midi(midi_files, subdir)

    for midi_file, npy_path in zip(midi_files, npy_paths):
        if overwrite or not os.path.exists(npy_path):
            print("Converting file '" + midi_file + "'")
            midi = pretty_midi.PrettyMIDI(midi_file)
            X, Y = convert_func(midi)
            np.savez(npy_path, X=X, Y=Y)


def convert_windowed(midi):
    ms_window = 20
    samples_per_sec = 1000 // ms_window
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

    data = hands
    batch_size = n_windows
    # Merge both hands in a single array
    X = np.logical_or(
        data[:, 0, :],
        data[:, 1, :]
    )

    Y = np.zeros((batch_size, 88))  # == WINDOWED_NOT_PLAYED_LABEL
    Y[data[:, 0, :]] = WINDOWED_LEFT_HAND_LABEL
    Y[data[:, 1, :]] = WINDOWED_RIGHT_HAND_LABEL

    return X.astype(np.float32), Y.astype(np.longlong)


def convert_event(midi):
    num_notes = 0
    for instrument in midi.instruments:
        num_notes += len(instrument.notes)

    # # Generate empty numpy array
    events = np.empty((2 * num_notes, 1 + 1 + 3))

    # Generate event list
    # Format:[    0    ,        1      ,     2   ,    3  ,     4     ]
    #        [timestamp, midi_pitch/127, is_start, is_end, left|right]
    i = 0
    for hand, instrument in enumerate(midi.instruments):
        notes = instrument.notes
        for note in notes:
            events[i:i + 2, 1] = note.pitch / 127
            events[i:i + 2, 4] = hand  # 0 = Right, 1 = Left

            events[i, 0] = note.start  # Timestamp note on
            events[i, 2:4] = [1, 0]  # One hot vector for note on

            events[i + 1, 0] = note.end  # Timestamp note off
            events[i + 1, 2:4] = [0, 1]  # One hot vector for note off

            i += 2

    # Compute timestamp deltas
    events = events[events[:, 0].argsort()]  # Sort by column 0
    events[1:, 0] = np.diff(events[:, 0])
    events[0, 0] = 0  # Find something more suitable for the first entry
    events[:, 0] = np.maximum(events[:, 0], 0)  # Don't allow negative time deltas (happens at file borders)

    return events[:, :4].astype(np.float32), events[:, 4].astype(np.longlong)


class HanndsDataset(Dataset):
    """
    Provides the Hannds dataset as sequences of size len_sequence.
    If len_sequenc == -1, it provides a single sequence of maximal
    length.
    """

    XY = namedtuple('XY', ['X', 'Y'])

    def __init__(self, midi_files, subdir, len_sequence, debug):
        self.len_sequence = len_sequence
        npz_files = hannds_files.npz_files_for_midi(midi_files, subdir)
        if debug:
            load_all = [np.load(npz_file) for npz_file in npz_files[:2]]
        else:
            load_all = [np.load(npz_file) for npz_file in npz_files]

        X = np.concatenate([item['X'] for item in load_all], axis=0)
        Y = np.concatenate([item['Y'] for item in load_all], axis=0)
        self.data = self.XY(X, Y)

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

    def len_features(self):
        return self.data.X.shape[1]

    def num_categories(self):
        return np.max(self.data.Y) + 1


class ContinuationSampler(Sampler):

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

        return


def main():
    print('Making windowed')
    make_npz_files(overwrite=True, subdir='windowed', convert_func=convert_windowed)
    print()
    print('Making event')
    make_npz_files(overwrite=True, subdir='event', convert_func=convert_event)
    print()

    f = hannds_files.TrainValidTestFiles()
    f.read_files_from_dir()
    data = HanndsDataset(f.train_files, 'windowed', 100, debug=False)

    import matplotlib.pyplot as plt

    batchX, batchY = data[0]

    batch_size = 50
    continuity = ContinuationSampler(len(data), batch_size)
    loader = DataLoader(data, batch_size, sampler=continuity)

    for idx, (X_batch, Y_batch) in enumerate(loader):
        X = X_batch[8]
        Y = Y_batch[8]
        img = np.full((X.shape[0] + 2, X.shape[1]), -0.2)
        img[:-2] = X
        img[-1] = Y[-1, :] - 1.0

        plt.imshow(img, cmap='bwr', origin='lower', vmin=-1, vmax=1)
        plt.show()

        if idx == 5: break


if __name__ == '__main__':
    main()
