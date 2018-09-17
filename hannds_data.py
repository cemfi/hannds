import glob
import logging
import math
import os
import random
from collections import namedtuple

import numpy as np
import pretty_midi
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

logging.basicConfig(level=logging.WARNING)

g_package_directory = os.path.dirname(os.path.abspath(__file__))


# Helpers

def get_files_from_path(path, extensions):
    if os.path.isfile(path):  # Load single file
        files = [path]
    else:  # Get list of all files with correct extensions in path
        files = []
        for file_type in extensions:
            files.extend(glob.glob(os.path.join(path, file_type)))

        if len(files) == 0:
            raise FileNotFoundError(f'No files found with correct extensions {str(extensions)} in {path}')
    return sorted(files)


class AllData(object):
    def __init__(self, debug=False):
        self._convert(os.path.join(g_package_directory, 'data'), overwrite=False)
        self.train_files = self.valid_files = self.test_files = None
        self.debug = debug

    def initialize_from_dir(self, len_train_sequence, cv_partition=1):
        all_files = get_files_from_path(os.path.join(g_package_directory, 'data'), ['*.npy'])
        r = random.Random(42)  # seed is arbitrary
        r.shuffle(all_files)

        test_begin, test_end = self._hold_out_range(cv_partition, len(all_files))
        n_valid = math.ceil(len(all_files) * 0.2)
        if test_end + n_valid < len(all_files):
            valid_begin = test_end
            valid_end = valid_begin + n_valid
        else:
            valid_end = test_begin
            valid_begin = valid_end - n_valid

        self.test_files = all_files[test_begin: test_end]
        self.valid_files = all_files[valid_begin: valid_end]
        self.train_files = [f for f in all_files if f not in self.test_files and f not in self.valid_files]

        self._make_datasets(len_train_sequence)

    def initialize_from_lists(self, train_files, valid_files, test_files, len_train_sequence):
        self.train_files = train_files.copy()
        self.valid_files = valid_files.copy()
        self.test_files = test_files.copy()
        self._make_datasets(len_train_sequence)

    def _n_hold_out(self, cv_partition, n_files):
        """
        How many MIDI files should be held out in cross validation step
        cv_step. Step index starts with 1.
        """
        assert cv_partition >= 1
        for step in range(1, cv_partition):
            n_in_past_step = math.ceil(n_files / (10 - step + 1))
            n_files -= n_in_past_step
        return math.ceil(n_files / (10 - cv_partition + 1))

    def _hold_out_range(self, cv_partition, n_files):
        begin = 0
        for step in range(1, cv_partition):
            begin += self._n_hold_out(step, n_files)

        end = begin + self._n_hold_out(cv_partition, n_files)
        return begin, end

    def _make_datasets(self, len_train_sequence):
        self.train_data = self._dataset_for_files(self.train_files, len_train_sequence, debug=self.debug)
        self.valid_data = self._dataset_for_files(self.valid_files, len_sequence=-1, debug=self.debug)
        self.test_data = self._dataset_for_files(self.test_files, len_sequence=-1, debug=self.debug)

    def _convert(self, path, ms_window=20, overwrite=True):
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

    def _dataset_for_files(self, npy_files, len_sequence, debug=False):
        if debug:
            npy_data = np.concatenate([np.load(npy_file) for npy_file in npy_files[:2]], axis=0)
        else:
            npy_data = np.concatenate([np.load(npy_file) for npy_file in npy_files], axis=0)

        data_set = HanndsDataset(npy_data, len_sequence)
        return data_set


XY = namedtuple('XY', ['X', 'Y'])

NOT_PLAYED_LABEL = 0
LEFT_HAND_LABEL = 1
RIGHT_HAND_LABEL = 2


class HanndsDataset(Dataset):
    """
    Provides the Hannds dataset as (overlapping) sequences of size
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

        Y = np.zeros((batch_size, 88))
        Y[data[:, 0, :]] = LEFT_HAND_LABEL
        Y[data[:, 1, :]] = RIGHT_HAND_LABEL
        return X.astype(np.float32), Y.astype(np.longlong)

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
    data = AllData(debug=True)
    data.initialize_from_dir(len_train_sequence=100)

    import matplotlib.pyplot as plt

    train_data = data.train_data
    batchX, batchY = train_data[0]

    batch_size = 50
    continuity = ContinuationSampler(len(train_data), batch_size)
    loader = DataLoader(train_data, batch_size, sampler=continuity)

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
