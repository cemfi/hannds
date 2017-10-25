import glob
import logging
import math
import os
import random

import pretty_midi
import numpy as np

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

    samples_per_sec = (1000 / ms_window)
    for midi_file in midi_files:
        npy_file = midi_file + '_' + str(ms_window) + 'ms' + '.npy'

        if overwrite or not os.path.exists(npy_file):
            midi = pretty_midi.PrettyMIDI(midi_file)
            logging.debug('Converting file \'' + midi_file + '\'...')
            midi_data = midi.instruments[0], midi.instruments[1]

            # Generate empty numpy arrays
            n_windows = math.ceil(midi.get_end_time() * samples_per_sec)
            hands = np.zeros((
                88,         # 88 keys on a piano
                n_windows,  # Number of windows to calculate
                2           # Left and right hand = 2 hands
            ), dtype=np.bool)

            # Fill array with data
            for hand, midi_hand in enumerate(midi_data):
                for note in midi_hand.notes:
                    start = int(math.floor(note.start * samples_per_sec))
                    end = int(math.ceil(note.end * samples_per_sec))
                    hands[note.pitch - 21, start:end, hand] = True

            # Save array to disk
            np.save(npy_file, hands)


class Dataset:
    def __init__(self, path, n_samples_past=0):
        self.n_windows_past = n_samples_past
        npy_files = get_files_from_path(path, ['*.npy'])

        # Load numpy array data in a single long array and fill start and end with zeros
        self.data = np.concatenate([
            np.zeros((88, n_samples_past, 2)),
            np.concatenate([np.load(npy_file) for npy_file in npy_files], axis=1),
            np.zeros((88, n_samples_past, 2))
        ], axis=1)

    def next_batch(self, n_samples):
        # Initialize hands array with zeros
        hands = np.zeros((
            88,                       # 88 keys on a piano
            self.n_windows_past + 1,  # Number of past windows considered
            2,                        # Left and right hand = 2 hands
            n_samples                 # Number of samples per batch
        ), dtype=np.bool)

        # Get total number of windows
        n_windows_total = self.data.shape[1] - self.n_windows_past

        for sample in range(n_samples):
            # Pick random starting point in dataset...
            start = random.randrange(n_windows_total)
            # ...and extract samples
            hands[:, :, :, sample] = self.data[:, start:start + self.n_windows_past + 1, :]

        # Merge both hands in a single array
        batch_x = np.logical_or(
            hands[:, :, 0, :],
            hands[:, :, 1, :]
        )

        # Mark if both hands are played simultaneously
        both = np.logical_and(
            hands[:, :, 0, :],
            hands[:, :, 1, :]
        )

        # Return last played window of every sample:
        #    -1 => left hand
        #    +1 => right hand
        #     0 => both hands
        #   nan => no hand
        batch_y = np.full((hands.shape[0], hands.shape[3]), np.nan)
        batch_y[hands[:, -1, 0, :]] = -1
        batch_y[hands[:, -1, 1, :]] = 1
        batch_y[both[:, -1, :]] = 0

        return batch_x, batch_y


if __name__ == '__main__':
    convert(path='data', ms_window=20, overwrite=False)
    foo = Dataset('data', n_samples_past=100)
    for i in range(10000):
        batch_x, batch_y = foo.next_batch(400)

        # import matplotlib.pyplot as plt
        # plt.imshow(batch_x[:, :, 0], cmap='bwr', origin='lower', vmin=-1, vmax=1)
        # mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        # plt.show()
