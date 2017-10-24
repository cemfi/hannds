import glob
import logging
import math
import os
import random

import pretty_midi
from numba import jit
import numpy as np

logging.basicConfig(level=logging.DEBUG)


@jit
def convert(path, ms_window=20):
    if os.path.isfile(path):
        midi_files = [path]
    else:
        # Get list of all midi files in path
        midi_files = []
        for file_type in ('*.mid', '*.midi'):
            midi_files.extend(glob.glob(os.path.join(path, file_type)))

        if len(midi_files) == 0:
            raise FileNotFoundError('No midi files found!')

    for midi_file in midi_files:
        midi = pretty_midi.PrettyMIDI(midi_file)
        logging.debug('Converting file \'' + midi_file + '\'...')
        midi_data = midi.instruments[0], midi.instruments[1]

        # Generate empty numpy arrays
        array_dim = math.ceil(midi.get_end_time() * ms_window)
        hands = np.zeros((88, array_dim, 2), dtype=np.bool)

        # Fill arrays with data
        for hand, midi_hand in enumerate(midi_data):
            for note in midi_hand.notes:
                start = int(math.floor(note.start * ms_window))
                end = int(math.ceil(note.end * ms_window))
                for width in range(start, end):
                    hands[note.pitch - 21, width, hand] = True

        np.save(midi_file + '.npy', hands)


class Dataset:
    def __init__(self, path):
        if os.path.isfile(path):
            npy_files = [path]
        else:
            npy_files = []
            npy_files.extend(glob.glob(os.path.join(path, '*.npy')))

            if len(npy_files) == 0:
                raise FileNotFoundError('No numpy arrays found!')

        self._data = []
        for npy_file in npy_files:
            self._data.append(np.load(npy_file))

    def next_batch(self, n_samples, n_past_windows=0):
        hands = np.zeros((88, n_past_windows + 1, 2, n_samples), dtype=np.bool)

        for sample in range(n_samples):
            file = random.choice(self._data)
            window = random.randrange(file.shape[1] + n_past_windows)

            if window >= file.shape[1]:
                data = np.concatenate([
                    file[:, window - n_past_windows:, :],
                    np.zeros((88, window - file.shape[1] + 1, 2), dtype=np.bool)
                ], axis=1)
            elif window < n_past_windows:
                data = np.concatenate([
                    np.zeros((88, n_past_windows - window, 2), dtype=np.bool),
                    file[:, 0:window + 1, :]
                ], axis=1)
            else:
                data = file[:, window - n_past_windows:window + 1, :]

            hands[:, :, :, sample] = data

            # import matplotlib.pyplot as plt
            # tmp = (data[:, :, 0].astype(np.int8) - data[:, :, 1].astype(np.int8))
            #
            # plt.imshow(tmp, cmap='bwr', origin='lower', vmin=-1, vmax=1)
            # plt.show()

        hands_88 = np.logical_or(hands[:, :, 0, :], hands[:, :, 1, :]).astype(np.int8)
        hands_176 = hands[:, :, 0, :].astype(np.int8) - hands[:, :, 1, :].astype(np.int8)

        return hands_88, hands_176


if __name__ == '__main__':
    # convert(path='data', ms_window=20)
    foo = Dataset('data')
    for i in range(100):
        foo.next_batch(400, n_past_windows=0)
