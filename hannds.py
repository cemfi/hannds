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
def convert(path, ms_window=20, overwrite=True):
    if os.path.isfile(path):
        # Load single midi file
        midi_files = [path]
    else:
        # Get list of all midi files in path
        midi_files = []
        for file_type in ('*.mid', '*.midi'):
            midi_files.extend(glob.glob(os.path.join(path, file_type)))

        if len(midi_files) == 0:
            raise FileNotFoundError('No midi files found!')

    samples_per_sec = (1000 / ms_window)
    for midi_file in midi_files:
        npy_file = midi_file + '_' + str(ms_window) + 'ms' + '.npy'

        if overwrite or not os.path.exists(npy_file):
            midi = pretty_midi.PrettyMIDI(midi_file)
            logging.debug('Converting file \'' + midi_file + '\'...')
            midi_data = midi.instruments[0], midi.instruments[1]

            # Generate empty numpy arrays
            n_samples = math.ceil(midi.get_end_time() * samples_per_sec)
            hands = np.zeros((
                88,         # 88 keys on a piano
                n_samples,  # Number of samples
                2           # Left and right hand = 2 hands
            ), dtype=np.bool)

            # Fill arrays with data
            for hand, midi_hand in enumerate(midi_data):
                for note in midi_hand.notes:
                    start = int(math.floor(note.start * samples_per_sec))
                    end = int(math.ceil(note.end * samples_per_sec))
                    for width in range(start, end):
                        hands[note.pitch - 21, width, hand] = True

            np.save(npy_file, hands)


class Dataset:
    def __init__(self, path, n_past_samples=0):
        self.n_past_samples = n_past_samples

        if os.path.isfile(path):
            # Load single numpy file
            npy_files = [path]
        else:
            # Get list of numpy files in path
            npy_files = []
            npy_files.extend(glob.glob(os.path.join(path, '*.npy')))

            if len(npy_files) == 0:
                raise FileNotFoundError('No numpy arrays found!')

        # Load numpy array data in a single long array and fill start and end with zeros
        self.data = np.concatenate([
            np.zeros((88, n_past_samples, 2)),
            np.concatenate([np.load(npy_file) for npy_file in npy_files], axis=1),
            np.zeros((88, n_past_samples, 2))
        ], axis=1)

    def next_batch(self, n_samples):
        # Initialize result array with zeros
        hands = np.zeros((
            88,                       # 88 keys on a piano
            self.n_past_samples + 1,  # Number of past samples considered
            2,                        # Left and right hand = 2 hands
            n_samples                 # Number of samples per batch
        ), dtype=np.bool)

        # Get total number of samples
        n_samples_total = self.data.shape[1] - self.n_past_samples

        for sample in range(n_samples):
            # Pick random starting point in dataset...
            start = random.randrange(n_samples_total)
            # ...and extract samples
            hands[:, :, :, sample] = self.data[:, start:start+self.n_past_samples + 1, :]

        # Merge both hands in a single array
        batch_x = np.logical_or(
            hands[:, :, 0, :],
            hands[:, :, 1, :]
        )

        # Subtract left from right hand, so that
        #   -1 => left hand
        #   +1 => right hand
        #    0 => not played (or both hands!)
        # Only last played sample of batch is returned since
        # it is the only one of relevance for the output
        batch_y = hands[:, -1, 0, :].astype(np.int8) -\
                  hands[:, -1, 1, :].astype(np.int8)

        return batch_x, batch_y


if __name__ == '__main__':
    convert(path='data', ms_window=20, overwrite=False)
    foo = Dataset('data', n_past_samples=100)
    for i in range(10000):
        batch_x, batch_y = foo.next_batch(400)


        # import matplotlib.pyplot as plt
        # tmp = batch_y[:, :, 0]
        # plt.imshow(tmp, cmap='bwr', origin='lower', vmin=-1, vmax=1)
        # mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        # plt.show()
