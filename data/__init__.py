import glob
import logging
import math
import os

import pretty_midi
from numba import jit
import numpy as np


class Dataset:
    def __init__(self, path, window_ms=20):
        self.window_ms = window_ms

        # Get list of all midi files in path
        file_types = ('*.mid', '*.midi')
        self.midi_files = []
        for file_type in file_types:
            self.midi_files.extend(glob.glob(os.path.join(path, file_type)))

        if len(self.midi_files) == 0:
            raise FileNotFoundError('No midi files found!')

        self.cur_file_idx = -1
        self.cnt_samples = 0
        self.np_left = None
        self.np_right = None

    def _read_file(self, file):
        # Read midi file
        midi = pretty_midi.PrettyMIDI(file)
        logging.debug('Loaded file \'' + file + '\'')
        midi_left = midi.instruments[1]
        midi_right = midi.instruments[0]

        # Generate empty numpy arrays
        array_dim = math.ceil(midi.get_end_time() * self.window_ms)
        np_left = np.zeros((88, array_dim), dtype=np.uint8)
        np_right = np.zeros((88, array_dim), dtype=np.uint8)

        # Fill arrays with data
        for note in midi_left.notes:
            start = math.floor(note.start * self.window_ms)
            end = math.ceil(note.end * self.window_ms)
            for w in range(start, end):
                np_left[note.pitch, w] = 1

        for note in midi_right.notes:
            start = math.floor(note.start * self.window_ms)
            end = math.ceil(note.end * self.window_ms)
            for w in range(start, end):
                np_right[note.pitch, w] = 1

        self.np_left = np_left
        self.np_right = np_right

    @jit
    def next_batch(self, n_samples, past_samples=0):
        # Init empty arrays with predefined size
        np_left = np.zeros((88 * past_samples, n_samples))
        np_right = np.zeros((88 * past_samples, n_samples))

        # If no file loaded or end of file reached...
        if self.cur_file_idx == -1 or self.cnt_samples >= self.np_left.shape[1] - past_samples:
            # ... load new file...
            self.cur_file_idx = (self.cur_file_idx + 1) % len(self.midi_files)
            self.cnt_samples = 0
            self._read_file(self.midi_files[self.cur_file_idx])

            # ... and preprocess the data
            # Fill start and end of data of new file with zeros to avoid 'jumps' in training data
            zeros_before = np.zeros((88, past_samples), dtype=np.uint8)
            zeros_after = np.zeros((88, n_samples - (self.np_left.shape[1] % n_samples)))
            self.np_left = np.concatenate([zeros_before, self.np_left, zeros_after], axis=1)
            self.np_right = np.concatenate([zeros_before, self.np_right, zeros_after], axis=1)

        # Insert data into arrays
        for sample in range(n_samples):
            for past in range(past_samples):
                sample_left = self.np_left[:, self.cnt_samples + sample + past_samples]
                np_left[past * 88:past * 88 + 88, sample] = sample_left

                sample_right = self.np_left[:, self.cnt_samples + sample + past_samples]
                np_right[past * 88:past * 88 + 88, sample] = sample_right

        self.cnt_samples += n_samples

        hands_88 = np.maximum(np_left, np_right).T
        hands_176 = np.concatenate([np_left, np_right])[past_samples - 1::past_samples].T

        return hands_88, hands_176
