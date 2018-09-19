import math
import os
import glob
import random


def all_midi_files(absolute_path=False):
    return _get_files_from_path(os.path.join(_g_package_directory, 'data'), ['*.mid', '*.midi'], absolute_path)


def npz_files_for_midi(midi_files, subdir):
    base_dir = _get_preprocessed_path(subdir)
    npz_files = [os.path.join(base_dir, os.path.splitext(os.path.basename(m))[0] + '.npz') for m in midi_files]
    return npz_files


class TrainValidTestFiles(object):
    def __init__(self):
        self.train_files = self.valid_files = self.test_files = None

    def read_files_from_dir(self, cv_partition=1):
        all_files = all_midi_files()
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

    def set_files_from_lists(self, train_files, valid_files, test_files, len_train_sequence):
        self.train_files = train_files.copy()
        self.valid_files = valid_files.copy()
        self.test_files = test_files.copy()

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

    def midi_dir(self):
        return os.path.join(_g_package_directory, 'data/')


def _get_files_from_path(path, extensions, absolute_path=False):
    if os.path.isfile(path):  # Load single file
        files = [path]
    else:  # Get list of all files with correct extensions in path
        files = []
        for file_type in extensions:
            coll = glob.glob(os.path.join(path, file_type))
            if not absolute_path:
                coll = [os.path.basename(c) for c in coll]
            files.extend(coll)

        if len(files) == 0:
            raise FileNotFoundError(f'No files found with correct extensions {str(extensions)} in {path}')
    return sorted(files)


def _get_preprocessed_path(subdir):
    output_path = os.path.join(_g_package_directory, 'preprocessed')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = os.path.join(output_path, subdir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return output_path


_g_package_directory = os.path.dirname(os.path.abspath(__file__))


def main():
    from pprint import pprint
    all = TrainValidTestFiles()
    all.read_files_from_dir()
    print('Training')
    pprint(all.train_files)
    print()
    print('Validation')
    pprint(all.valid_files)
    print()
    print('Testing')
    pprint(all.test_files)
    print()


if __name__ == '__main__':
    main()
