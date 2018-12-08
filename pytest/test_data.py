"""Tests for hannds_files.py"""

from hannds_files import TrainValidTestFiles
import hannds_data


def test_n_hold_out():
    data = TrainValidTestFiles()
    for step in range(1, 11):
        n = data._n_hold_out(cv_partition=step, n_files=94)
        if step <= 4:
            assert n == 10
        else:
            assert n == 9


def test_hold_out_range():
    data = TrainValidTestFiles()
    (begin, end) = data._hold_out_range(cv_partition=1, n_files=94)
    assert (begin, end) == (0, 10)
    (begin, end) = data._hold_out_range(cv_partition=2, n_files=94)
    assert (begin, end) == (10, 20)
    (begin, end) = data._hold_out_range(cv_partition=3, n_files=94)
    assert (begin, end) == (20, 30)
    (begin, end) = data._hold_out_range(cv_partition=4, n_files=94)
    assert (begin, end) == (30, 40)
    (begin, end) = data._hold_out_range(cv_partition=5, n_files=94)
    assert (begin, end) == (40, 49)
    (begin, end) = data._hold_out_range(cv_partition=6, n_files=94)
    assert (begin, end) == (49, 58)
    (begin, end) = data._hold_out_range(cv_partition=7, n_files=94)
    assert (begin, end) == (58, 67)
    (begin, end) = data._hold_out_range(cv_partition=8, n_files=94)
    assert (begin, end) == (67, 76)
    (begin, end) = data._hold_out_range(cv_partition=9, n_files=94)
    assert (begin, end) == (76, 85)
    (begin, end) = data._hold_out_range(cv_partition=10, n_files=94)
    assert (begin, end) == (85, 94)


def test_disjoint():
    files = TrainValidTestFiles()
    for cv_index in range(1, 11):
        files.read_files_from_dir(cv_partition=cv_index)
        a = set(files.train_files)
        b = set(files.valid_files)
        c = set(files.test_files)
        assert a.isdisjoint(b)
        assert a.isdisjoint(c)
        assert b.isdisjoint(c)


def test_data_main():
    hannds_data.main()
