import pytest

from zixy.utils import slice_index, slice_len, slice_of_slice, slice_to_tuple


def test_slice_len():
    assert slice_len(slice(0, 6, 1), 6) == 6
    assert slice_len(slice(0, 6, 1), 5) == 5
    assert slice_len(slice(0, 6, 1), 7) == 6
    assert slice_len(slice(1, 6, 1), 6) == 5
    assert slice_len(slice(2, 6, 1), 6) == 4
    assert slice_len(slice(1, None, -1), 2) == 2
    assert slice_len(slice(None, None, -1), 99) == 99
    assert slice_len(slice(99, None, -1), 99) == 99
    assert slice_len(slice(98, None, -1), 99) == 99
    assert slice_len(slice(97, None, -1), 99) == 98


def test_slice_to_tuple():
    assert slice_to_tuple(slice(2, 6, 1), 6) == (2, 3, 4, 5)


def test_slice_of_slice():
    test_cases = (
        (slice(0, 7, 2), slice(None, None, None), 7),
        (slice(None, None, 2), slice(None, None, 3), 20),
        (slice(None, None, 2), slice(None, None, -1), 20),
        (slice(None, None, -1), slice(None, None, -1), 20),
        (slice(None, None, -1), slice(None, None, 2), 20),
        (slice(3, None, -1), slice(30, None, 2), 20),
        (slice(3, None, 2), slice(30, None, -1), 20),
        (slice(3, None, -1), slice(30, None, -1), 20),
        (slice(3, None, -1), slice(30, None, -2), 20),
        (slice(30, None, -1), slice(3, None, 2), 20),
        (slice(30, None, 2), slice(3, None, -1), 20),
        (slice(30, None, -1), slice(3, None, -1), 20),
        (slice(30, None, -1), slice(3, None, -2), 20),
        (slice(None, None, -2), slice(None, None, -1), 6),
    )
    for s1, s2, length in test_cases:
        s = slice_of_slice(s1, s2, length)
        data = tuple(range(length))
        assert data[s] == data[s1][s2], (s, data[s], data[s1][s2])


def test_slice_index():
    assert slice_index(slice(None, None, 1), 0, 6) == 0
    assert slice_index(slice(None, None, 1), 1, 6) == 1
    assert slice_index(slice(None, None, 1), 2, 6) == 2
    assert slice_index(slice(None, None, 1), 3, 6) == 3
    assert slice_index(slice(None, None, 1), 5, 6) == 5
    assert slice_index(slice(None, None, 1), -1, 6) == 5
    assert slice_index(slice(None, None, 1), -2, 6) == 4
    assert slice_index(slice(None, None, 1), -3, 6) == 3
    assert slice_index(slice(None, None, 1), -5, 6) == 1
    assert slice_index(slice(None, None, 1), -6, 6) == 0
    assert slice_index(slice(None, None, -1), 0, 6) == 5
    assert slice_index(slice(None, None, -1), 1, 6) == 4
    assert slice_index(slice(None, None, -1), 5, 6) == 0
    assert slice_index(slice(None, None, -1), -1, 6) == 0
    assert slice_index(slice(None, None, -2), 0, 6) == 5
    assert slice_index(slice(None, None, -2), 1, 6) == 3
    assert slice_index(slice(None, None, -2), 2, 6) == 1
    assert slice_index(slice(None, None, -2), -1, 6) == 1
    assert slice_index(slice(None, None, -2), -2, 6) == 3
    assert slice_index(slice(None, None, -2), -3, 6) == 5

    with pytest.raises(IndexError):
        slice_index(slice(None, None, 1), -7, 6)

    with pytest.raises(IndexError):
        slice_index(slice(None, None, -2), 3, 6)

    with pytest.raises(IndexError):
        slice_index(slice(None, None, -2), -4, 6)
