# Copyright 2026 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Final

import numpy


def slice_index(s: slice, i: int, length: int) -> int:
    """Find an index in the original sequence corresponding to index :param:`i` of slice :param:`s`.

    Args:
        s: The slice object.
        i: Zero-based index relative to the slice.
        length: Length of the sequence being sliced.

    Returns:
        The index in the original sequence corresponding to index :param:`i` of slice :param:`s`.

    Raises:
        IndexError: If i is out of the slice bounds.
    """
    n = slice_len(s, length)
    if i < 0:
        i += n
    if i < 0 or i >= n:
        raise IndexError("slice index out of range")
    start, _, step = s.indices(length)
    i = start + i * step
    return i


def slice_single_item(s: slice, i: int, length: int) -> slice:
    """Narrow a slice to another slice for a single item within it.

    Args:
        s: The slice object.
        i: The index of the item to select, relative to the slice.
        length: The length of the underlying sequence.

    Returns:
        A slice that selects the item indexed ``i`` within the slice ``s``.
    """
    i = slice_index(s, i, length)
    return slice(i, i + 1, 1)


def slice_len(s: slice, length: int) -> int:
    """Get the length of slice :param:`s` for a sequence of length :param:`length`."""
    return len(range(length)[s])


def slice_equal(s1: slice, s2: slice, length: int) -> bool:
    """Check if two slices are equal for a sequence of length :param:`length`."""
    return slice_to_tuple(s1, length) == slice_to_tuple(s2, length)


def slice_index_gen(s: slice, length: int) -> Iterator[int]:
    """Generate indices in the original sequence corresponding to indices of slice :param:`s`.

    Args:
        s: The slice object.
        length: The length of the underlying sequence.

    Returns:
        An iterator over the indices in the original sequence corresponding to indices of slice
        :param:`s`.
    """
    n = slice_len(s, length)
    return (slice_index(s, i, length) for i in range(n))


def slice_to_tuple(s: slice, length: int) -> tuple[int, ...]:
    """Get all indices in the original sequence corresponding to indices of slice :param:`s`.

    Args:
        s: the slice to get as a tuple.
        length: the number of elements in the underlying array.

    Returns:
        A tuple of indices in the original sequence corresponding to indices of slice :param:`s`.
    """
    return tuple(slice_index_gen(s, length))


def slice_of_slice(s1: slice, s2: slice, length: int) -> slice:
    """Compose two slices.

    The resulting slice satisfies

    .. code:: python

        a[slice_of_slice(s1, s2, length)] == a[s1][s2]

    for some sequence ``a`` of length :param:`length`.

    Args:
        s1: The first slice.
        s2: The second slice.
        length: The number of elements in the data being sliced.

    Returns:
        The combined slice.
    """
    step1 = 1 if s1.step is None else s1.step
    step2 = 1 if s2.step is None else s2.step
    step = step1 * step2

    start1, stop1, _ = s1.indices(length)
    length1 = (abs(stop1 - start1) - 1) // abs(step1)

    # If we step in the same direction as the start,stop, we get at least one datapoint
    if (stop1 - start1) * step1 > 0:
        length1 += 1
    else:
        # Otherwise, The slice is zero length.
        return slice(0, 0, step)

    # Use the length after the first slice to get the indices returned from a
    # second slice starting at 0.
    start2, stop2, _ = s2.indices(length1)

    # if the final range length = 0, return
    if not (stop2 - start2) * step2 > 0:
        return slice(0, 0, step)

    # We shift slice2 indices by the starting index in slice1 and the
    # step size of slice1
    start = start1 + start2 * step1
    stop = start1 + stop2 * step1

    return slice(start, None, step) if start > stop and stop < 0 else slice(start, stop, step)


def _default_tols() -> tuple[float, float]:
    """Get the default absolute and relative tolerances for :func:`~numpy.allclose`."""
    params = inspect.signature(numpy.allclose).parameters
    return (params["atol"].default, params["rtol"].default)


DEFAULT_ATOL: Final[float] = _default_tols()[0]
DEFAULT_RTOL: Final[float] = _default_tols()[1]
DEFAULT_COMMUTES_ATOL: Final[float] = 1e-12
