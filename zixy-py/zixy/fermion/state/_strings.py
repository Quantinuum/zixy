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

"""Fermionic occupation-number state string components and collections of such strings.

State strings are occupation-number basis vectors, represented as bit strings, defined over a
space of fermionic modes.

The structure of this module parallels that of :mod:`~zixy.container.cmpnts`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias, overload

from typing_extensions import Self

from zixy._zixy import BinarySprings, FermionStateArray, Modes
from zixy.fermion._strings import (
    String as FermionString,
    Strings as FermionStrings,
    StringSet as FermionStringSet,
)

if TYPE_CHECKING:
    from zixy.fermion.state._terms import TermRegistry

StringSpec: TypeAlias = None | Sequence[bool] | set[int] | str
ElemT = bool
SpecT = StringSpec
ImplT = FermionStateArray


def _default_modes(source: SpecT = None) -> Modes:
    """Construct the default modes for a string specifier."""
    if isinstance(source, set):
        return Modes.from_count(max(source, default=-1) + 1)
    if isinstance(source, str):
        return Modes.from_count(BinarySprings(source).default_n_qubit())
    if source is None:
        return Modes.from_count(0)
    return Modes.from_count(len(source))


class String(FermionString[ImplT, SpecT, ElemT]):
    """A fermionic occupation-number state string.

    A single mode-based state string that may be an owning instance referencing a single element in
    a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT
    _term_registry: TermRegistry
    _get_default_modes = staticmethod(_default_modes)

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the input string.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        n = len(BinarySprings(source))
        if n != 1:
            raise ValueError(f"Source string should contain one state string, got {n}.")
        return cls(modes, source)

    def __getitem__(self, i: int) -> bool:
        """Return the occupation value of the string at index ``i``."""
        return self._impl.cmpnt_get_bit(self.index, i)

    def __setitem__(self, i: int, bit: bool | int) -> None:
        """Set the occupation value of the string at index ``i``."""
        if isinstance(bit, int):
            if not (bit == 0 or bit == 1):
                raise ValueError("Integer bit argument must be either 0 or 1")
            bit = bool(bit)
        elif not isinstance(bit, bool):
            raise TypeError("Bit argument should be bool or 0 or 1")
        self._impl.cmpnt_set_bit(self.index, i, bit)

    def get_list(self) -> list[bool]:
        """Get the string as a list of occupation flags."""
        return self._impl.cmpnt_get_list(self.index)

    def get_tuple(self) -> tuple[bool, ...]:
        """Get the string as a tuple of occupation flags."""
        return tuple(self.get_list())

    def get_set(self) -> set[int]:
        """Get the string as a set of occupied mode indices."""
        return self._impl.cmpnt_get_set(self.index)

    def set(self, source: SpecT | String | None) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        if source is None:
            self._impl.cmpnt_clear(self.index)
        elif isinstance(source, String):
            self._set_copy(source)
        elif isinstance(source, set):
            self._impl.cmpnt_set_from_set(self.index, source)
        elif isinstance(source, tuple | list):
            self._impl.cmpnt_set_from_list(self.index, list(source))
        elif isinstance(source, str):
            springs = BinarySprings(source)
            if len(springs) == 0:
                self.clear()
            elif len(springs) == 1:
                self._impl.cmpnt_set_from_spring(self.index, springs, 0)
            else:
                raise ValueError("String input has more than one component.")
        else:
            self.raise_spec_type_error(source)

    def count(self, bit: bool) -> int:
        """Return the number of occurrences of ``bit`` in the string."""
        return self._impl.cmpnt_count(self.index, bit)

    def hamming_weight(self) -> int:
        r"""Get the Hamming weight of this bit string.

        The Hamming weight is defined as

        .. math:: \sum_i s_i

        where :math:`s_i` is the occupation value at index :math:`i` in the string.
        """
        return self.count(True)

    def is_vacuum(self) -> bool:
        r"""Check whether the string is the vacuum state (:math:`\left[0, 0, \ldots, 0\right]`)."""
        return self.count(True) == 0

    def vdot(self, other: String) -> int:
        """Compute the inner product of this string with another."""
        return int(self == other)


class Strings(FermionStrings[ImplT, SpecT, ElemT]):
    """A collection of fermionic occupation-number state strings.

    An array-like container of mode-based state strings that may be an owning instance referencing
    a contiguous Rust-bound data object, or a view on a slice of the elements in another
    collection.
    """

    cmpnt_type = String
    _set_type: type[StringSet]

    @classmethod
    def new(cls, modes: int | Modes = 0, n: int = 0) -> Self:
        """Create a new instance with a given mode count and number of vacuum strings."""
        return cls(modes, n)

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the input string.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        if not source.strip():
            out = cls(modes if modes is not None else 0)
            out.resize(0)
            return out
        return cls._create(cls.cmpnt_type.impl_type(modes, BinarySprings(source)))

    @overload
    def __getitem__(self, indexer: int) -> String: ...
    @overload
    def __getitem__(self, indexer: slice) -> Self: ...
    def __getitem__(self, indexer: int | slice) -> String | Self:
        """Get the element or elements selected by ``indexer``."""
        return super().__getitem__(indexer)  # type: ignore[return-value]

    def get_sets(self) -> tuple[set[int], ...]:
        """Get the strings as a list of sets of occupied mode indices."""
        return tuple(self[i].get_set() for i in range(len(self)))


class StringSet(FermionStringSet[ImplT, SpecT, ElemT]):
    """A collection of unique fermionic occupation-number state strings.

    A set-like container of mode-based state strings that may be used to store unique components
    and perform set-like operations on them.
    """

    cmpnts_type = Strings


Strings._set_type = StringSet
