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

"""State string components and collections of such strings.

State strings are computational basis vectors, represented as bit strings, defined in a space of
qubits.

The structure of this module parallels that of :mod:`~zixy.container.cmpnts` and
:mod:`~zixy.qubit._strings`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

from zixy._zixy import BinarySprings, Qubits, QubitStateArray
from zixy.container.coeffs import ComplexSign
from zixy.qubit._strings import (
    String as StringBase,
    Strings as StringsBase,
    StringSet as StringSetBase,
    _default_qubits as _default_qubits_base,
)
from zixy.qubit.pauli import String as PauliString

if TYPE_CHECKING:
    from zixy.qubit.state._terms import TermRegistry

StringSpec = None | Sequence[bool] | set[int] | str
ElemT = bool
SpecT = StringSpec
ImplT = QubitStateArray


def _default_qubits(source: StringSpec = None) -> Qubits:
    """Construct the default qubits for a string specifier."""
    if isinstance(source, set):
        return Qubits.from_count(max(source) + 1)
    elif isinstance(source, str):
        return Qubits.from_count(BinarySprings(source).default_n_qubit())
    else:
        return _default_qubits_base(source)


class String(StringBase[ImplT, SpecT, ElemT]):
    """A state string.

    A single qubit-based state string that may be an owning instance referencing a single element
    in a Rust-bound data object, or a view on an element in another collection.
    """

    impl_type = ImplT
    _term_registry: TermRegistry

    _springs_type = BinarySprings

    @staticmethod
    def _get_default_qubits(source: SpecT | None = None) -> Qubits:
        """Get the default qubit space for :param:`source`."""
        return _default_qubits(source)

    def __getitem__(self, i: int) -> bool:
        """Return the bit value of the string at index :param:`i`."""
        return self._impl.cmpnt_get_bit(self.index, i)

    def __setitem__(self, i: int, bit: bool | int) -> None:
        """Set the bit value of the string at index :param:`i`."""
        if isinstance(bit, int):
            if not (bit == 0 or bit == 1):
                raise ValueError("Integer bit argument must be either 0 or 1")
            bit = bool(bit)
        elif not isinstance(bit, bool):
            raise TypeError("Bit argument should be bool or 0 or 1")
        self._impl.cmpnt_set_bit(self.index, i, bit)

    def get_set(self) -> set[int]:
        """Get the string as a set of the indices of bits with value 1."""
        return self._impl.cmpnt_get_set(self.index)

    def set(self, source: SpecT | StringBase[ImplT, SpecT, ElemT] | None) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        if isinstance(source, set):
            self._impl.cmpnt_set_from_set(self.index, source)
        else:
            super().set(source)

    def hamming_weight(self) -> int:
        r"""Get the Hamming weight of this bit string.

        The Hamming weight is defined as

        .. math:: \sum_i s_i

        where :math:`s_i` is the value of the bit at index :math:`i` in the string.
        """
        return self.count(True)

    def is_vacuum(self) -> bool:
        r"""Check whether the string is the vacuum state (:math:`\left[0, 0, \ldots, 0\right]`)."""
        return self.count(True) == 0

    def vdot(self, other: String) -> int:
        """Compute the inner product of this string with another."""
        return int(self == other)

    def imul_get_phase(self, op: PauliString) -> ComplexSign:
        """In-place multiplication of :param:`self` by :param:`op`, returning the phase."""
        return ComplexSign(self._impl.cmpnt_pauli_string_imul(self.index, op._impl, op.index))


class Strings(StringsBase[ImplT, SpecT, ElemT]):
    """A collection of state strings.

    An array-like container of qubit-based state strings that may be an owning instance referencing
    a contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    cmpnt_type = String

    _set_type: type[StringSet]

    @classmethod
    def new(cls, qubits: int | Qubits = 0, n: int = 0) -> Strings:
        """Create a new instance with a given qubit count and number of 00...0 strings.

        Args:
            qubits: Space of qubits or a number of qubits.
            n: Number of default elements with which to create the instance.
        """
        out = cls._create(
            ImplT(qubits if isinstance(qubits, Qubits) else Qubits.from_count(qubits))
        )
        out.resize(n)
        return out

    @overload
    def __getitem__(self, indexer: int) -> String: ...

    @overload
    def __getitem__(self, indexer: slice) -> Strings: ...

    def __getitem__(self, indexer: int | slice) -> String | Strings:
        """Get the element or elements selected by :param:`indexer`.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by :param:`indexer`.
        """
        return super().__getitem__(indexer)  # type: ignore[return-value]

    def get_sets(self) -> tuple[set[int], ...]:
        """Get the strings as a list of sets of the indices of bits with value 1."""
        return tuple(self[i].get_set() for i in range(len(self)))


class StringSet(StringSetBase[ImplT, SpecT, ElemT]):
    """A collection of unique state strings.

    A set-like container of qubit-based Pauli strings that may be used to store unique components
    and perform set-like operations on them.
    """

    cmpnts_type = Strings


Strings._set_type = StringSet
