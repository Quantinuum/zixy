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

"""String sequences for qubit systems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from zixy._zixy import Qubits
from zixy.container.strings import Strings as BaseStrings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zixy._zixy import QubitArray
    from zixy.container.string import String as BaseString
    from zixy.qubit._string import String


class Strings[ImplT: QubitArray[Any, Any], SpecT, ElemT](BaseStrings[ImplT, SpecT]):
    """A collection of strings.

    An array-like container of qubit-based strings that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    string_type: type[String[ImplT, SpecT, ElemT]]

    def __init__(self, qubits: int | Qubits = 0, *, size: int = 0):
        """Initialize the string array.

        Args:
            qubits: Qubits object or number of qubits to use.
            size: Number of default elements with which to create the instance.
        """
        super().__init__(
            self.string_type.impl_type(
                qubits if isinstance(qubits, Qubits) else Qubits.from_count(qubits)
            )
        )
        self.resize(size)

    @classmethod
    def from_iterable(
        cls, source: Iterable[SpecT | BaseString[ImplT, SpecT]], qubits: int | Qubits = 0
    ) -> Self:
        """Create an owning collection from an iterable of strings.

        Args:
            source: Iterable of compatible strings or string specifiers.
            qubits: The qubit register or qubit count.

        Returns:
            An independent owning instance of ``cls``.
        """
        return super().from_iterable(source, qubits)

    @property
    def qubits(self) -> Qubits:
        """The qubits corresponding to ``self``."""
        return self._impl.qubits

    def get_list(self) -> list[list[ElemT]]:
        """Get all the strings as a list of lists of their elements."""
        return [self[i].get_list() for i in range(len(self))]

    def get_tuples(self) -> tuple[tuple[ElemT, ...], ...]:
        """Get all the strings as a list of tuples of their elements."""
        return tuple(self[i].get_tuple() for i in range(len(self)))

    def relabel(self, qubits: Qubits) -> Self:
        """Relabel the contents of the strings according to a new qubit register.

        Args:
            qubits: The qubit register or qubit count.

        Returns:
            ``self`` for chaining.

        Note:
            This method operates in-place.
        """
        self._impl.relabel(qubits)
        return self

    def relabelled(self, qubits: Qubits) -> Self:
        """Relabel the contents of a clone of the strings according to a new set of qubit register.

        Args:
            qubits: The qubit register or qubit count.

        Returns:
            The resulting value.
        """
        out = self.clone()
        out.relabel(qubits)
        return out

    def standardize(self, n_qubit: int) -> Self:
        """Standardize the string labels according to a given number of qubits.

        Reorders the contents of the strings such that the associated qubit register can be
        reassigned to ``Qubits.from_count(n_qubit)`` without semantic relabelling. The given
        ``n_qubit`` may differ from the size of the original register. If larger, clear qubits
        are appended after the reordered contents; if smaller, only contents at qubit indices less
        than ``n_qubit`` will be retained.

        Args:
            n_qubit: The number of qubits.

        Returns:
            ``self`` for chaining.

        Note:
            This method operates in-place.
        """
        self._impl.standardize(n_qubit)
        return self

    def standardized(self, n_qubit: int) -> Self:
        """Standardize the labels of a clone of the string according to a given number of qubits.

        Reorders the contents of the strings such that the associated qubit register can be
        reassigned to ``Qubits.from_count(n_qubit)`` without semantic relabelling. The given
        ``n_qubit`` may differ from the size of the original register. If larger, clear qubits
        are appended after the reordered contents; if smaller, only contents at qubit indices less
        than ``n_qubit`` will be retained.

        Args:
            n_qubit: The number of qubits.

        Returns:
            The resulting value.
        """
        out = self.clone()
        out._impl = out._impl.standardized(n_qubit)
        return out

    @classmethod
    def from_str(cls, source: str, qubits: int | Qubits | None = None) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.
            qubits: The qubit register or qubit count. If ``None``, the qubit register is
                inferred from the string specifier.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        if not source.strip():
            out = cls._create(cls.string_type.impl_type(qubits))
            out.resize(0)
            return out
        return cls._create(cls.string_type.impl_type(qubits, cls.string_type._springs_type(source)))
