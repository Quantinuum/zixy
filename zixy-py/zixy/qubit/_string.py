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

"""Individual strings for qubit systems."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from zixy._zixy import Qubits
from zixy.container.string import String as BaseString

if TYPE_CHECKING:
    from zixy._zixy import QubitArray, Springs


def _default_qubits[SpecT](source: SpecT) -> Qubits:
    """Construct the default qubits for a string specifier."""
    if isinstance(source, tuple | list):
        return Qubits.from_count(len(source))
    else:
        raise TypeError("Source object is of an unsupported type.")


def _check_qubits_compatibility(*qubits: Qubits) -> None:
    """Check that all qubit registers are compatible."""
    if len(qubits) > 1 and any(qubits[0] != other for other in qubits[1:]):
        raise ValueError("Objects are defined over different qubits.")


class String[ImplT: QubitArray[Any, Any], SpecT, ElemT](BaseString[ImplT, SpecT]):
    """A string.

    A single qubit-based string that may be an owning instance referencing a single element in a
    Rust-bound data object, or a view on an element in another collection.
    """

    _springs_type: type[Springs]

    def __init__(
        self,
        qubits: int | Qubits | None = None,
        source: SpecT | BaseString[ImplT, SpecT] = "",  # type: ignore[assignment]
    ):
        """Initialize the string.

        Args:
            qubits: The qubit register or qubit count. If ``None``, the qubit register is
                inferred from the string specifier.
            source: Specification or compatible string to copy, also used to infer default qubits.
        """
        if isinstance(source, BaseString):
            if not isinstance(source._impl, self.impl_type):
                self.raise_spec_type_error(source)
            if qubits is None:
                qubits = source._impl.qubits
        elif qubits is None:
            qubits = self._get_default_qubits(source)
        impl = self.impl_type(qubits if isinstance(qubits, Qubits) else Qubits.from_count(qubits))
        impl.resize(1)
        super().__init__(impl)
        self.set(source)
        assert len(self._impl) == 1

    @staticmethod
    def _get_default_qubits(source: SpecT) -> Qubits:
        """Get the default qubits for this string type based on a string specifier."""
        return _default_qubits(source)

    @property
    def qubits(self) -> Qubits:
        """The qubits corresponding to ``self``."""
        return self._impl.qubits

    @abstractmethod
    def __getitem__(self, i: int) -> ElemT:
        """Return the element of the string at index ``i``."""
        pass

    @abstractmethod
    def __setitem__(self, i: int, elem: ElemT | Any) -> None:
        """Set the element of the string at index ``i``."""
        pass

    def count(self, elem: ElemT) -> int:
        """Return the number of occurrences of ``elem`` in the string.

        Args:
            elem: The element to count.

        Returns:
            The number of occurrences of ``elem`` in the string.
        """
        return self._impl.string_count(self.index, elem)

    def get_list(self) -> list[ElemT]:
        """Get the string as a list of its elements."""
        return self._impl.string_get_list(self.index)

    def get_tuple(self) -> tuple[ElemT, ...]:
        """Get the string as a tuple of its elements."""
        return tuple(self.get_list())

    def _set_spec(self, source: SpecT) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        if isinstance(source, Sequence) and not isinstance(source, str):
            try:
                self._impl.string_set_from_list(self.index, list(source))
            except TypeError:
                for i, c in enumerate(source):
                    self[i] = c
        elif isinstance(source, str):
            if not source.strip():
                self._impl.string_set_from_list(self.index, [])
                return
            springs = self._springs_type(source)
            if len(springs) == 0:
                self._impl.string_set_from_list(self.index, [])
            elif len(springs) == 1:
                self._impl.string_set_from_spring(self.index, springs, 0)
            else:
                raise ValueError("Input represents more than one string.")
        else:
            raise TypeError("Source object is of an unsupported type.")

    def to_sparse_matrix(self, big_endian: bool = False) -> Any:
        """Return ``self`` as a sparse matrix.

        Args:
            big_endian: Whether to use big-endian basis ordering.

        Returns:
            The sparse matrix representation of ``self``.
        """
        return self._impl.string_to_sparse_matrix(self.index, big_endian)  # type: ignore[attr-defined]
