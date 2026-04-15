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

"""Base qubit-based string components and collections of such strings."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

from typing_extensions import Self

from zixy._zixy import Qubits
from zixy.container.cmpnts import Cmpnt, Cmpnts, CmpntSet, SpecT

if TYPE_CHECKING:
    from zixy._zixy import QubitArray, Springs

ElemT = TypeVar("ElemT")
ImplT = TypeVar("ImplT", bound="QubitArray[Any, Any]")


def _default_qubits(source: SpecT | None = None) -> Qubits:
    """Construct the default qubits for a string specifier."""
    if source is None:
        return Qubits.from_count(0)
    elif isinstance(source, tuple | list):
        return Qubits.from_count(len(source))
    else:
        raise TypeError("Source object is of an unsupported type.")


class String(Generic[ImplT, SpecT, ElemT], Cmpnt[ImplT, SpecT]):
    """A string.

    A single qubit-based string that may be an owning instance referencing a single element in a
    Rust-bound data object, or a view on an element in another collection.
    """

    _springs_type: type[Springs]

    @staticmethod
    def _get_default_qubits(source: SpecT | None = None) -> Qubits:
        """Get the default qubits for this string type based on a string specifier."""
        return _default_qubits(source)

    def __init__(self, qubits: int | Qubits | None = None, source: SpecT | None = None):
        """Initialize the string.

        Args:
            qubits: The qubit register or qubit count.
            source: The string specifier to use for default qubits and initial value.
        """
        if qubits is None and source is not None:
            qubits = self._get_default_qubits(source)
        elif qubits is None:
            raise ValueError("At least one of qubits and source must be specified.")
        impl = self.impl_type(qubits if isinstance(qubits, Qubits) else Qubits.from_count(qubits))
        impl.resize(1)
        super().__init__(impl)
        if source is not None:
            self.set(source)
        assert len(self._impl) == 1

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return self._impl.cmpnt_to_string(self.index)

    @property
    def qubits(self) -> Qubits:
        """Get the qubits corresponding to :param:`self`."""
        return self._impl.qubits

    @abstractmethod
    def __getitem__(self, i: int) -> ElemT:
        """Return the element of the string at index :param:`i`."""
        pass

    @abstractmethod
    def __setitem__(self, i: int, elem: ElemT | Any) -> None:
        """Set the element of the string at index :param:`i`."""
        pass

    def get_list(self) -> list[ElemT]:
        """Get the string as a list of its elements."""
        return self._impl.cmpnt_get_list(self.index)

    def get_tuple(self) -> tuple[ElemT, ...]:
        """Get the string as a tuple of its elements."""
        return tuple(self.get_list())

    def _set_copy(self, source: String[ImplT, SpecT, ElemT]) -> None:
        """Set the value of :param:`self` to that of :param:`source`."""
        if not isinstance(source, String):
            raise TypeError("Source object must be a String.")
        if self._impl.same_as(source._impl):
            self._impl.cmpnt_copy_internal(self.index, source.index)
        else:
            self._impl.cmpnt_copy_external(self.index, source._impl, source.index)

    def set(self, source: SpecT | String[ImplT, SpecT, ElemT] | None) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        if source is None:
            source = tuple()  # type: ignore[assignment]
        if isinstance(source, String):
            self._set_copy(source)
        elif isinstance(source, tuple | list):
            try:
                self._impl.cmpnt_set_from_list(self.index, list(source))
            except TypeError:
                for i, c in enumerate(source):
                    self[i] = c
        elif isinstance(source, str):
            springs = self._springs_type(source)
            if len(springs) == 0:
                self.clear()
            elif len(springs) == 1:
                self._impl.cmpnt_set_from_spring(self.index, springs, 0)
            else:
                raise ValueError("String input has more than one component.")
        else:
            raise TypeError("Source object is of an unsupported type.")

    def count(self, elem: ElemT) -> int:
        """Return the number of occurrences of :param:`elem` in the string.

        Args:
            elem: The element to count.

        Returns:
            The number of occurrences of :param:`elem` in the string.
        """
        return self._impl.cmpnt_count(self.index, elem)

    def to_sparse_matrix(self, big_endian: bool = False) -> Any:
        """Return :param:`self` as a sparse matrix.

        Args:
            big_endian: Whether to use big-endian basis ordering.

        Returns:
            The sparse matrix representation of :param:`self`.
        """
        return self._impl.cmpnt_to_sparse_matrix(self.index, big_endian)  # type: ignore[attr-defined]


class Strings(Generic[ImplT, SpecT, ElemT], Cmpnts[ImplT, SpecT]):
    """A collection of strings.

    An array-like container of qubit-based strings that may be an owning instance referencing a
    contiguous Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    def __init__(self, qubits: int | Qubits = 0, n: int = 0):
        """Initialize the string array.

        Args:
            qubits: Qubits object or number of qubits to use.
            n: Number of default elements with which to create the instance.
        """
        super().__init__(
            self.cmpnt_type.impl_type(
                qubits if isinstance(qubits, Qubits) else Qubits.from_count(qubits)
            )
        )
        self.resize(n)

    @property
    def qubits(self) -> Qubits:
        """Get the qubits corresponding to :param:`self`."""
        return self._impl.qubits

    @overload
    def __getitem__(self, indexer: int) -> String[ImplT, SpecT, ElemT]: ...

    @overload
    def __getitem__(self, indexer: slice) -> Self: ...

    def __getitem__(self, indexer: int | slice) -> String[ImplT, SpecT, ElemT] | Self:
        """Get the element or elements selected by :param:`indexer`.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by :param:`indexer`.
        """
        return super().__getitem__(indexer)  # type: ignore[return-value]

    def relabel(self, qubits: Qubits) -> Self:
        """Relabel the contents of the strings according to a new qubit register.

        Args:
            qubits: The qubit register or qubit count.

        Returns:
            :param:`self` for chaining.

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
        :param:`n_qubit` may differ from the size of the original register. If larger, clear qubits
        are appended after the reordered contents; if smaller, only contents at qubit indices less
        than :param:`n_qubit` will be retained.

        Args:
            n_qubit: The number of qubits.

        Returns:
            :param:`self` for chaining.

        Note:
            This method operates in-place.
        """
        self._impl.standardize(n_qubit)
        return self

    def standardized(self, n_qubit: int) -> Self:
        """Standardize the labels of a clone of the string according to a given number of qubits.

        Reorders the contents of the strings such that the associated qubit register can be
        reassigned to ``Qubits.from_count(n_qubit)`` without semantic relabelling. The given
        :param:`n_qubit` may differ from the size of the original register. If larger, clear qubits
        are appended after the reordered contents; if smaller, only contents at qubit indices less
        than :param:`n_qubit` will be retained.

        Args:
            n_qubit: The number of qubits.

        Returns:
            The resulting value.
        """
        out = self.clone()
        out._impl = out._impl.standardized(n_qubit)
        return out

    def get_list(self) -> list[list[ElemT]]:
        """Get all the strings as a list of lists of their elements."""
        return [self[i].get_list() for i in range(len(self))]

    def get_tuples(self) -> tuple[tuple[ElemT, ...], ...]:
        """Get all the strings as a list of tuples of their elements."""
        return tuple(self[i].get_tuple() for i in range(len(self)))


class StringSet(Generic[ImplT, SpecT, ElemT], CmpntSet[ImplT, SpecT]):
    """A collection of unique strings.

    A set-like container of qubit-based strings that may be used to store unique components and
    perform set-like operations on them.
    """

    def __init__(self, qubits: int | Qubits = 0):
        """Initialize the string set.

        Args:
            qubits: The qubit register or qubit count.
        """
        super().__init__(
            self.cmpnts_type.cmpnt_type.impl_type(
                qubits if isinstance(qubits, Qubits) else Qubits.from_count(qubits)
            )
        )

    @classmethod
    def from_strings(cls, strings: Strings[ImplT, SpecT, ElemT]) -> StringSet[ImplT, SpecT, ElemT]:
        """Create a new instance of :param:`cls` from the strings in :param:`strings`.

        Args:
            strings: Owned or viewed strings with which to populate the new instance.

        Returns:
            A new instance of :param:`cls` containing the strings in :param:`strings`.
        """
        return cls.from_cmpnts(strings)

    def to_strings(self) -> Strings[ImplT, SpecT, ElemT]:
        """Create a new array of strings owning copies of all those contained in :param:`self`.

        Returns:
            New :class:`Strings` instance containing copies of the strings in :param:`self`. The
            type is determined by :attr:`~zixy.qubit._strings.StringSet.cmpnts_type`.
        """
        return cast(Strings[ImplT, SpecT, ElemT], self.to_cmpnts())
