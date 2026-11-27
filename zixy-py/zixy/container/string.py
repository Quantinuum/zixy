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

"""Individual strings."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self

from zixy.container.base import Representable, ViewableItem
from zixy.container.serialization import SerializableContainer
from zixy.utils import slice_index

if TYPE_CHECKING:
    from zixy._zixy import ImplArray
    from zixy.container.serialization import SerializedContainer


class String[ImplT: ImplArray, SpecT](ViewableItem[ImplT], Representable, SerializableContainer):
    """A string.

    A single string that may be an owning instance referencing a single element in a Rust-bound
    data object, or a view on an element in another collection.
    """

    impl_type: type[ImplT]
    _clear_spec: SpecT

    _impl: ImplT
    _index: int | None

    def __init__(self, impl: ImplT, indexer: int | None = None):
        """Initialize the string.

        Args:
            impl: Rust-bound object storing the data.
            indexer: Position within Rust-bound array at which the viewed string is stored. When
                ``indexer`` is ``None``, the :class:`~zixy.container.string.String` is an owning
                view on the sole element
                of ``impl``.
        """
        assert isinstance(impl, self.impl_type)
        self._impl = impl
        self._index = indexer
        self._check_bounds()

    def _check_bounds(self) -> None:
        """Validate that ``self`` references a valid element in its implementation array.

        Raises:
            IndexError: The index of ``self`` is out of bounds for the implementation array it
                views.
        """
        if len(self._impl) != 1 and self.is_owning():
            raise IndexError("Owning String must have exactly one element in the impl array.")
        index = self.index
        if index < 0:
            index += len(self._impl)
        if index < 0 or index >= len(self._impl):
            raise IndexError(
                f"Index {self.index} is out of bounds for an array of length {len(self._impl)}"
            )

    @classmethod
    def _create(cls, impl: ImplT) -> Self:
        """Create an owning instance of ``cls``, wrapping existing storage.

        Args:
            impl: Rust-bound object containing the data for this instance.

        Returns:
            A new owning instance of ``cls`` using the supplied storage without copying it.

        Raises:
            IndexError: The storage does not contain exactly one string.
        """
        out = cls.__new__(cls)
        String.__init__(out, impl)
        return out

    @classmethod
    def _view(cls, impl: ImplT, indexer: int) -> Self:
        """Create a mutable view on an existing string.

        Args:
            impl: Rust-bound object containing the data for this instance.
            indexer: Index selecting the viewed string.

        Returns:
            A non-owning instance of ``cls`` sharing the selected string without copying it.

        Raises:
            IndexError: The index is outside the storage.
        """
        out = cls.__new__(cls)
        String.__init__(out, impl, slice_index(slice(None), indexer, len(impl)))
        return out

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        return type(self)._create(self._impl.strings_clone([self.index]))

    @classmethod
    def raise_spec_type_error(cls, source: Any) -> None:
        """Raise a ``TypeError`` for unsupported string specifiers.

        Raises:
            TypeError: The specifier supplied to set the value of ``self`` is of an unsupported
                type.
        """
        raise TypeError(
            f"Cannot set value of a {cls} instance with a variable of type {type(source)}"
        )

    def set(self, source: SpecT | String[ImplT, SpecT]) -> None:
        """Set the value of the string.

        Args:
            source: Specification or compatible string to copy. Parent and subclass instances
                with the same backing implementation are supported.

        Note:
            This method operates in-place. Copying checks mode-space compatibility in the
            backing implementation.
        """
        if isinstance(source, String):
            self._set_copy(source)
        else:
            self._set_spec(source)

    def _set_copy(self, source: String[ImplT, SpecT]) -> None:
        """Copy a compatible string, preserving the source and checking its family."""
        if not isinstance(source._impl, self.impl_type):
            self.raise_spec_type_error(source)
        if self._impl.same_as(source._impl):
            self._impl.string_copy_internal(self.index, source.index)
        else:
            self._impl.string_copy_external(self.index, source._impl, source.index)

    @abstractmethod
    def _set_spec(self, source: SpecT) -> None:
        """Set the string from a domain-specific specification."""
        pass

    def clear(self) -> None:
        """Set the value of the string to zero.

        Note:
            This method operates in-place.
        """
        self.set(self._clear_spec)

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, String):
            return NotImplemented
        return self._impl.strings_eq([self.index], other._impl, [other.index])

    def aliases(self, other: String[ImplT, SpecT]) -> bool:
        """Determine whether ``self`` is a view of the same string as ``other``."""
        if type(self) is not type(other) or self.is_owning():
            # owners never alias other objects
            return False
        return self._impl.same_as(other._impl) and self.index == other.index

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return self._impl.string_to_string(self.index)

    @classmethod
    @abstractmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            A new owning instance of ``cls`` parsed from ``source``, with storage independent of
            the source dictionary and any existing containers.
        """
        pass
