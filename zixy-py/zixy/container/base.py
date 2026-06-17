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

"""Base classes for containers with an ownership model and underlying Rust-bound data."""

from __future__ import annotations

import builtins
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence, Sized
from functools import wraps
from typing import (
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import Self

from zixy.utils import slice_index, slice_len

P = ParamSpec("P")
R = TypeVar("R")
OwnT = TypeVar("OwnT", bound="SupportsOwnership")
OutT = TypeVar("OutT", bound="ViewableBase[Any, Any]")


class SupportsOwnership(Protocol):
    """Protocol for classes supporting the ownership model."""

    def is_owning(self) -> bool:
        """Check if ``self`` is owning (i.e. not a view)."""
        ...

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        ...


def requires_ownership(
    method: Callable[Concatenate[OwnT, P], R],
) -> Callable[Concatenate[OwnT, P], R]:
    """Decorator for methods that can only be called on owning views.

    Args:
        method: Method to decorate.

    Returns:
        Decorated method that raises a ``ValueError`` if called on a non-owning view.
    """

    @wraps(method)
    def wrapper(self: OwnT, *args: P.args, **kwargs: P.kwargs) -> R:
        if not self.is_owning():
            raise ValueError(
                f"Cannot call {method.__name__} on a non-owning view. Consider cloning the object "
                "first using the .clone() method."
            )
        return method(self, *args, **kwargs)

    return cast(Callable[Concatenate[OwnT, P], R], wrapper)


class Resizable(Sized, Protocol):
    """Protocol for :class:`Sized` classes that support resizing."""

    def resize(self, n: int) -> None:
        """Resize the underlying container.

        Args:
            n: The new size of the container.

        Note:
            This method operates in-place.

        Raises:
            ValueError: If the container is a view.
        """
        ...


T = TypeVar("T")
ImplT = TypeVar("ImplT", bound=Resizable)
IndexerT = TypeVar("IndexerT", int | None, slice)


class ViewableBase(Generic[ImplT, IndexerT]):
    """Abstract base class for classes with an ownership model and underlying Rust-bound data."""

    _impl: ImplT

    @classmethod
    @abstractmethod
    def _create(cls, impl: ImplT, indexer: IndexerT) -> Self:
        """Create an instance of ``cls``.

        Args:
            impl: Rust-bound object containing the data for this instance.
            indexer: Index or slice of the data in ``impl`` that this instance should view. The
                default value of ``None`` for an ``int`` or ``slice(None)`` for a ``slice``
                indicates that this instance is considered to be owning.

        Returns:
            A new instance of ``cls``.
        """
        pass

    @abstractmethod
    def is_owning(self) -> bool:
        """Check if ``self`` is owning (i.e. not a view)."""
        pass

    @abstractmethod
    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        pass

    def into(self, t: type[OutT]) -> OutT:
        """Clone ``self`` into a new related container of type ``t``.

        Args:
            t: Type of the new container to create.

        Returns:
            A new instance of ``t`` containing the same data as ``self``.
        """
        from zixy.container.convert import into  # noqa: PLC0415

        return into(self, t)


class ViewableItem(ViewableBase[ImplT, int | None]):
    """Abstract base class for items with an ownership model and underlying Rust-bound data."""

    _index: int | None

    @classmethod
    @abstractmethod
    def _create(cls, impl: ImplT, indexer: int | None = None) -> Self:
        """Create an instance of ``cls``.

        Args:
            impl: Rust-bound object containing the data for this item.
            indexer: Index of the item within ``impl``. If ``None``, this instance is considered to
                be owning.

        Returns:
            A new instance of ``cls``.
        """
        pass

    def is_owning(self) -> bool:
        """Check if ``self`` is owning (i.e. not a view)."""
        return self._index is None

    @property
    def index(self) -> int:
        """Get the index of ``self`` within its underlying data."""
        return 0 if self._index is None else self._index


class ViewableSequence(ViewableBase[ImplT, slice], Generic[T, ImplT], Sequence[T]):
    """Abstract base class for sequences with an ownership model and underlying Rust-bound data."""

    _slice: slice

    @classmethod
    @abstractmethod
    def _create(cls, impl: ImplT, indexer: slice = slice(None)) -> Self:
        """Create a new instance of ``cls``.

        Args:
            impl: Rust-bound object containing the data for this sequence.
            indexer: Slice of the data in ``impl`` that this instance should view. The default
                value of ``slice(None)`` indicates that this instance is considered to be owning.

        Returns:
            A new instance of ``cls``.
        """
        pass

    @classmethod
    def _create_view(cls, impl: ImplT) -> Self:
        """Factory method to create a viewing instance of ``cls``.

        Args:
            impl: Rust-bound object containing the data for this sequence.

        Returns:
            A viewing instance of ``cls``.
        """
        return cls._create(impl, slice(None, len(impl)))

    @property
    def slice(self) -> builtins.slice:
        """Get the slice of the underlying data that ``self`` views."""
        return self._slice

    def map_index(self, i: int) -> int:
        """Map an index in ``self`` to an index in the underlying data.

        Args:
            i: Index in ``self``.

        Returns:
            Corresponding index in the underlying data.
        """
        return slice_index(self._slice, i, len(self._impl))

    @overload
    def __getitem__(self, indexer: int) -> T: ...

    @overload
    def __getitem__(self, indexer: builtins.slice) -> Self: ...

    def __getitem__(self, indexer: int | builtins.slice) -> T | Self:
        """Get the element or elements selected by ``indexer``.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by ``indexer``.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        """Iterate over the elements of ``self``."""
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of ``self``."""
        pass

    def __len__(self) -> int:
        """Get the number of elements in ``self``."""
        return slice_len(self._slice, len(self._impl))

    def as_view(self) -> Self:
        """Return a view of ``self``.

        Returns:
            If ``self`` is owning, a new view on the same underlying data, otherwise
            ``self``.
        """
        if self.is_owning():
            return self._create_view(self._impl)
        else:
            return self

    def is_owning(self) -> bool:
        """Check if ``self`` is owning (i.e. not a view)."""
        return self.slice == slice(None)

    @requires_ownership
    def resize(self, n: int) -> Self:
        """Resize the underlying container.

        Args:
            n: The new size of the container.

        Note:
            This method operates in-place.

        Raises:
            ValueError: If the container is a view.
        """
        self._impl.resize(n)
        assert len(self) == n
        return self

    @classmethod
    def from_view(cls, source: Self) -> Self:
        """Create a new instance of ``cls`` from a view.

        Args:
            source: View to clone into the new instance.

        Returns:
            An owning clone of ``source``.
        """
        return source.clone()

    @classmethod
    def from_size(cls, n: int) -> Self:
        """Create a new instance of ``cls`` with the given size.

        Args:
            n: The size of the new instance.

        Returns:
            An instance of ``cls`` with the given size.
        """
        out = cls()
        out.resize(n)
        return out


class StringRepresentable(ABC):
    """Abstract base class for classes that can be represented as, and constructed from, strings."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        pass

    def __str__(self) -> str:
        """Return a string representation of ``self``."""
        return repr(self)

    def to_string(self) -> str:
        """Return a string representation of ``self``."""
        return str(self)

    @classmethod
    @abstractmethod
    def from_str(cls, s: str) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            s: String to parse.

        Returns:
            An instance of ``cls`` parsed from ``s``.
        """
        pass

    @classmethod
    def parse(cls, s: str) -> Self:
        """Parse a string into an instance of ``cls``.

        Args:
            s: String to parse.

        Returns:
            An instance of ``cls`` parsed from ``s``.
        """
        return cls.from_str(s)
