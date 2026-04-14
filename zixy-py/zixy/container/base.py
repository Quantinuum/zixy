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
from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence, Sized
from functools import wraps
from typing import (
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


class SupportsOwnership(Protocol):
    """Protocol for classes supporting the ownership model."""

    def is_owning(self) -> bool:
        """Check if :param:`self` is owning (i.e. not a view)."""
        ...

    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
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


class ViewableItem(Generic[ImplT]):
    """Abstract base class for items with an ownership model and underlying Rust-bound data."""

    _impl: ImplT
    _index: int | None

    @classmethod
    @abstractmethod
    def _create(cls, impl: ImplT, index: int | None = None) -> Self:
        """Create an instance of :param:`cls`.

        Args:
            impl: Rust-bound object containing the data for this item.
            index: Index of the item within :param:`impl`. If ``None``, this instance is
                considered to be owning.

        Returns:
            A new instance of :param:`cls`.
        """
        pass

    def is_owning(self) -> bool:
        """Check if :param:`self` is owning (i.e. not a view)."""
        return self._index is None

    @property
    def index(self) -> int:
        """Get the index of :param:`self` within its underlying data."""
        return 0 if self._index is None else self._index

    @abstractmethod
    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        pass


class ViewableSequence(Generic[T, ImplT], Sequence[T]):
    """Abstract base class for sequences with an ownership model and underlying Rust-bound data."""

    _impl: ImplT
    _slice: slice

    @classmethod
    @abstractmethod
    def _create(cls, impl: ImplT, s: slice = slice(None)) -> Self:
        """Create a new instance of :param:`cls`.

        Args:
            impl: Rust-bound object containing the data for this sequence.
            s: Slice of the data in :param:`impl` that this instance should view. If ``None``, this
                instance is considered to be owning.

        Returns:
            A new instance of :param:`cls`.
        """
        pass

    @classmethod
    def _create_view(cls, impl: ImplT) -> Self:
        """Factory method to create a viewing instance of :param:`cls`.

        Args:
            impl: Rust-bound object containing the data for this sequence.

        Returns:
            A viewing instance of :param:`cls`.
        """
        return cls._create(impl, slice(None, len(impl)))

    @property
    def slice(self) -> slice:
        """Get the slice of the underlying data that :param:`self` views."""
        return self._slice

    def map_index(self, i: int) -> int:
        """Map an index in :param:`self` to an index in the underlying data.

        Args:
            i: Index in :param:`self`.

        Returns:
            Corresponding index in the underlying data.
        """
        return slice_index(self._slice, i, len(self._impl))

    @overload
    def __getitem__(self, indexer: int) -> T: ...

    @overload
    def __getitem__(self, indexer: builtins.slice) -> Self: ...

    def __getitem__(self, indexer: int | builtins.slice) -> T | Self:
        """Get the element or elements selected by :param:`indexer`.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by :param:`indexer`.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[T]:
        """Iterate over the elements of :param:`self`."""
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of :param:`self`."""
        pass

    def __len__(self) -> int:
        """Get the number of elements in :param:`self`."""
        return slice_len(self._slice, len(self._impl))

    def as_view(self) -> Self:
        """Return a view of :param:`self`.

        Returns:
            If :param:`self` is owning, a new view on the same underlying data, otherwise
            :param:`self`.
        """
        if self.is_owning():
            return self._create_view(self._impl)
        else:
            return self

    @abstractmethod
    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        pass

    def is_owning(self) -> bool:
        """Check if :param:`self` is owning (i.e. not a view)."""
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
        """Create a new instance of :param:`cls` from a view.

        Args:
            source: View to clone into the new instance.

        Returns:
            An owning clone of :param:`source`.
        """
        return source.clone()

    @classmethod
    def from_size(cls, n: int) -> Self:
        """Create a new instance of :param:`cls` with the given size.

        Args:
            n: The size of the new instance.

        Returns:
            An instance of :param:`cls` with the given size.
        """
        out = cls()
        out.resize(n)
        return out
