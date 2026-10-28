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

"""Base classes and protocols for containers."""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence, Sized
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypedDict,
    cast,
    overload,
)

from zixy.utils import slice_index, slice_len

if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable, Iterator
    from typing import Any, Concatenate, NotRequired, Self

    from zixy.container.coeffs import Coeffs


class _Indexable[T](Protocol):
    """An integer-indexed container whose element type can be inferred independently."""

    def __getitem__(self, indexer: int) -> T:
        """Get an element by index."""
        ...

    def __len__(self) -> int:
        """Get the number of elements."""
        ...


class _HasStringType[StringT](Protocol):
    """A container exposing its string factory."""

    @property
    def string_type(self) -> type[StringT]: ...


class _HasStringsType[StringsT](Protocol):
    """A container exposing its string-collection factory."""

    @property
    def strings_type(self) -> type[StringsT]: ...


class _HasStrings[StringsT](Protocol):
    """Storage exposing its concrete string collection."""

    @property
    def strings(self) -> StringsT: ...


class _HasCoeffs[Cs](Protocol):
    """Storage exposing its concrete coefficient collection."""

    @property
    def coeffs(self) -> Cs: ...


class _HasCoeffsType[Cs](Protocol):
    """A container exposing its coefficient-collection factory."""

    coeffs_type: type[Cs]


class _HasCoeffType[C](Protocol):
    """A coefficient collection exposing its scalar type."""

    @property
    def coeff_type(self) -> type[C]: ...


class _HasEmptyClone[ContainerT](Protocol):
    """A container whose empty clone preserves its concrete type."""

    def _empty_clone(self) -> ContainerT: ...


class _FilterableStrings[ContainerT, StringT](
    _HasStringType[StringT], _HasEmptyClone[ContainerT], Protocol
):
    """String factories and clones used to infer filter inputs and results together."""

    def iter_filter_map(self, f: Callable[[StringT], bool]) -> Iterator[StringT]: ...

    def iter_filter_populated(self) -> Iterator[StringT]: ...


class _FilterableStringSet[ContainerT, StringT](
    _HasStringsType[_HasStringType[StringT]], _HasEmptyClone[ContainerT], Protocol
):
    """String-set factories and clones used to infer filter inputs and results together."""

    def iter_filter_map(self, f: Callable[[StringT], bool]) -> Iterator[StringT]: ...


class _HasTermType[TermT](Protocol):
    """A container exposing its term factory."""

    @property
    def term_type(self) -> type[TermT]: ...


class _HasTermsType[TermsT](Protocol):
    """A container exposing its term-collection factory."""

    @property
    def terms_type(self) -> type[TermsT]: ...


class _TermCoeffsView[Cs](_HasTermType[_HasCoeffsType[Cs]], Protocol):
    """A term-array factory and the storage and slice needed to access its coefficients."""

    @property
    def _data(self) -> _HasCoeffs[Coeffs[Any]]: ...

    @property
    def slice(self) -> slice: ...


class _TermSetCoeffs[Cs](_HasTermsType[_HasTermType[_HasCoeffsType[Cs]]], Protocol):
    """A term-set factory and the storage needed to access its coefficients."""

    @property
    def _data(self) -> _HasCoeffs[Coeffs[Any]]: ...


class SupportsOwnership(Protocol):
    """Protocol for classes supporting the ownership model."""

    def is_owning(self) -> bool:
        """Check if ``self`` is owning (i.e. not a view)."""
        ...

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        ...


def requires_ownership[T: ViewableBase[Any, Any], **P, R](
    method: Callable[Concatenate[T, P], R],
) -> Callable[Concatenate[T, P], R]:
    """Decorator for methods that can only be called on owning views.

    Args:
        method: Method to decorate.

    Returns:
        Decorated method that raises a ``ValueError`` if called on a non-owning view.
    """

    @wraps(method)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
        if not self.is_owning():
            raise ValueError(
                f"Cannot call {method.__name__} on a non-owning view. Consider cloning the object "
                "first using the .clone() method."
            )
        return method(self, *args, **kwargs)

    return cast("Callable[Concatenate[T, P], R]", wrapper)


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


class ViewableBase[ImplT: Resizable, IndexerT: (int | None, slice)]:
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

    def into[T: ViewableBase[Any, Any]](self, t: type[T]) -> T:
        """Clone ``self`` into a new related container of type ``t``.

        Args:
            t: Type of the new container to create.

        Returns:
            A new instance of ``t`` containing the same data as ``self``.
        """
        from zixy.container.convert import into  # noqa: PLC0415

        return into(self, t)


class ViewableItem[ImplT: Resizable](ViewableBase[ImplT, int | None]):
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
        """The index of ``self`` within its underlying data."""
        return 0 if self._index is None else self._index


class ViewableSequence[T, ImplT: Resizable](ViewableBase[ImplT, slice], Sequence[T]):
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
        """The slice of the underlying data that ``self`` views."""
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

    def __iter__[ElemT](self: _Indexable[ElemT]) -> Iterator[ElemT]:
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


class Representable(ABC):
    """Abstract base class for classes that can be represented as, and constructed from, strings."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        pass

    def to_str(self) -> str:
        """Return the canonical string representation of ``self``."""
        return repr(self)

    def __str__(self) -> str:
        """Return a string representation of ``self``."""
        return self.to_str()

    @classmethod
    @abstractmethod
    def from_str(cls, source: str, *args: Any, **kwargs: Any) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.
            *args: Positional arguments to forward to a subsequent constructor.
            **kwargs: Keyword arguments to forward to a subsequent constructor.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        pass

    @classmethod
    def parse(cls, source: str) -> Self:
        """Parse a string into an instance of ``cls``.

        Deprecated:
            use :meth:`from_str` instead.

        Args:
            source: String to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        warnings.warn(
            f"{cls.__name__}.parse is deprecated, use {cls.__name__}.from_str instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_str(source)


def to_hashable(value: Any) -> Hashable:
    """Convert a value to a hashable representation."""
    if isinstance(value, dict):
        return frozenset((to_hashable(k), to_hashable(v)) for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        return tuple(to_hashable(v) for v in value)
    elif isinstance(value, set):
        return frozenset(to_hashable(v) for v in value)
    if not isinstance(value, Hashable):
        raise TypeError(f"value of type {type(value)} is not hashable")
    return value


class SerializedContainer(TypedDict):
    """Serialized representation of a container."""

    version: str
    """Version of the package."""

    type_name: str
    """Name of the type of the container."""

    type_module: str
    """Module where the type of the container is defined."""

    strings: NotRequired[dict[str, Any]]
    """List of strings in the container."""

    coeffs: NotRequired[dict[str, Any]]
    """List of coefficients in the container."""

    space: NotRequired[dict[str, Any]]
    """Space for the domain of the container."""


class Serializable[SerializedT: dict[str, Any]](ABC):
    """Abstract base class for classes that can be serialized to and deserialized from JSON."""

    @abstractmethod
    def to_dict(self) -> SerializedT:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            A JSON-serializable dictionary representation of ``self``.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, source: SerializedT) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        pass

    def to_json(self) -> str:
        """Return a JSON string representation of ``self``.

        Returns:
            A JSON string representation of ``self``.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, source: str) -> Self:
        """Create an instance of ``cls`` from a JSON string representation.

        Args:
            source: JSON string representation to parse.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        return cls.from_dict(json.loads(source))

    def __hash__(self) -> int:
        """Return a hash of the object based on its dictionary representation."""
        return hash(to_hashable(self.to_dict()))


class SerializableContainer(Serializable[SerializedContainer], ABC):
    """Abstract base class for containers that can be serialized to and deserialized from JSON."""

    pass
