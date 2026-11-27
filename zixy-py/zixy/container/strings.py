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

"""String sequences."""

from __future__ import annotations

import builtins
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self, cast, overload

from zixy.container.base import Representable, ViewableSequence, requires_ownership
from zixy.container.serialization import SerializableContainer
from zixy.container.string import String
from zixy.utils import slice_index_gen, slice_len, slice_of_slice, slice_to_tuple

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from zixy._zixy import ImplArray
    from zixy.container.base import _FilterableStrings, _HasStringType
    from zixy.container.serialization import SerializedContainer
    from zixy.container.string_set import StringSet


class Strings[ImplT: ImplArray, SpecT](
    ViewableSequence[String[ImplT, SpecT], ImplT], Representable, SerializableContainer
):
    """A collection of strings.

    An array-like container of strings that may be an owning instance referencing a contiguous
    Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    string_type: type[String[ImplT, SpecT]]

    _impl: ImplT

    def __init__(self, impl: ImplT, indexer: slice = slice(None)):
        """Initialize the string array.

        Args:
            impl: Rust-bound object storing the data.
            indexer: Slice of elements within Rust-bound array that are to be viewed by ``self``.
                The default value of ``slice(None)`` indicates that ``self`` is taken to be an
                owning view on all elements of ``impl``.
        """
        assert self._set_type().strings_type is type(self)
        assert isinstance(impl, self.string_type.impl_type), type(impl)
        self._impl = impl
        self._slice = indexer

    @classmethod
    def _create(cls, impl: ImplT) -> Self:
        """Create an owning instance of ``cls``, wrapping existing storage.

        Args:
            impl: Rust-bound object containing the data for this instance.

        Returns:
            A new owning instance of ``cls`` using the supplied storage without copying it.
        """
        assert isinstance(impl, cls.string_type.impl_type)
        out = cls.__new__(cls)
        Strings.__init__(out, impl)
        return out

    @classmethod
    def _view(cls, impl: ImplT, indexer: slice) -> Self:
        """Create a mutable view on existing strings.

        Args:
            impl: Rust-bound object containing the data for this instance.
            indexer: Slice selecting the viewed strings.

        Returns:
            A non-owning instance of ``cls`` sharing the selected storage without copying it.
        """
        out = cls.__new__(cls)
        if indexer == slice(None):
            indexer = slice(0, len(impl), 1)
        Strings.__init__(out, impl, indexer)
        return out

    def _new(self) -> Self:
        """Create an empty instance with the same type and configuration."""
        return self._create(self._impl.strings_clone([]))

    @classmethod
    def from_string(cls, string: String[ImplT, SpecT]) -> Self:
        """Create a new one-element instance from a single string.

        Args:
            string: String to copy into the new instance.

        Returns:
            New owning instance containing only ``string``.
        """
        if not isinstance(string._impl, cls.string_type.impl_type):
            cls.string_type.raise_spec_type_error(string)
        return cls._create(string.clone()._impl)

    @classmethod
    def from_iterable(
        cls,
        source: Iterable[SpecT | String[ImplT, SpecT]],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of ``cls`` from an iterable.

        Args:
            source: Iterable returning specifiers of all the strings to be appended.
            args: Positional arguments to forward to the constructor of ``cls``.
            kwargs: Keyword arguments to forward to the constructor of ``cls``.

        Returns:
            New instance of ``cls`` containing the strings specified by ``source``.
        """
        out = cls(*args, **kwargs)
        out.extend(source)
        return out

    def new_string[StringT: String[Any, Any]](
        self: _HasStringType[StringT],
    ) -> StringT:
        """Create an independent string with the same space and configuration.

        Returns:
            An owning instance of ``self.string_type`` with independent storage.
        """
        strings = cast("Strings[Any, Any]", self)
        data = strings._new().resize(1)
        return cast("StringT", strings.string_type._create(data._impl))

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        inds = slice_to_tuple(self.slice, len(self._impl))
        return self._create(self._impl.strings_clone(inds))

    @classmethod
    @abstractmethod
    def _set_type(cls) -> type[StringSet[ImplT, SpecT]]:
        """Return the string set class paired with this collection."""

    @overload
    def __getitem__[StringT: String[Any, Any]](
        self: _HasStringType[StringT], indexer: int
    ) -> StringT: ...

    @overload
    def __getitem__(self, indexer: builtins.slice) -> Self: ...

    def __getitem__(self, indexer: int | builtins.slice) -> object:
        """Get the element or elements selected by ``indexer``.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by ``indexer``.
        """
        if isinstance(indexer, int):
            return self.string_type._view(self._impl, self.map_index(indexer))
        else:
            return type(self)._view(
                self._impl, slice_of_slice(self.slice, indexer, len(self._impl))
            )

    @overload
    def __setitem__(self, indexer: int, source: SpecT | String[ImplT, SpecT]) -> None: ...

    @overload
    def __setitem__(
        self,
        indexer: builtins.slice,
        source: SpecT | String[ImplT, SpecT] | Strings[ImplT, SpecT],
    ) -> None: ...

    def __setitem__(
        self,
        indexer: int | builtins.slice,
        source: SpecT | String[ImplT, SpecT] | Strings[ImplT, SpecT],
    ) -> None:
        """Set the string at ``indexer`` in ``self`` to ``source``.

        Args:
            indexer: Index of the string or slice of strings within ``self`` to assign.
            source: Value specifying the string or a view of many strings to assign at
                ``indexer``.
        """
        if isinstance(indexer, builtins.slice):
            if isinstance(source, Strings):
                if source._impl.same_as(self._impl):
                    # internal copy, need to ensure source is not overwritten in write.
                    source = source.clone()
                n = slice_len(indexer, len(self))
                # write a slice to a slice
                if n != len(source):
                    raise ValueError(
                        f"Length of source ({len(source)}) does not match "
                        f"length of destination ({n})"
                    )
                for i_src, i_dst in enumerate(slice_index_gen(indexer, len(self))):
                    self[i_dst].set(source[i_src])
            else:
                # write the same term value to each element of the slice.
                for i in slice_index_gen(indexer, len(self)):
                    self[i].set(source)
        else:
            if isinstance(source, Strings):
                raise ValueError(
                    f"Cannot assign a {self.__class__.__name__} instance to an integer indexer."
                )
            self[indexer].set(source)

    @requires_ownership
    def append(self, source: SpecT | String[ImplT, SpecT]) -> Self:
        """Append ``source`` to the end of ``self``.

        Args:
            source: Value to append.

        Note:
            This method operates in-place.
        """
        return self.append_n(1, source)

    @requires_ownership
    def append_n(self, n: int, source: SpecT | String[ImplT, SpecT]) -> Self:
        """Append ``source`` to the end of ``self`` ``n`` times.

        Args:
            n: Number of times to repeatedly append ``source``.
            source: Specification for the value to append.

        Note:
            This method operates in-place.
        """
        if source is None:
            source = self.string_type._clear_spec
        n_old = len(self)
        self.resize(len(self) + n)
        for i in range(n_old, len(self)):
            self[i] = source
        return self

    @requires_ownership
    def extend(self, source: Iterable[SpecT | String[ImplT, SpecT]] = tuple()) -> Self:
        """Append the elements of ``source`` to the end of ``self``.

        Args:
            source: Iterable of compatible strings or string specifiers.

        Returns:
            ``self`` with the source elements appended in iteration order.

        Raises:
            ValueError: If ``self`` is a non-owning view, including for an empty source.

        Note:
            This method operates in-place. Extending from ``self`` or an overlapping
            view appends the original visible elements once, in source order.
        """
        for item in source:
            self.append(item)
        return self

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, Strings):
            return NotImplemented
        inds = slice_to_tuple(self.slice, len(self._impl))
        other_inds = slice_to_tuple(other.slice, len(other._impl))
        return self._impl.strings_eq(inds, other._impl, other_inds)

    def iter_filter_map[StringT: String[Any, Any]](
        self: _HasStringType[StringT], f: Callable[[StringT], bool]
    ) -> Iterator[StringT]:
        """Lazily evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the strings of ``self``, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.string.String` instance as
                an argument, and return a
                boolean.

        Returns:
            Iterator over the selected (and possibly mutated) strings of ``self`` according
            to ``f``.
        """
        strings = cast("Strings[Any, Any]", self)
        for string in strings:
            tmp = cast("StringT", string.clone())
            if f(tmp):
                yield tmp

    def filter_map[StringsT: Strings[Any, Any], StringT: String[Any, Any]](
        self: _FilterableStrings[StringsT, StringT], f: Callable[[StringT], bool]
    ) -> StringsT:
        """Eagerly evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the strings of ``self``, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.string.String` instance as
                an argument, and return a
                boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) strings
            of ``self`` according to ``f``.
        """
        out = self._new()
        out.extend(self.iter_filter_map(f))
        return out

    def iter_filter_populated[StringT: String[Any, Any]](
        self: _HasStringType[StringT],
    ) -> Iterator[StringT]:
        """Lazily filter strings of ``self``, retaining only those that are not clear."""
        strings = cast("Strings[Any, Any]", self)
        tmp = strings.new_string()
        for view in strings:
            if view != tmp:
                yield cast("StringT", view)

    def filter_populated(self) -> Self:
        """Eagerly filter strings of ``self``, retaining only those that are not clear."""
        out = self._new()
        out.extend(self.iter_filter_populated())
        return out

    def filter_unique(self) -> Self:
        """Create a new collection containing the unique strings of ``self``."""
        unique = self._set_type().from_strings(self)
        return self._create(unique._impl.strings_clone(None))

    def reordered(self, inds: Sequence[int]) -> Self:
        """Get a new instance with the elements of ``self`` in a new order.

        Args:
            inds: Sequence of indices defining the new order. Should be a permutation of
                ``range(len(self))``.

        Returns:
            A new instance with the reordered elements.
        """
        if not all(i < len(self) for i in inds):
            raise IndexError("Index out of bounds in reorder.")
        out = self._new()
        for i in inds:
            out.append(self[i])
        return out

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return ", ".join(str(s) for s in self)

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
