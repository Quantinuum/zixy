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

"""Strings and collections of strings.

Strings (:class:`~zixy.container.strings.String`) are the general building blocks of terms. They
encapsulate an
underlying Rust-bound data object whose type depends on the domain of the string.

Collections of strings (:class:`~zixy.container.strings.Strings`) are array-like containers of such
strings. They can
either own the data they reference, or reference a slice of the data in another collection.

Sets of strings (:class:`~zixy.container.strings.StringSet`) are set-like containers of strings,
storing unique
strings, and allowing set-like operations on them. The set holds its own copy of the data, and is
not modified by changes to any other view.
"""

from __future__ import annotations

import builtins
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self, overload

from zixy._zixy import Map
from zixy.container.base import (
    StringRepresentable,
    ViewableItem,
    ViewableSequence,
    requires_ownership,
)
from zixy.utils import slice_index_gen, slice_len, slice_of_slice, slice_to_tuple

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from zixy._zixy import ImplArray


class String[ImplT: ImplArray, SpecT](ViewableItem[ImplT], StringRepresentable):
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
                ``indexer`` is ``None``, the :class:`~zixy.container.strings.String` is an owning
                view on the sole element
                of ``impl``.
        """
        assert isinstance(impl, self.impl_type)
        self._impl = impl
        self._index = indexer
        self._check_bounds()

    @classmethod
    def _create(cls, impl: ImplT, indexer: int | None = None) -> Self:
        """Create an instance of ``cls``.

        Args:
            impl: Rust-bound object containing the data for this item.
            indexer: Index of the item within ``impl``. If ``None``, this instance is
                considered to be owning.

        Returns:
            A new instance of ``cls``.
        """
        out = cls.__new__(cls)
        String.__init__(out, impl, indexer)
        return out

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        return type(self)._create(self._impl.strings_clone([self.index]))

    def aliases(self, other: String[ImplT, SpecT]) -> bool:
        """Determine whether ``self`` is a view of the same string as ``other``."""
        if type(self) is not type(other) or self.is_owning():
            # owners never alias other objects
            return False
        return self._impl.same_as(other._impl) and self.index == other.index

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, String):
            return NotImplemented
        return self._impl.strings_eq([self.index], other._impl, [other.index])

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return self._impl.string_to_string(self.index)

    @abstractmethod
    def set(self, source: SpecT | Self) -> None:
        """Set the value of the string.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        pass

    def clear(self) -> None:
        """Set the value of the string to zero.

        Note:
            This method operates in-place.
        """
        self.set(self._clear_spec)

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


class StringSet[ImplT: ImplArray, SpecT](StringRepresentable):
    """A collection of unique strings.

    A set-like container of strings that may be used to store unique strings and perform
    set-like operations on them.
    """

    strings_type: type[Strings[ImplT, SpecT]]
    map_type: type[Map] = Map

    _impl: ImplT
    _map: Map

    def __init__(self, impl: ImplT):
        """Initialize the string set.

        Args:
            impl: Rust-bound object storing the data. Unlike
            :class:`~zixy.container.strings.Strings`, this is copied from,
                not referenced directly by ``self``.
        """
        assert isinstance(impl, self.strings_type.string_type.impl_type), type(impl)
        self._impl = impl.strings_clone([])
        self._map = self.map_type()
        self._working_string = self.strings_type._create(impl.strings_clone([])).new_clear_string()
        self.insert_iterable(self.strings_type._create(impl))

    @classmethod
    def _create(cls, impl: ImplT, map_: Map | None = None) -> Self:
        """Create a new instance of ``cls``.

        Args:
            impl: Rust-bound object storing the data. Unlike
            :class:`~zixy.container.strings.Strings`, this is copied from,
                not referenced directly by ``self``.
            map_: Populated lookup map for ``impl``. If supplied, adopt both directly.

        Returns:
            A new instance of ``cls``.
        """
        out = cls.__new__(cls)
        if map_ is None:
            StringSet.__init__(out, impl)
        else:
            assert isinstance(impl, cls.strings_type.string_type.impl_type), type(impl)
            assert len(impl) == len(map_)
            out._impl = impl
            out._map = map_
            out._working_string = cls.strings_type._create(
                impl.strings_clone([])
            ).new_clear_string()
        return out

    @classmethod
    def from_strings(cls, strings: Strings[ImplT, SpecT]) -> Self:
        """Create a new instance of ``cls`` from the strings in ``strings``.

        Args:
            strings: Owned or viewed strings with which to populate the new instance.

        Returns:
            A new instance of ``cls`` containing the strings in ``strings``.
        """
        out = cls._create(strings._impl.strings_clone([]))
        out.insert_iterable(strings)
        return out

    @classmethod
    def from_str(cls, source: str, *args: Any, **kwargs: Any) -> Self:
        """Create an instance of ``cls`` from a string.

        Args:
            source: String to parse.
            *args: Positional arguments to forward to the string constructor.
            **kwargs: Keyword arguments to forward to the string constructor.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        strings = cls.strings_type.from_str(source, *args, **kwargs)
        return cls.from_strings(strings)

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of ``self``."""
        return self._create(self._impl.strings_clone([]))

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        out = self._empty_clone()
        out.insert_iterable(self)
        return out

    def to_strings(self) -> Strings[ImplT, SpecT]:
        """Create a new array of strings owning copies of all those contained in ``self``.

        Returns:
            New :class:`~zixy.container.strings.Strings` instance containing copies of the strings
            in ``self``. The
            type is determined by :attr:`~zixy.container.strings.StringSet.strings_type`.
        """
        return self.strings_type._create(self._impl.strings_clone(None))

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return ", ".join(str(s) for s in self)

    def __len__(self) -> int:
        """Get the number of elements in ``self``."""
        return len(self._impl)

    def _get_working_string(self, value: SpecT | String[ImplT, SpecT]) -> String[ImplT, SpecT]:
        """Prepare a string containing ``value``.

        Args:
            value: The string specifier.

        Returns:
            A :class:`~zixy.container.strings.String` instance containing the string specified by
            ``value``. If
            ``value`` is already a :class:`~zixy.container.strings.String` instance, it is returned
            directly, otherwise
            the value is set in a working :class:`~zixy.container.strings.String` instance and that
            instance is returned.
        """
        if isinstance(value, self.strings_type.string_type):
            return value
        else:
            self._working_string.set(value)
            return self._working_string

    def insert(self, value: SpecT | String[ImplT, SpecT]) -> int:
        """Try to insert the given string.

        Args:
            value: The string specifier.

        Returns:
            The index at which the term was inserted, or the index at which it already was
            stored if insertion is unsuccessful.

        Note:
            This method operates in-place.
        """
        value = self._get_working_string(value)
        out = self._impl.mapped_insert(self._map, value._impl, value.index)
        assert len(self) == len(self._map)
        return out

    def insert_iterable(self, source: Iterable[SpecT | String[ImplT, SpecT]] = tuple()) -> None:
        """Insert many strings from an iterable source.

        Args:
            source: Iterable over any string specifiers.

        Note:
            This method operates in-place.
        """
        for item in source:
            self.insert(item)

    def lookup(self, value: SpecT | String[ImplT, SpecT]) -> int | None:
        """Try to find the index of ``value`` in ``self``.

        Args:
            value: The string specifier.

        Returns:
            The index at which the value was inserted, or ``None`` if the value was not found.
        """
        value = self._get_working_string(value)
        return self._impl.mapped_lookup(self._map, value._impl, value.index)

    def contains(self, value: SpecT | String[ImplT, SpecT]) -> bool:
        """Check whether ``value`` is stored in ``self``.

        Args:
            value: The string specifier.

        Returns:
            Whether the lookup of ``value`` was successful.
        """
        return self.lookup(value) is not None

    def remove(self, value: SpecT | String[ImplT, SpecT]) -> int:
        """Try to remove ``value`` from ``self``.

        Args:
            value: The string specifier.

        Returns:
            The index at which the value was removed.

        Raises:
            KeyError: The value was not found.
        """
        value = self._get_working_string(value)
        i = self._impl.mapped_remove(self._map, value._impl, value.index)
        if i is None:
            raise KeyError(f"Value {value} not found in {self.__class__.__name__}.")
        return i

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, StringSet):
            return NotImplemented
        return self._impl.mapped_equal(self._map, other._impl)

    def __iter__(self) -> Iterator[String[ImplT, SpecT]]:
        """Iterate over the elements of ``self``."""
        for i in range(len(self)):
            yield self.strings_type.string_type._create(self._impl, i)

    def iter_filter_map(
        self, f: Callable[[String[ImplT, SpecT]], bool]
    ) -> Iterator[String[ImplT, SpecT]]:
        """Lazily evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the strings of ``self``, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.strings.String` instance
                as an argument, and return a
                boolean.

        Returns:
            Iterator over the selected (and possibly mutated) strings of ``self`` according
            to ``f``. The type of the strings is determined by
            :attr:`~zixy.container.strings.Strings.strings_type`.
        """
        tmp = self.strings_type._create(self._impl.strings_clone(None))
        return tmp.iter_filter_map(f)

    def filter_map(self, f: Callable[[String[ImplT, SpecT]], bool]) -> Self:
        """Eagerly evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the strings of ``self``, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.strings.String` instance
                as an argument, and return a
                boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) strings
            of ``self`` according to ``f``. The type of the strings is determined by
            :attr:`~zixy.container.strings.Strings.strings_type`.
        """
        out = self._empty_clone()
        out.insert_iterable(self.iter_filter_map(f))
        return out

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[SpecT | String[ImplT, SpecT]],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of ``cls`` from an iterable.

        Args:
            iterable: Iterable returning specifiers of all the strings to be appended.
            args: Positional arguments to forward to the constructor of ``cls``.
            kwargs: Keyword arguments to forward to the constructor of ``cls``.

        Returns:
            New instance of ``cls`` containing the strings specified by ``iterable``.
        """
        out = cls(*args, **kwargs)
        out.insert_iterable(iterable)
        return out


class Strings[ImplT: ImplArray, SpecT](
    ViewableSequence[String[ImplT, SpecT], ImplT], StringRepresentable
):
    """A collection of strings.

    An array-like container of strings that may be an owning instance referencing a contiguous
    Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    string_type: type[String[ImplT, SpecT]]

    _impl: ImplT
    _set_type: type[StringSet[ImplT, SpecT]] = StringSet

    def __init__(self, impl: ImplT, indexer: slice = slice(None)):
        """Initialize the string array.

        Args:
            impl: Rust-bound object storing the data.
            indexer: Slice of elements within Rust-bound array that are to be viewed by ``self``.
                The default value of ``slice(None)`` indicates that ``self`` is taken to be an
                owning view on all elements of ``impl``.
        """
        assert self._set_type.strings_type is type(self)
        assert isinstance(impl, self.string_type.impl_type), type(impl)
        self._impl = impl
        self._slice = indexer

    @classmethod
    def _create(cls, impl: ImplT, indexer: slice = slice(None)) -> Self:
        """Create a new instance of ``cls``.

        Args:
            impl: Rust-bound object containing the data for this sequence.
            indexer: Slice of the data in ``impl`` that this instance should view. The default
                value of ``slice(None)`` indicates that this instance is considered to be owning.

        Returns:
            A new instance of ``cls``.
        """
        assert isinstance(impl, cls.string_type.impl_type)
        out = cls.__new__(cls)
        Strings.__init__(out, impl, indexer)
        return out

    @classmethod
    def from_string(cls, string: String[ImplT, SpecT]) -> Self:
        """Create a new one-element instance from a single string.

        Args:
            string: String to copy into the new instance.

        Returns:
            New owning instance containing only ``string``.
        """
        assert isinstance(string, cls.string_type), type(string)
        return cls._create(string.clone()._impl)

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        inds = slice_to_tuple(self.slice, len(self._impl))
        return self._create(self._impl.strings_clone(inds))

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, Strings):
            return NotImplemented
        inds = slice_to_tuple(self.slice, len(self._impl))
        other_inds = slice_to_tuple(other.slice, len(other._impl))
        return self._impl.strings_eq(inds, other._impl, other_inds)

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return ", ".join(str(s) for s in self)

    @overload
    def __getitem__(self, indexer: int) -> String[ImplT, SpecT]: ...

    @overload
    def __getitem__(self, indexer: builtins.slice) -> Self: ...

    def __getitem__(self, indexer: int | builtins.slice) -> String[ImplT, SpecT] | Self:
        """Get the element or elements selected by ``indexer``.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by ``indexer``.
        """
        if isinstance(indexer, int):
            return self.string_type._create(self._impl, self.map_index(indexer))
        else:
            return type(self)._create(
                self._impl, slice_of_slice(self.slice, indexer, len(self._impl))
            )

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

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of ``self``."""
        return self._create(self._impl.strings_clone([]))

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
        out = self._empty_clone()
        for i in inds:
            out.append(self[i])
        return out

    def new_clear_string(self) -> String[ImplT, SpecT]:
        """Create a cleared string using the same implementation as ``self``."""
        return self._empty_clone().resize(1)[0]

    def iter_filter_map(
        self, f: Callable[[String[ImplT, SpecT]], bool]
    ) -> Iterator[String[ImplT, SpecT]]:
        """Lazily evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the strings of ``self``, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.strings.String` instance as
                an argument, and return a
                boolean.

        Returns:
            Iterator over the selected (and possibly mutated) strings of ``self`` according
            to ``f``.
        """
        for string in self:
            tmp = string.clone()
            if f(tmp):
                yield tmp

    def filter_map(self, f: Callable[[String[ImplT, SpecT]], bool]) -> Self:
        """Eagerly evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the strings of ``self``, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.strings.String` instance as
                an argument, and return a
                boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) strings
            of ``self`` according to ``f``.
        """
        out = self._empty_clone()
        out.append_iterable(self.iter_filter_map(f))
        return out

    def filter_unique(self) -> Strings[ImplT, SpecT]:
        """Create a new collection containing the unique strings of ``self``."""
        return self._set_type(self._impl).to_strings()

    def iter_filter_populated(self) -> Iterator[String[ImplT, SpecT]]:
        """Lazily filter strings of ``self``, retaining only those that are not clear."""
        tmp = self.new_clear_string()
        for view in self:
            if view != tmp:
                yield view

    def filter_populated(self) -> Self:
        """Eagerly filter strings of ``self``, retaining only those that are not clear."""
        out = self._empty_clone()
        out.append_iterable(self.iter_filter_populated())
        return out

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
    def append(self, source: SpecT | String[ImplT, SpecT]) -> Self:
        """Append ``source`` to the end of ``self``.

        Args:
            source: Value to append.

        Note:
            This method operates in-place.
        """
        return self.append_n(1, source)

    def append_iterable(self, source: Iterable[SpecT | String[ImplT, SpecT]] = tuple()) -> Self:
        """Append many values from an iterable source.

        Args:
            source: Iterable over any string specifiers.

        Note:
            This method operates in-place.
        """
        for item in source:
            self.append(item)
        return self

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[SpecT | String[ImplT, SpecT]],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of ``cls`` from an iterable.

        Args:
            iterable: Iterable returning specifiers of all the strings to be appended.
            args: Positional arguments to forward to the constructor of ``cls``.
            kwargs: Keyword arguments to forward to the constructor of ``cls``.

        Returns:
            New instance of ``cls`` containing the strings specified by ``iterable``.
        """
        out = cls(*args, **kwargs)
        out.append_iterable(iterable)
        return out
