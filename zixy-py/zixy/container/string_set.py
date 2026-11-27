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

"""String sets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

from zixy._zixy import Map
from zixy.container.base import Representable
from zixy.container.serialization import SerializableContainer
from zixy.container.string import String
from zixy.container.strings import Strings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from zixy._zixy import ImplArray
    from zixy.container.base import (
        _FilterableStrings,
        _FilterableStringSet,
        _HasStringsType,
        _HasStringType,
    )
    from zixy.container.serialization import SerializedContainer


class StringSet[ImplT: ImplArray, SpecT](Representable, SerializableContainer):
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
        self._bind(impl.strings_clone([]), self.map_type())
        self.insert_iterable(self.strings_type._view(impl, slice(None)))

    def _bind(self, impl: ImplT, map_: Map) -> None:
        """Set the string data and lookup map for ``self``.

        Args:
            impl: Rust-bound object containing the data for this instance.
            map_: Populated lookup map corresponding to the supplied data.
        """
        assert isinstance(impl, self.strings_type.string_type.impl_type), type(impl)
        assert len(impl) == len(map_)
        self._impl = impl
        self._map = map_
        self._working_string = self.strings_type._view(impl, slice(None)).new_string()

    @classmethod
    def _create(cls, impl: ImplT, *, map_: Map) -> Self:
        """Create an owning instance of ``cls``, wrapping existing storage.

        Args:
            impl: Rust-bound object containing the data for this instance.
            map_: Populated lookup map corresponding to the supplied data.

        Returns:
            A new owning instance of ``cls`` using the supplied storage without copying it.
        """
        out = cls.__new__(cls)
        out._bind(impl, map_)
        return out

    def _new(self) -> Self:
        """Create an empty instance with the same type and configuration."""
        return self._create(self._impl.strings_clone([]), map_=self.map_type())

    @classmethod
    def from_strings(cls, strings: Strings[ImplT, SpecT]) -> Self:
        """Create a new instance of ``cls`` from the strings in ``strings``.

        Args:
            strings: Owned or viewed strings with which to populate the new instance.

        Returns:
            A new instance of ``cls`` containing the strings in ``strings``.
        """
        out = cls._create(strings._impl.strings_clone([]), map_=cls.map_type())
        out.insert_iterable(strings)
        return out

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
        out.insert_iterable(source)
        return out

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        out = self._new()
        out.insert_iterable(self)
        return out

    def __len__(self) -> int:
        """Get the number of elements in ``self``."""
        return len(self._impl)

    def __iter__[StringT: String[Any, Any]](
        self: _HasStringsType[_HasStringType[StringT]],
    ) -> Iterator[StringT]:
        """Yield an independent, owning copy of each key."""
        strings = cast("StringSet[Any, Any]", self)
        for i in range(len(strings)):
            yield cast(
                "StringT",
                strings.strings_type.string_type._create(strings._impl.strings_clone([i])),
            )

    def __contains__(self, value: object) -> bool:
        """Check whether ``value`` is stored in ``self``.

        Args:
            value: String or string specifier to look up.

        Returns:
            Whether the string is present. Unsupported objects, out-of-range string
            specifiers, and strings from incompatible families or spaces return ``False``.
        """
        try:
            return self.lookup_index(cast("SpecT | String[ImplT, SpecT]", value)) is not None
        except (TypeError, ValueError, IndexError):
            return False

    def lookup_index(self, key: SpecT | String[ImplT, SpecT]) -> int | None:
        """Try to find the storage index of ``key`` in ``self``.

        Args:
            key: String or string specifier to look up.

        Returns:
            The current storage index, or ``None`` if the string is absent. Indices
            may change when strings are removed.

        Raises:
            TypeError: The key is unsupported or belongs to an incompatible family.
            ValueError: The key is invalid or belongs to an incompatible space.
        """
        value = self._get_working_string(key)
        return self._impl.mapped_lookup(self._map, value._impl, value.index)

    def _get_working_string(self, value: SpecT | String[ImplT, SpecT]) -> String[ImplT, SpecT]:
        """Prepare a string containing ``value``.

        Args:
            value: The string specifier.

        Returns:
            A :class:`~zixy.container.string.String` instance containing the string specified by
            ``value``. If
            ``value`` is already a :class:`~zixy.container.string.String` instance, it is returned
            directly, otherwise
            the value is set in a working :class:`~zixy.container.string.String` instance and that
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

    def iter_filter_map[StringT: String[Any, Any]](
        self: _HasStringsType[_HasStringType[StringT]], f: Callable[[StringT], bool]
    ) -> Iterator[StringT]:
        """Lazily evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the strings of ``self``, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.string.String` instance
                as an argument, and return a
                boolean.

        Returns:
            Iterator over the selected (and possibly mutated) strings of ``self`` according
            to ``f``. The type of the strings is determined by
            :attr:`~zixy.container.strings.Strings.strings_type`.
        """
        strings = cast("StringSet[Any, Any]", self)
        tmp = cast("_FilterableStrings[Strings[Any, Any], StringT]", strings.to_strings())
        return tmp.iter_filter_map(f)

    def filter_map[SetT: StringSet[Any, Any], StringT: String[Any, Any]](
        self: _FilterableStringSet[SetT, StringT], f: Callable[[StringT], bool]
    ) -> SetT:
        """Eagerly evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the strings of ``self``, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.string.String` instance
                as an argument, and return a
                boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) strings
            of ``self`` according to ``f``. The type of the strings is determined by
            :attr:`~zixy.container.strings.Strings.strings_type`.
        """
        out = self._new()
        out.insert_iterable(self.iter_filter_map(f))
        return out

    def __repr__(self) -> str:
        """Return a string representation of ``self``."""
        return ", ".join(str(s) for s in self)

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

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: JSON-serializable dictionary representation to parse.

        Returns:
            A new owning instance of ``cls`` parsed from ``source``, with storage independent of
            the source dictionary and any existing containers.
        """
        return cls.from_strings(cls.strings_type.from_dict(source))

    def to_strings[StringsT: Strings[Any, Any]](
        self: _HasStringsType[StringsT],
    ) -> StringsT:
        """Create a new array of strings owning copies of all those contained in ``self``.

        Returns:
            New :class:`~zixy.container.strings.Strings` instance containing copies of the strings
            in ``self``. The
            type is determined by :attr:`~zixy.container.string_set.StringSet.strings_type`.
        """
        strings = cast("StringSet[Any, Any]", self)
        return self.strings_type._create(strings._impl.strings_clone(None))
