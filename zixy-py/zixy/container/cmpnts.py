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

"""Components and collections of components.

Components (:class:`Cmpnt`) are the general building blocks of terms. They encapsulate an underlying
Rust-bound data object, the identity of which depends on the domain the component is defined for.

Collections of components (:class:`Cmpnts`) are array-like containers of such components. They can
either own the data they reference, or reference a slice of the data in another collection.

Sets of components (:class:`CmpntSet`) are set-like containers of components, storing unique
components, and allowing set-like operations on them. The set holds its own copy of the data, and is
not modified by changes to any other view.
"""

from __future__ import annotations

import builtins
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    overload,
)

from typing_extensions import Self

from zixy._zixy import Map
from zixy.container.base import ViewableItem, ViewableSequence, requires_ownership
from zixy.container.coeffs import Coeff, CoeffT, OtherCoeffT
from zixy.utils import slice_index_gen, slice_len, slice_of_slice, slice_to_tuple

if TYPE_CHECKING:
    from zixy.container.terms import Term, TermRegistry


if TYPE_CHECKING:
    from zixy._zixy import ImplArray
else:
    ImplArray = object


ImplT = TypeVar("ImplT", bound=ImplArray)
SpecT = TypeVar("SpecT")


class Cmpnt(ViewableItem[ImplT], Generic[ImplT, SpecT]):
    """A component.

    A single component that may be an owning instance referencing a single element in a Rust-bound
    data object, or a view on an element in another collection.
    """

    impl_type: type[ImplT]
    _term_registry: TermRegistry[ImplT, SpecT]

    _impl: ImplT
    _index: int | None

    def __init__(self, impl: ImplT, index: int | None = None):
        """Initialize the component.

        Args:
            impl: Rust-bound object storing the data.
            index: Position within Rust-bound array at which the viewed component is stored. When
                :param:`index` is ``None``, the :class:`Cmpnt` is an owning view on the sole element
                of :param:`impl`.
        """
        assert isinstance(impl, self.impl_type)
        self._impl = impl
        self._index = index
        self._check_bounds()

    @classmethod
    def _create(cls, impl: ImplT, index: int | None = None) -> Self:
        """Create an instance of :param:`cls`.

        Args:
            impl: Rust-bound object containing the data for this item.
            index: Index of the item within :param:`impl`. If ``None``, this instance is
                considered to be owning.

        Returns:
            A new instance of :param:`cls`.
        """
        out = cls.__new__(cls)
        Cmpnt.__init__(out, impl, index)
        return out

    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        return type(self)._create(self._impl.cmpnts_clone([self.index]))

    def aliases(self, other: Cmpnt[ImplT, SpecT]) -> bool:
        """Determine whether :param:`self` is a view of the same component as :param:`other`."""
        if type(self) is not type(other) or self.is_owning():
            # owners never alias other objects
            return False
        return self._impl.same_as(other._impl) and self.index == other.index

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, Cmpnt):
            return NotImplemented
        return self._impl.cmpnts_eq([self.index], other._impl, [other.index])

    @abstractmethod
    def set(self, source: SpecT | Self | None) -> None:
        """Set the value of the component.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        pass

    def clear(self) -> None:
        """Set the value of the component to zero.

        Note:
            This method operates in-place.
        """
        self.set(None)

    @classmethod
    def raise_spec_type_error(cls, source: Any) -> None:
        """Raise a ``TypeError`` for unsupported component specifiers.

        Raises:
            TypeError: The specifier supplied to set the value of :param:`self` is of an unsupported
                type.
        """
        raise TypeError(
            f"Cannot set value of a {cls} instance with a variable of type {type(source)}"
        )

    def _check_bounds(self) -> None:
        """Validate that :param:`self` references a valid element in its implementation array.

        Raises:
            IndexError: The index of :param:`self` is out of bounds for the implementation array it
                views.
        """
        if len(self._impl) != 1 and self.is_owning():
            raise IndexError("Owning Cmpnt must have exactly one element in the impl array.")
        index = self.index
        if index < 0:
            index += len(self._impl)
        if index < 0 or index >= len(self._impl):
            raise IndexError(
                f"Index {self.index} is out of bounds for an array of length {len(self._impl)}"
            )

    @overload
    def __mul__(self, rhs: Cmpnt[ImplT, SpecT]) -> Term[ImplT, SpecT, OtherCoeffT]: ...

    @overload
    def __mul__(self, rhs: CoeffT) -> Term[ImplT, SpecT, CoeffT]: ...

    def __mul__(
        self, rhs: CoeffT | Cmpnt[ImplT, SpecT]
    ) -> Term[ImplT, SpecT, CoeffT] | Term[ImplT, SpecT, OtherCoeffT]:
        """Return the product of :param:`self` and :param:`rhs`."""
        if not isinstance(rhs, Coeff):
            # Cmpnt multiplication is not defined for base Cmpnt, but may be define by derived class
            return NotImplemented
        term_type = self._term_registry[type(rhs)]
        return term_type.from_cmpnt_coeff(self, rhs)

    def __rmul__(self, lhs: CoeffT) -> Term[ImplT, SpecT, CoeffT]:
        """Return the product of :param:`lhs` and :param:`self`."""
        if not isinstance(lhs, Coeff):
            return NotImplemented
        term_type = self._term_registry[type(lhs)]
        return term_type.from_cmpnt_coeff(self, lhs)


class CmpntSet(Generic[ImplT, SpecT]):
    """A collection of unique components.

    A set-like container of components that may be used to store unique components and perform
    set-like operations on them.
    """

    cmpnts_type: type[Cmpnts[ImplT, SpecT]]
    map_type: type[Map] = Map

    _impl: ImplT
    _map: Map

    def __init__(self, impl: ImplT):
        """Initialize the component set.

        Args:
            impl: Rust-bound object storing the data. Unlike :class:`Cmpnts`, this is copied from,
                not referenced directly by :param:`self`.
        """
        assert isinstance(impl, self.cmpnts_type.cmpnt_type.impl_type), type(impl)
        self._impl = impl.cmpnts_clone([])
        self._map = self.map_type()
        self._working_cmpnt = self.cmpnts_type._create(impl.cmpnts_clone([])).new_clear_cmpnt()
        self.insert_iterable(self.cmpnts_type._create(impl))

    @classmethod
    def _create(cls, impl: ImplT) -> Self:
        """Create a new instance of :param:`cls`.

        Args:
            impl: Rust-bound object storing the data. Unlike :class:`Cmpnts`, this is copied from,
                not referenced directly by :param:`self`.

        Returns:
            A new instance of :param:`cls`.
        """
        out = cls.__new__(cls)
        CmpntSet.__init__(out, impl)
        return out

    @classmethod
    def from_cmpnts(cls, cmpnts: Cmpnts[ImplT, SpecT]) -> Self:
        """Create a new instance of :param:`cls` from the components in :param:`cmpnts`.

        Args:
            cmpnts: Owned or viewed components with which to populate the new instance.

        Returns:
            A new instance of :param:`cls` containing the components in :param:`cmpnts`.
        """
        out = cls._create(cmpnts._impl.cmpnts_clone([]))
        out.insert_iterable(cmpnts)
        return out

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of :param:`self`."""
        return self._create(self._impl.cmpnts_clone([]))

    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        out = self._empty_clone()
        out.insert_iterable(self)
        return out

    def to_cmpnts(self) -> Cmpnts[ImplT, SpecT]:
        """Create a new array of components owning copies of all those contained in :param:`self`.

        Returns:
            New :class:`Cmpnts` instance containing copies of the components in :param:`self`. The
            type is determined by :attr:`~zixy.container.cmpnts.CmpntSet.cmpnts_type`.
        """
        return self.cmpnts_type._create(self._impl.cmpnts_clone(None))

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return ", ".join(str(s) for s in self)

    def __len__(self) -> int:
        """Get the number of elements in :param:`self`."""
        return len(self._impl)

    def _get_working_cmpnt(
        self, value: SpecT | Cmpnt[ImplT, SpecT] | None = None
    ) -> Cmpnt[ImplT, SpecT]:
        """Get a :class:`Cmpnt` instance that contains the component specified by :param:`value`.

        Args:
            value: The component specifier.

        Returns:
            A :class:`Cmpnt` instance containing the component specified by :param:`value`. If
            :param:`value` is already a :class:`Cmpnt` instance, it is returned directly, otherwise
            the value is set in a working :class:`Cmpnt` instance and that instance is returned.
        """
        if isinstance(value, self.cmpnts_type.cmpnt_type):
            return value
        else:
            self._working_cmpnt.set(value)
            return self._working_cmpnt

    def insert(self, value: SpecT | Cmpnt[ImplT, SpecT]) -> int:
        """Try to insert the given component.

        Args:
            value: The component specifier.

        Returns:
            The index at which the term was inserted, or the index at which it already was
            stored if insertion is unsuccessful.

        Note:
            This method operates in-place.
        """
        value = self._get_working_cmpnt(value)
        out = self._impl.mapped_insert(self._map, value._impl, value.index)
        assert len(self) == len(self._map)
        return out

    def insert_iterable(self, source: Iterable[SpecT | Cmpnt[ImplT, SpecT]] = tuple()) -> None:
        """Insert many components from an iterable source.

        Args:
            source: Iterable over any component specifiers.

        Note:
            This method operates in-place.
        """
        for item in source:
            self.insert(item)

    def lookup(self, value: SpecT | Cmpnt[ImplT, SpecT]) -> int | None:
        """Try to find the index of :param:`value` in :param:`self`.

        Args:
            value: The component specifier.

        Returns:
            The index at which the value was inserted, or ``None`` if the value was not found.
        """
        value = self._get_working_cmpnt(value)
        return self._impl.mapped_lookup(self._map, value._impl, value.index)

    def contains(self, value: SpecT | Cmpnt[ImplT, SpecT]) -> bool:
        """Check whether :param:`value` is stored in :param:`self`.

        Args:
            value: The component specifier.

        Returns:
            Whether the lookup of :param:`value` was successful.
        """
        return self.lookup(value) is not None

    def remove(self, value: SpecT | Cmpnt[ImplT, SpecT]) -> int:
        """Try to remove :param:`value` from :param:`self`.

        Args:
            value: The component specifier.

        Returns:
            The index at which the value was removed.

        Raises:
            KeyError: The value was not found.
        """
        value = self._get_working_cmpnt(value)
        i = self._impl.mapped_remove(self._map, value._impl, value.index)
        if i is None:
            raise KeyError(f"Value {value} not found in {self.__class__.__name__}.")
        return i

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, CmpntSet):
            return NotImplemented
        return self._impl.mapped_equal(self._map, other._impl)

    def __iter__(self) -> Iterator[Cmpnt[ImplT, SpecT]]:
        """Iterate over the elements of :param:`self`."""
        if not len(self):
            return
        tmp = self._get_working_cmpnt()
        for i in range(len(self)):
            tmp._impl.cmpnt_copy_external(0, self._impl, i)
            yield tmp

    def iter_filter_map(
        self, f: Callable[[Cmpnt[ImplT, SpecT]], bool]
    ) -> Iterator[Cmpnt[ImplT, SpecT]]:
        """Lazily evaluate a filter-map operation over the components of :param:`self`.

        Args:
            f: Function which may mutate copies of the components of :param:`self`, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.cmpnts.Cmpnt` instance
                as an argument, and return a
                boolean.

        Returns:
            Iterator over the selected (and possibly mutated) components of :param:`self` according
            to :param:`f`. The type of the components is determined by
            :attr:`~zixy.container.cmpnts.Cmpnts.cmpnts_type`.
        """
        tmp = self.cmpnts_type._create(self._impl.cmpnts_clone(None))
        return tmp.iter_filter_map(f)

    def filter_map(self, f: Callable[[Cmpnt[ImplT, SpecT]], bool]) -> Self:
        """Eagerly evaluate a filter-map operation over the components of :param:`self`.

        Args:
            f: Function which may mutate copies of the components of :param:`self`, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`~zixy.container.cmpnts.Cmpnt` instance
                as an argument, and return a
                boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) components
            of :param:`self` according to :param:`f`. The type of the components is determined by
            :attr:`~zixy.container.cmpnts.Cmpnts.cmpnts_type`.
        """
        out = self._empty_clone()
        out.insert_iterable(self.iter_filter_map(f))
        return out

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[SpecT | Cmpnt[ImplT, SpecT]],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of :param:`cls` from an iterable.

        Args:
            iterable: Iterable returning specifiers of all the components to be appended.
            args: Positional arguments to forward to the constructor of :param:`cls`.
            kwargs: Keyword arguments to forward to the constructor of :param:`cls`.

        Returns:
            New instance of :param:`cls` containing the components specified by :param:`iterable`.
        """
        out = cls(*args, **kwargs)
        out.insert_iterable(iterable)
        return out


class Cmpnts(Generic[ImplT, SpecT], ViewableSequence[Cmpnt[ImplT, SpecT], ImplT]):
    """A collection of components.

    An array-like container of components that may be an owning instance referencing a contiguous
    Rust-bound data object, or a view on a slice of the elements in another collection.
    """

    cmpnt_type: type[Cmpnt[ImplT, SpecT]]

    _impl: ImplT
    _set_type: type[CmpntSet[ImplT, SpecT]] = CmpntSet

    def __init__(self, impl: ImplT, s: slice = slice(None)):
        """Initialize the component array.

        Args:
            impl: Rust-bound object storing the data.
            s: Slice of elements within Rust-bound array that are to be viewed by :param:`self`.
                When :param:`s` is ``None``, :param:`self` is taken to be an owning view on all
                elements of :param:`impl`.
        """
        assert self._set_type.cmpnts_type is type(self)
        assert isinstance(impl, self.cmpnt_type.impl_type), type(impl)
        self._impl = impl
        self._slice = s

    @classmethod
    def _create(cls, impl: ImplT, s: slice = slice(None)) -> Self:
        """Create a new instance of :param:`cls`.

        Args:
            impl: Rust-bound object containing the data for this sequence.
            s: Slice of the data in :param:`impl` that this instance should view. If ``None``, this
                instance is considered to be owning.

        Returns:
            A new instance of :param:`cls`.
        """
        assert isinstance(impl, cls.cmpnt_type.impl_type)
        out = cls.__new__(cls)
        Cmpnts.__init__(out, impl, s)
        return out

    @classmethod
    def from_cmpnt(cls, cmpnt: Cmpnt[ImplT, SpecT]) -> Self:
        """Create a new one-element instance from a single component.

        Args:
            cmpnt: Component to copy into the new instance.

        Returns:
            New owning instance containing only :param:`cmpnt`.
        """
        assert isinstance(cmpnt, cls.cmpnt_type), type(cmpnt)
        return cls._create(cmpnt.clone()._impl)

    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        inds = slice_to_tuple(self.slice, len(self._impl))
        return self._create(self._impl.cmpnts_clone(inds))

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, Cmpnts):
            return NotImplemented
        inds = slice_to_tuple(self.slice, len(self._impl))
        other_inds = slice_to_tuple(other.slice, len(other._impl))
        return self._impl.cmpnts_eq(inds, other._impl, other_inds)

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return ", ".join(str(s) for s in self)

    @overload
    def __getitem__(self, indexer: int) -> Cmpnt[ImplT, SpecT]: ...

    @overload
    def __getitem__(self, indexer: builtins.slice) -> Self: ...

    def __getitem__(self, indexer: int | builtins.slice) -> Cmpnt[ImplT, SpecT] | Self:
        """Get the element or elements selected by :param:`indexer`.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by :param:`indexer`.
        """
        if isinstance(indexer, int):
            return self.cmpnt_type._create(self._impl, self.map_index(indexer))
        else:
            return type(self)._create(
                self._impl, slice_of_slice(self.slice, indexer, len(self._impl))
            )

    def __setitem__(
        self,
        indexer: int | builtins.slice,
        source: SpecT | Cmpnt[ImplT, SpecT] | Cmpnts[ImplT, SpecT] | None,
    ) -> None:
        """Set the component at :param:`indexer` in :param:`self` to :param:`source`.

        Args:
            indexer: Index of the string or slice of strings within :param:`self` to assign.
            source: Value specifying the component or a view of many components to assign at
                :param:`indexer`.
        """
        if isinstance(indexer, builtins.slice):
            if isinstance(source, Cmpnts):
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
            if isinstance(source, Cmpnts):
                raise ValueError(
                    f"Cannot assign a {self.__class__.__name__} instance to an integer indexer."
                )
            self[indexer].set(source)

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of :param:`self`."""
        return self._create(self._impl.cmpnts_clone([]))

    def reordered(self, inds: Sequence[int]) -> Self:
        """Get a new instance with the elements of :param:`self` in a new order.

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

    def new_clear_cmpnt(self) -> Cmpnt[ImplT, SpecT]:
        """Get a new :class:`Cmpnt` instance using the same implementation as :param:`self`."""
        return self._empty_clone().resize(1)[0]

    def iter_filter_map(
        self, f: Callable[[Cmpnt[ImplT, SpecT]], bool]
    ) -> Iterator[Cmpnt[ImplT, SpecT]]:
        """Lazily evaluate a filter-map operation over the components of :param:`self`.

        Args:
            f: Function which may mutate copies of the components of :param:`self`, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`Cmpnt` instance as an argument, and return a
                boolean.

        Returns:
            Iterator over the selected (and possibly mutated) components of :param:`self` according
            to :param:`f`.
        """
        if not len(self):
            return
        tmp = self[0].clone()
        if f(tmp):
            yield tmp
        for i in range(1, len(self)):
            tmp.set(self[i])
            if f(tmp):
                yield tmp

    def filter_map(self, f: Callable[[Cmpnt[ImplT, SpecT]], bool]) -> Self:
        """Eagerly evaluate a filter-map operation over the components of :param:`self`.

        Args:
            f: Function which may mutate copies of the components of :param:`self`, returning
                ``True`` if those mutated copies are to be included in the generator. The function
                signature should take a single :class:`Cmpnt` instance as an argument, and return a
                boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) components
            of :param:`self` according to :param:`f`.
        """
        out = self._empty_clone()
        out.append_iterable(self.iter_filter_map(f))
        return out

    def filter_unique(self) -> Cmpnts[ImplT, SpecT]:
        """Get a new :class:`Cmpnts` instance containing the unique components of :param:`self`."""
        return self._set_type(self._impl).to_cmpnts()

    def iter_filter_populated(self) -> Iterator[Cmpnt[ImplT, SpecT]]:
        """Lazily filter components of :param:`self`, retaining only those that are not clear."""
        tmp = self.new_clear_cmpnt()
        for view in self:
            if view != tmp:
                yield view

    def filter_populated(self) -> Self:
        """Eagerly filter components of :param:`self`, retaining only those that are not clear."""
        out = self._empty_clone()
        out.append_iterable(self.iter_filter_populated())
        return out

    @requires_ownership
    def append_n(self, n: int, source: SpecT | Cmpnt[ImplT, SpecT] | None = None) -> Self:
        """Append :param:`source` to the end of :param:`self` :param:`n` times.

        Args:
            n: Number of times to repeatedly append :param:`source`.
            source: Specification for the value to append.

        Note:
            This method operates in-place.
        """
        n_old = len(self)
        self.resize(len(self) + n)
        for i in range(n_old, len(self)):
            self[i] = source
        return self

    @requires_ownership
    def append(self, source: SpecT | Cmpnt[ImplT, SpecT] | None = None) -> Self:
        """Append :param:`source` to the end of :param:`self`.

        Args:
            source: Value to append.

        Note:
            This method operates in-place.
        """
        return self.append_n(1, source)

    def append_iterable(
        self, source: Iterable[SpecT | Cmpnt[ImplT, SpecT] | None] = tuple()
    ) -> Self:
        """Append many values from an iterable source.

        Args:
            source: Iterable over any component specifiers.

        Note:
            This method operates in-place.
        """
        for item in source:
            self.append(item)
        return self

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[SpecT | Cmpnt[ImplT, SpecT]],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of :param:`cls` from an iterable.

        Args:
            iterable: Iterable returning specifiers of all the components to be appended.
            args: Positional arguments to forward to the constructor of :param:`cls`.
            kwargs: Keyword arguments to forward to the constructor of :param:`cls`.

        Returns:
            New instance of :param:`cls` containing the components specified by :param:`iterable`.
        """
        out = cls(*args, **kwargs)
        out.append_iterable(iterable)
        return out
