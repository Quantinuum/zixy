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

"""Terms and collections of terms.

Terms (:class:`Term`) are algebraic objects composed of a component and a coefficient. They
encapsulate a :class:`~zixy.container.data.TermData` object which wraps the component and
coefficient either as single owning instances or views of elements in another collection.

Terms are stored in collections of terms (:class:`Terms`), which are array-like containers that
similarly encapsulate a :class:`~zixy.container.data.TermData` object which wraps the
components and coefficients either as owning instances or views of slices of another collection.

Sets of terms (:class:`TermSet`) are set-like containers of terms, storing unique components and
their coefficients, and allowing set-like operations on them. They are implemented using a hash
map.

Sums of terms (:class:`TermSum`) extend the :class:`TermSet` interface with linear combination
operations.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeAlias,
    cast,
    overload,
)

import numpy as np
import pandas as pd
from sympy import Expr
from typing_extensions import Self

from zixy.container.base import ViewableItem, ViewableSequence, requires_ownership
from zixy.container.cmpnts import Cmpnt, Cmpnts, CmpntSet, ImplT, SpecT
from zixy.container.coeffs import (
    Coeff,
    Coeffs,
    CoeffT,
    ComplexSign,
    Number,
    NumberT,
    RootOfUnity,
    Sign,
    _is_complex,
    _is_complex_sign,
    _is_expr,
    _is_float,
    _is_int,
    _is_sign,
    convert,
    convert_vec,
    get_coeffs_type,
    typesafe_mul,
    unit,
    zero,
)
from zixy.container.data import TermData
from zixy.container.mixins import TermMulMixin
from zixy.utils import DEFAULT_ATOL, DEFAULT_RTOL, slice_index_gen, slice_of_slice

TermSpecT: TypeAlias = (
    Cmpnt[ImplT, SpecT] | SpecT | tuple[SpecT | Cmpnt[ImplT, SpecT] | None, CoeffT | None] | None
)


class Term(
    ViewableItem[TermData[ImplT, SpecT, CoeffT]],
    Generic[ImplT, SpecT, CoeffT],
    TermMulMixin[ImplT, SpecT, CoeffT],
):
    """A term consisting of a component and a coefficient.

    A single term consisting of a component and a coefficient that may be an owning instance
    referencing the sole element in an owned :class:`~zixy.container.data.TermData` instance, or
    a view on an element in another collection.
    """

    coeff_type: type[CoeffT]
    cmpnts_type: type[Cmpnts[ImplT, SpecT]]

    _impl: TermData[ImplT, SpecT, CoeffT]
    _index: int | None

    def __init__(self, data: TermData[ImplT, SpecT, CoeffT], index: int | None = None):
        """Initialize the term.

        Args:
            data: Raw term data object, of which :param:`self` views one.
            index: Index of the term within :param:`data` that :param:`self` views.
        """
        if data.cmpnts_type is not self.cmpnts_type:
            raise TypeError(
                f"Term data cmpnts type {data.cmpnts_type} does not match "
                f"the cmpnts type {self.cmpnts_type} of this {type(self)}."
            )
        if data.coeff_type is not self.coeff_type:
            raise TypeError(
                f"Term data coeff type {data.coeff_type} does not match "
                f"the coeff type {self.coeff_type} of this {type(self)}."
            )
        self._impl = data
        self._index = index

    @property
    def _data(self) -> TermData[ImplT, SpecT, CoeffT]:
        """Get the raw term data object underlying :param:`self`."""
        return self._impl

    @property
    def cmpnt_type(self) -> type[Cmpnt[ImplT, SpecT]]:
        """Get the component type of :param:`self`."""
        return self.cmpnts_type.cmpnt_type

    @property
    def coeffs_type(self) -> type[Coeffs[CoeffT]]:
        """Get the coefficient container type of :param:`self`."""
        return get_coeffs_type(self.coeff_type)

    @classmethod
    def _create(cls, data: TermData[ImplT, SpecT, CoeffT], index: int | None = None) -> Self:
        """Create an instance of :param:`cls`.

        Args:
            data: Raw term data object containing the data for this item.
            index: Index of the item within :param:`data`. If ``None``, this instance is
                considered to be owning.

        Returns:
            A new instance of :param:`cls`.
        """
        out = cls.__new__(cls)
        Term.__init__(out, data, index)
        return out

    @classmethod
    def from_cmpnt_coeff(cls, cmpnt: Cmpnt[ImplT, SpecT], coeff: CoeffT) -> Self:
        """Factory to make an instance of :param:`cls` from a component and a coefficient.

        Args:
            cmpnt: Component to copy from.
            coeff: Coefficient scaling the :param:`cmpnt` in the new instance.

        Returns:
            An instance of :param:`cls` with the given component and coefficient.
        """
        cmpnts = cls.cmpnts_type.from_cmpnt(cmpnt)
        coeffs = get_coeffs_type(cls.coeff_type)()
        coeffs.append(coeff)
        return cls._create(TermData(cmpnts, coeffs))

    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        return type(self)._create(self._impl.clone(self.index))

    @property
    def cmpnt(self) -> Cmpnt[ImplT, SpecT]:
        """Get a view on the component of :param:`self`."""
        return self._data.cmpnts[self.index]

    @property
    def coeff(self) -> CoeffT:
        """Get a copy of the coefficient of :param:`self`."""
        return self._data.coeffs[self.index]

    @coeff.setter
    def coeff(self, value: CoeffT) -> None:
        """Set the coefficient of :param:`self`.

        Args:
            value: Value to assign.

        Note:
            This method operates in-place.
        """
        self._data.coeffs[self.index] = convert(value, self._data.coeff_type)

    def into(self, t: type[Term[Any, Any, Any]]) -> Term[Any, Any, Any]:
        """Clone :param:`self` into a new term of type :param:`t`.

        Args:
            t: Type to return, must have the same
                :attr:`~zixy.container.terms.Term.cmpnt_type` as :param:`self`.

        Returns:
            A new instance of :param:`t`.
        """
        coeff = convert(self.coeff, t.coeff_type)
        out = t.from_cmpnt_coeff(self.cmpnt, coeff)
        if out.cmpnt_type is not self.cmpnt_type:
            raise TypeError(
                f"Cannot convert a term with component type "
                f"{self.cmpnt_type} into one with a different component "
                f"type {out.cmpnt_type}."
            )
        return out

    def aliases(self, other: Term[Any, Any, Any]) -> bool:
        """Determine whether :param:`self` is a view of the same component as :param:`other`."""
        if type(self) is not type(other):
            return False
        return self.cmpnt.aliases(other.cmpnt)

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, Term):
            return NotImplemented
        result: bool = (
            self._impl._cmpnts._impl.cmpnts_eq(
                [self.index], other._impl._cmpnts._impl, [other.index]
            )
            and self._impl._coeffs[self.index] == other._impl._coeffs[other.index]
        )
        return result

    def set(self, source: Self | TermSpecT[ImplT, SpecT, CoeffT]) -> None:
        """Set the value of the term.

        Args:
            source: Specification for the new value.

        Note:
            This method operates in-place.
        """
        try:
            # first, attempt to set just the cmpnt
            self.cmpnt.set(cast(SpecT | Cmpnt[ImplT, SpecT] | None, source))
            self.coeff = unit(self.coeff_type)
            return
        except (TypeError, ValueError):
            pass
        if isinstance(source, type(self)):
            self.cmpnt.set(source.cmpnt)
            self.coeff = source.coeff
            return
        if isinstance(source, tuple) and len(source) == 2:
            cmpnt, coeff = source
        else:
            cmpnt, coeff = source, None
        self.cmpnt.set(cmpnt)
        self.coeff = coeff if coeff is not None else unit(self.coeff_type)

    def clear(self) -> None:
        """Set the value of the component to zero and the coefficient to unity.

        Note:
            This method operates in-place.
        """
        self.set(None)

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        coeff = repr(self.coeff)
        coeff = coeff.replace("-0j", "+0j")
        return f"({coeff}, {self.cmpnt})"


class Terms(
    Generic[ImplT, SpecT, CoeffT],
    ViewableSequence[Term[ImplT, SpecT, CoeffT], TermData[ImplT, SpecT, CoeffT]],
):
    """A collection of terms consisting of components and coefficients.

    An array-like container of terms consisting of components and coefficients that may be an
    owning instance referencing a :class:`~zixy.container.data.TermData` instance, or a view on
    a slice of the elements in another collection.
    """

    term_type: type[Term[ImplT, SpecT, CoeffT]]

    _impl: TermData[ImplT, SpecT, CoeffT]

    def __init__(self, data: TermData[ImplT, SpecT, CoeffT], s: slice = slice(None)):
        """Initialize the term array.

        Args:
            data: Raw term data object, of which :param:`self` views a slice.
            s: Slice of the data in :param:`data` that :param:`self` will view. If ``None``, this
                instance is considered to be owning.
        """
        self._impl = data
        self._slice = s

    @property
    def _data(self) -> TermData[ImplT, SpecT, CoeffT]:
        """Return the raw term data object underlying :param:`self`."""
        return self._impl

    @classmethod
    def _create(cls, data: TermData[ImplT, SpecT, CoeffT], s: slice = slice(None)) -> Self:
        """Create a new instance of :param:`cls`.

        Args:
            data: Raw term data object containing the data for this sequence.
            s: Slice of the data in :param:`data` that this instance should view. If ``None``, this
                instance is considered to be owning.

        Returns:
            A new instance of :param:`cls`.
        """
        out = cls.__new__(cls)
        Terms.__init__(out, data, s)
        return out

    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        return self._create(self._impl.clone(self.slice))

    def into(self, t: type[Terms[Any, Any, Any]]) -> Terms[Any, Any, Any]:
        """Clone :param:`self` into a new collection of terms with type :param:`t`.

        Args:
            t: Type to return, must have the same
                :attr:`~zixy.container.terms.Terms.cmpnt_type` as :param:`self`.

        Returns:
            A new instance of :param:`t`.
        """
        coeffs = convert_vec(self.coeffs, get_coeffs_type(t.term_type.coeff_type))
        out = t._create(TermData(self.cmpnts.clone(), coeffs))
        if out.cmpnt_type is not self.cmpnt_type:
            raise TypeError(
                f"Cannot convert terms with component type "
                f"{out.cmpnt_type} into an instance with a different "
                f"component type {self.cmpnt_type}."
            )
        return out

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return all(self[i] == other[i] for i in range(len(self)))

    @property
    def coeffs_type(self) -> type[Coeffs[CoeffT]]:
        """Get the coefficient container type of :param:`self`."""
        return get_coeffs_type(self._data.coeff_type)

    @property
    def coeff_type(self) -> type[CoeffT]:
        """Get the coefficient type of :param:`self`."""
        return self._impl.coeff_type

    @property
    def cmpnts_type(self) -> type[Cmpnts[ImplT, SpecT]]:
        """Get the component container type of :param:`self`."""
        return self._data.cmpnts_type

    @property
    def cmpnt_type(self) -> type[Cmpnt[ImplT, SpecT]]:
        """Get the component type of :param:`self`."""
        return self._impl.cmpnt_type

    @property
    def cmpnts(self) -> Cmpnts[ImplT, SpecT]:
        """Get the components of :param:`self`."""
        return self._data.cmpnts[self.slice]

    @property
    def coeffs(self) -> Coeffs[CoeffT]:
        """Get the coefficients of :param:`self`."""
        return self._data.coeffs[self.slice]

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return ", ".join(str(s) for s in self)

    @overload
    def __getitem__(self, indexer: int) -> Term[ImplT, SpecT, CoeffT]: ...

    @overload
    def __getitem__(self, indexer: builtins.slice) -> Terms[ImplT, SpecT, CoeffT]: ...

    def __getitem__(
        self, indexer: int | builtins.slice
    ) -> Term[ImplT, SpecT, CoeffT] | Terms[ImplT, SpecT, CoeffT]:
        """Get the element or elements selected by :param:`indexer`.

        Args:
            indexer: Index or slice selecting the element(s) to return.

        Returns:
            Element or slice selected by :param:`indexer`.
        """
        if isinstance(indexer, int):
            if self.map_index(indexer) >= len(self._impl):
                raise IndexError
            return self.term_type._create(self._impl, self.map_index(indexer))
        else:
            return type(self)._create(
                self._data, slice_of_slice(self.slice, indexer, len(self._data))
            )

    @overload
    def __setitem__(
        self,
        indexer: int,
        source: TermSpecT[ImplT, SpecT, CoeffT] | Term[ImplT, SpecT, CoeffT],
    ) -> None: ...

    @overload
    def __setitem__(self, indexer: builtins.slice, source: Terms[ImplT, SpecT, CoeffT]) -> None: ...

    def __setitem__(
        self,
        indexer: int | builtins.slice,
        source: TermSpecT[ImplT, SpecT, CoeffT]
        | Term[ImplT, SpecT, CoeffT]
        | Terms[ImplT, SpecT, CoeffT]
        | None,
    ) -> None:
        """Set the term at :param:`indexer` in :param:`self` to :param:`source`.

        Args:
            indexer: Index of the string or slice of strings within :param:`self` to assign.
            source: Value specifying the term or a view of many components to assign at
                :param:`indexer`.
        """
        if isinstance(indexer, slice):
            if isinstance(source, Terms):
                self.cmpnts[indexer] = source.cmpnts
                self.coeffs[indexer] = source.coeffs
            else:
                # write the same term value to each element of the slice.
                for i in slice_index_gen(indexer, len(self)):
                    self[i].set(source)
        else:
            if isinstance(source, Terms):
                raise ValueError(
                    "Cannot assign a Terms object to a single index. "
                    "Consider assigning a single term or using a slice "
                    "index instead."
                )
            self[indexer].set(source)

    def scale(self, scalar: CoeffT) -> None:
        """Scale all elements by a given factor.

        Args:
            scalar: Scalar factor by which to scale all elements.

        Note:
            This method operates in-place.
        """
        for term in self:
            term.coeff = typesafe_mul(term.coeff, scalar)

    def __imul__(self, scalar: CoeffT) -> Self:
        """Multiply :param:`self` by :param:`scalar` in-place."""
        self.scale(scalar)
        return self

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of :param:`self`."""
        return self._create(self._impl._empty_clone())

    def reordered(self, inds: Sequence[int]) -> Self:
        """Get a new instance with the elements of :param:`self` in a new order.

        Args:
            inds: Sequence of indices defining the new order. Should be a permutation of
                ``tuple(range(len(self)))``.

        Returns:
            A new instance with the reordered elements.
        """
        if not all(i < len(self) for i in inds):
            raise IndexError("Index out of bounds in reorder.")
        out = self._empty_clone()
        for i in inds:
            out.append(self[i])
        return out

    def new_clear_term(self) -> Term[ImplT, SpecT, CoeffT]:
        """Factory to make a new term with a zero component and a unit coefficient.

        Returns:
            A new owing term with the component and coefficient type of :param:`self`.
        """
        try:
            tmp = self[0].clone()
            tmp.clear()
        except IndexError:
            copy = self.clone()
            copy.resize(1)
            tmp = copy[0]
        return tmp

    def iter_filter_map(
        self, f: Callable[[Term[ImplT, SpecT, CoeffT]], bool]
    ) -> Iterator[Term[ImplT, SpecT, CoeffT]]:
        """Lazily evaluate a filter-map operation over the components of :param:`self`.

        Args:
            f: Function which may mutate copies of the terms of :param:`self`, returning ``True`` if
                those mutated copies are to be included in the generator. The function signature
                should take a single :class:`Term` instance as an argument, and return a boolean.

        Returns:
            Iterator over the selected (and possibly mutated) terms of :param:`self` according
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

    def filter_map(self, f: Callable[[Term[ImplT, SpecT, CoeffT]], bool]) -> Self:
        """Eagerly evaluate a filter-map operation over the terms of :param:`self`.

        Args:
            f: Function which may mutate copies of the terms of :param:`self`, returning ``True``
                if those mutated copies are to be included in the generator. The function signature
                should take a single :class:`Term` instance as an argument, and return a boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) components
            of :param:`self` according to :param:`f`.
        """
        out = self._empty_clone()
        out.append_iterable(self.iter_filter_map(f))
        return out

    def append_n(
        self,
        n: int,
        source: TermSpecT[ImplT, SpecT, CoeffT] | Term[ImplT, SpecT, CoeffT] | None = None,
    ) -> Self:
        """Append :param:`source` to the end of :param:`self` :param:`n` times.

        Args:
            n: Number of times to repeatedly append :param:`source`.
            source: Value to append.

        Note:
            This method operates in-place.
        """
        n_old = len(self)
        self.resize(n_old + n)
        for i in range(n_old, len(self)):
            self[i] = source
        assert len(self) == n_old + n
        return self

    @requires_ownership
    def append(
        self,
        source: TermSpecT[ImplT, SpecT, CoeffT] | Term[ImplT, SpecT, CoeffT] | None = None,
    ) -> Self:
        """Append :param:`source` to the end of :param:`self`.

        Args:
            source: Value to append.

        Note:
            This method operates in-place.
        """
        return self.append_n(1, source)

    def append_iterable(
        self,
        source: Iterable[
            TermSpecT[ImplT, SpecT, CoeffT] | Term[ImplT, SpecT, CoeffT] | None
        ] = tuple(),
    ) -> Self:
        """Append the elements of :param:`source` to the end of :param:`self`.

        Args:
            source: Other iterable whose terms to append to :param:`self`.

        Note:
            This method operates in-place.
        """
        for item in source:
            self.append(item)
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Convert :param:`self` to a :class:`~pandas.DataFrame`."""
        return pd.DataFrame(
            {
                "Component": (str(c) for c in self.cmpnts),
                "Coefficient": (str(c) for c in self.coeffs),
            }
        )

    @classmethod
    def from_iterable(
        cls,
        source: Iterable[TermSpecT[ImplT, SpecT, CoeffT] | Term[ImplT, SpecT, CoeffT]],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of :param:`cls` from an iterable.

        Args:
            source: Iterable returning specifiers of all the terms to be appended.
            args: Positional arguments to forward to the constructor of :param:`cls`.
            kwargs: Keyword arguments to forward to the constructor of :param:`cls`.

        Returns:
            New instance of :param:`cls` containing the terms specified by :param:`source`.
        """
        out = cls(*args, **kwargs)
        out.append_iterable(source)
        return out


class NumericTerms(Terms[ImplT, SpecT, NumberT]):
    """A collection of terms consisting of components and numeric coefficients.

    An array-like container of terms consisting of components and coefficients that may be an
    owning instance referencing a :class:`~zixy.container.data.TermData` instance, or a view on
    a slice of the elements in another collection.
    """

    def allclose(self, other: Self, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> bool:
        """Check whether :param:`self` and :param:`other` are within a certain tolerance.

        Args:
            other: Other instance to compare to.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Whether :param:`self` and :param:`other` are within the given tolerances of each other.

        Note:
            This method operates by comparing the NumPy arrays returned by
            :attr:`~zixy.container.coeffs.Coeffs.np_array`.
        """
        out: bool = self.cmpnts == other.cmpnts and self.coeffs.allclose(other.coeffs, rtol, atol)
        return out


class TermSet(Generic[ImplT, SpecT, CoeffT]):
    """A collection of unique terms consisting of components and coefficients.

    A set-like container of terms that may be used to store unique terms and perform set-like
    operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[ImplT, SpecT, CoeffT]]

    _impl: TermData[ImplT, SpecT, CoeffT]
    _cmpnt_set: CmpntSet[ImplT, SpecT]
    _working_term: Term[ImplT, SpecT, CoeffT]

    def __init__(self, terms: Terms[ImplT, SpecT, CoeffT]):
        """Initialize the term set.

        Args:
            terms: Terms-derived object from which to construct the set of components and their
                coefficients.

        Note:
            Components which appear multiple times in :param:`terms` will appear only once in
                :param:`self`.
        """
        self._impl = terms._empty_clone()._impl
        cmpnt_set_type = terms.cmpnts_type._set_type
        self._cmpnt_set = cmpnt_set_type.__new__(cmpnt_set_type)
        CmpntSet.__init__(self._cmpnt_set, self._impl._cmpnts._impl)
        # data and set should point at the same cmpnts
        self._impl._cmpnts._impl = self._cmpnt_set._impl
        self._working_term = terms.new_clear_term()
        self.insert_iterable(terms)

    @property
    def _data(self) -> TermData[ImplT, SpecT, CoeffT]:
        """Get the raw term data object underlying :param:`self`."""
        return self._impl

    @property
    def coeffs_type(self) -> type[Coeffs[CoeffT]]:
        """Get the coefficient container type of :param:`self`."""
        return self._impl.coeffs_type

    @property
    def coeff_type(self) -> type[CoeffT]:
        """Get the coefficient type of :param:`self`."""
        return self._data.coeff_type

    @property
    def cmpnts_type(self) -> type[Cmpnts[ImplT, SpecT]]:
        """Get the component container type of :param:`self`."""
        return self._impl.cmpnts_type

    @property
    def cmpnt_type(self) -> type[Cmpnt[ImplT, SpecT]]:
        """Get the component type of :param:`self`."""
        return self._data.cmpnt_type

    @classmethod
    def _create(cls, data: TermData[ImplT, SpecT, CoeffT]) -> Self:
        """Create a new instance of :param:`cls`.

        Args:
            data: Raw term data object.

        Returns:
            A new instance of :param:`cls`.
        """
        out = cls.__new__(cls)
        TermSet.__init__(out, cls.terms_type._create(data))
        return out

    def _empty_clone(self) -> Self:
        """Get an empty (owning, contiguous) clone of :param:`self`."""
        return self._create(
            TermData(self._impl._cmpnts._empty_clone(), self._impl._coeffs._empty_clone())
        )

    def clone(self) -> Self:
        """Return a deep copy of :param:`self`."""
        out = self._empty_clone()
        out.insert_iterable(self)
        return out

    def to_terms(self) -> Terms[ImplT, SpecT, CoeffT]:
        """Get a collection of terms containing the same data as :param:`self`."""
        return self.terms_type._create(self._impl.clone())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert :param:`self` to a :class:`~pandas.DataFrame`."""
        return self.to_terms().to_dataframe()

    def into(self, t: type[TermSet[Any, Any, Any]]) -> TermSet[Any, Any, Any]:
        """Clone :param:`self` into a new set of terms with type :param:`t`.

        Args:
            t: Type to return, must have the same
                :attr:`~zixy.container.terms.TermSet.cmpnt_type` as :param:`self`.

        Returns:
            A new instance of :param:`t`.
        """
        terms = self.to_terms().into(t.terms_type)
        out = t._create(terms._impl)
        if out.cmpnt_type is not self.cmpnt_type:
            raise TypeError(
                f"Cannot convert a term with component type "
                f"{self.cmpnt_type} into one with a different component "
                f"type {out.cmpnt_type}."
            )
        return out

    def __repr__(self) -> str:
        """Return a string representation of :param:`self`."""
        return ", ".join(str(s) for s in self)

    def __len__(self) -> int:
        """Get the number of elements in :param:`self`."""
        return len(self._impl)

    def _get_working_term(
        self, value: Term[ImplT, SpecT, CoeffT] | TermSpecT[ImplT, SpecT, CoeffT] = None
    ) -> Term[ImplT, SpecT, CoeffT]:
        """Get a term that contains the data specified by :param:`value`.

        Args:
            value: Term or term specifier required.

        Returns:
            If :param:`value` is an instance of :class:`Term`, return it. Otherwise, set the
            internally-allocated working term to the data specified by :param:`value` and return
            that working term.
        """
        if isinstance(value, Term):
            return value
        else:
            self._working_term.set(value)
            return self._working_term

    def _get_working_cmpnt(self, value: Cmpnt[ImplT, SpecT] | SpecT | None) -> Cmpnt[ImplT, SpecT]:
        """Get a component that contains the data specified by :param:`value`.

        Args:
            value: Component or component specifier required.

        Returns:
            If :param:`value` is an instance of :class:`~zixy.container.cmpnts.Cmpnt`, return
            it. Otherwise, set the component of the internally-allocated working term to the data
            specified by :param:`value` and return that component.
        """
        cmpnt_type = self._impl.cmpnt_type
        assert issubclass(cmpnt_type, Cmpnt)
        if isinstance(value, cmpnt_type):
            return value
        else:
            self._working_term.cmpnt.set(value)
            return self._working_term.cmpnt

    def insert(self, key: Term[ImplT, SpecT, CoeffT] | TermSpecT[ImplT, SpecT, CoeffT]) -> int:
        """Try to insert the given term.

        Args:
            key: The term specifier.

        Returns:
            The index at which the term was inserted, or the index at which it already was
            stored if insertion is unsuccessful.

        Note:
            This method operates in-place.
        """
        term = self._get_working_term(key)
        index = self._cmpnt_set.insert(term.cmpnt)
        self._impl._coeffs.resize(len(self._impl._cmpnts))
        self._impl._coeffs[index] = term.coeff
        return index

    def soft_insert(
        self, key: Term[ImplT, SpecT, CoeffT] | TermSpecT[ImplT, SpecT, CoeffT]
    ) -> tuple[int, bool]:
        """Insertion method which does not overwrite coefficient values.

        Operates similarly to :meth:`insert`, but does not overwrite coefficient values if the
        component specified by :param:`key` is already present in :param:`self`.

        Args:
            key: The term specifier.

        Returns:
            Index at which the term was inserted or found, and a boolean indicating whether
            insertion was successful (i.e. whether the component specified by :param:`key` was not
            already present in :param:`self`).

        Note:
            This method operates in-place.
        """
        term = self._get_working_term(key)
        index = self.lookup_index(term)
        if index is None:
            return self.insert(term), True
        return index, False

    def insert_iterable(
        self,
        source: Iterable[Term[ImplT, SpecT, CoeffT] | TermSpecT[ImplT, SpecT, CoeffT]] = tuple(),
    ) -> None:
        """Insert many terms from an iterable source.

        Args:
            source: Iterable over any term specifiers.

        Note:
            This method operates in-place.
        """
        for item in source:
            self.insert(item)

    def lookup_index(
        self, value: SpecT | Cmpnt[ImplT, SpecT] | Term[ImplT, SpecT, CoeffT]
    ) -> int | None:
        """Try to find the index of the component specified by :param:`value` in :param:`self`.

        Args:
            value: The component or term specifier.

        Returns:
            The index of the component specified by :param:`value` if it is present in
            :param:`self`, and ``None`` otherwise.
        """
        value = self._get_working_term(value)
        return self._cmpnt_set.lookup(value.cmpnt)

    def lookup_coeff(
        self, value: SpecT | Cmpnt[ImplT, SpecT] | Term[ImplT, SpecT, CoeffT]
    ) -> CoeffT | None:
        """Try to find the coefficient of the term specified by :param:`value` in :param:`self`.

        Args:
            value: The component or term specifier.

        Returns:
            The coefficient of the term specified by :param:`value` if it is present in
            :param:`self`, and ``None`` otherwise.
        """
        index = self.lookup_index(value)
        if index is None:
            return None
        return self._impl._coeffs[index]

    def lookup(
        self, value: SpecT | Cmpnt[ImplT, SpecT] | Term[ImplT, SpecT, CoeffT]
    ) -> tuple[int, CoeffT] | None:
        """Try to find the index and coefficient of the component specified by :param:`value`.

        Args:
            value: The component or term specifier.

        Returns:
            The index at which the value was inserted, or ``None`` if the value was not found.
        """
        index = self.lookup_index(value)
        if index is None:
            return None
        return index, self._impl._coeffs[index]

    def contains(self, value: SpecT | Cmpnt[ImplT, SpecT] | Term[ImplT, SpecT, CoeffT]) -> bool:
        """Check whether the component specified by :param:`value` is stored in :param:`self`.

        Args:
            value: The component specifier.

        Returns:
            Whether the lookup of :param:`value` was successful.
        """
        return self.lookup(value) is not None

    def remove(self, value: SpecT | Cmpnt[ImplT, SpecT] | Term[ImplT, SpecT, CoeffT]) -> int:
        """Try to remove the component specified by :param:`value` from :param:`self`.

        If the component is found, removal proceeds via swap-remove.

        Args:
            value: The component specifier.

        Returns:
            The index at which the component was removed.

        Raises:
            KeyError: The component was not found.
        """
        value = self._get_working_term(value)
        i = self._cmpnt_set.remove(value.cmpnt)
        if i is None:
            raise KeyError(f"Value {value} not found in {self.__class__}.")
        self._impl._coeffs.swap_remove(i)
        return i

    def __getitem__(
        self, value: SpecT | Cmpnt[ImplT, SpecT] | Term[ImplT, SpecT, CoeffT]
    ) -> CoeffT:
        """Get the coefficient of the term with component specified by :param:`value`.

        Args:
            value: The component or term specifier.

        Returns:
            The coefficient of the term with component specified by :param:`value`.

        Raises:
            KeyError: The component specified by :param:`value` was not found in :param:`self`.
        """
        out = self.lookup_coeff(value)
        if out is None:
            raise KeyError(f"Value {value} not found in TermSet.")
        return out

    def __setitem__(self, key: SpecT | Term[ImplT, SpecT, CoeffT], coeff: CoeffT) -> None:
        """Set the coefficient of the term with component specified by :param:`key`."""
        self._impl._coeffs[self.insert(key)] = coeff

    def __eq__(self, other: object) -> bool:
        """Return whether :param:`self` and :param:`other` are equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self._cmpnt_set != other._cmpnt_set:
            return False
        return all(self.lookup_coeff(term.cmpnt) == term.coeff for term in other)

    def __iter__(self) -> Iterator[Term[ImplT, SpecT, CoeffT]]:
        """Iterate over the elements of :param:`self`."""
        if not len(self):
            return
        tmp = self._get_working_term().clone()
        for i in range(len(self)):
            tmp.coeff = self._impl._coeffs[i]
            tmp._impl._cmpnts._impl.cmpnt_copy_external(0, self._impl._cmpnts._impl, i)
            yield tmp

    def _from_generator(self, gen: Iterator[Term[ImplT, SpecT, CoeffT]]) -> Self:
        """Creates a new instance from a generator yielding instances of :class:`Term`."""
        out = self._empty_clone()
        out.insert_iterable(gen)
        return out

    def iter_filter_map(
        self, f: Callable[[Term[ImplT, SpecT, CoeffT]], bool]
    ) -> Iterator[Term[ImplT, SpecT, CoeffT]]:
        """Lazily evaluate a filter-map operation over the terms of :param:`self`.

        Args:
            f: Function which may mutate copies of the terms of :param:`self`, returning ``True`` if
                those mutated copies are to be included in the generator. The function signature
                should take a single :class:`Term` instance as an argument, and return a boolean.

        Returns:
            Iterator over the selected (and possibly mutated) components of :param:`self` according
            to :param:`f`.

        Note:
            The resulting generator does not enforce uniqueness of components.
        """
        return (term for term in self if f(term))

    def filter_map(self, f: Callable[[Term[ImplT, SpecT, CoeffT]], bool]) -> Self:
        """Eagerly evaluate a filter-map operation over the terms of :param:`self`.

        Args:
            f: Function which may mutate copies of the terms of :param:`self`, returning ``True``
                if those mutated copies are to be included in the generator. The function signature
                should take a single :class:`Term` instance as an argument, and return a boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) components
            of :param:`self` according to :param:`f`.

        Note:
            The resulting generator enforces uniqueness of components by overwriting coefficients of
            duplicate components.
        """
        return self._from_generator(self.iter_filter_map(f))

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[Term[ImplT, SpecT, CoeffT] | TermSpecT[ImplT, SpecT, CoeffT]],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of :param:`cls` from an iterable.

        Args:
            iterable: Iterable returning specifiers of all the terms to be appended.
            args: Positional arguments to forward to the constructor of :param:`cls`.
            kwargs: Keyword arguments to forward to the constructor of :param:`cls`.

        Returns:
            New instance of :param:`cls` containing the terms specified by :param:`iterable`.
        """
        out = cls(*args, **kwargs)
        out.insert_iterable(iterable)
        return out

    def iter_filter_nonzero(self) -> Iterator[Term[ImplT, SpecT, CoeffT]]:
        """Iterate over the non-zero terms of :param:`self`."""
        return (term for term in self if term.coeff != 0)

    def filter_nonzero(self) -> Self:
        """Filter :param:`self` to only the non-zero terms."""
        return self._from_generator(self.iter_filter_nonzero())


class TermSum(TermSet[ImplT, SpecT, CoeffT]):
    """A sum of terms consisting of components and coefficients.

    A set-like container of terms that may be used to store unique terms and perform algebraic
    operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    def _scaled_iadd(self, rhs: Term[ImplT, SpecT, CoeffT] | Self, scalar: Coeff) -> None:
        """Add :param:`scalar` times :param:`rhs` to :param:`self` in-place."""
        if not isinstance(rhs, Term | TermSum):
            rhs = self._get_working_term(rhs)
        if self.coeff_type != rhs.coeff_type:
            raise TypeError(
                f"Cannot add a term with coefficient type {rhs.coeff_type} "
                f"to a TermSum with coefficient type {self.coeff_type}."
            )
        assert not issubclass(self.coeff_type, RootOfUnity)  # TODO: reflect in typing
        if isinstance(rhs, Term):
            index, inserted = self.soft_insert(rhs)
            if not inserted:
                self._impl._coeffs[index] += rhs.coeff * convert(scalar, self.coeff_type)
        else:
            # todo: delegate rust
            for term in rhs:
                self._scaled_iadd(term, scalar)
            TermSet.__init__(self, self.filter_nonzero().to_terms())

    def __iadd__(self, rhs: Term[ImplT, SpecT, CoeffT] | Self) -> Self:
        """Add :param:`rhs` to :param:`self` in-place."""
        self._scaled_iadd(rhs, 1)
        return self

    def __isub__(self, rhs: Term[ImplT, SpecT, CoeffT] | Self) -> Self:
        """Subtract :param:`rhs` from :param:`self` in-place."""
        self._scaled_iadd(rhs, -1)
        return self

    def __add__(self, rhs: Term[ImplT, SpecT, CoeffT] | Self) -> Self:
        """Addition of :param:`self` and :param:`rhs`."""
        out = self.clone()
        out += rhs
        return out

    def __sub__(self, rhs: Term[ImplT, SpecT, CoeffT] | Self) -> Self:
        """Subtraction of :param:`rhs` from :param:`self`."""
        out = self.clone()
        out -= rhs
        return out

    def __imul__(self, scalar: Coeff) -> Self:
        """In-place multiplication of :param:`self` by :param:`scalar`."""
        self._impl._coeffs.scale(scalar)
        return self

    def __itruediv__(self, scalar: Coeff) -> Self:
        """In-place division of :param:`self` by :param:`scalar`."""
        if isinstance(scalar, RootOfUnity):
            self *= 1 / scalar.to_numeric()
        else:
            self *= 1 / scalar
        return self

    def __mul__(self, scalar: Coeff) -> Self:
        """Multiplication of :param:`self` by :param:`scalar`."""
        out = self.clone()
        out *= scalar
        return out

    def __rmul__(self, scalar: Coeff) -> Self:
        """Multiplication of :param:`scalar` by :param:`self`."""
        return self * scalar

    def __truediv__(self, scalar: Coeff) -> Self:
        """Division of :param:`self` by :param:`scalar`."""
        factor = 1 / scalar if not isinstance(scalar, RootOfUnity) else 1 / scalar.to_numeric()
        return self * factor

    def add_iterable(self, iterable: Iterable[Term[ImplT, SpecT, CoeffT]]) -> None:
        """In-place addition of the terms in :param:`iterable` to :param:`self`.

        Args:
            iterable: Iterable of terms to add to :param:`self`.

        Note:
            This method operates in-place.
        """
        for item in iterable:
            self += item

    def _from_generator(self, gen: Iterator[Term[ImplT, SpecT, CoeffT]]) -> Self:
        """Create a new instance based on :param:`self` with contents given by a generator.

        Args:
            gen: Generator of terms to be summed into the new instance.

        Returns:
            New instance with contents given by :param:`gen`.
        """
        out = self._empty_clone()
        out.add_iterable(gen)
        return out

    # TODO: solve override with base class
    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[Term[ImplT, SpecT, CoeffT]],  # type: ignore[override]
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of :param:`cls` from an iterable.

        Args:
            iterable: Iterable returning specifiers of all the terms to be appended.
            args: Positional arguments to forward to the constructor of :param:`cls`.
            kwargs: Keyword arguments to forward to the constructor of :param:`cls`.

        Returns:
            New instance of :param:`cls` containing the terms specified by :param:`iterable`.
        """
        out = cls(*args, **kwargs)
        out.add_iterable(iterable)
        return out


class NumericTermSum(TermSum[ImplT, SpecT, NumberT]):
    """A sum of terms consisting of components and numeric coefficients.

    A set-like container of terms that may be used to store unique terms and perform algebraic
    operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    @property
    def l1_norm(self) -> NumberT:
        """Get the L1 norm of :param:`self`."""
        result: NumberT = zero(self.coeff_type)
        for coeff in self._impl._coeffs:
            result += self.coeff_type(abs(coeff))
        return result

    @property
    def l2_norm_square(self) -> NumberT:
        """Get the square of the L2 norm of :param:`self`."""
        result: NumberT = zero(self.coeff_type)
        for coeff in self._impl._coeffs:
            result += self.coeff_type(abs(coeff)) ** 2
        return result

    @property
    def l2_norm(self) -> Number:
        """Get the L2 norm of :param:`self`."""
        return cast(Number, self.l2_norm_square**0.5)

    def l1_normalize(self) -> None:
        """Divide :param:`self` by its L1 norm.

        Note:
            This method operates in-place.
        """
        self._impl._coeffs.scale(1 / self.l1_norm)

    def l2_normalize(self) -> None:
        """Divide :param:`self` by its L2 norm.

        Note:
            This method operates in-place.
        """
        self._impl._coeffs.scale(1 / self.l2_norm)

    def iter_filter_significant(
        self, atol: float = DEFAULT_ATOL
    ) -> Iterator[Term[ImplT, SpecT, NumberT]]:
        """Lazily generate terms in :param:`self` with coefficients no less than :param:`atol`.

        Args:
            atol: Absolute tolerance.

        Returns:
            Iterator of terms that meet the criterion.
        """
        return (term for term in self if not np.isclose(term.coeff, 0, atol=atol))

    def filter_significant(self, atol: float = DEFAULT_ATOL) -> Self:
        """Eagerly get terms in :param:`self` with coefficients no less than :param:`atol`.

        Args:
            atol: Absolute tolerance.

        Returns:
            New instance containing all terms that meet the criterion.
        """
        return self._from_generator(self.iter_filter_significant(atol))

    def iter_filter_insignificant(
        self, atol: float = DEFAULT_ATOL
    ) -> Iterator[Term[ImplT, SpecT, NumberT]]:
        """Lazily generate terms in :param:`self` with coefficients less than :param:`atol`.

        Args:
            atol: Absolute tolerance.

        Returns:
            Iterator of terms that meet the criterion.
        """
        return (term for term in self if np.isclose(term.coeff, 0, atol=atol))

    def filter_insignificant(self, atol: float = DEFAULT_ATOL) -> Self:
        """Eagerly get terms in :param:`self` with coefficients less than :param:`atol`.

        Args:
            atol: Absolute tolerance.

        Returns:
            New instance containing all terms that meet the criterion.
        """
        return self._from_generator(self.iter_filter_insignificant(atol))


@dataclass
class TermRegistry(Generic[ImplT, SpecT]):
    """Registry of term types for each different coefficient type."""

    term_type_sign: type[Term[ImplT, SpecT, Sign]]
    term_type_complex_sign: type[Term[ImplT, SpecT, ComplexSign]]
    term_type_real: type[Term[ImplT, SpecT, float]]
    term_type_complex: type[Term[ImplT, SpecT, complex]]
    term_type_symbolic: type[Term[ImplT, SpecT, Expr]]

    def __getitem__(self, coeff_type: type[CoeffT]) -> type[Term[ImplT, SpecT, CoeffT]]:
        """Get the term type corresponding to :param:`coeff_type`."""
        if _is_int(coeff_type) or _is_float(coeff_type):
            return self.term_type_real
        elif _is_complex(coeff_type):
            return self.term_type_complex
        elif _is_sign(coeff_type):
            return self.term_type_sign
        elif _is_complex_sign(coeff_type):
            return self.term_type_complex_sign
        elif _is_expr(coeff_type):
            return self.term_type_symbolic
        else:
            raise TypeError(f"Unsupported coefficient type {coeff_type} for term registry lookup.")
