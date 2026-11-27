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

"""Term sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast, overload

import pandas as pd
from sympy import Expr

from zixy.container.base import Representable, ViewableSequence, requires_ownership
from zixy.container.coeff import ComplexSign, Sign, normalize_scalar, typesafe_mul
from zixy.container.coeffs import Coeffs
from zixy.container.data import TermData, _TermData
from zixy.container.serialization import SerializableContainer, check_schema_version
from zixy.container.string import String
from zixy.container.strings import Strings
from zixy.container.term import Term
from zixy.container.utils import _coeffs_from_str, _parse_term_str
from zixy.utils import DEFAULT_ATOL, DEFAULT_RTOL, classproperty, slice_index_gen, slice_of_slice

if TYPE_CHECKING:
    import builtins
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from zixy._zixy import ImplArray
    from zixy.container.base import (
        _HasStringsType,
        _HasStringType,
        _HasTermType,
        _TermCoeffsView,
    )
    from zixy.container.serialization import SerializedContainer
    from zixy.container.term import TermSpecT


class Terms[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    ViewableSequence[Term[ImplT, SpecT, C], _TermData[ImplT, SpecT, C]],
    Representable,
    SerializableContainer,
):
    """A collection of terms consisting of strings and coefficients.

    An array-like container of terms consisting of strings and coefficients that may be an
    owning instance referencing a :class:`~zixy.container.data.TermData` instance, or a view on
    a slice of the elements in another collection.
    """

    term_type: type[Term[ImplT, SpecT, C]]

    _impl: _TermData[ImplT, SpecT, C]

    __array_priority__ = 1000

    def __init__(self, data: _TermData[ImplT, SpecT, C], indexer: slice = slice(None)):
        """Initialize the term array.

        Args:
            data: Raw term data object, of which ``self`` views a slice.
            indexer: Slice of the data in ``data`` that ``self`` will view. The default value of
                ``slice(None)`` indicates that this instance is considered to be owning.
        """
        if type(data.strings) is not self.term_type.strings_type:
            raise TypeError("String storage does not match its factory")
        if not isinstance(data.coeffs, self.term_type.coeffs_type):
            raise TypeError("Coefficient storage does not match its factory")
        self._impl = data
        self._slice = indexer

    @classmethod
    def _create(cls, data: _TermData[ImplT, SpecT, C]) -> Self:
        """Create an owning instance of ``cls``, wrapping existing storage.

        Args:
            data: Term data object containing the data for this instance.

        Returns:
            A new owning instance of ``cls`` using the supplied storage without copying it.
        """
        out = cls.__new__(cls)
        Terms.__init__(out, data)
        return out

    @classmethod
    def _view(cls, data: _TermData[ImplT, SpecT, C], indexer: slice) -> Self:
        """Create a mutable view on existing terms.

        Args:
            data: Term data object containing the data for this instance.
            indexer: Slice selecting the viewed terms.

        Returns:
            A non-owning instance of ``cls`` sharing the selected storage without copying it.
        """
        out = cls.__new__(cls)
        if indexer == slice(None):
            indexer = slice(0, len(data), 1)
        Terms.__init__(out, data, indexer)
        return out

    def _new(self) -> Self:
        """Create an empty instance with the same type and configuration."""
        return self._create(self._impl._new())

    @classmethod
    def from_iterable(
        cls,
        source: Iterable[TermSpecT[ImplT, SpecT, C]],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Create a new instance of ``cls`` from an iterable.

        Args:
            source: Iterable returning specifiers of all the terms to be appended.
            args: Positional arguments to forward to the constructor of ``cls``.
            kwargs: Keyword arguments to forward to the constructor of ``cls``.

        Returns:
            New instance of ``cls`` containing the terms specified by ``source``.
        """
        out = cls(*args, **kwargs)
        out.extend(source)
        return out

    def new_term[TermT: Term[Any, Any, Any]](self: _HasTermType[TermT]) -> TermT:
        """Create an independent term with the same space and configuration.

        Returns:
            An owning instance of ``self.term_type`` with independent string and
            coefficient storage.
        """
        terms = cast("Terms[Any, Any, Any]", self)
        data = terms._new().resize(1)
        return self.term_type._create(data._impl)

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        return self._create(self._impl.clone(self.slice))

    @classproperty
    def coeff_type(cls: type[Terms[ImplT, SpecT, C]]) -> type[C]:
        """The scalar type supplied by the term class."""
        return cls.term_type.coeff_type

    @property
    def coeffs_type[Cs: Coeffs[Any]](
        self: _TermCoeffsView[Cs],
    ) -> type[Cs]:
        """The concrete coefficient container type of ``self``."""
        return cast("type[Cs]", type(self._data.coeffs))

    @property
    def string_type[StringT: String[Any, Any]](
        self: _HasTermType[_HasStringsType[_HasStringType[StringT]]],
    ) -> type[StringT]:
        """The concrete string type of ``self``."""
        terms = cast("Terms[Any, Any, Any]", self)
        return cast("type[StringT]", terms._impl.strings.string_type)

    @property
    def strings_type[StringsT: Strings[Any, Any]](
        self: _HasTermType[_HasStringsType[StringsT]],
    ) -> type[StringsT]:
        """The concrete string container type of ``self``."""
        terms = cast("Terms[Any, Any, Any]", self)
        return cast("type[StringsT]", type(terms._data.strings))

    @property
    def _data(self) -> _TermData[ImplT, SpecT, C]:
        """The raw term data object underlying ``self``."""
        return self._impl

    @property
    def strings[StringsT: Strings[Any, Any]](
        self: _HasTermType[_HasStringsType[StringsT]],
    ) -> StringsT:
        """The strings of ``self``, preserving their concrete container type."""
        terms = cast("Terms[Any, Any, Any]", self)
        return cast("StringsT", terms._data.strings[terms.slice])

    @property
    def coeffs[Cs: Coeffs[Any]](
        self: _TermCoeffsView[Cs],
    ) -> Cs:
        """The coefficients of ``self``, preserving their concrete container type."""
        return cast("Cs", self._data.coeffs[self.slice])

    @overload
    def __getitem__[TermT: Term[Any, Any, Any]](
        self: _HasTermType[TermT], indexer: int
    ) -> TermT: ...

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
            if self.map_index(indexer) >= len(self._impl):
                raise IndexError
            return self.term_type._view(self._impl, self.map_index(indexer))
        else:
            return type(self)._view(
                self._data, slice_of_slice(self.slice, indexer, len(self._data))
            )

    @overload
    def __setitem__(
        self,
        indexer: int,
        source: TermSpecT[ImplT, SpecT, C],
    ) -> None: ...

    @overload
    def __setitem__(
        self, indexer: builtins.slice, source: TermSpecT[ImplT, SpecT, C] | Terms[ImplT, SpecT, C]
    ) -> None: ...

    def __setitem__(
        self,
        indexer: int | builtins.slice,
        source: TermSpecT[ImplT, SpecT, C] | Terms[ImplT, SpecT, C],
    ) -> None:
        """Set the term at ``indexer`` in ``self`` to ``source``.

        Args:
            indexer: Index of the string or slice of strings within ``self`` to assign.
            source: Value specifying the term or a view of many strings to assign at
                ``indexer``.
        """
        if isinstance(indexer, slice):
            if isinstance(source, Terms):
                self.strings[indexer] = source.strings
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

    @requires_ownership
    def append(
        self,
        source: TermSpecT[ImplT, SpecT, C] = "",  # type: ignore[assignment]
    ) -> Self:
        """Append ``source`` to the end of ``self``.

        Args:
            source: Value to append.

        Note:
            This method operates in-place.
        """
        return self.append_n(1, source)

    def append_n(
        self,
        n: int,
        source: TermSpecT[ImplT, SpecT, C] = "",  # type: ignore[assignment]
    ) -> Self:
        """Append ``source`` to the end of ``self`` ``n`` times.

        Args:
            n: Number of times to repeatedly append ``source``.
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
    def extend(
        self,
        source: Iterable[TermSpecT[ImplT, SpecT, C]] = tuple(),
    ) -> Self:
        """Append the elements of ``source`` to the end of ``self``.

        Args:
            source: Iterable of compatible terms or term specifiers.

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
        if not isinstance(other, type(self)):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(self[i] == other[i] for i in range(len(self)))

    def __pos__(self) -> Self:
        """Return ``self``."""
        return self.clone()

    def __neg__(self) -> Self:
        """Return the negation of ``self``."""
        out = self.clone()
        out *= cast("C", -1)
        return out

    def __mul__(self, coeff: C | Coeffs[C]) -> Self:
        """Return ``self`` multiplied by a scalar or coefficient vector."""
        coeff = normalize_scalar(coeff)
        out = self.clone()
        out *= coeff
        return out

    def __rmul__(self, coeff: C | Coeffs[C]) -> Self:
        """Return ``self`` multiplied by a scalar or coefficient vector."""
        coeff = normalize_scalar(coeff)
        out = self.clone()
        out *= coeff
        return out

    def __imul__(self, coeff: C | Coeffs[C]) -> Self:
        """Multiply ``self`` in-place by a scalar or coefficient vector."""
        coeff = normalize_scalar(coeff)
        if isinstance(coeff, Coeffs):
            self.scale_by_coeffs(coeff)
        else:
            self.scale(coeff)
        return self

    def scale(self, scalar: C) -> None:
        """Scale all elements by a scalar factor.

        Args:
            scalar: Scalar factor by which to scale all elements.

        Note:
            This method operates in-place.
        """
        scalar = normalize_scalar(scalar)
        for term in self:
            term.coeff = typesafe_mul(term.coeff, scalar)

    def scale_by_coeffs(self, coeffs: Coeffs[C]) -> None:
        """Scale all elements by the corresponding coefficient vector.

        Args:
            coeffs: Coefficients by which to scale the elements of ``self``.

        Note:
            This method operates in-place.
        """
        if len(coeffs) != len(self):
            raise ValueError(
                f"Cannot scale {type(self)} of length {len(self)} by "
                f"{type(coeffs)} of length {len(coeffs)}."
            )

        if type(coeffs) is self.coeffs_type and coeffs._impl.same_as(self._impl.coeffs._impl):
            coeffs = coeffs.clone()

        for item, coeff in zip(self, coeffs):
            item.coeff = typesafe_mul(item.coeff, coeff)

    def iter_filter_map(
        self, f: Callable[[Term[ImplT, SpecT, C]], bool]
    ) -> Iterator[Term[ImplT, SpecT, C]]:
        """Lazily evaluate a filter-map operation over the strings of ``self``.

        Args:
            f: Function which may mutate copies of the terms of ``self``, returning ``True`` if
                those mutated copies are to be included in the generator. The function signature
                should take a single :class:`Term` instance as an argument, and return a boolean.

        Returns:
            Iterator over the selected (and possibly mutated) terms of ``self`` according
            to ``f``.
        """
        for term in self:
            tmp = term.clone()
            if f(tmp):
                yield tmp

    def filter_map(self, f: Callable[[Term[ImplT, SpecT, C]], bool]) -> Self:
        """Eagerly evaluate a filter-map operation over the terms of ``self``.

        Args:
            f: Function which may mutate copies of the terms of ``self``, returning ``True``
                if those mutated copies are to be included in the generator. The function signature
                should take a single :class:`Term` instance as an argument, and return a boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) strings
            of ``self`` according to ``f``.
        """
        out = self._new()
        out.extend(self.iter_filter_map(f))
        return out

    def reordered(self, inds: Sequence[int]) -> Self:
        """Get a new instance with the elements of ``self`` in a new order.

        Args:
            inds: Sequence of indices defining the new order. Should be a permutation of
                ``tuple(range(len(self)))``.

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
    def from_str(cls, source: str, *args: Any, **kwargs: Any) -> Self:
        """Create a new instance of ``cls`` from a string.

        Args:
            source: String to parse.
            *args: Positional arguments to forward to the string constructor.
            **kwargs: Keyword arguments to forward to the string constructor.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        if not source.strip():
            return cls(*args, **kwargs)
        source_coeff, source_strings = _parse_term_str(source)
        strings = cls.term_type.strings_type.from_str(source_strings, *args, **kwargs)
        coeffs = _coeffs_from_str(source_coeff, cls.term_type.coeffs_type, len(strings))
        term_data = TermData(strings, coeffs)
        return cls._create(term_data)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            An independent dictionary containing the container metadata, full space, and aligned
            lists of strings and coefficients, using their respective serialization formats.

        Note:
            For a view, only the selected terms are included, in iteration order. Empty
            collections retain their space. View metadata is not serialized.
        """
        data = self.strings.to_dict()
        data["container"] = type(self).__name__
        data["coeffs"] = self.coeffs.to_dict()["coeffs"]
        return data

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: Dictionary containing the schema version, space, strings, and coefficients.
                String and coefficient values must be compatible with the factories of ``cls``.

        Returns:
            A new owning instance of ``cls`` with independent string and coefficient storage.

        Raises:
            KeyError: A required field is missing.
            TypeError: A string, coefficient, or space value has an incompatible type.
            ValueError: The schema version or contents are invalid, or the string and
                coefficient counts differ.
            IndexError: A serialized mode index is outside its space.

        Note:
            The family and container tags are not checked. The term factory determines the
            decoded string and coefficient types; view metadata is not restored.
        """
        check_schema_version(source)
        strings = cls.term_type.strings_type.from_dict(source)
        coeffs = cls.term_type.coeffs_type.from_dict(source)
        data = TermData(strings, coeffs)
        return cls._create(data)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ``self`` to a :class:`~pandas.DataFrame`."""
        return pd.DataFrame(
            {
                "String": (str(c) for c in self.strings),
                "Coefficient": (str(c) for c in self.coeffs),
            }
        )


class NumericTerms[ImplT: ImplArray, SpecT, C: (float, complex)](Terms[ImplT, SpecT, C]):
    """A collection of terms consisting of strings and numeric coefficients.

    An array-like container of terms consisting of strings and coefficients that may be an
    owning instance referencing a :class:`~zixy.container.data.TermData` instance, or a view on
    a slice of the elements in another collection.
    """

    def allclose(self, other: Self, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> bool:
        """Check whether ``self`` and ``other`` are within a certain tolerance.

        Args:
            other: Other instance to compare to.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Returns:
            Whether ``self`` and ``other`` are within the given tolerances of each other.

        Note:
            This method operates by comparing the NumPy arrays returned by
            :meth:`~zixy.container.coeffs.Coeffs.to_numpy`.
        """
        out: bool = self.strings == other.strings and self.coeffs.allclose(other.coeffs, rtol, atol)
        return out
