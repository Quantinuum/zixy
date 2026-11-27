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

"""Term sets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast, overload

import pandas as pd  # noqa: TC002 (runtime type hints)
from sympy import Expr, simplify, sympify

from zixy.container.base import Representable, ViewableBase
from zixy.container.coeff import ComplexSign, Sign
from zixy.container.coeffs import Coeffs, ComplexCoeffs, RealCoeffs, convert_vec
from zixy.container.data import TermData
from zixy.container.serialization import SerializableContainer
from zixy.container.string import String
from zixy.container.strings import Strings
from zixy.container.term import Term
from zixy.utils import classproperty

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from zixy._zixy import ImplArray, Map
    from zixy.container.base import (
        _HasStringsType,
        _HasStringType,
        _HasTermsType,
        _HasTermType,
        _TermSetCoeffs,
    )
    from zixy.container.data import _TermData
    from zixy.container.serialization import SerializedContainer
    from zixy.container.string_set import StringSet
    from zixy.container.term import TermSpecT, _AnyCoeffTerm
    from zixy.container.terms import Terms


class TermSet[ImplT: ImplArray, SpecT, C: (float, complex, Sign, ComplexSign, Expr)](
    Representable, SerializableContainer
):
    """A collection of unique terms consisting of strings and coefficients.

    A set-like container of terms that may be used to store unique terms and perform set-like
    operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[ImplT, SpecT, C]]

    _impl: _TermData[ImplT, SpecT, C]
    _string_set: StringSet[ImplT, SpecT]
    _working_term: Term[ImplT, SpecT, C]

    def __init__(self, terms: Terms[ImplT, SpecT, C]):
        """Initialize the term set.

        Args:
            terms: Terms-derived object from which to construct the set of strings and their
                coefficients.

        Note:
            Strings which appear multiple times in ``terms`` will appear only once in
                ``self``.
        """
        if type(terms._data.strings) is not self.terms_type.term_type.strings_type:
            raise TypeError("String storage does not match its factory")
        if not isinstance(terms._data.coeffs, self.terms_type.term_type.coeffs_type):
            raise TypeError("Coefficient storage does not match its factory")
        self._bind(terms._new()._impl, terms.strings_type._set_type().map_type())
        if isinstance(terms._impl.coeffs, RealCoeffs | ComplexCoeffs):
            self._collect_terms(terms, sum_duplicates=False)
        else:
            self.insert_iterable(terms)

    def _bind(self, data: _TermData[ImplT, SpecT, C], map_: Map) -> None:
        """Set the term data and lookup map for ``self``.

        Args:
            data: Term data object containing the data for this instance.
            map_: Populated string lookup map corresponding to the supplied data.

        Raises:
            TypeError: The storage does not match the concrete container factories.
        """
        if type(data.strings) is not self.terms_type.term_type.strings_type:
            raise TypeError("String storage does not match its factory")
        if not isinstance(data.coeffs, self.terms_type.term_type.coeffs_type):
            raise TypeError("Coefficient storage does not match its factory")
        assert len(data) == len(map_)
        self._impl = data
        string_set_type = data.strings._set_type()
        self._string_set = string_set_type._create(data.strings._impl, map_=map_)
        self._working_term = self.terms_type._view(data, slice(None)).new_term()

    @classmethod
    def _create(cls, data: _TermData[ImplT, SpecT, C], *, map_: Map) -> Self:
        """Create an owning instance of ``cls``, wrapping existing storage.

        Args:
            data: Term data object containing the data for this instance.
            map_: Populated lookup map corresponding to the supplied data.

        Returns:
            A new owning instance of ``cls`` using the supplied storage without copying it.
        """
        out = cls.__new__(cls)
        out._bind(data, map_)
        return out

    def _new(self) -> Self:
        """Create an empty instance with the same type and configuration."""
        term_data = TermData(self._impl.strings._new(), self._impl.coeffs._new())
        return self._create(term_data, map_=self._string_set.map_type())

    @classmethod
    def from_terms(cls, terms: Terms[ImplT, SpecT, C]) -> Self:
        """Create a new instance of ``cls`` from ``terms``.

        Args:
            terms: Terms-derived object from which to construct the set of terms.

        Returns:
            A new instance of ``cls`` containing the terms in ``terms``.
        """
        out = cls.__new__(cls)
        TermSet.__init__(out, terms)
        return out

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
        out.insert_iterable(source)
        return out

    def _from_generator(self, gen: Iterator[Term[ImplT, SpecT, C]]) -> Self:
        """Creates a new instance from a generator yielding instances of :class:`Term`."""
        out = self._new()
        out.insert_iterable(gen)
        return out

    def _collect_terms(self, terms: Terms[ImplT, SpecT, C], sum_duplicates: bool) -> None:
        """Collect numeric terms in Rust, normalizing sliced inputs first."""
        if (
            not terms.is_owning()
            or not terms._impl.strings.is_owning()
            or not terms._impl.coeffs.is_owning()
        ):
            terms = terms.clone()
        source = terms._impl.strings._impl
        coeffs = terms._impl.coeffs
        out_coeffs = self._impl.coeffs
        if isinstance(coeffs, RealCoeffs):
            assert isinstance(out_coeffs, RealCoeffs)
            array, mapping, values = source.collect_terms_real(coeffs._impl, sum_duplicates)
            out_coeffs._impl = values
        else:
            assert isinstance(coeffs, ComplexCoeffs)
            assert isinstance(out_coeffs, ComplexCoeffs)
            array, mapping, complex_values = source.collect_terms_complex(
                coeffs._impl, sum_duplicates
            )
            out_coeffs._impl = complex_values
        self._impl.strings._impl = array
        self._bind(self._impl, mapping)

    def clone(self) -> Self:
        """Return a deep copy of ``self``."""
        if isinstance(self._impl.coeffs, RealCoeffs | ComplexCoeffs):
            return self._copy_terms(nonzero=False)
        out = self._new()
        out.insert_iterable(self)
        return out

    def _copy_terms(self, nonzero: bool) -> Self:
        """Copy numeric terms in Rust, optionally omitting exact zeros."""
        out = self._new()
        coeffs = self._impl.coeffs
        out_coeffs = out._impl.coeffs
        if isinstance(coeffs, RealCoeffs):
            assert isinstance(out_coeffs, RealCoeffs)
            array, mapping, values = self._impl.strings._impl.copy_terms_real(coeffs._impl, nonzero)
            out_coeffs._impl = values
        else:
            assert isinstance(coeffs, ComplexCoeffs)
            assert isinstance(out_coeffs, ComplexCoeffs)
            array, mapping, complex_values = self._impl.strings._impl.copy_terms_complex(
                coeffs._impl, nonzero
            )
            out_coeffs._impl = complex_values
        out._impl.strings._impl = array
        out._bind(out._impl, mapping)
        return out

    @classproperty
    def coeff_type(cls: type[TermSet[ImplT, SpecT, C]]) -> type[C]:
        """The scalar type supplied by the term sequence class."""
        return cls.terms_type.coeff_type

    @property
    def coeffs_type[Cs: Coeffs[Any]](
        self: _TermSetCoeffs[Cs],
    ) -> type[Cs]:
        """The concrete coefficient container type of ``self``."""
        return cast("type[Cs]", type(self._data.coeffs))

    @property
    def string_type[StringT: String[Any, Any]](
        self: _HasTermsType[_HasTermType[_HasStringsType[_HasStringType[StringT]]]],
    ) -> type[StringT]:
        """The concrete string type of ``self``."""
        terms = cast("TermSet[Any, Any, Any]", self)
        return cast("type[StringT]", terms._data.strings.string_type)

    @property
    def strings_type[StringsT: Strings[Any, Any]](
        self: _HasTermsType[_HasTermType[_HasStringsType[StringsT]]],
    ) -> type[StringsT]:
        """The concrete string container type of ``self``."""
        terms = cast("TermSet[Any, Any, Any]", self)
        return cast("type[StringsT]", type(terms._impl.strings))

    @property
    def _data(self) -> _TermData[ImplT, SpecT, C]:
        """The raw term data object underlying ``self``."""
        return self._impl

    @property
    def strings[StringsT: Strings[Any, Any]](
        self: _HasTermsType[_HasTermType[_HasStringsType[StringsT]]],
    ) -> StringsT:
        """An independent copy of the keys in iteration order."""
        terms = cast("TermSet[Any, Any, Any]", self)
        return cast("StringsT", terms._impl.strings.clone())

    @property
    def coeffs[Cs: Coeffs[Any]](
        self: _TermSetCoeffs[Cs],
    ) -> Cs:
        """A fixed-length view supporting write-through coefficient updates."""
        return cast("Cs", self._data.coeffs.as_view())

    def __len__(self) -> int:
        """Get the number of elements in ``self``."""
        return len(self._impl)

    def __iter__[TermT: Term[Any, Any, Any]](
        self: _HasTermsType[_HasTermType[TermT]],
    ) -> Iterator[TermT]:
        """Yield an independent, owning copy of each term."""
        terms = cast("TermSet[Any, Any, Any]", self)
        for i in range(len(terms)):
            yield cast("TermT", terms.terms_type.term_type._create(terms._impl.clone(i)))

    def __getitem__(self, key: SpecT | String[ImplT, SpecT] | _AnyCoeffTerm[ImplT, SpecT]) -> C:
        """Get the coefficient of the term with string specified by ``key``.

        Args:
            key: String, string specifier, or compatible term. A term's coefficient
                is ignored and is not converted to this collection's coefficient type.

        Returns:
            The coefficient of the term with string specified by ``key``.

        Raises:
            KeyError: The string specified by ``key`` was not found in ``self``.
            TypeError: The key is unsupported or belongs to an incompatible family.
            ValueError: The key is invalid or belongs to an incompatible space.
        """
        out = self.get(key)
        if out is None:
            raise KeyError(f"Value {key} not found in TermSet.")
        return out

    def __setitem__(self, key: SpecT | Term[ImplT, SpecT, C], coeff: C) -> None:
        """Set the coefficient of the term with string specified by ``key``."""
        self._impl.coeffs[self.insert(key)] = coeff

    def __contains__(self, value: object) -> bool:
        """Check whether a string key or complete term is stored in ``self``.

        Args:
            value: String, string specifier, or term to look up. Strings and string
                specifiers test key membership; terms also require an equal coefficient.

        Returns:
            Whether the key or complete term is present. Unsupported objects, out-of-range
            string specifiers, and objects from incompatible families or spaces return
            ``False``. A missing key is not a member, including when querying a
            zero-coefficient term.
        """
        if not isinstance(value, Term):
            return value in self._string_set
        try:
            index = self._string_set.lookup_index(value.string)
        except (TypeError, ValueError, IndexError):
            return False
        return index is not None and self.terms_type.term_type._view(self._impl, index) == value

    def lookup_index(
        self, key: SpecT | String[ImplT, SpecT] | _AnyCoeffTerm[ImplT, SpecT]
    ) -> int | None:
        """Try to find the storage index of the string specified by ``key``.

        Args:
            key: String, string specifier, or compatible term. A term's coefficient
                is ignored and is not converted to this collection's coefficient type.

        Returns:
            The current storage index, or ``None`` if the string is absent. Indices
            may change when terms are removed.

        Raises:
            TypeError: The key is unsupported or belongs to an incompatible family.
            ValueError: The key is invalid or belongs to an incompatible space.
        """
        if isinstance(key, Term):
            key = key.string
        string = self._get_working_string(key)
        return self._string_set.lookup_index(string)

    @overload
    def get(
        self,
        key: SpecT | String[ImplT, SpecT] | _AnyCoeffTerm[ImplT, SpecT],
        default: None = None,
    ) -> C | None: ...

    @overload
    def get[T](
        self,
        key: SpecT | String[ImplT, SpecT] | _AnyCoeffTerm[ImplT, SpecT],
        default: T,
    ) -> C | T: ...

    def get[T](
        self,
        key: SpecT | String[ImplT, SpecT] | _AnyCoeffTerm[ImplT, SpecT],
        default: T | None = None,
    ) -> C | T | None:
        """Get the stored coefficient for ``key``, or ``default`` if it is absent.

        Args:
            key: String, string specifier, or compatible term. A term's coefficient
                is ignored and is not converted to this collection's coefficient type.
            default: Value returned for a missing key, without coefficient conversion.

        Returns:
            The stored coefficient if the key is present, including a stored zero,
            or the supplied default otherwise.

        Raises:
            TypeError: The key is unsupported or belongs to an incompatible family.
            ValueError: The key is invalid or belongs to an incompatible space.
        """
        index = self.lookup_index(key)
        if index is None:
            return default
        return self._impl.coeffs[index]

    def _check_string(self, value: String[ImplT, SpecT]) -> None:
        """Hook to check whether a string is valid for insertion into ``self``."""
        pass

    def _check_term(self, value: Term[ImplT, SpecT, C]) -> None:
        """Hook to check whether a term is valid for insertion into ``self``."""
        pass

    def _get_working_string(self, value: String[ImplT, SpecT] | SpecT) -> String[ImplT, SpecT]:
        """Get a string that contains the data specified by ``value``.

        Args:
            value: String or string specifier required.

        Returns:
            If ``value`` is an instance of :class:`~zixy.container.string.String`, return
            it. Otherwise, set the string of the internally-allocated working term to the data
            specified by ``value`` and return that string.
        """
        string_type = self._impl.strings.string_type
        assert issubclass(string_type, String)
        if isinstance(value, string_type):
            string = value
        else:
            self._working_term.string.set(value)
            string = self._working_term.string
        self._check_string(string)
        return string

    def _get_working_term(self, value: TermSpecT[ImplT, SpecT, C]) -> Term[ImplT, SpecT, C]:
        """Get a term that contains the data specified by ``value``.

        Args:
            value: Term or term specifier required.

        Returns:
            If ``value`` already has the destination term type, return it. Otherwise, copy or
            parse it into the working term, applying checked coefficient conversion.
        """
        if isinstance(value, self.terms_type.term_type):
            term = value
        else:
            self._working_term.set(value)
            term = self._working_term
        self._check_term(term)
        return term

    def insert(self, key: TermSpecT[ImplT, SpecT, C]) -> int:
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
        index = self._string_set.insert(term.string)
        self._impl.coeffs.resize(len(self._impl.strings))
        self._impl.coeffs[index] = term.coeff
        return index

    def soft_insert(self, key: TermSpecT[ImplT, SpecT, C]) -> tuple[int, bool]:
        """Insertion method which does not overwrite coefficient values.

        Operates similarly to :meth:`insert`, but does not overwrite coefficient values if the
        string specified by ``key`` is already present in ``self``.

        Args:
            key: The term specifier.

        Returns:
            Index at which the term was inserted or found, and a boolean indicating whether
            insertion was successful (i.e. whether the string specified by ``key`` was not
            already present in ``self``).

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
        source: Iterable[TermSpecT[ImplT, SpecT, C]] = tuple(),
    ) -> None:
        """Insert many terms from an iterable source.

        Args:
            source: Iterable over any term specifiers.

        Note:
            This method operates in-place.
        """
        for item in source:
            self.insert(item)

    def remove(self, value: SpecT | String[ImplT, SpecT] | Term[ImplT, SpecT, C]) -> int:
        """Try to remove the string specified by ``value`` from ``self``.

        If the string is found, removal proceeds via swap-remove.

        Args:
            value: The string specifier.

        Returns:
            The index at which the string was removed.

        Raises:
            KeyError: The string was not found.
        """
        value = self._get_working_term(value)
        i = self._string_set.remove(value.string)
        if i is None:
            raise KeyError(f"Value {value} not found in {self.__class__}.")
        self._impl.coeffs.swap_remove(i)
        return i

    def __eq__(self, other: object) -> bool:
        """Return whether ``self`` and ``other`` are equal."""
        if not isinstance(other, type(self)):
            return NotImplemented
        if self._string_set != other._string_set:
            return False
        return all(self.get(term.string) == term.coeff for term in other)

    def _symbolic_vdot(self: TermSet[ImplT, SpecT, Expr], rhs: TermSet[ImplT, SpecT, Expr]) -> Expr:
        """Compute a sparse symbolic inner product between compatible term sets."""
        result = sympify(0)
        for term in rhs:
            lhs_coeff = self.get(term.string)
            if lhs_coeff is not None:
                conjugated = lhs_coeff.conjugate() if hasattr(lhs_coeff, "conjugate") else lhs_coeff
                result += conjugated * term.coeff
        return simplify(result)

    def iter_filter_map(
        self, f: Callable[[Term[ImplT, SpecT, C]], bool]
    ) -> Iterator[Term[ImplT, SpecT, C]]:
        """Lazily evaluate a filter-map operation over the terms of ``self``.

        Args:
            f: Function which may mutate copies of the terms of ``self``, returning ``True`` if
                those mutated copies are to be included in the generator. The function signature
                should take a single :class:`Term` instance as an argument, and return a boolean.

        Returns:
            Iterator over the selected (and possibly mutated) strings of ``self`` according
            to ``f``.

        Note:
            The resulting generator does not enforce uniqueness of strings.
        """
        for term in self:
            if f(term):
                yield term

    def filter_map(self, f: Callable[[Term[ImplT, SpecT, C]], bool]) -> Self:
        """Eagerly evaluate a filter-map operation over the terms of ``self``.

        Args:
            f: Function which may mutate copies of the terms of ``self``, returning ``True``
                if those mutated copies are to be included in the generator. The function signature
                should take a single :class:`Term` instance as an argument, and return a boolean.

        Returns:
            New instance containing the occurrences of selected (and possibly mutated) strings
            of ``self`` according to ``f``.

        Note:
            The resulting generator enforces uniqueness of strings by overwriting coefficients of
            duplicate strings.
        """
        return self._from_generator(self.iter_filter_map(f))

    def iter_filter_nonzero(self) -> Iterator[Term[ImplT, SpecT, C]]:
        """Iterate over the non-zero terms of ``self``."""
        return (term for term in self if term.coeff != 0)

    def filter_nonzero(self) -> Self:
        """Filter ``self`` to only the non-zero terms."""
        if isinstance(self._impl.coeffs, RealCoeffs | ComplexCoeffs):
            return self._copy_terms(nonzero=True)
        out = self._new()
        strings = out._string_set
        coeffs = out._impl.coeffs
        # Strings are already unique: copy them without constructing term views
        # or repeating coefficient conversion and string compatibility checks.
        for index, coeff in enumerate(self._impl.coeffs):
            if coeff != 0:
                strings._impl.mapped_insert(strings._map, self._impl.strings._impl, index)
                coeffs.append(coeff)
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
        terms = cls.terms_type.from_str(source, *args, **kwargs)
        return cls.from_terms(terms)

    def to_dict(self) -> SerializedContainer:
        """Return a JSON-serializable dictionary representation of ``self``.

        Returns:
            An independent dictionary containing the container metadata, full space, and aligned
            lists of unique strings and coefficients, using their serialization formats.

        Note:
            Terms are stored in iteration order, including terms with zero coefficients. The
            string lookup map is not serialized. Term sums use the same representation.
        """
        data = self._data.strings.to_dict()
        data["container"] = type(self).__name__
        data["coeffs"] = self._data.coeffs.to_dict()["coeffs"]
        return data

    @classmethod
    def from_dict(cls, source: SerializedContainer) -> Self:
        """Create an instance of ``cls`` from a JSON-serializable dictionary representation.

        Args:
            source: Dictionary containing the schema version, space, strings, and coefficients.
                Values must be compatible with the factories of ``terms_type``.

        Returns:
            A new instance of ``cls`` owning independent string and coefficient storage and
            a rebuilt string lookup map.

        Raises:
            KeyError: A required field is missing.
            TypeError: A string, coefficient, or space value has an incompatible type.
            ValueError: The schema version or contents are invalid, or the string and
                coefficient counts differ.
            IndexError: A serialized mode index is outside its space.

        Note:
            The family and container tags are not checked. Duplicate strings follow
            ``from_terms`` semantics: term sets retain the last coefficient, while term sums
            add coefficients together.
        """
        return cls.from_terms(cls.terms_type.from_dict(source))

    @overload
    def into[T: TermSet[Any, Any, Any]](self, t: type[T]) -> T: ...
    @overload
    def into[T: ViewableBase[Any, Any]](self, t: type[T]) -> T: ...

    def into[T: ViewableBase[Any, Any], COther: (float, complex, Sign, ComplexSign, Expr)](
        self, t: type[T] | type[TermSet[ImplT, SpecT, COther]]
    ) -> T | TermSet[ImplT, SpecT, COther]:
        """Clone ``self`` into a new related container of type ``t``.

        Args:
            t: Type of the new container to create.

        Returns:
            A new instance of ``t`` containing the same data as ``self``.
        """
        if issubclass(t, TermSet):
            coeffs_type = t.terms_type.term_type.coeffs_type
            strings = self._impl.strings
            coeffs = convert_vec(self._impl.coeffs, coeffs_type)
            term_data = TermData(strings, coeffs)
            return t.from_terms(t.terms_type._create(term_data))
        try:
            return self.to_terms().into(t)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Cannot convert {type(self)} into unsupported target type {t}.") from e

    def to_terms(self) -> Terms[ImplT, SpecT, C]:
        """Get a collection of terms containing the same data as ``self``."""
        return self.terms_type._create(self._impl.clone())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert ``self`` to a :class:`~pandas.DataFrame`."""
        return self.to_terms().to_dataframe()
