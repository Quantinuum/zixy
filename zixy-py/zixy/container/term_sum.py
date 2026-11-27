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

"""Term sums."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
from sympy import Expr

from zixy.container.coeff import Coeff, RootOfUnity, convert, normalize_scalar, unit, zero
from zixy.container.coeffs import Coeffs, ComplexCoeffs, RealCoeffs, SymbolicCoeffs
from zixy.container.term import Term
from zixy.container.term_set import TermSet
from zixy.utils import DEFAULT_ATOL

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from sympy import Symbol

    from zixy._zixy import ImplArray
    from zixy.container.coeff import Number
    from zixy.container.term import TermSpecT
    from zixy.container.terms import Terms


class TermSum[ImplT: ImplArray, SpecT, C: (float, complex, Expr)](TermSet[ImplT, SpecT, C]):
    """A sum of terms consisting of strings and coefficients.

    A set-like container of terms that may be used to store unique terms and perform algebraic
    operations on them.

    Coefficients may be real, complex, or symbolic so that like terms can be added.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    __array_priority__ = 1000

    @classmethod
    def from_terms(cls, terms: Terms[ImplT, SpecT, C]) -> Self:
        """Create a term sum from a collection of terms.

        Args:
            terms: Owned or viewed terms from which to construct the sum.

        Returns:
            A new instance of ``cls`` owning independent storage, with coefficients of
            duplicate strings added together.
        """
        out = cls.__new__(cls)
        TermSet.__init__(out, terms._new())
        if isinstance(terms._impl.coeffs, RealCoeffs | ComplexCoeffs):
            out._collect_terms(terms, sum_duplicates=True)
        else:
            out.add_iterable(terms)
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
        return cls.from_terms(cls.terms_type.from_iterable(source, *args, **kwargs))

    def _from_generator(self, gen: Iterator[Term[ImplT, SpecT, C]]) -> Self:
        """Create a new instance based on ``self`` with contents given by a generator.

        Args:
            gen: Generator of terms to be summed into the new instance.

        Returns:
            New instance with contents given by ``gen``.
        """
        out = self._new()
        out.add_iterable(gen)
        return out

    def add_iterable(self, source: Iterable[TermSpecT[ImplT, SpecT, C]]) -> None:
        """In-place addition of the terms in ``iterable`` to ``self``.

        Args:
            source: Iterable of term specifiers or compatible terms to add to ``self``.

        Note:
            This method operates in-place.
        """
        for item in source:
            term = self._get_working_term(item)
            self += term

    def _identity(self) -> Self:
        """Create a multiplicative identity of the same space and configuration as ``self``."""
        out = self._new()
        term = out._working_term
        term.coeff = unit(term.coeff_type)
        out.insert(term)
        return out

    def __pos__(self) -> Self:
        """Return ``self``."""
        return self.clone()

    def __neg__(self) -> Self:
        """Return the negation of ``self``."""
        return self * -1

    def __add__(self, rhs: Term[ImplT, SpecT, C] | Self) -> Self:
        """Addition of ``self`` and ``rhs``."""
        out = self.clone()
        out += rhs
        return out

    def __iadd__(self, rhs: Term[ImplT, SpecT, C] | Self) -> Self:
        """Add ``rhs`` to ``self`` in-place."""
        self._scaled_iadd(rhs, 1)
        return self

    def _scaled_iadd(self, rhs: Term[ImplT, SpecT, C] | Self, scalar: Coeff) -> None:
        """Add ``scalar`` times ``rhs`` to ``self`` in-place."""
        if not isinstance(rhs, Term | TermSum):
            rhs = self._get_working_term(rhs)
        if self.coeff_type != rhs.coeff_type:
            raise TypeError(
                f"Cannot add a term with coefficient type {rhs.coeff_type} "
                f"to a TermSum with coefficient type {self.coeff_type}."
            )
        if isinstance(rhs, Term):
            scaled_coeff = rhs.coeff * convert(scalar, self.coeff_type)
            index, inserted = self.soft_insert(rhs)
            if inserted:
                self._impl.coeffs[index] = scaled_coeff
            else:
                self._impl.coeffs[index] += scaled_coeff
        else:
            if rhs is self:
                rhs = rhs.clone()
            coeffs = self._impl.coeffs
            rhs_coeffs = rhs._impl.coeffs
            array = self._impl.strings._impl
            if isinstance(coeffs, RealCoeffs):
                assert isinstance(rhs_coeffs, RealCoeffs)
                array.scaled_iadd_real(
                    coeffs._impl,
                    self._string_set._map,
                    rhs._impl.strings._impl,
                    rhs_coeffs._impl,
                    convert(scalar, float),
                )
            elif isinstance(coeffs, ComplexCoeffs):
                assert isinstance(rhs_coeffs, ComplexCoeffs)
                array.scaled_iadd_complex(
                    coeffs._impl,
                    self._string_set._map,
                    rhs._impl.strings._impl,
                    rhs_coeffs._impl,
                    convert(scalar, complex),
                )
            else:
                for term in rhs:
                    self._scaled_iadd(term, scalar)
            filtered = self.filter_nonzero()
            self._bind(filtered._impl, filtered._string_set._map)

    def __sub__(self, rhs: Term[ImplT, SpecT, C] | Self) -> Self:
        """Subtraction of ``rhs`` from ``self``."""
        out = self.clone()
        out -= rhs
        return out

    def __isub__(self, rhs: Term[ImplT, SpecT, C] | Self) -> Self:
        """Subtract ``rhs`` from ``self`` in-place."""
        self._scaled_iadd(rhs, -1)
        return self

    def __mul__(self, other: Coeff | Coeffs[C]) -> Self:
        """Return ``self`` multiplied by a scalar or coefficient vector."""
        other = normalize_scalar(other)
        if not isinstance(other, Coeff | Coeffs):
            return NotImplemented
        out = self.clone()
        out *= other
        return out

    def __rmul__(self, coeff: Coeff | Coeffs[C]) -> Self:
        """Return ``self`` multiplied by a scalar or coefficient vector."""
        return self * coeff

    def __imul__(self, other: Coeff | Coeffs[C]) -> Self:
        """Multiply ``self`` in-place by a scalar or coefficient vector."""
        other = normalize_scalar(other)
        if not isinstance(other, Coeff | Coeffs):
            return NotImplemented
        coeffs = self._impl.coeffs
        coeffs *= other
        return self

    def __truediv__(self, scalar: Coeff) -> Self:
        """Division of ``self`` by ``scalar``."""
        scalar = normalize_scalar(scalar)
        factor = 1 / scalar if not isinstance(scalar, RootOfUnity) else 1 / scalar.to_numeric()
        return self * factor

    def __itruediv__(self, scalar: Coeff) -> Self:
        """In-place division of ``self`` by ``scalar``."""
        scalar = normalize_scalar(scalar)
        if isinstance(scalar, RootOfUnity):
            self *= 1 / scalar.to_numeric()
        else:
            self *= 1 / scalar
        return self

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


class SymbolicTermSum[ImplT: ImplArray, SpecT](TermSum[ImplT, SpecT, Expr]):
    """A sum of terms consisting of strings and symbolic coefficients.

    Coefficient substitution preserves the string keys and their space. Out-of-place
    substitution returns an independent owning sum of the same concrete type.
    """

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> Self:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            An independent owning sum of the same concrete type with the substitution
            applied. String keys, their space, and explicitly stored zeros are retained.
        """
        out = self.clone()
        out.isubs(values)
        return out

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place. String keys and explicitly stored zero
            coefficients are retained.
        """
        coeffs = self._impl.coeffs
        assert isinstance(coeffs, SymbolicCoeffs)
        coeffs.isubs(values)


class NumericTermSum[ImplT: ImplArray, SpecT, C: (float, complex)](TermSum[ImplT, SpecT, C]):
    """A sum of terms consisting of strings and numeric coefficients.

    A set-like container of terms that may be used to store unique terms and perform algebraic
    operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    @property
    def l1_norm(self) -> C:
        """The L1 norm of ``self``."""
        coeffs = self._impl.coeffs
        if isinstance(coeffs, RealCoeffs | ComplexCoeffs):
            return self.coeff_type(coeffs._impl.norm_slice(coeffs.slice, False))
        result: C = zero(self.coeff_type)
        for coeff in self._impl.coeffs:
            result += self.coeff_type(abs(coeff))
        return result

    def l1_normalize(self) -> None:
        """Divide ``self`` by its L1 norm.

        Note:
            This method operates in-place.
        """
        self._impl.coeffs.scale(1 / self.l1_norm)

    @property
    def l2_norm_square(self) -> C:
        """The square of the L2 norm of ``self``."""
        coeffs = self._impl.coeffs
        if isinstance(coeffs, RealCoeffs | ComplexCoeffs):
            return self.coeff_type(coeffs._impl.norm_slice(coeffs.slice, True))
        result: C = zero(self.coeff_type)
        for coeff in self._impl.coeffs:
            result += self.coeff_type(abs(coeff)) ** 2
        return result

    @property
    def l2_norm(self) -> Number:
        """The L2 norm of ``self``."""
        return cast("Number", self.l2_norm_square**0.5)

    def l2_normalize(self) -> None:
        """Divide ``self`` by its L2 norm.

        Note:
            This method operates in-place.
        """
        self._impl.coeffs.scale(1 / self.l2_norm)

    def iter_filter_significant(
        self, atol: float = DEFAULT_ATOL
    ) -> Iterator[Term[ImplT, SpecT, C]]:
        """Lazily generate terms in ``self`` with coefficients no less than ``atol``.

        Args:
            atol: Absolute tolerance.

        Returns:
            Iterator of terms that meet the criterion.
        """
        return (term for term in self if not np.isclose(term.coeff, 0, atol=atol))

    def filter_significant(self, atol: float = DEFAULT_ATOL) -> Self:
        """Eagerly get terms in ``self`` with coefficients no less than ``atol``.

        Args:
            atol: Absolute tolerance.

        Returns:
            New instance containing all terms that meet the criterion.
        """
        return self._from_generator(self.iter_filter_significant(atol))

    def iter_filter_insignificant(
        self, atol: float = DEFAULT_ATOL
    ) -> Iterator[Term[ImplT, SpecT, C]]:
        """Lazily generate terms in ``self`` with coefficients less than ``atol``.

        Args:
            atol: Absolute tolerance.

        Returns:
            Iterator of terms that meet the criterion.
        """
        return (term for term in self if np.isclose(term.coeff, 0, atol=atol))

    def filter_insignificant(self, atol: float = DEFAULT_ATOL) -> Self:
        """Eagerly get terms in ``self`` with coefficients less than ``atol``.

        Args:
            atol: Absolute tolerance.

        Returns:
            New instance containing all terms that meet the criterion.
        """
        return self._from_generator(self.iter_filter_insignificant(atol))
