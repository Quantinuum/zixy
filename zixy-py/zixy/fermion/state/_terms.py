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

"""Term sequences for fermionic states."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy._zixy import FermionStateArray
from zixy.container.coeff import ComplexSign, Sign
from zixy.container.data import TermData
from zixy.container.terms import NumericTerms
from zixy.fermion._terms import Terms as FermionTerms
from zixy.fermion.state._string import ElemT, ImplT, SpecT, StringSpec
from zixy.fermion.state._term import ComplexSignTerm, ComplexTerm, RealTerm, SignTerm, SymbolicTerm

if TYPE_CHECKING:
    from sympy import Symbol

    from zixy.container.coeff import Number
    from zixy.fermion.state._term import Term


class Terms[C: (float, complex, Sign, ComplexSign, Expr)](FermionTerms[ImplT, SpecT, C, ElemT]):
    """A collection of terms consisting of fermionic state strings and coefficients.

    An array-like container of mode-based terms consisting of state strings and coefficients that
    may be an owning instance referencing a :class:`~zixy.container.data.TermData` instance, or
    a view on a slice of the elements in another collection.
    """

    term_type: type[Term[C]]


class SignTerms(Terms[Sign]):
    """A collection of terms consisting of state strings and sign coefficients.

    An array-like container of mode-based terms consisting of state strings and
    :class:`~zixy.container.coeff.Sign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = SignTerm


class ComplexSignTerms(Terms[ComplexSign]):
    """A collection of terms consisting of state strings and complex sign coefficients.

    An array-like container of mode-based terms consisting of state strings and
    :class:`~zixy.container.coeff.ComplexSign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = ComplexSignTerm


class RealTerms(NumericTerms[FermionStateArray, StringSpec, float], Terms[float]):
    """A collection of terms consisting of state strings and real coefficients.

    An array-like container of mode-based terms consisting of state strings and ``float``
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = RealTerm


class ComplexTerms(NumericTerms[FermionStateArray, StringSpec, complex], Terms[complex]):
    """A collection of terms consisting of state strings and complex coefficients.

    An array-like container of mode-based terms consisting of state strings and ``complex``
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = ComplexTerm


class SymbolicTerms(Terms[Expr]):
    """A collection of terms consisting of state strings and symbolic coefficients.

    An array-like container of mode-based terms consisting of state strings and
    :class:`~sympy.Expr` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = SymbolicTerm

    @property
    def free_symbols(self) -> set[Symbol]:
        """The set of free (unsubstituted) symbols in ``self``.

        Returns:
            Union of the sets of free symbols across all coefficients in ``self``.
        """
        return self.coeffs.free_symbols

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerms:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        term_data = TermData(self.strings.clone(), self.coeffs.subs(values))
        return SymbolicTerms._create(term_data)

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        self.coeffs.isubs(values)

    def diff(self, variable: Symbol | str) -> SymbolicTerms:
        """Differentiate partially with respect to ``variable`` out of place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Returns:
            A new contiguously stored instance with the differentiation applied.
        """
        term_data = TermData(self.strings.clone(), self.coeffs.diff(variable))
        return SymbolicTerms._create(term_data)

    def idiff(self, variable: Symbol | str) -> None:
        """Differentiate partially with respect to ``variable`` in-place.

        Args:
            variable: Symbol or name of symbol by which to differentiate the viewed symbolic
                expressions.

        Note:
            This method operates in-place.
        """
        self.coeffs.idiff(variable)

    def try_to_real(self) -> RealTerms:
        """Try to evaluate ``self`` as terms containing real coefficients.

        Returns:
            An instance of :class:`~zixy.fermion.state._terms.RealTerms` with the evaluated
            coefficients.

        Raises:
            TypeError: If ``self`` cannot be evaluated with real coefficients.
        """
        term_data = TermData(self.strings.clone(), self.coeffs.try_to_real())
        return RealTerms._create(term_data)

    def try_to_complex(self) -> ComplexTerms:
        """Try to evaluate ``self`` as terms containing complex coefficients.

        Returns:
            An instance of :class:`~zixy.fermion.state._terms.ComplexTerms` with the evaluated
            coefficients.

        Raises:
            TypeError: If ``self`` cannot be evaluated with complex coefficients.
        """
        term_data = TermData(self.strings.clone(), self.coeffs.try_to_complex())
        return ComplexTerms._create(term_data)
