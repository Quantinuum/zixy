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

"""Term sequences for normal-ordered fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy._zixy import NormalFermionOperatorArray
from zixy.container.coeff import ComplexSign, Sign
from zixy.container.data import TermData
from zixy.container.terms import NumericTerms
from zixy.fermion.operator._terms import Terms as OperatorTerms
from zixy.fermion.operator.normal._string import ElemT, ImplT, SpecT, StringSpec
from zixy.fermion.operator.normal._term import (
    ComplexSignTerm,
    ComplexTerm,
    RealTerm,
    SignTerm,
    SymbolicTerm,
)

if TYPE_CHECKING:
    from sympy import Symbol

    from zixy.container.coeff import Number
    from zixy.fermion.operator.normal._term import Term


class Terms[C: (float, complex, Sign, ComplexSign, Expr)](OperatorTerms[ImplT, SpecT, C, ElemT]):
    """A collection of terms consisting of normal-ordered fermionic strings and coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings and
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type: type[Term[C]]


class SignTerms(Terms[Sign]):
    """A collection of terms with normal-ordered fermionic strings and sign coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings
    and :class:`~zixy.container.coeff.Sign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = SignTerm


class ComplexSignTerms(Terms[ComplexSign]):
    """A collection of terms with normal-ordered fermionic strings and complex sign coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings and
    :class:`~zixy.container.coeff.ComplexSign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = ComplexSignTerm


class RealTerms(NumericTerms[NormalFermionOperatorArray, StringSpec, float], Terms[float]):
    """A collection of terms with normal-ordered fermionic strings and real coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings
    and ``float`` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = RealTerm


class ComplexTerms(NumericTerms[NormalFermionOperatorArray, StringSpec, complex], Terms[complex]):
    """A collection of terms with normal-ordered fermionic strings and complex coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings
    and ``complex`` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = ComplexTerm


class SymbolicTerms(Terms[Expr]):
    """A collection of terms with normal-ordered fermionic strings and symbolic coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings
    and :class:`~sympy.Expr` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = SymbolicTerm

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
