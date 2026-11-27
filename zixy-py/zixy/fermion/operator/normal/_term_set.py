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

"""Term sets for normal-ordered fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr

from zixy.container.coeff import ComplexSign, Sign
from zixy.fermion.operator._term_set import TermSet as OperatorTermSet
from zixy.fermion.operator.normal._string import ElemT, ImplT, SpecT
from zixy.fermion.operator.normal._terms import (
    ComplexSignTerms,
    ComplexTerms,
    RealTerms,
    SignTerms,
    SymbolicTerms,
)

if TYPE_CHECKING:
    from sympy import Symbol

    from zixy.container.coeff import Number
    from zixy.fermion.operator.normal._terms import Terms


class TermSet[C: (float, complex, Sign, ComplexSign, Expr)](
    OperatorTermSet[ImplT, SpecT, C, ElemT]
):
    """A collection of unique terms consisting of normal-ordered fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[C]]


class SignTermSet(TermSet[Sign]):
    """A collection of unique terms with normal-ordered fermionic strings and sign coefficients.

    A set-like container of mode-based terms with :class:`~zixy.container.coeff.Sign`
    coefficients that may be used to store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SignTerms


class ComplexSignTermSet(TermSet[ComplexSign]):
    """Unique normal-ordered fermionic strings with complex sign coefficients.

    A set-like container of mode-based terms with :class:`~zixy.container.coeff.ComplexSign`
    coefficients that may be used to store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexSignTerms


class RealTermSet(TermSet[float]):
    """A collection of unique terms with normal-ordered fermionic strings and real coefficients.

    A set-like container of mode-based terms with ``float`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms with normal-ordered strings and complex coefficients.

    A set-like container of mode-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms with normal-ordered strings and symbolic coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTermSet:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        out = self.clone()
        out.isubs(values)
        return out

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        self.coeffs.isubs(values)
