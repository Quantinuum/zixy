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

"""Base fermionic ladder-operator terms and collections of such terms."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, cast

from sympy import Expr, sympify
from typing_extensions import Self

from zixy.container.cmpnts import SpecT
from zixy.container.coeffs import CoeffT, Sign, typesafe_mul, unit
from zixy.fermion._strings import ImplT
from zixy.fermion._terms import (
    Term as FermionTerm,
    Terms as FermionTerms,
    TermSet as FermionTermSet,
    TermSum as FermionTermSum,
)
from zixy.fermion.operator._strings import (
    ElemT,
    String as OperatorString,
    Strings as OperatorStrings,
)


def _sign_value(sign: Any) -> Sign:
    return Sign(sign)


def _signed_coeff(coeff: CoeffT, sign: Any) -> Any:
    return typesafe_mul(coeff, _sign_value(sign))


def _factor_coeff(coeff: CoeffT, factor: complex) -> Any:
    if abs(factor.imag) < 1e-14:
        factor = factor.real
    return typesafe_mul(coeff, factor)


def _parse_coeff(text: str | None, coeff_type: type[CoeffT]) -> Any:
    if text is None:
        return unit(coeff_type)
    if coeff_type is Sign:
        value: Any = Sign.from_int(int(text))
        return value
    if coeff_type is float:
        value = float(text)
        return value
    if coeff_type is complex:
        value = complex(text.replace("i", "j"))
        return value
    if issubclass(coeff_type, Expr):
        value = sympify(text)
        return value
    parser: Any = coeff_type
    return parser(text)


def _product_sign(cre: list[int], ann: list[int]) -> Sign:
    n_cre = len(cre)
    n_ann = len(ann)
    return Sign(((n_cre * (n_cre - 1) // 2) + (n_ann * (n_ann - 1) // 2)) & 1)


class Term(Generic[ImplT, SpecT, CoeffT, ElemT], FermionTerm[ImplT, SpecT, CoeffT, ElemT]):
    """A term consisting of a fermionic ladder-operator string and a coefficient."""

    cmpnts_type: type[OperatorStrings[ImplT, SpecT, ElemT]]

    @property
    def string(self) -> OperatorString[ImplT, SpecT, ElemT]:
        """Get the string component of the term."""
        return cast(OperatorString[ImplT, SpecT, ElemT], self.cmpnt)

    @abstractmethod
    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        pass

    @abstractmethod
    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        pass


class Terms(Generic[ImplT, SpecT, CoeffT, ElemT], FermionTerms[ImplT, SpecT, CoeffT, ElemT]):
    """A collection of terms consisting of fermionic ladder-operator strings and coefficients."""

    term_type: type[Term[ImplT, SpecT, CoeffT, ElemT]]

    @property
    def strings(self) -> OperatorStrings[ImplT, SpecT, ElemT]:
        """Get the string components of the terms."""
        return cast(OperatorStrings[ImplT, SpecT, ElemT], self.cmpnts)


class TermSet(Generic[ImplT, SpecT, CoeffT, ElemT], FermionTermSet[ImplT, SpecT, CoeffT, ElemT]):
    """A collection of unique terms consisting of fermionic ladder-operator strings."""

    terms_type: type[Terms[ImplT, SpecT, CoeffT, ElemT]]

    @property
    def strings(self) -> OperatorStrings[ImplT, SpecT, ElemT]:
        """Get the string components of the terms."""
        return cast(OperatorStrings[ImplT, SpecT, ElemT], self._impl._cmpnts)

    @property
    def coeffs(self) -> Any:
        """Get the coefficients of ``self``."""
        return self._impl._coeffs


class TermSum(
    Generic[ImplT, SpecT, CoeffT, ElemT],
    FermionTermSum[ImplT, SpecT, CoeffT, ElemT],
    TermSet[ImplT, SpecT, CoeffT, ElemT],
):
    """A sum of terms consisting of fermionic ladder-operator strings and coefficients."""

    @abstractmethod
    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        pass

    @abstractmethod
    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        pass
