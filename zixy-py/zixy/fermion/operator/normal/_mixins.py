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

"""Mixin classes for normal-ordered fermionic ladder-operator terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, overload

from sympy import Expr

from zixy.container.cmpnts import Cmpnt
from zixy.container.coeffs import (
    Coeff,
    CoeffT,
    ComplexSign,
    RootOfUnity,
    Sign,
    get_coeffs_type,
)
from zixy.container.data import TermData
from zixy.fermion.operator._terms import _signed_coeff

if TYPE_CHECKING:
    from zixy._zixy import Modes
    from zixy.container.coeffs import OtherCoeffT
    from zixy.fermion.operator.normal._strings import String, Strings
    from zixy.fermion.operator.normal._terms import TermSum


class TermMulMixin(Generic[CoeffT]):
    """Mixin class for normal-ordered fermionic term multiplication."""

    coeff_type: type[CoeffT]
    cmpnts_type: type[Strings]
    coeff: CoeffT

    _impl: TermData[Any, Any, CoeffT]

    if TYPE_CHECKING:

        @property
        def cmpnt(self) -> Cmpnt[Any, Any]:
            """Get the component associated with ``self``."""
            pass

        @property
        def modes(self) -> Modes:
            """Get the modes corresponding to ``self``."""
            pass

    @overload
    def __mul__(self: TermMulMixin[float], rhs: float) -> TermMulMixin[float]: ...

    @overload
    def __mul__(self: TermMulMixin[float], rhs: complex) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[float], rhs: Sign) -> TermMulMixin[float]: ...

    @overload
    def __mul__(self: TermMulMixin[float], rhs: ComplexSign) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[float], rhs: Expr) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[complex], rhs: float) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[complex], rhs: complex) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[complex], rhs: Sign) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[complex], rhs: ComplexSign) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[complex], rhs: Expr) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[Sign], rhs: float) -> TermMulMixin[float]: ...

    @overload
    def __mul__(self: TermMulMixin[Sign], rhs: complex) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[Sign], rhs: Sign) -> TermMulMixin[Sign]: ...

    @overload
    def __mul__(self: TermMulMixin[Sign], rhs: ComplexSign) -> TermMulMixin[ComplexSign]: ...

    @overload
    def __mul__(self: TermMulMixin[Sign], rhs: Expr) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[ComplexSign], rhs: float) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[ComplexSign], rhs: complex) -> TermMulMixin[complex]: ...

    @overload
    def __mul__(self: TermMulMixin[ComplexSign], rhs: Sign) -> TermMulMixin[ComplexSign]: ...

    @overload
    def __mul__(self: TermMulMixin[ComplexSign], rhs: ComplexSign) -> TermMulMixin[ComplexSign]: ...

    @overload
    def __mul__(self: TermMulMixin[ComplexSign], rhs: Expr) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[Expr], rhs: float) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[Expr], rhs: complex) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[Expr], rhs: Sign) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[Expr], rhs: ComplexSign) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[Expr], rhs: Expr) -> TermMulMixin[Expr]: ...

    @overload
    def __mul__(self: TermMulMixin[CoeffT], rhs: String) -> TermSum[CoeffT]: ...

    @overload
    def __mul__(
        self: TermMulMixin[float],
        rhs: TermMulMixin[float],
    ) -> TermSum[float]: ...

    @overload
    def __mul__(
        self: TermMulMixin[float],
        rhs: TermMulMixin[complex],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[float],
        rhs: TermMulMixin[Sign],
    ) -> TermSum[float]: ...

    @overload
    def __mul__(
        self: TermMulMixin[float],
        rhs: TermMulMixin[ComplexSign],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[float],
        rhs: TermMulMixin[Expr],
    ) -> TermSum[Expr]: ...

    @overload
    def __mul__(
        self: TermMulMixin[complex],
        rhs: TermMulMixin[float],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[complex],
        rhs: TermMulMixin[complex],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[complex],
        rhs: TermMulMixin[Sign],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[complex],
        rhs: TermMulMixin[ComplexSign],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[complex],
        rhs: TermMulMixin[Expr],
    ) -> TermSum[Expr]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Sign],
        rhs: TermMulMixin[float],
    ) -> TermSum[float]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Sign],
        rhs: TermMulMixin[complex],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Sign],
        rhs: TermMulMixin[Sign],
    ) -> TermSum[Sign]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Sign],
        rhs: TermMulMixin[ComplexSign],
    ) -> TermSum[ComplexSign]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Sign],
        rhs: TermMulMixin[Expr],
    ) -> TermSum[Expr]: ...

    @overload
    def __mul__(
        self: TermMulMixin[ComplexSign],
        rhs: TermMulMixin[float],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[ComplexSign],
        rhs: TermMulMixin[complex],
    ) -> TermSum[complex]: ...

    @overload
    def __mul__(
        self: TermMulMixin[ComplexSign],
        rhs: TermMulMixin[Sign],
    ) -> TermSum[ComplexSign]: ...

    @overload
    def __mul__(
        self: TermMulMixin[ComplexSign],
        rhs: TermMulMixin[ComplexSign],
    ) -> TermSum[ComplexSign]: ...

    @overload
    def __mul__(
        self: TermMulMixin[ComplexSign],
        rhs: TermMulMixin[Expr],
    ) -> TermSum[Expr]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Expr],
        rhs: TermMulMixin[float],
    ) -> TermSum[Expr]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Expr],
        rhs: TermMulMixin[complex],
    ) -> TermSum[Expr]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Expr],
        rhs: TermMulMixin[Sign],
    ) -> TermSum[Expr]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Expr],
        rhs: TermMulMixin[ComplexSign],
    ) -> TermSum[Expr]: ...

    @overload
    def __mul__(
        self: TermMulMixin[Expr],
        rhs: TermMulMixin[Expr],
    ) -> TermSum[Expr]: ...

    def __mul__(
        self,
        rhs: OtherCoeffT | String | TermMulMixin[OtherCoeffT],
    ) -> TermMulMixin[Any] | TermSum[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        if not isinstance(rhs, Coeff | Cmpnt | TermMulMixin):
            return NotImplemented
        if isinstance(rhs, Coeff):
            scalar_product = self.coeff * rhs
            term_type = self.cmpnts_type.cmpnt_type._term_registry[type(scalar_product)]
            coeffs_type = get_coeffs_type(type(scalar_product))
            data = TermData(self._impl._cmpnts, coeffs_type.from_scalar(scalar_product))
            return term_type._create(data)
        elif isinstance(rhs, Cmpnt):
            term_type = self.cmpnts_type.cmpnt_type._term_registry[type(self.coeff)]  # type: ignore
            out = term_type._term_sum_type(self.modes)
            impl, signs = self.cmpnt._impl.cmpnt_mul(self.cmpnt.index, rhs._impl, rhs.index)
            cmpnts = self.cmpnts_type._create(impl)
            for i in range(len(impl)):
                coeff = _signed_coeff(self.coeff, signs[i])
                term = term_type.from_cmpnt_coeff(cmpnts[i], coeff)
                if issubclass(term_type.coeff_type, RootOfUnity):
                    out.insert(term)
                else:
                    out += term
            return out
        else:
            base_coeff = self.coeff * rhs.coeff
            term_type = self.cmpnts_type.cmpnt_type._term_registry[type(base_coeff)]
            out = term_type._term_sum_type(self.modes)
            impl, signs = self.cmpnt._impl.cmpnt_mul(
                self.cmpnt.index, rhs.cmpnt._impl, rhs.cmpnt.index
            )
            cmpnts = self.cmpnts_type._create(impl)
            for i in range(len(impl)):
                coeff = _signed_coeff(base_coeff, signs[i])
                term = term_type.from_cmpnt_coeff(cmpnts[i], coeff)
                if issubclass(term_type.coeff_type, RootOfUnity):
                    out.insert(term)
                else:
                    out += term
            return out
