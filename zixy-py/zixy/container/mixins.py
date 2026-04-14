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

"""Mixin classes for containers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, overload

from sympy import Expr

from zixy.container.cmpnts import Cmpnt, Cmpnts, ImplT, SpecT
from zixy.container.coeffs import (
    Coeff,
    CoeffT,
    ComplexSign,
    OtherCoeffT,
    Sign,
    get_coeffs_type,
)
from zixy.container.data import TermData


class CoeffMulMixin(Generic[CoeffT]):
    """Mixin class for typing the multiplication operator for coefficient types."""

    @overload
    def __mul__(self: CoeffMulMixin[float], rhs: float) -> CoeffMulMixin[float]: ...
    @overload
    def __mul__(self: CoeffMulMixin[float], rhs: complex) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[float], rhs: Sign) -> CoeffMulMixin[float]: ...
    @overload
    def __mul__(self: CoeffMulMixin[float], rhs: ComplexSign) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[float], rhs: Expr) -> CoeffMulMixin[Expr]: ...
    @overload
    def __mul__(self: CoeffMulMixin[complex], rhs: float) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[complex], rhs: complex) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[complex], rhs: Sign) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[complex], rhs: ComplexSign) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[complex], rhs: Expr) -> CoeffMulMixin[Expr]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Sign], rhs: float) -> CoeffMulMixin[float]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Sign], rhs: complex) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Sign], rhs: Sign) -> CoeffMulMixin[Sign]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Sign], rhs: ComplexSign) -> CoeffMulMixin[ComplexSign]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Sign], rhs: Expr) -> CoeffMulMixin[Expr]: ...
    @overload
    def __mul__(self: CoeffMulMixin[ComplexSign], rhs: float) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[ComplexSign], rhs: complex) -> CoeffMulMixin[complex]: ...
    @overload
    def __mul__(self: CoeffMulMixin[ComplexSign], rhs: Sign) -> CoeffMulMixin[ComplexSign]: ...
    @overload
    def __mul__(
        self: CoeffMulMixin[ComplexSign], rhs: ComplexSign
    ) -> CoeffMulMixin[ComplexSign]: ...
    @overload
    def __mul__(self: CoeffMulMixin[ComplexSign], rhs: Expr) -> CoeffMulMixin[Expr]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Expr], rhs: float) -> CoeffMulMixin[Expr]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Expr], rhs: complex) -> CoeffMulMixin[Expr]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Expr], rhs: Sign) -> CoeffMulMixin[Expr]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Expr], rhs: ComplexSign) -> CoeffMulMixin[Expr]: ...
    @overload
    def __mul__(self: CoeffMulMixin[Expr], rhs: Expr) -> CoeffMulMixin[Expr]: ...

    @abstractmethod
    def __mul__(self, rhs: OtherCoeffT) -> CoeffMulMixin[Any]:
        """Multiply :param:`self` with :param:`rhs`."""
        pass

    @overload
    def __rmul__(self: CoeffMulMixin[float], lhs: float) -> CoeffMulMixin[float]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[float], lhs: complex) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[float], lhs: Sign) -> CoeffMulMixin[float]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[float], lhs: ComplexSign) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[float], lhs: Expr) -> CoeffMulMixin[Expr]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[complex], lhs: float) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[complex], lhs: complex) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[complex], lhs: Sign) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[complex], lhs: ComplexSign) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[complex], lhs: Expr) -> CoeffMulMixin[Expr]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Sign], lhs: float) -> CoeffMulMixin[float]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Sign], lhs: complex) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Sign], lhs: Sign) -> CoeffMulMixin[Sign]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Sign], lhs: ComplexSign) -> CoeffMulMixin[ComplexSign]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Sign], lhs: Expr) -> CoeffMulMixin[Expr]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[ComplexSign], lhs: float) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[ComplexSign], lhs: complex) -> CoeffMulMixin[complex]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[ComplexSign], lhs: Sign) -> CoeffMulMixin[ComplexSign]: ...
    @overload
    def __rmul__(
        self: CoeffMulMixin[ComplexSign], lhs: ComplexSign
    ) -> CoeffMulMixin[ComplexSign]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[ComplexSign], lhs: Expr) -> CoeffMulMixin[Expr]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Expr], lhs: float) -> CoeffMulMixin[Expr]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Expr], lhs: complex) -> CoeffMulMixin[Expr]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Expr], lhs: Sign) -> CoeffMulMixin[Expr]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Expr], lhs: ComplexSign) -> CoeffMulMixin[Expr]: ...
    @overload
    def __rmul__(self: CoeffMulMixin[Expr], lhs: Expr) -> CoeffMulMixin[Expr]: ...

    def __rmul__(self, lhs: OtherCoeffT) -> CoeffMulMixin[Any]:
        """Multiply :param:`lhs` with :param:`self`."""
        return self.__mul__(lhs)


class CoeffDivMixin(Generic[CoeffT]):
    """Mixin class for typing the division operator for coefficient types."""

    @overload
    def __truediv__(self: CoeffDivMixin[float], rhs: float) -> CoeffDivMixin[float]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[float], rhs: complex) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[float], rhs: Sign) -> CoeffDivMixin[float]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[float], rhs: ComplexSign) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[float], rhs: Expr) -> CoeffDivMixin[Expr]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[complex], rhs: float) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[complex], rhs: complex) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[complex], rhs: Sign) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[complex], rhs: ComplexSign) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[complex], rhs: Expr) -> CoeffDivMixin[Expr]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Sign], rhs: float) -> CoeffDivMixin[float]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Sign], rhs: complex) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Sign], rhs: Sign) -> CoeffDivMixin[Sign]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Sign], rhs: ComplexSign) -> CoeffDivMixin[ComplexSign]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Sign], rhs: Expr) -> CoeffDivMixin[Expr]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[ComplexSign], rhs: float) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[ComplexSign], rhs: complex) -> CoeffDivMixin[complex]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[ComplexSign], rhs: Sign) -> CoeffDivMixin[ComplexSign]: ...
    @overload
    def __truediv__(
        self: CoeffDivMixin[ComplexSign], rhs: ComplexSign
    ) -> CoeffDivMixin[ComplexSign]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[ComplexSign], rhs: Expr) -> CoeffDivMixin[Expr]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Expr], rhs: float) -> CoeffDivMixin[Expr]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Expr], rhs: complex) -> CoeffDivMixin[Expr]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Expr], rhs: Sign) -> CoeffDivMixin[Expr]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Expr], rhs: ComplexSign) -> CoeffDivMixin[Expr]: ...
    @overload
    def __truediv__(self: CoeffDivMixin[Expr], rhs: Expr) -> CoeffDivMixin[Expr]: ...

    @abstractmethod
    def __truediv__(self, rhs: OtherCoeffT) -> CoeffDivMixin[Any]:
        """Divide :param:`self` by :param:`rhs`."""
        pass

    @overload
    def __rtruediv__(self: CoeffDivMixin[float], lhs: float) -> CoeffDivMixin[float]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[float], lhs: complex) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[float], lhs: Sign) -> CoeffDivMixin[float]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[float], lhs: ComplexSign) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[float], lhs: Expr) -> CoeffDivMixin[Expr]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[complex], lhs: float) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[complex], lhs: complex) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[complex], lhs: Sign) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[complex], lhs: ComplexSign) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[complex], lhs: Expr) -> CoeffDivMixin[Expr]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Sign], lhs: float) -> CoeffDivMixin[float]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Sign], lhs: complex) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Sign], lhs: Sign) -> CoeffDivMixin[Sign]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Sign], lhs: ComplexSign) -> CoeffDivMixin[ComplexSign]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Sign], lhs: Expr) -> CoeffDivMixin[Expr]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[ComplexSign], lhs: float) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[ComplexSign], lhs: complex) -> CoeffDivMixin[complex]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[ComplexSign], lhs: Sign) -> CoeffDivMixin[ComplexSign]: ...
    @overload
    def __rtruediv__(
        self: CoeffDivMixin[ComplexSign], lhs: ComplexSign
    ) -> CoeffDivMixin[ComplexSign]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[ComplexSign], lhs: Expr) -> CoeffDivMixin[Expr]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Expr], lhs: float) -> CoeffDivMixin[Expr]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Expr], lhs: complex) -> CoeffDivMixin[Expr]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Expr], lhs: Sign) -> CoeffDivMixin[Expr]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Expr], lhs: ComplexSign) -> CoeffDivMixin[Expr]: ...
    @overload
    def __rtruediv__(self: CoeffDivMixin[Expr], lhs: Expr) -> CoeffDivMixin[Expr]: ...

    @abstractmethod
    def __rtruediv__(self, lhs: OtherCoeffT) -> CoeffDivMixin[Any]:
        """Divide :param:`lhs` by :param:`self`."""
        pass


class TermMulMixin(Generic[ImplT, SpecT, CoeffT]):
    """Mixin class for typing the multiplication operator for term types."""

    coeff_type: type[CoeffT]
    cmpnts_type: type[Cmpnts[ImplT, SpecT]]

    _impl: TermData[ImplT, SpecT, CoeffT]

    @property
    @abstractmethod
    def coeff(self) -> CoeffT:
        """Get the coefficient associated with :param:`self`."""
        pass

    @coeff.setter
    @abstractmethod
    def coeff(self, value: CoeffT) -> None:
        """Set the coefficient associated with :param:`self`."""
        pass

    @property
    @abstractmethod
    def cmpnt(self) -> Cmpnt[ImplT, SpecT]:
        """Get the component associated with :param:`self`."""
        pass

    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float], rhs: float
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float], rhs: complex
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float], rhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float], rhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float], rhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex], rhs: float
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex], rhs: complex
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex], rhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex], rhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex], rhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: float
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: complex
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, Sign]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], rhs: float
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], rhs: complex
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], rhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], rhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], rhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: float
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: complex
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float], rhs: TermMulMixin[ImplT, SpecT, float]
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float],
        rhs: TermMulMixin[ImplT, SpecT, complex],
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float], rhs: TermMulMixin[ImplT, SpecT, Sign]
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float],
        rhs: TermMulMixin[ImplT, SpecT, ComplexSign],
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, float], rhs: TermMulMixin[ImplT, SpecT, Expr]
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex],
        rhs: TermMulMixin[ImplT, SpecT, float],
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex],
        rhs: TermMulMixin[ImplT, SpecT, complex],
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex], rhs: TermMulMixin[ImplT, SpecT, Sign]
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex],
        rhs: TermMulMixin[ImplT, SpecT, ComplexSign],
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, complex], rhs: TermMulMixin[ImplT, SpecT, Expr]
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: TermMulMixin[ImplT, SpecT, float]
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: TermMulMixin[ImplT, SpecT, complex]
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: TermMulMixin[ImplT, SpecT, Sign]
    ) -> TermMulMixin[ImplT, SpecT, Sign]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign],
        rhs: TermMulMixin[ImplT, SpecT, ComplexSign],
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Sign], rhs: TermMulMixin[ImplT, SpecT, Expr]
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign],
        rhs: TermMulMixin[ImplT, SpecT, float],
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign],
        rhs: TermMulMixin[ImplT, SpecT, complex],
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign],
        rhs: TermMulMixin[ImplT, SpecT, Sign],
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign],
        rhs: TermMulMixin[ImplT, SpecT, ComplexSign],
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign],
        rhs: TermMulMixin[ImplT, SpecT, Expr],
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: TermMulMixin[ImplT, SpecT, float]
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: TermMulMixin[ImplT, SpecT, complex]
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: TermMulMixin[ImplT, SpecT, Sign]
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr],
        rhs: TermMulMixin[ImplT, SpecT, ComplexSign],
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, Expr], rhs: TermMulMixin[ImplT, SpecT, Expr]
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __mul__(
        self: TermMulMixin[ImplT, SpecT, CoeffT], rhs: Cmpnt[ImplT, SpecT]
    ) -> TermMulMixin[ImplT, SpecT, CoeffT]: ...

    def __mul__(
        self,
        rhs: OtherCoeffT | Cmpnt[ImplT, SpecT] | TermMulMixin[ImplT, SpecT, OtherCoeffT],
    ) -> TermMulMixin[ImplT, SpecT, Any]:
        """Multiply :param:`self` with :param:`rhs`."""
        if not isinstance(rhs, Coeff | Cmpnt | TermMulMixin):
            return NotImplemented
        if isinstance(rhs, Coeff):
            scalar_product = self.coeff * rhs
            term_type = self.cmpnts_type.cmpnt_type._term_registry[type(scalar_product)]
            coeffs_type = get_coeffs_type(type(scalar_product))
            data = TermData(self._impl._cmpnts, coeffs_type.from_scalar(scalar_product))
            return term_type._create(data)
        elif isinstance(rhs, Cmpnt):
            # Undefined behaviour for base Cmpnt, but may be defined by derived classes
            return self.cmpnt * rhs * self.coeff
        else:
            # Undefined behaviour for base Cmpnt, but may be defined by derived classes
            return self.cmpnt * rhs.cmpnt * self.coeff * rhs.coeff

    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, float], lhs: float
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, float], lhs: complex
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, float], lhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, float], lhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, float], lhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, complex], lhs: float
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, complex], lhs: complex
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, complex], lhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, complex], lhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, complex], lhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Sign], lhs: float
    ) -> TermMulMixin[ImplT, SpecT, float]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Sign], lhs: complex
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Sign], lhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, Sign]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Sign], lhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Sign], lhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], lhs: float
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], lhs: complex
    ) -> TermMulMixin[ImplT, SpecT, complex]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], lhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], lhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, ComplexSign]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, ComplexSign], lhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Expr], lhs: float
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Expr], lhs: complex
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Expr], lhs: Sign
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Expr], lhs: ComplexSign
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, Expr], lhs: Expr
    ) -> TermMulMixin[ImplT, SpecT, Expr]: ...
    @overload
    def __rmul__(
        self: TermMulMixin[ImplT, SpecT, CoeffT], lhs: Cmpnt[ImplT, SpecT]
    ) -> TermMulMixin[ImplT, SpecT, CoeffT]: ...

    def __rmul__(self, lhs: OtherCoeffT | Cmpnt[ImplT, SpecT]) -> TermMulMixin[ImplT, SpecT, Any]:
        """Multiply :param:`lhs` with :param:`self`."""
        if not isinstance(lhs, Coeff | Cmpnt):
            return NotImplemented
        if isinstance(lhs, Coeff):
            return self.__mul__(lhs)
        else:
            # Undefined behaviour for base Cmpnt, but may be defined by derived classes
            return lhs * self.cmpnt * self.coeff
