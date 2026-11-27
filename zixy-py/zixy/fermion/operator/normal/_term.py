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

"""Individual terms and term specifications for normal-ordered fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, overload

from sympy import Expr

from zixy._zixy import NormalFermionOperatorArray
from zixy.container.coeff import (
    Coeff,
    ComplexSign,
    Sign,
    _integer_exponent,
    normalize_scalar,
    typesafe_mul,
)
from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSignCoeffs,
    RealCoeffs,
    SignCoeffs,
    SymbolicCoeffs,
)
from zixy.container.term import TermSpecT
from zixy.container.utils import _positive_power
from zixy.fermion.operator import normal as zfon
from zixy.fermion.operator._term import Term as OperatorTerm, _product_sign
from zixy.fermion.operator.normal._string import ElemT, ImplT, SpecT, String, StringSpec
from zixy.fermion.operator.normal._strings import Strings

if TYPE_CHECKING:
    import numpy as np
    from sympy import Symbol

    from zixy.container.coeff import Number

type TermSpec[C: (float, complex, Sign, ComplexSign, Expr)] = TermSpecT[
    NormalFermionOperatorArray, StringSpec, C
]

SignTermSpec = TermSpec[Sign]
ComplexSignTermSpec = TermSpec[ComplexSign]
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


def _mul[
    C: (float, complex, Sign, ComplexSign, Expr),
    COther: (float, complex, Sign, ComplexSign, Expr),
](lhs: Term[C], rhs: COther | String | Term[COther]) -> Term[Any] | zfon._term_sum.TermSum[Any]:
    """Driver for multiplication of a term with another term, a string, or a coefficient."""
    from zixy.fermion.operator.normal._term_sum import _term_sum_from_product  # noqa: PLC0415

    if isinstance(rhs, Coeff):
        scalar_product = lhs.coeff * rhs
        return get_term_type(type(scalar_product)).from_string(lhs.string, scalar_product)
    if isinstance(rhs, String):
        return _term_sum_from_product(lhs.modes, lhs.string, rhs, lhs.coeff)
    base_coeff = lhs.coeff * rhs.coeff
    return _term_sum_from_product(lhs.modes, lhs.string, rhs.string, base_coeff)


def _rmul[
    C: (float, complex, Sign, ComplexSign, Expr),
    COther: (float, complex, Sign, ComplexSign, Expr),
](rhs: Term[C], lhs: COther | String | Term[COther]) -> Term[Any] | zfon._term_sum.TermSum[Any]:
    """Driver for multiplication of another term, a string, or a coefficient with a term."""
    from zixy.fermion.operator.normal._term_sum import _term_sum_from_product  # noqa: PLC0415

    if isinstance(lhs, Coeff):
        return _mul(rhs, lhs)
    if isinstance(lhs, String):
        return _term_sum_from_product(rhs.modes, lhs, rhs.string, rhs.coeff)
    base_coeff = lhs.coeff * rhs.coeff
    return _term_sum_from_product(rhs.modes, lhs.string, rhs.string, base_coeff)


class Term[C: (float, complex, Sign, ComplexSign, Expr)](OperatorTerm[ImplT, SpecT, C, ElemT]):
    """A term consisting of a normal-ordered fermionic string and a coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a coefficient that
    may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another collection.
    """

    strings_type = Strings

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place, including the fermionic sign."""
        cre, ann = self.string.get_sets()
        self.string.set((ann, cre))
        coeff = self.coeff.conjugate() if hasattr(self.coeff, "conjugate") else self.coeff
        self.coeff = typesafe_mul(coeff, _product_sign(len(cre), len(ann)))

    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        out = self.clone()
        out.dagger()
        return out


class SignTerm(Term[Sign]):
    """A term consisting of a normal-ordered fermionic string and a sign coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a
    :class:`~zixy.container.coeff.Sign` that may be an owning instance referencing a single
    element in a :class:`~zixy.container.data.TermData` instance, or a view on an element in
    another collection.

    Products with strings or other :class:`SignTerm` instances return real-coefficient term sums.
    """

    coeffs_type = SignCoeffs

    @overload
    def __mul__(self, rhs: Sign) -> SignTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, rhs: float) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: String) -> zfon.RealTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> zfon.RealTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> zfon.RealTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        if not isinstance(rhs, Coeff | String | Term):
            return NotImplemented
        return _mul(self, rhs)

    @overload
    def __rmul__(self, lhs: Sign) -> SignTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, lhs: float) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: String) -> zfon.RealTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> zfon.RealTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> zfon.RealTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> Self | zfon.RealTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(
            zfon.RealTermSum.from_iterable([self.into(RealTerm)], self.modes),
            exponent,
            lambda lhs, rhs: lhs * rhs,
        )


class ComplexSignTerm(Term[ComplexSign]):
    """A term consisting of a normal-ordered fermionic string and a complex sign coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a
    :class:`~zixy.container.coeff.ComplexSign` coefficient that may be an owning instance
    referencing a single element in a :class:`~zixy.container.data.TermData` instance, or a view
    on an element in another collection.

    Products with strings or other sign-valued terms return complex-coefficient term sums.
    """

    coeffs_type = ComplexSignCoeffs

    @overload
    def __mul__(self, rhs: Sign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, rhs: float) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: String) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        if not isinstance(rhs, Coeff | String | Term):
            return NotImplemented
        return _mul(self, rhs)

    @overload
    def __rmul__(self, lhs: Sign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, lhs: float) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: String) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> Self | zfon.ComplexTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(
            zfon.ComplexTermSum.from_iterable([self.into(ComplexTerm)], self.modes),
            exponent,
            lambda lhs, rhs: lhs * rhs,
        )


class RealTerm(Term[float]):
    """A term consisting of a normal-ordered fermionic string and a real coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a ``float``
    coefficient that may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeffs_type = RealCoeffs

    @overload
    def __mul__(self, rhs: Sign) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSign) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: float) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: String) -> zfon.RealTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> zfon.RealTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> zfon.RealTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        if not isinstance(rhs, Coeff | String | Term):
            return NotImplemented
        return _mul(self, rhs)

    @overload
    def __rmul__(self, lhs: Sign) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: float) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: String) -> zfon.RealTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> zfon.RealTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> zfon.RealTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> Self | zfon.RealTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(
            zfon.RealTermSum.from_iterable([self.into(RealTerm)], self.modes),
            exponent,
            lambda lhs, rhs: lhs * rhs,
        )


class ComplexTerm(Term[complex]):
    """A term consisting of a normal-ordered fermionic string and a complex coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a ``complex``
    coefficient that may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeffs_type = ComplexCoeffs

    @overload
    def __mul__(self, rhs: Sign) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSign) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: float) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: String) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        if not isinstance(rhs, Coeff | String | Term):
            return NotImplemented
        return _mul(self, rhs)

    @overload
    def __rmul__(self, lhs: Sign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: float) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: String) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> zfon.ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> Self | zfon.ComplexTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(
            zfon.ComplexTermSum.from_iterable([self.into(ComplexTerm)], self.modes),
            exponent,
            lambda lhs, rhs: lhs * rhs,
        )


class SymbolicTerm(Term[Expr]):
    """A term consisting of a normal-ordered fermionic string and a symbolic coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a
    :class:`~sympy.Expr` coefficient that may be an owning instance referencing a single element
    in a :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeffs_type = SymbolicCoeffs

    @overload
    def __mul__(self, rhs: Sign) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSign) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: float) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: complex) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: String) -> zfon.SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> zfon.SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> zfon.SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> zfon.SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> zfon.SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        if not isinstance(rhs, Coeff | String | Term):
            return NotImplemented
        return _mul(self, rhs)

    @overload
    def __rmul__(self, lhs: Sign) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSign) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: float) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: complex) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: String) -> zfon.SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> zfon.SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> zfon.SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> zfon.SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> zfon.SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> zfon.SymbolicTermSum: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any] | zfon._term_sum.TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> Self | zfon.SymbolicTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        if exponent < 0:
            raise ValueError(f"Term powers require a nonnegative integer exponent, got {exponent}.")
        result = self * self
        for _ in range(2, exponent):
            product = zfon.SymbolicTermSum(self.modes)
            for term in result:
                product += term * self
            result = product
        return result

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerm:
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

        See Also:
            :meth:`~zixy.container.coeffs.SymbolicCoeffs.isubs`
        """
        self.coeff = self.coeff.subs(values)


def get_term_type[C: (float, complex, Sign, ComplexSign, Expr)](
    coeff_type: type[C],
) -> type[Term[C]]:
    """Get the term type corresponding to a given coefficient type.

    Args:
        coeff_type: The coefficient type for which to get the corresponding term type.

    Returns:
        The term type corresponding to ``coeff_type``.
    """
    if issubclass(coeff_type, Sign):
        return SignTerm
    elif issubclass(coeff_type, ComplexSign):
        return ComplexSignTerm
    if not issubclass(coeff_type, complex) and issubclass(coeff_type, float | int):
        return RealTerm
    if not issubclass(coeff_type, float | int) and issubclass(coeff_type, complex):
        return ComplexTerm
    elif issubclass(coeff_type, Expr):
        return SymbolicTerm
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type}.")
