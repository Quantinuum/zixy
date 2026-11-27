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

"""Individual terms and term specifications for fermionic states."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from sympy import Expr

from zixy._zixy import FermionStateArray
from zixy.container.coeff import Coeff, ComplexSign, Sign, normalize_scalar
from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSignCoeffs,
    RealCoeffs,
    SignCoeffs,
    SymbolicCoeffs,
)
from zixy.container.data import TermData
from zixy.container.term import TermSpecT
from zixy.fermion._term import Term as FermionTerm
from zixy.fermion.state._string import ElemT, ImplT, SpecT, StringSpec
from zixy.fermion.state._strings import Strings

if TYPE_CHECKING:
    from sympy import Symbol

    from zixy.container.coeff import Number


type TermSpec[C: (float, complex, Sign, ComplexSign, Expr)] = TermSpecT[
    FermionStateArray, StringSpec, C
]

SignTermSpec = TermSpec[Sign]
ComplexSignTermSpec = TermSpec[ComplexSign]
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


def _mul[
    COther: (float, complex, Sign, ComplexSign, Expr),
    C: (float, complex, Sign, ComplexSign, Expr),
](lhs: Term[C], rhs: COther) -> Term[Any]:
    """Driver for multiplication of a term with a coefficient."""
    if not isinstance(rhs, Coeff):
        return NotImplemented
    scalar_product = lhs.coeff * rhs
    return get_term_type(type(scalar_product)).from_string(lhs.string, scalar_product)


def _rmul[
    COther: (float, complex, Sign, ComplexSign, Expr),
    C: (float, complex, Sign, ComplexSign, Expr),
](rhs: Term[C], lhs: COther) -> Term[Any]:
    """Driver for multiplication of a coefficient with a term."""
    return _mul(rhs, lhs)


class Term[C: (float, complex, Sign, ComplexSign, Expr)](FermionTerm[ImplT, SpecT, C, ElemT]):
    """A term consisting of a fermionic state string and a coefficient.

    A single mode-based term consisting of a state string and a coefficient that may be an owning
    instance referencing a single element in a :class:`~zixy.container.data.TermData` instance, or
    a view on an element in another collection.
    """

    strings_type = Strings


class SignTerm(Term[Sign]):
    """A term consisting of a fermionic state string and a sign coefficient.

    A single mode-based term consisting of a state string and a
    :class:`~zixy.container.coeff.Sign` that may be an owning instance referencing a single
    element in a :class:`~zixy.container.data.TermData` instance, or a view on an element in
    another collection.
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

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
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

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        return _rmul(self, lhs)


class ComplexSignTerm(Term[ComplexSign]):
    """A term consisting of a fermionic state string and a complex sign coefficient.

    A single mode-based term consisting of a state string and a
    :class:`~zixy.container.coeff.ComplexSign` coefficient that may be an owning instance
    referencing a single element in a :class:`~zixy.container.data.TermData` instance, or a view
    on an element in another collection.
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

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
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

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)


class RealTerm(Term[float]):
    """A term consisting of a fermionic state string and a real coefficient.

    A single mode-based term consisting of a state string and a ``float`` coefficient that may be
    an owning instance referencing a single element in a
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

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
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

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        return _rmul(self, lhs)


class ComplexTerm(Term[complex]):
    """A term consisting of a fermionic state string and a complex coefficient.

    A single mode-based term consisting of a state string and a ``complex`` coefficient that may
    be an owning instance referencing a single element in a
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

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
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

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        return _rmul(self, lhs)


class SymbolicTerm(Term[Expr]):
    """A term consisting of a fermionic state string and a symbolic coefficient.

    A single mode-based term consisting of a state string and a :class:`~sympy.Expr` coefficient
    that may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
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

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](self, rhs: C) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
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

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](self, lhs: C) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        return _rmul(self, lhs)

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

    def try_to_real(self) -> RealTerm:
        """Try to evaluate ``self`` as a term containing a real coefficient.

        Returns:
            An instance of :class:`~zixy.fermion.state._term.RealTerm` with the evaluated
            coefficient.

        Raises:
            TypeError: If ``self`` cannot be evaluated with a real coefficient.

        See Also:
            :meth:`~zixy.container.coeffs.SymbolicCoeffs.try_to_real`
        """
        strings = RealTerm.strings_type.from_string(self.string.clone())
        coeffs = RealTerm.coeffs_type.from_scalar(float(self.coeff.evalf()))
        term_data = TermData(strings, coeffs)
        return RealTerm._create(term_data)

    def try_to_complex(self) -> ComplexTerm:
        """Try to evaluate ``self`` as a term containing a complex coefficient.

        Returns:
            An instance of :class:`~zixy.fermion.state._term.ComplexTerm` with the evaluated
            coefficient.

        Raises:
            TypeError: If ``self`` cannot be evaluated with a complex coefficient.

        See Also:
            :meth:`~zixy.container.coeffs.SymbolicCoeffs.try_to_complex`
        """
        strings = ComplexTerm.strings_type.from_string(self.string.clone())
        coeffs = ComplexTerm.coeffs_type.from_scalar(complex(self.coeff.evalf()))
        term_data = TermData(strings, coeffs)
        return ComplexTerm._create(term_data)


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
