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

"""Individual terms and term specifications for general fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, overload

from sympy import Expr

from zixy._zixy import FermionSprings, GeneralFermionOperatorArray
from zixy.container.coeff import Coeff, ComplexSign, Sign, _integer_exponent, normalize_scalar
from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSignCoeffs,
    RealCoeffs,
    SignCoeffs,
    SymbolicCoeffs,
)
from zixy.container.data import TermData
from zixy.container.string import String as BaseString
from zixy.container.term import Term as TermBase, TermSpecT
from zixy.container.utils import _positive_power
from zixy.fermion._string import _check_modes_compatibility
from zixy.fermion.operator._term import Term as OperatorTerm
from zixy.fermion.operator.general._string import (
    ElemT,
    ImplT,
    SpecT,
    String,
    StringSpec,
    _string_length,
)
from zixy.fermion.operator.general._strings import Strings

if TYPE_CHECKING:
    import numpy as np
    from sympy import Symbol

    from zixy._zixy import Modes
    from zixy.container.coeff import Number


type TermSpec[C: (float, complex, Sign, ComplexSign, Expr)] = TermSpecT[
    GeneralFermionOperatorArray, StringSpec, C
]

SignTermSpec = TermSpec[Sign]
ComplexSignTermSpec = TermSpec[ComplexSign]
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


def _max_string_len_from_source[C: (float, complex, Sign, ComplexSign, Expr)](
    source: TermSpecT[ImplT, SpecT, C],
) -> int:
    """Get the maximum operator-product length from a term source."""
    if (
        isinstance(source, tuple)
        and len(source) == 2
        and (source[1] is None or isinstance(source[1], Coeff))
    ):
        return _max_string_len_from_source(source[0])
    if isinstance(source, TermBase):
        return _string_length(source.string)
    if isinstance(source, BaseString):
        return _string_length(source)
    if isinstance(source, str):
        if not source.strip():
            return 0
        springs = FermionSprings(source)
        return max((len(springs.get_ops(i)) for i in range(len(springs))), default=0)
    return len(source)


def _mul[
    C: (float, complex, Sign, ComplexSign, Expr),
    COther: (float, complex, Sign, ComplexSign, Expr),
](lhs: Term[C], rhs: COther | String | Term[COther]) -> Term[Any]:
    """Driver for multiplication of a term with another term, a string, or a coefficient."""
    if isinstance(rhs, Coeff):
        scalar_product = lhs.coeff * rhs
        return get_term_type(type(scalar_product)).from_string(lhs.string, scalar_product)
    if isinstance(rhs, String):
        _check_modes_compatibility(lhs.modes, rhs.modes)
        string = String(lhs.modes, lhs.string.get_ops() + rhs.get_ops())
        return get_term_type(type(lhs.coeff)).from_string(string, lhs.coeff)
    _check_modes_compatibility(lhs.modes, rhs.modes)
    coeff = lhs.coeff * rhs.coeff
    string = String(lhs.modes, lhs.string.get_ops() + rhs.string.get_ops())
    return get_term_type(type(coeff)).from_string(string, coeff)


def _rmul[
    C: (float, complex, Sign, ComplexSign, Expr),
    COther: (float, complex, Sign, ComplexSign, Expr),
](rhs: Term[C], lhs: COther | String | Term[COther]) -> Term[Any]:
    """Driver for multiplication of another term, a string, or a coefficient with a term."""
    if isinstance(lhs, Coeff):
        return _mul(rhs, lhs)
    if isinstance(lhs, String):
        _check_modes_compatibility(lhs.modes, rhs.modes)
        string = String(rhs.modes, lhs.get_ops() + rhs.string.get_ops())
        return get_term_type(type(rhs.coeff)).from_string(string, rhs.coeff)
    _check_modes_compatibility(lhs.modes, rhs.modes)
    coeff = lhs.coeff * rhs.coeff
    string = String(rhs.modes, lhs.string.get_ops() + rhs.string.get_ops())
    return get_term_type(type(coeff)).from_string(string, coeff)


class Term[C: (float, complex, Sign, ComplexSign, Expr)](OperatorTerm[ImplT, SpecT, C, ElemT]):
    """A term consisting of a raw fermionic string and a coefficient.

    A single mode-based term consisting of a raw fermionic string and a coefficient that may be an
    owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another collection.
    """

    strings_type = Strings

    def __init__(
        self,
        modes: int | Modes = 0,
        source: TermSpec[C] = "",
        *,
        max_string_len: int = 0,
    ):
        """Initialize the term.

        Args:
            modes: The mode space or number of modes.
            source: The term specifier to use for initial value.
            max_string_len: Maximum number of operators per string.
        """
        strings = self.strings_type(modes, size=1, max_string_len=max_string_len)
        strings._impl._reserve_string_length(_max_string_len_from_source(source), warn=False)
        coeffs = self.coeffs_type(size=1)
        term_data = TermData(strings, coeffs)
        TermBase.__init__(self, term_data)
        self.set(source)

    @property
    def max_string_len(self) -> int:
        """The maximum number of operators per string."""
        return self.string.max_string_len

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        self.string.dagger()
        if hasattr(self.coeff, "conjugate"):
            self.coeff = self.coeff.conjugate()

    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        out = self.clone()
        out.dagger()
        return out


class SignTerm(Term[Sign]):
    """A term consisting of a raw fermionic string and a sign coefficient.

    A single mode-based term consisting of a raw fermionic string and a
    :class:`~zixy.container.coeff.Sign` coefficient that may be an owning instance referencing a
    single element in a :class:`~zixy.container.data.TermData` instance, or a view on an element
    in another collection.
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
    def __mul__(self, rhs: String) -> SignTerm: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> SignTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
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
    def __rmul__(self, lhs: String) -> SignTerm: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> SignTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> SignTerm:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, SignTerm.__mul__)


class ComplexSignTerm(Term[ComplexSign]):
    """A term consisting of a raw fermionic string and a complex sign coefficient.

    A single mode-based term consisting of a raw fermionic string and a
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
    @overload
    def __mul__(self, rhs: String) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
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
    def __rmul__(self, lhs: String) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> ComplexSignTerm:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, ComplexSignTerm.__mul__)


class RealTerm(Term[float]):
    """A term consisting of a raw fermionic string and a real coefficient.

    A single mode-based term consisting of a raw fermionic string and a ``float`` coefficient
    that may be an owning instance referencing a single element in a
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
    def __mul__(self, rhs: String) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
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
    def __rmul__(self, lhs: String) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> RealTerm:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, RealTerm.__mul__)


class ComplexTerm(Term[complex]):
    """A term consisting of a raw fermionic string and a complex coefficient.

    A single mode-based term consisting of a raw fermionic string and a ``complex`` coefficient
    that may be an owning instance referencing a single element in a
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
    def __mul__(self, rhs: String) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
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
    def __rmul__(self, lhs: String) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> ComplexTerm:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, ComplexTerm.__mul__)


class SymbolicTerm(Term[Expr]):
    """A term consisting of a raw fermionic string and a symbolic coefficient.

    A single mode-based term consisting of a raw fermionic string and a
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
    def __mul__(self, rhs: String) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSignTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
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
    def __rmul__(self, lhs: String) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSignTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> SymbolicTerm:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, SymbolicTerm.__mul__)

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
