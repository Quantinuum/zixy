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

"""Individual terms and term specifications for qubit operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, overload

import numpy as np  # noqa: TC002 (runtime type hints)
from sympy import Expr

from zixy._zixy import PauliMatrix, QubitPauliArray
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
from zixy.container.data import TermData
from zixy.container.term import TermSpecT
from zixy.container.utils import _positive_power
from zixy.qubit._term import Term as TermBase
from zixy.qubit.operator._string import String, StringSpec
from zixy.qubit.operator._strings import Strings

if TYPE_CHECKING:
    from sympy import Symbol

    from zixy.container.coeff import Number
    from zixy.qubit.clifford import GateList


type TermSpec[C: (float, complex, Sign, ComplexSign, Expr)] = TermSpecT[
    QubitPauliArray, StringSpec, C
]

SignTermSpec = TermSpec[Sign]
ComplexSignTermSpec = TermSpec[ComplexSign]
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


def _mul[
    C: (float, complex, Sign, ComplexSign, Expr),
    COther: (float, complex, Sign, ComplexSign, Expr),
](
    lhs: Term[C],
    rhs: COther | String | Term[COther],
) -> Term[Any]:
    """Driver for multiplication of a term with another term, a string, or a coefficient."""
    if isinstance(rhs, Coeff):
        scalar_product = lhs.coeff * rhs
        return get_term_type(type(scalar_product)).from_string(lhs.string, scalar_product)
    elif isinstance(rhs, String):
        product = lhs.string * rhs
        assert isinstance(product, ComplexSignTerm)
        return product * lhs.coeff
    elif isinstance(rhs, Term):
        product = lhs.string * rhs.string
        assert isinstance(product, ComplexSignTerm)
        return product * (lhs.coeff * rhs.coeff)
    return NotImplemented


def _rmul[
    C: (float, complex, Sign, ComplexSign, Expr),
    COther: (float, complex, Sign, ComplexSign, Expr),
](
    rhs: Term[C],
    lhs: COther | String | Term[COther],
) -> Term[Any]:
    """Driver for multiplication of another term, a string, or a coefficient with a term."""
    if isinstance(lhs, Coeff):
        return _mul(rhs, lhs)
    elif isinstance(lhs, String):
        product = lhs * rhs.string
        assert isinstance(product, ComplexSignTerm)
        return product * rhs.coeff
    elif isinstance(lhs, Term):
        product = lhs.string * rhs.string
        assert isinstance(product, ComplexSignTerm)
        return product * (lhs.coeff * rhs.coeff)
    return NotImplemented


class Term[C: (float, complex, Sign, ComplexSign, Expr)](
    TermBase[QubitPauliArray, StringSpec, C, PauliMatrix]
):
    """A term consisting of a Pauli string and a coefficient.

    A single qubit-based term consisting of a Pauli string and a coefficient that may be an owning
    instance referencing a single element in a :class:`~zixy.container.data.TermData`
    instance, or a view on an element in another collection.
    """

    strings_type = Strings

    def __imul__(self, rhs: Coeff | String | Term[C]) -> Self:
        """In-place multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        init_coeff = None
        different_qubits = True
        try:
            if isinstance(rhs, Coeff):
                different_qubits = False
                self.coeff = typesafe_mul(self.coeff, rhs)
            elif isinstance(rhs, String):
                phase = self.string.phase_of_mul(rhs)
                different_qubits = False
                self.coeff = typesafe_mul(self.coeff, phase)
                self.string.imul_ignore_phase(rhs)
            else:
                phase = self.string.phase_of_mul(rhs.string)
                different_qubits = False
                init_coeff = self.coeff
                self.coeff = typesafe_mul(self.coeff, phase)
                self.coeff = typesafe_mul(self.coeff, rhs.coeff)
                self.string.imul_ignore_phase(rhs.string)
        except ValueError as err:
            if different_qubits:
                raise err
            if init_coeff is not None:
                self.coeff = init_coeff
            extra = (
                f"Coefficient {rhs} is not {type(self.coeff)} representable."
                if isinstance(rhs, Coeff)
                else (
                    "Pauli string multiplication gives a ComplexSign factor "
                    f"that is not {type(self.coeff)} representable."
                )
            )
            raise ValueError(
                f"Cannot multiply {type(self)} {self} in-place by {type(rhs)} {rhs}. {extra}"
            )
        return self

    def conj_clifford_list(self, gates: GateList) -> None:
        """Conjugate ``self`` by a list of Clifford gates.

        Args:
            gates: The Clifford gates to conjugate by.

        Note:
            This method operates in-place.
        """
        signs = self._data.strings._impl.conj_clifford_list(gates._impl)
        coeffs = self._impl.coeffs
        coeffs *= SignCoeffs._create(signs)


class SignTerm(Term[Sign]):
    """A term consisting of a Pauli string and a sign coefficient.

    A single qubit-based term consisting of a Pauli string and a
    :class:`~zixy.container.coeff.Sign` that may be an owning instance referencing a single
    element in a :class:`~zixy.container.data.TermData` instance, or a view on an element in
    another collection.
    """

    coeffs_type = SignCoeffs

    @overload
    def __mul__(self, other: Sign) -> SignTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: float) -> RealTerm: ...
    @overload
    def __mul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: String) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: SignTerm) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: ComplexSignTerm) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: RealTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> SignTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: float) -> RealTerm: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: String) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: SignTerm) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: ComplexSignTerm) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: RealTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> Self | ComplexSignTerm:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self.into(ComplexSignTerm), exponent, lambda lhs, rhs: lhs * rhs)


class ComplexSignTerm(Term[ComplexSign]):
    """A term consisting of a Pauli string and a complex sign coefficient.

    A single qubit-based term consisting of a Pauli string and a
    :class:`~zixy.container.coeff.ComplexSign` that may be an owning instance referencing a
    single element in a :class:`~zixy.container.data.TermData` instance, or a view on an
    element in another collection.
    """

    coeffs_type = ComplexSignCoeffs

    @overload
    def __mul__(self, other: Sign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: float) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: String) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: SignTerm) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: ComplexSignTerm) -> ComplexSignTerm: ...
    @overload
    def __mul__(self, other: RealTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: float) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: String) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: SignTerm) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: ComplexSignTerm) -> ComplexSignTerm: ...
    @overload
    def __rmul__(self, other: RealTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
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
    """A term consisting of a Pauli string and a real coefficient.

    A single qubit-based term consisting of a Pauli string and a ``float`` coefficient that may be
    an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeffs_type = RealCoeffs

    @overload
    def __mul__(self, other: Sign) -> RealTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: float) -> RealTerm: ...
    @overload
    def __mul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: String) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: SignTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: ComplexSignTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: RealTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> RealTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: float) -> RealTerm: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: String) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: SignTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: ComplexSignTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: RealTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
        return _rmul(self, lhs)

    def __pow__(self, exponent: int | np.integer[Any]) -> Self | ComplexTerm:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self.into(ComplexTerm), exponent, lambda lhs, rhs: lhs * rhs)


class ComplexTerm(Term[complex]):
    """A term consisting of a Pauli string and a complex coefficient.

    A single qubit-based term consisting of a Pauli string and a ``complex`` coefficient that may
    be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeffs_type = ComplexCoeffs

    @overload
    def __mul__(self, other: Sign) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: float) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: String) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: SignTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: ComplexSignTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: RealTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: float) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: String) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: SignTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: ComplexSignTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: RealTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
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
    """A term consisting of a Pauli string and a symbolic coefficient.

    A single qubit-based term consisting of a Pauli string and a :class:`~sympy.Expr`
    coefficient that may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

    coeffs_type = SymbolicCoeffs

    @overload
    def __mul__(self, other: Sign) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: ComplexSign) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: float) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: complex) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: String) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: SignTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: ComplexSignTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: RealTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: ComplexTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, rhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        return _mul(self, rhs)

    @overload
    def __rmul__(self, other: Sign) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: ComplexSign) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: float) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: complex) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: Expr) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: String) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: SignTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: ComplexSignTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: RealTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: ComplexTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, other: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__[C: (float, complex, Sign, ComplexSign, Expr)](
        self, lhs: C | String | Term[C]
    ) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        lhs = normalize_scalar(lhs)
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

    def try_to_real(self) -> RealTerm:
        """Try to evaluate ``self`` as a term containing a vector of real coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.operator._term.RealTerm` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as real or there are free symbols.

        See Also:
            :meth:`~zixy.container.coeffs.SymbolicCoeffs.try_to_real`
        """
        strings = RealTerm.strings_type.from_string(self.string.clone())
        coeffs = RealTerm.coeffs_type.from_scalar(float(self.coeff.evalf()))
        term_data = TermData(strings, coeffs)
        return RealTerm._create(term_data)

    def try_to_complex(self) -> ComplexTerm:
        """Try to evaluate ``self`` as a term containing a vector of complex coefficients.

        Returns:
            An instance of :class:`~zixy.qubit.operator._term.ComplexTerm` with the evaluated
            coefficients.

        Raises:
            TypeError: A coefficient is not representable as complex or there are free symbols.

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
