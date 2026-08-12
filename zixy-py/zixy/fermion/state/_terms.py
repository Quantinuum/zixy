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

"""Terms containing fermionic occupation-number state strings.

The structure of this module parallels that of :mod:`~zixy.container.terms`, but with components
that are fermionic occupation-number state strings, as defined in
:mod:`~zixy.fermion.state._strings`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast, overload

import numpy as np
from numpy.typing import NDArray
from sympy import Expr, Symbol, sympify
from typing_extensions import Self

from zixy._zixy import BinarySprings, FermionStateArray, Modes
from zixy.container.coeffs import (
    Coeff,
    CoeffT,
    ComplexCoeffs,
    ComplexSign,
    Number,
    OtherCoeffT,
    RealCoeffs,
    Sign,
    SymbolicCoeffs,
    _is_complex,
    _is_complex_sign,
    _is_expr,
    _is_float,
    _is_int,
    _is_sign,
    get_coeffs_type,
)
from zixy.container.data import TermData
from zixy.container.terms import (
    NumericTerms,
    NumericTermSum,
)
from zixy.fermion._terms import (
    Term as FermionTerm,
    Terms as FermionTerms,
    TermSet as FermionTermSet,
    TermSum as FermionTermSum,
)
from zixy.fermion.state._strings import String, Strings, StringSpec
from zixy.utils import split_top_level

TermSpec: TypeAlias = String | tuple[StringSpec | String | None, CoeffT | None] | None
ElemT = bool
SpecT = StringSpec
ImplT = FermionStateArray
SignTermSpec = TermSpec[Sign]
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


def _parse_term_source(source: str) -> tuple[str, str | None]:
    source = source.strip()
    if source.startswith("(") and source.endswith(")"):
        parts = split_top_level(source[1:-1])
        if len(parts) == 2:
            return parts[1], parts[0]
    return source, None


def _parse_coeff(text: str | None, coeff_type: type[CoeffT]) -> Any:
    if text is None:
        return get_coeffs_type(coeff_type).from_size(1)[0]
    if coeff_type is Sign:
        return Sign.from_int(int(text))
    if coeff_type is float:
        return float(text)
    if coeff_type is complex:
        return complex(text.replace("i", "j"))
    if issubclass(coeff_type, Expr):
        return sympify(text)
    parser: Any = coeff_type
    return parser(text)


def _modes_from_dense_len(n: int) -> Modes:
    if n <= 1:
        return Modes.from_count(0)
    if n & (n - 1):
        raise ValueError("Dense state vector length must be a power of two.")
    return Modes.from_count(n.bit_length() - 1)


def _mul(lhs: Term[CoeffT], rhs: OtherCoeffT) -> Term[Any]:
    """Driver for multiplication of a term with a coefficient."""
    if not isinstance(rhs, Coeff):
        return NotImplemented
    scalar_product = lhs.coeff * rhs
    term_type = get_term_type(type(scalar_product))
    coeffs_type = get_coeffs_type(type(scalar_product))
    data = TermData(lhs._impl._cmpnts, coeffs_type.from_scalar(scalar_product))
    return term_type._create(data)


def _rmul(rhs: Term[CoeffT], lhs: OtherCoeffT) -> Term[Any]:
    """Driver for multiplication of a coefficient with a term."""
    return _mul(rhs, lhs)


class Term(FermionTerm[ImplT, SpecT, CoeffT, ElemT]):
    """A term consisting of a fermionic state string and a coefficient.

    A single mode-based term consisting of a state string and a coefficient that may be an owning
    instance referencing a single element in a :class:`~zixy.container.data.TermData` instance, or
    a view on an element in another collection.
    """

    cmpnts_type = Strings
    coeff_type: type[CoeffT]

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            A new instance containing the state string and coefficient in ``source``.
        """
        string_text, coeff_text = _parse_term_source(source)
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        impl = FermionStateArray(modes, BinarySprings(string_text))
        cmpnts = cls.cmpnts_type._create(impl)
        coeffs_type = get_coeffs_type(cls.coeff_type)
        coeffs = coeffs_type()
        coeffs.append(_parse_coeff(coeff_text, cls.coeff_type))
        data = TermData(cmpnts, coeffs)
        if len(data) != 1:
            raise ValueError(
                f"There should be exactly one Term string in the input, not {len(data)}."
            )
        return cls._create(data)


class Terms(FermionTerms[ImplT, SpecT, CoeffT, ElemT]):
    """A collection of terms consisting of fermionic state strings and coefficients."""

    term_type: type[Term[CoeffT]]

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            A new instance containing the state strings and coefficients in ``source``.
        """
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        term_strs = split_top_level(source)
        parsed_terms = [cls.term_type.from_str(term_str, modes) for term_str in term_strs]
        out_modes = modes
        if out_modes is None:
            out_modes = parsed_terms[0].modes if parsed_terms else Modes.from_count(0)
        return cls.from_iterable(parsed_terms, out_modes)


class TermSet(FermionTermSet[ImplT, SpecT, CoeffT, ElemT]):
    """A collection of unique terms consisting of fermionic state strings and coefficients."""

    terms_type: type[Terms[CoeffT]]


class TermSum(FermionTermSum[ImplT, SpecT, CoeffT, ElemT], TermSet[CoeffT]):
    """A sum of terms consisting of fermionic state strings and coefficients."""

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            A new instance containing the state strings and coefficients in ``source``.
        """
        terms_ = cls.terms_type.from_str(source, modes)
        return cls.from_iterable(terms_, terms_.modes)


class SignTerm(Term[Sign]):
    """A term consisting of a fermionic state string and a sign coefficient."""

    coeff_type = Sign

    @overload
    def __mul__(self, rhs: Sign) -> SignTerm: ...
    @overload
    def __mul__(self, rhs: ComplexSign) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: float) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: complex) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: Expr) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        return _mul(self, rhs)

    @overload
    def __rmul__(self, lhs: Sign) -> SignTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexSign) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: float) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: complex) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: Expr) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)


class SignTerms(Terms[Sign]):
    """A collection of terms consisting of state strings and sign coefficients."""

    term_type = SignTerm


class SignTermSet(TermSet[Sign]):
    """A collection of unique terms consisting of state strings and sign coefficients."""

    terms_type = SignTerms


class RealTerm(Term[float]):
    """A term consisting of a fermionic state string and a real coefficient."""

    coeff_type = float

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

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
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

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)


class RealTerms(NumericTerms[FermionStateArray, StringSpec, float], Terms[float]):
    """A collection of terms consisting of state strings and real coefficients."""

    term_type = RealTerm


class RealTermSet(TermSet[float]):
    """A collection of unique terms consisting of state strings and real coefficients."""

    terms_type = RealTerms


class RealTermSum(NumericTermSum[FermionStateArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of state strings and real coefficients."""

    terms_type = RealTerms

    @classmethod
    def from_dense(
        cls,
        modes: int | Modes | None = None,
        source: Sequence[float] = tuple(),
        big_endian: bool = False,
    ) -> Self:
        """Create an instance of ``cls`` from a dense vector.

        Args:
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the dense vector length.
            source: The vector to read from.
            big_endian: Whether to use big endian ordering for the resulting vector.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        if modes is None:
            modes = _modes_from_dense_len(len(source))
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        out = cls(modes)
        assert isinstance(out._impl._coeffs, RealCoeffs)
        out._impl._cmpnts._impl.from_dense_real(
            out._cmpnt_set._map, out._impl._coeffs._impl, list(source), big_endian
        )
        return out

    def to_dense(self, big_endian: bool = False) -> NDArray[np.float64]:
        """Convert ``self`` to a dense vector."""
        assert isinstance(self._impl._coeffs, RealCoeffs)
        return np.asarray(
            self._impl._cmpnts._impl.to_dense_real(self._impl._coeffs._impl, big_endian),
            dtype=np.float64,
        )

    def vdot(self, rhs: RealTermSum) -> float:
        """Compute the inner product of ``self`` with ``rhs``."""
        assert isinstance(self._impl._coeffs, RealCoeffs)
        assert isinstance(rhs._impl._coeffs, RealCoeffs)
        return float(
            self._impl._cmpnts._impl.vdot_real(
                self._cmpnt_set._map,
                self._impl._coeffs._impl,
                rhs._impl._cmpnts._impl,
                rhs._impl._coeffs._impl,
            )
        )


class ComplexTerm(Term[complex]):
    """A term consisting of a fermionic state string and a complex coefficient."""

    coeff_type = complex

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

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
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

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)


class ComplexTerms(NumericTerms[FermionStateArray, StringSpec, complex], Terms[complex]):
    """A collection of terms consisting of state strings and complex coefficients."""

    term_type = ComplexTerm


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms consisting of state strings and complex coefficients."""

    terms_type = ComplexTerms


class ComplexTermSum(NumericTermSum[FermionStateArray, StringSpec, complex], TermSum[complex]):
    """A sum of terms consisting of state strings and complex coefficients."""

    terms_type = ComplexTerms

    @classmethod
    def from_dense(
        cls,
        modes: int | Modes | None = None,
        source: Sequence[complex] = tuple(),
        big_endian: bool = False,
    ) -> Self:
        """Create an instance of ``cls`` from a dense vector.

        Args:
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the dense vector length.
            source: The vector to read from.
            big_endian: Whether to use big endian ordering for the resulting vector.

        Returns:
            An instance of ``cls`` parsed from ``source``.
        """
        if modes is None:
            modes = _modes_from_dense_len(len(source))
        if isinstance(modes, int):
            modes = Modes.from_count(modes)
        out = cls(modes)
        assert isinstance(out._impl._coeffs, ComplexCoeffs)
        out._impl._cmpnts._impl.from_dense_complex(
            out._cmpnt_set._map, out._impl._coeffs._impl, list(source), big_endian
        )
        return out

    @property
    def real_part(self) -> RealTermSum:
        """Return the real part of ``self``."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)
        data = TermData(self._data.cmpnts.clone(), self._data.coeffs.real_part)
        return RealTermSum._create(data)

    @property
    def imag_part(self) -> RealTermSum:
        """Return the imaginary part of ``self``."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)
        data = TermData(self._data.cmpnts.clone(), self._data.coeffs.imag_part)
        return RealTermSum._create(data)

    def to_dense(self, big_endian: bool = False) -> NDArray[np.complex128]:
        """Convert ``self`` to a dense vector."""
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        return np.asarray(
            self._impl._cmpnts._impl.to_dense_complex(self._impl._coeffs._impl, big_endian),
            dtype=np.complex128,
        )

    def vdot(self, rhs: ComplexTermSum) -> complex:
        """Compute the inner product of ``self`` with ``rhs``."""
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        assert isinstance(rhs._impl._coeffs, ComplexCoeffs)
        return complex(
            self._impl._cmpnts._impl.vdot_complex(
                self._cmpnt_set._map,
                self._impl._coeffs._impl,
                rhs._impl._cmpnts._impl,
                rhs._impl._coeffs._impl,
            )
        )


class SymbolicTerm(Term[Expr]):
    """A term consisting of a fermionic state string and a symbolic coefficient."""

    coeff_type = Expr

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

    def __mul__(self, rhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
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

    def __rmul__(self, lhs: OtherCoeffT) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        return _rmul(self, lhs)

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Substitute values into the symbolic coefficient in-place."""
        self.coeff = self.coeff.subs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerm:
        """Return a copy with values substituted into the symbolic coefficient."""
        out = self.clone()
        out.isubs(values)
        return out

    def try_to_real(self) -> RealTerm:
        """Try to evaluate ``self`` as a term with a real coefficient."""
        cmpnts = RealTerm.cmpnts_type.from_cmpnt(self.string.clone())
        coeffs = get_coeffs_type(RealTerm.coeff_type).from_scalar(float(self.coeff.evalf()))
        return RealTerm._create(TermData(cmpnts, coeffs))

    def try_to_complex(self) -> ComplexTerm:
        """Try to evaluate ``self`` as a term with a complex coefficient."""
        cmpnts = ComplexTerm.cmpnts_type.from_cmpnt(self.string.clone())
        coeffs = get_coeffs_type(ComplexTerm.coeff_type).from_scalar(complex(self.coeff.evalf()))
        return ComplexTerm._create(TermData(cmpnts, coeffs))


class SymbolicTerms(Terms[Expr]):
    """A collection of terms consisting of state strings and symbolic coefficients."""

    term_type = SymbolicTerm

    @property
    def coeffs(self) -> SymbolicCoeffs:
        """Get the coefficients of ``self``."""
        return cast(SymbolicCoeffs, self._data.coeffs[self.slice])

    @property
    def free_symbols(self) -> set[Symbol]:
        """Get the set of free symbols in ``self``."""
        return self.coeffs.free_symbols

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Substitute values into the symbolic coefficients in-place."""
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerms:
        """Return a copy with values substituted into the symbolic coefficients."""
        return SymbolicTerms._create(TermData(self.strings.clone(), self.coeffs.subs(values)))

    def idiff(self, variable: Symbol | str) -> None:
        """Differentiate partially with respect to ``variable`` in-place."""
        self.coeffs.idiff(variable)

    def diff(self, variable: Symbol | str) -> SymbolicTerms:
        """Differentiate partially with respect to ``variable`` out of place."""
        return SymbolicTerms._create(TermData(self.strings.clone(), self.coeffs.diff(variable)))

    def try_to_real(self) -> RealTerms:
        """Try to evaluate ``self`` as terms with real coefficients."""
        return RealTerms._create(TermData(self.strings.clone(), self.coeffs.try_to_real()))

    def try_to_complex(self) -> ComplexTerms:
        """Try to evaluate ``self`` as terms with complex coefficients."""
        return ComplexTerms._create(TermData(self.strings.clone(), self.coeffs.try_to_complex()))


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms consisting of state strings and symbolic coefficients."""

    terms_type = SymbolicTerms


class SymbolicTermSum(TermSum[Expr]):
    """A sum of terms consisting of state strings and symbolic coefficients."""

    terms_type = SymbolicTerms

    @property
    def coeffs(self) -> SymbolicCoeffs:
        """Get the coefficients of ``self``."""
        return cast(SymbolicCoeffs, self._data.coeffs)

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Substitute values into the symbolic coefficients in-place."""
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTermSum:
        """Return a copy with values substituted into the symbolic coefficients."""
        out = self.clone()
        out.isubs(values)
        return out


def get_term_type(coeff_type: type[CoeffT]) -> type[Term[CoeffT]]:
    """Get the term type corresponding to ``coeff_type``."""
    if _is_sign(coeff_type):
        return cast(type[Term[CoeffT]], SignTerm)
    elif _is_int(coeff_type) or _is_float(coeff_type):
        return cast(type[Term[CoeffT]], RealTerm)
    elif _is_complex(coeff_type) or _is_complex_sign(coeff_type):
        return cast(type[Term[CoeffT]], ComplexTerm)
    elif _is_expr(coeff_type):
        return cast(type[Term[CoeffT]], SymbolicTerm)
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type} for term type lookup.")
