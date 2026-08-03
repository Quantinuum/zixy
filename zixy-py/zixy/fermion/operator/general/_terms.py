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

"""Terms containing raw fermionic strings as components and collections of such terms.

The structure of this module parallels that of :mod:`~zixy.container.terms`, but with components
that are raw fermionic strings, as defined in
:mod:`~zixy.fermion.operator.general._strings`.
"""

from __future__ import annotations

from typing import Any, TypeAlias, cast

from sympy import Expr, Symbol
from typing_extensions import Self

from zixy._zixy import GeneralFermionOperatorArray, Modes
from zixy.container import terms
from zixy.container.coeffs import (
    Coeff,
    Coeffs,
    CoeffT,
    ComplexCoeffs,
    Number,
    RealCoeffs,
    SymbolicCoeffs,
    get_coeffs_type,
)
from zixy.container.data import TermData
from zixy.container.terms import (
    NumericTerms,
    NumericTermSum,
    Term as TermBase,
    Terms as TermsBase,
    TermSet as TermSetBase,
    TermSum as TermSumBase,
)
from zixy.fermion.operator._strings import parse_ladder_product, parse_term_source
from zixy.fermion.operator._terms import _parse_coeff
from zixy.fermion.operator.general._strings import (
    String,
    Strings,
    StringSpec,
    _default_modes,
)

TermSpec: TypeAlias = String | tuple[StringSpec | String | None, CoeffT | None] | None


class Term(TermBase[GeneralFermionOperatorArray, StringSpec, CoeffT]):
    """A term consisting of a raw fermionic string and a coefficient.

    A single mode-based term consisting of a raw fermionic string and a coefficient that may be an
    owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another collection.
    """

    cmpnts_type = Strings
    coeff_type: type[CoeffT]

    def __init__(self, modes: int | Modes = 0, source: TermSpec[CoeffT] = None):
        cmpnts = self.cmpnts_type(modes, 1)
        coeffs = get_coeffs_type(self.coeff_type).from_size(1)
        TermBase.__init__(self, TermData(cmpnts, coeffs))
        self.set(source)

    @property
    def string(self) -> String:
        """Get the string component of the term."""
        return cast(String, self.cmpnt)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self.string.modes

    @classmethod
    def term_data_from_str(
        cls, source: str, modes: int | Modes | None = None
    ) -> TermData[GeneralFermionOperatorArray, StringSpec, CoeffT]:
        """Create a new instance of :class:`~zixy.container.data.TermData`.

        Args:
            source: Input string to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

        Returns:
            A new instance containing the raw fermionic strings and coefficients in the
            ``source``.
        """
        parsed = parse_term_source(source)
        ops_by_term = [parse_ladder_product(cmpnt) for cmpnt, _ in parsed]
        if modes is None:
            modes = _default_modes([op for ops in ops_by_term for op in ops])
        elif isinstance(modes, int):
            modes = Modes.from_count(modes)
        cmpnts = Strings(modes, max_len=max((len(ops) for ops in ops_by_term), default=0))
        coeffs = get_coeffs_type(cls.coeff_type)()
        for ops, (_, coeff_text) in zip(ops_by_term, parsed, strict=True):
            cmpnts.append(ops)
            coeffs.append(_parse_coeff(coeff_text, cls.coeff_type))
        return TermData(cmpnts, coeffs)

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

        Returns:
            A new instance containing the raw fermionic string and coefficient in the ``source``.
        """
        data = cls.term_data_from_str(source, modes)
        if len(data) != 1:
            raise ValueError(
                f"There should be exactly one Term string in the input, not {len(data)}."
            )
        return cls._create(data)


class Terms(TermsBase[GeneralFermionOperatorArray, StringSpec, CoeffT]):
    """A collection of terms consisting of raw fermionic strings and coefficients.

    An array-like container of mode-based terms consisting of raw fermionic strings and
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type: type[Term[CoeffT]]

    def __init__(self, modes: int | Modes = 0, n: int = 0, max_len: int = 0):
        cmpnts = self.term_type.cmpnts_type(modes, n, max_len)
        coeffs = get_coeffs_type(self.term_type.coeff_type).from_size(n)
        TermsBase.__init__(self, TermData(cmpnts, coeffs))

    @property
    def strings(self) -> Strings:
        """Get the components of ``self``."""
        return cast(Strings, self.cmpnts)

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self.strings.modes

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

        Returns:
            A new instance containing the raw fermionic strings and coefficients in the
            ``source``.
        """
        return cls._create(cls.term_type.term_data_from_str(source, modes))


class TermSet(TermSetBase[GeneralFermionOperatorArray, StringSpec, CoeffT]):
    """A collection of unique terms consisting of raw fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[CoeffT]]

    def __init__(self, modes: int | Modes = 0, max_len: int = 0):
        TermSetBase.__init__(self, self.terms_type(modes, max_len=max_len))

    @property
    def strings(self) -> Strings:
        """Get the components of ``self``."""
        return cast(Strings, self._impl._cmpnts)

    @property
    def coeffs(self) -> Any:
        """Get the coefficients of ``self``."""
        return self._impl._coeffs

    @property
    def modes(self) -> Modes:
        """Get the modes corresponding to ``self``."""
        return self.strings.modes

    @property
    def max_len(self) -> int:
        """Get the maximum operator-product length supported by the backing array."""
        return self.strings.max_len


class TermSum(TermSumBase[GeneralFermionOperatorArray, StringSpec, CoeffT], TermSet[CoeffT]):
    """A sum of terms consisting of raw fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    linear combination operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

        Returns:
            A new instance containing the raw fermionic strings and coefficients in the
            ``source``.
        """
        terms_ = cls.terms_type.from_str(source, modes)
        return cls.from_iterable(terms_, terms_.modes, terms_.strings.max_len)

    @classmethod
    def from_iterable(cls, source: Any, modes: int | Modes = 0, max_len: int = 0) -> Self:
        """Create a new instance of ``cls`` from an iterable of terms."""
        out = cls(modes, max_len=max_len)
        out.add_iterable(source)
        return out

    def __mul__(self, rhs: Any) -> Any:
        """Multiplication of ``self`` by ``rhs``.

        Term-sum multiplication in the general representation concatenates raw ladder-operator
        products without applying normal-ordering identities.
        """
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        if isinstance(rhs, Term):
            rhs = type(self).from_iterable((rhs,), self.modes, rhs.string.max_len)
        if not isinstance(rhs, TermSum):
            return NotImplemented
        method = (
            self.strings._impl.lincomb_mul_real
            if self.terms_type.term_type.coeff_type is float
            else self.strings._impl.lincomb_mul_complex
        )
        impl, coeffs = method(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        if self.terms_type.term_type.coeff_type is float:
            data: Any = TermData(Strings._create(impl), RealCoeffs._create(coeffs))
            return type(self)._create(data)
        data = TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        return type(self)._create(data)

    def normal_ordered(self) -> Any:
        """Return ``self`` converted to normal-ordered form."""
        from zixy.fermion.operator.normal import (  # noqa: PLC0415
            ComplexTermSum,
            Strings as NormalStrings,
        )

        method = (
            self.strings._impl.lincomb_to_normal_order_real
            if self.terms_type.term_type.coeff_type is float
            else self.strings._impl.lincomb_to_normal_order_complex
        )
        impl, coeffs = method(self.strings._impl, self.coeffs._impl)
        return ComplexTermSum._create(
            TermData(
                NormalStrings._create(impl),
                ComplexCoeffs._create(coeffs),
            )
        )


class RealTerm(Term[float]):
    """A term consisting of a raw fermionic string and a real coefficient."""

    coeff_type = float


class RealTerms(NumericTerms[GeneralFermionOperatorArray, StringSpec, float], Terms[float]):
    """A collection of terms consisting of raw fermionic strings and real coefficients."""

    term_type = RealTerm


class RealTermSet(TermSet[float]):
    """A collection of unique terms consisting of raw fermionic strings and real coefficients."""

    terms_type = RealTerms


class RealTermSum(NumericTermSum[GeneralFermionOperatorArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of raw fermionic strings and real coefficients."""

    terms_type = RealTerms


class ComplexTerm(Term[complex]):
    """A term consisting of a raw fermionic string and a complex coefficient."""

    coeff_type = complex


class ComplexTerms(NumericTerms[GeneralFermionOperatorArray, StringSpec, complex], Terms[complex]):
    """A collection of terms consisting of raw fermionic strings and complex coefficients."""

    term_type = ComplexTerm


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms with raw fermionic strings and complex coefficients."""

    terms_type = ComplexTerms


class ComplexTermSum(
    NumericTermSum[GeneralFermionOperatorArray, StringSpec, complex], TermSum[complex]
):
    """A sum of terms consisting of raw fermionic strings and complex coefficients."""

    terms_type = ComplexTerms


class SymbolicTerm(Term[Expr]):
    """A term consisting of a raw fermionic string and a symbolic coefficient."""

    coeff_type = Expr

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Substitute values into the symbolic coefficient in-place."""
        self.coeff = self.coeff.subs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerm:
        """Return a copy with values substituted into the symbolic coefficient."""
        out = self.clone()
        out.isubs(values)
        return out


class SymbolicTerms(Terms[Expr]):
    """A collection of terms consisting of raw fermionic strings and symbolic coefficients."""

    term_type = SymbolicTerm

    @property
    def coeffs(self) -> SymbolicCoeffs:
        """Get the coefficients of ``self``."""
        return cast(SymbolicCoeffs, self._data.coeffs[self.slice])


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms with raw fermionic strings and symbolic coefficients."""

    terms_type = SymbolicTerms


class SymbolicTermSum(TermSum[Expr]):
    """A sum of terms consisting of raw fermionic strings and symbolic coefficients."""

    terms_type = SymbolicTerms


class TermRegistry(terms.TermRegistry[GeneralFermionOperatorArray, StringSpec]):
    term_type_sign: type[Any]
    term_type_complex_sign: type[Any]
    term_type_real: type[RealTerm]
    term_type_complex: type[ComplexTerm]
    term_type_symbolic: type[SymbolicTerm]

    def __init__(
        self,
        term_type_sign: type[Any],
        term_type_complex_sign: type[Any],
        term_type_real: type[RealTerm],
        term_type_complex: type[ComplexTerm],
        term_type_symbolic: type[SymbolicTerm],
    ) -> None:
        self.term_type_sign = term_type_sign
        self.term_type_complex_sign = term_type_complex_sign
        self.term_type_real = term_type_real
        self.term_type_complex = term_type_complex
        self.term_type_symbolic = term_type_symbolic

    def __getitem__(self, coeff_type: type[CoeffT]) -> type[Term[CoeffT]]:
        return cast(type[Term[CoeffT]], super().__getitem__(coeff_type))


String._term_registry = TermRegistry(
    term_type_sign=RealTerm,
    term_type_complex_sign=ComplexTerm,
    term_type_real=RealTerm,
    term_type_complex=ComplexTerm,
    term_type_symbolic=SymbolicTerm,
)

for _term_type, _sum_type in (
    (RealTerm, RealTermSum),
    (ComplexTerm, ComplexTermSum),
    (SymbolicTerm, SymbolicTermSum),
):
    setattr(_term_type, "_term_sum_type", _sum_type)
