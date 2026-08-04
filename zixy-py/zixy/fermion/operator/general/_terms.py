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

from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

from sympy import Expr, Symbol
from typing_extensions import Self

from zixy._zixy import GeneralFermionOperatorArray, Modes, Qubits
from zixy.container import terms
from zixy.container.coeffs import (
    Coeff,
    Coeffs,
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
    Term as TermBase,
    Terms as TermsBase,
    TermSet as TermSetBase,
)
from zixy.fermion.operator._strings import LadderOp, parse_ladder_product, parse_term_source
from zixy.fermion.operator._terms import (
    Term as OperatorTerm,
    Terms as OperatorTerms,
    TermSet as OperatorTermSet,
    TermSum as OperatorTermSum,
    _parse_coeff,
)
from zixy.fermion.operator.general._strings import (
    String,
    Strings,
    StringSpec,
    _default_modes,
)
from zixy.qubit.pauli._terms import ComplexTermSum as PauliComplexTermSum

if TYPE_CHECKING:
    from zixy.fermion.mappings import Mapper
    from zixy.fermion.operator.normal._terms import (
        ComplexTermSum as NormalComplexTermSum,
        RealTermSum as NormalRealTermSum,
    )

TermSpec: TypeAlias = String | tuple[StringSpec | String | None, CoeffT | None] | None
ElemT = list[LadderOp]
SpecT = StringSpec
ImplT = GeneralFermionOperatorArray
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


def _max_len_from_string_source(source: StringSpec | String | None) -> int:
    if isinstance(source, String):
        return len(source.get_ops())
    if isinstance(source, str):
        return len(parse_ladder_product(source))
    if source is None:
        return 0
    return len(source)


def _max_len_from_term_source(source: TermSpec[Any]) -> int:
    if isinstance(source, tuple) and len(source) == 2:
        return _max_len_from_string_source(source[0])
    return _max_len_from_string_source(source)


def _mul(lhs: Term[CoeffT], rhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any]:
    """Driver for multiplication of a term with another term, a string, or a coefficient."""
    if isinstance(rhs, Coeff):
        scalar_product = lhs.coeff * rhs
        return get_term_type(type(scalar_product)).from_cmpnt_coeff(lhs.string, scalar_product)
    if isinstance(rhs, String):
        if lhs.modes != rhs.modes:
            raise ValueError("Cannot multiply terms defined over different modes.")
        string = String(lhs.modes, lhs.string.get_ops() + rhs.get_ops())
        return get_term_type(type(lhs.coeff)).from_cmpnt_coeff(string, lhs.coeff)
    if lhs.modes != rhs.modes:
        raise ValueError("Cannot multiply terms defined over different modes.")
    coeff = lhs.coeff * rhs.coeff
    string = String(lhs.modes, lhs.string.get_ops() + rhs.string.get_ops())
    return get_term_type(type(coeff)).from_cmpnt_coeff(string, coeff)


def _rmul(rhs: Term[CoeffT], lhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any]:
    """Driver for multiplication of another term, a string, or a coefficient with a term."""
    if isinstance(lhs, Coeff):
        return _mul(rhs, lhs)
    if isinstance(lhs, String):
        if lhs.modes != rhs.modes:
            raise ValueError("Cannot multiply terms defined over different modes.")
        string = String(rhs.modes, lhs.get_ops() + rhs.string.get_ops())
        return get_term_type(type(rhs.coeff)).from_cmpnt_coeff(string, rhs.coeff)
    if lhs.modes != rhs.modes:
        raise ValueError("Cannot multiply terms defined over different modes.")
    coeff = lhs.coeff * rhs.coeff
    string = String(rhs.modes, lhs.string.get_ops() + rhs.string.get_ops())
    return get_term_type(type(coeff)).from_cmpnt_coeff(string, coeff)


class Term(OperatorTerm[ImplT, SpecT, CoeffT, ElemT]):
    """A term consisting of a raw fermionic string and a coefficient.

    A single mode-based term consisting of a raw fermionic string and a coefficient that may be an
    owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another collection.
    """

    cmpnts_type = Strings
    coeff_type: type[CoeffT]

    def __init__(self, modes: int | Modes = 0, source: TermSpec[CoeffT] = None, max_len: int = 0):
        if max_len == 0:
            max_len = _max_len_from_term_source(source)
        cmpnts = self.cmpnts_type(modes, 1, max_len)
        coeffs = get_coeffs_type(self.coeff_type).from_size(1)
        TermBase.__init__(self, TermData(cmpnts, coeffs))
        self.set(source)

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

    @property
    def string(self) -> String:
        """Get the string component of the term."""
        return cast(String, self.cmpnt)

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


class Terms(OperatorTerms[ImplT, SpecT, CoeffT, ElemT]):
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
        """Get the string components of the terms."""
        return cast(Strings, self.cmpnts)

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


class TermSet(OperatorTermSet[ImplT, SpecT, CoeffT, ElemT]):
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
        """Get the string components of the terms."""
        return cast(Strings, self._impl._cmpnts)

    @property
    def coeffs(self) -> Any:
        """Get the coefficients of ``self``."""
        return self._impl._coeffs

    @property
    def max_len(self) -> int:
        """Get the maximum operator-product length supported by the backing array."""
        return self.strings.max_len


class TermSum(OperatorTermSum[ImplT, SpecT, CoeffT, ElemT], TermSet[CoeffT]):
    """A sum of terms consisting of raw fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    linear combination operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    def __init__(self, modes: int | Modes = 0, max_len: int = 0):
        TermSet.__init__(self, modes, max_len=max_len)

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

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        out = type(self)(self.modes, max_len=self.max_len)
        for term in self:
            out += term.into(self.terms_type.term_type).daggered()
        terms.TermSum.__init__(self, out.to_terms())

    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        out = self.clone()
        out.dagger()
        return out


class RealTerm(Term[float]):
    """A term consisting of a raw fermionic string and a real coefficient."""

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
    @overload
    def __mul__(self, rhs: String) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> RealTerm: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
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
    def __rmul__(self, lhs: RealTerm) -> RealTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)


class RealTerms(NumericTerms[GeneralFermionOperatorArray, StringSpec, float], Terms[float]):
    """A collection of terms consisting of raw fermionic strings and real coefficients."""

    term_type = RealTerm


class RealTermSet(TermSet[float]):
    """A collection of unique terms consisting of raw fermionic strings and real coefficients."""

    terms_type = RealTerms


class RealTermSum(NumericTermSum[GeneralFermionOperatorArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of raw fermionic strings and real coefficients."""

    terms_type = RealTerms

    @overload
    def __mul__(self, rhs: Coeff | Coeffs[float]) -> Self: ...
    @overload
    def __mul__(self, rhs: Self) -> RealTermSum: ...

    def __mul__(self, rhs: Coeff | Coeffs[float] | Self) -> Self | RealTermSum:
        """Multiplication of ``self`` by ``rhs``.

        Term-sum multiplication in the general representation concatenates raw ladder-operator
        products without applying normal-ordering identities.
        """
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        elif not isinstance(rhs, RealTermSum):
            return NotImplemented
        assert isinstance(self._impl._coeffs, RealCoeffs)
        assert isinstance(rhs._impl._coeffs, RealCoeffs)
        lhs_impl = self._impl._cmpnts._impl
        lhs_coeffs = self._impl._coeffs._impl
        rhs_impl = rhs._impl._cmpnts._impl
        rhs_coeffs = rhs._impl._coeffs._impl
        # TODO: support output by reference
        impl, coeffs = lhs_impl.lincomb_mul_real(lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs)
        data = TermData(Strings._create(impl), RealCoeffs._create(coeffs))
        return RealTermSum._create(data)

    def to_qubit(
        self,
        mapper: type[Mapper] | None = None,
        qubits: int | Qubits | None = None,
    ) -> PauliComplexTermSum:
        """Map this fermionic term sum to a qubit Pauli term sum.

        Args:
            mapper: The mapper class to use. If ``None``, use
                :class:`~zixy.fermion.mappings.JordanWignerMapper`.
            qubits: The qubit register or qubit count. If ``None``, infer from the number of
                fermionic modes.

        Returns:
            The mapped Pauli term sum.
        """
        from zixy.fermion.mappings import JordanWignerMapper  # noqa: PLC0415

        mapper = JordanWignerMapper if mapper is None else mapper
        if qubits is None:
            qubits = Qubits.from_count(len(self.modes))
        elif isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        mapper_ = mapper(qubits)
        out = PauliComplexTermSum(qubits)
        for term in self:
            out += mapper_.encode(term.cmpnt.into(String), term.coeff)
        return out

    def to_normal_ordered(self) -> NormalRealTermSum:
        """Convert this raw general term sum to the normal-ordered representation."""
        from zixy.fermion.operator.normal._strings import Strings as NormalStrings  # noqa: PLC0415
        from zixy.fermion.operator.normal._terms import (  # noqa: PLC0415
            RealTermSum as NormalRealTermSum,
        )

        impl, coeffs = self.strings._impl.lincomb_to_normal_order_real(
            self.strings._impl, self.coeffs._impl
        )
        return NormalRealTermSum._create(
            TermData(
                NormalStrings._create(impl),
                RealCoeffs._create(coeffs),
            )
        )


class ComplexTerm(Term[complex]):
    """A term consisting of a raw fermionic string and a complex coefficient."""

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
    @overload
    def __mul__(self, rhs: String) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
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
    def __rmul__(self, lhs: RealTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTerm: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)


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

    @overload
    def __mul__(self, rhs: Coeff | Coeffs[complex]) -> Self: ...
    @overload
    def __mul__(self, rhs: Self) -> ComplexTermSum: ...

    def __mul__(self, rhs: Coeff | Coeffs[complex] | Self) -> Self | ComplexTermSum:
        """Multiplication of ``self`` by ``rhs``.

        Term-sum multiplication in the general representation concatenates raw ladder-operator
        products without applying normal-ordering identities.
        """
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        elif not isinstance(rhs, ComplexTermSum):
            return NotImplemented
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        assert isinstance(rhs._impl._coeffs, ComplexCoeffs)
        lhs_impl = self._impl._cmpnts._impl
        lhs_coeffs = self._impl._coeffs._impl
        rhs_impl = rhs._impl._cmpnts._impl
        rhs_coeffs = rhs._impl._coeffs._impl
        # TODO: support output by reference
        impl, coeffs = lhs_impl.lincomb_mul_complex(lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs)
        data = TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        return ComplexTermSum._create(data)

    def to_qubit(
        self,
        mapper: type[Mapper] | None = None,
        qubits: int | Qubits | None = None,
    ) -> PauliComplexTermSum:
        """Map this fermionic term sum to a qubit Pauli term sum.

        Args:
            mapper: The mapper class to use. If ``None``, use
                :class:`~zixy.fermion.mappings.JordanWignerMapper`.
            qubits: The qubit register or qubit count. If ``None``, infer from the number of
                fermionic modes.

        Returns:
            The mapped Pauli term sum.
        """
        from zixy.fermion.mappings import JordanWignerMapper  # noqa: PLC0415

        mapper = JordanWignerMapper if mapper is None else mapper
        if qubits is None:
            qubits = Qubits.from_count(len(self.modes))
        elif isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        mapper_ = mapper(qubits)
        out = PauliComplexTermSum(qubits)
        for term in self:
            out += mapper_.encode(term.cmpnt.into(String), term.coeff)
        return out

    def to_normal_ordered(self) -> NormalComplexTermSum:
        """Convert this raw general term sum to the normal-ordered representation."""
        from zixy.fermion.operator.normal._strings import Strings as NormalStrings  # noqa: PLC0415
        from zixy.fermion.operator.normal._terms import (  # noqa: PLC0415
            ComplexTermSum as NormalComplexTermSum,
        )

        impl, coeffs = self.strings._impl.lincomb_to_normal_order_complex(
            self.strings._impl, self.coeffs._impl
        )
        return NormalComplexTermSum._create(
            TermData(
                NormalStrings._create(impl),
                ComplexCoeffs._create(coeffs),
            )
        )


class SymbolicTerm(Term[Expr]):
    """A term consisting of a raw fermionic string and a symbolic coefficient."""

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
    @overload
    def __mul__(self, rhs: String) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> SymbolicTerm: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTerm: ...

    def __mul__(self, rhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any]:
        """Multiplication of ``self`` by ``rhs``."""
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
    def __rmul__(self, lhs: RealTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> SymbolicTerm: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTerm: ...

    def __rmul__(self, lhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

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

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Substitute values into the symbolic coefficients in-place."""
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerms:
        """Return a copy with values substituted into the symbolic coefficients."""
        return SymbolicTerms._create(TermData(self.strings.clone(), self.coeffs.subs(values)))


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms with raw fermionic strings and symbolic coefficients."""

    terms_type = SymbolicTerms

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Substitute values into the symbolic coefficients in-place."""
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTermSet:
        """Return a copy with values substituted into the symbolic coefficients."""
        out = self.clone()
        out.isubs(values)
        return out


class SymbolicTermSum(TermSum[Expr]):
    """A sum of terms consisting of raw fermionic strings and symbolic coefficients."""

    terms_type = SymbolicTerms

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
    if _is_int(coeff_type) or _is_float(coeff_type) or _is_sign(coeff_type):
        return cast(type[Term[CoeffT]], RealTerm)
    elif _is_complex(coeff_type) or _is_complex_sign(coeff_type):
        return cast(type[Term[CoeffT]], ComplexTerm)
    elif _is_expr(coeff_type):
        return cast(type[Term[CoeffT]], SymbolicTerm)
    else:
        raise TypeError(f"Unsupported coefficient type {coeff_type} for term type lookup.")


for _term_type, _sum_type in (
    (RealTerm, RealTermSum),
    (ComplexTerm, ComplexTermSum),
    (SymbolicTerm, SymbolicTermSum),
):
    setattr(_term_type, "_term_sum_type", _sum_type)
