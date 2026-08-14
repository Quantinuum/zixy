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

"""Terms containing normal-ordered fermionic strings as components and collections of such terms.

The structure of this module parallels that of :mod:`~zixy.container.terms`, but with components
that are normal-ordered fermionic strings, as defined in
:mod:`~zixy.fermion.operator.normal._strings`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

from sympy import Expr, Symbol
from typing_extensions import Self

from zixy._zixy import Modes, NormalFermionOperatorArray, Qubits
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
    RootOfUnity,
    Sign,
    SymbolicCoeffs,
    _is_complex,
    _is_complex_sign,
    _is_expr,
    _is_float,
    _is_int,
    _is_sign,
    get_coeffs_type,
    typesafe_mul,
)
from zixy.container.data import TermData
from zixy.container.terms import (
    NumericTerms,
    NumericTermSum,
)
from zixy.fermion._strings import _check_modes_compatibility
from zixy.fermion.operator._strings import (
    parse_ladder_product,
    parse_term_source,
)
from zixy.fermion.operator._terms import (
    Term as OperatorTerm,
    Terms as OperatorTerms,
    TermSet as OperatorTermSet,
    TermSum as OperatorTermSum,
    _product_sign,
)
from zixy.fermion.operator.normal._strings import (
    String,
    Strings,
    StringSpec,
    _default_modes,
)
from zixy.fermion.state._terms import (
    ComplexTermSum as ComplexState,
    RealTermSum as RealState,
)
from zixy.qubit.pauli._terms import (
    ComplexTermSum as PauliComplexTermSum,
    RealTermSum as PauliRealTermSum,
)

if TYPE_CHECKING:
    from zixy.fermion.mappings import Mapper
    from zixy.fermion.operator.general._terms import (
        ComplexTermSum as GeneralComplexTermSum,
        RealTermSum as GeneralRealTermSum,
    )

TermSpec: TypeAlias = String | StringSpec | tuple[StringSpec | String, CoeffT | None]
ElemT = tuple[list[int], list[int]]
SpecT = StringSpec
ImplT = NormalFermionOperatorArray
SignTermSpec = TermSpec[Sign]
RealTermSpec = TermSpec[float]
ComplexTermSpec = TermSpec[complex]
SymbolicTermSpec = TermSpec[Expr]


def _term_sum_from_product(
    term_type: type[Term[Any]],
    modes: Modes,
    lhs: String,
    rhs: String,
    base_coeff: Any,
) -> TermSum[Any]:
    """Compute the product of two normal-ordered strings."""
    out = term_type._term_sum_type(modes)
    impl, signs = lhs._impl.cmpnt_mul(lhs.index, rhs._impl, rhs.index)
    cmpnts = term_type.cmpnts_type._create(impl)
    for i in range(len(impl)):
        coeff = typesafe_mul(base_coeff, Sign(signs[i]))
        term = term_type.from_cmpnt_coeff(cmpnts[i], coeff)
        if issubclass(term_type.coeff_type, RootOfUnity):
            out.insert(term)
        else:
            out += term
    return out


def _data_from_str(
    source: str, modes: int | Modes | None, coeff_type: type[CoeffT]
) -> TermData[NormalFermionOperatorArray, StringSpec, CoeffT]:
    """Create a new instance of :class:`~zixy.container.data.TermData` by parsing an input string.

    Args:
        source: Input string to parse.
        modes: The mode space or mode count. If ``None``, the mode space is inferred from
            the string specifier.
        coeff_type: The coefficient type.

    Returns:
        A new instance containing the normal-ordered fermionic strings and coefficients in the
        ``source``.
    """
    parsed = parse_term_source(source)
    if modes is None:
        modes = _default_modes(" ".join(cmpnt for cmpnt, _ in parsed))
    elif isinstance(modes, int):
        modes = Modes.from_count(modes)
    cmpnts = Strings(modes)
    coeffs_type = get_coeffs_type(coeff_type)
    base_coeffs = (
        coeffs_type.from_str(source) if "(" in source else coeffs_type.from_size(len(parsed))
    )
    coeffs = coeffs_type()
    for (cmpnt_text, _), base_coeff in zip(parsed, base_coeffs, strict=True):
        impl, coeffs_ = NormalFermionOperatorArray.from_ladder_product(
            modes, parse_ladder_product(cmpnt_text)
        )
        for i in range(len(impl)):
            cmpnts.append(Strings._create(impl)[i])
            coeffs.append(typesafe_mul(base_coeff, coeffs_[i]))
    return TermData(cmpnts, coeffs)


def _mul(
    lhs: Term[CoeffT], rhs: OtherCoeffT | String | Term[OtherCoeffT]
) -> Term[Any] | TermSum[Any]:
    """Driver for multiplication of a term with another term, a string, or a coefficient."""
    if isinstance(rhs, Coeff):
        scalar_product = lhs.coeff * rhs
        return get_term_type(type(scalar_product)).from_cmpnt_coeff(lhs.cmpnt, scalar_product)
    if isinstance(rhs, String):
        return _term_sum_from_product(
            get_term_type(type(lhs.coeff)), lhs.modes, lhs.string, rhs, lhs.coeff
        )
    base_coeff = lhs.coeff * rhs.coeff
    return _term_sum_from_product(
        get_term_type(type(base_coeff)), lhs.modes, lhs.string, rhs.string, base_coeff
    )


def _rmul(
    rhs: Term[CoeffT], lhs: OtherCoeffT | String | Term[OtherCoeffT]
) -> Term[Any] | TermSum[Any]:
    """Driver for multiplication of another term, a string, or a coefficient with a term."""
    if isinstance(lhs, Coeff):
        return _mul(rhs, lhs)
    if isinstance(lhs, String):
        return _term_sum_from_product(
            get_term_type(type(rhs.coeff)), rhs.modes, lhs, rhs.string, rhs.coeff
        )
    base_coeff = lhs.coeff * rhs.coeff
    return _term_sum_from_product(
        get_term_type(type(base_coeff)), rhs.modes, lhs.string, rhs.string, base_coeff
    )


class Term(OperatorTerm[ImplT, SpecT, CoeffT, ElemT]):
    """A term consisting of a normal-ordered fermionic string and a coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a coefficient that
    may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another collection.
    """

    cmpnts_type = Strings
    coeff_type: type[CoeffT]
    _term_sum_type: type[TermSum[CoeffT]]

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            A new instance containing the normal-ordered fermionic string and coefficient in the
            ``source``.
        """
        data = _data_from_str(source, modes, cls.coeff_type)
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


class Terms(OperatorTerms[ImplT, SpecT, CoeffT, ElemT]):
    """A collection of terms consisting of normal-ordered fermionic strings and coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings and
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type: type[Term[CoeffT]]

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            A new instance containing the normal-ordered fermionic strings and coefficients in the
            ``source``.
        """
        return cls._create(_data_from_str(source, modes, cls.term_type.coeff_type))


class TermSet(OperatorTermSet[ImplT, SpecT, CoeffT, ElemT]):
    """A collection of unique terms consisting of normal-ordered fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[CoeffT]]

    @property
    def coeffs(self) -> Any:
        """Get the coefficients of ``self``."""
        return self._impl._coeffs


class TermSum(OperatorTermSum[ImplT, SpecT, CoeffT, ElemT], TermSet[CoeffT]):
    """A sum of terms consisting of normal-ordered fermionic strings and coefficients.

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
            modes: The mode space or mode count. If ``None``, the mode space is inferred from
                the string specifier.

        Returns:
            A new instance containing the normal-ordered fermionic strings and coefficients in the
            ``source``.
        """
        terms_ = cls.terms_type.from_str(source, modes)
        return cls.from_iterable(terms_, terms_.modes)

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        out = type(self)(self.modes)
        for term in self:
            out += term.into(self.terms_type.term_type).daggered()
        terms.TermSum.__init__(self, out.to_terms())

    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        out = self.clone()
        out.dagger()
        return out


class SignTerm(Term[Sign]):
    """A term consisting of a normal-ordered fermionic string and a sign coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a
    :class:`~zixy.container.coeffs.Sign` that may be an owning instance referencing a single
    element in a :class:`~zixy.container.data.TermData` instance, or a view on an element in
    another collection.
    """

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
    @overload
    def __mul__(self, rhs: String) -> SignTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> SignTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> RealTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTermSum: ...

    def __mul__(self, rhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any] | TermSum[Any]:
        """Multiplication of ``self`` by ``rhs``."""
        if not isinstance(rhs, Coeff | String | Term):
            return NotImplemented
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
    @overload
    def __rmul__(self, lhs: String) -> SignTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> SignTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> RealTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTermSum: ...

    def __rmul__(self, lhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any] | TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)


class SignTerms(Terms[Sign]):
    """A collection of terms with normal-ordered fermionic strings and sign coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings
    and :class:`~zixy.container.coeffs.Sign` coefficients that may be an owning instance
    referencing a :class:`~zixy.container.data.TermData` instance, or a view on a slice of the
    elements in another collection.
    """

    term_type = SignTerm


class SignTermSet(TermSet[Sign]):
    """A collection of unique terms with normal-ordered fermionic strings and sign coefficients.

    A set-like container of mode-based terms with :class:`~zixy.container.coeffs.Sign`
    coefficients that may be used to store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SignTerms


class SignTermSum(TermSum[Sign]):
    """A sum of terms consisting of normal-ordered fermionic strings and sign coefficients."""

    terms_type = SignTerms


class RealTerm(Term[float]):
    """A term consisting of a normal-ordered fermionic string and a real coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a ``float``
    coefficient that may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

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
    def __mul__(self, rhs: String) -> RealTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> RealTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> RealTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTermSum: ...

    def __mul__(self, rhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any] | TermSum[Any]:
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
    def __rmul__(self, lhs: String) -> RealTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> RealTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> RealTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTermSum: ...

    def __rmul__(self, lhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any] | TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)


class RealTerms(NumericTerms[NormalFermionOperatorArray, StringSpec, float], Terms[float]):
    """A collection of terms with normal-ordered fermionic strings and real coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings
    and ``float`` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = RealTerm


class RealTermSet(TermSet[float]):
    """A collection of unique terms with normal-ordered fermionic strings and real coefficients.

    A set-like container of mode-based terms with ``float`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms


class RealTermSum(NumericTermSum[NormalFermionOperatorArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of normal-ordered fermionic strings and real coefficients.

    A set-like container of mode-based terms with ``float`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms

    @overload  # type: ignore[override]
    def __mul__(self, rhs: Coeff | Coeffs[float]) -> Self: ...
    @overload
    def __mul__(self, rhs: Self) -> RealTermSum: ...

    def __mul__(self, rhs: Coeff | Coeffs[float] | Self) -> Self | RealTermSum:
        """Multiplication of ``self`` by ``rhs``."""
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

    def commutator(self, rhs: TermSum[float]) -> RealTermSum:
        """Return the commutator ``[self, rhs]``."""
        impl, coeffs = self.strings._impl.lincomb_commutator_real(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        data = TermData(Strings._create(impl), RealCoeffs._create(coeffs))
        return RealTermSum._create(data)

    def anticommutator(self, rhs: TermSum[float]) -> RealTermSum:
        """Return the anticommutator ``{self, rhs}``."""
        impl, coeffs = self.strings._impl.lincomb_anticommutator_real(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        data = TermData(Strings._create(impl), RealCoeffs._create(coeffs))
        return RealTermSum._create(data)

    def is_hermitian(self, atol: float = 1e-10) -> bool:
        """Check whether ``self`` is Hermitian within the given tolerance."""
        return self.strings._impl.lincomb_is_hermitian_real(
            self.strings._impl, self.coeffs._impl, atol
        )

    def conserves_particle_number(self, atol: float = 1e-10) -> bool:
        """Check whether ``self`` conserves particle number within the given tolerance."""
        return self.strings._impl.lincomb_conserves_particle_number_real(
            self.strings._impl, self.coeffs._impl, atol
        )

    def max_n_body(self) -> int:
        """Return the maximum n-body order across all terms."""
        return self.strings._impl.lincomb_max_n_body_real(self.strings._impl, self.coeffs._impl)

    def active_modes(self) -> set[int]:
        """Return the set of mode indices on which ``self`` acts."""
        return self.strings._impl.lincomb_active_modes_real(self.strings._impl, self.coeffs._impl)

    def _to_qubit_real(
        self,
        mapper: type[Mapper],
        qubits: Qubits,
    ) -> PauliRealTermSum:
        """Map this fermionic term sum to a qubit Pauli term sum."""
        mapper_ = mapper(qubits)
        out = PauliRealTermSum(qubits)
        for term in self:
            out += mapper_.encode_real(term.cmpnt.into(String), term.coeff)
        return out

    def _to_qubit_complex(
        self,
        mapper: type[Mapper],
        qubits: Qubits,
    ) -> PauliComplexTermSum:
        """Map this fermionic term sum to a qubit Pauli term sum."""
        mapper_ = mapper(qubits)
        out = PauliComplexTermSum(qubits)
        for term in self:
            out += mapper_.encode_complex(term.cmpnt.into(String), term.coeff)
        return out

    def to_general(self) -> GeneralRealTermSum:
        """Convert this normal-ordered term sum to the raw general representation."""
        from zixy.fermion.operator.general._strings import (  # noqa: PLC0415
            Strings as GeneralStrings,
        )
        from zixy.fermion.operator.general._terms import (  # noqa: PLC0415
            RealTermSum as GeneralRealTermSum,
        )

        impl, coeffs = self.strings._impl.lincomb_to_general_real(
            self.strings._impl, self._cmpnt_set._map, self.coeffs._impl
        )
        return GeneralRealTermSum._create(
            TermData(
                GeneralStrings._create(impl),
                RealCoeffs._create(coeffs),
            )
        )

    def apply(self, state: RealState) -> RealState:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        _check_modes_compatibility(self.modes, state.modes)
        out = RealState(self.modes)
        assert isinstance(self._impl._coeffs, RealCoeffs)
        assert isinstance(state._impl._coeffs, RealCoeffs)
        assert isinstance(out._impl._coeffs, RealCoeffs)
        self._impl._cmpnts._impl.apply_to_state_real(
            self._impl._coeffs._impl,
            state._impl._cmpnts._impl,
            state._impl._coeffs._impl,
            out._impl._cmpnts._impl,
            out._cmpnt_set._map,
            out._impl._coeffs._impl,
        )
        return out

    def mat_elem(self, bra: RealState, ket: RealState) -> float:
        """Evaluate the matrix element of ``self`` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting matrix element.
        """
        _check_modes_compatibility(self.modes, bra.modes, ket.modes)
        assert isinstance(self._impl._coeffs, RealCoeffs)
        assert isinstance(bra._impl._coeffs, RealCoeffs)
        assert isinstance(ket._impl._coeffs, RealCoeffs)
        return float(
            self._impl._cmpnts._impl.mat_elem_real(
                self._impl._coeffs._impl,
                bra._impl._cmpnts._impl,
                bra._impl._coeffs._impl,
                ket._impl._cmpnts._impl,
                ket._impl._coeffs._impl,
            )
        )

    def exp_val(self, state: RealState) -> float:
        """Evaluate the expectation value of ``self`` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting expectation value.
        """
        return self.mat_elem(state, state)


class ComplexTerm(Term[complex]):
    """A term consisting of a normal-ordered fermionic string and a complex coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a ``complex``
    coefficient that may be an owning instance referencing a single element in a
    :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

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
    def __mul__(self, rhs: String) -> ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> ComplexTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTermSum: ...

    def __mul__(self, rhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any] | TermSum[Any]:
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
    def __rmul__(self, lhs: String) -> ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> ComplexTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTermSum: ...

    def __rmul__(self, lhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any] | TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)


class ComplexTerms(NumericTerms[NormalFermionOperatorArray, StringSpec, complex], Terms[complex]):
    """A collection of terms with normal-ordered fermionic strings and complex coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings
    and ``complex`` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = ComplexTerm


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms with normal-ordered strings and complex coefficients.

    A set-like container of mode-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms


class ComplexTermSum(
    NumericTermSum[NormalFermionOperatorArray, StringSpec, complex], TermSum[complex]
):
    """A sum of terms consisting of normal-ordered fermionic strings and complex coefficients.

    A set-like container of mode-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms

    @overload  # type: ignore[override]
    def __mul__(self, rhs: Coeff | Coeffs[complex]) -> Self: ...
    @overload
    def __mul__(self, rhs: Self) -> ComplexTermSum: ...

    def __mul__(self, rhs: Coeff | Coeffs[complex] | Self) -> Self | ComplexTermSum:
        """Multiplication of ``self`` by ``rhs``."""
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

    def commutator(self, rhs: TermSum[complex]) -> ComplexTermSum:
        """Return the commutator ``[self, rhs]``."""
        impl, coeffs = self.strings._impl.lincomb_commutator_complex(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        data = TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        return ComplexTermSum._create(data)

    def anticommutator(self, rhs: TermSum[complex]) -> ComplexTermSum:
        """Return the anticommutator ``{self, rhs}``."""
        impl, coeffs = self.strings._impl.lincomb_anticommutator_complex(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        data = TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        return ComplexTermSum._create(data)

    def is_hermitian(self, atol: float = 1e-10) -> bool:
        """Check whether ``self`` is Hermitian within the given tolerance."""
        return self.strings._impl.lincomb_is_hermitian_complex(
            self.strings._impl, self.coeffs._impl, atol
        )

    def conserves_particle_number(self, atol: float = 1e-10) -> bool:
        """Check whether ``self`` conserves particle number within the given tolerance."""
        return self.strings._impl.lincomb_conserves_particle_number_complex(
            self.strings._impl, self.coeffs._impl, atol
        )

    def max_n_body(self) -> int:
        """Return the maximum n-body order across all terms."""
        return self.strings._impl.lincomb_max_n_body_complex(self.strings._impl, self.coeffs._impl)

    def active_modes(self) -> set[int]:
        """Return the set of mode indices on which ``self`` acts."""
        return self.strings._impl.lincomb_active_modes_complex(
            self.strings._impl, self.coeffs._impl
        )

    def to_qubit(
        self,
        mapper: type[Mapper] | None = None,
        qubits: int | Qubits | None = None,
    ) -> PauliComplexTermSum:
        """Map this fermionic term sum to a qubit Pauli term sum.

        Args:
            mapper: The mapper class to use. If ``None``, use
                :class:`~zixy.fermion.mappings.JordanWignerMapper`.
            qubits: The qubit register or qubit count. If ``None``, the qubit register is
                inferred from the number of fermionic modes.

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
            out += mapper_.encode_complex(term.cmpnt.into(String), term.coeff)
        return out

    def to_general(self) -> GeneralComplexTermSum:
        """Convert this normal-ordered term sum to the raw general representation."""
        from zixy.fermion.operator.general._strings import (  # noqa: PLC0415
            Strings as GeneralStrings,
        )
        from zixy.fermion.operator.general._terms import (  # noqa: PLC0415
            ComplexTermSum as GeneralComplexTermSum,
        )

        impl, coeffs = self.strings._impl.lincomb_to_general_complex(
            self.strings._impl, self._cmpnt_set._map, self.coeffs._impl
        )
        return GeneralComplexTermSum._create(
            TermData(
                GeneralStrings._create(impl),
                ComplexCoeffs._create(coeffs),
            )
        )

    def apply(self, state: ComplexState) -> ComplexState:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        _check_modes_compatibility(self.modes, state.modes)
        out = ComplexState(self.modes)
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        assert isinstance(state._impl._coeffs, ComplexCoeffs)
        assert isinstance(out._impl._coeffs, ComplexCoeffs)
        self._impl._cmpnts._impl.apply_to_state_complex(
            self._impl._coeffs._impl,
            state._impl._cmpnts._impl,
            state._impl._coeffs._impl,
            out._impl._cmpnts._impl,
            out._cmpnt_set._map,
            out._impl._coeffs._impl,
        )
        return out

    def mat_elem(self, bra: ComplexState, ket: ComplexState) -> complex:
        """Evaluate the matrix element of ``self`` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting matrix element.
        """
        _check_modes_compatibility(self.modes, bra.modes, ket.modes)
        assert isinstance(self._impl._coeffs, ComplexCoeffs)
        assert isinstance(bra._impl._coeffs, ComplexCoeffs)
        assert isinstance(ket._impl._coeffs, ComplexCoeffs)
        return complex(
            self._impl._cmpnts._impl.mat_elem_complex(
                self._impl._coeffs._impl,
                bra._impl._cmpnts._impl,
                bra._impl._coeffs._impl,
                ket._impl._cmpnts._impl,
                ket._impl._coeffs._impl,
            )
        )

    def exp_val(self, state: ComplexState) -> complex:
        """Evaluate the expectation value of ``self`` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting expectation value.
        """
        return self.mat_elem(state, state)


class SymbolicTerm(Term[Expr]):
    """A term consisting of a normal-ordered fermionic string and a symbolic coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a
    :class:`~sympy.Expr` coefficient that may be an owning instance referencing a single element
    in a :class:`~zixy.container.data.TermData` instance, or a view on an element in another
    collection.
    """

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
    def __mul__(self, rhs: String) -> SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: SignTerm) -> SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: RealTerm) -> SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: ComplexTerm) -> SymbolicTermSum: ...
    @overload
    def __mul__(self, rhs: SymbolicTerm) -> SymbolicTermSum: ...

    def __mul__(self, rhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any] | TermSum[Any]:
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
    def __rmul__(self, lhs: String) -> SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: SignTerm) -> SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: RealTerm) -> SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: ComplexTerm) -> SymbolicTermSum: ...
    @overload
    def __rmul__(self, lhs: SymbolicTerm) -> SymbolicTermSum: ...

    def __rmul__(self, lhs: OtherCoeffT | String | Term[OtherCoeffT]) -> Term[Any] | TermSum[Any]:
        """Multiplication of ``lhs`` by ``self``."""
        if not isinstance(lhs, Coeff | String | Term):
            return NotImplemented
        return _rmul(self, lhs)

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


class SymbolicTerms(Terms[Expr]):
    """A collection of terms with normal-ordered fermionic strings and symbolic coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings
    and :class:`~sympy.Expr` coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type = SymbolicTerm

    @property
    def coeffs(self) -> SymbolicCoeffs:
        """Get the coefficients of ``self``."""
        return cast(SymbolicCoeffs, self._data.coeffs[self.slice])

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTerms:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        return SymbolicTerms._create(TermData(self.strings.clone(), self.coeffs.subs(values)))


class SymbolicTermSet(TermSet[Expr]):
    """A collection of unique terms with normal-ordered strings and symbolic coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTermSet:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
        out = self.clone()
        out.isubs(values)
        return out


class SymbolicTermSum(TermSum[Expr]):
    """A sum of terms consisting of normal-ordered fermionic strings and symbolic coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Apply a partial substitution of the symbols in-place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Note:
            This method operates in-place.
        """
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTermSum:
        """Apply a partial substitution of the symbols out of place.

        Args:
            values: Map from a symbol or symbol name to its new expression or numeric value.

        Returns:
            A new contiguously stored instance with the substitution applied.
        """
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


for _term_type, _sum_type in (
    (SignTerm, SignTermSum),
    (RealTerm, RealTermSum),
    (ComplexTerm, ComplexTermSum),
    (SymbolicTerm, SymbolicTermSum),
):
    setattr(_term_type, "_term_sum_type", _sum_type)
