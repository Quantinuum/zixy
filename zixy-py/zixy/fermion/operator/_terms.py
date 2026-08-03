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
:mod:`~zixy.fermion.operator._strings`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias, cast

from sympy import Expr, Symbol, sympify
from typing_extensions import Self

from zixy._zixy import Modes, NormalFermionOperatorArray
from zixy.container import terms
from zixy.container.coeffs import (
    Coeff,
    Coeffs,
    CoeffT,
    ComplexCoeffs,
    Number,
    RealCoeffs,
    Sign,
    SymbolicCoeffs,
    get_coeffs_type,
    typesafe_mul,
    unit,
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
from zixy.fermion.operator._strings import (
    String,
    Strings,
    StringSpec,
    _default_modes,
    parse_ladder_product,
    parse_term_source,
)
from zixy.fermion.state._terms import (
    ComplexTermSum as ComplexState,
    RealTermSum as RealState,
)

TermSpec: TypeAlias = String | tuple[StringSpec | String | None, CoeffT | None] | None


def _sign_value(sign: Any) -> Sign:
    return Sign(sign)


def _signed_coeff(coeff: CoeffT, sign: Any) -> Any:
    return typesafe_mul(coeff, _sign_value(sign))


def _factor_coeff(coeff: CoeffT, factor: complex) -> Any:
    if abs(factor.imag) < 1e-14:
        factor = factor.real
    return typesafe_mul(coeff, factor)


def _parse_coeff(text: str | None, coeff_type: type[CoeffT]) -> Any:
    if text is None:
        return unit(coeff_type)
    if coeff_type is Sign:
        value: Any = Sign.from_int(int(text))
        return value
    if coeff_type is float:
        value = float(text)
        return value
    if coeff_type is complex:
        value = complex(text.replace("i", "j"))
        return value
    if issubclass(coeff_type, Expr):
        value = sympify(text)
        return value
    parser: Any = coeff_type
    return parser(text)


def _product_sign(cre: list[int], ann: list[int]) -> Sign:
    n_cre = len(cre)
    n_ann = len(ann)
    return Sign(((n_cre * (n_cre - 1) // 2) + (n_ann * (n_ann - 1) // 2)) & 1)


class Term(TermBase[NormalFermionOperatorArray, StringSpec, CoeffT]):
    """A term consisting of a normal-ordered fermionic string and a coefficient.

    A single mode-based term consisting of a normal-ordered fermionic string and a coefficient that
    may be an owning instance referencing a single element in a
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
    ) -> TermData[NormalFermionOperatorArray, StringSpec, CoeffT]:
        """Create a new instance of :class:`~zixy.container.data.TermData`.

        Args:
            source: Input string to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

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
        coeffs = get_coeffs_type(cls.coeff_type)()
        for cmpnt_text, coeff_text in parsed:
            base_coeff = _parse_coeff(coeff_text, cls.coeff_type)
            impl, coeffs_ = NormalFermionOperatorArray.from_ladder_product(
                modes, parse_ladder_product(cmpnt_text)
            )
            for i in range(len(impl)):
                cmpnts.append(Strings._create(impl)[i])
                coeffs.append(_factor_coeff(base_coeff, coeffs_[i]))
        return TermData(cmpnts, coeffs)

    @classmethod
    def from_str(cls, source: str, modes: int | Modes | None = None) -> Self:
        """Create a new instance of ``cls`` by parsing an input string.

        Args:
            source: Input string to parse.
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

        Returns:
            A new instance containing the normal-ordered fermionic string and coefficient in the
            ``source``.
        """
        data = cls.term_data_from_str(source, modes)
        if len(data) != 1:
            raise ValueError(
                f"There should be exactly one Term string in the input, not {len(data)}."
            )
        return cls._create(data)

    def __mul__(self, rhs: Any) -> Any:
        """Multiplication of ``self`` by ``rhs``."""
        if isinstance(rhs, Coeff):
            coeff = self.coeff * rhs
            term_type = cast(Any, self.string._term_registry)[type(coeff)]
            return term_type.from_cmpnt_coeff(self.string, coeff)
        if isinstance(rhs, String):
            rhs = type(self).from_cmpnt_coeff(rhs, unit(self.coeff_type))
        if not isinstance(rhs, Term):
            return NotImplemented
        out = self._term_sum_type(self.modes)  # type: ignore[attr-defined]
        impl, signs = self.string._impl.cmpnt_mul(
            self.string.index, rhs.string._impl, rhs.string.index
        )
        for i in range(len(impl)):
            coeff = typesafe_mul(self.coeff, rhs.coeff)
            coeff = _signed_coeff(coeff, signs[i])
            out += type(self).from_cmpnt_coeff(Strings._create(impl)[i], coeff)
        return out

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place, including the fermionic sign."""
        cre, ann = self.string.get_sets()
        self.string.set((ann, cre))
        coeff = self.coeff.conjugate() if hasattr(self.coeff, "conjugate") else self.coeff
        self.coeff = _signed_coeff(coeff, _product_sign(cre, ann))

    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        out = self.clone()
        out.dagger()
        return out


class Terms(TermsBase[NormalFermionOperatorArray, StringSpec, CoeffT]):
    """A collection of terms consisting of normal-ordered fermionic strings and coefficients.

    An array-like container of mode-based terms consisting of normal-ordered fermionic strings and
    coefficients that may be an owning instance referencing a
    :class:`~zixy.container.data.TermData` instance, or a view on a slice of the elements in
    another collection.
    """

    term_type: type[Term[CoeffT]]

    def __init__(self, modes: int | Modes = 0, n: int = 0):
        cmpnts = self.term_type.cmpnts_type(modes, n)
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
            A new instance containing the normal-ordered fermionic strings and coefficients in the
            ``source``.
        """
        return cls._create(cls.term_type.term_data_from_str(source, modes))


class TermSet(TermSetBase[NormalFermionOperatorArray, StringSpec, CoeffT]):
    """A collection of unique terms consisting of normal-ordered fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    set-like operations on them.

    Note:
        Coefficients are mutable in-place, but components are the keys of a hashmap and therefore
        are not.
    """

    terms_type: type[Terms[CoeffT]]

    def __init__(self, modes: int | Modes = 0):
        TermSetBase.__init__(self, self.terms_type(modes))

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


class TermSum(TermSumBase[NormalFermionOperatorArray, StringSpec, CoeffT], TermSet[CoeffT]):
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
            modes: Space of modes or a number of modes. If ``None``, infer from the max mode
                index in the input string.

        Returns:
            A new instance containing the normal-ordered fermionic strings and coefficients in the
            ``source``.
        """
        terms_ = cls.terms_type.from_str(source, modes)
        return cls.from_iterable(terms_, terms_.modes)

    def __mul__(self, rhs: Any) -> Any:
        """Multiplication of ``self`` by ``rhs``."""
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        if isinstance(rhs, Term):
            rhs = type(self).from_iterable((rhs,), self.modes)
        if not isinstance(rhs, TermSum):
            return NotImplemented
        out = type(self)(self.modes)
        for lhs_term in self:
            for rhs_term in rhs:
                out += cast(Term[CoeffT], lhs_term) * cast(Term[CoeffT], rhs_term)
        return out

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        out = type(self)(self.modes)
        for term in cast(Iterable[Term[CoeffT]], self):
            out += term.daggered()
        TermSumBase.__init__(self, out.to_terms())

    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        out = self.clone()
        out.dagger()
        return out

    def to_general(self) -> Any:
        """Convert this normal-ordered term sum to the raw general representation."""
        from zixy.fermion.operator.general import (  # noqa: PLC0415
            ComplexTermSum,
            Strings as GeneralStrings,
        )

        method = (
            self.strings._impl.lincomb_to_general_real
            if self.terms_type.term_type.coeff_type is float
            else self.strings._impl.lincomb_to_general_complex
        )
        impl, coeffs = method(self.strings._impl, self._cmpnt_set._map, self.coeffs._impl)
        return ComplexTermSum._create(
            TermData(
                GeneralStrings._create(impl),
                ComplexCoeffs._create(coeffs),
            )
        )


class SignTerm(Term[Sign]):
    """A term consisting of a normal-ordered fermionic string and a sign coefficient."""

    coeff_type = Sign


class SignTerms(Terms[Sign]):
    """A collection of terms with normal-ordered fermionic strings and sign coefficients."""

    term_type = SignTerm


class SignTermSet(TermSet[Sign]):
    """A collection of unique terms with normal-ordered fermionic strings and sign coefficients."""

    terms_type = SignTerms


class SignTermSum(TermSum[Sign]):
    """A sum of terms consisting of normal-ordered fermionic strings and sign coefficients."""

    terms_type = SignTerms


class RealTerm(Term[float]):
    """A term consisting of a normal-ordered fermionic string and a real coefficient."""

    coeff_type = float


class RealTerms(NumericTerms[NormalFermionOperatorArray, StringSpec, float], Terms[float]):
    """A collection of terms with normal-ordered fermionic strings and real coefficients."""

    term_type = RealTerm


class RealTermSet(TermSet[float]):
    """A collection of unique terms with normal-ordered fermionic strings and real coefficients."""

    terms_type = RealTerms


class RealTermSum(NumericTermSum[NormalFermionOperatorArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of normal-ordered fermionic strings and real coefficients."""

    terms_type = RealTerms

    @staticmethod
    def _complex_from_impls(impl: NormalFermionOperatorArray, coeffs: Any) -> ComplexTermSum:
        return ComplexTermSum._create(
            TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        )

    def __mul__(self, rhs: Any) -> Any:
        """Multiplication of ``self`` by ``rhs``."""
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        if isinstance(rhs, Term):
            rhs = type(self).from_iterable((rhs,), self.modes)
        if not isinstance(rhs, TermSum):
            return NotImplemented
        impl, coeffs = self.strings._impl.lincomb_mul_real(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        return self._complex_from_impls(impl, coeffs)

    def commutator(self, rhs: TermSum[float]) -> ComplexTermSum:
        """Return the commutator ``[self, rhs]``."""
        impl, coeffs = self.strings._impl.lincomb_commutator_real(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        return self._complex_from_impls(impl, coeffs)

    def anticommutator(self, rhs: TermSum[float]) -> ComplexTermSum:
        """Return the anticommutator ``{self, rhs}``."""
        impl, coeffs = self.strings._impl.lincomb_anticommutator_real(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        return self._complex_from_impls(impl, coeffs)

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

    def to_qubit(self, mapper: Any) -> Any:
        """Map this fermionic term sum to a qubit Pauli term sum."""
        from zixy.qubit.pauli import (  # noqa: PLC0415
            RealTerm as PauliRealTerm,
            RealTermSum as PauliRealTermSum,
            String as PauliString,
        )

        out = PauliRealTermSum(mapper.qubits)
        for term in self:
            term = cast(RealTerm, term)
            if term.string.get_sets() == ([], []):
                out += PauliRealTerm.from_cmpnt_coeff(PauliString(mapper.qubits), term.coeff)
            else:
                out += mapper.encode(term.string) * term.coeff
        return out

    def to_ladder_ops(self) -> list[tuple[list[tuple[int, bool]], float]]:
        """Return the terms as raw ladder-operator products and coefficients."""
        return [(_string_to_ladder_ops(cast(RealTerm, term).string), term.coeff) for term in self]

    def apply(self, state: RealState) -> ComplexState:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        out = ComplexState(self.modes)
        assert isinstance(self._impl._coeffs, RealCoeffs)
        assert isinstance(state._impl._coeffs, RealCoeffs)
        assert isinstance(out._impl._coeffs, ComplexCoeffs)
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
    """A term consisting of a normal-ordered fermionic string and a complex coefficient."""

    coeff_type = complex


class ComplexTerms(NumericTerms[NormalFermionOperatorArray, StringSpec, complex], Terms[complex]):
    """A collection of terms with normal-ordered fermionic strings and complex coefficients."""

    term_type = ComplexTerm


class ComplexTermSet(TermSet[complex]):
    """A collection of unique terms with normal-ordered strings and complex coefficients."""

    terms_type = ComplexTerms


class ComplexTermSum(
    NumericTermSum[NormalFermionOperatorArray, StringSpec, complex], TermSum[complex]
):
    """A sum of terms consisting of normal-ordered fermionic strings and complex coefficients."""

    terms_type = ComplexTerms

    @staticmethod
    def _from_impls(impl: NormalFermionOperatorArray, coeffs: Any) -> ComplexTermSum:
        return ComplexTermSum._create(
            TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        )

    def __mul__(self, rhs: Any) -> Any:
        """Multiplication of ``self`` by ``rhs``."""
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        if isinstance(rhs, Term):
            rhs = type(self).from_iterable((rhs,), self.modes)
        if not isinstance(rhs, TermSum):
            return NotImplemented
        impl, coeffs = self.strings._impl.lincomb_mul_complex(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        return self._from_impls(impl, coeffs)

    def commutator(self, rhs: TermSum[complex]) -> ComplexTermSum:
        """Return the commutator ``[self, rhs]``."""
        impl, coeffs = self.strings._impl.lincomb_commutator_complex(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        return self._from_impls(impl, coeffs)

    def anticommutator(self, rhs: TermSum[complex]) -> ComplexTermSum:
        """Return the anticommutator ``{self, rhs}``."""
        impl, coeffs = self.strings._impl.lincomb_anticommutator_complex(
            self.strings._impl, self.coeffs._impl, rhs.strings._impl, rhs.coeffs._impl
        )
        return self._from_impls(impl, coeffs)

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

    def apply(self, state: ComplexState) -> ComplexState:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
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
    """A term consisting of a normal-ordered fermionic string and a symbolic coefficient."""

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
    """A collection of terms with normal-ordered fermionic strings and symbolic coefficients."""

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
    """A collection of unique terms with normal-ordered strings and symbolic coefficients."""

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
    """A sum of terms consisting of normal-ordered fermionic strings and symbolic coefficients."""

    terms_type = SymbolicTerms

    def isubs(self, values: dict[Symbol | str, Number | Expr]) -> None:
        """Substitute values into the symbolic coefficients in-place."""
        self.coeffs.isubs(values)

    def subs(self, values: dict[Symbol | str, Number | Expr]) -> SymbolicTermSum:
        """Return a copy with values substituted into the symbolic coefficients."""
        out = self.clone()
        out.isubs(values)
        return out


def _string_to_ladder_ops(string: String) -> list[tuple[int, bool]]:
    cre, ann = string.get_sets()
    return [(i, True) for i in cre] + [(i, False) for i in ann]


class TermRegistry(terms.TermRegistry[NormalFermionOperatorArray, StringSpec]):
    term_type_sign: type[SignTerm]
    term_type_complex_sign: type[Any]
    term_type_real: type[RealTerm]
    term_type_complex: type[ComplexTerm]
    term_type_symbolic: type[SymbolicTerm]

    def __init__(
        self,
        term_type_sign: type[SignTerm],
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
    term_type_sign=SignTerm,
    term_type_complex_sign=ComplexTerm,  # fermion signs are real; promote complex signs to complex
    term_type_real=RealTerm,
    term_type_complex=ComplexTerm,
    term_type_symbolic=SymbolicTerm,
)

for _term_type, _sum_type in (
    (SignTerm, SignTermSum),
    (RealTerm, RealTermSum),
    (ComplexTerm, ComplexTermSum),
    (SymbolicTerm, SymbolicTermSum),
):
    setattr(_term_type, "_term_sum_type", _sum_type)
