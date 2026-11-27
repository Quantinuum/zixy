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

"""Term sums for normal-ordered fermionic operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, overload

from sympy import Expr

import zixy.fermion.state as zfs
from zixy._zixy import NormalFermionOperatorArray
from zixy.container.coeff import (
    Coeff,
    ComplexSign,
    Sign,
    _integer_exponent,
    normalize_scalar,
    typesafe_mul,
)
from zixy.container.coeffs import Coeffs, ComplexCoeffs, RealCoeffs
from zixy.container.data import TermData
from zixy.container.term_sum import (
    NumericTermSum,
    SymbolicTermSum as SymbolicTermSumBase,
    TermSum as BaseTermSum,
)
from zixy.container.utils import _positive_power
from zixy.fermion._string import _check_modes_compatibility
from zixy.fermion.operator._term_sum import TermSum as OperatorTermSum
from zixy.fermion.operator.normal._string import ElemT, ImplT, SpecT, StringSpec
from zixy.fermion.operator.normal._strings import Strings
from zixy.fermion.operator.normal._term import SymbolicTerm
from zixy.fermion.operator.normal._term_set import TermSet
from zixy.fermion.operator.normal._terms import ComplexTerms, RealTerms, SymbolicTerms

if TYPE_CHECKING:
    import numpy as np

    import zixy.fermion.operator.general as zfog
    from zixy._zixy import Modes
    from zixy.fermion.operator.normal._string import String

type NumericState = zfs.RealTermSum | zfs.ComplexTermSum

type State = NumericState | zfs.SymbolicTermSum


def _new_product_sum(base_coeff: Coeff, modes: Modes) -> TermSum[Any]:
    """Create an empty normal-ordered sum for the product coefficient."""
    if isinstance(base_coeff, Sign | int | float):
        return RealTermSum(modes)
    if isinstance(base_coeff, ComplexSign | complex):
        return ComplexTermSum(modes)
    if isinstance(base_coeff, Expr):
        return SymbolicTermSum(modes)
    raise TypeError(f"Unsupported coefficient type {type(base_coeff)}.")


def _term_sum_from_product(
    modes: Modes,
    lhs: String,
    rhs: String,
    base_coeff: Any,
) -> TermSum[Any]:
    """Compute the product of two normal-ordered strings."""
    out = _new_product_sum(base_coeff, modes)
    impl, signs = lhs._impl.string_mul(lhs.index, rhs._impl, rhs.index)
    strings = Strings._create(impl)
    for i in range(len(impl)):
        coeff = typesafe_mul(base_coeff, Sign(signs[i]))
        term = out.terms_type.term_type.from_string(strings[i], coeff)
        out += term
    return out


class TermSum[C: (float, complex, Expr)](OperatorTermSum[ImplT, SpecT, C, ElemT], TermSet[C]):
    """A sum of terms consisting of normal-ordered fermionic strings and coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    linear combination operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    def dagger(self) -> None:
        """Take the adjoint of ``self`` in-place."""
        out = type(self)(self.modes)
        for term in self:
            out += term.into(self.terms_type.term_type).daggered()
        BaseTermSum.__init__(self, out.to_terms())

    def daggered(self) -> Self:
        """Return the adjoint of ``self``."""
        out = self.clone()
        out.dagger()
        return out


class RealTermSum(NumericTermSum[NormalFermionOperatorArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of normal-ordered fermionic strings and real coefficients.

    A set-like container of mode-based terms with ``float`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms

    def is_hermitian(self, atol: float = 1e-10) -> bool:
        """Check whether ``self`` is Hermitian within the given tolerance."""
        return self._string_set._impl.lincomb_is_hermitian_real(
            self._string_set._impl, self.coeffs._impl, atol
        )

    @overload  # type: ignore[override]
    def __mul__(self, rhs: Coeff | Coeffs[float]) -> Self: ...
    @overload
    def __mul__(self, rhs: Self) -> RealTermSum: ...

    def __mul__(self, rhs: Coeff | Coeffs[float] | Self) -> Self | RealTermSum:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        elif not isinstance(rhs, RealTermSum):
            return NotImplemented
        assert isinstance(self._impl.coeffs, RealCoeffs)
        assert isinstance(rhs._impl.coeffs, RealCoeffs)
        lhs_impl = self._impl.strings._impl
        lhs_coeffs = self._impl.coeffs._impl
        rhs_impl = rhs._impl.strings._impl
        rhs_coeffs = rhs._impl.coeffs._impl
        # TODO: support output by reference
        impl, coeffs, map_ = lhs_impl.lincomb_mul_real(lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs)
        data = TermData(Strings._create(impl), RealCoeffs._create(coeffs))
        return RealTermSum._create(data, map_=map_)

    def __pow__(self, exponent: int | np.integer[Any]) -> RealTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, RealTermSum.__mul__)

    def commutator(self, rhs: TermSum[float]) -> RealTermSum:
        """Return the commutator ``[self, rhs]``."""
        rhs_coeffs = rhs.coeffs
        assert isinstance(rhs_coeffs, RealCoeffs)
        impl, coeffs, map_ = self._string_set._impl.lincomb_commutator_real(
            self._string_set._impl, self.coeffs._impl, rhs._string_set._impl, rhs_coeffs._impl
        )
        data = TermData(Strings._create(impl), RealCoeffs._create(coeffs))
        return RealTermSum._create(data, map_=map_)

    def anticommutator(self, rhs: TermSum[float]) -> RealTermSum:
        """Return the anticommutator ``{self, rhs}``."""
        rhs_coeffs = rhs.coeffs
        assert isinstance(rhs_coeffs, RealCoeffs)
        impl, coeffs, map_ = self._string_set._impl.lincomb_anticommutator_real(
            self._string_set._impl, self.coeffs._impl, rhs._string_set._impl, rhs_coeffs._impl
        )
        data = TermData(Strings._create(impl), RealCoeffs._create(coeffs))
        return RealTermSum._create(data, map_=map_)

    @overload
    def mat_elem(self, bra: zfs.RealTermSum, ket: zfs.RealTermSum) -> float: ...
    @overload
    def mat_elem(self, bra: zfs.ComplexTermSum, ket: NumericState) -> complex: ...
    @overload
    def mat_elem(self, bra: zfs.RealTermSum, ket: zfs.ComplexTermSum) -> complex: ...
    @overload
    def mat_elem(self, bra: zfs.SymbolicTermSum, ket: State) -> Expr: ...
    @overload
    def mat_elem(self, bra: NumericState, ket: zfs.SymbolicTermSum) -> Expr: ...

    def mat_elem(self, bra: State, ket: State) -> float | complex | Expr:
        """Evaluate the matrix element of ``self`` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting matrix element.
        """
        _check_modes_compatibility(self.modes, bra.modes, ket.modes)
        if isinstance(bra, zfs.SymbolicTermSum) or isinstance(ket, zfs.SymbolicTermSum):
            return self.into(SymbolicTermSum).mat_elem(bra, ket)
        if isinstance(bra, zfs.ComplexTermSum) or isinstance(ket, zfs.ComplexTermSum):
            return self.into(ComplexTermSum).mat_elem(bra, ket)
        assert isinstance(self._impl.coeffs, RealCoeffs)
        assert isinstance(bra._impl.coeffs, RealCoeffs)
        assert isinstance(ket._impl.coeffs, RealCoeffs)
        return float(
            self._impl.strings._impl.mat_elem_real(
                self._impl.coeffs._impl,
                bra._impl.strings._impl,
                bra._impl.coeffs._impl,
                ket._impl.strings._impl,
                ket._impl.coeffs._impl,
            )
        )

    @overload
    def exp_val(self, state: zfs.RealTermSum) -> float: ...
    @overload
    def exp_val(self, state: zfs.ComplexTermSum) -> complex: ...
    @overload
    def exp_val(self, state: zfs.SymbolicTermSum) -> Expr: ...

    def exp_val(self, state: State) -> float | complex | Expr:
        """Evaluate the expectation value of ``self`` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting expectation value.
        """
        return self.mat_elem(state, state)

    @overload
    def apply(self, state: zfs.RealTermSum) -> zfs.RealTermSum: ...
    @overload
    def apply(self, state: zfs.ComplexTermSum) -> zfs.ComplexTermSum: ...
    @overload
    def apply(self, state: zfs.SymbolicTermSum) -> zfs.SymbolicTermSum: ...

    def apply(self, state: State) -> State:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        _check_modes_compatibility(self.modes, state.modes)
        if isinstance(state, zfs.ComplexTermSum):
            return self.into(ComplexTermSum).apply(state)
        if isinstance(state, zfs.SymbolicTermSum):
            return self.into(SymbolicTermSum).apply(state)
        out = zfs.RealTermSum(self.modes)
        assert isinstance(self._impl.coeffs, RealCoeffs)
        assert isinstance(state._impl.coeffs, RealCoeffs)
        assert isinstance(out._impl.coeffs, RealCoeffs)
        self._impl.strings._impl.apply_to_state_real(
            self._impl.coeffs._impl,
            state._impl.strings._impl,
            state._impl.coeffs._impl,
            out._impl.strings._impl,
            out._string_set._map,
            out._impl.coeffs._impl,
        )
        return out.filter_nonzero()

    def active_modes(self) -> set[int]:
        """Return the set of mode indices on which ``self`` acts."""
        return self._string_set._impl.lincomb_active_modes_real(
            self._string_set._impl, self.coeffs._impl
        )

    def max_n_body(self) -> int:
        """Return the maximum n-body order across all terms."""
        return self._string_set._impl.lincomb_max_n_body_real(
            self._string_set._impl, self.coeffs._impl
        )

    def conserves_particle_number(self, atol: float = 1e-10) -> bool:
        """Check whether ``self`` conserves particle number within the given tolerance."""
        return self._string_set._impl.lincomb_conserves_particle_number_real(
            self._string_set._impl, self.coeffs._impl, atol
        )

    def to_general(self) -> zfog.RealTermSum:
        """Convert this normal-ordered term sum to the raw general representation."""
        import zixy.fermion.operator.general as zfog  # noqa: PLC0415

        impl, coeffs, map_ = self._string_set._impl.lincomb_to_general_real(
            self._string_set._impl, self._string_set._map, self.coeffs._impl
        )
        term_data = TermData(
            zfog.Strings._create(impl),
            RealCoeffs._create(coeffs),
        )
        return zfog.RealTermSum._create(term_data, map_=map_)


class ComplexTermSum(
    NumericTermSum[NormalFermionOperatorArray, StringSpec, complex], TermSum[complex]
):
    """A sum of terms consisting of normal-ordered fermionic strings and complex coefficients.

    A set-like container of mode-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms

    def is_hermitian(self, atol: float = 1e-10) -> bool:
        """Check whether ``self`` is Hermitian within the given tolerance."""
        return self._string_set._impl.lincomb_is_hermitian_complex(
            self._string_set._impl, self.coeffs._impl, atol
        )

    @overload  # type: ignore[override]
    def __mul__(self, rhs: Coeff | Coeffs[complex]) -> Self: ...
    @overload
    def __mul__(self, rhs: Self) -> ComplexTermSum: ...

    def __mul__(self, rhs: Coeff | Coeffs[complex] | Self) -> Self | ComplexTermSum:
        """Multiplication of ``self`` by ``rhs``."""
        rhs = normalize_scalar(rhs)
        if isinstance(rhs, Coeff | Coeffs):
            return super().__mul__(rhs)
        elif not isinstance(rhs, ComplexTermSum):
            return NotImplemented
        assert isinstance(self._impl.coeffs, ComplexCoeffs)
        assert isinstance(rhs._impl.coeffs, ComplexCoeffs)
        lhs_impl = self._impl.strings._impl
        lhs_coeffs = self._impl.coeffs._impl
        rhs_impl = rhs._impl.strings._impl
        rhs_coeffs = rhs._impl.coeffs._impl
        # TODO: support output by reference
        impl, coeffs, map_ = lhs_impl.lincomb_mul_complex(
            lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs
        )
        data = TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        return ComplexTermSum._create(data, map_=map_)

    def __pow__(self, exponent: int | np.integer[Any]) -> ComplexTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, ComplexTermSum.__mul__)

    def commutator(self, rhs: TermSum[complex]) -> ComplexTermSum:
        """Return the commutator ``[self, rhs]``."""
        rhs_coeffs = rhs.coeffs
        assert isinstance(rhs_coeffs, ComplexCoeffs)
        impl, coeffs, map_ = self._string_set._impl.lincomb_commutator_complex(
            self._string_set._impl, self.coeffs._impl, rhs._string_set._impl, rhs_coeffs._impl
        )
        data = TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        return ComplexTermSum._create(data, map_=map_)

    def anticommutator(self, rhs: TermSum[complex]) -> ComplexTermSum:
        """Return the anticommutator ``{self, rhs}``."""
        rhs_coeffs = rhs.coeffs
        assert isinstance(rhs_coeffs, ComplexCoeffs)
        impl, coeffs, map_ = self._string_set._impl.lincomb_anticommutator_complex(
            self._string_set._impl, self.coeffs._impl, rhs._string_set._impl, rhs_coeffs._impl
        )
        data = TermData(Strings._create(impl), ComplexCoeffs._create(coeffs))
        return ComplexTermSum._create(data, map_=map_)

    @overload
    def mat_elem(self, bra: NumericState, ket: NumericState) -> complex: ...
    @overload
    def mat_elem(self, bra: zfs.SymbolicTermSum, ket: State) -> Expr: ...
    @overload
    def mat_elem(self, bra: NumericState, ket: zfs.SymbolicTermSum) -> Expr: ...

    def mat_elem(self, bra: State, ket: State) -> complex | Expr:
        """Evaluate the matrix element of ``self`` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting matrix element.
        """
        _check_modes_compatibility(self.modes, bra.modes, ket.modes)
        if isinstance(bra, zfs.SymbolicTermSum) or isinstance(ket, zfs.SymbolicTermSum):
            return self.into(SymbolicTermSum).mat_elem(bra, ket)
        if isinstance(bra, zfs.RealTermSum):
            bra = bra.into(zfs.ComplexTermSum)
        if isinstance(ket, zfs.RealTermSum):
            ket = ket.into(zfs.ComplexTermSum)
        assert isinstance(self._impl.coeffs, ComplexCoeffs)
        assert isinstance(bra._impl.coeffs, ComplexCoeffs)
        assert isinstance(ket._impl.coeffs, ComplexCoeffs)
        return complex(
            self._impl.strings._impl.mat_elem_complex(
                self._impl.coeffs._impl,
                bra._impl.strings._impl,
                bra._impl.coeffs._impl,
                ket._impl.strings._impl,
                ket._impl.coeffs._impl,
            )
        )

    @overload
    def exp_val(self, state: NumericState) -> complex: ...
    @overload
    def exp_val(self, state: zfs.SymbolicTermSum) -> Expr: ...

    def exp_val(self, state: State) -> complex | Expr:
        """Evaluate the expectation value of ``self`` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting expectation value.
        """
        return self.mat_elem(state, state)

    @overload
    def apply(self, state: NumericState) -> zfs.ComplexTermSum: ...
    @overload
    def apply(self, state: zfs.SymbolicTermSum) -> zfs.SymbolicTermSum: ...

    def apply(self, state: State) -> zfs.ComplexTermSum | zfs.SymbolicTermSum:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        _check_modes_compatibility(self.modes, state.modes)
        if isinstance(state, zfs.RealTermSum):
            state = state.into(zfs.ComplexTermSum)
        elif isinstance(state, zfs.SymbolicTermSum):
            return self.into(SymbolicTermSum).apply(state)
        out = zfs.ComplexTermSum(self.modes)
        assert isinstance(self._impl.coeffs, ComplexCoeffs)
        assert isinstance(state._impl.coeffs, ComplexCoeffs)
        assert isinstance(out._impl.coeffs, ComplexCoeffs)
        self._impl.strings._impl.apply_to_state_complex(
            self._impl.coeffs._impl,
            state._impl.strings._impl,
            state._impl.coeffs._impl,
            out._impl.strings._impl,
            out._string_set._map,
            out._impl.coeffs._impl,
        )
        return out.filter_nonzero()

    def active_modes(self) -> set[int]:
        """Return the set of mode indices on which ``self`` acts."""
        return self._string_set._impl.lincomb_active_modes_complex(
            self._string_set._impl, self.coeffs._impl
        )

    def max_n_body(self) -> int:
        """Return the maximum n-body order across all terms."""
        return self._string_set._impl.lincomb_max_n_body_complex(
            self._string_set._impl, self.coeffs._impl
        )

    def conserves_particle_number(self, atol: float = 1e-10) -> bool:
        """Check whether ``self`` conserves particle number within the given tolerance."""
        return self._string_set._impl.lincomb_conserves_particle_number_complex(
            self._string_set._impl, self.coeffs._impl, atol
        )

    def to_general(self) -> zfog.ComplexTermSum:
        """Convert this normal-ordered term sum to the raw general representation."""
        import zixy.fermion.operator.general as zfog  # noqa: PLC0415

        impl, coeffs, map_ = self._string_set._impl.lincomb_to_general_complex(
            self._string_set._impl, self._string_set._map, self.coeffs._impl
        )
        term_data = TermData(
            zfog.Strings._create(impl),
            ComplexCoeffs._create(coeffs),
        )
        return zfog.ComplexTermSum._create(term_data, map_=map_)


class SymbolicTermSum(SymbolicTermSumBase[ImplT, SpecT], TermSum[Expr]):
    """A sum of terms consisting of normal-ordered fermionic strings and symbolic coefficients.

    A set-like container of mode-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms

    def mat_elem(self, bra: State, ket: State) -> Expr:
        """Evaluate the matrix element of ``self`` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting symbolic matrix element.
        """
        _check_modes_compatibility(self.modes, bra.modes, ket.modes)
        if not isinstance(bra, zfs.SymbolicTermSum):
            bra = bra.into(zfs.SymbolicTermSum)
        if not isinstance(ket, zfs.SymbolicTermSum):
            ket = ket.into(zfs.SymbolicTermSum)
        return bra.vdot(self.apply(ket))

    def exp_val(self, state: State) -> Expr:
        """Evaluate the expectation value of ``self`` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting symbolic expectation value.
        """
        return self.mat_elem(state, state)

    def apply(self, state: State) -> zfs.SymbolicTermSum:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting symbolic state.
        """
        _check_modes_compatibility(self.modes, state.modes)
        if not isinstance(state, zfs.SymbolicTermSum):
            state = state.into(zfs.SymbolicTermSum)
        out = zfs.SymbolicTermSum(self.modes)
        for op_term in self:
            assert isinstance(op_term, SymbolicTerm)
            for state_term in state:
                assert isinstance(state_term, zfs.SymbolicTerm)
                string = state_term.string.clone()
                assert isinstance(string, zfs.String)
                sign = string._impl.string_operator_string_imul(
                    string.index, op_term.string._impl, op_term.string.index
                )
                if sign is not None:
                    out += string * (op_term.coeff * state_term.coeff * Sign(sign).to_symbolic())
        return out.filter_nonzero()
