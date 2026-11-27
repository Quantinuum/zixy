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

"""Term sums for qubit operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

import numpy as np  # noqa: TC002 (runtime type hints)
from sympy import Expr

import zixy.qubit.state as zqs
from zixy._zixy import PauliMatrix, QubitPauliArray
from zixy.container.coeff import Coeff, _integer_exponent, normalize_scalar
from zixy.container.coeffs import Coeffs, ComplexCoeffs, RealCoeffs, SymbolicCoeffs
from zixy.container.data import TermData
from zixy.container.term_sum import NumericTermSum, SymbolicTermSum as SymbolicTermSumBase
from zixy.container.utils import _positive_power
from zixy.qubit._string import _check_qubits_compatibility
from zixy.qubit._term_sum import TermSum as TermSumBase
from zixy.qubit.operator._string import StringSpec
from zixy.qubit.operator._term import ComplexTerm, RealTerm, SymbolicTerm  # noqa: TC001 (runtime type hints)
from zixy.qubit.operator._term_set import TermSet
from zixy.qubit.operator._terms import ComplexTerms, RealTerms, SymbolicTerms
from zixy.utils import DEFAULT_COMMUTES_ATOL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


type NumericState = zqs.RealTermSum | zqs.ComplexTermSum

type State = NumericState | zqs.SymbolicTermSum


class TermSum[C: (float, complex, Expr)](
    TermSumBase[QubitPauliArray, StringSpec, C, PauliMatrix], TermSet[C]
):
    """A sum of terms consisting of Pauli strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    linear combination operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    def commutes_with(self, other: TermSum[C], atol: float = DEFAULT_COMMUTES_ATOL) -> bool:
        """Check whether ``self`` commutes with ``other``.

        .. math::
            [A, B] = 0.

        Args:
            other: The other operand.
            atol: The absolute tolerance.

        Returns:
            Whether ``self`` commutes with ``other``.

        See Also:
            :meth:`commutator`
        """
        coeffs = self.commutator(other)._data.coeffs
        if isinstance(coeffs, RealCoeffs | ComplexCoeffs):
            return not coeffs.any_significant(atol)
        return len(coeffs) == 0

    def commutator(self, other: TermSum[C]) -> TermSum[Any]:
        """Compute the commutator of ``self`` with ``other``.

        .. math::
            [A, B] = AB - BA.
        """
        out = cast("Any", self) * other - cast("Any", other) * self
        return cast("TermSum[Any]", out)

    def conserves_hamming_weight(self, atol: float = DEFAULT_COMMUTES_ATOL) -> bool:
        """Check whether ``self`` conserves Hamming weight.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            Whether ``self`` conserves Hamming weight.

        See Also:
            :meth:`~zixy.qubit.operator.num_ops.create_num_op`
        """
        from .num_ops import create_num_op  # noqa: PLC0415

        return self.commutes_with(
            cast("Any", create_num_op(cast("Any", type(self)), self.qubits)), atol
        )

    def conserves_odd_bit_hamming_weight(self, atol: float = DEFAULT_COMMUTES_ATOL) -> bool:
        """Check whether ``self`` conserves odd-bit Hamming weight.

        Args:
            atol: The absolute tolerance to use.

        Returns:
            Whether ``self`` conserves odd-bit Hamming weight.

        See Also:
            :meth:`~zixy.qubit.operator.num_ops.create_num_op_odd_bits`
        """
        from .num_ops import create_num_op_odd_bits  # noqa: PLC0415

        return self.commutes_with(
            cast("Any", create_num_op_odd_bits(cast("Any", type(self)), self.qubits)), atol
        )


class RealTermSum(NumericTermSum[QubitPauliArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of Pauli strings and real coefficients.

    A set-like container of qubit-based terms with ``float`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms

    def __iadd__(self, rhs: RealTerm | Self) -> Self:  # type: ignore[override,misc]
        """In-place addition of ``self`` by ``rhs``."""
        super().__iadd__(rhs)
        return self

    def __isub__(self, rhs: RealTerm | Self) -> Self:  # type: ignore[override,misc]
        """In-place subtraction of ``self`` by ``rhs``."""
        super().__isub__(rhs)
        return self

    @overload  # type: ignore[override]
    def __mul__(self, other: Coeff | Coeffs[float]) -> Self: ...
    @overload
    def __mul__(self, other: Self) -> ComplexTermSum: ...

    def __mul__(self, other: Self | Coeff | Coeffs[float]) -> Self | ComplexTermSum:
        """Multiplication of ``self`` by ``other``."""
        other = normalize_scalar(other)
        if isinstance(other, Coeff | Coeffs):
            return super().__mul__(other)
        elif not isinstance(other, RealTermSum):
            return NotImplemented
        assert isinstance(self._impl.coeffs, RealCoeffs)
        assert isinstance(other._impl.coeffs, RealCoeffs)
        lhs_impl = self._impl.strings._impl
        lhs_coeffs = self._impl.coeffs._impl
        rhs_impl = other._impl.strings._impl
        rhs_coeffs = other._impl.coeffs._impl
        term_data = TermData(other._impl.strings._new(), ComplexCoeffs())
        out = ComplexTermSum._create(term_data, map_=other._string_set.map_type())
        assert isinstance(out._impl.coeffs, ComplexCoeffs)
        out_impl = out._impl.strings._impl
        out_map = out._string_set._map
        out_coeffs = out._impl.coeffs._impl
        QubitPauliArray.lincomb_mul_real(
            lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs, out_impl, out_map, out_coeffs
        )
        return out

    def __pow__(self, exponent: int | np.integer[Any]) -> Self | ComplexTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self.into(ComplexTermSum), exponent, lambda lhs, rhs: lhs * rhs)

    @overload
    def mat_elem(
        self, bra: zqs.RealTermSum, ket: zqs.RealTermSum, *, real: Literal[True]
    ) -> float: ...
    @overload
    def mat_elem(
        self, bra: NumericState, ket: NumericState, *, real: Literal[False] = False
    ) -> complex: ...
    @overload
    def mat_elem(
        self, bra: zqs.RealTermSum, ket: zqs.RealTermSum, *, real: bool
    ) -> float | complex: ...
    @overload
    def mat_elem(
        self, bra: zqs.SymbolicTermSum, ket: State, *, real: Literal[False] = False
    ) -> Expr: ...
    @overload
    def mat_elem(
        self, bra: NumericState, ket: zqs.SymbolicTermSum, *, real: Literal[False] = False
    ) -> Expr: ...

    def mat_elem(self, bra: State, ket: State, *, real: bool = False) -> float | complex | Expr:
        """Evaluate the matrix element of ``self`` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.
            real: Use the optimized real-valued implementation. The bra and ket state must be
                real, and the resulting matrix element is therein assumed to be real,
                irrespective of the operator.

        Returns:
            The resulting matrix element.

        Raises:
            TypeError: If ``real`` is ``True`` and either state has non-real coefficients.
        """
        _check_qubits_compatibility(self.qubits, bra.qubits, ket.qubits)
        if real:
            if not isinstance(bra, zqs.RealTermSum) or not isinstance(ket, zqs.RealTermSum):
                raise TypeError("real=True requires real bra and ket states.")
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
        if isinstance(bra, zqs.SymbolicTermSum) or isinstance(ket, zqs.SymbolicTermSum):
            return self.into(SymbolicTermSum).mat_elem(bra, ket)
        return self.into(ComplexTermSum).mat_elem(bra, ket)

    @overload
    def exp_val(self, state: zqs.RealTermSum, *, real: Literal[True]) -> float: ...
    @overload
    def exp_val(self, state: NumericState, *, real: Literal[False] = False) -> complex: ...
    @overload
    def exp_val(self, state: zqs.RealTermSum, *, real: bool) -> float | complex: ...
    @overload
    def exp_val(self, state: zqs.SymbolicTermSum, *, real: Literal[False] = False) -> Expr: ...

    def exp_val(self, state: State, *, real: bool = False) -> float | complex | Expr:
        """Evaluate the expectation value of ``self`` with respect to a state.

        Args:
            state: The state to evaluate with respect to.
            real: If ``True``, use the optimized real-valued implementation and return a
                ``float``. This is supported only for states with real coefficients. The default
                is ``False``, which uses the complex-valued implementation.

        Returns:
            The resulting expectation value.

        Raises:
            TypeError: If ``real`` is ``True`` and the state has non-real coefficients.
        """
        if real:
            if not isinstance(state, zqs.RealTermSum):
                raise TypeError("real=True requires a real state.")
            return self.mat_elem(state, state, real=True)
        return self.mat_elem(state, state)

    @overload
    def apply(self, state: NumericState) -> zqs.ComplexTermSum: ...
    @overload
    def apply(self, state: zqs.SymbolicTermSum) -> zqs.SymbolicTermSum: ...

    def apply(self, state: State) -> zqs.ComplexTermSum | zqs.SymbolicTermSum:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        _check_qubits_compatibility(self.qubits, state.qubits)
        if isinstance(state, zqs.ComplexTermSum):
            return self.into(ComplexTermSum).apply(state)
        if isinstance(state, zqs.SymbolicTermSum):
            return self.into(SymbolicTermSum).apply(state)
        out = zqs.ComplexTermSum(self.qubits)
        assert isinstance(self._impl.coeffs, RealCoeffs)
        assert isinstance(state._impl.coeffs, RealCoeffs)
        assert isinstance(out._impl.coeffs, ComplexCoeffs)
        self._impl.strings._impl.apply_to_state_real(
            self._impl.coeffs._impl,
            state._impl.strings._impl,
            state._impl.coeffs._impl,
            out._impl.strings._impl,
            out._string_set._map,
            out._impl.coeffs._impl,
        )
        return out.filter_nonzero()

    def project_into_ortho_subspace(self, subspace: zqs.Strings) -> NDArray[np.float64]:
        """Project the term into an orthogonal subspace defined by an ordered sequence of states.

        Args:
            subspace: The subspace to project into.

        Returns:
            The resulting vector of coefficients in the subspace basis.
        """
        assert isinstance(self._impl.coeffs, RealCoeffs)
        return QubitPauliArray.lincomb_project_into_ortho_subspace_real(
            self._impl.strings._impl,
            self._impl.coeffs._impl,
            subspace._impl,
        )

    def project_into_nonortho_subspace(
        self, subspace: Sequence[zqs.RealTermSum]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Project the term into a non-orthogonal subspace defined by an ordered sequence of states.

        Args:
            subspace: The subspace to project into.

        Returns:
            A tuple containing the resulting vector of coefficients in the subspace basis and the
            overlap matrix of the subspace states.
        """
        assert isinstance(self._impl.coeffs, RealCoeffs)
        strings = tuple(item._string_set._impl for item in subspace)
        maps = tuple(item._string_set._map for item in subspace)
        coeffs = []
        for item in subspace:
            assert isinstance(item._impl.coeffs, RealCoeffs)
            coeffs.append(item._impl.coeffs._impl)
        return QubitPauliArray.lincomb_project_into_nonortho_subspace_real(
            self._impl.strings._impl,
            self._impl.coeffs._impl,
            strings,
            maps,
            tuple(coeffs),
        )

    def to_sparse_matrix(self, big_endian: bool = False) -> Any:
        """Convert ``self`` to a sparse matrix.

        Args:
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The resulting sparse matrix.
        """
        assert isinstance(self._impl.coeffs, RealCoeffs)
        return QubitPauliArray.lincomb_to_sparse_real(
            self._impl.strings._impl,
            self._impl.coeffs._impl,
            big_endian,
        )


class ComplexTermSum(NumericTermSum[QubitPauliArray, StringSpec, complex], TermSum[complex]):
    """A sum of terms consisting of Pauli strings and complex coefficients.

    A set-like container of qubit-based terms with ``complex`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = ComplexTerms

    @property
    def real_part(self) -> RealTermSum:
        """The real part of ``self``."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)
        data = TermData(self._data.strings.clone(), self._data.coeffs.real_part)
        return RealTermSum._create(data, map_=self._string_set._map.clone())

    @property
    def imag_part(self) -> RealTermSum:
        """The imaginary part of ``self``."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)
        data = TermData(self._data.strings.clone(), self._data.coeffs.imag_part)
        return RealTermSum._create(data, map_=self._string_set._map.clone())

    def __iadd__(self, rhs: ComplexTerm | Self) -> Self:  # type: ignore[override,misc]
        """In-place addition of ``self`` by ``rhs``."""
        super().__iadd__(rhs)
        return self

    def __isub__(self, rhs: ComplexTerm | Self) -> Self:  # type: ignore[override,misc]
        """In-place subtraction of ``self`` by ``rhs``."""
        super().__isub__(rhs)
        return self

    @overload  # type: ignore[override]
    def __mul__(self, other: Coeff | Coeffs[complex]) -> Self: ...
    @overload
    def __mul__(self, other: Self) -> ComplexTermSum: ...

    def __mul__(self, other: Self | Coeff | Coeffs[complex]) -> Self | ComplexTermSum:
        """Multiplication of ``self`` by ``other``."""
        other = normalize_scalar(other)
        if isinstance(other, Coeff | Coeffs):
            return super().__mul__(other)
        elif not isinstance(other, ComplexTermSum):
            return NotImplemented
        assert isinstance(self._impl.coeffs, ComplexCoeffs)
        assert isinstance(other._impl.coeffs, ComplexCoeffs)
        lhs_impl = self._impl.strings._impl
        lhs_coeffs = self._impl.coeffs._impl
        rhs_impl = other._impl.strings._impl
        rhs_coeffs = other._impl.coeffs._impl
        term_data = TermData(other._impl.strings._new(), ComplexCoeffs())
        out = ComplexTermSum._create(term_data, map_=other._string_set.map_type())
        assert isinstance(out._impl.coeffs, ComplexCoeffs)
        out_impl = out._impl.strings._impl
        out_map = out._string_set._map
        out_coeffs = out._impl.coeffs._impl
        QubitPauliArray.lincomb_mul_complex(
            lhs_impl, lhs_coeffs, rhs_impl, rhs_coeffs, out_impl, out_map, out_coeffs
        )
        return out

    def __pow__(self, exponent: int | np.integer[Any]) -> ComplexTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, ComplexTermSum.__mul__)

    @overload
    def mat_elem(self, bra: NumericState, ket: NumericState) -> complex: ...
    @overload
    def mat_elem(self, bra: zqs.SymbolicTermSum, ket: State) -> Expr: ...
    @overload
    def mat_elem(self, bra: NumericState, ket: zqs.SymbolicTermSum) -> Expr: ...

    def mat_elem(self, bra: State, ket: State) -> complex | Expr:
        """Evaluate the matrix element of ``self`` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting matrix element.
        """
        _check_qubits_compatibility(self.qubits, bra.qubits, ket.qubits)
        if isinstance(bra, zqs.SymbolicTermSum) or isinstance(ket, zqs.SymbolicTermSum):
            return self.into(SymbolicTermSum).mat_elem(bra, ket)
        if isinstance(bra, zqs.RealTermSum):
            bra = bra.into(zqs.ComplexTermSum)
        if isinstance(ket, zqs.RealTermSum):
            ket = ket.into(zqs.ComplexTermSum)
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
    def exp_val(self, state: zqs.SymbolicTermSum) -> Expr: ...

    def exp_val(self, state: State) -> complex | Expr:
        """Evaluate the expectation value of ``self`` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting expectation value.
        """
        return self.mat_elem(state, state)

    @overload
    def apply(self, state: NumericState) -> zqs.ComplexTermSum: ...
    @overload
    def apply(self, state: zqs.SymbolicTermSum) -> zqs.SymbolicTermSum: ...

    def apply(self, state: State) -> zqs.ComplexTermSum | zqs.SymbolicTermSum:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting state.
        """
        _check_qubits_compatibility(self.qubits, state.qubits)
        if isinstance(state, zqs.RealTermSum):
            state = state.into(zqs.ComplexTermSum)
        elif isinstance(state, zqs.SymbolicTermSum):
            return self.into(SymbolicTermSum).apply(state)
        out = zqs.ComplexTermSum(self.qubits)
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

    def project_into_ortho_subspace(self, subspace: zqs.Strings) -> NDArray[np.complex128]:
        """Project the term into an orthogonal subspace defined by a set of states.

        Args:
            subspace: The subspace to project into.

        Returns:
            The resulting vector of coefficients in the subspace basis.
        """
        assert isinstance(self._impl.coeffs, ComplexCoeffs)
        return QubitPauliArray.lincomb_project_into_ortho_subspace_complex(
            self._impl.strings._impl,
            self._impl.coeffs._impl,
            subspace._impl,
        )

    def to_sparse_matrix(self, big_endian: bool = False) -> Any:
        """Convert ``self`` to a sparse matrix.

        Args:
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The resulting sparse matrix.
        """
        assert isinstance(self._impl.coeffs, ComplexCoeffs)
        return QubitPauliArray.lincomb_to_sparse_complex(
            self._impl.strings._impl,
            self._impl.coeffs._impl,
            big_endian,
        )


class SymbolicTermSum(SymbolicTermSumBase[QubitPauliArray, StringSpec], TermSum[Expr]):
    """A sum of terms consisting of Pauli strings and symbolic coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms

    @overload  # type: ignore[override]
    def __mul__(self, other: Coeff) -> Self: ...
    @overload
    def __mul__(self, other: Self) -> Self: ...
    @overload
    def __mul__(self, other: Coeffs[Expr]) -> Self: ...

    def __mul__(self, other: Self | Coeff | Coeffs[Expr]) -> Self | SymbolicTermSum:
        """Multiplication of ``self`` by ``other``."""
        other = normalize_scalar(other)
        if isinstance(other, Coeff | Coeffs):
            return super().__mul__(other)
        elif not isinstance(other, SymbolicTermSum):
            return NotImplemented
        term_data = TermData(other._impl.strings._new(), SymbolicCoeffs())
        out = SymbolicTermSum._create(term_data, map_=other._string_set.map_type())
        for l_term in self:
            assert isinstance(l_term, SymbolicTerm)
            for r_term in other:
                assert isinstance(r_term, SymbolicTerm)
                product = l_term * r_term
                product.coeff = product.coeff.simplify()
                out += product
        return out

    def __pow__(self, exponent: int | np.integer[Any]) -> SymbolicTermSum:
        """Return the operator raised to a nonnegative integer power by multiplication."""
        exponent = _integer_exponent(exponent)
        if exponent == 0:
            return self._identity()
        if exponent == 1:
            return self.clone()
        return _positive_power(self, exponent, SymbolicTermSum.__mul__)

    def mat_elem(self, bra: State, ket: State) -> Expr:
        """Evaluate the matrix element of ``self`` between a bra and ket state.

        Args:
            bra: The bra state.
            ket: The ket state.

        Returns:
            The resulting symbolic matrix element.
        """
        _check_qubits_compatibility(self.qubits, bra.qubits, ket.qubits)
        if not isinstance(bra, zqs.SymbolicTermSum):
            bra = bra.into(zqs.SymbolicTermSum)
        if not isinstance(ket, zqs.SymbolicTermSum):
            ket = ket.into(zqs.SymbolicTermSum)
        return bra.vdot(self.apply(ket))

    def exp_val(self, state: State) -> Expr:
        """Evaluate the expectation value of ``self`` with respect to a state.

        Args:
            state: The state to evaluate with respect to.

        Returns:
            The resulting symbolic expectation value.
        """
        return self.mat_elem(state, state)

    def apply(self, state: State) -> zqs.SymbolicTermSum:
        """Apply ``self`` to a state.

        Args:
            state: The state to apply to.

        Returns:
            The resulting symbolic state.
        """
        _check_qubits_compatibility(self.qubits, state.qubits)
        if not isinstance(state, zqs.SymbolicTermSum):
            state = state.into(zqs.SymbolicTermSum)
        out = zqs.SymbolicTermSum(self.qubits)
        for op_term in self:
            assert isinstance(op_term, SymbolicTerm)
            for state_term in state:
                assert isinstance(state_term, zqs.SymbolicTerm)
                string = state_term.string.clone()
                phase = string.imul_get_phase(op_term.string)
                out += string * (op_term.coeff * state_term.coeff * phase.to_symbolic())
        return out.filter_nonzero()
