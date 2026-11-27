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

"""Term sums for qubit states."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from sympy import Expr

from zixy._zixy import Qubits, QubitStateArray
from zixy.container.coeffs import ComplexCoeffs, RealCoeffs
from zixy.container.data import TermData
from zixy.container.term_sum import NumericTermSum, SymbolicTermSum as SymbolicTermSumBase
from zixy.qubit._string import _check_qubits_compatibility
from zixy.qubit._term_sum import TermSum as TermSumBase
from zixy.qubit.state._string import StringSpec
from zixy.qubit.state._term_set import TermSet
from zixy.qubit.state._terms import ComplexTerms, RealTerms, SymbolicTerms

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class TermSum[C: (float, complex, Expr)](
    TermSumBase[QubitStateArray, StringSpec, C, bool], TermSet[C]
):
    """A sum of terms consisting of state strings and coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """


class RealTermSum(NumericTermSum[QubitStateArray, StringSpec, float], TermSum[float]):
    """A sum of terms consisting of state strings and real coefficients.

    A set-like container of qubit-based terms with ``float`` coefficients that may be used to
    store unique terms and perform algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = RealTerms

    def vdot(self, rhs: RealTermSum) -> float:
        """Compute the inner product of ``self`` with ``rhs``."""
        _check_qubits_compatibility(self.qubits, rhs.qubits)
        assert isinstance(self._impl.coeffs, RealCoeffs)  # TODO: resolve
        assert isinstance(rhs._impl.coeffs, RealCoeffs)  # TODO: resolve
        return float(
            self._impl.strings._impl.vdot_real(
                self._string_set._map,
                self._impl.coeffs._impl,
                rhs._impl.strings._impl,
                rhs._impl.coeffs._impl,
            )
        )

    @classmethod
    def from_dense(
        cls,
        qubits: int | Qubits | None = None,
        source: Sequence[float] = tuple(),
        big_endian: bool = False,
    ) -> Self:
        """Create an instance of ``cls`` from a dense vector.

        Args:
            qubits: The qubit register or qubit count. If ``None``, the qubit register is
                inferred from the dense vector length.
            source: The vector to read from.
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The constructed instance.
        """
        if qubits is None:
            qubits = Qubits.from_hilbert_space_dim(len(source))
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        out = cls(qubits)
        assert isinstance(out._impl.coeffs, RealCoeffs)  # TODO: resolve
        out._impl.strings._impl.from_dense_real(
            out._string_set._map, out._impl.coeffs._impl, list(source), big_endian
        )
        return out

    def to_dense(self, big_endian: bool = False) -> NDArray[np.float64]:
        """Convert ``self`` to a dense vector.

        Args:
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The dense vector.
        """
        assert isinstance(self._impl.coeffs, RealCoeffs)  # TODO: resolve
        return self._impl.strings._impl.to_dense_real(self._impl.coeffs._impl, big_endian)


class ComplexTermSum(NumericTermSum[QubitStateArray, StringSpec, complex], TermSum[complex]):
    """A sum of terms consisting of state strings and complex coefficients.

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
        assert isinstance(self._data.coeffs, ComplexCoeffs)  # TODO: resolve
        data = TermData(self._data.strings.clone(), self._data.coeffs.real_part)
        return RealTermSum._create(data, map_=self._string_set._map.clone())

    @property
    def imag_part(self) -> RealTermSum:
        """The imaginary part of ``self``."""
        assert isinstance(self._data.coeffs, ComplexCoeffs)  # TODO: resolve
        data = TermData(self._data.strings.clone(), self._data.coeffs.imag_part)
        return RealTermSum._create(data, map_=self._string_set._map.clone())

    def vdot(self, rhs: ComplexTermSum) -> complex:
        """Compute the inner product of ``self`` with ``rhs``."""
        _check_qubits_compatibility(self.qubits, rhs.qubits)
        assert isinstance(self._impl.coeffs, ComplexCoeffs)  # TODO: resolve
        assert isinstance(rhs._impl.coeffs, ComplexCoeffs)  # TODO: resolve
        return complex(
            self._impl.strings._impl.vdot_complex(
                self._string_set._map,
                self._impl.coeffs._impl,
                rhs._impl.strings._impl,
                rhs._impl.coeffs._impl,
            )
        )

    @classmethod
    def from_dense(
        cls,
        qubits: int | Qubits | None = None,
        source: Sequence[float] = tuple(),
        big_endian: bool = False,
    ) -> ComplexTermSum:
        """Create an instance of ``cls`` from a dense vector.

        Args:
            qubits: The qubit register or qubit count. If ``None``, the qubit register is
                inferred from the dense vector length.
            source: The vector to read from.
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The constructed instance.
        """
        if qubits is None:
            qubits = Qubits.from_hilbert_space_dim(len(source))
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        out = cls(qubits)
        assert isinstance(out._impl.coeffs, ComplexCoeffs)  # TODO: resolve
        out._impl.strings._impl.from_dense_complex(
            out._string_set._map, out._impl.coeffs._impl, list(source), big_endian
        )
        return out

    def to_dense(self, big_endian: bool = False) -> NDArray[np.complex128]:
        """Convert ``self`` to a dense vector.

        Args:
            big_endian: Whether to use big endian ordering for the resulting matrix. If ``False``,
                little endian ordering is used. In big (little) endian ordering, the least
                significant bit in the basis index integer is determined by the last (first) qubit
                in the register.

        Returns:
            The dense vector.
        """
        assert isinstance(self._impl.coeffs, ComplexCoeffs)  # TODO: resolve
        return self._impl.strings._impl.to_dense_complex(self._impl.coeffs._impl, big_endian)


class SymbolicTermSum(SymbolicTermSumBase[QubitStateArray, StringSpec], TermSum[Expr]):
    """A sum of terms consisting of state strings and symbolic coefficients.

    A set-like container of qubit-based terms that may be used to store unique terms and perform
    algebraic operations on them.

    Note:
        Coefficients are mutable in-place, but strings are the keys of a hashmap and therefore
        are not.
    """

    terms_type = SymbolicTerms

    def vdot(self, rhs: SymbolicTermSum) -> Expr:
        """Compute the inner product of ``self`` with ``rhs``."""
        _check_qubits_compatibility(self.qubits, rhs.qubits)
        return self._symbolic_vdot(rhs)
