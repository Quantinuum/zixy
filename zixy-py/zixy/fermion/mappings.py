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

"""Fermion to qubit mappings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from zixy._zixy import JordanWignerMapper as Impl, Qubits
from zixy.container.coeffs import ComplexCoeffs
from zixy.fermion.operator._strings import String as NormalString
from zixy.fermion.operator.general._strings import String as GeneralString
from zixy.qubit.pauli import ComplexTermSum as PauliComplexTermSum

FermionString = NormalString | GeneralString


class Mapper(ABC):
    """Base class for fermion to qubit mappers."""

    _impl: Impl
    qubits: Qubits

    @abstractmethod
    def encode(self, fermion_string: FermionString) -> PauliComplexTermSum:
        """Encode a fermionic string to qubit operators.

        Args:
            fermion_string: The fermionic string to encode.

        Returns:
            The encoded linear combination of qubit Pauli strings.
        """
        pass

    def encode_ca(self, c: int, a: int) -> PauliComplexTermSum:
        r"""Encode the product of a creation operator and an annihilation operator.

        The operator is defined as

        .. math:: a^\dagger_c a_a.

        Args:
            c: The fermionic mode of the creation operator.
            a: The fermionic mode of the annihilation operator.

        Returns:
            The encoded linear combination of qubit Pauli strings.
        """
        return self.encode(GeneralString(len(self.qubits), [(c, True), (a, False)]))

    def encode_n(self, i: int) -> PauliComplexTermSum:
        r"""Encode the local number operator for a given mode.

        The operator is defined as

        .. math:: n_i = a^\dagger_i a_i.

        Args:
            i: The fermionic mode to encode.

        Returns:
            The encoded linear combination of qubit Pauli strings.
        """
        return self.encode_ca(i, i)

    def encode_caca(self, c1: int, a1: int, c2: int, a2: int) -> PauliComplexTermSum:
        r"""Encode the product of two creation-annihilation operator products.

        The operator is defined as

        .. math:: a^\dagger_{c1} a_{a1} a^\dagger_{c2} a_{a2}.

        Args:
            c1: The fermionic mode of the first creation operator.
            a1: The fermionic mode of the first annihilation operator.
            c2: The fermionic mode of the second creation operator.
            a2: The fermionic mode of the second annihilation operator.

        Returns:
            The encoded linear combination of qubit Pauli strings.
        """
        return self.encode(
            GeneralString(
                len(self.qubits),
                [(c1, True), (a1, False), (c2, True), (a2, False)],
            )
        )

    def encode_nn(self, i: int, j: int) -> PauliComplexTermSum:
        r"""Encode the product of two local number operators.

        The operator is defined as

        .. math:: n_i n_j = a^\dagger_i a_i a^\dagger_j a_j.

        Args:
            i: The fermionic mode of the first number operator.
            j: The fermionic mode of the second number operator.

        Returns:
            The encoded linear combination of qubit Pauli strings.
        """
        return self.encode_caca(i, i, j, j)

    def encode_ccaa(self, c1: int, c2: int, a1: int, a2: int) -> PauliComplexTermSum:
        r"""Encode the product of two creation operators and two annihilation operators.

        The operator is defined as

        .. math:: a^\dagger_{c1} a^\dagger_{c2} a_{a1} a_{a2}.

        Args:
            c1: The fermionic mode of the first creation operator.
            c2: The fermionic mode of the second creation operator.
            a1: The fermionic mode of the first annihilation operator.
            a2: The fermionic mode of the second annihilation operator.

        Returns:
            The encoded linear combination of qubit Pauli strings.
        """
        return self.encode(
            GeneralString(
                len(self.qubits),
                [(c1, True), (c2, True), (a1, False), (a2, False)],
            )
        )


class JordanWignerMapper(Mapper):
    """Jordan--Wigner fermion to qubit mapper."""

    def __init__(self, qubits: int | Qubits, mode_ordering: Sequence[int] | None = None):
        """Initialize the mapper.

        Args:
            qubits: The qubits, or number thereof.
            mode_ordering: The ordering of the modes with respect to the qubits. If ``None``, the
                modes are ordered in the same way as the qubits.
        """
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        self.qubits = qubits
        self._impl = Impl(qubits, list(mode_ordering) if mode_ordering is not None else None)

    def encode(self, fermion_string: FermionString) -> PauliComplexTermSum:
        """Encode a fermionic string.

        Args:
            fermion_string: The fermionic string to encode.

        Returns:
            The encoded linear combination of qubit Pauli strings.
        """
        self._impl.op_load_product(fermion_string.get_ops())
        out = PauliComplexTermSum(self.qubits)
        assert isinstance(out._impl._coeffs, ComplexCoeffs)  # for mypy
        self._impl.op_contribute_complex(
            out._impl._cmpnts._impl,
            out._cmpnt_set._map,
            out._impl._coeffs._impl,
            1.0 + 0.0j,
        )
        return out
