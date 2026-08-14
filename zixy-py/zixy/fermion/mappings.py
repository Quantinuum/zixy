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

from collections.abc import Sequence
from typing import Any, TypeAlias

from zixy._zixy import (
    BravyiKitaevMapper as BravyiKitaevImpl,
    JordanWignerMapper as JordanWignerImpl,
    Mapper as Impl,
    ParaparticularMapper as ParaparticularImpl,
    ParityMapper as ParityImpl,
    Qubits,
)
from zixy.container.coeffs import ComplexCoeffs, RealCoeffs
from zixy.fermion.operator._strings import String as FermionString
from zixy.qubit.pauli import ComplexTermSum as PauliComplexTermSum, RealTermSum as PauliRealTermSum

ImplType: TypeAlias = (
    type[JordanWignerImpl] | type[BravyiKitaevImpl] | type[ParityImpl] | type[ParaparticularImpl]
)


class Mapper:
    """Base class for fermion to qubit mappers."""

    impl_type: ImplType
    _impl: Impl
    qubits: Qubits

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
        self._impl = self.impl_type(
            qubits, list(mode_ordering) if mode_ordering is not None else None
        )

    def encode_real(
        self, fermion_string: FermionString[Any, Any, Any], coeff: float = 1.0
    ) -> PauliRealTermSum:
        """Encode a fermionic string to real qubit operators.

        Args:
            fermion_string: The fermionic string to encode.
            coeff: A real factor to multiply the encoded string by.

        Returns:
            The encoded linear combination of real qubit Pauli strings.
        """
        self._impl.op_load_product(fermion_string.get_ops())
        out = PauliRealTermSum(self.qubits)
        assert isinstance(out._impl._coeffs, RealCoeffs)  # for mypy
        self._impl.op_contribute_real(
            out._impl._cmpnts._impl,
            out._cmpnt_set._map,
            out._impl._coeffs._impl,
            coeff,
        )
        return out

    def encode_complex(
        self, fermion_string: FermionString[Any, Any, Any], coeff: complex = 1.0
    ) -> PauliComplexTermSum:
        """Encode a fermionic string to complex qubit operators.

        Args:
            fermion_string: The fermionic string to encode.
            coeff: A complex factor to multiply the encoded string by.

        Returns:
            The encoded linear combination of complex qubit Pauli strings.
        """
        self._impl.op_load_product(fermion_string.get_ops())
        out = PauliComplexTermSum(self.qubits)
        assert isinstance(out._impl._coeffs, ComplexCoeffs)  # for mypy
        self._impl.op_contribute_complex(
            out._impl._cmpnts._impl,
            out._cmpnt_set._map,
            out._impl._coeffs._impl,
            coeff,
        )
        return out

    def encode(
        self, fermion_string: FermionString[Any, Any, Any], coeff: complex = 1.0
    ) -> PauliComplexTermSum:
        """Encode a fermionic string to qubit operators.

        Args:
            fermion_string: The fermionic string to encode.
            coeff: A factor to multiply the encoded string by.

        Returns:
            The encoded linear combination of qubit Pauli strings.
        """
        return self.encode_complex(fermion_string, coeff)


class JordanWignerMapper(Mapper):
    """Jordan--Wigner fermion to qubit mapper."""

    impl_type = JordanWignerImpl


class BravyiKitaevMapper(Mapper):
    """Bravyi--Kitaev fermion to qubit mapper."""

    impl_type = BravyiKitaevImpl


class ParityMapper(Mapper):
    """Parity fermion to qubit mapper."""

    impl_type = ParityImpl


class ParaparticularMapper(Mapper):
    """Paraparticular fermion to qubit mapper."""

    impl_type = ParaparticularImpl
