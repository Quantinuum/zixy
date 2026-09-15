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

"""Parity mapper implementation."""

from collections.abc import Sequence
from typing import Any

from zixy._zixy import ParityMapper as ParityImpl, Qubits
from zixy.container.coeffs import ComplexCoeffs
from zixy.container.data import TermData
from zixy.fermion.operator._strings import String as FermionString
from zixy.mappings.base import Mapper
from zixy.qubit.pauli._strings import Strings as PauliStrings
from zixy.qubit.pauli._terms import ComplexTermSum as PauliComplexTermSum


class ParityMapper(Mapper[FermionString[Any, Any, Any], PauliComplexTermSum]):
    """Parity mapper from fermionic strings to Pauli term sums."""

    _impl: ParityImpl
    qubits: Qubits

    def __init__(self, qubits: int | Qubits, mode_ordering: Sequence[int] | None = None):
        """Initialize the mapper for a qubit register."""
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        self.qubits = qubits
        self._impl = ParityImpl(qubits, list(mode_ordering) if mode_ordering is not None else None)

    def apply(self, value: FermionString[Any, Any, Any], /) -> PauliComplexTermSum:
        """Map ``value`` to a complex Pauli term sum."""
        cmpnts, coeffs = self._impl.apply(value.get_ops())
        return PauliComplexTermSum._create(
            TermData(PauliStrings._create(cmpnts), ComplexCoeffs._create(coeffs))
        )
