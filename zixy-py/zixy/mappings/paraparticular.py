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

"""Paraparticular mapper implementation."""

from collections.abc import Sequence
from typing import Any, overload

from zixy._zixy import ParaparticularMapper as ParaparticularImpl, Qubits
from zixy.container.coeffs import ComplexCoeffs
from zixy.container.data import TermData
from zixy.fermion.operator._strings import String as FermionOperatorString
from zixy.fermion.state import String as FermionStateString
from zixy.mappings.base import Mapper
from zixy.qubit.pauli._strings import Strings as PauliStrings
from zixy.qubit.pauli._terms import ComplexTermSum as PauliComplexTermSum
from zixy.qubit.state import String as QubitStateString


class ParaparticularMapper(Mapper):
    """Paraparticular mapper from fermionic operators and states to qubit values."""

    _impl: ParaparticularImpl
    qubits: Qubits

    def __init__(self, qubits: int | Qubits, mode_ordering: Sequence[int] | None = None):
        """Initialize the mapper for a qubit register."""
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        self.qubits = qubits
        self._impl = ParaparticularImpl(
            qubits, list(mode_ordering) if mode_ordering is not None else None
        )

    @overload
    def apply(self, value: FermionOperatorString[Any, Any, Any], /) -> PauliComplexTermSum: ...

    @overload
    def apply(self, value: FermionStateString, /) -> QubitStateString: ...

    def apply(
        self, value: FermionOperatorString[Any, Any, Any] | FermionStateString, /
    ) -> PauliComplexTermSum | QubitStateString:
        """Map a fermionic operator string or occupation-number state string."""
        if isinstance(value, FermionStateString):
            if len(value.modes) != len(self.qubits):
                raise ValueError("Fermion mode count must equal qubit count.")
            return QubitStateString._create(self._impl.apply_state(value._impl, value.index))
        if isinstance(value, FermionOperatorString):
            cmpnts, coeffs, map_ = self._impl.apply(value.get_ops())
            return PauliComplexTermSum._create(
                TermData(PauliStrings._create(cmpnts), ComplexCoeffs._create(coeffs)), map_
            )
        raise TypeError(f"Cannot map an instance of {type(value).__name__}.")
