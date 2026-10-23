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

"""Jordan--Wigner mapper implementation."""

from collections.abc import Sequence
from typing import Any, Literal, overload

from zixy._zixy import JordanWignerMapper as JordanWignerImpl, Qubits
from zixy.container.coeffs import ComplexCoeffs, RealCoeffs
from zixy.container.data import TermData
from zixy.fermion.operator._strings import String as FermionOperatorString
from zixy.fermion.state import String as FermionStateString
from zixy.mappings.base import Mapper
from zixy.qubit.operator._strings import Strings as PauliStrings
from zixy.qubit.operator._terms import (
    ComplexTermSum as PauliComplexTermSum,
    RealTermSum as PauliRealTermSum,
)
from zixy.qubit.state import String as QubitStateString


class JordanWignerMapper(Mapper):
    """Jordan--Wigner mapper from fermionic operators and states to qubit values."""

    _impl: JordanWignerImpl
    qubits: Qubits

    def __init__(self, qubits: int | Qubits, mode_ordering: Sequence[int] | None = None):
        """Initialize the mapper for a qubit register."""
        if isinstance(qubits, int):
            qubits = Qubits.from_count(qubits)
        self.qubits = qubits
        self._impl = JordanWignerImpl(
            qubits, list(mode_ordering) if mode_ordering is not None else None
        )

    @overload
    def apply(
        self, value: FermionOperatorString[Any, Any, Any], /, *, real: Literal[False] = False
    ) -> PauliComplexTermSum: ...

    @overload
    def apply(
        self, value: FermionOperatorString[Any, Any, Any], /, *, real: Literal[True] = True
    ) -> PauliRealTermSum: ...

    @overload
    def apply(self, value: FermionStateString, /, *, real: None = None) -> QubitStateString: ...

    def apply(
        self,
        value: FermionOperatorString[Any, Any, Any] | FermionStateString,
        /,
        *,
        real: bool | None = None,
    ) -> PauliComplexTermSum | PauliRealTermSum | QubitStateString:
        """Apply the mapper to ``value``."""
        if isinstance(value, FermionStateString):
            if real is not None:
                raise TypeError("real must be None for fermionic state strings.")
            if len(value.modes) != len(self.qubits):
                raise ValueError("Fermion mode count must equal qubit count.")
            return QubitStateString._create(self._impl.apply_state(value._impl, value.index))
        if isinstance(value, FermionOperatorString):
            if real:
                real_strings, real_coeffs, map_ = self._impl.apply_real(value.get_ops())
                real_data = TermData(
                    PauliStrings._create(real_strings), RealCoeffs._create(real_coeffs)
                )
                return PauliRealTermSum._create(
                    real_data,
                    map_,
                )
            complex_strings, complex_coeffs, map_ = self._impl.apply_complex(value.get_ops())
            complex_data = TermData(
                PauliStrings._create(complex_strings), ComplexCoeffs._create(complex_coeffs)
            )
            return PauliComplexTermSum._create(
                complex_data,
                map_,
            )
        raise TypeError(f"Cannot map an instance of {type(value).__name__}.")
