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
from typing import Any, Literal, overload

import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy._zixy import ParaparticularMapper as ParaparticularImpl, Qubits
from zixy.container.coeffs import ComplexCoeffs, RealCoeffs
from zixy.container.data import TermData
from zixy.fermion.operator._strings import String as FermionOperatorString
from zixy.mappings.base import Mapper


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
    def apply(
        self, value: FermionOperatorString[Any, Any, Any], /, *, real: Literal[False] = False
    ) -> zqo.ComplexTermSum: ...

    @overload
    def apply(
        self, value: FermionOperatorString[Any, Any, Any], /, *, real: Literal[True] = True
    ) -> zqo.RealTermSum: ...

    @overload
    def apply(self, value: zfs.String, /, *, real: None = None) -> zqs.String: ...

    def apply(
        self,
        value: FermionOperatorString[Any, Any, Any] | zfs.String,
        /,
        *,
        real: bool | None = None,
    ) -> zqo.ComplexTermSum | zqo.RealTermSum | zqs.String:
        """Apply the mapper to ``value``."""
        if isinstance(value, zfs.String):
            if real is not None:
                raise TypeError("real must be None for fermionic state strings.")
            if len(value.modes) != len(self.qubits):
                raise ValueError("Fermion mode count must equal qubit count.")
            return zqs.String._create(self._impl.apply_state(value._impl, value.index))
        if isinstance(value, FermionOperatorString):
            if real:
                real_strings, real_coeffs, map_ = self._impl.apply_real(value.get_ops())
                real_data = TermData(
                    zqo.Strings._create(real_strings), RealCoeffs._create(real_coeffs)
                )
                return zqo.RealTermSum._create(
                    real_data,
                    map_,
                )
            complex_strings, complex_coeffs, map_ = self._impl.apply_complex(value.get_ops())
            complex_data = TermData(
                zqo.Strings._create(complex_strings), ComplexCoeffs._create(complex_coeffs)
            )
            return zqo.ComplexTermSum._create(
                complex_data,
                map_,
            )
        raise TypeError(f"Cannot map an instance of {type(value).__name__}.")
