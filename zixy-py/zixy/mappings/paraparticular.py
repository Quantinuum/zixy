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

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy._zixy import ParaparticularMapper as ParaparticularImpl, Qubits
from zixy.container.coeffs import ComplexCoeffs, RealCoeffs
from zixy.container.data import TermData
from zixy.mappings._operators import _apply_operator, _get_operator_ops
from zixy.mappings.base import Mapper

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    import zixy.fermion.operator.general as zfog
    import zixy.fermion.operator.normal as zfon
    from zixy.mappings._operators import _NumericOperator


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
        self,
        value: zfog.String | zfon.String | _NumericOperator,
        /,
        *,
        real: Literal[False] | None = None,
    ) -> zqo.ComplexTermSum: ...

    @overload
    def apply(
        self,
        value: zfog.String | zfon.String | _NumericOperator,
        /,
        *,
        real: Literal[True],
    ) -> zqo.RealTermSum: ...

    @overload
    def apply(
        self,
        value: zfog.String | zfon.String | _NumericOperator,
        /,
        *,
        real: bool | None,
    ) -> zqo.ComplexTermSum | zqo.RealTermSum: ...

    @overload
    def apply(self, value: zfs.String, /, *, real: None = None) -> zqs.String: ...

    def apply(
        self,
        value: zfog.String | zfon.String | _NumericOperator | zfs.String,
        /,
        *,
        real: bool | None = None,
    ) -> zqo.ComplexTermSum | zqo.RealTermSum | zqs.String:
        """Apply the mapper to ``value``."""
        import zixy.fermion.operator.general as zfog  # noqa: PLC0415
        import zixy.fermion.operator.normal as zfon  # noqa: PLC0415

        if isinstance(value, zfs.String):
            if real is not None:
                raise TypeError("real must be None for fermionic state strings.")
            if len(value.modes) != len(self.qubits):
                raise ValueError("Fermion mode count must equal qubit count.")
            return zqs.String._create(self._impl.apply_state(value._impl, value.index))
        if isinstance(value, zfog.String | zfon.String):
            ops = _get_operator_ops(value, self.qubits)
            if real:
                real_strings, real_coeffs, map_ = self._impl.apply_real(ops)
                real_data = TermData(
                    zqo.Strings._create(real_strings), RealCoeffs._create(real_coeffs)
                )
                return zqo.RealTermSum._create(
                    real_data,
                    map_,
                )
            complex_strings, complex_coeffs, map_ = self._impl.apply_complex(ops)
            complex_data = TermData(
                zqo.Strings._create(complex_strings), ComplexCoeffs._create(complex_coeffs)
            )
            return zqo.ComplexTermSum._create(
                complex_data,
                map_,
            )
        return _apply_operator(value, self.apply, self.qubits, real=bool(real))
