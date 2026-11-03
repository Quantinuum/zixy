from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.coeffs import Sign
from zixy.mappings._operators import _apply_operator

if TYPE_CHECKING:
    from collections.abc import Callable

    from zixy._zixy import Qubits
    from zixy.mappings import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParaparticularMapper,
        ParityMapper,
    )


def check_mapping_types(
    mapper: JordanWignerMapper | BravyiKitaevMapper | ParityMapper | ParaparticularMapper,
    real: bool,
) -> None:
    string = zfon.String(2, "F0^")
    normal = zfon.RealTerm(2, ("F0^", 2.0))
    general = zfog.ComplexTerm(2, ("F0^", 2j))
    sign = zfon.SignTerm(2, ("F0^", Sign.from_int(-1)))
    normal_sum = zfon.RealTermSum(2)
    general_sum = zfog.ComplexTermSum(2)

    assert_type(mapper.apply(zfog.String(2, "F0^")), zqo.ComplexTermSum)
    assert_type(mapper.apply(string), zqo.ComplexTermSum)
    assert_type(mapper.apply(string, real=None), zqo.ComplexTermSum)
    assert_type(mapper.apply(string, real=True), zqo.RealTermSum)
    assert_type(mapper.apply(string, real=real), zqo.ComplexTermSum | zqo.RealTermSum)
    assert_type(mapper.apply(normal), zqo.ComplexTermSum)
    assert_type(mapper.apply(general), zqo.ComplexTermSum)
    assert_type(mapper.apply(sign), zqo.ComplexTermSum)
    assert_type(mapper.apply(normal_sum), zqo.ComplexTermSum)
    assert_type(mapper.apply(general_sum), zqo.ComplexTermSum)
    assert_type(mapper.apply(general, real=True), zqo.RealTermSum)
    assert_type(mapper.apply(general_sum, real=True), zqo.RealTermSum)
    assert_type(mapper.apply(normal, real=real), zqo.ComplexTermSum | zqo.RealTermSum)
    assert_type(mapper.apply(general_sum, real=real), zqo.ComplexTermSum | zqo.RealTermSum)
    assert_type(mapper.apply(zfs.String(2)), zqs.String)
    assert_type(normal_sum.to_qubit(), zqo.ComplexTermSum)
    assert_type(general_sum.to_qubit(), zqo.ComplexTermSum)

    if TYPE_CHECKING:
        mapper.apply(object())  # type: ignore[call-overload]
        mapper.apply(zfon.SymbolicTerm(2))  # type: ignore[call-overload]


def check_linear_extension_types(
    map_string: Callable[[zfog.String | zfon.String], zqo.ComplexTermSum],
    map_normal: Callable[[zfon.String], zqo.ComplexTermSum],
    qubits: Qubits,
    real: bool,
) -> None:
    normal = zfon.RealTerm(2, ("F0^", 2.0))
    normal_sum = zfon.ComplexTermSum(2)
    general = zfog.RealTerm(2, ("F0^", 2.0))

    assert_type(_apply_operator(normal, map_string, qubits), zqo.ComplexTermSum)
    assert_type(_apply_operator(normal_sum, map_string, qubits), zqo.ComplexTermSum)
    assert_type(_apply_operator(normal, map_string, qubits, real=True), zqo.RealTermSum)
    assert_type(
        _apply_operator(normal_sum, map_string, qubits, real=real),
        zqo.ComplexTermSum | zqo.RealTermSum,
    )
    assert_type(_apply_operator(general, map_string, qubits), zqo.ComplexTermSum)
    if TYPE_CHECKING:
        # The shared mapper must accept both supported operator-string families.
        _apply_operator(general, map_normal, qubits)  # type: ignore[arg-type]
