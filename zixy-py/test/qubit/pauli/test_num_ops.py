from zixy.qubit.pauli import (
    ComplexTermSum,
    RealTermSum,
    SymbolicTermSum,
)
from zixy.qubit.pauli.num_ops import create_num_op


def test_create_num_op():
    nop = create_num_op(RealTermSum, 6)
    assert (
        str(nop)
        == "(3.0, ), (-0.5, Z0), (-0.5, Z1), (-0.5, Z2), (-0.5, Z3), (-0.5, Z4), (-0.5, Z5)"
    )
    nop = create_num_op(ComplexTermSum, 6)
    assert (
        str(nop)
        == "((3+0j), ), ((-0.5+0j), Z0), ((-0.5+0j), Z1), ((-0.5+0j), Z2), ((-0.5+0j), Z3), ((-0.5+0j), Z4), ((-0.5+0j), Z5)"
    )
    nop = create_num_op(SymbolicTermSum, 6)
    assert (
        str(nop) == "(3, ), (-1/2, Z0), (-1/2, Z1), (-1/2, Z2), (-1/2, Z3), (-1/2, Z4), (-1/2, Z5)"
    )
