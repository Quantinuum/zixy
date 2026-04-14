import pytest
from sympy import sympify

from zixy.qubit.pauli import (
    ComplexTermSum,
    I,
    RealTerm,
    RealTermSum,
    SymbolicTermSum,
    X,
    Y,
    Z,
)


def test_real_term_sum():
    lc = RealTermSum(6)
    assert len(lc) == 0
    lc += RealTerm(6, (X, X, X, I, I, I))
    assert len(lc) == 1
    assert lc[(X, X, X, I, I, I)] == 1.0
    lc += RealTerm(6, ((X, X, X, I, I, I), 2.0))
    assert len(lc) == 1
    assert lc[(X, X, X, I, I, I)] == 3.0
    lc += RealTerm(6, ((X, X, Y, I, I, I), -4.0))
    assert lc.l1_norm == 7.0
    assert lc.l2_norm == 5.0
    lc.l2_normalize()
    assert lc.l2_norm == 1.0
    lc *= 5.0
    assert lc[(X, X, Y, I, I, I)] == -4.0
    assert len(lc) == 2


def test_real_term_add_iterable():
    lc = RealTermSum(6)
    lc.add_iterable(
        (
            ((X, X, X, I, I, I), 0.5),
            ((X, X, Y, I, I, I), 1.5),
            ((X, X, Y, Z, I, I), -1.0),
            ((X, X, X, I, I, I), -1.0),
            ((Z, X, X, I, I, I), 1.3),
            ((X, X, Y, I, I, I), 1.6),
        )
    )
    assert len(lc) == 4
    assert str(lc) == "(-0.5, X0 X1 X2), (3.1, X0 X1 Y2), (-1.0, X0 X1 Y2 Z3), (1.3, Z0 X1 X2)"

    lc = RealTermSum.from_iterable(
        (
            ((X, X, X, I, I, I), 0.5),
            ((X, X, Y, I, I, I), 1.5),
            ((X, X, Y, Z, I, I), -1.0),
            ((X, X, X, I, I, I), -1.0),
            ((Z, X, X, I, I, I), 1.3),
            ((X, X, Y, I, I, I), 1.6),
        ),
        6,
    )
    assert len(lc) == 4
    assert str(lc) == "(-0.5, X0 X1 X2), (3.1, X0 X1 Y2), (-1.0, X0 X1 Y2 Z3), (1.3, Z0 X1 X2)"


def test_real_term_into_other_types():
    lc = RealTermSum.from_iterable(
        (
            ((X, X, X, I, I, I), 0.5),
            ((X, X, Y, I, I, I), 1.5),
            ((X, X, Y, Z, I, I), -1.0),
        ),
        6,
    )
    assert type(lc.into(ComplexTermSum) is ComplexTermSum)
    assert all(type(c) is complex for c in lc.into(ComplexTermSum)._data.coeffs)
    assert tuple(lc.into(ComplexTermSum)._data.coeffs) == (
        complex(0.5),
        complex(1.5),
        complex(-1.0),
    )
    lcc = lc.into(ComplexTermSum)
    lcc._data.coeffs.scale(1j)
    assert tuple(lcc._data.coeffs) == (complex(0, 0.5), complex(0, 1.5), complex(0, -1.0))
    with pytest.raises(ValueError) as err:
        lcc.into(RealTermSum)
    assert (
        str(err.value)
        == "Cannot represent coefficient value 0.5j of type <class 'complex'> as type <class 'float'>"
    )


def test_real_term_product():
    lc = RealTermSum.from_iterable(
        (
            ((X, X, Y, Z, I, I), -1.0),
            ((X, X, X, I, I, I), -1.0),
            ((Z, X, X, I, I, I), 2.0),
            ((X, X, Y, I, I, I), 1.5),
        ),
        6,
    )
    lc_square = lc * lc
    re_lc_square = lc_square.real_part
    im_lc_square = lc_square.imag_part
    assert str(re_lc_square) == "(8.25, ), (-3.0, Z3), (4.0, Y0 Z2 Z3), (-6.0, Y0 Z2)"
    assert str(im_lc_square.filter_significant()) == ""
    assert str(lc_square) == "((8.25+0j), ), ((-3+0j), Z3), ((4+0j), Y0 Z2 Z3), ((-6+0j), Y0 Z2)"


def test_complex_term_product():
    lc = ComplexTermSum.from_iterable(
        (
            ((X, X, Y, Z, I, I), -1.0),
            ((X, X, X, I, I, I), -1.0 + 1j),
            ((Z, X, X, I, I, I), 2.0 - 1j),
            ((X, X, Y, I, I, I), 1.5),
        ),
        6,
    )
    lc_square = lc * lc
    assert str(lc_square) == "((6.25-6j), ), ((-3+0j), Z3), ((4-2j), Y0 Z2 Z3), ((-6+3j), Y0 Z2)"


def test_symbolic_term_product():
    a, b, c = sympify("a, b, c")
    lc = SymbolicTermSum.from_iterable(
        (
            ((X, X, Y, Z, I, I), a),
            ((X, X, X, I, I, I), b),
            ((Z, X, X, I, I, I), c),
            ((X, X, Y, I, I, I), -1),
        ),
        6,
    )
    term = next(iter(lc))
    assert str(term * 2) == "(2*a, X0 X1 Y2 Z3)"
    lc_square = (lc * lc).filter_nonzero()
    assert (
        str(lc_square) == "(a**2 + b**2 + c**2 + 1, ), (-2*a*c, Y0 Z2 Z3), (-2*a, Z3), (2*c, Y0 Z2)"
    )
