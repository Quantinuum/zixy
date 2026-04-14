from __future__ import annotations

import numpy as np
from mock_cmpnts import Strings
from sympy import sympify

from zixy import _zixy
from zixy.container.coeffs import (
    ComplexCoeffs,
    ExprListWrapper,
    RealCoeffs,
    Sign,
    SignCoeffs,
    SymbolicCoeffs,
)
from zixy.container.data import TermData


def test_sign_term_data() -> None:
    td = TermData(Strings(0), SignCoeffs.from_size(0))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (0,) * 3
    td = TermData(Strings(1), SignCoeffs.from_sequence([Sign(True)]))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (1,) * 3
    td = TermData(Strings(6), SignCoeffs.from_sequence([Sign(False)] * 6))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (6,) * 3
    assert type(td._coeffs) is SignCoeffs
    assert type(td._coeffs._impl) is _zixy.SignVec
    assert td._coeffs[0] == Sign(False)
    assert all(td._coeffs[i] == Sign(False) for i in range(6))
    td._coeffs[0] = Sign(True)
    assert td._coeffs[0] == Sign(True)
    td = TermData(Strings(6), SignCoeffs.from_sequence([Sign(True)] * 6))
    assert all(td._coeffs[i] == Sign(True) for i in range(6))
    td = TermData(Strings(6), SignCoeffs.from_sequence([Sign(True), Sign(False)] * 3))
    assert all(td._coeffs[i] == Sign(True) for i in range(0, 6, 2))
    assert all(td._coeffs[i] == Sign(False) for i in range(1, 6, 2))
    assert td._coeffs.clone() == td._coeffs
    td = TermData(Strings(6), td._coeffs.clone())
    assert all(td._coeffs[i] == Sign(True) for i in range(0, 6, 2))
    assert all(td._coeffs[i] == Sign(False) for i in range(1, 6, 2))
    assert len(td.clone(slice(None, None, 2))) == len(td) // 2
    assert td.clone(slice(None, None, 2)) == td.clone(slice(None, None, 2))
    assert td.clone() is not td
    assert td.clone() == td
    assert str(td) == "(-1, ), (+1, ), (-1, ), (+1, ), (-1, ), (+1, )"
    td._cmpnts._impl._list[2] = "hello"
    assert str(td) == "(-1, ), (+1, ), (-1, hello), (+1, ), (-1, ), (+1, )"


def test_real_term_data() -> None:
    td = TermData(Strings(0), RealCoeffs.from_sequence([]))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (0,) * 3
    td = TermData(Strings(1), RealCoeffs.from_sequence([1.234]))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (1,) * 3
    td = TermData(Strings(6), RealCoeffs.from_sequence([1.0] * 6))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (6,) * 3
    assert type(td._coeffs) is RealCoeffs
    assert all(
        td._coeffs.np_array
        == [
            1.0,
        ]
        * 6
    )
    td = TermData(Strings(6), RealCoeffs.from_scalar(1.234, 6))
    assert all(
        td._coeffs.np_array
        == [
            1.234,
        ]
        * 6
    )
    td = TermData(Strings(6), RealCoeffs.from_sequence(list(range(6))))
    assert all(td._coeffs.np_array == np.arange(6))
    assert td._coeffs.clone() == td._coeffs
    td = TermData(Strings(6), td._coeffs.clone())
    assert all(td._coeffs.np_array == np.arange(6))
    assert td.clone(slice(None, None, 2)) == td.clone(slice(None, None, 2))
    assert td.clone() is not td
    assert td.clone() == td
    assert str(td) == "(0.0, ), (1.0, ), (2.0, ), (3.0, ), (4.0, ), (5.0, )"
    td._cmpnts._impl._list[2] = "hello"
    assert str(td) == "(0.0, ), (1.0, ), (2.0, hello), (3.0, ), (4.0, ), (5.0, )"


def test_complex_term_data() -> None:
    td = TermData(Strings(0), ComplexCoeffs.from_sequence([]))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (0,) * 3
    td = TermData(Strings(1), ComplexCoeffs.from_sequence([complex(1.2, -3.4)]))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (1,) * 3
    td = TermData(Strings(6), ComplexCoeffs.from_sequence([complex(1, 0)] * 6))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (6,) * 3
    assert type(td._coeffs) is ComplexCoeffs
    assert td._coeffs.np_array.shape == (6,)
    assert all(td._coeffs.np_array == complex(1, 0))
    td = TermData(Strings(6), ComplexCoeffs.from_scalar(complex(1.2, -3.4), 6))
    assert all(
        td._coeffs.np_array
        == [
            complex(1.2, -3.4),
        ]
        * 6
    )
    td = TermData(Strings(6), ComplexCoeffs.from_sequence(list(complex(i) for i in range(6))))
    assert all(td._coeffs.np_array == np.arange(6))
    assert td._coeffs.clone() == td._coeffs
    td = TermData(Strings(6), td._coeffs.clone())
    assert all(td._coeffs.np_array == np.arange(6))
    assert td.clone(slice(None, None, 2)) == td.clone(slice(None, None, 2))
    assert td.clone() is not td
    assert td.clone() == td
    assert str(td) == "(0j, ), ((1+0j), ), ((2+0j), ), ((3+0j), ), ((4+0j), ), ((5+0j), )"
    td._cmpnts._impl._list[2] = "hello"
    assert str(td) == "(0j, ), ((1+0j), ), ((2+0j), hello), ((3+0j), ), ((4+0j), ), ((5+0j), )"


def test_symbolic_term_data() -> None:
    td = TermData(Strings(0), SymbolicCoeffs.from_sequence([]))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (0,) * 3
    td = TermData(Strings(1), SymbolicCoeffs.from_sequence([sympify(1)]))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (1,) * 3
    td = TermData(Strings(6), SymbolicCoeffs.from_sequence([sympify(1)] * 6))
    assert (len(td), len(td._cmpnts), len(td._coeffs)) == (6,) * 3
    assert type(td._coeffs) is SymbolicCoeffs
    assert type(td._coeffs._impl) is ExprListWrapper
    assert isinstance(td._coeffs._impl, ExprListWrapper)
    assert (
        td._coeffs._impl._list
        == [
            sympify(1),
        ]
        * 6
    )
    x = sympify("x")
    td = TermData(Strings(6), SymbolicCoeffs.from_scalar(x, 6))
    assert isinstance(td._coeffs._impl, ExprListWrapper)
    assert (
        td._coeffs._impl._list
        == [
            x,
        ]
        * 6
    )
    td = TermData(Strings(6), SymbolicCoeffs.from_sequence(list(x * i for i in range(6))))
    assert isinstance(td._coeffs._impl, ExprListWrapper)
    assert td._coeffs._impl._list == [sympify(0), x, 2 * x, 3 * x, 4 * x, 5 * x]
    assert td._coeffs.clone() == td._coeffs
    td = TermData(Strings(6), td._coeffs.clone())
    assert isinstance(td._coeffs._impl, ExprListWrapper)
    assert td._coeffs._impl._list == [sympify(0), x, 2 * x, 3 * x, 4 * x, 5 * x]
    assert td.clone(slice(None, None, 2)) == td.clone(slice(None, None, 2))
    assert td.clone() is not td
    assert td.clone() == td
    assert str(td) == "(0, ), (x, ), (2*x, ), (3*x, ), (4*x, ), (5*x, )"
    td._cmpnts._impl._list[2] = "hello"
    assert str(td) == "(0, ), (x, ), (2*x, hello), (3*x, ), (4*x, ), (5*x, )"
