import numpy as np
import pytest
from sympy import cos, sin, sympify

from zixy.container.coeffs import SymbolicCoeffs


def test_init_append():
    x, y = sympify("x, y")
    numbers = (0, x, 1.234, complex(1, 2), 2, -2, x * y)
    v = SymbolicCoeffs.from_sequence(numbers)
    assert len(v) == 7
    assert str(v) == "[0, x, 1.23400000000000, 1 + 2*I, 2, -2, x*y]"
    v.append(sympify("1"))
    assert len(v) == 8
    assert str(v) == "[0, x, 1.23400000000000, 1 + 2*I, 2, -2, x*y, 1]"
    view = v[::-1]
    assert str(view) == "[1, x*y, -2, 2, 1 + 2*I, 1.23400000000000, x, 0]"
    view = view[::-1]
    assert view == v
    assert str(v[::2]) == "[0, 1.23400000000000, 2, x*y]"


def test_subs_convert():
    x, y, z = sympify("x, y, z")
    v = SymbolicCoeffs.from_sequence([x, 2 * x, x**2, x * z, sin(x)])
    assert v.subs({x: 3}) == SymbolicCoeffs.from_sequence([3, 6, 9, 3 * z, sin(3)])
    assert v == SymbolicCoeffs.from_sequence([x, 2 * x, x**2, x * z, sin(x)])
    v.isubs({x: 3})
    assert v == SymbolicCoeffs.from_sequence([3, 6, 9, 3 * z, sin(3)])
    # both real and complex conversions fail because z is unsubstituted
    with pytest.raises(TypeError):
        v.try_to_real()
    with pytest.raises(TypeError):
        v.try_to_real()
    v.isubs({z: y})
    assert v == SymbolicCoeffs.from_sequence([3, 6, 9, 3 * y, sin(3)])
    v.isubs({y: 3j})
    # both real conversion fails because of non-real coeff
    with pytest.raises(TypeError):
        v.try_to_real()
    assert v.try_to_complex().np_array.dtype.type is np.complex128
    assert v.try_to_complex().to_tuple() == (3, 6, 9, 9j, np.sin(3))
    v[3] *= 1j
    assert v.try_to_complex().np_array.dtype.type is np.complex128
    assert v.try_to_complex().to_tuple() == (3, 6, 9, -9, np.sin(3))
    assert v.try_to_real().np_array.dtype.type is np.float64
    assert v.try_to_real().to_tuple() == (3, 6, 9, -9, np.sin(3))


def test_diff():
    x, y = sympify("x, y")
    v = SymbolicCoeffs.from_sequence([x, 2 * x, x**2, x * y, sin(x)])
    assert v.diff(x) == SymbolicCoeffs.from_sequence([1, 2, 2 * x, y, cos(x)])
    v.idiff(x)
    assert v == SymbolicCoeffs.from_sequence([1, 2, 2 * x, y, cos(x)])


def test_symbolic_vec_view():
    v = SymbolicCoeffs.from_sequence([sympify(c) for c in ["x", "y", "z", "x*y", 1, "z+1"]])
    view = v[::2]
    assert view == SymbolicCoeffs.from_sequence([sympify(c) for c in ["x", "z", 1]])
    assert view.free_symbols == {sympify("x"), sympify("z")}
    assert v.free_symbols == {sympify("x"), sympify("y"), sympify("z")}


def test_symbolic_vec_view_reverse():
    v = SymbolicCoeffs.from_sequence([sympify(c) for c in ["x", 2, "y", 3, "z"]])
    view = v[::-2]
    assert view == SymbolicCoeffs.from_sequence([sympify(c) for c in ["z", "y", "x"]])
    assert view.free_symbols == {sympify("x"), sympify("y"), sympify("z")}


def test_inplace_sub_view():
    v = SymbolicCoeffs.from_sequence([sympify(c) for c in ["x", "y", "x+y", "z", "x*z", 2]])
    view = v[1::2]
    assert view == SymbolicCoeffs.from_sequence([sympify(c) for c in ["y", "z", 2]])
    view.isubs({sympify("y"): 3, sympify("z"): 4})
    assert view == SymbolicCoeffs.from_sequence([sympify(c) for c in ["3", "4", 2]])
    assert v == SymbolicCoeffs.from_sequence([sympify(c) for c in ["x", "3", "x+y", "4", "x*z", 2]])


def test_inplace_diff_view():
    v = SymbolicCoeffs.from_sequence(
        [sympify(c) for c in ["x", "y", "x*y", "x**2", "y**2", "sin(x)"]]
    )
    view = v[::2]
    assert view == SymbolicCoeffs.from_sequence([sympify(c) for c in ["x", "x*y", "y**2"]])
    view.idiff(sympify("x"))
    assert view == SymbolicCoeffs.from_sequence([sympify(c) for c in ["1", "y", "0"]])
    assert v == SymbolicCoeffs.from_sequence(
        [sympify(c) for c in ["1", "y", "y", "x**2", "0", "sin(x)"]]
    )


def test_inplace_diff_view_reverse():
    v = SymbolicCoeffs.from_sequence(
        [sympify(c) for c in ["x", "y", "x*y", "x**2", "y**2", "sin(x)"]]
    )
    view = v[::-2]
    assert view == SymbolicCoeffs.from_sequence([sympify(c) for c in ["sin(x)", "x**2", "y"]])
    view.idiff(sympify("x"))
    assert view == SymbolicCoeffs.from_sequence([sympify(c) for c in ["cos(x)", "2*x", "0"]])
    assert v == SymbolicCoeffs.from_sequence(
        [sympify(c) for c in ["x", "0", "x*y", "2*x", "y**2", "cos(x)"]]
    )
