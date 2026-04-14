import pytest
from sympy import Expr, sympify

from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSign,
    ComplexSignCoeffs,
    RealCoeffs,
    Sign,
    SignCoeffs,
    SymbolicCoeffs,
    convert,
    typesafe_mul,
)


def test_conversion():
    assert int(Sign(0)) == 1
    assert int(Sign(1)) == -1
    assert float(Sign(0)) == 1
    assert float(Sign(1)) == -1
    assert complex(Sign(0)) == 1
    assert complex(Sign(1)) == -1
    assert int(ComplexSign(0)) == 1
    assert int(ComplexSign(2)) == -1
    assert float(ComplexSign(0)) == 1
    assert float(ComplexSign(2)) == -1
    assert complex(ComplexSign(0)) == 1
    assert complex(ComplexSign(1)) == 1j
    assert complex(ComplexSign(2)) == -1
    assert complex(ComplexSign(3)) == -1j
    with pytest.raises(ValueError):
        int(ComplexSign(1))
    with pytest.raises(ValueError):
        int(ComplexSign(3))
    with pytest.raises(ValueError):
        float(ComplexSign(1))
    with pytest.raises(ValueError):
        float(ComplexSign(3))


@pytest.mark.parametrize(
    "types",
    [
        (Sign, int, int),
        (Sign, float, float),
        (Sign, complex, complex),
        (Sign, Expr, Expr),
        (Sign, Sign, Sign),
        (Sign, ComplexSign, ComplexSign),
        (ComplexSign, int, complex),
        (ComplexSign, float, complex),
        (ComplexSign, complex, complex),
        (ComplexSign, Expr, Expr),
        (ComplexSign, ComplexSign, ComplexSign),
    ],
)
def test_scalar_mul(types):
    l_type, r_type, p_type = types
    assert isinstance(l_type() * r_type(), p_type)
    assert isinstance(r_type() * l_type(), p_type)


@pytest.mark.parametrize(
    "values_and_validity",
    [
        (Sign(), int(1), True),
        (Sign(), int(-1), True),
        (Sign(), int(2), False),
        (Sign(), float(1), True),
        (Sign(), float(-1), True),
        (Sign(), float(1.01), False),
        (Sign(), complex(1), True),
        (Sign(), complex(-1), True),
        (Sign(), complex(0, -1), False),
        (Sign(), complex(1, 0.01), False),
        (Sign(), sympify(1), True),
        (Sign(), sympify(-1), True),
        (Sign(), sympify("x"), False),
        (Sign(), sympify(0.01), False),
        (Sign(), sympify("I"), False),
        (Sign(), sympify("-I"), False),
        (Sign(), sympify("1-I"), False),
        (Sign(), Sign(True), True),
        (Sign(), Sign(False), True),
        (Sign(), ComplexSign(0), True),
        (Sign(), ComplexSign(1), False),
        (Sign(), ComplexSign(2), True),
        (Sign(), ComplexSign(3), False),
        (ComplexSign(), int(1), True),
        (ComplexSign(), int(-1), True),
        (ComplexSign(), int(2), False),
        (ComplexSign(), float(1), True),
        (ComplexSign(), float(-1), True),
        (ComplexSign(), float(1.01), False),
        (ComplexSign(), complex(1), True),
        (ComplexSign(), complex(-1), True),
        (ComplexSign(), complex(1, 0.01), False),
        (ComplexSign(), complex(0, -1), True),
        (ComplexSign(), sympify(1), True),
        (ComplexSign(), sympify(-1), True),
        (ComplexSign(), sympify("x"), False),
        (ComplexSign(), sympify(0.01), False),
        (ComplexSign(), sympify("I"), True),
        (ComplexSign(), sympify("-I"), True),
        (ComplexSign(), sympify("1-I"), False),
        (ComplexSign(), Sign(True), True),
        (ComplexSign(), Sign(False), True),
        (ComplexSign(), ComplexSign(0), True),
        (ComplexSign(), ComplexSign(1), True),
        (ComplexSign(), ComplexSign(2), True),
        (ComplexSign(), ComplexSign(3), True),
    ],
)
def test_scalar_imul(values_and_validity):
    l, r, valid = values_and_validity
    l_type = type(l)
    if valid:
        l *= r
    else:
        with pytest.raises(ValueError):
            l *= r
    assert type(l) is l_type


@pytest.mark.parametrize(
    "values_and_validity",
    [
        (int(1), int(1), True),
        (int(1), int(-1), True),
        (int(1), int(2), True),
        (int(1), float(1), True),
        (int(1), float(-1), True),
        (int(1), float(1.01), False),
        (int(1), complex(1), True),
        (int(1), complex(1.01), False),
        (int(1), complex(-1), True),
        (int(1), complex(0, -1), False),
        (int(1), complex(1, 0.01), False),
        (int(1), sympify(1), True),
        (int(1), sympify(2), True),
        (int(1), sympify(-1), True),
        (int(1), sympify("x"), False),
        (int(1), sympify(0.01), False),
        (int(1), sympify("I"), False),
        (int(1), sympify("-I"), False),
        (int(1), sympify("1-I"), False),
        (int(1), Sign(True), True),
        (int(1), Sign(False), True),
        (int(1), ComplexSign(0), True),
        (int(1), ComplexSign(1), False),
        (int(1), ComplexSign(2), True),
        (int(1), ComplexSign(3), False),
        (float(1), int(1), True),
        (float(1), int(-1), True),
        (float(1), int(2), True),
        (float(1), float(1), True),
        (float(1), float(-1), True),
        (float(1), float(1.01), True),
        (float(1), complex(1), True),
        (float(1), complex(1.01), True),
        (float(1), complex(-1), True),
        (float(1), complex(0, -1), False),
        (float(1), complex(1, 0.01), False),
        (float(1), sympify(1), True),
        (float(1), sympify(2), True),
        (float(1), sympify(-1), True),
        (float(1), sympify("x"), False),
        (float(1), sympify(0.01), True),
        (float(1), sympify("I"), False),
        (float(1), sympify("-I"), False),
        (float(1), sympify("1-I"), False),
        (float(1), Sign(True), True),
        (float(1), Sign(False), True),
        (float(1), ComplexSign(0), True),
        (float(1), ComplexSign(1), False),
        (float(1), ComplexSign(2), True),
        (float(1), ComplexSign(3), False),
        (complex(1), int(1), True),
        (complex(1), int(-1), True),
        (complex(1), int(2), True),
        (complex(1), float(1), True),
        (complex(1), float(-1), True),
        (complex(1), float(1.01), True),
        (complex(1), complex(1), True),
        (complex(1), complex(1.01), True),
        (complex(1), complex(-1), True),
        (complex(1), complex(0, -1), True),
        (complex(1), complex(1, 0.01), True),
        (complex(1), sympify(1), True),
        (complex(1), sympify(2), True),
        (complex(1), sympify(-1), True),
        (complex(1), sympify("x"), False),
        (complex(1), sympify(0.01), True),
        (complex(1), sympify("I"), True),
        (complex(1), sympify("-I"), True),
        (complex(1), sympify("1-I"), True),
        (complex(1), Sign(True), True),
        (complex(1), Sign(False), True),
        (complex(1), ComplexSign(0), True),
        (complex(1), ComplexSign(1), True),
        (complex(1), ComplexSign(2), True),
        (complex(1), ComplexSign(3), True),
        (Sign(), int(1), True),
        (Sign(), int(-1), True),
        (Sign(), int(2), False),
        (Sign(), float(1), True),
        (Sign(), float(-1), True),
        (Sign(), float(1.01), False),
        (Sign(), complex(1), True),
        (Sign(), complex(1.01), False),
        (Sign(), complex(-1), True),
        (Sign(), complex(0, -1), False),
        (Sign(), complex(1, 0.01), False),
        (Sign(), sympify(1), True),
        (Sign(), sympify(2), False),
        (Sign(), sympify(-1), True),
        (Sign(), sympify("x"), False),
        (Sign(), sympify(0.01), False),
        (Sign(), sympify("I"), False),
        (Sign(), sympify("-I"), False),
        (Sign(), sympify("1-I"), False),
        (Sign(), Sign(True), True),
        (Sign(), Sign(False), True),
        (Sign(), ComplexSign(0), True),
        (Sign(), ComplexSign(1), False),
        (Sign(), ComplexSign(2), True),
        (Sign(), ComplexSign(3), False),
        (ComplexSign(), int(1), True),
        (ComplexSign(), int(-1), True),
        (ComplexSign(), int(2), False),
        (ComplexSign(), float(1), True),
        (ComplexSign(), float(-1), True),
        (ComplexSign(), float(1.01), False),
        (ComplexSign(), complex(1), True),
        (ComplexSign(), complex(1.01), False),
        (ComplexSign(), complex(-1), True),
        (ComplexSign(), complex(0, -1), True),
        (ComplexSign(), complex(1, 0.01), False),
        (ComplexSign(), sympify(1), True),
        (ComplexSign(), sympify(2), False),
        (ComplexSign(), sympify(-1), True),
        (ComplexSign(), sympify("x"), False),
        (ComplexSign(), sympify(0.01), False),
        (ComplexSign(), sympify("I"), True),
        (ComplexSign(), sympify("-I"), True),
        (ComplexSign(), sympify("1-I"), False),
        (ComplexSign(), Sign(True), True),
        (ComplexSign(), Sign(False), True),
        (ComplexSign(), ComplexSign(0), True),
        (ComplexSign(), ComplexSign(1), True),
        (ComplexSign(), ComplexSign(2), True),
        (ComplexSign(), ComplexSign(3), True),
        (Expr(sympify(1)), int(1), True),
        (Expr(sympify(1)), int(-1), True),
        (Expr(sympify(1)), int(2), True),
        (Expr(sympify(1)), float(1), True),
        (Expr(sympify(1)), float(-1), True),
        (Expr(sympify(1)), float(1.01), True),
        (Expr(sympify(1)), complex(1), True),
        (Expr(sympify(1)), complex(-1), True),
        (Expr(sympify(1)), complex(1, 0.01), True),
        (Expr(sympify(1)), complex(0, -1), True),
        (Expr(sympify(1)), sympify(1), True),
        (Expr(sympify(1)), sympify(-1), True),
        (Expr(sympify(1)), sympify("x"), True),
        (Expr(sympify(1)), sympify(0.01), True),
        (Expr(sympify(1)), sympify("I"), True),
        (Expr(sympify(1)), sympify("-I"), True),
        (Expr(sympify(1)), sympify("1-I"), True),
        (Expr(sympify(1)), Sign(True), True),
        (Expr(sympify(1)), Sign(False), True),
        (Expr(sympify(1)), ComplexSign(0), True),
        (Expr(sympify(1)), ComplexSign(1), True),
        (Expr(sympify(1)), ComplexSign(2), True),
        (Expr(sympify(1)), ComplexSign(3), True),
    ],
)
def test_scalar_typesafe_imul(values_and_validity):
    l, r, valid = values_and_validity
    l_type = type(l)
    if valid:
        l = typesafe_mul(l, r)
    else:
        with pytest.raises(ValueError):
            l = typesafe_mul(l, r)
    if issubclass(l_type, Expr):
        assert isinstance(l, Expr)
    else:
        # stricter in the non-symbolic cases
        assert type(l) is l_type


@pytest.mark.parametrize(
    "test_case",
    [
        (int(1), int, int(1)),
        (int(2), float, float(2)),
        (int(2), complex, complex(2)),
        (int(1), Sign, Sign(0)),
        (int(-1), Sign, Sign(1)),
        (int(2), Sign, None),
        (int(1), ComplexSign, ComplexSign(0)),
        (int(-1), ComplexSign, ComplexSign(2)),
        (int(2), ComplexSign, None),
        (int(2), Expr, sympify(2)),
        (float(1), int, int(1)),
        (float(-2), int, int(-2)),
        (float(-2.1), int, None),
        (float(2), float, float(2)),
        (float(2), complex, complex(2)),
        (float(-1.2), complex, complex(-1.2)),
        (float(2), Expr, sympify(2.0)),
        (float(1), Sign, Sign(0)),
        (float(-1), Sign, Sign(1)),
        (float(-1.01), Sign, None),
        (float(2), Sign, None),
        (float(1), ComplexSign, ComplexSign(0)),
        (float(-1), ComplexSign, ComplexSign(2)),
        (float(1.01), ComplexSign, None),
        (float(2), ComplexSign, None),
        (float(1), Expr, sympify(1.0)),
        (float(1.2), Expr, sympify(1.2)),
        (complex(1), int, int(1)),
        (complex(-2), int, int(-2)),
        (complex(-2.1), int, None),
        (complex(0, 1), int, None),
        (complex(2), float, float(2)),
        (complex(2.2), float, float(2.2)),
        (complex(1, 0.01), float, None),
        (complex(2), complex, complex(2)),
        (complex(-1.2, 1), complex, complex(-1.2, 1)),
        (complex(2, 1), Expr, sympify("2.0+1.0*I")),
        (complex(1), Sign, Sign(0)),
        (complex(-1), Sign, Sign(1)),
        (complex(-1j), Sign, None),
        (complex(-1.01), Sign, None),
        (complex(1 - 1.01j), Sign, None),
        (complex(2), Sign, None),
        (complex(1), ComplexSign, ComplexSign(0)),
        (complex(-1), ComplexSign, ComplexSign(2)),
        (complex(1j), ComplexSign, ComplexSign(1)),
        (complex(-1j), ComplexSign, ComplexSign(3)),
        (complex(0.01 - 1j), ComplexSign, None),
        (complex(1.01), ComplexSign, None),
        (complex(2), ComplexSign, None),
        (complex(1), Expr, sympify("1.0")),
        (complex(1.2), Expr, sympify("1.2")),
        (complex(1 - 1.2j), Expr, sympify("1.0-1.2*I")),
        (Sign(0), int, int(1)),
        (Sign(1), int, int(-1)),
        (Sign(0), float, float(1)),
        (Sign(1), float, float(-1)),
        (Sign(0), complex, complex(1)),
        (Sign(1), complex, complex(-1)),
        (Sign(0), Sign, Sign(0)),
        (Sign(1), Sign, Sign(1)),
        (Sign(0), ComplexSign, ComplexSign(0)),
        (Sign(1), ComplexSign, ComplexSign(2)),
        (Sign(0), Expr, sympify(1)),
        (Sign(1), Expr, sympify(-1)),
        (ComplexSign(0), int, int(1)),
        (ComplexSign(1), int, None),
        (ComplexSign(2), int, int(-1)),
        (ComplexSign(3), int, None),
        (ComplexSign(0), float, float(1)),
        (ComplexSign(1), float, None),
        (ComplexSign(2), float, float(-1)),
        (ComplexSign(3), float, None),
        (ComplexSign(0), complex, complex(1)),
        (ComplexSign(1), complex, complex(1j)),
        (ComplexSign(2), complex, complex(-1)),
        (ComplexSign(3), complex, complex(-1j)),
        (ComplexSign(0), Sign, Sign(0)),
        (ComplexSign(1), Sign, None),
        (ComplexSign(2), Sign, Sign(1)),
        (ComplexSign(3), Sign, None),
        (ComplexSign(0), ComplexSign, ComplexSign(0)),
        (ComplexSign(1), ComplexSign, ComplexSign(1)),
        (ComplexSign(2), ComplexSign, ComplexSign(2)),
        (ComplexSign(3), ComplexSign, ComplexSign(3)),
        (ComplexSign(0), Expr, sympify(1)),
        (ComplexSign(1), Expr, sympify("I")),
        (ComplexSign(2), Expr, sympify(-1)),
        (ComplexSign(3), Expr, sympify("-I")),
        (sympify(1), int, int(1)),
        (sympify(1), float, float(1)),
        (sympify(1), complex, complex(1)),
        (sympify(1), Sign, Sign(0)),
        (sympify(1), ComplexSign, ComplexSign(0)),
        (sympify(1), Expr, sympify(1)),
        (sympify(1.0), int, int(1)),
        (sympify(1.0), float, float(1)),
        (sympify(1.0), complex, complex(1)),
        (sympify(1.0), Sign, Sign(0)),
        (sympify(1.0), ComplexSign, ComplexSign(0)),
        (sympify(1.0), Expr, sympify(1.0)),
        (sympify(0.1), int, None),
        (sympify(0.1), float, float(0.1)),
        (sympify(0.1), complex, complex(0.1)),
        (sympify(0.1), Sign, None),
        (sympify(0.1), ComplexSign, None),
        (sympify(0.1), Expr, sympify(0.1)),
        (sympify("1+I"), int, None),
        (sympify("1+I"), float, None),
        (sympify("1+I"), complex, complex(1, 1)),
        (sympify("1+I"), Sign, None),
        (sympify("1+I"), ComplexSign, None),
        (sympify("1+I"), Expr, sympify("1+I")),
        (sympify("I"), int, None),
        (sympify("I"), float, None),
        (sympify("I"), complex, complex(0, 1)),
        (sympify("I"), Sign, None),
        (sympify("I"), ComplexSign, ComplexSign(1)),
        (sympify("I"), Expr, sympify("I")),
    ],
)
def test_convert(test_case):
    source, t, result = test_case
    if result is not None:
        converted = convert(source, t)
        assert isinstance(converted, t)
        assert converted == result
    else:
        with pytest.raises(ValueError):
            convert(source, t)


def test_sign_vec():
    assert len(SignCoeffs.from_scalar(Sign(False))) == 1
    assert len(SignCoeffs.from_scalar(Sign(False), 5)) == 5
    assert SignCoeffs.from_scalar(Sign(False))[0] == Sign(False)
    v = SignCoeffs.from_size(10)
    assert len(v) == 10
    assert all(c == Sign(False) for c in v)
    v.append(Sign(False))
    assert len(v) == 11
    assert all(c == Sign(False) for c in v)
    v.resize(6)
    assert len(v) == 6
    assert all(c == Sign(False) for c in v)
    v[-1] = Sign(True)
    v[0] = Sign(True)
    assert len(v) == 6
    assert [c for c in v] == [Sign(i) for i in [1, 0, 0, 0, 0, 1]]
    assert v == SignCoeffs.from_phases([1, 0, 0, 0, 0, 1])
    v.extend(SignCoeffs.from_size(1))
    assert len(v) == 7
    assert v == SignCoeffs.from_sequence([Sign(i) for i in [1, 0, 0, 0, 0, 1, 0]])
    v[3] = Sign(True)
    assert v == SignCoeffs.from_sequence([Sign(i) for i in [1, 0, 0, 1, 0, 1, 0]])
    # view it backwards
    assert v[::-1] == SignCoeffs.from_phases([1, 0, 0, 1, 0, 1, 0][::-1])
    # view every second item
    assert v[::2] == SignCoeffs.from_phases([1, 0, 0, 1, 0, 1, 0][::2])
    assert SignCoeffs.from_view(v[::2]) == v[::2]
    v.resize(1)
    v.resize(7)
    assert v == SignCoeffs.from_sequence([Sign(i) for i in [1, 0, 0, 0, 0, 0, 0]])
    assert v.clone() is not v
    assert v.clone() == v
    v = SignCoeffs.from_phases([1, 0, 1, 0, 1, 0])
    v.swap_remove(0)
    assert v == SignCoeffs.from_phases([0, 0, 1, 0, 1])
    v.swap_remove(0)
    assert v == SignCoeffs.from_phases([1, 0, 1, 0])
    v[:] = Sign(True)
    assert all(c == Sign(True) for c in v)
    assert str(v) == "[-1, -1, -1, -1]"
    v[-2:] = Sign(False)
    assert str(v) == "[-1, -1, +1, +1]"
    copy = v[:].clone()
    assert copy == v
    rcopy = v[::-1].clone()
    assert rcopy == copy[::-1]
    assert rcopy[::-1] == copy
    # check that inplace updates are handled correctly
    copy[:] = copy[::-1]
    assert copy == rcopy
    # check iteration
    for i, el in enumerate(v):
        assert el == v[i]
        assert el is not v[i]
    view = v[::2]
    with pytest.raises(ValueError):
        view.resize(1)


def test_complex_sign_vec():
    assert len(ComplexSignCoeffs.from_scalar(ComplexSign(3))) == 1
    assert len(ComplexSignCoeffs.from_scalar(ComplexSign(3), 5)) == 5
    assert ComplexSignCoeffs.from_scalar(ComplexSign(3))[0] == ComplexSign(3)
    v = ComplexSignCoeffs.from_size(10)
    assert len(v) == 10
    assert all(c == ComplexSign(0) for c in v)
    v.append(ComplexSign(0))
    assert len(v) == 11
    assert all(c == ComplexSign(0) for c in v)
    v.resize(6)
    assert len(v) == 6
    assert all(c == ComplexSign(0) for c in v)
    v[-1] = ComplexSign(1)
    v[0] = ComplexSign(2)
    assert len(v) == 6
    assert v == ComplexSignCoeffs.from_phases([2, 0, 0, 0, 0, 1])
    v.extend(ComplexSignCoeffs.from_size(1))
    assert len(v) == 7
    assert v == ComplexSignCoeffs.from_phases([2, 0, 0, 0, 0, 1, 0])
    v[3] = ComplexSign(3)
    assert v == ComplexSignCoeffs.from_phases([2, 0, 0, 3, 0, 1, 0])
    # view it backwards
    assert v[::-1] == ComplexSignCoeffs.from_phases([2, 0, 0, 3, 0, 1, 0][::-1])
    # view every second item
    assert v[::2] == ComplexSignCoeffs.from_phases([2, 0, 0, 3, 0, 1, 0][::2])
    assert v[1::2] == ComplexSignCoeffs.from_phases([2, 0, 0, 3, 0, 1, 0][1::2])
    assert ComplexSignCoeffs.from_view(v[::2]) == v[::2]
    v.resize(1)
    v.resize(7)
    assert v == ComplexSignCoeffs.from_phases([2, 0, 0, 0, 0, 0, 0])
    assert v.clone() is not v
    assert v.clone() == v
    v = ComplexSignCoeffs.from_phases([0, 1, 2, 3, 2, 1])
    v.swap_remove(0)
    assert v == ComplexSignCoeffs.from_phases([1, 1, 2, 3, 2])
    v.swap_remove(0)
    assert v == ComplexSignCoeffs.from_phases([2, 1, 2, 3])
    v[:] = ComplexSign(1)
    assert all(c == ComplexSign(1) for c in v)
    assert str(v) == "[+i, +i, +i, +i]"
    v[-2:] = ComplexSign(2)
    assert str(v) == "[+i, +i, -1, -1]"
    copy = v[:].clone()
    assert copy == v
    rcopy = v[::-1].clone()
    assert rcopy == copy[::-1]
    assert rcopy[::-1] == copy
    # check that inplace updates are handled correctly
    copy[:] = copy[::-1]
    assert copy == rcopy
    # check iteration
    for i, el in enumerate(v):
        assert el == v[i]
        assert el is not v[i]
    view = v[:]
    with pytest.raises(ValueError):
        view.resize(1)


def test_real_vec():
    assert len(RealCoeffs.from_scalar(2.0)) == 1
    assert len(RealCoeffs.from_scalar(2.0, 5)) == 5
    assert RealCoeffs.from_scalar(2.0)[0] == 2
    v = RealCoeffs.from_size(10)
    assert len(v) == 10
    assert all(c == 1 for c in v)
    v.append(1)
    assert len(v) == 11
    assert all(c == 1 for c in v)
    v.resize(6)
    assert len(v) == 6
    assert all(c == 1 for c in v)
    v[-1] = 2.345
    v[0] = -9.8
    assert len(v) == 6
    assert [c for c in v] == [-9.8, 1, 1, 1, 1, 2.345]
    assert v == RealCoeffs.from_sequence([-9.8, 1, 1, 1, 1, 2.345])
    v.extend(RealCoeffs.from_size(1))
    assert len(v) == 7
    assert v == RealCoeffs.from_sequence([-9.8, 1, 1, 1, 1, 2.345, 1])
    v[3] = 0.4
    assert v == RealCoeffs.from_sequence([-9.8, 1, 1, 0.4, 1, 2.345, 1])
    # view it backwards
    assert v[::-1] == RealCoeffs.from_sequence([-9.8, 1, 1, 0.4, 1, 2.345, 1][::-1])
    # view every second item
    assert v[::2] == RealCoeffs.from_sequence([-9.8, 1, 1, 0.4, 1, 2.345, 1][::2])
    assert RealCoeffs.from_view(v[::2]) == v[::2]
    v.resize(1)
    v.resize(7)
    assert v == RealCoeffs.from_sequence([-9.8, 1, 1, 1, 1, 1, 1])
    assert v.clone() is not v
    assert v.clone() == v
    v[3] += 1e-9
    assert v.allclose(RealCoeffs.from_sequence([-9.8, 1, 1, 1, 1, 1, 1]), 1e-5, 0.0)
    v[3] += 1e-5
    assert not v.allclose(RealCoeffs.from_sequence([-9.8, 1, 1, 1, 1, 1, 1]), 1e-5, 0.0)
    v = RealCoeffs.from_sequence([1.0, 2.0, 2.4, -9.8, 1.2])
    v.swap_remove(0)
    assert v == RealCoeffs.from_sequence([1.2, 2.0, 2.4, -9.8])
    v.swap_remove(0)
    assert v == RealCoeffs.from_sequence([-9.8, 2.0, 2.4])
    v[:] = 1
    assert all(c == 1 for c in v)
    assert str(v) == "[1.0, 1.0, 1.0]"
    v[-2:] = 2.0
    assert str(v) == "[1.0, 2.0, 2.0]"
    copy = v[:].clone()
    assert copy == v
    rcopy = v[::-1].clone()
    assert rcopy == copy[::-1]
    assert rcopy[::-1] == copy
    # check that inplace updates are handled correctly
    copy[:] = copy[::-1]
    assert copy == rcopy
    # check iteration
    for i, el in enumerate(v):
        assert el == v[i]
        assert el is not v[i]
    view = v[:]
    with pytest.raises(ValueError):
        view.resize(1)


def test_complex_vec():
    assert len(ComplexCoeffs.from_scalar(2 + 1j)) == 1
    assert len(ComplexCoeffs.from_scalar(2 + 1j, 5)) == 5
    assert ComplexCoeffs.from_scalar(2 + 1j)[0] == 2 + 1j
    v = ComplexCoeffs.from_size(10)
    assert len(v) == 10
    assert all(c == 1 for c in v)
    v.append(1)
    assert len(v) == 11
    assert all(c == 1 for c in v)
    v.resize(6)
    assert len(v) == 6
    assert all(c == 1 for c in v)
    v[-1] = 2.345 + 1j
    v[0] = -9.8 + 1j
    assert len(v) == 6
    example = [-9.8 + 1j, 1, 1, 1, 1, 2.345 + 1j]
    assert [c for c in v] == example
    assert v == ComplexCoeffs.from_sequence(example)
    v.extend(ComplexCoeffs.from_size(1))
    assert len(v) == 7
    example += [1]
    assert v == ComplexCoeffs.from_sequence(example)
    v[3] = 0.4 + 1j
    example[3] = 0.4 + 1j
    assert v == ComplexCoeffs.from_sequence(example)
    # view it backwards
    assert v[::-1] == ComplexCoeffs.from_sequence(example[::-1])
    # view every second item
    assert v[::2] == ComplexCoeffs.from_sequence(example[::2])
    assert ComplexCoeffs.from_view(v[::2]) == v[::2]
    v.resize(1)
    v.resize(7)
    assert v == ComplexCoeffs.from_sequence([-9.8 + 1j, 1, 1, 1, 1, 1, 1])
    assert v.clone() is not v
    assert v.clone() == v
    v[3] += 1e-9
    assert v.allclose(ComplexCoeffs.from_sequence([-9.8 + 1j, 1, 1, 1, 1, 1, 1]), 1e-5, 0.0)
    v[3] += 1e-5
    assert not v.allclose(ComplexCoeffs.from_sequence([-9.8 + 1j, 1, 1, 1, 1, 1, 1]), 1e-5, 0.0)
    v = ComplexCoeffs.from_sequence(example)
    assert v.real_part == RealCoeffs.from_sequence([complex(c).real for c in example])
    assert v.imag_part == RealCoeffs.from_sequence([complex(c).imag for c in example])
    v.resize(4)
    v[:] = 1 + 1j
    assert all(c == 1 + 1j for c in v)
    assert str(v) == "[(1+1j), (1+1j), (1+1j), (1+1j)]"
    v[-2:] = 2.0
    assert str(v) == "[(1+1j), (1+1j), (2+0j), (2+0j)]"
    copy = v[:].clone()
    assert copy == v
    rcopy = v[::-1].clone()
    assert rcopy == copy[::-1]
    assert rcopy[::-1] == copy
    # check that inplace updates are handled correctly
    copy[:] = copy[::-1]
    assert copy == rcopy
    # check iteration
    for i, el in enumerate(v):
        assert el == v[i]
        assert el is not v[i]
    view = v[::-1]
    with pytest.raises(ValueError):
        view.resize(1)


def test_symbolic_vec():
    assert len(SymbolicCoeffs.from_scalar("x")) == 1
    assert len(SymbolicCoeffs.from_scalar("x", 5)) == 5
    assert SymbolicCoeffs.from_scalar("x")[0] == sympify("x")
    v = SymbolicCoeffs.from_size(10)
    assert len(v) == 10
    assert all(c == sympify(1) for c in v)
    v.append(1)
    assert len(v) == 11
    assert all(c == sympify(1) for c in v)
    v.resize(6)
    assert len(v) == 6
    assert all(c == sympify(1) for c in v)
    x = sympify("x")
    v[-1] = 3 * x
    v[0] = x * x
    assert len(v) == 6
    assert [c for c in v] == [sympify(c) for c in ["x**2", 1, 1, 1, 1, "3*x"]]
    assert v == SymbolicCoeffs.from_sequence([sympify(c) for c in ["x**2", 1, 1, 1, 1, "3*x"]])
    v.extend(SymbolicCoeffs.from_size(1))
    assert len(v) == 7
    assert v == SymbolicCoeffs.from_sequence([sympify(c) for c in ["x**2", 1, 1, 1, 1, "3*x", 1]])
    v[3] = sympify("y")
    assert v == SymbolicCoeffs.from_sequence([sympify(c) for c in ["x**2", 1, 1, "y", 1, "3*x", 1]])
    # view it backwards
    assert v[::-1] == SymbolicCoeffs.from_sequence(["x**2", 1, 1, "y", 1, "3*x", 1][::-1])
    # view every second item
    assert v[::2] == SymbolicCoeffs.from_sequence(["x**2", 1, 1, "y", 1, "3*x", 1][::2])
    assert SymbolicCoeffs.from_view(v[::2]) == v[::2]
    v.resize(1)
    v.resize(7)
    assert v == SymbolicCoeffs.from_sequence([sympify(c) for c in ["x**2", 1, 1, 1, 1, 1, 1]])
    assert v == SymbolicCoeffs.from_sequence([sympify(c) for c in ["x*x", 1, 1, 1, 1, 1, 1]])
    assert v.clone() is not v
    assert v.clone() == v
    v = SymbolicCoeffs.from_sequence(
        [sympify(c) for c in ["x**2", "2*x", "y**3", "y", 1, "3*x", 1]]
    )
    assert v.subs({"x": 5}) == SymbolicCoeffs.from_sequence(
        [sympify(c) for c in [25, 10, "y**3", "y", 1, 15, 1]]
    )
    assert v.subs({"y": 2}) == SymbolicCoeffs.from_sequence(
        [sympify(c) for c in ["x**2", "2*x", 8, 2, 1, "3*x", 1]]
    )
    assert v.subs({sympify("x"): 1, "y": 2}) == SymbolicCoeffs.from_sequence(
        [sympify(c) for c in [1, 2, 8, 2, 1, 3, 1]]
    )
    assert v.subs({sympify("x"): 1, "y": 2}).try_to_real() == RealCoeffs.from_sequence(
        [1, 2, 8, 2, 1, 3, 1]
    )
    v.resize(4)
    v[:] = x
    assert all(c == x for c in v)
    assert str(v) == "[x, x, x, x]"
    v[-3:] = sympify("y")
    assert str(v) == "[x, y, y, y]"
    copy = v[:].clone()
    assert copy == v
    rcopy = v[::-1].clone()
    assert rcopy == copy[::-1]
    assert rcopy[::-1] == copy
    # check that inplace updates are handled correctly
    copy[:] = copy[::-1]
    assert copy == rcopy
    # check iteration
    for i, el in enumerate(v):
        assert el == v[i]
        with pytest.xfail(
            reason="Symbolic elements are not actually cloned, but they are immutable."
        ):
            assert el is not v[i]
        el *= sympify(2)
        assert el == v[i]
