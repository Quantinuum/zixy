import pytest
from sympy import sympify

from zixy.container.coeffs import ComplexSign, Sign
from zixy.qubit.pauli import (
    ComplexSignTerm,
    ComplexTerm,
    I,
    RealTerm,
    SignTerm,
    String,
    SymbolicTerm,
    X,
    Y,
    Z,
)


@pytest.mark.parametrize(
    "types",
    [
        (String, String, ComplexSignTerm),
        (String, SignTerm, ComplexSignTerm),
        (String, ComplexSignTerm, ComplexSignTerm),
        (String, RealTerm, ComplexTerm),
        (String, ComplexTerm, ComplexTerm),
        (String, int, RealTerm),
        (String, float, RealTerm),
        (String, complex, ComplexTerm),
        (String, sympify, SymbolicTerm),
        (String, Sign, SignTerm),
        (String, ComplexSign, ComplexSignTerm),
        (SignTerm, SignTerm, ComplexSignTerm),
        (SignTerm, ComplexSignTerm, ComplexSignTerm),
        (SignTerm, RealTerm, ComplexTerm),
        (SignTerm, ComplexTerm, ComplexTerm),
        (SignTerm, int, RealTerm),
        (SignTerm, float, RealTerm),
        (SignTerm, complex, ComplexTerm),
        (SignTerm, sympify, SymbolicTerm),
        (SignTerm, Sign, SignTerm),
        (SignTerm, ComplexSign, ComplexSignTerm),
        (ComplexSignTerm, ComplexSignTerm, ComplexSignTerm),
        (ComplexSignTerm, RealTerm, ComplexTerm),
        (ComplexSignTerm, ComplexTerm, ComplexTerm),
        (ComplexSignTerm, int, ComplexTerm),
        (ComplexSignTerm, float, ComplexTerm),
        (ComplexSignTerm, complex, ComplexTerm),
        (ComplexSignTerm, sympify, SymbolicTerm),
        (ComplexSignTerm, Sign, ComplexSignTerm),
        (ComplexSignTerm, ComplexSign, ComplexSignTerm),
        (RealTerm, RealTerm, ComplexTerm),
        (RealTerm, ComplexTerm, ComplexTerm),
        (RealTerm, int, RealTerm),
        (RealTerm, float, RealTerm),
        (RealTerm, complex, ComplexTerm),
        (RealTerm, sympify, SymbolicTerm),
        (RealTerm, Sign, RealTerm),
        (RealTerm, ComplexSign, ComplexTerm),
        (ComplexTerm, ComplexTerm, ComplexTerm),
        (ComplexTerm, int, ComplexTerm),
        (ComplexTerm, float, ComplexTerm),
        (ComplexTerm, complex, ComplexTerm),
        (ComplexTerm, sympify, SymbolicTerm),
        (ComplexTerm, Sign, ComplexTerm),
        (ComplexTerm, ComplexSign, ComplexTerm),
    ],
)
def test_out_of_place(types):
    value = 1
    l_type, r_type, p_type = types
    assert type(l_type(value) * r_type(value)) is p_type
    # typing is independent of operand ordering
    assert type(r_type(value) * l_type(value)) is p_type


@pytest.mark.parametrize(
    "test_case",
    [
        (String(6, (X,) * 6), String(6, (X,) * 6), String(6, (I,) * 6)),
        (String(6, (X,) * 6), String(6, (Y,) + (X,) * 5), None),
        (SignTerm(6, (X,) * 6), SignTerm(6, (X,) * 6), SignTerm(6, (I,) * 6)),
        (SignTerm(6, (X,) * 6), SignTerm(6, (Y,) + (X,) * 5), None),
    ],
)
def test_in_place(test_case):
    l, r, o = test_case
    l_type = type(l)
    if o is not None:
        l *= r
        assert l == o
    else:
        with pytest.raises(ValueError):
            l *= r
    assert type(l) is l_type


def test_sign_term_products():
    prod = SignTerm(2, (X, X))
    prod *= SignTerm(2, (Z, Z))
    assert type(prod) is SignTerm
    assert str(prod) == "(-1, Y0 Y1)"

    prod = SignTerm(2, (X, X)).cmpnt * SignTerm(2, (Z, Z)).cmpnt
    assert type(prod) is ComplexSignTerm
    assert str(prod) == "(-1, Y0 Y1)"

    prod = SignTerm(2, (X, X)) * SignTerm(2, (Z, Z))
    assert type(prod) is ComplexSignTerm
    assert str(prod) == "(-1, Y0 Y1)"
