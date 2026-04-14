import pytest
from sympy import Expr, Symbol, sympify

from zixy.container.coeffs import ComplexSign, Sign
from zixy.qubit.pauli import (
    ComplexTerm,
    ComplexTerms,
    I,
    RealTerm,
    RealTerms,
    SignTerms,
    String,
    SymbolicTerm,
    SymbolicTerms,
    X,
    Y,
    Z,
)


def test_real_term():
    with pytest.raises(IndexError):
        RealTerm(5, ((X, Y, Z, X, Y, Z), 2.0))
    lhs = RealTerm(6, ((X, Y, Z, X, Y, Z), 2.0))
    # in-place mul by non-complex scalar types is valid
    lhs *= 1
    lhs *= float(1)
    lhs *= Sign()
    assert type(lhs.coeff) is float
    assert lhs.coeff == 2.0
    rhs = RealTerm(6, ((Y, Y, X, Y, Z, Y), 4.0))
    assert type(rhs.coeff) is float
    assert rhs.coeff == 4.0
    # can't in-place multiply by imag complex
    with pytest.raises(ValueError):
        rhs.coeff *= complex(0, 1)
    with pytest.raises(ValueError):
        rhs *= complex(0, 1)
    # or imag ComplexSign
    with pytest.raises(ValueError):
        rhs.coeff *= ComplexSign(1)
    with pytest.raises(ValueError):
        rhs *= ComplexSign(1)
    # but real values of complex types are fine:
    rhs.coeff *= complex(-1)
    assert type(rhs.coeff) is float
    assert rhs.coeff == -4.0
    rhs.coeff *= ComplexSign(2)
    assert type(rhs.coeff) is float
    assert rhs.coeff == 4.0
    assert lhs.string.phase_of_mul(rhs.string) == ComplexSign(3)
    # string product is imaginary, so can't in-place multiply by String or Term
    with pytest.raises(ValueError) as err:
        lhs *= rhs.string
    assert (
        str(err.value)
        == "Cannot multiply <class 'zixy.qubit.pauli._terms.RealTerm'> (2.0, X0 Y1 Z2 X3 Y4 Z5) "
        "in-place by <class 'zixy.qubit.pauli._strings.String'> Y0 Y1 X2 Y3 Z4 Y5. Pauli string "
        "multiplication gives a ComplexSign factor that is not <class 'float'> representable."
    )
    assert lhs.string.get_tuple() == (X, Y, Z, X, Y, Z)
    with pytest.raises(ValueError) as err:
        lhs *= rhs
    assert (
        str(err.value)
        == "Cannot multiply <class 'zixy.qubit.pauli._terms.RealTerm'> (2.0, X0 Y1 Z2 X3 Y4 Z5) "
        "in-place by <class 'zixy.qubit.pauli._terms.RealTerm'> (4.0, Y0 Y1 X2 Y3 Z4 Y5). Pauli "
        "string multiplication gives a ComplexSign factor that is not <class 'float'> representable."
    )
    assert lhs.string.get_tuple() == (X, Y, Z, X, Y, Z)
    assert lhs.coeff == 2.0
    with pytest.raises(ValueError) as err:
        lhs *= ComplexSign(3)
    assert (
        str(err.value)
        == "Cannot multiply <class 'zixy.qubit.pauli._terms.RealTerm'> (2.0, X0 Y1 Z2 X3 Y4 Z5) "
        "in-place by <class 'zixy.container.coeffs.ComplexSign'> -i. Coefficient -i is not "
        "<class 'float'> representable."
    )
    assert lhs.string.get_tuple() == (X, Y, Z, X, Y, Z)

    assert rhs.coeff == 4.0
    with pytest.raises(TypeError):
        lhs.coeff *= "invalid"
    with pytest.raises(ValueError):
        lhs.coeff *= complex(1, 0.1)

    tmp = ComplexTerm(6, (Y, Y, X, Y, Z, Y))
    assert str(tmp) == "((1+0j), Y0 Y1 X2 Y3 Z4 Y5)"
    tmp.coeff *= 12
    assert type(tmp.coeff) is complex
    assert tmp.coeff == complex(12, 0)

    product = lhs * rhs
    assert type(product) is ComplexTerm
    assert type(product.coeff) is complex
    assert product.coeff == -8.0j

    assert str(product) == "(-8j, Z0 Y2 Z3 X4 X5)"


def test_complex_term():
    with pytest.raises(IndexError):
        ComplexTerm(5, ((X, Y, Z, X, Y, Z), 2.0j))
    lhs = ComplexTerm(6, ((X, Y, Z, X, Y, Z), 2.0j))
    assert type(lhs.coeff) is complex
    assert lhs.coeff == complex(0, 2)
    # in-place mul by any of the scalar types is valid
    lhs *= 1
    lhs *= float(1)
    lhs *= complex(1)
    lhs *= Sign()
    lhs *= ComplexSign()
    assert lhs.coeff == complex(0, 2)
    lhs *= lhs.string
    assert lhs.coeff == complex(0, 2)
    assert lhs.string.is_identity()


def test_real_terms():
    terms = RealTerms(6)
    assert len(terms) == 0
    terms.resize(10)
    assert sum(terms.coeffs) == 10
    assert len(terms) == 10
    assert terms[0].string.is_identity()
    assert terms[0].string.is_identity()
    assert terms[0] is not terms[0]
    assert type(terms[:4]) is RealTerms
    assert len(terms[:4]) == 4
    assert len(terms[:4].strings) == 4
    assert len(terms[:4].coeffs) == 4
    assert len(terms[:4]) == 4
    terms[4].string.set((X, Y, Z, Z, Y, X))
    assert str(terms[2:6]) == "(1.0, ), (1.0, ), (1.0, X0 Y1 Z2 Z3 Y4 X5), (1.0, )"
    for i, term in enumerate(terms):
        if i == 4:
            assert term.string.get_tuple() == (X, Y, Z, Z, Y, X)
        else:
            assert term.string.is_identity()
    terms[0].coeff = 1.23
    assert terms[0].coeff == 1.23
    terms[0] = String(6, (X,) * 6)
    assert terms[0].coeff == 1.0
    terms[0] = RealTerm(6, (X,) * 6)
    assert terms[0].string.get_tuple() == (X,) * 6
    assert type(terms.into(ComplexTerms)) is ComplexTerms
    assert all(terms.into(ComplexTerms).coeffs.np_array == 1)
    terms[1:4].scale(-1)
    assert type(terms.into(SignTerms)) is SignTerms
    assert str(terms.into(SignTerms).coeffs) == "[+1, -1, -1, -1, +1, +1, +1, +1, +1, +1]"


def test_complex_terms():
    terms = ComplexTerms(6)
    assert len(terms) == 0
    terms.resize(10)
    assert sum(terms.coeffs) == 10
    assert len(terms) == 10
    assert terms[0].string.is_identity()
    assert terms[0].string.is_identity()
    assert terms[0] is not terms[0]
    assert type(terms[:4]) is ComplexTerms
    assert len(terms[:4]) == 4
    assert len(terms[:4].strings) == 4
    assert len(terms[:4].coeffs) == 4
    assert len(terms[:4]) == 4
    terms[4].string.set((X, Y, Z, Z, Y, X))
    assert str(terms[2:6]) == "((1+0j), ), ((1+0j), ), ((1+0j), X0 Y1 Z2 Z3 Y4 X5), ((1+0j), )"
    for i, term in enumerate(terms):
        if i == 4:
            assert term.string.get_tuple() == (X, Y, Z, Z, Y, X)
        else:
            assert term.string.is_identity()
    terms[0] = ComplexTerm(6, (X,) * 6)
    assert terms[0].string.get_tuple() == (X,) * 6
    terms[0].coeff = 1.23
    assert terms[0].coeff == 1.23
    terms[0] = String(6, (X,) * 6)
    assert terms[0].coeff == 1.0
    terms[0] = ComplexTerm(6, ((X,) * 6, 1.23))
    assert terms[0].string.get_tuple() == (X,) * 6
    assert type(terms[0].into(RealTerm)) is RealTerm
    assert terms[0].into(RealTerm).coeff == 1.23
    assert terms[0].into(RealTerm).cmpnt.get_tuple() == (X,) * 6
    assert type(terms.into(RealTerms)) is RealTerms
    assert type(terms.into(RealTerms).strings) is type(terms.strings)
    assert terms.into(RealTerms).strings is not terms.strings
    terms.coeffs[-1] *= 1j
    with pytest.raises(ValueError):
        terms.into(RealTerms)


def test_symbolic_terms():
    terms = SymbolicTerms(6)
    assert len(terms) == 0
    terms.resize(10)
    assert sum(terms.coeffs) == 10
    assert len(terms) == 10
    terms.append()
    assert len(terms._impl._cmpnts) == len(terms._impl._coeffs)
    assert sum(terms.coeffs) == 11
    assert len(terms) == 11
    terms.resize(10)
    assert len(terms._impl._cmpnts) == len(terms._impl._coeffs)
    assert sum(terms.coeffs) == 10
    assert len(terms) == 10
    assert terms[0].string.is_identity()
    assert terms[0].string.is_identity()
    assert terms[0] is not terms[0]
    assert type(terms[:4]) is SymbolicTerms
    assert len(terms[:4]) == 4
    assert len(terms[:4].strings) == 4
    assert len(terms[:4].coeffs) == 4
    assert len(terms[:4]) == 4
    terms[4].string.set((X, Y, Z, Z, Y, X))
    assert str(terms[2:6]) == "(1, ), (1, ), (1, X0 Y1 Z2 Z3 Y4 X5), (1, )"
    for i, term in enumerate(terms):
        if i == 4:
            assert term.string.get_tuple() == (X, Y, Z, Z, Y, X)
        else:
            assert term.string.is_identity()
    terms[0].coeff = 1.23
    assert isinstance(terms[0].coeff, Expr)
    assert terms[0].coeff == sympify(1.23)
    terms[0] = String(6, (X,) * 6)
    assert terms[0].coeff == 1
    terms[0] = SymbolicTerm(6, (X,) * 6)
    assert terms[0].string.get_tuple() == (X,) * 6
    real_terms = terms.try_to_real()
    assert type(real_terms) is RealTerms
    assert len(real_terms) == len(terms)
    terms[0] *= sympify(1j)
    # now with the imag unit, should not be able to convert to real
    with pytest.raises(TypeError):
        terms.try_to_real()
    # but can be converted to complex
    assert all(
        terms.try_to_complex().coeffs.np_array
        == [
            1j,
        ]
        + [1] * 9
    )
    v = terms[-1].clone()
    assert isinstance(v, SymbolicTerm)
    assert len(v._impl) == 1
    assert bool(v.coeff)
    assert v.coeff == terms.coeffs[-1]
    v *= Symbol("x")
    terms[-1] *= Symbol("x")
    terms[-2] *= Symbol("y")
    # but not if there are free symbols
    assert len(terms.free_symbols) == 2
    assert terms.free_symbols == {Symbol("x"), Symbol("y")}
    with pytest.raises(TypeError):
        terms.try_to_complex()

    assert terms[-1].clone().subs({"x": 3}).try_to_real().coeff == 3

    # in-place subs in the terms array:
    terms.isubs({"x": 5})
    assert len(terms.free_symbols) == 1
    with pytest.raises(TypeError):
        terms.try_to_complex()
    terms[-2].isubs({"y": 4})
    assert len(terms.free_symbols) == 0
    assert all(terms.try_to_complex().coeffs.np_array == [1j, 1, 1, 1, 1, 1, 1, 1, 4, 5])


def test_append_iterable():
    n_qubit = 6
    tuples = (
        (X, Y, Z, Z, Y, X),
        (Y, Z, Z, I, X, Z),
    )
    a = RealTerms(n_qubit)
    a.append_iterable(tuples)
    assert a[0].string.get_tuple() == (X, Y, Z, Z, Y, X)
    assert a[1].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert a[0].string.get_tuple() == (X, Y, Z, Z, Y, X)
    assert a[1].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert len(a[:]) == len(a)
    assert len(a[0 : len(a) : 1]) == len(a)
    # view them in reverse order
    assert len(a[len(a) :: -1]) == len(a)
    assert len(a[::-1]) == len(a)
    assert a[::-1][0].string.get_tuple() == (Y, Z, Z, I, X, Z)
    assert a[::-1][1].string.get_tuple() == (X, Y, Z, Z, Y, X)
