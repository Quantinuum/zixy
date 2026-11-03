import pytest
from sympy import Expr, Symbol, sympify

import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.coeffs import ComplexSign, Sign
from zixy.qubit.operator import I, X, Y, Z


@pytest.mark.parametrize(
    ("term_type", "source", "qubits", "expected", "terms_type"),
    (
        (zqo.SignTerm, "(1, X0)", 2, "(+1, X0)", None),
        (zqo.SignTerms, "(1, X0), (-1, Z1), (1, X0)", 2, "(+1, X0), (-1, Z1), (+1, X0)", None),
        (zqo.SignTermSet, "(1, X0), (-1, Z1), (1, X0)", 2, "(+1, X0), (-1, Z1)", zqo.SignTerms),
        (zqo.ComplexSignTerm, "(1j, X0)", 2, "(+i, X0)", None),
        (
            zqo.ComplexSignTerms,
            "(1j, X0), (-1, Z1), (1, X0)",
            2,
            "(+i, X0), (-1, Z1), (+1, X0)",
            None,
        ),
        (
            zqo.ComplexSignTermSet,
            "(1j, X0), (-1, Z1), (1, X0)",
            2,
            "(+1, X0), (-1, Z1)",
            zqo.ComplexSignTerms,
        ),
        (zqo.RealTerm, " ( 2.5 , X0 ) ", 2, "(2.5, X0)", None),
        (
            zqo.RealTerms,
            " (2.0, X0) , (-0.5, Z1) , (3.0, X0) ",
            2,
            "(2.0, X0), (-0.5, Z1), (3.0, X0)",
            None,
        ),
        (
            zqo.RealTermSet,
            "(2.0, X0), (-0.5, Z1), (3.0, X0)",
            2,
            "(3.0, X0), (-0.5, Z1)",
            zqo.RealTerms,
        ),
        (zqo.RealTermSum, "(2.0, X0), (-0.5, Z1), (3.0, X0)", 2, "(5.0, X0), (-0.5, Z1)", None),
        (zqo.ComplexTerm, "(1j, X0)", 2, "(1j, X0)", None),
        (
            zqo.ComplexTerms,
            "((1j), X0), ((2), Z1), ((3), X0)",
            2,
            "(1j, X0), ((2+0j), Z1), ((3+0j), X0)",
            None,
        ),
        (
            zqo.ComplexTermSet,
            "((1j), X0), ((2), Z1), ((3), X0)",
            2,
            "((3+0j), X0), ((2+0j), Z1)",
            zqo.ComplexTerms,
        ),
        (
            zqo.ComplexTermSum,
            "((1j), X0), ((2), Z1), ((3), X0)",
            2,
            "((3+1j), X0), ((2+0j), Z1)",
            None,
        ),
        (zqo.RealTerm, "(2, X0 Y1 Z2)", 3, "(2.0, X0 Y1 Z2)", None),
        (zqo.RealTerms, "\n(1, X0),\n(2, Y1)\n", 2, "(1.0, X0), (2.0, Y1)", None),
        (zqo.RealTermSum, "(2, X0 Y1 Z2), (-2, X0 Y1 Z2)", 3, "(0.0, X0 Y1 Z2)", None),
    ),
)
def test_from_str(term_type, source, qubits, expected, terms_type):
    parsed = term_type.from_str(source, qubits)
    assert str(parsed) == expected

    if terms_type is not None:
        assert parsed == term_type.from_terms(terms_type.from_str(source, qubits))


@pytest.mark.parametrize(
    "term_type",
    (
        zqo.SignTerms,
        zqo.SignTermSet,
        zqo.ComplexSignTerms,
        zqo.ComplexSignTermSet,
        zqo.RealTerms,
        zqo.RealTermSet,
        zqo.RealTermSum,
        zqo.ComplexTerms,
        zqo.ComplexTermSet,
        zqo.ComplexTermSum,
    ),
)
def test_from_str_empty_containers(term_type):
    assert str(term_type.from_str("", 2)) == ""


@pytest.mark.parametrize(
    ("term_type", "source"),
    (
        (zqo.ComplexTerm, "((1j), X0 Y1)"),
        (zqo.ComplexTerms, "((1j), X0), ((2), Y1)"),
        (zqo.ComplexTermSet, "((1j), X0), ((2), Y1)"),
        (zqo.ComplexTermSum, "((1j), X0), ((2), Y1)"),
    ),
)
def test_from_str_round_trip(term_type, source):
    parsed = term_type.from_str(source, 2)
    assert term_type.from_str(str(parsed), 2) == parsed


@pytest.mark.parametrize(
    "term_type", (zqo.SymbolicTerm, zqo.SymbolicTerms, zqo.SymbolicTermSet, zqo.SymbolicTermSum)
)
def test_from_str_symbolic_not_implemented(term_type):
    with pytest.raises(NotImplementedError):
        term_type.from_str("(a, X0)", 2)


@pytest.mark.parametrize("source", ("1, X0", "(1 X0)", "(1, X0, extra)", "(, X0)", "(1, )"))
def test_from_str_errors(source):
    with pytest.raises(ValueError):
        zqo.RealTerm.from_str(source, 2)

    with pytest.raises(IndexError):
        zqo.RealTerm.from_str("(1, X2)", 2)

    with pytest.raises(ValueError):
        zqo.SignTerm.from_str("(2, X0)", 2)

    with pytest.raises(ValueError):
        zqo.ComplexSignTerm.from_str("(2, X0)", 2)

    with pytest.raises(ValueError):
        zqo.RealTerm.from_str("(1j, X0)", 2)


def test_real_term():
    with pytest.raises(IndexError):
        zqo.RealTerm(5, ((X, Y, Z, X, Y, Z), 2.0))
    lhs = zqo.RealTerm(6, ((X, Y, Z, X, Y, Z), 2.0))
    # in-place mul by non-complex scalar types is valid
    lhs *= 1
    lhs *= float(1)
    lhs *= Sign()
    assert type(lhs.coeff) is float
    assert lhs.coeff == 2.0
    rhs = zqo.RealTerm(6, ((Y, Y, X, Y, Z, Y), 4.0))
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
        == "Cannot multiply <class 'zixy.qubit.operator._terms.RealTerm'> (2.0, X0 Y1 Z2 X3 Y4 Z5) "
        "in-place by <class 'zixy.qubit.operator._strings.String'> Y0 Y1 X2 Y3 Z4 Y5. Pauli string "
        "multiplication gives a ComplexSign factor that is not <class 'float'> representable."
    )
    assert lhs.string.get_tuple() == (X, Y, Z, X, Y, Z)
    with pytest.raises(ValueError) as err:
        lhs *= rhs
    assert (
        str(err.value)
        == "Cannot multiply <class 'zixy.qubit.operator._terms.RealTerm'> (2.0, X0 Y1 Z2 X3 Y4 Z5) "
        "in-place by <class 'zixy.qubit.operator._terms.RealTerm'> (4.0, Y0 Y1 X2 Y3 Z4 Y5). Pauli "
        "string multiplication gives a ComplexSign factor that is not <class 'float'> representable."
    )
    assert lhs.string.get_tuple() == (X, Y, Z, X, Y, Z)
    assert lhs.coeff == 2.0
    with pytest.raises(ValueError) as err:
        lhs *= ComplexSign(3)
    assert (
        str(err.value)
        == "Cannot multiply <class 'zixy.qubit.operator._terms.RealTerm'> (2.0, X0 Y1 Z2 X3 Y4 Z5) "
        "in-place by <class 'zixy.container.coeffs.ComplexSign'> -i. Coefficient -i is not "
        "<class 'float'> representable."
    )
    assert lhs.string.get_tuple() == (X, Y, Z, X, Y, Z)

    assert rhs.coeff == 4.0
    with pytest.raises(TypeError):
        lhs.coeff *= "invalid"
    with pytest.raises(ValueError):
        lhs.coeff *= complex(1, 0.1)

    tmp = zqo.ComplexTerm(6, (Y, Y, X, Y, Z, Y))
    assert str(tmp) == "((1+0j), Y0 Y1 X2 Y3 Z4 Y5)"
    tmp.coeff *= 12
    assert type(tmp.coeff) is complex
    assert tmp.coeff == complex(12, 0)

    product = lhs * rhs
    assert type(product) is zqo.ComplexTerm
    assert type(product.coeff) is complex
    assert product.coeff == -8.0j

    assert str(product) == "(-8j, Z0 Y2 Z3 X4 X5)"

    term = zqo.RealTerm.from_str("X0 Y1")
    assert str(term) == "(1.0, X0 Y1)"
    assert term.coeff == 1.0


def test_term_scalar_mul_preserves_viewed_string():
    terms = zqo.RealTerms.from_str("X0, Z1", 2)

    right_scaled = terms[1] * 3.0
    left_scaled = 3.0 * terms[1]

    assert right_scaled.string == terms[1].string
    assert right_scaled.string != terms[0].string
    assert right_scaled.coeff == 3.0
    assert not right_scaled.string.aliases(terms[1].string)
    assert not right_scaled.string.aliases(terms[0].string)
    assert left_scaled.string == terms[1].string
    assert left_scaled.string != terms[0].string
    assert left_scaled.coeff == 3.0
    assert not left_scaled.string.aliases(terms[1].string)
    assert not left_scaled.string.aliases(terms[0].string)


def test_complex_term():
    with pytest.raises(IndexError):
        zqo.ComplexTerm(5, ((X, Y, Z, X, Y, Z), 2.0j))
    lhs = zqo.ComplexTerm(6, ((X, Y, Z, X, Y, Z), 2.0j))
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


def test_real_apply_rejects_different_qubits():
    op = zqo.RealTermSum.from_str("X0", 2)
    state = zqs.RealTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        op.apply(state)


def test_real_mat_elem_rejects_different_qubits():
    op = zqo.RealTermSum.from_str("X0", 2)
    bra = zqs.RealTermSum.from_str("[1, 0]", 2)
    ket = zqs.RealTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        op.mat_elem(bra, ket)


def test_complex_apply_rejects_different_qubits():
    op = zqo.ComplexTermSum.from_str("X0", 2)
    state = zqs.ComplexTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        op.apply(state)


def test_complex_mat_elem_rejects_different_qubits():
    op = zqo.ComplexTermSum.from_str("X0", 2)
    bra = zqs.ComplexTermSum.from_str("[1, 0]", 2)
    ket = zqs.ComplexTermSum.from_str("[1, 0, 0]", 3)

    with pytest.raises(ValueError, match="different qubits"):
        op.mat_elem(bra, ket)


def test_real_terms():
    terms = zqo.RealTerms(6)
    assert len(terms) == 0
    terms.resize(10)
    assert sum(terms.coeffs) == 10
    assert len(terms) == 10
    assert terms[0].string.is_identity()
    assert terms[0].string.is_identity()
    assert terms[0] is not terms[0]
    assert type(terms[:4]) is zqo.RealTerms
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
    terms[0] = zqo.String(6, (X,) * 6)
    assert terms[0].coeff == 1.0
    terms[0] = zqo.RealTerm(6, (X,) * 6)
    assert terms[0].string.get_tuple() == (X,) * 6
    assert type(terms.into(zqo.ComplexTerms)) is zqo.ComplexTerms
    assert all(terms.into(zqo.ComplexTerms).coeffs.np_array == 1)
    terms[1:4].scale(-1)
    assert type(terms.into(zqo.SignTerms)) is zqo.SignTerms
    assert str(terms.into(zqo.SignTerms).coeffs) == "[+1, -1, -1, -1, +1, +1, +1, +1, +1, +1]"


def test_complex_terms():
    terms = zqo.ComplexTerms(6)
    assert len(terms) == 0
    terms.resize(10)
    assert sum(terms.coeffs) == 10
    assert len(terms) == 10
    assert terms[0].string.is_identity()
    assert terms[0].string.is_identity()
    assert terms[0] is not terms[0]
    assert type(terms[:4]) is zqo.ComplexTerms
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
    terms[0] = zqo.ComplexTerm(6, (X,) * 6)
    assert terms[0].string.get_tuple() == (X,) * 6
    terms[0].coeff = 1.23
    assert terms[0].coeff == 1.23
    terms[0] = zqo.String(6, (X,) * 6)
    assert terms[0].coeff == 1.0
    terms[0] = zqo.ComplexTerm(6, ((X,) * 6, 1.23))
    assert terms[0].string.get_tuple() == (X,) * 6
    assert type(terms[0].into(zqo.RealTerm)) is zqo.RealTerm
    assert terms[0].into(zqo.RealTerm).coeff == 1.23
    assert terms[0].into(zqo.RealTerm).string.get_tuple() == (X,) * 6
    assert type(terms.into(zqo.RealTerms)) is zqo.RealTerms
    assert type(terms.into(zqo.RealTerms).strings) is type(terms.strings)
    assert terms.into(zqo.RealTerms).strings is not terms.strings
    terms.coeffs[-1] *= 1j
    with pytest.raises(ValueError):
        terms.into(zqo.RealTerms)


def test_symbolic_terms():
    terms = zqo.SymbolicTerms(6)
    assert len(terms) == 0
    terms.resize(10)
    assert sum(terms.coeffs) == 10
    assert len(terms) == 10
    terms.append()
    assert len(terms._impl._strings) == len(terms._impl._coeffs)
    assert sum(terms.coeffs) == 11
    assert len(terms) == 11
    terms.resize(10)
    assert len(terms._impl._strings) == len(terms._impl._coeffs)
    assert sum(terms.coeffs) == 10
    assert len(terms) == 10
    assert terms[0].string.is_identity()
    assert terms[0].string.is_identity()
    assert terms[0] is not terms[0]
    assert type(terms[:4]) is zqo.SymbolicTerms
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
    terms[0] = zqo.String(6, (X,) * 6)
    assert terms[0].coeff == 1
    terms[0] = zqo.SymbolicTerm(6, (X,) * 6)
    assert terms[0].string.get_tuple() == (X,) * 6
    real_terms = terms.try_to_real()
    assert type(real_terms) is zqo.RealTerms
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
    assert isinstance(v, zqo.SymbolicTerm)
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
    a = zqo.RealTerms(n_qubit)
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
