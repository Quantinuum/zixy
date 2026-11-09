import pytest
from sympy import sympify

import zixy.qubit.operator as zqo
from zixy.container.coeffs import ComplexSign, Sign
from zixy.qubit.operator import I, SymplecticPart, X, Y, Z


@pytest.mark.parametrize(
    "types",
    [
        (zqo.String, zqo.String, zqo.ComplexSignTerm),
        (zqo.String, zqo.SignTerm, zqo.ComplexSignTerm),
        (zqo.String, zqo.ComplexSignTerm, zqo.ComplexSignTerm),
        (zqo.String, zqo.RealTerm, zqo.ComplexTerm),
        (zqo.String, zqo.ComplexTerm, zqo.ComplexTerm),
        (zqo.String, int, zqo.RealTerm),
        (zqo.String, float, zqo.RealTerm),
        (zqo.String, complex, zqo.ComplexTerm),
        (zqo.String, sympify, zqo.SymbolicTerm),
        (zqo.String, Sign, zqo.SignTerm),
        (zqo.String, ComplexSign, zqo.ComplexSignTerm),
        (zqo.SignTerm, zqo.SignTerm, zqo.ComplexSignTerm),
        (zqo.SignTerm, zqo.ComplexSignTerm, zqo.ComplexSignTerm),
        (zqo.SignTerm, zqo.RealTerm, zqo.ComplexTerm),
        (zqo.SignTerm, zqo.ComplexTerm, zqo.ComplexTerm),
        (zqo.SignTerm, int, zqo.RealTerm),
        (zqo.SignTerm, float, zqo.RealTerm),
        (zqo.SignTerm, complex, zqo.ComplexTerm),
        (zqo.SignTerm, sympify, zqo.SymbolicTerm),
        (zqo.SignTerm, Sign, zqo.SignTerm),
        (zqo.SignTerm, ComplexSign, zqo.ComplexSignTerm),
        (zqo.ComplexSignTerm, zqo.ComplexSignTerm, zqo.ComplexSignTerm),
        (zqo.ComplexSignTerm, zqo.RealTerm, zqo.ComplexTerm),
        (zqo.ComplexSignTerm, zqo.ComplexTerm, zqo.ComplexTerm),
        (zqo.ComplexSignTerm, int, zqo.ComplexTerm),
        (zqo.ComplexSignTerm, float, zqo.ComplexTerm),
        (zqo.ComplexSignTerm, complex, zqo.ComplexTerm),
        (zqo.ComplexSignTerm, sympify, zqo.SymbolicTerm),
        (zqo.ComplexSignTerm, Sign, zqo.ComplexSignTerm),
        (zqo.ComplexSignTerm, ComplexSign, zqo.ComplexSignTerm),
        (zqo.RealTerm, zqo.RealTerm, zqo.ComplexTerm),
        (zqo.RealTerm, zqo.ComplexTerm, zqo.ComplexTerm),
        (zqo.RealTerm, int, zqo.RealTerm),
        (zqo.RealTerm, float, zqo.RealTerm),
        (zqo.RealTerm, complex, zqo.ComplexTerm),
        (zqo.RealTerm, sympify, zqo.SymbolicTerm),
        (zqo.RealTerm, Sign, zqo.RealTerm),
        (zqo.RealTerm, ComplexSign, zqo.ComplexTerm),
        (zqo.ComplexTerm, zqo.ComplexTerm, zqo.ComplexTerm),
        (zqo.ComplexTerm, int, zqo.ComplexTerm),
        (zqo.ComplexTerm, float, zqo.ComplexTerm),
        (zqo.ComplexTerm, complex, zqo.ComplexTerm),
        (zqo.ComplexTerm, sympify, zqo.SymbolicTerm),
        (zqo.ComplexTerm, Sign, zqo.ComplexTerm),
        (zqo.ComplexTerm, ComplexSign, zqo.ComplexTerm),
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
        (zqo.String(6, (X,) * 6), zqo.String(6, (X,) * 6), zqo.String(6, (I,) * 6)),
        (zqo.String(6, (X,) * 6), zqo.String(6, (Y,) + (X,) * 5), None),
        (zqo.SignTerm(6, (X,) * 6), zqo.SignTerm(6, (X,) * 6), zqo.SignTerm(6, (I,) * 6)),
        (zqo.SignTerm(6, (X,) * 6), zqo.SignTerm(6, (Y,) + (X,) * 5), None),
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
    prod = zqo.SignTerm(2, (X, X))
    prod *= zqo.SignTerm(2, (Z, Z))
    assert type(prod) is zqo.SignTerm
    assert str(prod) == "(-1, Y0 Y1)"

    prod = zqo.SignTerm(2, (X, X)).string * zqo.SignTerm(2, (Z, Z)).string
    assert type(prod) is zqo.ComplexSignTerm
    assert str(prod) == "(-1, Y0 Y1)"

    prod = zqo.SignTerm(2, (X, X)) * zqo.SignTerm(2, (Z, Z))
    assert type(prod) is zqo.ComplexSignTerm
    assert str(prod) == "(-1, Y0 Y1)"


def test_sign_terms_canonicalize():
    tuples = (
        (Z, Z, Z),
        (X, X, I),
        (I, X, X),
    )
    terms = zqo.SignTerms(3)
    terms.extend(tuples)
    imul_ops = terms.canonicalize_all()
    # check imul ops are done in the right order
    correct_ops = [
        # move X0 up
        (0, 1),
        # eliminate X0s
        (1, 0),
        # move X1 up
        (1, 2),
        # eliminate X1s
        (0, 1),
        (2, 1),
        # X2 already solved
        # Z0 already in place
        # eliminate Z0s
        (1, 2),
    ]
    assert imul_ops == correct_ops
    # check final object is in canonical form
    assert terms[0].string.get_tuple() == (X, I, X)
    assert terms[1].string.get_tuple() == (I, X, X)
    assert terms[2].string.get_tuple() == (Z, Z, Z)
    # use manual canonicalization order to revert back
    imul_ops = terms.canonicalize(
        [(0, SymplecticPart.Z), (0, SymplecticPart.X), (2, SymplecticPart.X)], [0, 1, 2], []
    )
    correct_ops = [
        # move Z0 up
        (0, 2),
        # eliminate Z0s
        (2, 0),
        # move X0 up
        (1, 2),
        # eliminate X0s
        (0, 1),
        (2, 1),
        # X2 already in place
        # eliminate X2s
        (0, 2),
    ]
    assert imul_ops == correct_ops
    assert terms[0].string.get_tuple() == tuples[0]
    assert terms[1].string.get_tuple() == tuples[1]
    assert terms[2].string.get_tuple() == tuples[2]

    null = zqo.SignTerms(0)
    # the unique tableau with 0 qubits is in canonical form, so no imuls
    assert null.canonicalize_all() == []

    empty = zqo.SignTerms(3)
    # the unique tableau with no strings is in canonical form, so no imuls
    assert empty.canonicalize_all() == []
