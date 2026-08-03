import pytest
from sympy import sympify

from zixy.container.coeffs import ComplexSign, Sign
from zixy.fermion.operator.normal import (
    ComplexTerm,
    ComplexTerms,
    ComplexTermSet,
    ComplexTermSum,
    RealTerm,
    RealTerms,
    RealTermSet,
    RealTermSum,
    SignTerm,
    String,
    SymbolicTerm,
    SymbolicTerms,
    SymbolicTermSum,
)


def test_real_term():
    term = RealTerm(4, ("F0^ F1", 2.0))
    assert type(term.coeff) is float
    assert term.coeff == 2.0
    assert term.string.get_sets() == ([0], [1])

    scaled = term * 3
    assert type(scaled) is RealTerm
    assert str(scaled) == "(6.0, F0^ F1)"

    complex_scaled = term * 1j
    assert type(complex_scaled) is ComplexTerm
    assert str(complex_scaled) == "(2j, F0^ F1)"

    symbolic_scaled = term * sympify("a")
    assert type(symbolic_scaled) is SymbolicTerm
    assert str(symbolic_scaled) == "(2.0*a, F0^ F1)"

    sign_scaled = SignTerm(4, "F0^ F1") * Sign()
    assert type(sign_scaled) is SignTerm
    assert str(sign_scaled) == "(+1, F0^ F1)"

    complex_sign_scaled = term * ComplexSign(1)
    assert type(complex_sign_scaled) is ComplexTerm
    assert str(complex_sign_scaled) == "(2j, F0^ F1)"

    out_of_order = RealTerm.from_str("F1 F0^", 4)
    assert str(out_of_order) == "(-1.0, F0^ F1)"
    assert str(out_of_order.daggered()) == "(-1.0, F1^ F0)"


def test_complex_term():
    term = ComplexTerm(3, ("F0^ F1", 1 + 2j))
    assert type(term.coeff) is complex
    assert term.coeff == 1 + 2j

    scaled = term * 2
    assert type(scaled) is ComplexTerm
    assert str(scaled) == "((2+4j), F0^ F1)"

    adjoint = ComplexTerm.from_str("(1j, F0^ F1)", 2).daggered()
    assert str(adjoint) == "(-1j, F1^ F0)"


def test_real_terms():
    terms = RealTerms(4)
    assert len(terms) == 0
    terms.resize(5)

    assert len(terms) == 5
    assert len(terms[:3].strings) == 3
    assert len(terms[:3].coeffs) == 3
    assert str(terms[0]) == "(1.0, )"

    terms[1].string.set("F0^ F1")
    terms[1].coeff = 2.5
    terms[2] = String(4, "F2^")
    terms[3] = RealTerm(4, ("F3", -1.0))

    assert str(terms[1:4]) == "(2.5, F0^ F1), (1.0, F2^), (-1.0, F3)"
    assert type(terms.into(ComplexTerms)) is ComplexTerms
    assert tuple(terms.into(ComplexTerms).coeffs) == (
        1 + 0j,
        2.5 + 0j,
        1 + 0j,
        -1 + 0j,
        1 + 0j,
    )


def test_complex_terms():
    terms = ComplexTerms(3)
    terms.append_iterable(
        (
            ("F0^ F1", 1 + 2j),
            ("F2", -1j),
        )
    )

    assert type(terms[0]) is ComplexTerm
    assert str(terms) == "((1+2j), F0^ F1), ((-0-1j), F2)"

    terms[0].coeff = 1.25
    terms[1].coeff = -2.0
    assert terms[0].coeff == 1.25 + 0j
    assert type(terms.into(RealTerms)) is RealTerms
    terms[1].coeff = 1j
    with pytest.raises(ValueError):
        terms.into(RealTerms)


def test_symbolic_terms():
    a = sympify("a")
    terms = SymbolicTerms(3)
    terms.resize(3)
    terms[1] = ("F0^ F1", a)
    terms[2] = SymbolicTerm(3, ("F2", 2 * a))

    assert str(terms) == "(1, ), (a, F0^ F1), (2*a, F2)"
    assert str(terms[1].subs({a: 3})) == "(3, F0^ F1)"
    assert str(terms.subs({a: 3})) == "(1, ), (3, F0^ F1), (6, F2)"

    term_sum = SymbolicTermSum.from_str("(a, F0^ F1)", 2)
    assert str(term_sum.subs({a: 2})) == "(2, F0^ F1)"


def test_append_iterable():
    terms = RealTerms(4)
    terms.append_iterable(
        (
            ("F0^ F1", 2.0),
            ("F2", -1.0),
            ("F3^", 0.5),
        )
    )

    assert len(terms[:]) == 3
    assert str(terms[::-1]) == "(0.5, F3^), (-1.0, F2), (2.0, F0^ F1)"
    assert terms.clone() == terms


def test_real_term_sum():
    term_sum = RealTermSum(4)
    term_sum += RealTerm(4, ("F0^ F1", 2.0))
    term_sum += RealTerm(4, ("F0^ F1", -0.5))
    term_sum += RealTerm(4, ("F2^", 1.0))

    assert len(term_sum) == 2
    assert term_sum["F0^ F1"] == 1.5
    assert str(term_sum) == "(1.5, F0^ F1), (1.0, F2^)"
    assert term_sum.l1_norm == 2.5
    assert term_sum.l2_norm == pytest.approx((1.5**2 + 1.0**2) ** 0.5)
    assert str(term_sum.filter_significant(atol=1.1)) == "(1.5, F0^ F1)"

    contraction = RealTermSum.from_str("F0 F0^", 2)
    assert str(contraction) == "(1.0, ), (-1.0, F0^ F0)"


def test_real_term_add_iterable():
    term_sum = RealTermSum.from_iterable(
        (
            RealTerm(4, ("F0^ F1", 1.0)),
            RealTerm(4, ("F0^ F1", -1.0)),
            RealTerm(4, ("F2", 0.25)),
        ),
        4,
    )

    assert str(term_sum) == "(0.0, F0^ F1), (0.25, F2)"
    assert str(term_sum.filter_nonzero()) == "(0.25, F2)"
    term_sum["F2"] = 0.0
    assert str(term_sum.filter_nonzero()) == ""


def test_real_term_into_other_types():
    term_set = RealTermSet(4)

    assert term_set.insert(("F0^ F1", 2.0)) == 0
    assert term_set.insert(("F2", -1.0)) == 1
    assert term_set.insert(("F0^ F1", 3.0)) == 0
    assert len(term_set) == 2
    assert term_set["F0^ F1"] == 3.0
    assert term_set.lookup("F2") == (1, -1.0)
    assert term_set.contains("F2")

    terms = term_set.to_terms()
    assert type(terms) is RealTerms
    assert RealTermSet.from_terms(terms) == term_set
    assert type(term_set.into(ComplexTermSet)) is ComplexTermSet

    assert term_set.remove("F0^ F1") == 0
    assert not term_set.contains("F0^ F1")
    with pytest.raises(KeyError):
        term_set.remove("F0^ F1")


def test_real_term_product():
    creation = RealTermSum.from_str("F0^", 2)
    annihilation = RealTermSum.from_str("F0", 2)

    assert str(creation * annihilation) == "((1+0j), F0^ F0)"
    assert str(annihilation * creation) == "((1+0j), ), ((-1+0j), F0^ F0)"
    assert str(annihilation.commutator(creation)) == "((1+0j), ), ((-2+0j), F0^ F0)"
    assert str(annihilation.anticommutator(creation)) == "((1+0j), )"


def test_complex_term_product():
    lhs = ComplexTermSum.from_str("(1j, F0)", 2)
    rhs = ComplexTermSum.from_str("(2, F0^)", 2)

    assert str(lhs * rhs) == "(2j, ), (-2j, F0^ F0)"


def test_symbolic_term_product():
    lhs = SymbolicTerm.from_str("(a, F0)", 2)
    rhs = SymbolicTerm.from_str("(b, F0^)", 2)

    assert str(lhs * rhs) == "(a*b, ), (-a*b, F0^ F0)"


def test_operator_properties():
    number_op = RealTermSum.from_str("F0^ F0", 2)
    hopping = RealTermSum.from_str("F0^ F1, F1^ F0", 2)
    creation = RealTermSum.from_str("F0^", 2)

    assert number_op.is_hermitian()
    assert hopping.is_hermitian()
    assert number_op.conserves_particle_number()
    assert hopping.conserves_particle_number()
    assert not creation.is_hermitian()
    assert not creation.conserves_particle_number()
    assert hopping.max_n_body() == 1
    assert hopping.active_modes() == {0, 1}


def test_to_general():
    terms = RealTermSum.from_str("F2^ F0 F1", 3)

    general_terms = terms.to_general()

    assert str(general_terms) == "((1+0j), F2^ F0 F1)"
    assert general_terms.to_terms()[0].string.get_ops() == [(2, True), (0, False), (1, False)]

    complex_terms = ComplexTermSum.from_str("(1j, F0^ F1)", 2)
    assert str(complex_terms.to_general()) == "(1j, F0^ F1)"
