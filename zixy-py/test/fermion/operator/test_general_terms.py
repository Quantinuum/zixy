import pytest
from sympy import sympify

from zixy.fermion.operator.general import (
    ComplexTerm,
    ComplexTerms,
    ComplexTermSet,
    ComplexTermSum,
    RealTerm,
    RealTerms,
    RealTermSet,
    RealTermSum,
    String,
    SymbolicTerm,
    SymbolicTerms,
    SymbolicTermSum,
)


def test_real_term():
    with pytest.raises(ValueError, match="mode index out of bounds"):
        RealTerm(1, ("F2^", 2.0))

    term = RealTerm(4, ("F1 F0^", 2.0))
    assert type(term.coeff) is float
    assert term.coeff == 2.0
    assert term.string.get_ops() == [(1, False), (0, True)]

    scaled = term * 3
    assert type(scaled) is RealTerm
    assert str(scaled) == "(6.0, F1 F0^)"

    complex_scaled = term * 1j
    assert type(complex_scaled) is ComplexTerm
    assert str(complex_scaled) == "(2j, F1 F0^)"

    symbolic_scaled = term * sympify("a")
    assert type(symbolic_scaled) is SymbolicTerm
    assert str(symbolic_scaled) == "(2.0*a, F1 F0^)"

    parsed = RealTerm.from_str("F1 F0^", 2)
    assert str(parsed) == "(1.0, F1 F0^)"
    assert parsed.string.get_ops() == [(1, False), (0, True)]


def test_complex_term():
    term = ComplexTerm(3, ("F1 F0^", 1 + 2j))
    assert type(term.coeff) is complex
    assert term.coeff == 1 + 2j

    scaled = term * 2
    assert type(scaled) is ComplexTerm
    assert str(scaled) == "((2+4j), F1 F0^)"


def test_real_terms():
    terms = RealTerms(4, max_len=2)
    assert len(terms) == 0
    terms.resize(4)

    assert len(terms) == 4
    assert len(terms[:2].strings) == 2
    assert terms[0].string.get_ops() == []

    terms[1].string.set("F1 F0^")
    terms[1].coeff = 2.5
    terms[2] = String(4, "F2^")
    terms[3] = RealTerm(4, ("F3", -1.0))

    assert str(terms[1:]) == "(2.5, F1 F0^), (1.0, F2^), (-1.0, F3)"
    assert type(terms.into(ComplexTerms)) is ComplexTerms
    assert tuple(terms.into(ComplexTerms).coeffs) == (
        1 + 0j,
        2.5 + 0j,
        1 + 0j,
        -1 + 0j,
    )


def test_complex_terms():
    terms = ComplexTerms(3, max_len=2)
    terms.append_iterable(
        (
            ("F1 F0^", 1 + 2j),
            ("F2", -1j),
        )
    )

    assert type(terms[0]) is ComplexTerm
    assert str(terms) == "((1+2j), F1 F0^), ((-0-1j), F2)"

    terms[0].coeff = 1.25
    terms[1].coeff = -2.0
    assert terms[0].coeff == 1.25 + 0j
    assert type(terms.into(RealTerms)) is RealTerms
    terms[1].coeff = 1j
    with pytest.raises(ValueError):
        terms.into(RealTerms)


def test_symbolic_terms():
    a = sympify("a")
    terms = SymbolicTerms(3, max_len=2)
    terms.resize(3)
    terms[1] = ("F1 F0^", a)
    terms[2] = SymbolicTerm(3, ("F2", 2 * a))

    assert str(terms) == "(1, ), (a, F1 F0^), (2*a, F2)"
    assert str(terms[1].subs({a: 3})) == "(3, F1 F0^)"
    assert str(terms.subs({a: 3})) == "(1, ), (3, F1 F0^), (6, F2)"

    term_sum = SymbolicTermSum.from_str("(a, F1 F0^)", 2)
    assert str(term_sum.subs({a: 2})) == "(2, F1 F0^)"


def test_append_iterable():
    terms = RealTerms(4, max_len=2)
    terms.append_iterable(
        (
            ("F1 F0^", 2.0),
            ("F2", -1.0),
            ("F3^", 0.5),
        )
    )

    assert len(terms[:]) == 3
    assert str(terms[::-1]) == "(0.5, F3^), (-1.0, F2), (2.0, F1 F0^)"
    assert terms.clone() == terms


def test_real_term_sum():
    term_sum = RealTermSum.from_str("(2, F1 F0^), (-0.5, F1 F0^), F0^ F1", 2)

    assert len(term_sum) == 2
    assert str(term_sum) == "(1.5, F1 F0^), (1.0, F0^ F1)"
    assert term_sum.l1_norm == 2.5
    assert term_sum.l2_norm == pytest.approx((1.5**2 + 1.0**2) ** 0.5)
    assert str(term_sum.filter_significant(atol=1.1)) == "(1.5, F1 F0^)"


def test_real_term_add_iterable():
    term_sum = RealTermSum.from_iterable(
        (
            RealTerm(4, ("F1 F0^", 1.0)),
            RealTerm(4, ("F1 F0^", -1.0)),
            RealTerm(4, ("F2", 0.25)),
        ),
        4,
        max_len=2,
    )

    assert str(term_sum) == "(0.0, F1 F0^), (0.25, F2)"
    assert str(term_sum.filter_nonzero()) == "(0.25, F2)"
    term_sum["F2"] = 0.0
    assert str(term_sum.filter_nonzero()) == ""


def test_real_term_into_other_types():
    term_set = RealTermSet(4, max_len=2)

    assert term_set.insert(("F1 F0^", 2.0)) == 0
    assert term_set.insert(("F2", -1.0)) == 1
    assert term_set.insert(("F1 F0^", 3.0)) == 0
    assert len(term_set) == 2
    assert term_set["F1 F0^"] == 3.0
    assert term_set.lookup("F2") == (1, -1.0)
    assert term_set.contains("F2")

    terms = term_set.to_terms()
    assert type(terms) is RealTerms
    assert RealTermSet.from_terms(terms) == term_set
    assert type(term_set.into(ComplexTermSet)) is ComplexTermSet

    assert term_set.remove("F1 F0^") == 0
    assert not term_set.contains("F1 F0^")
    with pytest.raises(KeyError):
        term_set.remove("F1 F0^")


def test_real_term_product():
    lhs = RealTermSum.from_str("F0", 2)
    rhs = RealTermSum.from_str("F0^", 2)

    product = lhs * rhs

    assert str(product) == "(1.0, F0 F0^)"
    assert product.to_terms()[0].string.get_ops() == [(0, False), (0, True)]


def test_complex_term_product():
    lhs = ComplexTermSum.from_str("(1j, F0)", 2)
    rhs = ComplexTermSum.from_str("(2, F1^)", 2)

    assert str(lhs * rhs) == "(2j, F0 F1^)"


def test_symbolic_term_product():
    lhs = SymbolicTerm.from_str("(a, F0)", 2)
    rhs = SymbolicTerm.from_str("(b, F0^)", 2)

    assert str(lhs * rhs) == "(a*b, F0 F0^)"


def test_normal_ordered():
    assert str(RealTermSum.from_str("F0 F0^", 2).normal_ordered()) == (
        "((1+0j), ), ((-1+0j), F0^ F0)"
    )
    assert str(RealTermSum.from_str("F0 F0^, F0^ F0", 2).normal_ordered()) == ("((1+0j), )")
    assert str(RealTermSum.from_str("F1 F0^", 2).normal_ordered()) == ("((-1+0j), F0^ F1)")
