import pytest
from sympy import sympify

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon


@pytest.mark.parametrize(
    ("term_type", "source", "modes", "expected", "terms_type"),
    (
        (zfog.RealTerm, " ( 2.5 , F0^ F1 ) ", 2, "(2.5, F0^ F1)", None),
        (
            zfog.RealTerms,
            " (2.0, F0^ F1) , (-0.5, F1 F0^) , (3.0, F0^ F1) ",
            2,
            "(2.0, F0^ F1), (-0.5, F1 F0^), (3.0, F0^ F1)",
            None,
        ),
        (
            zfog.RealTermSet,
            "(2.0, F0^ F1), (-0.5, F1 F0^), (3.0, F0^ F1)",
            2,
            "(3.0, F0^ F1), (-0.5, F1 F0^)",
            zfog.RealTerms,
        ),
        (
            zfog.RealTermSum,
            "(2.0, F0^ F1), (-0.5, F1 F0^), (3.0, F0^ F1)",
            2,
            "(5.0, F0^ F1), (-0.5, F1 F0^)",
            None,
        ),
        (zfog.ComplexTerm, "(1j, F0^ F1)", 2, "(1j, F0^ F1)", None),
        (
            zfog.ComplexTerms,
            "((1j), F0^ F1), ((2), F1 F0^), ((3), F0^ F1)",
            2,
            "(1j, F0^ F1), ((2+0j), F1 F0^), ((3+0j), F0^ F1)",
            None,
        ),
        (
            zfog.ComplexTermSet,
            "((1j), F0^ F1), ((2), F1 F0^), ((3), F0^ F1)",
            2,
            "((3+0j), F0^ F1), ((2+0j), F1 F0^)",
            zfog.ComplexTerms,
        ),
        (
            zfog.ComplexTermSum,
            "((1j), F0^ F1), ((2), F1 F0^), ((3), F0^ F1)",
            2,
            "((3+1j), F0^ F1), ((2+0j), F1 F0^)",
            None,
        ),
        (zfog.RealTerm, "(2, F0^ F1 F2^ F0)", 3, "(2.0, F0^ F1 F2^ F0)", None),
        (zfog.RealTermSum, "(2, F0^ F1^ F2), (-2, F0^ F1^ F2)", 3, "(0.0, F0^ F1^ F2)", None),
    ),
)
def test_from_str(term_type, source, modes, expected, terms_type):
    parsed = term_type.from_str(source, modes)
    assert str(parsed) == expected

    if terms_type is not None:
        assert parsed == term_type.from_terms(terms_type.from_str(source, modes))


@pytest.mark.parametrize(
    "term_type",
    (
        zfog.RealTerms,
        zfog.RealTermSet,
        zfog.RealTermSum,
        zfog.ComplexTerms,
        zfog.ComplexTermSet,
        zfog.ComplexTermSum,
    ),
)
def test_from_str_empty_containers(term_type):
    assert str(term_type.from_str("", 2)) == ""


@pytest.mark.parametrize(
    ("term_type", "source"),
    (
        (zfog.ComplexTerm, "((1j), F0^ F1)"),
        (zfog.ComplexTerms, "((1j), F0^ F1), ((2), F1 F0^)"),
        (zfog.ComplexTermSet, "((1j), F0^ F1), ((2), F1 F0^)"),
        (zfog.ComplexTermSum, "((1j), F0^ F1), ((2), F1 F0^)"),
    ),
)
def test_from_str_round_trip(term_type, source):
    parsed = term_type.from_str(source, 2)
    assert term_type.from_str(str(parsed), 2) == parsed


@pytest.mark.parametrize(
    "term_type", (zfog.SymbolicTerm, zfog.SymbolicTerms, zfog.SymbolicTermSet, zfog.SymbolicTermSum)
)
def test_from_str_symbolic_not_implemented(term_type):
    with pytest.raises(NotImplementedError):
        term_type.from_str("(a, F0^ F1)", 2)


@pytest.mark.parametrize(
    "source", ("1, F0^ F1", "(1 F0^ F1)", "(1, F0^ F1, extra)", "(, F0^ F1)", "(1, )")
)
def test_from_str_errors(source):
    with pytest.raises(ValueError):
        zfog.RealTerm.from_str(source, 2)

    with pytest.raises(ValueError):
        zfog.RealTerm.from_str("(1, F2^ F0)", 2)

    with pytest.raises(ValueError):
        zfog.RealTerm.from_str("(1j, F0^ F1)", 2)


def test_real_term():
    with pytest.raises(ValueError, match="mode index out of bounds"):
        zfog.RealTerm(1, ("F2^", 2.0))

    term = zfog.RealTerm(4, ("F1 F0^", 2.0))
    assert type(term.coeff) is float
    assert term.coeff == 2.0
    assert term.string.get_ops() == [(1, False), (0, True)]

    scaled = term * 3
    assert type(scaled) is zfog.RealTerm
    assert str(scaled) == "(6.0, F1 F0^)"

    complex_scaled = term * 1j
    assert type(complex_scaled) is zfog.ComplexTerm
    assert str(complex_scaled) == "(2j, F1 F0^)"

    symbolic_scaled = term * sympify("a")
    assert type(symbolic_scaled) is zfog.SymbolicTerm
    assert str(symbolic_scaled) == "(2.0*a, F1 F0^)"

    parsed = zfog.RealTerm.from_str("F1 F0^", 2)
    assert str(parsed) == "(1.0, F1 F0^)"
    assert parsed.string.get_ops() == [(1, False), (0, True)]


def test_complex_term():
    term = zfog.ComplexTerm(3, ("F1 F0^", 1 + 2j))
    assert type(term.coeff) is complex
    assert term.coeff == 1 + 2j

    scaled = term * 2
    assert type(scaled) is zfog.ComplexTerm
    assert str(scaled) == "((2+4j), F1 F0^)"


@pytest.mark.parametrize(
    ("term", "expected"),
    (
        (zfog.RealTerm.from_str("(2, F0^ F1)", 2), "(2.0, F1^ F0)"),
        (zfog.ComplexTerm.from_str("((1j), F0^ F1)", 2), "(-1j, F1^ F0)"),
        (zfog.SymbolicTerm(2, ("F0^ F1", sympify("a"))), "(conjugate(a), F1^ F0)"),
    ),
)
def test_term_dagger(term, expected):
    out = term.dagger()

    assert out is None
    assert str(term) == expected


@pytest.mark.parametrize(
    ("term", "original", "expected"),
    (
        (zfog.RealTerm.from_str("(2, F0^ F1)", 2), "(2.0, F0^ F1)", "(2.0, F1^ F0)"),
        (zfog.ComplexTerm.from_str("((1j), F0^ F1)", 2), "(1j, F0^ F1)", "(-1j, F1^ F0)"),
        (
            zfog.SymbolicTerm(2, ("F0^ F1", sympify("a"))),
            "(a, F0^ F1)",
            "(conjugate(a), F1^ F0)",
        ),
    ),
)
def test_term_daggered(term, original, expected):
    adjoint = term.daggered()

    assert adjoint is not term
    assert str(adjoint) == expected
    assert str(term) == original


def test_real_terms():
    terms = zfog.RealTerms(4, max_string_len=2)
    assert len(terms) == 0
    terms.resize(4)

    assert len(terms) == 4
    assert len(terms[:2].strings) == 2
    assert terms[0].string.get_ops() == []

    terms[1].string.set("F1 F0^")
    terms[1].coeff = 2.5
    terms[2] = zfog.String(4, "F2^")
    terms[3] = zfog.RealTerm(4, ("F3", -1.0))

    assert str(terms[1:]) == "(2.5, F1 F0^), (1.0, F2^), (-1.0, F3)"
    assert type(terms.into(zfog.ComplexTerms)) is zfog.ComplexTerms
    assert tuple(terms.into(zfog.ComplexTerms).coeffs) == (
        1 + 0j,
        2.5 + 0j,
        1 + 0j,
        -1 + 0j,
    )


def test_complex_terms():
    terms = zfog.ComplexTerms(3, max_string_len=2)
    terms.append_iterable(
        (
            ("F1 F0^", 1 + 2j),
            ("F2", -1j),
        )
    )

    assert type(terms[0]) is zfog.ComplexTerm
    assert str(terms) == "((1+2j), F1 F0^), ((-0-1j), F2)"

    terms[0].coeff = 1.25
    terms[1].coeff = -2.0
    assert terms[0].coeff == 1.25 + 0j
    assert type(terms.into(zfog.RealTerms)) is zfog.RealTerms
    terms[1].coeff = 1j
    with pytest.raises(ValueError):
        terms.into(zfog.RealTerms)


def test_append_iterable():
    terms = zfog.RealTerms(4, max_string_len=2)
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
    term_sum = zfog.RealTermSum.from_str("(2, F1 F0^), (-0.5, F1 F0^), (1, F0^ F1)", 2)

    assert len(term_sum) == 2
    assert str(term_sum) == "(1.5, F1 F0^), (1.0, F0^ F1)"
    assert term_sum.l1_norm == 2.5
    assert term_sum.l2_norm == pytest.approx((1.5**2 + 1.0**2) ** 0.5)
    assert str(term_sum.filter_significant(atol=1.1)) == "(1.5, F1 F0^)"


def test_real_term_add_iterable():
    term_sum = zfog.RealTermSum.from_iterable(
        (
            zfog.RealTerm(4, ("F1 F0^", 1.0)),
            zfog.RealTerm(4, ("F1 F0^", -1.0)),
            zfog.RealTerm(4, ("F2", 0.25)),
        ),
        4,
        max_string_len=2,
    )

    assert str(term_sum) == "(0.0, F1 F0^), (0.25, F2)"
    assert str(term_sum.filter_nonzero()) == "(0.25, F2)"
    term_sum["F2"] = 0.0
    assert str(term_sum.filter_nonzero()) == ""


def test_term_sum_rejects_term_over_different_modes():
    term_sum = zfog.RealTermSum(max_string_len=4)
    term = zfog.RealTerm.from_string_coeff(zfog.String(4, "F0^ F0"), 1.0)

    with pytest.raises(ValueError, match="different modes"):
        term_sum += term


def test_term_set_check_term():
    term_set = zfog.RealTermSet(3, max_string_len=2)
    term = zfog.RealTerm.from_str("F0^ F1", 3)

    assert term_set.insert(term) == 0
    term_set._check_term(term)

    with pytest.raises(ValueError, match="different modes"):
        term_set._check_term(zfog.RealTerm.from_str("F0^ F1", 4))

    with pytest.raises(TypeError, match="Expected a RealTerm instance"):
        term_set._check_term(zfon.RealTerm.from_str("F0^ F1", 3))


def test_term_set_check_string():
    term_set = zfog.RealTermSet(3, max_string_len=2)
    string = zfog.String(3, "F0^ F1")

    term_set._check_string(string)

    with pytest.raises(ValueError, match="different modes"):
        term_set._check_string(zfog.String(4, "F0^ F1"))

    with pytest.raises(TypeError, match="Expected a String instance"):
        term_set._check_string(zfon.String(3, "F0^ F1"))


def test_real_term_into_other_types():
    term_set = zfog.RealTermSet(4, max_string_len=2)

    assert term_set.insert(("F1 F0^", 2.0)) == 0
    assert term_set.insert(("F2", -1.0)) == 1
    assert term_set.insert(("F1 F0^", 3.0)) == 0
    assert len(term_set) == 2
    assert term_set["F1 F0^"] == 3.0
    assert term_set.lookup("F2") == (1, -1.0)
    assert term_set.contains("F2")

    terms = term_set.to_terms()
    assert type(terms) is zfog.RealTerms
    assert zfog.RealTermSet.from_terms(terms) == term_set
    assert type(term_set.into(zfog.ComplexTermSet)) is zfog.ComplexTermSet

    assert term_set.remove("F1 F0^") == 0
    assert not term_set.contains("F1 F0^")
    with pytest.raises(KeyError):
        term_set.remove("F1 F0^")


def test_real_term_product():
    lhs = zfog.RealTermSum.from_str("F0", 2)
    rhs = zfog.RealTermSum.from_str("F0^", 2)

    product = lhs * rhs

    assert str(product) == "(1.0, F0 F0^)"
    assert product.to_terms()[0].string.get_ops() == [(0, False), (0, True)]


def test_complex_term_product():
    lhs = zfog.ComplexTermSum.from_str("((1j), F0)", 2)
    rhs = zfog.ComplexTermSum.from_str("((2), F1^)", 2)

    assert str(lhs * rhs) == "(2j, F0 F1^)"


def test_complex_term_sum_dagger():
    term_sum = zfog.ComplexTermSum.from_str("((1j), F0^ F1), ((2), F2)", 3)

    adjoint = term_sum.daggered()

    assert adjoint is not term_sum
    assert str(term_sum) == "(1j, F0^ F1), ((2+0j), F2)"
    assert str(adjoint) == "(-1j, F1^ F0), ((2+0j), F2^)"

    term_sum.dagger()
    assert term_sum == adjoint


@pytest.mark.parametrize(
    ("term_sum_type", "expected_type", "source", "modes", "expected"),
    (
        (
            zfog.RealTermSum,
            zfon.RealTermSum,
            "F0 F0^",
            2,
            "(1.0, ), (-1.0, F0^ F0)",
        ),
        (
            zfog.ComplexTermSum,
            zfon.ComplexTermSum,
            "F0 F0^",
            2,
            "((1+0j), ), ((-1+0j), F0^ F0)",
        ),
        (
            zfog.RealTermSum,
            zfon.RealTermSum,
            "F0 F0^, F0^ F0",
            2,
            "(1.0, )",
        ),
        (
            zfog.ComplexTermSum,
            zfon.ComplexTermSum,
            "F0 F0^, F0^ F0",
            2,
            "((1+0j), )",
        ),
        (
            zfog.RealTermSum,
            zfon.RealTermSum,
            "F1 F0^",
            2,
            "(-1.0, F0^ F1)",
        ),
        (
            zfog.ComplexTermSum,
            zfon.ComplexTermSum,
            "F1 F0^",
            2,
            "((-1+0j), F0^ F1)",
        ),
        (
            zfog.RealTermSum,
            zfon.RealTermSum,
            "F0 F1 F1^",
            2,
            "(1.0, F0), (1.0, F1^ F0 F1)",
        ),
        (
            zfog.ComplexTermSum,
            zfon.ComplexTermSum,
            "((1j), F0 F1 F1^)",
            2,
            "(1j, F0), (1j, F1^ F0 F1)",
        ),
        (
            zfog.RealTermSum,
            zfon.RealTermSum,
            "F2 F0^ F1^",
            3,
            # F2 crosses two distinct creators, giving (-1)**2 = +1.
            "(1.0, F0^ F1^ F2)",
        ),
        (
            zfog.ComplexTermSum,
            zfon.ComplexTermSum,
            "F2 F0^ F1^",
            3,
            "((1+0j), F0^ F1^ F2)",
        ),
        (
            zfog.RealTermSum,
            zfon.RealTermSum,
            "F0 F0^ F1 F1^",
            3,
            "(1.0, ), (-1.0, F1^ F1), (-1.0, F0^ F0), (-1.0, F0^ F1^ F0 F1)",
        ),
        (
            zfog.ComplexTermSum,
            zfon.ComplexTermSum,
            "((2j), F0 F0^ F1 F1^), ((1-1j), F0 F0^)",
            3,
            "((1+1j), ), (-2j, F1^ F1), ((-1-1j), F0^ F0), (-2j, F0^ F1^ F0 F1)",
        ),
    ),
)
def test_to_normal_ordered(term_sum_type, expected_type, source, modes, expected):
    terms = term_sum_type.from_str(source, modes)

    normal_ordered = terms.to_normal_ordered()

    assert type(normal_ordered) is expected_type
    assert str(normal_ordered) == expected
