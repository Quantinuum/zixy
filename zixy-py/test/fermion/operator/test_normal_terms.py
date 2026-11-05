import pytest
from sympy import sympify

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
from zixy.container.coeffs import ComplexSign, Sign
from zixy.container.terms import TermSet


@pytest.mark.parametrize(
    ("term_type", "source", "modes", "expected", "terms_type"),
    (
        (zfon.SignTerm, "(1, F0^ F1)", 2, "(+1, F0^ F1)", None),
        (
            zfon.SignTerms,
            "(1, F0^ F1), (-1, F1^ F0), (1, F0^ F1)",
            2,
            "(+1, F0^ F1), (-1, F1^ F0), (+1, F0^ F1)",
            None,
        ),
        (
            zfon.SignTermSet,
            "(1, F0^ F1), (-1, F1^ F0), (1, F0^ F1)",
            2,
            "(+1, F0^ F1), (-1, F1^ F0)",
            zfon.SignTerms,
        ),
        (zfon.RealTerm, " ( 2.5 , F0^ F1 ) ", 2, "(2.5, F0^ F1)", None),
        (
            zfon.RealTerms,
            " (2.0, F0^ F1) , (-0.5, F1^ F0) , (3.0, F0^ F1) ",
            2,
            "(2.0, F0^ F1), (-0.5, F1^ F0), (3.0, F0^ F1)",
            None,
        ),
        (
            zfon.RealTermSet,
            "(2.0, F0^ F1), (-0.5, F1^ F0), (3.0, F0^ F1)",
            2,
            "(3.0, F0^ F1), (-0.5, F1^ F0)",
            zfon.RealTerms,
        ),
        (
            zfon.RealTermSum,
            "(2.0, F0^ F1), (-0.5, F1^ F0), (3.0, F0^ F1)",
            2,
            "(5.0, F0^ F1), (-0.5, F1^ F0)",
            None,
        ),
        (zfon.ComplexTerm, "(1j, F0^ F1)", 2, "(1j, F0^ F1)", None),
        (
            zfon.ComplexTerms,
            "((1j), F0^ F1), ((2), F1^ F0), ((3), F0^ F1)",
            2,
            "(1j, F0^ F1), ((2+0j), F1^ F0), ((3+0j), F0^ F1)",
            None,
        ),
        (
            zfon.ComplexTermSet,
            "((1j), F0^ F1), ((2), F1^ F0), ((3), F0^ F1)",
            2,
            "((3+0j), F0^ F1), ((2+0j), F1^ F0)",
            zfon.ComplexTerms,
        ),
        (
            zfon.ComplexTermSum,
            "((1j), F0^ F1), ((2), F1^ F0), ((3), F0^ F1)",
            2,
            "((3+1j), F0^ F1), ((2+0j), F1^ F0)",
            None,
        ),
        (zfon.RealTerm, "(2, F0^ F1^ F2 F3)", 4, "(2.0, F0^ F1^ F2 F3)", None),
        (zfon.RealTerms, "\n(1, F0^ F1),\n(2, F1^ F0)\n", 2, "(1.0, F0^ F1), (2.0, F1^ F0)", None),
        (zfon.RealTermSum, "(2, F0^ F1^ F2), (-2, F0^ F1^ F2)", 3, "(0.0, F0^ F1^ F2)", None),
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
        zfon.SignTerms,
        zfon.SignTermSet,
        zfon.RealTerms,
        zfon.RealTermSet,
        zfon.RealTermSum,
        zfon.ComplexTerms,
        zfon.ComplexTermSet,
        zfon.ComplexTermSum,
    ),
)
def test_from_str_empty_containers(term_type):
    assert str(term_type.from_str("", 2)) == ""


@pytest.mark.parametrize(
    ("term_type", "source"),
    (
        (zfon.ComplexTerm, "((1j), F0^ F1)"),
        (zfon.ComplexTerms, "((1j), F0^ F1), ((2), F1^ F0)"),
        (zfon.ComplexTermSet, "((1j), F0^ F1), ((2), F1^ F0)"),
        (zfon.ComplexTermSum, "((1j), F0^ F1), ((2), F1^ F0)"),
    ),
)
def test_from_str_round_trip(term_type, source):
    parsed = term_type.from_str(source, 2)
    assert term_type.from_str(str(parsed), 2) == parsed


@pytest.mark.parametrize(
    "term_type", (zfon.SymbolicTerm, zfon.SymbolicTerms, zfon.SymbolicTermSet, zfon.SymbolicTermSum)
)
def test_from_str_symbolic_not_implemented(term_type):
    with pytest.raises(NotImplementedError):
        term_type.from_str("(a, F0^ F1)", 2)


@pytest.mark.parametrize(
    "source", ("1, F0^ F1", "(1 F0^ F1)", "(1, F0^ F1, extra)", "(, F0^ F1)", "(1, )")
)
def test_from_str_errors(source):
    with pytest.raises(ValueError):
        zfon.RealTerm.from_str(source, 2)

    with pytest.raises(ValueError):
        zfon.RealTerm.from_str("(1, F2^ F0)", 2)

    with pytest.raises(ValueError):
        zfon.SignTerm.from_str("(2, F0^ F1)", 2)

    with pytest.raises(ValueError):
        zfon.RealTerm.from_str("(1j, F0^ F1)", 2)


def test_real_term():
    term = zfon.RealTerm(4, ("F0^ F1", 2.0))
    assert type(term.coeff) is float
    assert term.coeff == 2.0
    assert term.string.get_sets() == ([0], [1])

    scaled = term * 3
    assert type(scaled) is zfon.RealTerm
    assert str(scaled) == "(6.0, F0^ F1)"

    complex_scaled = term * 1j
    assert type(complex_scaled) is zfon.ComplexTerm
    assert str(complex_scaled) == "(2j, F0^ F1)"

    symbolic_scaled = term * sympify("a")
    assert type(symbolic_scaled) is zfon.SymbolicTerm
    assert str(symbolic_scaled) == "(2.0*a, F0^ F1)"

    sign_scaled = zfon.SignTerm(4, "F0^ F1") * Sign()
    assert type(sign_scaled) is zfon.SignTerm
    assert str(sign_scaled) == "(+1, F0^ F1)"

    complex_sign_scaled = term * ComplexSign(1)
    assert type(complex_sign_scaled) is zfon.ComplexTerm
    assert str(complex_sign_scaled) == "(2j, F0^ F1)"

    ordered = zfon.RealTerm.from_str("F0^ F1", 4)
    assert str(ordered) == "(1.0, F0^ F1)"
    assert str(ordered.daggered()) == "(1.0, F1^ F0)"


def test_complex_term():
    term = zfon.ComplexTerm(3, ("F0^ F1", 1 + 2j))
    assert type(term.coeff) is complex
    assert term.coeff == 1 + 2j

    scaled = term * 2
    assert type(scaled) is zfon.ComplexTerm
    assert str(scaled) == "((2+4j), F0^ F1)"

    adjoint = zfon.ComplexTerm.from_str("((1j), F0^ F1)", 2).daggered()
    assert str(adjoint) == "(-1j, F1^ F0)"


def test_term_scalar_mul_preserves_viewed_string():
    terms = zfon.RealTerms.from_str("(1.0, F0^ F0), (2.0, F1^ F1)", 2)

    right_scaled = terms[1] * 3.0
    left_scaled = 3.0 * terms[1]

    assert right_scaled.string == terms[1].string
    assert right_scaled.string != terms[0].string
    assert right_scaled.coeff == 6.0
    assert not right_scaled.string.aliases(terms[1].string)
    assert not right_scaled.string.aliases(terms[0].string)
    assert left_scaled.string == terms[1].string
    assert left_scaled.string != terms[0].string
    assert left_scaled.coeff == 6.0
    assert not left_scaled.string.aliases(terms[1].string)
    assert not left_scaled.string.aliases(terms[0].string)


@pytest.mark.parametrize(
    ("constructor", "source"),
    (
        (zfon.RealTerm.from_str, "F1 F0^"),
        (zfon.RealTerms.from_str, "(1.0, F1 F0^)"),
        (zfon.RealTermSet.from_str, "(1.0, F1 F0^)"),
        (zfon.RealTermSum.from_str, "(1.0, F1 F0^)"),
    ),
)
def test_rejects_non_normal_ordered_inputs(constructor, source):
    with pytest.raises(ValueError, match="normal-order"):
        constructor(source, 2)


@pytest.mark.parametrize(
    ("term", "expected"),
    (
        (zfon.SignTerm.from_str("F0^ F1", 2), "(+1, F1^ F0)"),
        (zfon.SignTerm.from_str("F0^ F1^ F2", 3), "(-1, F2^ F0 F1)"),
        (zfon.RealTerm.from_str("(2, F0^ F1)", 2), "(2.0, F1^ F0)"),
        (zfon.RealTerm.from_str("F0^ F1^ F2", 3), "(-1.0, F2^ F0 F1)"),
        (zfon.RealTerm.from_str("(2, F0^ F1^ F2 F3)", 4), "(2.0, F2^ F3^ F0 F1)"),
        (zfon.ComplexTerm.from_str("((1j), F0^ F1)", 2), "(-1j, F1^ F0)"),
        (zfon.ComplexTerm.from_str("((1j), F0^ F1^ F2)", 3), "(1j, F2^ F0 F1)"),
        (zfon.SymbolicTerm(2, ("F0^ F1", sympify("a"))), "(conjugate(a), F1^ F0)"),
        (zfon.SymbolicTerm(3, ("F0^ F1^ F2", sympify("a"))), "(-conjugate(a), F2^ F0 F1)"),
    ),
)
def test_term_dagger(term, expected):
    out = term.dagger()

    assert out is None
    assert str(term) == expected


@pytest.mark.parametrize(
    ("term", "original", "expected"),
    (
        (zfon.SignTerm.from_str("F0^ F1", 2), "(+1, F0^ F1)", "(+1, F1^ F0)"),
        (zfon.SignTerm.from_str("F0^ F1^ F2", 3), "(+1, F0^ F1^ F2)", "(-1, F2^ F0 F1)"),
        (zfon.RealTerm.from_str("(2, F0^ F1)", 2), "(2.0, F0^ F1)", "(2.0, F1^ F0)"),
        (zfon.RealTerm.from_str("F0^ F1^ F2", 3), "(1.0, F0^ F1^ F2)", "(-1.0, F2^ F0 F1)"),
        (
            zfon.RealTerm.from_str("(2, F0^ F1^ F2 F3)", 4),
            "(2.0, F0^ F1^ F2 F3)",
            "(2.0, F2^ F3^ F0 F1)",
        ),
        (zfon.ComplexTerm.from_str("((1j), F0^ F1)", 2), "(1j, F0^ F1)", "(-1j, F1^ F0)"),
        (
            zfon.ComplexTerm.from_str("((1j), F0^ F1^ F2)", 3),
            "(1j, F0^ F1^ F2)",
            "(1j, F2^ F0 F1)",
        ),
        (
            zfon.SymbolicTerm(2, ("F0^ F1", sympify("a"))),
            "(a, F0^ F1)",
            "(conjugate(a), F1^ F0)",
        ),
        (
            zfon.SymbolicTerm(3, ("F0^ F1^ F2", sympify("a"))),
            "(a, F0^ F1^ F2)",
            "(-conjugate(a), F2^ F0 F1)",
        ),
    ),
)
def test_term_daggered(term, original, expected):
    adjoint = term.daggered()

    assert adjoint is not term
    assert str(adjoint) == expected
    assert str(term) == original


def test_sign_term_mul_by_string():
    term = zfon.SignTerm(2, "F0")
    string = zfon.String(2, "F0^")

    product = term * string

    assert type(product) is zfon.RealTermSum
    assert str(product) == "(1.0, ), (-1.0, F0^ F0)"


def test_real_term_mul_by_string():
    term = zfon.RealTerm(2, ("F0", 2.0))
    string = zfon.String(2, "F0^")

    product = term * string

    assert type(product) is zfon.RealTermSum
    assert str(product) == "(2.0, ), (-2.0, F0^ F0)"


def test_complex_term_mul_by_string():
    term = zfon.ComplexTerm(2, ("F0", 1j))
    string = zfon.String(2, "F0^")

    product = term * string

    assert type(product) is zfon.ComplexTermSum
    assert str(product) == "(1j, ), (-1j, F0^ F0)"


def test_symbolic_term_mul_by_string():
    a = sympify("a")
    term = zfon.SymbolicTerm(2, ("F0", a))
    string = zfon.String(2, "F0^")

    product = term * string

    assert type(product) is zfon.SymbolicTermSum
    assert str(product) == "(a, ), (-a, F0^ F0)"


def test_sign_term_mul_by_sign_term():
    lhs = zfon.SignTerm(2, "F0")
    rhs = zfon.SignTerm(2, "F0^")

    product = lhs * rhs

    assert type(product) is zfon.RealTermSum
    assert str(product) == "(1.0, ), (-1.0, F0^ F0)"


@pytest.mark.parametrize("lhs_is_term", [False, True])
@pytest.mark.parametrize("rhs_is_term", [False, True])
@pytest.mark.parametrize("rhs_string", ["F0^", "F0"])
def test_sign_product_supports_addition(lhs_is_term, rhs_is_term, rhs_string):
    lhs_type = zfon.SignTerm if lhs_is_term else zfon.String
    rhs_type = zfon.SignTerm if rhs_is_term else zfon.String
    lhs = lhs_type(2, "F0")
    rhs = rhs_type(2, rhs_string)
    before = str(lhs), str(rhs)

    product = lhs * rhs

    assert type(product) is zfon.RealTermSum
    expected = zfon.RealTermSum(2)
    if rhs_string == "F0^":
        expected = zfon.RealTermSum.from_str("(2, ), (-2, F0^ F0)", 2)
    assert product + product == expected
    assert len(product - product) == 0
    assert (str(lhs), str(rhs)) == before


def test_real_term_mul_by_real_term():
    lhs = zfon.RealTerm(2, ("F0", 2.0))
    rhs = zfon.RealTerm(2, ("F0^", 3.0))

    product = lhs * rhs

    assert type(product) is zfon.RealTermSum
    assert str(product) == "(6.0, ), (-6.0, F0^ F0)"


def test_real_term_mul_by_complex_term():
    lhs = zfon.RealTerm(2, ("F0", 2.0))
    rhs = zfon.ComplexTerm(2, ("F0^", 1j))

    product = lhs * rhs

    assert type(product) is zfon.ComplexTermSum
    assert str(product) == "(2j, ), (-2j, F0^ F0)"


def test_symbolic_term_mul_by_symbolic_term():
    lhs = zfon.SymbolicTerm(2, ("F0", sympify("a")))
    rhs = zfon.SymbolicTerm(2, ("F0^", sympify("b")))

    product = lhs * rhs

    assert type(product) is zfon.SymbolicTermSum
    assert str(product) == "(a*b, ), (-a*b, F0^ F0)"


def test_real_terms():
    terms = zfon.RealTerms(4)
    assert len(terms) == 0
    terms.resize(5)

    assert len(terms) == 5
    assert len(terms[:3].strings) == 3
    assert len(terms[:3].coeffs) == 3
    assert str(terms[0]) == "(1.0, )"

    terms[1].string.set("F0^ F1")
    terms[1].coeff = 2.5
    terms[2] = zfon.String(4, "F2^")
    terms[3] = zfon.RealTerm(4, ("F3", -1.0))

    assert str(terms[1:4]) == "(2.5, F0^ F1), (1.0, F2^), (-1.0, F3)"
    assert type(terms.into(zfon.ComplexTerms)) is zfon.ComplexTerms
    assert tuple(terms.into(zfon.ComplexTerms).coeffs) == (
        1 + 0j,
        2.5 + 0j,
        1 + 0j,
        -1 + 0j,
        1 + 0j,
    )


def test_complex_terms():
    terms = zfon.ComplexTerms(3)
    terms.append_iterable(
        (
            ("F0^ F1", 1 + 2j),
            ("F2", -1j),
        )
    )

    assert type(terms[0]) is zfon.ComplexTerm
    assert str(terms) == "((1+2j), F0^ F1), ((-0-1j), F2)"

    terms[0].coeff = 1.25
    terms[1].coeff = -2.0
    assert terms[0].coeff == 1.25 + 0j
    assert type(terms.into(zfon.RealTerms)) is zfon.RealTerms
    terms[1].coeff = 1j
    with pytest.raises(ValueError):
        terms.into(zfon.RealTerms)


def test_append_iterable():
    terms = zfon.RealTerms(4)
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
    term_sum = zfon.RealTermSum(4)
    term_sum += zfon.RealTerm(4, ("F0^ F1", 2.0))
    term_sum += zfon.RealTerm(4, ("F0^ F1", -0.5))
    term_sum += zfon.RealTerm(4, ("F2^", 1.0))

    assert len(term_sum) == 2
    assert term_sum["F0^ F1"] == 1.5
    assert str(term_sum) == "(1.5, F0^ F1), (1.0, F2^)"
    assert term_sum.l1_norm == 2.5
    assert term_sum.l2_norm == pytest.approx((1.5**2 + 1.0**2) ** 0.5)
    assert str(term_sum.filter_significant(atol=1.1)) == "(1.5, F0^ F1)"

    number = zfon.RealTermSum.from_str("F0^ F0", 2)
    assert str(number) == "(1.0, F0^ F0)"


def test_real_term_add_iterable():
    term_sum = zfon.RealTermSum.from_iterable(
        (
            zfon.RealTerm(4, ("F0^ F1", 1.0)),
            zfon.RealTerm(4, ("F0^ F1", -1.0)),
            zfon.RealTerm(4, ("F2", 0.25)),
        ),
        4,
    )

    assert str(term_sum) == "(0.0, F0^ F1), (0.25, F2)"
    assert str(term_sum.filter_nonzero()) == "(0.25, F2)"
    term_sum["F2"] = 0.0
    assert str(term_sum.filter_nonzero()) == ""


def test_real_term_into_other_types():
    term_set = zfon.RealTermSet(4)

    assert term_set.insert(("F0^ F1", 2.0)) == 0
    assert term_set.insert(("F2", -1.0)) == 1
    assert term_set.insert(("F0^ F1", 3.0)) == 0
    assert len(term_set) == 2
    assert term_set["F0^ F1"] == 3.0
    assert term_set.lookup("F2") == (1, -1.0)
    assert term_set.contains("F2")

    terms = term_set.to_terms()
    assert type(terms) is zfon.RealTerms
    assert zfon.RealTermSet.from_terms(terms) == term_set
    assert type(term_set.into(zfon.ComplexTermSet)) is zfon.ComplexTermSet

    assert term_set.remove("F0^ F1") == 0
    assert not term_set.contains("F0^ F1")
    with pytest.raises(KeyError):
        term_set.remove("F0^ F1")


def test_term_set_check_term():
    term_set = zfon.RealTermSet(3)
    term = zfon.RealTerm.from_str("F0^ F1", 3)

    assert term_set.insert(term) == 0
    term_set._check_term(term)

    with pytest.raises(ValueError, match="different modes"):
        term_set._check_term(zfon.RealTerm.from_str("F0^ F1", 4))

    with pytest.raises(TypeError, match="Expected a RealTerm instance"):
        term_set._check_term(zfog.RealTerm.from_str("F0^ F1", 3))


def test_term_set_check_string():
    term_set = zfon.RealTermSet(3)
    string = zfon.String(3, "F0^ F1")

    term_set._check_string(string)

    with pytest.raises(ValueError, match="different modes"):
        term_set._check_string(zfon.String(4, "F0^ F1"))

    with pytest.raises(TypeError, match="Expected a String instance"):
        term_set._check_string(zfog.String(3, "F0^ F1"))


def test_real_term_product():
    creation = zfon.RealTermSum.from_str("F0^", 2)
    annihilation = zfon.RealTermSum.from_str("F0", 2)

    assert str(creation * annihilation) == "(1.0, F0^ F0)"
    assert str(annihilation * creation) == "(1.0, ), (-1.0, F0^ F0)"
    assert str(annihilation.commutator(creation)) == "(1.0, ), (-2.0, F0^ F0)"
    assert str(annihilation.anticommutator(creation)) == "(1.0, )"


@pytest.mark.parametrize(
    ("product", "expected"),
    (
        (
            zfon.RealTerm.from_str("F1", 3) * zfon.RealTerm.from_str("F0^", 3),
            "(-1.0, F0^ F1)",
        ),
        (
            zfon.RealTerm.from_str("F1 F2", 3) * zfon.RealTerm.from_str("F0^ F1^", 3),
            "(-1.0, F0^ F2), (1.0, F0^ F1^ F1 F2)",
        ),
        (
            zfon.RealTerm.from_str("F0 F1", 3) * zfon.String(3, "F1^ F2^"),
            "(-1.0, F2^ F0), (1.0, F1^ F2^ F0 F1)",
        ),
        (
            zfon.RealTermSum.from_str("F1 F2", 3) * zfon.RealTermSum.from_str("F0^ F1^", 3),
            "(-1.0, F0^ F2), (1.0, F0^ F1^ F1 F2)",
        ),
    ),
)
def test_products_remain_normal_ordered(product, expected):
    assert str(product) == expected
    for term in product:
        ops = term.string.get_ops()
        first_annihilation = next(
            (i for i, (_, is_creation) in enumerate(ops) if not is_creation), len(ops)
        )
        assert all(not is_creation for _, is_creation in ops[first_annihilation:])


def test_complex_term_product():
    lhs = zfon.ComplexTermSum.from_str("((1j), F0)", 2)
    rhs = zfon.ComplexTermSum.from_str("((2), F0^)", 2)

    assert str(lhs * rhs) == "(2j, ), (-2j, F0^ F0)"


@pytest.mark.parametrize("term_sum_type", [zfon.RealTermSum, zfon.ComplexTermSum])
@pytest.mark.parametrize("empty_side", [None, "lhs", "rhs"])
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("__mul__", [("", 1.0), ("F0^ F0", -1.0)]),
        ("commutator", [("", 1.0), ("F0^ F0", -2.0)]),
        ("anticommutator", [("", 1.0)]),
    ],
)
def test_numeric_operations_retain_rust_map(
    term_sum_type, empty_side, operation, expected, monkeypatch
):
    lhs = term_sum_type.from_str("F0", 2)
    rhs = term_sum_type.from_str("F0^", 2)
    scale = 2j if term_sum_type is zfon.ComplexTermSum else 2.0
    lhs *= scale
    expected_sum = term_sum_type.from_iterable([(s, c * scale) for s, c in expected], 2)
    if empty_side == "lhs":
        lhs = term_sum_type(2)
    elif empty_side == "rhs":
        rhs = term_sum_type(2)
    if empty_side is not None:
        expected_sum = term_sum_type(2)

    def fail_collect(*args, **kwargs):
        raise AssertionError("The Rust result should already have its lookup map.")

    monkeypatch.setattr(TermSet, "_collect_terms", fail_collect)
    result = getattr(lhs, operation)(rhs)

    assert type(result) is term_sum_type
    assert result == expected_sum
    assert result.strings.is_owning()
    assert result.coeffs.is_owning()
    assert len(result._string_set._map) == len(result)
    for term in expected_sum:
        assert result.lookup_coeff(term.string) == term.coeff
        index, inserted = result.soft_insert(term)
        assert not inserted
        assert result._impl.coeffs[index] == term.coeff


def test_ladder_product_retains_rust_map():
    strings, coeffs, map_ = zfon.NormalFermionOperatorArray.from_ladder_product(
        zfon.Modes.from_count(2), [(0, False), (0, True)]
    )

    assert len(strings) == len(coeffs) == len(map_) == 2
    for index in range(len(strings)):
        assert strings.mapped_lookup(map_, strings, index) == index


def test_operator_properties():
    number_op = zfon.RealTermSum.from_str("F0^ F0", 2)
    hopping = zfon.RealTermSum.from_str("F0^ F1, F1^ F0", 2)
    creation = zfon.RealTermSum.from_str("F0^", 2)

    assert number_op.is_hermitian()
    assert hopping.is_hermitian()
    assert number_op.conserves_particle_number()
    assert hopping.conserves_particle_number()
    assert not creation.is_hermitian()
    assert not creation.conserves_particle_number()
    assert hopping.max_n_body() == 1
    assert hopping.active_modes() == {0, 1}


@pytest.mark.parametrize(
    ("term_sum_type", "expected_type", "source", "expected", "expected_ops"),
    (
        (
            zfon.RealTermSum,
            zfog.RealTermSum,
            "F2^ F0 F1",
            "(1.0, F2^ F0 F1)",
            [(2, True), (0, False), (1, False)],
        ),
        (
            zfon.ComplexTermSum,
            zfog.ComplexTermSum,
            "((1j), F0^ F1)",
            "(1j, F0^ F1)",
            [(0, True), (1, False)],
        ),
    ),
)
def test_to_general(term_sum_type, expected_type, source, expected, expected_ops):
    terms = term_sum_type.from_str(source, 3)

    general_terms = terms.to_general()

    assert type(general_terms) is expected_type
    assert str(general_terms) == expected
    assert general_terms.to_terms()[0].string.get_ops() == expected_ops
