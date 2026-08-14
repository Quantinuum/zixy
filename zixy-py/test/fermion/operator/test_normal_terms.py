import pytest
from sympy import sympify

from zixy.container.coeffs import ComplexSign, Sign
from zixy.fermion.operator.general import (
    ComplexTermSum as GeneralComplexTermSum,
    RealTerm as GeneralRealTerm,
    RealTermSum as GeneralRealTermSum,
    String as GeneralString,
)
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
    SignTermSum,
    String,
    SymbolicTerm,
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

    ordered = RealTerm.from_str("F0^ F1", 4)
    assert str(ordered) == "(1.0, F0^ F1)"
    assert str(ordered.daggered()) == "(1.0, F1^ F0)"


def test_complex_term():
    term = ComplexTerm(3, ("F0^ F1", 1 + 2j))
    assert type(term.coeff) is complex
    assert term.coeff == 1 + 2j

    scaled = term * 2
    assert type(scaled) is ComplexTerm
    assert str(scaled) == "((2+4j), F0^ F1)"

    adjoint = ComplexTerm.from_str("((1j), F0^ F1)", 2).daggered()
    assert str(adjoint) == "(-1j, F1^ F0)"


def test_term_scalar_mul_preserves_viewed_string():
    terms = RealTerms.from_str("(1.0, F0^ F0), (2.0, F1^ F1)", 2)

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
        (RealTerm.from_str, "F1 F0^"),
        (RealTerms.from_str, "(1.0, F1 F0^)"),
    ),
)
def test_rejects_non_normal_ordered_inputs(constructor, source):
    with pytest.raises(ValueError, match="normal-order"):
        constructor(source, 2)


@pytest.mark.parametrize(
    ("term", "expected"),
    (
        (SignTerm.from_str("F0^ F1", 2), "(+1, F1^ F0)"),
        (SignTerm.from_str("F0^ F1^ F2", 3), "(-1, F2^ F0 F1)"),
        (RealTerm.from_str("(2, F0^ F1)", 2), "(2.0, F1^ F0)"),
        (RealTerm.from_str("F0^ F1^ F2", 3), "(-1.0, F2^ F0 F1)"),
        (RealTerm.from_str("(2, F0^ F1^ F2 F3)", 4), "(2.0, F2^ F3^ F0 F1)"),
        (ComplexTerm.from_str("((1j), F0^ F1)", 2), "(-1j, F1^ F0)"),
        (ComplexTerm.from_str("((1j), F0^ F1^ F2)", 3), "(1j, F2^ F0 F1)"),
        (SymbolicTerm(2, ("F0^ F1", sympify("a"))), "(conjugate(a), F1^ F0)"),
        (SymbolicTerm(3, ("F0^ F1^ F2", sympify("a"))), "(-conjugate(a), F2^ F0 F1)"),
    ),
)
def test_term_dagger(term, expected):
    out = term.dagger()

    assert out is None
    assert str(term) == expected


@pytest.mark.parametrize(
    ("term", "original", "expected"),
    (
        (SignTerm.from_str("F0^ F1", 2), "(+1, F0^ F1)", "(+1, F1^ F0)"),
        (SignTerm.from_str("F0^ F1^ F2", 3), "(+1, F0^ F1^ F2)", "(-1, F2^ F0 F1)"),
        (RealTerm.from_str("(2, F0^ F1)", 2), "(2.0, F0^ F1)", "(2.0, F1^ F0)"),
        (RealTerm.from_str("F0^ F1^ F2", 3), "(1.0, F0^ F1^ F2)", "(-1.0, F2^ F0 F1)"),
        (
            RealTerm.from_str("(2, F0^ F1^ F2 F3)", 4),
            "(2.0, F0^ F1^ F2 F3)",
            "(2.0, F2^ F3^ F0 F1)",
        ),
        (ComplexTerm.from_str("((1j), F0^ F1)", 2), "(1j, F0^ F1)", "(-1j, F1^ F0)"),
        (
            ComplexTerm.from_str("((1j), F0^ F1^ F2)", 3),
            "(1j, F0^ F1^ F2)",
            "(1j, F2^ F0 F1)",
        ),
        (
            SymbolicTerm(2, ("F0^ F1", sympify("a"))),
            "(a, F0^ F1)",
            "(conjugate(a), F1^ F0)",
        ),
        (
            SymbolicTerm(3, ("F0^ F1^ F2", sympify("a"))),
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
    term = SignTerm(2, "F0")
    string = String(2, "F0^")

    product = term * string

    assert type(product) is SignTermSum
    assert str(product) == "(+1, ), (-1, F0^ F0)"


def test_real_term_mul_by_string():
    term = RealTerm(2, ("F0", 2.0))
    string = String(2, "F0^")

    product = term * string

    assert type(product) is RealTermSum
    assert str(product) == "(2.0, ), (-2.0, F0^ F0)"


def test_complex_term_mul_by_string():
    term = ComplexTerm(2, ("F0", 1j))
    string = String(2, "F0^")

    product = term * string

    assert type(product) is ComplexTermSum
    assert str(product) == "(1j, ), ((-0-1j), F0^ F0)"


def test_symbolic_term_mul_by_string():
    a = sympify("a")
    term = SymbolicTerm(2, ("F0", a))
    string = String(2, "F0^")

    product = term * string

    assert type(product) is SymbolicTermSum
    assert str(product) == "(a, ), (-a, F0^ F0)"


def test_sign_term_mul_by_sign_term():
    lhs = SignTerm(2, "F0")
    rhs = SignTerm(2, "F0^")

    product = lhs * rhs

    assert type(product) is SignTermSum
    assert str(product) == "(+1, ), (-1, F0^ F0)"


def test_real_term_mul_by_real_term():
    lhs = RealTerm(2, ("F0", 2.0))
    rhs = RealTerm(2, ("F0^", 3.0))

    product = lhs * rhs

    assert type(product) is RealTermSum
    assert str(product) == "(6.0, ), (-6.0, F0^ F0)"


def test_real_term_mul_by_complex_term():
    lhs = RealTerm(2, ("F0", 2.0))
    rhs = ComplexTerm(2, ("F0^", 1j))

    product = lhs * rhs

    assert type(product) is ComplexTermSum
    assert str(product) == "(2j, ), ((-0-2j), F0^ F0)"


def test_symbolic_term_mul_by_symbolic_term():
    lhs = SymbolicTerm(2, ("F0", sympify("a")))
    rhs = SymbolicTerm(2, ("F0^", sympify("b")))

    product = lhs * rhs

    assert type(product) is SymbolicTermSum
    assert str(product) == "(a*b, ), (-a*b, F0^ F0)"


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

    number = RealTermSum.from_str("F0^ F0", 2)
    assert str(number) == "(1.0, F0^ F0)"


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


def test_term_set_check_term():
    term_set = RealTermSet(3)
    term = RealTerm.from_str("F0^ F1", 3)

    assert term_set.insert(term) == 0
    term_set._check_term(term)

    with pytest.raises(ValueError, match="different modes"):
        term_set._check_term(RealTerm.from_str("F0^ F1", 4))

    with pytest.raises(TypeError, match="Expected a RealTerm instance"):
        term_set._check_term(GeneralRealTerm.from_str("F0^ F1", 3))


def test_term_set_check_cmpnt():
    term_set = RealTermSet(3)
    string = String(3, "F0^ F1")

    term_set._check_cmpnt(string)

    with pytest.raises(ValueError, match="different modes"):
        term_set._check_cmpnt(String(4, "F0^ F1"))

    with pytest.raises(TypeError, match="Expected a String instance"):
        term_set._check_cmpnt(GeneralString(3, "F0^ F1"))


def test_real_term_product():
    creation = RealTermSum.from_str("F0^", 2)
    annihilation = RealTermSum.from_str("F0", 2)

    assert str(creation * annihilation) == "(1.0, F0^ F0)"
    assert str(annihilation * creation) == "(1.0, ), (-1.0, F0^ F0)"
    assert str(annihilation.commutator(creation)) == "(1.0, ), (-2.0, F0^ F0)"
    assert str(annihilation.anticommutator(creation)) == "(1.0, )"


@pytest.mark.parametrize(
    ("product", "expected"),
    (
        (
            RealTerm.from_str("F1", 3) * RealTerm.from_str("F0^", 3),
            "(-1.0, F0^ F1)",
        ),
        (
            RealTerm.from_str("F1 F2", 3) * RealTerm.from_str("F0^ F1^", 3),
            "(-1.0, F0^ F2), (1.0, F0^ F1^ F1 F2)",
        ),
        (
            RealTerm.from_str("F0 F1", 3) * String(3, "F1^ F2^"),
            "(-1.0, F2^ F0), (1.0, F1^ F2^ F0 F1)",
        ),
        (
            RealTermSum.from_str("F1 F2", 3) * RealTermSum.from_str("F0^ F1^", 3),
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
    lhs = ComplexTermSum.from_str("((1j), F0)", 2)
    rhs = ComplexTermSum.from_str("((2), F0^)", 2)

    assert str(lhs * rhs) == "(2j, ), (-2j, F0^ F0)"


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


@pytest.mark.parametrize(
    ("term_sum_type", "expected_type", "source", "expected", "expected_ops"),
    (
        (
            RealTermSum,
            GeneralRealTermSum,
            "F2^ F0 F1",
            "(1.0, F2^ F0 F1)",
            [(2, True), (0, False), (1, False)],
        ),
        (
            ComplexTermSum,
            GeneralComplexTermSum,
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
