from sympy import sympify

from zixy.fermion.operator import general


def test_raw_term_from_str_preserves_order():
    term = general.RealTerm.from_str("F1 F0^", 2)

    assert str(term) == "(1.0, F1 F0^)"
    assert term.string.get_ops() == [(1, False), (0, True)]


def test_raw_term_sum_addition_and_deduplication():
    terms = general.RealTermSum.from_str("(2, F1 F0^), (-0.5, F1 F0^), F0^ F1", 2)

    assert len(terms) == 2
    assert str(terms) == "(1.5, F1 F0^), (1.0, F0^ F1)"


def test_raw_multiplication_concatenates_without_normal_ordering():
    lhs = general.RealTermSum.from_str("F0", 2)
    rhs = general.RealTermSum.from_str("F0^", 2)

    product = lhs * rhs

    assert str(product) == "(1.0, F0 F0^)"
    assert product.to_terms()[0].string.get_ops() == [(0, False), (0, True)]


def test_raw_complex_multiplication():
    lhs = general.ComplexTermSum.from_str("(1j, F0)", 2)
    rhs = general.ComplexTermSum.from_str("(2, F1^)", 2)

    assert str(lhs * rhs) == "(2j, F0 F1^)"


def test_normal_ordered_conversion():
    terms = general.RealTermSum.from_str("F0 F0^", 2)

    assert str(terms.normal_ordered()) == "((1+0j), ), ((-1+0j), F0^ F0)"


def test_normal_ordered_conversion_combines_terms():
    terms = general.RealTermSum.from_str("F0 F0^, F0^ F0", 2)

    assert str(terms.normal_ordered()) == "((1+0j), )"


def test_normal_ordered_conversion_of_out_of_order_product():
    terms = general.RealTermSum.from_str("F1 F0^", 2)

    assert str(terms.normal_ordered()) == "((-1+0j), F0^ F1)"


def test_symbolic_raw_terms_substitution():
    a = sympify("a")
    term = general.SymbolicTerm.from_str("(a, F1 F0^)", 2)

    assert str(term.subs({a: 2})) == "(2, F1 F0^)"
