from sympy import sympify

from zixy.fermion.operator import normal


def test_term_from_out_of_order_product_absorbs_sign():
    term = normal.RealTerm.from_str("F1 F0^", 4)

    assert str(term) == "(-1.0, F0^ F1)"


def test_term_sum_from_contraction_product_expands():
    terms = normal.RealTermSum.from_str("F0 F0^", 2)

    assert str(terms) == "(1.0, ), (-1.0, F0^ F0)"


def test_real_term_sum_addition_and_deduplication():
    terms = normal.RealTermSum(4)
    terms += normal.RealTerm(4, ("F0^ F1", 2.0))
    terms += normal.RealTerm(4, ("F0^ F1", -0.5))
    terms += normal.RealTerm(4, ("F2^", 1.0))

    assert len(terms) == 2
    assert terms["F0^ F1"] == 1.5
    assert str(terms) == "(1.5, F0^ F1), (1.0, F2^)"


def test_term_sum_multiplication_uses_fermion_normal_ordering():
    creation = normal.RealTermSum.from_str("F0^", 2)
    annihilation = normal.RealTermSum.from_str("F0", 2)

    assert str(creation * annihilation) == "((1+0j), F0^ F0)"
    assert str(annihilation * creation) == "((1+0j), ), ((-1+0j), F0^ F0)"


def test_complex_term_sum_multiplication():
    lhs = normal.ComplexTermSum.from_str("(1j, F0)", 2)
    rhs = normal.ComplexTermSum.from_str("(2, F0^)", 2)

    assert str(lhs * rhs) == "(2j, ), (-2j, F0^ F0)"


def test_commutator_and_anticommutator():
    annihilation = normal.RealTermSum.from_str("F0", 2)
    creation = normal.RealTermSum.from_str("F0^", 2)

    assert str(annihilation.commutator(creation)) == "((1+0j), ), ((-2+0j), F0^ F0)"
    assert str(annihilation.anticommutator(creation)) == "((1+0j), )"


def test_adjoint_and_daggered():
    term = normal.RealTerm.from_str("F0^ F1", 3)
    terms = normal.RealTermSum.from_str("(2, F0^ F1), (3, F2^)", 3)

    assert str(term.daggered()) == "(1.0, F1^ F0)"
    assert str(terms.daggered()) == "(2.0, F1^ F0), (3.0, F2)"


def test_operator_properties():
    number_op = normal.RealTermSum.from_str("F0^ F0", 2)
    hopping = normal.RealTermSum.from_str("F0^ F1, F1^ F0", 2)
    creation = normal.RealTermSum.from_str("F0^", 2)

    assert number_op.is_hermitian()
    assert hopping.is_hermitian()
    assert number_op.conserves_particle_number()
    assert hopping.conserves_particle_number()
    assert not creation.is_hermitian()
    assert not creation.conserves_particle_number()
    assert hopping.max_n_body() == 1
    assert hopping.active_modes() == {0, 1}


def test_normal_to_general_conversion_preserves_normal_order():
    terms = normal.RealTermSum.from_str("F2^ F0 F1", 3)

    general_terms = terms.to_general()

    assert str(general_terms) == "((1+0j), F2^ F0 F1)"
    assert general_terms.to_terms()[0].string.get_ops() == [(2, True), (0, False), (1, False)]


def test_symbolic_terms_substitution():
    a = sympify("a")
    term = normal.SymbolicTerm.from_str("(a, F0^ F1)", 2)
    terms = normal.SymbolicTermSum.from_str("(a, F0^ F1)", 2)

    assert str(term.subs({a: 2})) == "(2, F0^ F1)"
    assert str(terms) == "(a, F0^ F1)"
