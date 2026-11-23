"""Normal-ordered products contain only surviving contributions."""

import pytest

import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs


@pytest.mark.parametrize(
    "term_type", [zfon.SignTerm, zfon.RealTerm, zfon.ComplexTerm, zfon.SymbolicTerm]
)
@pytest.mark.parametrize(("source", "sign"), [("F0^ F0", 1), ("F0^ F1^ F0 F1", -1)])
def test_term_squares_have_no_unused_identity_rows(term_type, source, sign):
    term = term_type(2, source)

    product = term * term

    assert len(product) == 1
    assert next(iter(product)) == term_type(2, (source, sign))


@pytest.mark.parametrize(
    ("sum_type", "coefficient"), [(zfon.RealTermSum, 2.0), (zfon.ComplexTermSum, 2j)]
)
@pytest.mark.parametrize(("source", "sign"), [("F0^ F0", 1), ("F0^ F1^ F0 F1", -1)])
def test_sum_squares_have_no_unused_identity_rows(sum_type, coefficient, source, sign):
    terms = sum_type.from_iterable([(source, coefficient)], 2)

    product = terms * terms

    assert len(product) == 1
    expected = sum_type.terms_type.term_type(2, (source, sign * coefficient**2))
    assert next(iter(product)) == expected


@pytest.mark.parametrize(
    "term_type", [zfon.SignTerm, zfon.RealTerm, zfon.ComplexTerm, zfon.SymbolicTerm]
)
def test_contraction_preserves_legitimate_identity(term_type):
    annihilation = term_type(2, "F0")
    creation = term_type(2, "F0^")

    product = annihilation * creation

    assert list(product) == [term_type(2, ""), term_type(2, ("F0^ F0", -1))]


@pytest.mark.parametrize(
    "term_type", [zfon.SignTerm, zfon.RealTerm, zfon.ComplexTerm, zfon.SymbolicTerm]
)
@pytest.mark.parametrize(
    ("lhs", "rhs", "expected", "sign"),
    [
        ("F0^ F0", "F1^ F1", "F0^ F1^ F0 F1", -1),
        ("F1^ F1", "F0^ F0", "F0^ F1^ F0 F1", -1),
        ("F0", "F1^ F1", "F1^ F0 F1", -1),
        ("F0^ F0", "F1^", "F0^ F1^ F0", -1),
        ("F1", "F2^ F0 F2", "F2^ F0 F1 F2", 1),
        ("F0^ F2^ F3", "F1^", "F0^ F1^ F2^ F3", 1),
    ],
)
def test_product_insertion_signs(term_type, lhs, rhs, expected, sign):
    product = term_type(4, lhs) * term_type(4, rhs)

    assert len(product) == 1
    assert next(iter(product)) == term_type(4, (expected, sign))


@pytest.mark.parametrize(
    ("sum_type", "state_type"),
    [(zfon.RealTermSum, zfs.RealTermSum), (zfon.ComplexTermSum, zfs.ComplexTermSum)],
)
@pytest.mark.parametrize("modes", [(0, 1), (1, 0)])
@pytest.mark.parametrize("occupied", [set(), {0}, {1}, {0, 1}])
def test_number_operator_product_expectation(sum_type, state_type, modes, occupied):
    lhs, rhs = [sum_type.from_iterable([(f"F{mode}^ F{mode}", 1.0)], 2) for mode in modes]
    state = state_type.from_iterable([(occupied, 1.0)], 2)

    product = lhs * rhs

    assert len(product) == 1
    assert next(iter(product)) == sum_type.terms_type.term_type(2, ("F0^ F1^ F0 F1", -1))
    assert product.exp_val(state) == pytest.approx(float(occupied == {0, 1}))
