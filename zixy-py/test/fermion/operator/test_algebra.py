"""Fermionic identities checked against independent Jordan--Wigner matrices.

The reference matrices use tensor products of two-dimensional operators, without
Zixy's multiplication, normal ordering, or adjoint routines.
Basis indices use mode zero as the least significant occupation bit.
"""

from functools import cache
from itertools import product

import numpy as np
import pytest
from sympy import symbols

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon

GENERAL_TYPES = [zfog.RealTermSum, zfog.ComplexTermSum]
NUMERIC_TYPES = [
    pytest.param(zfog.RealTermSum, id="general-real"),
    pytest.param(zfog.ComplexTermSum, id="general-complex"),
    pytest.param(zfon.RealTermSum, id="normal-real"),
    pytest.param(zfon.ComplexTermSum, id="normal-complex"),
]
COMPLEX_TYPES = [zfog.ComplexTermSum, zfon.ComplexTermSum]


@cache
def _ladder_matrix(mode, creation, n_mode):
    """Build a ladder operator from its Jordan--Wigner tensor factors."""
    annihilator = np.array([[0, 1], [0, 0]])
    local = annihilator.T if creation else annihilator
    matrix = np.ones((1, 1))
    for position in reversed(range(n_mode)):
        if position < mode:
            factor = np.diag([1, -1])
        elif position == mode:
            factor = local
        else:
            factor = np.eye(2)
        matrix = np.kron(matrix, factor)
    return matrix


def _word_matrix(operators, n_mode):
    """Multiply reference matrices in the supplied operator order."""
    matrix = np.eye(1 << n_mode, dtype=complex)
    for mode, creation in operators:
        matrix = matrix @ _ladder_matrix(mode, creation, n_mode)
    return matrix


def _reference(terms, n_mode):
    """Construct a reference matrix directly from the input specifications."""
    matrix = np.zeros((1 << n_mode, 1 << n_mode), dtype=complex)
    for source, coefficient in terms:
        operators = [(int(token[1:].rstrip("^")), token.endswith("^")) for token in source.split()]
        matrix += coefficient * _word_matrix(operators, n_mode)
    return matrix


def _matrix(operator):
    """Interpret returned terms using reference matrices, preserving their coefficients."""
    n_mode = len(operator.modes)
    matrix = np.zeros((1 << n_mode, 1 << n_mode), dtype=complex)
    for term in operator:
        matrix += complex(term.coeff) * _word_matrix(term.string.get_ops(), n_mode)
    return matrix


def _sum(sum_type, terms, n_mode):
    """Construct a sum with sufficient initial capacity for raw operator words."""
    kwargs = {"max_string_len": 12} if sum_type in GENERAL_TYPES else {}
    return sum_type.from_iterable(terms, n_mode, **kwargs)


def _assert_matrix(operator, expected, *, context=""):
    """Compare algebraic action without depending on term order or stored zero rows."""
    np.testing.assert_allclose(
        _matrix(operator), expected, atol=1e-12, rtol=1e-12, err_msg=f"{context}\n{operator}"
    )


@pytest.mark.parametrize("sum_type", NUMERIC_TYPES)
@pytest.mark.parametrize(("i", "j"), product(range(3), repeat=2))
def test_canonical_anticommutation(sum_type, i, j):
    a_i, a_j = [_sum(sum_type, [(f"F{mode}", 1)], 3) for mode in (i, j)]
    c_i, c_j = [_sum(sum_type, [(f"F{mode}^", 1)], 3) for mode in (i, j)]
    zero = np.zeros((8, 8))

    _assert_matrix(a_i * a_j + a_j * a_i, zero)
    _assert_matrix(c_i * c_j + c_j * c_i, zero)
    _assert_matrix(a_i * c_j + c_j * a_i, np.eye(8) if i == j else zero)


@pytest.mark.parametrize("sum_type", NUMERIC_TYPES)
@pytest.mark.parametrize(("i", "j"), product(range(3), repeat=2))
def test_number_operator_algebra(sum_type, i, j):
    number = _sum(sum_type, [(f"F{i}^ F{i}", 1)], 3)
    other_number = _sum(sum_type, [(f"F{j}^ F{j}", 1)], 3)
    annihilation = _sum(sum_type, [(f"F{j}", 1)], 3)
    creation = _sum(sum_type, [(f"F{j}^", 1)], 3)
    n_matrix = _ladder_matrix(i, True, 3) @ _ladder_matrix(i, False, 3)

    _assert_matrix(number * number, n_matrix)
    _assert_matrix(number * other_number - other_number * number, np.zeros((8, 8)))
    _assert_matrix(
        number * annihilation - annihilation * number, -(i == j) * _ladder_matrix(j, False, 3)
    )
    _assert_matrix(number * creation - creation * number, (i == j) * _ladder_matrix(j, True, 3))


@pytest.mark.parametrize("sum_type", NUMERIC_TYPES)
@pytest.mark.parametrize(("i", "j", "k", "l"), product(range(3), repeat=4))
def test_bilinear_commutator(sum_type, i, j, k, l):
    lhs = _sum(sum_type, [(f"F{i}^ F{j}", 1)], 3)
    rhs = _sum(sum_type, [(f"F{k}^ F{l}", 1)], 3)
    # [a_i^dagger a_j, a_k^dagger a_l] = delta_jk a_i^dagger a_l - delta_il a_k^dagger a_j.
    expected = (j == k) * (_ladder_matrix(i, True, 3) @ _ladder_matrix(l, False, 3))
    expected -= (i == l) * (_ladder_matrix(k, True, 3) @ _ladder_matrix(j, False, 3))

    _assert_matrix(lhs * rhs - rhs * lhs, expected)


@pytest.mark.parametrize("sum_type", NUMERIC_TYPES)
@pytest.mark.parametrize(
    ("left_word", "right_word", "sign"),
    [
        ("F0", "F2", -1),
        ("F0^", "F2", -1),
        ("F0^ F1", "F2^", 1),
        ("F0^ F1", "F2^ F3", 1),
        ("F0^ F1^ F0", "F2^", -1),
    ],
)
def test_disjoint_support_graded_commutation(sum_type, left_word, right_word, sign):
    lhs = _sum(sum_type, [(left_word, 1)], 4)
    rhs = _sum(sum_type, [(right_word, 1)], 4)
    expected = _reference([(left_word, 1)], 4) @ _reference([(right_word, 1)], 4)

    _assert_matrix(lhs * rhs, expected)
    _assert_matrix(rhs * lhs, sign * expected)
    _assert_matrix(lhs * rhs - sign * (rhs * lhs), np.zeros((16, 16)))


@pytest.mark.parametrize("sum_type", NUMERIC_TYPES)
@pytest.mark.parametrize(
    ("i", "j"), [(0, 63), (63, 64), (64, 127), (127, 128), (64, 64), (128, 128)]
)
def test_multiword_canonical_anticommutation(sum_type, i, j):
    n_mode = max(i, j) + 1
    annihilators = [_sum(sum_type, [(f"F{mode}", 1)], n_mode) for mode in (i, j)]
    creators = [_sum(sum_type, [(f"F{mode}^", 1)], n_mode) for mode in (i, j)]

    def coefficients(operator):
        if sum_type in GENERAL_TYPES:
            operator = operator.to_normal_ordered()
        return {
            tuple(term.string.get_ops()): complex(term.coeff) for term in operator.filter_nonzero()
        }

    a_i, a_j = annihilators
    c_i, c_j = creators
    assert coefficients(a_i * a_j + a_j * a_i) == {}
    assert coefficients(c_i * c_j + c_j * c_i) == {}
    assert coefficients(a_i * c_j + c_j * a_i) == ({(): 1} if i == j else {})


def _weighted_operators(sum_type):
    """Return mixed-parity sums and references, including complex weights and duplicate inputs."""
    specifications = [
        [("", 0.5), ("F0^ F1", 2), ("F1^ F2^ F0 F2", -0.5), ("F0^ F1", -0.25), ("F0", -0.5)],
        [("F1^ F0", 1), ("F2 F3", -2), ("F2^ F3", 0.5), ("F1^", 0.25)],
        [("F0^ F3", 0.25), ("F0 F1", 1), ("F2^ F2", -1), ("F3", 0.25)],
    ]
    if sum_type in COMPLEX_TYPES:
        specifications = [
            [(word, value * (1 + (index + 1) * 0.25j)) for index, (word, value) in enumerate(terms)]
            for terms in specifications
        ]
    return (
        [_sum(sum_type, terms, 4) for terms in specifications],
        [_reference(terms, 4) for terms in specifications],
    )


@pytest.fixture(params=NUMERIC_TYPES)
def weighted_operators(request):
    return request.param, *_weighted_operators(request.param)


@pytest.mark.parametrize("sum_type", NUMERIC_TYPES)
def test_identity_plus_number_squared(sum_type):
    operator = _sum(sum_type, [("", 1), ("F0^ F0", 1)], 1)
    _assert_matrix(operator * operator, np.diag([1, 4]))


@pytest.mark.parametrize("sum_type", GENERAL_TYPES)
@pytest.mark.parametrize("length", [2, 63, 64, 65])
def test_general_product_collects_equal_words(sum_type, length):
    word = ((0, True), (0, False)) * length
    word = word[:length]
    scale = 1j if sum_type is zfog.ComplexTermSum else 1
    lhs = _sum(sum_type, [("", 1), (list(word), 2), (list(word * 2), -1)], 1)
    rhs = _sum(sum_type, [("", scale), (list(word), -2 * scale), (list(word * 2), 3 * scale)], 1)

    result = lhs * rhs
    coeffs = {tuple(term.string.get_ops()): term.coeff for term in result}
    assert len(coeffs) == len(result)
    assert {w: c for w, c in coeffs.items() if c} == {
        (): scale,
        word * 2: -2 * scale,
        word * 3: 8 * scale,
        word * 4: -3 * scale,
    }


@pytest.mark.parametrize("sum_type", GENERAL_TYPES)
def test_normal_ordering_cancels_across_terms(sum_type):
    scale = 1j if sum_type is zfog.ComplexTermSum else 1
    raw = _sum(
        sum_type,
        [("F0 F0^", scale), ("F1", 2 * scale), ("F0^ F0", scale), ("", -scale)],
        2,
    )

    result = raw.to_normal_ordered().filter_nonzero()
    assert [(term.string.get_ops(), term.coeff) for term in result] == [([(1, False)], 2 * scale)]


def test_sum_multiplication_and_associativity(weighted_operators):
    _, (a, b, c), (ma, mb, mc) = weighted_operators
    before = [str(operator) for operator in (a, b, c)]

    _assert_matrix(a * b, ma @ mb)
    _assert_matrix((a * b) * c, ma @ mb @ mc)
    _assert_matrix(a * (b * c), ma @ mb @ mc)
    assert [str(operator) for operator in (a, b, c)] == before


def test_distributivity_and_scalar_linearity(weighted_operators):
    sum_type, (a, b, c), (ma, mb, mc) = weighted_operators
    scalar = -0.5j if sum_type in COMPLEX_TYPES else -0.5

    _assert_matrix(a * (b + c), ma @ (mb + mc))
    _assert_matrix((a + b) * c, (ma + mb) @ mc)
    _assert_matrix((a * scalar) * b, scalar * (ma @ mb))
    _assert_matrix(a * (b * scalar), scalar * (ma @ mb))


def test_adjoint_algebra(weighted_operators):
    _, (a, b, _), (ma, mb, _) = weighted_operators

    _assert_matrix(a.daggered(), ma.conj().T)
    _assert_matrix(a.daggered().daggered(), ma)
    _assert_matrix((a * b).daggered(), mb.conj().T @ ma.conj().T)
    _assert_matrix(b.daggered() * a.daggered(), mb.conj().T @ ma.conj().T)
    _assert_matrix(a.daggered() * a, ma.conj().T @ ma)


def test_zero_and_identity_algebra(weighted_operators):
    sum_type, (a, _, _), (ma, _, _) = weighted_operators
    zero = _sum(sum_type, [], 4)
    identity = _sum(sum_type, [("", 1)], 4)

    _assert_matrix(a * identity, ma)
    _assert_matrix(identity * a, ma)
    _assert_matrix(a * zero, np.zeros((16, 16)))
    _assert_matrix(zero * a, np.zeros((16, 16)))
    _assert_matrix(a - a, np.zeros((16, 16)))


@pytest.mark.parametrize("sum_type", NUMERIC_TYPES)
def test_zero_mode_scalar_algebra(sum_type):
    zero = _sum(sum_type, [], 0)
    scalar = _sum(sum_type, [("", 2)], 0)

    _assert_matrix(scalar, np.array([[2]]))
    _assert_matrix(scalar * scalar, np.array([[4]]))
    _assert_matrix(scalar * zero, np.array([[0]]))
    _assert_matrix(scalar.daggered(), np.array([[2]]))


@pytest.mark.parametrize("sum_type", [zfon.RealTermSum, zfon.ComplexTermSum])
def test_commutator_methods(sum_type):
    (a, b, _), (ma, mb, _) = _weighted_operators(sum_type)

    _assert_matrix(a.commutator(b), ma @ mb - mb @ ma)
    _assert_matrix(a.anticommutator(b), ma @ mb + mb @ ma)


@pytest.mark.parametrize("sum_type", [zfon.RealTermSum, zfon.ComplexTermSum])
def test_jacobi_identity(sum_type):
    (a, b, c), _ = _weighted_operators(sum_type)
    result = (
        a.commutator(b.commutator(c))
        + b.commutator(c.commutator(a))
        + c.commutator(a.commutator(b))
    )

    _assert_matrix(result, np.zeros((16, 16)))


@pytest.mark.parametrize("sum_type", [zfog.RealTermSum, zfog.ComplexTermSum])
@pytest.mark.parametrize("length", range(5))
def test_normal_ordering_all_short_words(sum_type, length):
    alphabet = [(mode, creation) for mode in range(2) for creation in (False, True)]
    coefficient = 0.5j if sum_type in COMPLEX_TYPES else -0.5
    for operators in product(alphabet, repeat=length):
        raw = _sum(sum_type, [(list(operators), coefficient)], 2)
        expected = coefficient * _word_matrix(operators, 2)
        normal = raw.to_normal_ordered()

        _assert_matrix(normal, expected, context=f"Raw word: {operators!r}")
        _assert_matrix(normal.to_general(), expected)
        _assert_matrix(normal.to_general().to_normal_ordered(), expected)
        _assert_matrix(raw.daggered().to_normal_ordered(), expected.conj().T)


@pytest.mark.parametrize("sum_type", [zfog.RealTermSum, zfog.ComplexTermSum])
def test_long_contractions_and_conversion_of_sums(sum_type):
    specifications = [
        ("F0 F1 F2 F2^ F1^ F0^", 1),
        ("F1 F0^ F2 F1^ F0 F2^", -0.5),
        ("F0 F0^ F0 F0^ F0 F0^", 0.25),
        ("F2^ F2^ F1 F0", 2),
    ]
    if sum_type in COMPLEX_TYPES:
        specifications = [(word, coefficient * 1j) for word, coefficient in specifications]
    raw = _sum(sum_type, specifications, 3)
    expected = _reference(specifications, 3)

    _assert_matrix(raw.to_normal_ordered(), expected)
    _assert_matrix(raw.daggered().to_normal_ordered(), expected.conj().T)


@pytest.mark.parametrize("source", ["F0", "F0^", "F0^ F1", "F0^ F1^", "F0 F1"])
@pytest.mark.parametrize(
    "term_type", [zfon.SignTerm, zfon.RealTerm, zfon.ComplexTerm, zfon.SymbolicTerm]
)
def test_nilpotent_terms(term_type, source):
    term = term_type(3, source)

    assert len(term * term) == 0


@pytest.mark.parametrize(
    ("term_type", "coefficient"),
    [
        pytest.param(zfon.SignTerm, -1, id="normal-sign"),
        pytest.param(zfon.RealTerm, -0.5, id="normal-real"),
        pytest.param(zfon.ComplexTerm, 1 + 2j, id="normal-complex"),
        pytest.param(zfon.SymbolicTerm, 1 + 2j, id="normal-symbolic"),
        pytest.param(zfog.RealTerm, -0.5, id="general-real"),
        pytest.param(zfog.ComplexTerm, 1 + 2j, id="general-complex"),
        pytest.param(zfog.SymbolicTerm, 1 + 2j, id="general-symbolic"),
    ],
)
@pytest.mark.parametrize(("n_cre", "n_ann"), product(range(5), repeat=2))
def test_term_adjoints_at_each_rank(term_type, coefficient, n_cre, n_ann):
    operators = [(mode, True) for mode in range(n_cre)] + [(mode, False) for mode in range(n_ann)]
    source = " ".join(f"F{mode}{'^' if creation else ''}" for mode, creation in operators)
    term = term_type(4, (source, coefficient))
    before = str(term)
    expected = coefficient * _word_matrix(operators, 4)

    adjoint = term.daggered()
    restored = adjoint.daggered()

    for result, reference in [(adjoint, expected.conj().T), (restored, expected)]:
        matrix = complex(result.coeff) * _word_matrix(result.string.get_ops(), 4)
        np.testing.assert_allclose(matrix, reference, atol=1e-12, rtol=1e-12, err_msg=source)
    assert str(term) == before


def test_symbolic_coefficients_and_adjoint():
    x, y = symbols("x y")
    lhs = zfon.SymbolicTerm(2, ("F0^ F1", x))
    rhs = zfon.SymbolicTerm(2, ("F1^ F0", y))
    substitutions = {x: 2 + 3j, y: -0.5j}
    expected = _reference([("F0^ F1", substitutions[x])], 2) @ _reference(
        [("F1^ F0", substitutions[y])], 2
    )

    _assert_matrix((lhs * rhs).subs(substitutions), expected)
    _assert_matrix((lhs * rhs).daggered().subs(substitutions), expected.conj().T)
