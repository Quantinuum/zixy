"""Numeric bulk operations preserve views, ordering and container invariants."""

import pytest

from zixy.container.coeffs import ComplexCoeffs, RealCoeffs
from zixy.container.data import TermData
from zixy.container.terms import TermSet
from zixy.fermion import state as fermion_state
from zixy.fermion.operator import general, normal
from zixy.qubit import pauli, state as qubit_state

DOMAINS = (
    (pauli, ("X0", "Y1", "Z2"), {}),
    (general, ("F0^", "F1 F2^", "F2^ F0 F1^"), {"max_len": 5}),
    (normal, ("F0^", "F1", "F2^"), {}),
    (fermion_state, ("[1, 0, 0]", "[0, 1, 0]", "[0, 0, 1]"), {}),
    (qubit_state, ("[1, 0, 0]", "[0, 1, 0]", "[0, 0, 1]"), {}),
)


def snapshot(terms):
    return [(str(term.cmpnt), term.coeff) for term in terms]


@pytest.mark.parametrize("domain,keys,kwargs", DOMAINS)
@pytest.mark.parametrize("name", ("RealTermSum", "ComplexTermSum", "RealTermSet", "ComplexTermSet"))
def test_bulk_copy_and_collection(domain, keys, kwargs, name):
    cls = getattr(domain, name)
    data = [(keys[0], 1), (keys[1], 0), (keys[2], 1e-20), (keys[0], 2)]
    source = cls.from_iterable(data, 3, **kwargs)
    assert len(source) == 3
    assert source.lookup_coeff(keys[0]) == (3 if name.endswith("Sum") else 2)
    before = snapshot(source)
    for nonzero in (False, True):
        copy = source.filter_nonzero() if nonzero else source.clone()
        assert snapshot(copy) == [(key, c) for key, c in before if not nonzero or c != 0]
        for key, c in snapshot(copy):
            assert copy.lookup_coeff(key) == c
        copy.remove(keys[0])
        copy.insert((keys[0], 7))
        assert copy.lookup_coeff(keys[0]) == 7
        assert snapshot(source) == before
    # Construction from a sliced array must not copy unselected terms.
    terms = source.to_terms()
    sliced = cls.from_terms(terms[::-2])
    assert snapshot(sliced) == snapshot(terms[::-2])
    assert snapshot(cls.from_terms(terms)) == before


@pytest.mark.parametrize("cls", (RealCoeffs, ComplexCoeffs))
@pytest.mark.parametrize(
    "indexer", (slice(None), slice(None, None, -1), slice(1, 6, 2), slice(5, 0, -2), slice(2, 2))
)
def test_numeric_vector_views(cls, indexer):
    values = [cls.coeff_type(i + 1) for i in range(6)]
    vector = cls.from_sequence(values)
    view = vector[indexer]
    clone = view.clone()
    assert list(clone) == values[indexer]
    assert clone.is_owning()
    assert list(-view) == [-v for v in values[indexer]]
    view.scale(2)
    expected = values.copy()
    for i in range(6)[indexer]:
        expected[i] *= 2
    assert list(vector) == expected
    assert list(clone) == values[indexer]
    view.fill(cls.coeff_type(9))
    for i in range(6)[indexer]:
        expected[i] = 9
    assert list(vector) == expected


@pytest.mark.parametrize("cls", (pauli.RealTermSum, pauli.ComplexTermSum))
def test_bulk_addition(cls):
    lhs = cls.from_iterable([("X0", 1), ("Y0", 2), ("Z0", 0)], 2)
    rhs = cls.from_iterable([("Y0", -2), ("X1", 3), ("Z1", 1e-20)], 2)
    rhs_before = snapshot(rhs)
    lhs += rhs
    assert snapshot(lhs) == snapshot(cls.from_iterable([("X0", 1), ("X1", 3), ("Z1", 1e-20)], 2))
    assert snapshot(rhs) == rhs_before
    lhs += lhs
    assert lhs.lookup_coeff("X0") == 2
    assert lhs.lookup_coeff("X1") == 6
    lhs -= lhs
    assert len(lhs) == 0
    lhs.insert(("Z0", 4))
    assert lhs.lookup_coeff("Z0") == 4
    with pytest.raises(ValueError):
        lhs += cls.from_iterable([("X0", 1)], 3)


@pytest.mark.parametrize("cls", (general.RealTermSum, general.ComplexTermSum))
def test_product_preserves_lengths_and_lookup(cls):
    lhs = cls.from_iterable([("F0^", 2), ("F1 F2^", 3)], 3, max_len=4)
    rhs = cls.from_iterable([("F2", 5), ("F0^ F1", 7)], 3, max_len=3)
    product = lhs * rhs
    expected = cls.from_iterable(
        [
            ("F0^ F2", 10),
            ("F0^ F0^ F1", 14),
            ("F1 F2^ F2", 15),
            ("F1 F2^ F0^ F1", 21),
        ],
        3,
        max_len=7,
    )
    assert snapshot(product) == snapshot(expected)
    for key, coeff in snapshot(expected):
        assert product.lookup_coeff(key) == coeff


@pytest.mark.parametrize("domain,keys,kwargs", DOMAINS)
@pytest.mark.parametrize("name", ("RealTermSum", "ComplexTermSum"))
def test_numeric_norms(domain, keys, kwargs, name):
    cls = getattr(domain, name)
    values = (3, -4, 0) if name == "RealTermSum" else (3 + 4j, -4j, 0j)
    terms = cls.from_iterable(zip(keys, values), 3, **kwargs)
    assert terms.l1_norm == pytest.approx(sum(abs(c) for c in values))
    assert terms.l2_norm_square == pytest.approx(sum(abs(c) ** 2 for c in values))
    assert type(terms.l1_norm) is (complex if name == "ComplexTermSum" else float)


@pytest.mark.parametrize("cls", (RealCoeffs, ComplexCoeffs))
def test_native_slice_validation(cls):
    vector = cls.from_sequence([1, 2, 3])
    with pytest.raises(ValueError):
        vector._impl.copy_slice(slice(None, None, 0))
    with pytest.raises(ValueError):
        vector._impl.scale_slice(slice(None, None, 0), 2)
    assert list(vector) == [1, 2, 3]


@pytest.mark.parametrize("domain", (fermion_state, qubit_state))
def test_reused_state_string_assignment(domain):
    state = domain.String.from_str("[1, 0, 0]", 3)
    state.set("[0, 1, 0]")
    assert str(state) == "[0, 1, 0]"
    state.set("[]")
    assert str(state) == "[0, 0, 0]"


@pytest.mark.parametrize("domain,keys,kwargs", DOMAINS)
@pytest.mark.parametrize("name", ("RealTermSum", "ComplexTermSum"))
def test_required_numeric_bulk_api(domain, keys, kwargs, name, monkeypatch):
    cls = getattr(domain, name)
    lhs = cls.from_iterable([(keys[0], 2), (keys[1], 3)], 3, **kwargs)
    rhs = cls.from_iterable([(keys[0], -2), (keys[2], 4)], 3, **kwargs)
    array = cls.terms_type.from_iterable([(keys[0], 1), (keys[1], 2), (keys[0], 3)], 3, **kwargs)

    def no_python_insertion(*args, **kwargs):
        pytest.fail("Numeric bulk operations must not use Python insertion")

    monkeypatch.setattr(TermSet, "insert", no_python_insertion)
    monkeypatch.setattr(TermSet, "insert_iterable", no_python_insertion)
    assert len(lhs.clone()) == 2
    assert len(lhs.filter_nonzero()) == 2
    constructed = cls.from_iterable([(keys[0], 1), (keys[0], 2)], 3, **kwargs)
    assert constructed.lookup_coeff(keys[0]) == 3
    copied = cls.from_terms(array[::-1])
    assert copied.lookup_coeff(keys[0]) == 1
    assert copied.lookup_coeff(keys[1]) == 2
    # Component and coefficient views can also be nested inside an owning Terms.
    nested = cls.terms_type._create(TermData(array.cmpnts[::-1], array.coeffs[::-1]))
    assert snapshot(cls.from_terms(nested)) == snapshot(copied)
    lhs += rhs
    assert lhs.lookup_coeff(keys[0]) is None
    assert lhs.lookup_coeff(keys[1]) == 3
    assert lhs.lookup_coeff(keys[2]) == 4
    lhs += lhs
    assert lhs.lookup_coeff(keys[1]) == 6
    lhs -= lhs
    assert len(lhs) == 0
    lhs += cls(3, **kwargs)
    assert len(lhs) == 0


@pytest.mark.parametrize("cls", (general.RealTermSum, general.ComplexTermSum))
def test_general_bulk_addition_different_storage_layouts(cls):
    lhs = cls.from_iterable([("F1^", 2)], 3, max_len=5)
    rhs = cls.from_iterable([("F1^", 3), ("F2 F1^", 4)], 3, max_len=128)
    lhs += rhs
    assert len(lhs) == 2
    assert lhs.lookup_coeff("F1^") == 5
    assert lhs.lookup_coeff("F2 F1^") == 4
    assert rhs.lookup_coeff("F1^") == 3
    # Validate all source strings before mutating the destination.
    bad = cls.from_iterable([("F1^", 10), ("F1^ " * 6, 1)], 3, max_len=128)
    before = snapshot(lhs)
    with pytest.raises(ValueError, match="max_len"):
        lhs += bad
    assert snapshot(lhs) == before


@pytest.mark.parametrize("domain,keys,kwargs", DOMAINS)
@pytest.mark.parametrize("name", ("RealTermSum", "ComplexTermSum"))
def test_bulk_addition_rejects_incompatible_spaces(domain, keys, kwargs, name):
    cls = getattr(domain, name)
    lhs = cls.from_iterable([(keys[0], 1)], 3, **kwargs)
    rhs = cls.from_iterable([(keys[0], 2)], 4, **kwargs)
    before = snapshot(lhs)
    with pytest.raises(ValueError):
        lhs += rhs
    assert snapshot(lhs) == before
