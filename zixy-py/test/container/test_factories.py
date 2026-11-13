"""Factories distinguish storage adoption, independent allocation, and shared views."""

import numpy as np
import pytest
from sympy import Symbol

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy._zixy import Modes, Qubits
from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSign,
    ComplexSignCoeffs,
    RealCoeffs,
    Sign,
    SignCoeffs,
    SymbolicCoeffs,
    unit,
)
from zixy.container.terms import TermSet


@pytest.fixture(params=[zqo, zqs, zfog, zfon, zfs], ids=lambda family: family.__name__)
def family_case(request):
    family = request.param
    space = (
        Qubits.from_inds([5, 2]) if family in (zqo, zqs) else Modes.from_pair_count_spin_minor(1)
    )
    specs = {
        zqo: ["X0", "Y1"],
        zqs: ["[1, 0]", "[0, 1]"],
        zfog: ["F0^ F1", "F1 F0^"],
        zfon: ["F0^", "F1"],
        zfs: ["[1, 0]", "[0, 1]"],
    }[family]
    kwargs = {"max_string_len": 64} if family is zfog else {}
    strings = family.Strings.from_iterable(specs, space, **kwargs)
    terms = family.RealTerms.from_iterable(zip(strings, [2.0, 3.0]), space, **kwargs)
    return family, strings, terms


@pytest.mark.parametrize("kind", ["strings", "terms"])
@pytest.mark.parametrize("indexer", [slice(None), slice(0, 0), slice(None, None, -1)])
def test_sequence_factories_preserve_ownership_and_configuration(family_case, kind, indexer):
    _, strings, terms = family_case
    source = strings if kind == "strings" else terms
    storage = source.clone()._impl
    owner = type(source)._create(storage)
    view = type(source)._view(storage, indexer)
    assert owner._impl is view._impl is storage
    assert owner.is_owning()
    assert not view.is_owning()
    assert str(view) == str(owner[indexer])
    with pytest.raises(ValueError):
        view.resize(3)

    empty = view._new()
    assert type(empty) is type(source)
    assert empty.is_owning()
    assert len(empty) == 0
    assert empty.to_dict()["space"] == source.to_dict()["space"]
    if hasattr(source, "max_string_len"):
        assert empty.max_string_len == source.max_string_len
    clone = view.clone()
    assert clone.is_owning()
    before = clone.to_dict()
    if len(view):
        view[0].clear()
        assert str(view[0]) == str(owner[indexer][0])
        assert clone.to_dict() == before


@pytest.mark.parametrize("empty", [False, True])
def test_item_allocation_is_owning_and_independent(family_case, empty):
    _, strings, terms = family_case
    string_source = strings[:0] if empty else strings[::-1]
    term_source = terms[:0] if empty else terms[::-1]
    before_strings, before_terms = strings.to_dict(), terms.to_dict()
    string, term = string_source.new_string(), term_source.new_term()
    assert type(string) is strings.string_type
    assert type(term) is terms.term_type
    assert string.is_owning()
    assert term.is_owning()
    assert string.to_dict()["space"] == strings.to_dict()["space"]
    assert term.to_dict()["space"] == terms.to_dict()["space"]
    assert term.coeff == 1.0
    if hasattr(strings, "max_string_len"):
        assert string.max_string_len == strings.max_string_len
        assert term.string.max_string_len == terms.max_string_len
    string.set(strings[0])
    term.set(terms[0])
    assert strings.to_dict() == before_strings
    assert terms.to_dict() == before_terms


def test_item_views_share_storage_and_validate_indices(family_case):
    family, strings, terms = family_case
    for cls, source in [(family.String, strings), (family.RealTerm, terms)]:
        view = cls._view(source._impl, -1)
        assert not view.is_owning()
        assert view.index == len(source) - 1
        assert view._impl is source._impl
        assert str(view) == str(source[-1])
        before = str(source[-1])
        view.clear()
        assert str(source[-1]) != before
        assert str(view) == str(source[-1])
        with pytest.raises(IndexError):
            cls._view(source._impl, len(source))


@pytest.mark.parametrize("source_kind", ["owning", "view", "default"])
@pytest.mark.parametrize("coefficient_mode", ["omitted", "none", "explicit"])
def test_term_from_string_defaults_and_ownership(family_case, source_kind, coefficient_mode):
    family, strings, _ = family_case
    coefficients = {
        "Sign": Sign.from_int(-1),
        "ComplexSign": ComplexSign.from_complex(-1j),
        "Real": 2.5,
        "Complex": 1 + 2j,
        "Symbolic": Symbol("x"),
    }
    for prefix, coeff in coefficients.items():
        cls = getattr(family, prefix + "Term", None)
        if cls is None:
            continue
        template = strings.clone()
        source = {
            "owning": template[1].clone(),
            "view": template[1],
            "default": template.new_string(),
        }[source_kind]
        before = source.to_dict()
        if coefficient_mode == "omitted":
            result = cls.from_string(source)
        else:
            result = cls.from_string(source, coeff=None if coefficient_mode == "none" else coeff)

        expected = coeff if coefficient_mode == "explicit" else unit(cls.coeff_type)
        assert type(result) is cls
        assert result.is_owning()
        assert result._data.strings.is_owning()
        assert result._data.coeffs.is_owning()
        assert isinstance(result.coeff, cls.coeff_type)
        assert result.coeff == expected
        assert result.string == source
        assert result.to_dict()["space"] == before["space"]
        if hasattr(source, "max_string_len"):
            assert result.string.max_string_len == source.max_string_len
        result.string.clear()
        result.coeff = -expected
        assert source.to_dict() == before
        snapshot = result.to_dict()
        source.set(template[0])
        assert result.to_dict() == snapshot


@pytest.mark.parametrize("prefix", ["Real", "Complex", "Symbolic"])
def test_term_from_string_retains_zero_coefficient(family_case, prefix):
    family, strings, _ = family_case
    cls = getattr(family, prefix + "Term")
    zero = 0 * unit(cls.coeff_type)

    result = cls.from_string(strings[0], coeff=zero)

    assert result.coeff == zero
    assert isinstance(result.coeff, cls.coeff_type)
    assert result.string == strings[0]


def test_public_factories_copy_visible_values(family_case):
    family, strings, terms = family_case
    string_view, term_view = strings[::-1], terms[::-1]
    results = [
        family.Strings.from_string(string_view[0]),
        family.StringSet.from_strings(string_view),
        family.RealTerm.from_string(string_view[0], 2.0),
        family.RealTermSet.from_terms(term_view),
        family.RealTermSum.from_terms(term_view),
    ]
    snapshots = [result.to_dict() for result in results]
    assert snapshots[0]["strings"] == string_view.to_dict()["strings"][:1]
    assert snapshots[-1]["strings"] == term_view.to_dict()["strings"]
    for result in results:
        if hasattr(result, "is_owning"):
            assert result.is_owning()
    for source in [strings, terms]:
        for item in source:
            item.clear()
    assert [result.to_dict() for result in results] == snapshots


def test_keyed_factories_adopt_data_and_maps_together(family_case):
    family, strings, terms = family_case
    string_set = family.StringSet.from_strings(strings)
    adopted_strings = type(string_set)._create(string_set._impl, map_=string_set._map)
    assert adopted_strings._impl is string_set._impl
    assert adopted_strings._map is string_set._map
    for cls in [family.RealTermSet, family.RealTermSum]:
        source = cls.from_terms(terms)
        adopted = cls._create(source._impl, map_=source._string_set._map)
        assert adopted._impl is source._impl
        assert adopted._string_set._map is source._string_set._map
        assert adopted._string_set._impl is adopted._impl.strings._impl
        empty = adopted._new()
        assert len(empty) == 0
        assert empty.to_dict()["space"] == adopted.to_dict()["space"]
        assert empty._string_set._impl is empty._impl.strings._impl
        empty.insert(terms[0])
        assert len(adopted) == len(source) == 2
        assert empty.lookup_coeff(terms[0].string) == terms[0].coeff


@pytest.mark.parametrize(
    "cls", [SignCoeffs, ComplexSignCoeffs, RealCoeffs, ComplexCoeffs, SymbolicCoeffs]
)
@pytest.mark.parametrize("size", [0, 1, 4])
def test_coefficient_size_construction(cls, size):
    one = unit(cls.coeff_type)
    coeffs = cls(size=size)
    other = cls(size=size)

    assert coeffs.is_owning()
    assert len(coeffs) == size
    assert list(coeffs) == [one] * size
    assert len(cls()) == 0
    if size:
        coeffs[0] = -one
        assert list(other) == [one] * size


@pytest.mark.parametrize(
    "cls", [SignCoeffs, ComplexSignCoeffs, RealCoeffs, ComplexCoeffs, SymbolicCoeffs]
)
def test_coefficient_size_rejects_negative_values(cls):
    with pytest.raises(ValueError, match="size must be non-negative"):
        cls(size=-1)


@pytest.mark.parametrize(
    "cls", [SignCoeffs, ComplexSignCoeffs, RealCoeffs, ComplexCoeffs, SymbolicCoeffs]
)
def test_coefficient_factories_share_only_for_views(cls):
    one = unit(cls.coeff_type)
    data = cls.from_scalar(one, 3)._impl
    owner, view = cls._create(data), cls._view(data, slice(None))
    assert owner.is_owning()
    assert not view.is_owning()
    assert owner._impl is view._impl is data
    copy = view.clone()
    view.fill(-one)
    assert list(owner) == [-one] * 3
    assert list(copy) == [one] * 3
    assert copy.is_owning()
    empty = view._new()
    assert type(empty) is cls
    assert empty.is_owning()
    assert len(empty) == 0


@pytest.mark.parametrize(
    "cls", [SignCoeffs, ComplexSignCoeffs, RealCoeffs, ComplexCoeffs, SymbolicCoeffs]
)
@pytest.mark.parametrize("source_kind", ["list", "tuple", "iterator", "generator", "empty"])
def test_coefficient_iterable_construction(cls, source_kind):
    one = unit(cls.coeff_type)
    values = [one, -one, one]
    source = {
        "list": values,
        "tuple": tuple(values),
        "iterator": iter(values),
        "generator": (value for value in values),
        "empty": iter(()),
    }[source_kind]
    expected = [] if source_kind == "empty" else values.copy()

    result = cls.from_iterable(source=source)

    assert type(result) is cls
    assert result.is_owning()
    assert list(result) == expected
    if expected:
        result[0] = -one
        assert values == expected


@pytest.mark.parametrize("cls", [RealCoeffs, ComplexCoeffs])
@pytest.mark.parametrize("source_kind", ["sequence", "array", "strided-array", "generator"])
def test_numeric_iterable_uses_bulk_conversion(cls, source_kind, monkeypatch):
    values = np.arange(6, dtype=cls.coeff_type)
    source = {
        "sequence": list(values),
        "array": values,
        "strided-array": values[::2],
        "generator": (value for value in values),
    }[source_kind]
    expected = list(values[::2] if source_kind == "strided-array" else values)

    def no_python_append(*args, **kwargs):
        pytest.fail("Numeric iterable construction must use the bulk array path")

    monkeypatch.setattr(cls, "append", no_python_append)
    result = cls.from_iterable(source)

    assert list(result) == expected
    assert result.is_owning()
    values[:] = 0
    assert list(result) == expected


@pytest.mark.parametrize(
    "kind", ["Strings", "StringSet", "RealTerms", "RealTermSet", "RealTermSum"]
)
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_domain_iterable_signatures_and_duplicate_policies(family_case, kind, streamed, empty):
    family, strings, _ = family_case
    source = [strings[0], strings[1], strings[0]]
    if kind.startswith("Real"):
        source = list(zip(source, [2.0, 3.0, 4.0]))
    if empty:
        source = []
    kwargs = {"qubits": strings.qubits} if family in (zqo, zqs) else {"modes": strings.modes}
    if family is zfog:
        kwargs["max_string_len"] = 64
    cls = getattr(family, kind)

    result = cls.from_iterable(source=(value for value in source) if streamed else source, **kwargs)

    assert type(result) is cls
    assert len(result) == (0 if empty else 3 if kind in ("Strings", "RealTerms") else 2)
    assert result.to_dict()["space"] == strings.to_dict()["space"]
    if hasattr(result, "is_owning"):
        assert result.is_owning()
    if family is zfog:
        assert result.max_string_len == 64
    if not empty and kind.startswith("Real"):
        expected = {
            "RealTerms": [2.0, 3.0, 4.0],
            "RealTermSet": [4.0, 3.0],
            "RealTermSum": [6.0, 3.0],
        }
        assert list(result.coeffs) == expected[kind]
    snapshot = result.to_dict()
    for string in strings:
        string.clear()
    assert result.to_dict() == snapshot


@pytest.mark.parametrize("prefix", ["Real", "Complex", "Symbolic"])
@pytest.mark.parametrize("streamed", [False, True])
def test_sum_iterable_accumulates_duplicates(family_case, prefix, streamed):
    family, strings, _ = family_case
    cls = getattr(family, prefix + "TermSum")
    coefficient = {"Real": 2.0, "Complex": 1j, "Symbolic": Symbol("x")}[prefix]
    source = [(strings[0], coefficient), (strings[0], 2 * coefficient)]
    kwargs = {"qubits": strings.qubits} if family in (zqo, zqs) else {"modes": strings.modes}
    if family is zfog:
        kwargs["max_string_len"] = 64

    result = cls.from_iterable(source=iter(source) if streamed else source, **kwargs)

    assert type(result) is cls
    assert len(result) == 1
    assert result.lookup_coeff(strings[0]) == 3 * coefficient
    assert result == cls.from_terms(cls.terms_type.from_iterable(source, **kwargs))


@pytest.mark.parametrize("family", [zqo, zqs, zfs])
@pytest.mark.parametrize(("part", "coefficient"), [("real_part", 1.0), ("imag_part", 2.0)])
def test_projection_copies_existing_lookup_map(family, part, coefficient, monkeypatch):
    specs = ["X0", "Y1", "Z0"] if family is zqo else ["[1, 0]", "[0, 1]", "[0, 0]"]
    space = (
        Qubits.from_inds([5, 2]) if family in (zqo, zqs) else Modes.from_pair_count_spin_minor(1)
    )
    source = family.ComplexTermSum.from_iterable(zip(specs, [1 + 2j, 0j]), space)
    before = source.to_dict()

    def fail_collect(*args, **kwargs):
        raise AssertionError("Projection should reuse the existing lookup map.")

    monkeypatch.setattr(TermSet, "_collect_terms", fail_collect)
    projected = getattr(source, part)

    assert type(projected) is family.RealTermSum
    assert projected._string_set._map is not source._string_set._map
    assert projected.strings.is_owning()
    assert projected.coeffs.is_owning()
    assert projected.to_dict()["space"] == before["space"]
    assert projected.lookup_coeff(specs[0]) == coefficient
    assert projected.lookup_coeff(specs[1]) == 0.0
    projected.insert((specs[2], 3.0))
    assert source.lookup_coeff(specs[2]) is None
    assert source.to_dict() == before


@pytest.mark.parametrize("family", [zfog, zfon])
@pytest.mark.parametrize("prefix", ["Real", "Complex"])
@pytest.mark.parametrize("source_string", ["", "F0^, F1"])
def test_operator_domain_conversion_retains_rust_map(family, prefix, source_string, monkeypatch):
    target = zfon if family is zfog else zfog
    space = Modes.from_pair_count_spin_minor(1)
    source = getattr(family, prefix + "TermSum").from_str(source_string, space)
    expected = getattr(target, prefix + "TermSum").from_str(source_string, space)
    source *= 2j if prefix == "Complex" else 2.0
    expected *= 2j if prefix == "Complex" else 2.0

    def fail_collect(*args, **kwargs):
        raise AssertionError("Domain conversion should retain the Rust lookup map.")

    monkeypatch.setattr(TermSet, "_collect_terms", fail_collect)
    result = source.to_normal_ordered() if family is zfog else source.to_general()

    assert result == expected
    assert result.to_dict()["space"] == source.to_dict()["space"]
    assert len(result._string_set._map) == len(result)
    for term in expected:
        assert result.lookup_coeff(term.string) == term.coeff


@pytest.mark.parametrize("term_sum_type", [zqo.RealTermSum, zqo.ComplexTermSum])
def test_pauli_product_uses_prepared_empty_map(term_sum_type, monkeypatch):
    scale = 1j if term_sum_type is zqo.ComplexTermSum else 1.0
    lhs = term_sum_type.from_iterable([("X0", 2 * scale)], 2)
    rhs = term_sum_type.from_iterable([("Y0", 3 * scale)], 2)
    expected = zqo.ComplexTermSum.from_iterable([("Z0", 6j * scale * scale)], 2)

    def fail_collect(*args, **kwargs):
        raise AssertionError("Pauli multiplication should populate the supplied empty map.")

    monkeypatch.setattr(TermSet, "_collect_terms", fail_collect)
    product = lhs * rhs

    assert product == expected
    assert product.lookup_coeff("Z0") == 6j * scale * scale
