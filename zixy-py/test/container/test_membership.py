"""String-keyed lookup and complete-term membership use the same lookup map."""

import pytest
from sympy import sympify

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy._zixy import Modes, Qubits
from zixy.container.coeffs import ComplexSign, Sign


@pytest.fixture(params=[zqo, zqs, zfog, zfon, zfs], ids=lambda family: family.__name__)
def family_strings(request):
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
    strings = family.Strings.from_iterable(specs, space)
    return family, space, specs, strings


def test_string_set_membership(family_strings):
    family, space, specs, strings = family_strings
    collection = family.StringSet.from_strings(strings)
    before = collection.to_dict()

    for spec, string in zip(specs, strings, strict=True):
        assert spec in collection
        assert string in collection
        assert string.clone() in collection
    assert all(string in collection for string in collection)
    assert family.String(space) not in collection
    assert family.String(3) not in collection
    for unsupported in (None, object(), 42, family.RealTerm.from_string(strings[0], 2.0)):
        assert unsupported not in collection
    assert collection.to_dict() == before
    assert not hasattr(collection, "contains")


def test_membership_rejects_out_of_range_specifiers(family_strings):
    family, space, _, strings = family_strings
    invalid = {
        zqo: "X2",
        zqs: {2},
        zfog: [(2, True)],
        zfon: ([2], []),
        zfs: {2},
    }[family]
    kwargs = {"max_string_len": strings.max_string_len} if family is zfog else {}
    terms = family.RealTerms.from_iterable(zip(strings, [2.0, 3.0]), space, **kwargs)
    collections = [
        family.StringSet.from_strings(strings),
        family.RealTermSet.from_terms(terms),
        family.RealTermSum.from_terms(terms),
    ]
    for collection in collections:
        before = collection.to_dict()
        assert invalid not in collection
        assert strings[0] in collection
        with pytest.raises((IndexError, ValueError)):
            collection.lookup_index(invalid)
        assert collection.to_dict() == before


def test_membership_rejects_invalidated_views(family_strings):
    family, space, _, strings = family_strings
    kwargs = {"max_string_len": strings.max_string_len} if family is zfog else {}
    terms = family.RealTerms.from_iterable(zip(strings, [2.0, 3.0]), space, **kwargs)
    string_view, term_view = strings[-1], terms[-1]
    collections = [
        family.StringSet.from_strings(strings),
        family.RealTermSet.from_terms(terms),
        family.RealTermSum.from_terms(terms),
    ]
    strings.resize(0)
    terms.resize(0)
    for collection in collections:
        before = collection.to_dict()
        assert string_view not in collection
        if not isinstance(collection, family.StringSet):
            assert term_view not in collection
        assert collection.to_dict() == before


def test_string_set_lookup_index(family_strings):
    family, space, specs, strings = family_strings
    collection = family.StringSet.from_strings(strings)
    for index, (spec, string) in enumerate(zip(specs, strings, strict=True)):
        assert collection.lookup_index(key=spec) == index
        assert collection.lookup_index(string.clone()) == index
    assert collection.lookup_index(family.String(space)) is None
    with pytest.raises(TypeError):
        collection.lookup_index(object())
    with pytest.raises(ValueError):
        collection.lookup_index(family.String(3))
    assert not hasattr(collection, "lookup")


@pytest.mark.parametrize("suffix", ["TermSet", "TermSum"])
@pytest.mark.parametrize(
    ("prefix", "coefficient", "different"),
    [("Real", 2.0, 2.0 + 1e-9), ("Complex", 2j, -2j), ("Symbolic", sympify("x"), sympify("y"))],
)
def test_term_membership_checks_coefficient(family_strings, suffix, prefix, coefficient, different):
    family, space, specs, strings = family_strings
    collection_type = getattr(family, prefix + suffix)
    term_type = collection_type.terms_type.term_type
    term = term_type.from_string(strings[0], coefficient)
    collection = collection_type.from_terms(collection_type.terms_type.from_iterable([term], space))
    before = collection.to_dict()

    assert term in collection
    assert term.clone() in collection
    assert specs[0] in collection
    assert term.string in collection
    assert all(value in collection for value in collection)
    changed = term_type.from_string(strings[0], different)
    assert changed not in collection
    assert changed.string in collection
    assert term_type.from_string(strings[1], coefficient) not in collection
    assert term_type.from_string(family.String(3), coefficient) not in collection
    assert family.String(3) not in collection
    for unsupported in (None, object(), 42):
        assert unsupported not in collection
    assert collection.to_dict() == before
    assert not hasattr(collection, "contains")


@pytest.mark.parametrize("suffix", ["TermSet", "TermSum"])
@pytest.mark.parametrize("prefix", ["Real", "Complex", "Symbolic"])
def test_membership_distinguishes_stored_zero_from_missing(family_strings, suffix, prefix):
    family, space, specs, strings = family_strings
    collection_type = getattr(family, prefix + suffix)
    term_type = collection_type.terms_type.term_type
    kwargs = {"max_string_len": strings.max_string_len} if family is zfog else {}
    collection = collection_type(space, **kwargs)
    coefficient = sympify(0) if prefix == "Symbolic" else 0.0
    stored = term_type.from_string(strings[0], coefficient)
    missing = term_type.from_string(strings[1], coefficient)

    assert stored not in collection
    collection.insert(stored)
    assert stored in collection
    assert stored.string in collection
    assert specs[0] in collection
    assert missing not in collection
    assert missing.string not in collection
    assert specs[1] not in collection


@pytest.mark.parametrize(
    ("family", "prefix", "coefficient", "different"),
    [
        (zqo, "Sign", Sign.from_scalar(1), Sign.from_scalar(-1)),
        (zqs, "Sign", Sign.from_scalar(1), Sign.from_scalar(-1)),
        (zfon, "Sign", Sign.from_scalar(1), Sign.from_scalar(-1)),
        (zfs, "Sign", Sign.from_scalar(1), Sign.from_scalar(-1)),
        (zqo, "ComplexSign", ComplexSign.from_scalar(1j), ComplexSign.from_scalar(-1j)),
        (zqs, "ComplexSign", ComplexSign.from_scalar(1j), ComplexSign.from_scalar(-1j)),
    ],
)
def test_root_of_unity_membership(family, prefix, coefficient, different):
    collection_type = getattr(family, prefix + "TermSet")
    term_type = collection_type.terms_type.term_type
    term = term_type.from_string(family.String(2), coefficient)
    collection = collection_type(2)
    collection.insert(term)

    assert term in collection
    assert next(iter(collection)) in collection
    assert term_type.from_string(term.string, different) not in collection
    assert term_type.from_string(term.string, different).string in collection
    assert collection.get(term.string) == coefficient
    assert collection.get(term_type.from_string(term.string, different)) == coefficient
    assert collection.lookup_index(term) == 0
    missing_spec = "X0" if family is zqo else "F0" if family is zfon else "[1, 0]"
    missing = family.String(2, missing_spec)
    marker = object()
    assert collection.get(missing) is None
    assert collection.get(missing, 0.0) == 0.0
    assert collection.get(missing, marker) is marker
    with pytest.raises(KeyError):
        _ = collection[missing]


@pytest.mark.parametrize("suffix", ["TermSet", "TermSum"])
@pytest.mark.parametrize(
    ("prefix", "coefficient", "different"),
    [("Real", 2.0, 3.0), ("Complex", 2j, -2j), ("Symbolic", sympify("x"), sympify("y"))],
)
def test_term_get_and_index_lookup(family_strings, suffix, prefix, coefficient, different):
    family, space, specs, strings = family_strings
    collection_type = getattr(family, prefix + suffix)
    term_type = collection_type.terms_type.term_type
    term = term_type.from_string(strings[0], coefficient)
    collection = collection_type.from_terms(collection_type.terms_type.from_iterable([term], space))
    marker = object()
    before = collection.to_dict()

    for key in (specs[0], strings[0], term, term_type.from_string(strings[0], different)):
        assert collection.lookup_index(key=key) == 0
        assert collection.get(key=key) == coefficient
        assert collection.get(key, default=marker) == coefficient
        assert collection[key] == coefficient
    for missing in (specs[1], strings[1], term_type.from_string(strings[1], coefficient)):
        assert collection.lookup_index(missing) is None
        assert collection.get(missing) is None
        assert collection.get(missing, None) is None
        assert collection.get(missing, default=marker) is marker
        assert collection.get(missing, "absent") == "absent"
        assert collection.get(missing, 0.0) == 0.0
        with pytest.raises(KeyError):
            _ = collection[missing]
    with pytest.raises(TypeError):
        collection.get(object(), marker)
    with pytest.raises(ValueError):
        collection.get(family.String(3), marker)
    assert collection.to_dict() == before
    assert not hasattr(collection, "lookup")
    assert not hasattr(collection, "lookup_coeff")


@pytest.mark.parametrize("suffix", ["TermSet", "TermSum"])
@pytest.mark.parametrize("prefix", ["Real", "Complex", "Symbolic"])
def test_get_preserves_stored_zero(family_strings, suffix, prefix):
    family, space, _, strings = family_strings
    collection_type = getattr(family, prefix + suffix)
    term_type = collection_type.terms_type.term_type
    kwargs = {"max_string_len": strings.max_string_len} if family is zfog else {}
    collection = collection_type(space, **kwargs)
    coefficient = sympify(0) if prefix == "Symbolic" else 0.0
    collection.insert(term_type.from_string(strings[0], coefficient))
    marker = object()

    assert collection.lookup_index(strings[0]) == 0
    assert collection.get(strings[0], marker) == coefficient
    assert collection[strings[0]] == coefficient
    assert collection.get(strings[1], marker) is marker


def test_membership_rejects_other_families(family_strings):
    family, space, _, strings = family_strings
    other_family = zqs if family is zqo else zqo
    foreign_string = other_family.String(2)
    foreign_term = other_family.RealTerm.from_string(foreign_string, 2.0)
    collections = [
        family.StringSet.from_strings(strings),
        family.RealTermSet(space),
        family.RealTermSum(space),
    ]
    for collection in collections:
        before = collection.to_dict()
        assert foreign_string not in collection
        assert foreign_term not in collection
        assert collection.to_dict() == before


def test_membership_checks_space_metadata(family_strings):
    family, space, _, _ = family_strings
    other_space = Qubits.from_count(2) if family in (zqo, zqs) else Modes.from_count(2)
    assert len(space) == len(other_space)
    assert space != other_space
    string = family.String(space)
    foreign_string = family.String(other_space)
    term = family.RealTerm.from_string(string, 2.0)
    foreign_term = family.RealTerm.from_string(foreign_string, 2.0)
    strings = family.StringSet(space)
    strings.insert(string)
    assert string in strings
    assert foreign_string not in strings
    with pytest.raises(ValueError):
        strings.lookup_index(foreign_string)
    for collection_type in (family.RealTermSet, family.RealTermSum):
        collection = collection_type(space)
        collection.insert(term)
        assert term in collection
        assert foreign_string not in collection
        assert foreign_term not in collection
        with pytest.raises(ValueError):
            collection.lookup_index(foreign_term)
        with pytest.raises(ValueError):
            collection.get(foreign_term, object())


def test_membership_does_not_convert_coefficients():
    strings = zqo.Strings.from_iterable(["X0"], 2)
    collection = zqo.RealTermSet.from_iterable([("X0", 2.0)], 2)
    equal = zqo.ComplexTerm.from_string(strings[0], 2 + 0j)
    different = zqo.ComplexTerm.from_string(strings[0], 2 + 1j)
    stored = next(iter(collection))

    assert equal == stored
    assert equal in collection
    assert different != stored
    assert different not in collection
    assert different.string in collection
    assert collection.lookup_index(different) == 0
    assert collection.get(different) == 2.0
    assert collection[different] == 2.0


@pytest.mark.parametrize("operation", ["get", "getitem", "lookup_index"])
def test_coefficient_lookup_uses_one_map_lookup(operation, monkeypatch):
    collection = zqo.RealTermSet.from_iterable([("X0", 2.0)], 2)
    lookup_index = zqo.StringSet.lookup_index
    lookups = 0

    def count_lookup(self, key):
        nonlocal lookups
        lookups += 1
        return lookup_index(self, key)

    monkeypatch.setattr(zqo.StringSet, "lookup_index", count_lookup)
    if operation == "get":
        assert collection.get("X0") == 2.0
    elif operation == "getitem":
        assert collection["X0"] == 2.0
    else:
        assert collection.lookup_index("X0") == 0
    assert lookups == 1


@pytest.mark.parametrize("collection_type", [zqo.StringSet, zqo.RealTermSet, zqo.RealTermSum])
def test_membership_uses_lookup_without_iteration(collection_type, monkeypatch):
    string = zqo.String(2, "X0")
    term = zqo.RealTerm.from_string(string, 2.0)
    collection = collection_type(2)
    collection.insert(string if collection_type is zqo.StringSet else term)

    def reject_iteration(self):
        raise AssertionError("Membership must use the lookup map rather than scan the collection.")

    monkeypatch.setattr(collection_type, "__iter__", reject_iteration)
    assert string in collection
    assert "X0" in collection
    if collection_type is not zqo.StringSet:
        assert term in collection
        assert zqo.RealTerm.from_string(string, 3.0) not in collection
