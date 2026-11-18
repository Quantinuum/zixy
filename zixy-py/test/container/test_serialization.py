import json
from types import MappingProxyType

import pytest
from sympy import sympify

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy import __version__, deserialize, serialize
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
from zixy.container.data import TermData
from zixy.container.serialization import __schema_version__
from zixy.container.terms import Term


@pytest.fixture(
    params=[
        (
            SignCoeffs,
            [Sign.from_scalar(x) for x in [1, -1, 1, -1]],
            [1, -1, 1, -1],
        ),
        (
            ComplexSignCoeffs,
            [ComplexSign.from_scalar(x) for x in [1, 1j, -1, -1j]],
            [[1, 0], [0, 1], [-1, 0], [0, -1]],
        ),
        (RealCoeffs, [1.25, -2.5, 0.0, 4.0], [1.25, -2.5, 0.0, 4.0]),
        (
            ComplexCoeffs,
            [1 + 2j, 3 - 4j, 0j, -1j],
            [[1, 2], [3, -4], [0, 0], [0, -1]],
        ),
        (
            SymbolicCoeffs,
            [sympify(x) for x in ["x", "y + 1", "2", "x**2"]],
            ["x", "y + 1", "2", "x**2"],
        ),
    ],
    ids=["sign", "complex-sign", "real", "complex", "symbolic"],
)
def coefficient_case(request):
    coeffs_type, values, encoded = request.param
    return coeffs_type.from_iterable(values), encoded


def test_coeffs_serialization_format(coefficient_case):
    coeffs, encoded = coefficient_case
    expected = {
        "schema_version": __schema_version__,
        "package_version": __version__,
        "family": "container",
        "container": type(coeffs).__name__,
        "coeffs": encoded,
    }
    assert coeffs.to_dict() == expected
    assert json.loads(coeffs.to_json()) == expected
    pretty = coeffs.to_json(indent=2)
    assert pretty.startswith('{\n  "schema_version"')
    assert json.loads(pretty) == expected


@pytest.mark.parametrize("format_", ["dict", "json"])
@pytest.mark.parametrize(
    "slices",
    [
        pytest.param((), id="owning"),
        pytest.param((slice(None),), id="full-view"),
        pytest.param((slice(1, 3),), id="offset"),
        pytest.param((slice(1, None, 2),), id="stepped"),
        pytest.param((slice(None, None, -1),), id="reversed"),
        pytest.param((slice(0, 0),), id="empty"),
        pytest.param((slice(None, None, -1), slice(1, None, 2)), id="nested"),
    ],
)
def test_coeffs_serialization_round_trip(coefficient_case, format_, slices):
    original, encoded = coefficient_case
    view = original
    for indexer in slices:
        view = view[indexer]
        encoded = encoded[indexer]

    assert view.to_dict()["coeffs"] == encoded
    if format_ == "dict":
        restored = type(original).from_dict(view.to_dict())
    else:
        restored = type(original).from_json(view.to_json())

    assert type(restored) is type(original)
    assert list(restored) == list(view)
    assert restored.is_owning()

    # Both storage directions must be independent, including when loading a view.
    if len(restored):
        before = list(original)
        restored[0] = original[1]
        assert list(original) == before
    before = list(restored)
    original[0] = original[1]
    assert list(restored) == before


def test_coeffs_serialization_snapshot(coefficient_case):
    coeffs, encoded = coefficient_case
    payload = coeffs.to_dict()
    restored = type(coeffs).from_dict(payload)

    coeffs[0] = coeffs[1]
    assert payload["coeffs"] == encoded
    payload["coeffs"].clear()
    assert len(coeffs) == len(encoded)
    assert restored.to_dict()["coeffs"] == encoded


@pytest.mark.parametrize("format_", ["dict", "json"])
def test_coeffs_serialization_schema_mismatch(coefficient_case, format_):
    coeffs, _ = coefficient_case
    payload = coeffs.to_dict()
    payload["schema_version"] = "unsupported"
    with pytest.raises(ValueError, match="schema version mismatch"):
        if format_ == "dict":
            type(coeffs).from_dict(payload)
        else:
            type(coeffs).from_json(json.dumps(payload))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("coeffs_type", [RealCoeffs, ComplexCoeffs])
def test_coeffs_serialization_nonfinite(coeffs_type, value):
    coeffs = coeffs_type.from_iterable([coeffs_type.coeff_type(value)])
    with pytest.raises(ValueError, match="Out of range float values"):
        coeffs.to_json()


def test_coeffs_deserialization_cross_type():
    signs = SignCoeffs.from_iterable([Sign.from_scalar(1), Sign.from_scalar(-1)])
    assert list(RealCoeffs.from_dict(signs.to_dict())) == [1.0, -1.0]
    reals = RealCoeffs.from_iterable([1.25, -2.5])
    assert list(ComplexCoeffs.from_json(reals.to_json())) == [1.25 + 0j, -2.5 + 0j]


@pytest.fixture(
    params=[
        (Qubits.from_count(3), [0, 1, 2]),
        (Qubits.from_offset(4, 3), [4, 5, 6]),
        (Qubits.from_inds([7, 2, 5]), [7, 2, 5]),
    ],
    ids=["default", "offset", "mapped"],
)
def qubit_space(request):
    return request.param


@pytest.mark.parametrize("container", ["String", "Strings", "StringSet"])
def test_pauli_serialization_format(qubit_space, container):
    qubits, indices = qubit_space
    rows = [["I", "X", "Y"], ["Z", "Y", "X"]]
    strings = zqo.Strings.from_iterable([[zqo.I, zqo.X, zqo.Y], [zqo.Z, zqo.Y, zqo.X]], qubits)
    if container == "String":
        source = strings[1]
        rows = rows[1:]
    elif container == "Strings":
        source = strings
    else:
        source = zqo.StringSet.from_strings(strings)

    expected = {
        "schema_version": __schema_version__,
        "package_version": __version__,
        "family": "qubit.operator",
        "container": container,
        "space": {"qubits": indices},
        "strings": rows,
    }
    assert source.to_dict() == expected
    assert json.loads(source.to_json()) == expected


@pytest.mark.parametrize("format_", ["dict", "json"])
@pytest.mark.parametrize(
    "slices",
    [
        (),
        (slice(1, None),),
        (slice(1, None, 2),),
        (slice(None, None, -1),),
        (slice(0, 0),),
        (slice(None, None, -1), slice(1, None, 2)),
    ],
)
def test_pauli_strings_serialization_round_trip(qubit_space, format_, slices):
    qubits, _ = qubit_space
    source = zqo.Strings.from_iterable(
        [[zqo.X, zqo.I, zqo.Y], [zqo.Z, zqo.X, zqo.I], [zqo.I, zqo.Y, zqo.Z]], qubits
    )
    view = source
    for indexer in slices:
        view = view[indexer]
    if format_ == "dict":
        restored = zqo.Strings.from_dict(view.to_dict())
    else:
        restored = zqo.Strings.from_json(view.to_json())
    assert restored == view
    assert restored.qubits == qubits
    assert restored.is_owning()

    before = source.clone()
    if len(restored):
        restored[0].clear()
        assert source == before
    before = restored.clone()
    source[0].clear()
    assert restored == before


def test_pauli_string_serialization_snapshot(qubit_space):
    qubits, _ = qubit_space
    source = zqo.Strings.from_iterable([[zqo.X, zqo.Y, zqo.Z], [zqo.Z, zqo.X, zqo.Y]], qubits)
    view = source[1]
    payload = view.to_dict()
    restored = zqo.String.from_dict(payload)
    assert restored == view
    assert restored.qubits == qubits
    assert restored.is_owning()

    source[1].clear()
    assert payload["strings"] == [["Z", "X", "Y"]]
    payload["strings"][0][0] = "I"
    assert restored.get_tuple() == (zqo.Z, zqo.X, zqo.Y)
    restored.clear()
    assert source[0].get_tuple() == (zqo.X, zqo.Y, zqo.Z)


def test_pauli_set_serialization_rebuilds_map(qubit_space):
    qubits, _ = qubit_space
    strings = zqo.Strings.from_iterable([[zqo.X], [zqo.Y], [zqo.X]], qubits)
    source = zqo.StringSet.from_strings(strings)
    restored = zqo.StringSet.from_json(source.to_json())
    assert restored == source
    assert restored.to_strings().qubits == qubits
    assert restored.insert([zqo.X]) == 0
    assert len(restored) == 2
    restored.remove([zqo.Y])
    assert len(restored) == 1
    assert len(source) == 2
    assert restored.insert([zqo.Z]) == 1


@pytest.mark.parametrize("container_type", [zqo.String, zqo.Strings, zqo.StringSet])
@pytest.mark.parametrize("n_qubits", [0, 3])
def test_pauli_serialization_empty(container_type, n_qubits):
    # Construct a zero-qubit identity through the array, avoiding the text parser.
    source = (
        zqo.Strings(n_qubits, size=1)[0].clone()
        if container_type is zqo.String
        else container_type(n_qubits)
    )
    restored = container_type.from_json(source.to_json())
    assert restored == source
    space = restored.qubits if container_type is not zqo.StringSet else restored.to_strings().qubits
    assert len(space) == n_qubits


@pytest.mark.parametrize("n_strings", [0, 2])
def test_pauli_string_deserialization_requires_one_string(n_strings):
    payload = zqo.Strings(3, size=n_strings).to_dict()
    with pytest.raises(ValueError, match="exactly one string"):
        zqo.String.from_dict(payload)


@pytest.mark.parametrize("container_type", [zqo.String, zqo.Strings, zqo.StringSet])
def test_pauli_serialization_schema_mismatch(container_type):
    payload = container_type(3).to_dict()
    payload["schema_version"] = "unsupported"
    with pytest.raises(ValueError, match="schema version mismatch"):
        container_type.from_dict(payload)


@pytest.mark.parametrize("row", [["X"], ["X", "Y", "invalid"]])
def test_pauli_serialization_invalid_row(row):
    payload = zqo.Strings(3).to_dict()
    payload["strings"] = [row]
    with pytest.raises(ValueError):
        zqo.Strings.from_dict(payload)


@pytest.fixture(params=[zqo, zqs, zfs, zfon, zfog], ids=lambda family: family.__name__)
def string_family(request):
    return request.param


@pytest.fixture(params=["count", "labelled", "alternate", "empty"])
def string_case(request, string_family):
    return _make_string_case(string_family, request.param)


def _make_string_case(family, space_kind):
    if family in (zqo, zqs):
        space, indices = {
            "count": (Qubits.from_count(4), [0, 1, 2, 3]),
            "labelled": (Qubits.from_offset(4, 4), [4, 5, 6, 7]),
            "alternate": (Qubits.from_inds([7, 2, 8, 3]), [7, 2, 8, 3]),
            "empty": (Qubits.from_count(0), []),
        }[space_kind]
        encoded_space = {"qubits": indices}
    else:
        space, ordering = {
            "count": (Modes.from_count(4), None),
            "labelled": (Modes.from_pair_count_spin_major(2), "major"),
            "alternate": (Modes.from_pair_count_spin_minor(2), "minor"),
            "empty": (Modes.from_count(0), None),
        }[space_kind]
        encoded_space = {"modes": len(space), "spin_ordering": ordering}

    if family is zqo:
        rows = [[zqo.I, zqo.X, zqo.Y, zqo.Z], [zqo.X] * 4, [zqo.Z] * 4]
        encoded = [["I", "X", "Y", "Z"], ["X"] * 4, ["Z"] * 4]
    elif family in (zqs, zfs):
        rows = [[True, False, False, True], [False, True, False, False], [True] * 4]
        encoded = rows
    elif family is zfon:
        rows = [([0], [1]), ([2], [3]), ([1], [2])]
        encoded = [[[0], [1]], [[2], [3]], [[1], [2]]]
    else:
        # Raw products must preserve both the order and repeated operators.
        rows = [
            [(0, True), (1, False), (0, True)],
            [(3, False), (1, True)],
            [(0, False), (2, True), (0, False)],
        ]
        encoded = [[[mode, creation] for mode, creation in row] for row in rows]

    if len(space) == 0:
        strings = family.Strings(space, size=3)
        encoded = [[[], []] if family is zfon else [] for _ in range(3)]
    else:
        strings = family.Strings.from_iterable(rows, space)
    return strings, encoded, encoded_space


@pytest.mark.parametrize("container", ["String", "Strings", "StringSet"])
@pytest.mark.parametrize("format_", ["dict", "json"])
def test_string_family_serialization(string_family, string_case, container, format_):
    strings, encoded, space = string_case
    if container == "String":
        source = strings[1]
        encoded = encoded[1:2]
    elif container == "StringSet":
        source = string_family.StringSet.from_strings(strings)
        if len(source) == 1:
            encoded = encoded[:1]
    else:
        source = strings
    expected = {
        "schema_version": __schema_version__,
        "package_version": __version__,
        "family": string_family.__name__.removeprefix("zixy."),
        "container": container,
        "space": space,
        "strings": encoded,
    }
    assert source.to_dict() == expected
    cls = getattr(string_family, container)
    if format_ == "dict":
        restored = cls.from_dict(expected)
    else:
        restored = cls.from_json(source.to_json())
    assert type(restored) is cls
    assert restored == source
    assert restored.to_dict() == expected
    if container != "StringSet":
        assert restored.is_owning()


@pytest.mark.parametrize(
    "slices",
    [
        (slice(1, None),),
        (slice(1, None, 2),),
        (slice(None, None, -1),),
        (slice(0, 0),),
        (slice(None, None, -1), slice(1, None, 2)),
    ],
)
def test_string_family_serialization_views(string_case, slices):
    source, encoded, space = string_case
    view = source
    for indexer in slices:
        view = view[indexer]
        encoded = encoded[indexer]
    payload = view.to_dict()
    assert payload["strings"] == encoded
    assert payload["space"] == space
    restored = type(source).from_json(view.to_json())
    assert restored == view
    assert restored.is_owning()
    assert restored.to_dict()["space"] == space

    before = source.clone()
    # Zero-mode identities have no content to mutate.
    if space.get("modes", len(space.get("qubits", []))):
        for string in restored:
            string.clear()
    assert source == before
    before = restored.clone()
    source.resize(0)
    assert restored == before
    assert payload["strings"] == encoded
    payload["strings"].clear()
    assert restored == before


def test_string_family_set_rebuilds_map(string_family, string_case):
    strings, _, _ = string_case
    restored = string_family.StringSet.from_json(strings.to_json())
    first = strings[0].clone()
    size = len(restored)
    restored.insert(first)
    assert len(restored) == size
    restored.remove(first)
    assert len(restored) == size - 1
    restored.insert(first)
    assert len(restored) == size
    assert len(strings) == 3


@pytest.mark.parametrize("container", ["String", "Strings", "StringSet"])
def test_string_family_schema_mismatch(string_family, string_case, container):
    strings, _, _ = string_case
    payload = strings.to_dict()
    payload["schema_version"] = "unsupported"
    with pytest.raises(ValueError, match="schema version mismatch"):
        getattr(string_family, container).from_dict(payload)


@pytest.mark.parametrize("n_strings", [0, 2])
def test_string_family_requires_one_string(string_family, string_case, n_strings):
    strings, _, _ = string_case
    payload = strings.to_dict()
    payload["strings"] = payload["strings"][:n_strings]
    with pytest.raises(ValueError, match="exactly one string"):
        string_family.String.from_dict(payload)


@pytest.mark.parametrize("family", [zfs, zfon, zfog])
@pytest.mark.parametrize("ordering", ["major", "minor"])
def test_fermion_serialization_empty_spin_space(family, ordering):
    modes = (
        Modes.from_pair_count_spin_major(0)
        if ordering == "major"
        else Modes.from_pair_count_spin_minor(0)
    )
    source = family.Strings(modes)
    restored = family.Strings.from_json(source.to_json())
    assert restored.modes == modes
    assert restored.to_dict()["space"] == {"modes": 0, "spin_ordering": ordering}


@pytest.mark.parametrize(
    "space",
    [
        {"modes": -1, "spin_ordering": None},
        {"modes": True, "spin_ordering": None},
        {"modes": 2.5, "spin_ordering": None},
        {"modes": 3, "spin_ordering": "major"},
        {"modes": 4, "spin_ordering": "unknown"},
    ],
)
def test_fermion_serialization_invalid_space(space):
    payload = zfs.Strings(4).to_dict()
    payload["space"] = space
    with pytest.raises(ValueError):
        zfs.Strings.from_dict(payload)


_COEFFICIENT_FAMILIES = [
    (family, prefix)
    for family in (zqo, zqs, zfs, zfon, zfog)
    for prefix in ("Sign", "ComplexSign", "Real", "Complex", "Symbolic")
    if hasattr(family, f"{prefix}Terms")
]


def _make_term_case(family, prefix, container, space_kind):
    strings, encoded_strings, space = _make_string_case(family, space_kind)
    values, encoded_coeffs = {
        "Sign": ([Sign.from_scalar(x) for x in [1, -1, 1]], [1, -1, 1]),
        "ComplexSign": (
            [ComplexSign.from_scalar(x) for x in [1, 1j, -1j]],
            [[1, 0], [0, 1], [0, -1]],
        ),
        "Real": ([1.25, 0.0, -2.5], [1.25, 0.0, -2.5]),
        "Complex": ([1 + 2j, 0j, -3 + 4j], [[1, 2], [0, 0], [-3, 4]]),
        "Symbolic": ([sympify(x) for x in ["x", "0", "y + 1"]], ["x", "0", "y + 1"]),
    }[prefix]
    terms_type = getattr(family, f"{prefix}Terms")
    coeffs = terms_type.term_type.coeffs_type.from_iterable(values)
    terms = terms_type._create(TermData(strings, coeffs))
    if container == "Term":
        source = terms[1]
        encoded_strings = encoded_strings[1:2]
        encoded_coeffs = encoded_coeffs[1:2]
    elif container == "Terms":
        source = terms
    else:
        source = getattr(family, f"{prefix}{container}").from_terms(terms)
        if len(source) == 1:
            encoded_strings = encoded_strings[:1]
            if container == "TermSum":
                encoded_coeffs = {
                    "Real": [-1.25],
                    "Complex": [[-2, 6]],
                    "Symbolic": ["x + y + 1"],
                }[prefix]
            else:
                encoded_coeffs = encoded_coeffs[-1:]
    expected = {
        "schema_version": __schema_version__,
        "package_version": __version__,
        "family": family.__name__.removeprefix("zixy."),
        "container": type(source).__name__,
        "space": space,
        "strings": encoded_strings,
        "coeffs": encoded_coeffs,
    }
    return source, expected


@pytest.fixture(params=["count", "labelled", "alternate", "empty"])
def term_space_kind(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param((family, prefix, container), id=f"{family.__name__}.{prefix}{container}")
        for family, prefix in _COEFFICIENT_FAMILIES
        for container in ("Term", "Terms", "TermSet", "TermSum")
        if hasattr(family, f"{prefix}{container}")
    ]
)
def term_case(request, term_space_kind):
    return _make_term_case(*request.param, term_space_kind)


@pytest.mark.parametrize("format_", ["dict", "json"])
def test_term_serialization_round_trip(term_case, format_):
    source, expected = term_case
    assert source.to_dict() == expected
    assert json.loads(source.to_json()) == expected
    cls = type(source)
    if format_ == "dict":
        restored = cls.from_dict(source.to_dict())
    else:
        restored = cls.from_json(source.to_json())
    assert type(restored) is cls
    assert restored.to_dict() == expected
    assert type(restored._data.strings) is type(source._data.strings)
    assert type(restored._data.coeffs) is type(source._data.coeffs)
    assert restored._data.strings.is_owning()
    assert restored._data.coeffs.is_owning()
    if hasattr(restored, "is_owning"):
        assert restored.is_owning()

    # Coefficients are mutable in all four container kinds, including sets and sums.
    one = unit(source.coeff_type)
    source_index = source.index if isinstance(source, Term) else 0
    replacement = -one if source._data.coeffs[source_index] == one else one
    before = source.to_dict()
    restored._data.coeffs[0] = replacement
    assert source.to_dict() == before
    before = restored.to_dict()
    source._data.coeffs[source_index] = replacement
    assert restored.to_dict() == before


def test_term_serialization_snapshot(term_case):
    source, expected = term_case
    payload = source.to_dict()
    restored = type(source).from_dict(payload)
    before = restored.to_dict()
    payload["strings"].clear()
    payload["coeffs"].clear()
    payload["space"].clear()
    assert source.to_dict() == expected
    assert restored.to_dict() == before


@pytest.mark.parametrize(
    "family,prefix",
    [
        pytest.param(family, prefix, id=f"{family.__name__}.{prefix}")
        for family, prefix in _COEFFICIENT_FAMILIES
    ],
)
@pytest.mark.parametrize(
    "slices",
    [
        (slice(1, None),),
        (slice(1, None, 2),),
        (slice(None, None, -1),),
        (slice(0, 0),),
        (slice(None, None, -1), slice(1, None, 2)),
    ],
)
def test_terms_serialization_views(family, prefix, term_space_kind, slices):
    source, expected = _make_term_case(family, prefix, "Terms", term_space_kind)
    view = source
    for indexer in slices:
        view = view[indexer]
        expected["strings"] = expected["strings"][indexer]
        expected["coeffs"] = expected["coeffs"][indexer]
    assert view.to_dict() == expected
    restored = type(source).from_json(view.to_json())
    assert restored.to_dict() == expected
    assert restored.is_owning()
    if len(view):
        term = view[0]
        term_expected = dict(expected)
        term_expected["container"] = type(term).__name__
        term_expected["strings"] = expected["strings"][:1]
        term_expected["coeffs"] = expected["coeffs"][:1]
        assert term.to_dict() == term_expected
        single = type(term).from_json(term.to_json())
        assert single.to_dict() == term_expected
        assert single.is_owning()
    source.resize(0)
    assert restored.to_dict() == expected


def test_term_serialization_schema_mismatch(term_case):
    source, _ = term_case
    payload = source.to_dict()
    payload["schema_version"] = "unsupported"
    with pytest.raises(ValueError, match="schema version mismatch"):
        type(source).from_dict(payload)


def test_term_serialization_mismatched_lengths(term_case):
    source, _ = term_case
    payload = source.to_dict()
    payload["coeffs"].pop()
    with pytest.raises(ValueError, match="equal lengths"):
        type(source).from_dict(payload)


@pytest.mark.parametrize("n_terms", [0, 2])
def test_single_term_deserialization_requires_one_term(string_family, string_case, n_terms):
    strings, _, _ = string_case
    cls = string_family.RealTerm
    payload = strings.to_dict()
    payload["strings"] = payload["strings"][:n_terms]
    payload["coeffs"] = [1.0] * n_terms
    with pytest.raises(ValueError, match="exactly one term"):
        cls.from_dict(payload)


@pytest.mark.parametrize(
    "family,prefix,container",
    [
        pytest.param(family, prefix, container, id=f"{family.__name__}.{prefix}{container}")
        for family, prefix in _COEFFICIENT_FAMILIES
        for container in ("TermSet", "TermSum")
        if hasattr(family, f"{prefix}{container}")
    ],
)
def test_term_serialization_rebuilds_map(family, prefix, container, term_space_kind):
    source, expected = _make_term_case(family, prefix, container, term_space_kind)
    restored = type(source).from_json(source.to_json())
    key = source.strings[0].clone()
    coeff = source.coeffs[0]
    size = len(source)
    assert restored.lookup_index(key) == 0
    assert restored.lookup_coeff(key) == coeff

    replacement = -unit(source.coeff_type)
    restored.insert((key, replacement))
    assert restored.lookup_coeff(key) == replacement
    assert len(restored) == size
    restored.remove(key)
    assert restored.lookup_index(key) is None
    assert len(restored) == size - 1
    restored.insert((key, coeff))
    assert restored.lookup_coeff(key) == coeff
    assert len(restored) == size
    assert source.to_dict() == expected


@pytest.mark.parametrize("family", [zqo, zqs, zfs, zfon, zfog])
@pytest.mark.parametrize("prefix", ["Real", "Complex", "Symbolic"])
@pytest.mark.parametrize("container", ["TermSet", "TermSum"])
@pytest.mark.parametrize("format_", ["dict", "json", "dispatch"])
def test_term_deserialization_duplicate_strings(family, prefix, container, format_):
    terms, _ = _make_term_case(family, prefix, "Terms", "count")
    terms.strings[1] = terms.strings[0]
    terms.coeffs[1] = terms.coeffs[2]
    cls = getattr(family, f"{prefix}{container}")
    payload = terms.to_dict()
    payload["container"] = cls.__name__
    before = terms.to_dict()

    if format_ == "dict":
        restored = cls.from_dict(payload)
    elif format_ == "json":
        restored = cls.from_json(json.dumps(payload))
    else:
        restored = deserialize(json.dumps(payload))

    expected = terms.coeffs[1]
    if container == "TermSum":
        expected += terms.coeffs[0]
    assert type(restored) is cls
    assert len(restored) == 2
    assert restored.lookup_coeff(terms.strings[0]) == expected
    assert restored.lookup_coeff(terms.strings[2]) == terms.coeffs[2]
    assert terms.to_dict() == before


def test_promoted_sign_product_serialization():
    product = zfon.SignTerm(2, "F0") * zfon.SignTerm(2, "F0^")

    payload = product.to_dict()
    assert payload["container"] == "RealTermSum"
    assert payload["coeffs"] == [1.0, -1.0]
    restored = deserialize(serialize(product))
    assert type(restored) is zfon.RealTermSum
    assert restored == product


def test_coefficient_serialization_dispatch(coefficient_case):
    source, _ = coefficient_case
    encoded = serialize(source)
    assert json.loads(encoded) == source.to_dict()
    restored = deserialize(encoded)
    assert type(restored) is type(source)
    assert restored.to_dict() == source.to_dict()
    assert restored.is_owning()


@pytest.mark.parametrize("container", ["String", "Strings", "StringSet"])
def test_string_serialization_dispatch(string_family, string_case, container):
    strings, _, _ = string_case
    if container == "String":
        source = strings[1]
    elif container == "StringSet":
        source = string_family.StringSet.from_strings(strings)
    else:
        source = strings
    encoded = serialize(source)
    assert json.loads(encoded) == source.to_dict()
    restored = deserialize(encoded)
    assert type(restored) is type(source)
    assert restored.to_dict() == source.to_dict()


def test_term_serialization_dispatch(term_case):
    source, expected = term_case
    encoded = serialize(source)
    assert json.loads(encoded) == expected
    restored = deserialize(encoded)
    assert type(restored) is type(source)
    assert restored.to_dict() == expected


def test_serialize_json_options():
    source = RealCoeffs.from_iterable([1.25, -2.5])
    encoded = serialize(source, indent=2, sort_keys=True)
    assert encoded.startswith('{\n  "coeffs"')
    assert json.loads(encoded) == source.to_dict()


@pytest.mark.parametrize("source", [None, 1, {}, [1], Sign.from_scalar(1)])
def test_serialize_unsupported_object(source):
    with pytest.raises(TypeError, match="serializable Zixy container"):
        serialize(source)


@pytest.mark.parametrize("mapping_type", [dict, MappingProxyType])
def test_deserialize_rejects_mapping(mapping_type):
    source = RealCoeffs.from_iterable([1.25, -2.5])
    payload = source.to_dict()
    with pytest.raises(TypeError, match="JSON string"):
        deserialize(mapping_type(payload))


@pytest.mark.parametrize("source", [None, 1, [], {}, b"{}", bytearray(b"{}")])
def test_deserialize_requires_string(source):
    with pytest.raises(TypeError, match="JSON string"):
        deserialize(source)


@pytest.mark.parametrize("source", ["null", "[]", "1", "true", '"text"'])
def test_deserialize_requires_object(source):
    with pytest.raises(TypeError, match="must contain an object"):
        deserialize(source)


def test_deserialize_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        deserialize("{")


@pytest.mark.parametrize("field", ["schema_version", "package_version", "family", "container"])
def test_deserialize_missing_metadata(field):
    payload = RealCoeffs.from_iterable([1.0]).to_dict()
    del payload[field]
    with pytest.raises(ValueError, match="Missing serialization metadata"):
        deserialize(json.dumps(payload))


@pytest.mark.parametrize("field", ["schema_version", "package_version", "family", "container"])
def test_deserialize_invalid_metadata_type(field):
    payload = RealCoeffs.from_iterable([1.0]).to_dict()
    payload[field] = None
    with pytest.raises(TypeError, match="must be a string"):
        deserialize(json.dumps(payload))


def test_deserialize_schema_mismatch():
    payload = RealCoeffs.from_iterable([1.0]).to_dict()
    payload["schema_version"] = "unsupported"
    with pytest.raises(ValueError, match="schema version mismatch"):
        deserialize(json.dumps(payload))


def test_deserialize_other_package_version():
    source = RealCoeffs.from_iterable([1.0])
    payload = source.to_dict()
    payload["package_version"] = "0.0.0"
    assert deserialize(json.dumps(payload)).to_dict() == source.to_dict()


@pytest.mark.parametrize("family", ["unknown", "qubit", "os", "container.coeffs"])
def test_deserialize_unknown_family(family):
    payload = RealCoeffs.from_iterable([1.0]).to_dict()
    payload["family"] = family
    with pytest.raises(ValueError, match="Unknown serialization family"):
        deserialize(json.dumps(payload))


@pytest.mark.parametrize("container", ["Unknown", "Sign", "Coeffs", "convert_vec", "RealTerm"])
def test_deserialize_unknown_container(container):
    payload = RealCoeffs.from_iterable([1.0]).to_dict()
    payload["container"] = container
    with pytest.raises(ValueError, match="Unknown serializable container"):
        deserialize(json.dumps(payload))


@pytest.mark.parametrize("field,value", [("strings", {}), ("coeffs", {}), ("space", [])])
def test_deserialize_invalid_payload_type(field, value):
    payload = zqo.RealTerm.from_str("(1.0, X0)", 2).to_dict()
    payload[field] = value
    with pytest.raises(TypeError):
        deserialize(json.dumps(payload))


@pytest.mark.parametrize("field", ["strings", "coeffs", "space"])
def test_deserialize_missing_payload(field):
    payload = zqo.RealTerm.from_str("(1.0, X0)", 2).to_dict()
    del payload[field]
    with pytest.raises(KeyError):
        deserialize(json.dumps(payload))


def test_deserialize_mismatched_lengths():
    payload = zqo.RealTerms.from_str("(1.0, X0), (2.0, Y0)", 2).to_dict()
    payload["coeffs"].pop()
    with pytest.raises(ValueError, match="equal lengths"):
        deserialize(json.dumps(payload))
