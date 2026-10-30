import json

import pytest
from sympy import sympify

from zixy import __version__
from zixy.container.base import __schema_version__
from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSign,
    ComplexSignCoeffs,
    RealCoeffs,
    Sign,
    SignCoeffs,
    SymbolicCoeffs,
)


@pytest.fixture(
    params=[
        (
            SignCoeffs,
            [Sign.from_int(x) for x in [1, -1, 1, -1]],
            [1, -1, 1, -1],
        ),
        (
            ComplexSignCoeffs,
            [ComplexSign.from_complex(x) for x in [1, 1j, -1, -1j]],
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
    return coeffs_type.from_sequence(values), encoded


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
    coeffs = coeffs_type.from_sequence([coeffs_type.coeff_type(value)])
    with pytest.raises(ValueError, match="Out of range float values"):
        coeffs.to_json()


def test_coeffs_deserialization_cross_type():
    signs = SignCoeffs.from_sequence([Sign.from_int(1), Sign.from_int(-1)])
    assert list(RealCoeffs.from_dict(signs.to_dict())) == [1.0, -1.0]
    reals = RealCoeffs.from_sequence([1.25, -2.5])
    assert list(ComplexCoeffs.from_json(reals.to_json())) == [1.25 + 0j, -2.5 + 0j]
