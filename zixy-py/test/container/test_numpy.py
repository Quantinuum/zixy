"""Coefficient arrays expose independent NumPy copies of their visible elements."""

import warnings

import numpy as np
import pytest
from sympy import sympify

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSign,
    ComplexSignCoeffs,
    RealCoeffs,
    Sign,
    SignCoeffs,
    SymbolicCoeffs,
    convert,
)


@pytest.fixture(
    params=[
        (RealCoeffs, [1.0, -2.0, 3.0, -4.0], [1.0, -2.0, 3.0, -4.0], np.float64),
        (ComplexCoeffs, [1j, -2j, 3 + 1j, -4], [1j, -2j, 3 + 1j, -4], np.complex128),
        (SignCoeffs, [Sign.from_scalar(v) for v in [1, -1, -1, 1]], [1, -1, -1, 1], np.int64),
        (
            ComplexSignCoeffs,
            [ComplexSign.from_scalar(v) for v in [1, 1j, -1, -1j]],
            [1, 1j, -1, -1j],
            np.complex128,
        ),
        (SymbolicCoeffs, [sympify(v) for v in [1, -2, 3, -4]], [1, -2, 3, -4], np.float64),
        (SymbolicCoeffs, [sympify(v) for v in [1j, -2j, 3, -4]], [1j, -2j, 3, -4], np.complex128),
    ],
    ids=["real", "complex", "sign", "complex_sign", "symbolic_real", "symbolic_complex"],
)
def coefficient_array(request):
    cls, values, expected, dtype = request.param
    return cls.from_iterable(values), np.array(expected, dtype=dtype)


@pytest.mark.parametrize("conversion", ["to_numpy", "asarray", "array"])
@pytest.mark.parametrize(
    "indexers",
    [
        (),
        (slice(None),),
        (slice(1, 3),),
        (slice(None, None, 2),),
        (slice(None, None, -1),),
        (slice(None, None, 2), slice(None, None, -1)),
        (slice(0, 0),),
    ],
)
def test_array_conversion_copies_visible_elements(coefficient_array, conversion, indexers):
    owner, expected = coefficient_array
    source = owner
    for indexer in indexers:
        source, expected = source[indexer], expected[indexer]
    original = list(owner)

    result = source.to_numpy() if conversion == "to_numpy" else getattr(np, conversion)(source)

    # Empty symbolic arrays have the default real dtype, regardless of their backing values.
    expected_dtype = (
        np.dtype(np.float64)
        if isinstance(source, SymbolicCoeffs) and not len(source)
        else expected.dtype
    )
    assert result.dtype == expected_dtype
    assert result.shape == (len(source),)
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, expected)
    result[:] = 0
    assert list(owner) == original
    owner[0] = original[1]
    np.testing.assert_array_equal(result, np.zeros_like(result))
    assert not hasattr(source, "np_array")


@pytest.mark.parametrize("dtype", [np.complex64, object])
def test_asarray_respects_requested_dtype(coefficient_array, dtype):
    source, expected = coefficient_array
    result = np.asarray(source, dtype=dtype)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, expected.astype(dtype))


@pytest.mark.parametrize("copy", [None, True])
def test_array_protocol_copy_options(coefficient_array, copy):
    source, expected = coefficient_array
    result = source.__array__(copy=copy)
    np.testing.assert_array_equal(result, expected)
    result[:] = 0
    np.testing.assert_array_equal(source.to_numpy(), expected)
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        np.testing.assert_array_equal(np.asarray(source, copy=copy), expected)


def test_array_protocol_rejects_copy_false(coefficient_array):
    source, _ = coefficient_array
    with pytest.raises(ValueError, match="requires a copy"):
        source.__array__(copy=False)
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        with pytest.raises(ValueError, match="requires a copy"):
            np.asarray(source, copy=False)


@pytest.mark.parametrize(
    "cls", [RealCoeffs, ComplexCoeffs, SignCoeffs, ComplexSignCoeffs, SymbolicCoeffs]
)
def test_empty_array_conversion(cls):
    source = cls()
    assert source.to_numpy().shape == (0,)
    assert np.asarray(source).shape == (0,)
    assert np.asarray(source, dtype=np.complex64).dtype == np.dtype(np.complex64)


def test_array_protocol_dispatches_to_subclass_method():
    class CustomCoeffs(RealCoeffs):
        def to_numpy(self):
            return np.array([9.0])

    np.testing.assert_array_equal(np.asarray(CustomCoeffs(size=1)), [9.0])


@pytest.mark.parametrize("cls", [RealCoeffs, ComplexCoeffs])
@pytest.mark.parametrize("indexer", [slice(50, 53), slice(None, None, -1000), slice(0, 0)])
def test_numeric_view_conversion_avoids_full_backing_array(cls, indexer, monkeypatch):
    source = cls.from_iterable(np.arange(10000, dtype=cls.coeff_type))
    view = source[indexer]
    native_type = type(source._impl)
    to_array = native_type.to_array
    converted_lengths = []

    def record_conversion(self):
        converted_lengths.append(len(self))
        return to_array(self)

    monkeypatch.setattr(native_type, "to_array", record_conversion)
    result = view.to_numpy()

    np.testing.assert_array_equal(result, np.arange(10000, dtype=cls.coeff_type)[indexer])
    assert converted_lengths == [len(view)]


def test_symbolic_conversion_requires_only_visible_symbols_to_be_resolved():
    source = SymbolicCoeffs.from_iterable([sympify("x"), sympify(2), sympify("I")])
    for conversion in (source.to_numpy, lambda: np.asarray(source)):
        with pytest.raises(TypeError):
            conversion()
    np.testing.assert_array_equal(source[1:].to_numpy(), [2 + 0j, 1j])
    np.testing.assert_array_equal(np.asarray(source[1:]), [2 + 0j, 1j])


def _complex_source(values, form):
    if form == "list":
        return values
    if form == "tuple":
        return tuple(values)
    if form == "iterator":
        return iter(values)
    if form == "generator":
        return (value for value in values)
    if form == "array":
        return np.array(values, dtype=np.complex128)
    if form == "object_array":
        return np.array([np.complex128(value) for value in values], dtype=object)
    if form == "complex64_array":
        return np.array(values, dtype=np.complex64)
    return (value for value in np.array(values, dtype=np.complex64))


@pytest.mark.parametrize(
    "form",
    [
        "list",
        "tuple",
        "iterator",
        "generator",
        "array",
        "object_array",
        "complex64_array",
        "complex64_generator",
    ],
)
def test_real_construction_accepts_zero_imaginary_parts(form):
    values = [1 + 0j, -2 - 0j, complex(float("nan"), 0), complex(float("inf"), 0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = RealCoeffs.from_iterable(_complex_source(values, form))
    np.testing.assert_array_equal(result.to_numpy(), [1, -2, float("nan"), float("inf")])


@pytest.mark.parametrize(
    "form",
    [
        "list",
        "tuple",
        "iterator",
        "generator",
        "array",
        "object_array",
        "complex64_array",
        "complex64_generator",
    ],
)
@pytest.mark.parametrize("imaginary", [1.0, float("nan"), float("inf")])
def test_real_construction_rejects_nonzero_imaginary_parts(form, imaginary):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="nonzero imaginary"):
            RealCoeffs.from_iterable(_complex_source([1 + 0j, complex(2, imaginary)], form))


@pytest.mark.parametrize("dtype", [np.float32, np.int64, bool])
def test_array_dtype_conversion_rejects_nonzero_imaginary_parts(dtype):
    source = ComplexCoeffs.from_iterable([1 + 0j, 2 + 1j])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="nonzero imaginary"):
            np.asarray(source, dtype=dtype)
    np.testing.assert_array_equal(np.asarray(source[:1], dtype=dtype), np.array([1], dtype=dtype))


def test_real_conversion_rejects_even_small_imaginary_parts():
    with pytest.raises(ValueError, match="nonzero imaginary"):
        RealCoeffs.from_iterable([1 + 1e-300j])


@pytest.fixture(params=[(zfs, "[1]"), (zqs, "[1]"), (zqo, "X0"), (zfog, "F0"), (zfon, "F0")])
def family_word(request):
    return request.param


@pytest.mark.parametrize(
    "scalar",
    [
        np.int64(-2),
        np.uint8(2),
        np.float32(0.5),
        np.float64(0.5),
        np.complex64(1j),
        np.complex128(1j),
    ],
)
def test_numpy_scalar_multiplication(family_word, scalar):
    family, word = family_word
    value = scalar.item()
    sum_type = family.ComplexTermSum if isinstance(value, complex) else family.RealTermSum
    terms = sum_type.from_iterable([(word, 1)], 1)
    for operand in (terms, terms.to_terms(), next(iter(terms)), next(iter(terms)).string):
        expected = operand * value
        for actual in (operand * scalar, scalar * operand):
            assert type(actual) is type(expected)
            assert actual == expected


@pytest.mark.parametrize(
    "weights",
    [
        np.array([-2, 3], dtype=np.int64),
        np.array([[1j, 0], [-0.5j, 2]], dtype=np.complex64),
        np.empty((0, 2)),
    ],
    ids=["vector", "matrix", "empty"],
)
def test_numpy_array_multiplies_individual_strings_and_terms(family_word, weights):
    family, word = family_word
    term = family.RealTerm.from_string(family.String(1, word), 2.0)
    for operand in (term.string, term):
        for result in (weights * operand, operand * weights):
            assert isinstance(result, np.ndarray)
            assert result.shape == weights.shape
            assert result.dtype == object
            for index in np.ndindex(weights.shape):
                expected = operand * weights[index].item()
                assert type(result[index]) is type(expected)
                assert result[index] == expected


def test_add_numpy_scaled_fermion_states():
    first = zfs.RealTermSum.from_str("[1, 1, 0, 0, 0, 0]", modes=6)
    second = zfs.RealTermSum.from_str("[0, 0, 0, 0, 1, 1]", modes=6)
    amplitude = np.sqrt(0.5)
    result = amplitude * first + amplitude * second
    assert isinstance(result, zfs.RealTermSum)
    assert result == float(amplitude) * first + float(amplitude) * second


def test_numpy_complex_promotes_real_term():
    term = zfs.RealTerm.from_str("[1]", modes=1)
    scalar = np.complex64(1j)
    for result in (scalar * term, term * scalar):
        assert isinstance(result, zfs.ComplexTerm)
        assert result == 1j * term


def test_numpy_complex_multiplies_sign():
    sign = Sign.from_scalar(-1)
    scalar = np.complex64(1j)
    for result in (scalar * sign, sign * scalar):
        assert type(result) is complex
        assert result == -1j


def test_numpy_scalar_inplace_scaling_and_division():
    terms = zfs.RealTermSum.from_iterable([("[1]", 3.0)], 1)
    alias = terms
    factor = np.float32(2)
    terms *= factor
    assert terms is alias
    assert list(alias.coeffs) == [6.0]

    quotient = terms / factor
    assert isinstance(quotient, zfs.RealTermSum)
    assert list(quotient.coeffs) == [3.0]
    assert list(terms.coeffs) == [6.0]

    terms /= factor
    assert terms is alias
    assert list(alias.coeffs) == [3.0]


@pytest.mark.parametrize("scalar_type", [complex, np.complex64])
def test_convert_complex_scalar_to_float(scalar_type):
    result = convert(scalar_type(2 + 0j), float)
    assert type(result) is float
    assert result == 2.0
    with pytest.raises(ValueError):
        convert(scalar_type(2 + 1j), float)
