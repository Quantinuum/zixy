"""Symbolic sums share coefficient substitution across all string families."""

import pytest
from sympy import symbols, sympify

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy._zixy import Modes, Qubits


@pytest.fixture(params=[zqo, zqs, zfog, zfon, zfs], ids=lambda family: family.__name__)
def symbolic_sum(request):
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
    x, y = symbols("x y")
    source = family.SymbolicTermSum.from_iterable(zip(specs, [x + y, 2 * x - y]), space, **kwargs)
    return family, space, specs, kwargs, source


def test_subs_preserves_keys_space_and_independence(symbolic_sum):
    _, _, specs, _, source = symbolic_sum
    x, _, z = symbols("x y z")
    before = source.to_dict()

    result = source.subs({x: 2, "y": z})

    assert type(result) is type(source)
    assert list(result.coeffs) == [2 + z, 4 - z]
    assert result.to_dict()["strings"] == before["strings"]
    assert result.to_dict()["space"] == before["space"]
    assert result.strings.is_owning()
    assert result.coeffs.is_owning()
    assert result._string_set._map is not source._string_set._map
    for index, spec in enumerate(specs):
        assert result.lookup_index(spec) == index
        assert result.get(spec) == result.coeffs[index]
    if hasattr(source, "max_string_len"):
        assert result.max_string_len == source.max_string_len == 64
    result.coeffs[0] = sympify(10)
    result.remove(specs[1])
    assert source.to_dict() == before


def test_isubs_updates_existing_views_without_changing_keys(symbolic_sum):
    _, _, specs, _, source = symbolic_sum
    x, _, z = symbols("x y z")
    before = source.to_dict()
    term_view = next(iter(source))
    coeff_view = source.coeffs[::-1]
    mapping = source._string_set._map

    assert source.isubs({x: 2, "y": z}) is None

    assert list(source.coeffs) == [2 + z, 4 - z]
    assert term_view.coeff == 2 + z
    assert list(coeff_view) == [4 - z, 2 + z]
    assert source._string_set._map is mapping
    assert source.to_dict()["strings"] == before["strings"]
    assert source.to_dict()["space"] == before["space"]
    assert [source.lookup_index(spec) for spec in specs] == [0, 1]


def test_subs_retains_explicit_zero_terms(symbolic_sum):
    _, _, specs, _, source = symbolic_sum
    result = source.subs({"x": 0, "y": 0})

    assert len(result) == len(source) == 2
    assert list(result.coeffs) == [0, 0]
    assert all(spec in result for spec in specs)
    assert all(term in result for term in result)
    assert [result.get(spec) for spec in specs] == [0, 0]


def test_subs_with_empty_mapping_clones_and_preserves_subclass(symbolic_sum):
    family, space, specs, kwargs, source = symbolic_sum

    class DerivedSum(family.SymbolicTermSum):
        pass

    derived = DerivedSum.from_iterable(zip(specs, source.coeffs), space, **kwargs)
    result = derived.subs({})

    assert type(result) is DerivedSum
    assert result == derived
    result.coeffs[0] = sympify(0)
    assert result != derived
    assert derived == source


def test_empty_sum_substitution_preserves_configuration(symbolic_sum):
    family, space, _, kwargs, _ = symbolic_sum
    source = family.SymbolicTermSum(space, **kwargs)
    before = source.to_dict()

    assert source.isubs({"x": 2}) is None
    result = source.subs({"x": 2})

    assert type(result) is type(source)
    assert len(result) == 0
    assert result.to_dict() == source.to_dict() == before
    assert result.coeffs.is_owning()
    assert result.strings.is_owning()
    if hasattr(source, "max_string_len"):
        assert result.max_string_len == source.max_string_len == 64


def test_numeric_sums_do_not_expose_substitution(symbolic_sum):
    family, space, _, kwargs, _ = symbolic_sum
    for collection_type in (family.RealTermSum, family.ComplexTermSum):
        collection = collection_type(space, **kwargs)
        assert not hasattr(collection, "subs")
        assert not hasattr(collection, "isubs")
