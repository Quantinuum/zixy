import pytest

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs


@pytest.fixture(
    params=[
        (zqo, ["X0", "Y0", "Z0"], {}),
        (zqs, ["[0, 0]", "[1, 0]", "[0, 1]"], {}),
        (zfog, ["F0", "F1", "F0^ F1"], {"max_string_len": 2}),
        (zfon, ["F0", "F1", "F0^ F1"], {}),
        (zfs, ["[0, 0]", "[1, 0]", "[0, 1]"], {}),
    ],
    ids=["qubit-operator", "qubit-state", "general-fermion", "normal-fermion", "fermion-state"],
)
def family(request):
    return request.param


def test_component_set_retains_distinct_views(family):
    api, specs, kwargs = family
    strings = api.StringSet.from_iterable(specs, 2, **kwargs)
    expected = [str(string) for string in strings]
    iterator = iter(strings)
    first = next(iterator)
    saved = str(first)
    remaining = list(iterator)
    collected = [first, *remaining]

    assert str(first) == saved
    assert [str(string) for string in collected] == expected
    assert len({id(string) for string in collected}) == len(specs)
    assert all(string._impl is strings._impl for string in collected)
    assert all(not string.is_owning() for string in collected)
    assert list(api.StringSet(2, **kwargs)) == []
    single = api.StringSet.from_iterable(specs[:1], 2, **kwargs)
    assert [str(string) for string in single] == expected[:1]


@pytest.mark.parametrize("as_set", [False, True], ids=["array", "set"])
def test_component_filter_map_retains_distinct_copies(family, as_set):
    api, specs, kwargs = family
    array = api.Strings.from_iterable(specs, 2, **kwargs)
    source = api.StringSet.from_strings(array) if as_set else array[::-1]
    expected = [str(string) for string in source]
    collected = list(source.iter_filter_map(lambda string: True))

    assert [str(string) for string in collected] == expected
    assert len({id(string) for string in collected}) == len(specs)
    collected[0].set(specs[1])
    assert [str(string) for string in source] == expected
    assert [str(string) for string in collected[1:]] == expected[1:]

    # Callbacks may retain every argument, including rejected elements.
    seen = []

    def mutate(string):
        keep = str(string) != expected[1]
        seen.append(string)
        string.set(specs[0])
        return keep

    mapped = list(source.iter_filter_map(mutate))
    assert len(mapped) == 2
    assert len({id(string) for string in seen}) == len(specs)
    assert mapped[0] is seen[0]
    assert mapped[1] is seen[2]
    assert [str(string) for string in source] == expected
    empty = array[:0]
    assert list(empty.iter_filter_map(mutate)) == []
    assert len(seen) == len(specs)


@pytest.mark.parametrize("as_set", [False, True], ids=["array", "set"])
def test_term_filter_map_retains_distinct_copies(family, as_set):
    api, specs, kwargs = family
    array = api.RealTerms.from_iterable(
        [(spec, float(i + 1)) for i, spec in enumerate(specs)], 2, **kwargs
    )
    source = api.RealTermSet.from_terms(array) if as_set else array[::-1]
    expected = [str(term) for term in source]
    expected_coeffs = [term.coeff for term in source]
    expected_strings = [str(term.string) for term in source]
    collected = list(source.iter_filter_map(lambda term: True))
    assert [str(term) for term in collected] == expected
    assert len({id(term) for term in collected}) == len(specs)

    seen = []

    def mutate(term):
        keep = term.coeff != 2.0
        seen.append(term)
        term.coeff += 10.0
        term.string.set(specs[0])
        return keep

    mapped = list(source.iter_filter_map(mutate))
    assert [term.coeff for term in mapped] == [c + 10.0 for c in expected_coeffs if c != 2.0]
    assert [term.coeff for term in seen] == [c + 10.0 for c in expected_coeffs]
    assert len({id(term) for term in seen}) == len(specs)
    assert [str(term) for term in source] == expected
    assert [str(term.string) for term in collected] == expected_strings
    assert [term.coeff for term in collected] == expected_coeffs
    assert list(array[:0].iter_filter_map(mutate)) == []
    assert len(seen) == len(specs)


def test_filtered_set_iterator_preserves_views(family):
    api, specs, kwargs = family
    array = api.RealTerms.from_iterable([(spec, 1.0) for spec in specs], 2, **kwargs)
    terms = api.RealTermSet.from_terms(array)
    collected = list(terms.iter_filter_nonzero())
    assert [str(term.string) for term in collected] == [str(term.string) for term in array]
    assert len({id(term) for term in collected}) == len(specs)
    collected[0].coeff = 2.0
    assert terms[specs[0]] == 2.0
    assert [term.coeff for term in collected[1:]] == [1.0, 1.0]
