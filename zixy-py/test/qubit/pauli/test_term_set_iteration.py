import pytest

from zixy.container.coeffs import unit
from zixy.qubit import pauli


@pytest.fixture(
    params=[
        pauli.SignTermSet,
        pauli.ComplexSignTermSet,
        pauli.RealTermSet,
        pauli.ComplexTermSet,
        pauli.SymbolicTermSet,
        pauli.RealTermSum,
        pauli.ComplexTermSum,
        pauli.SymbolicTermSum,
    ],
    ids=lambda cls: cls.__name__,
)
def terms(request):
    terms_type = request.param.terms_type
    coeff = unit(terms_type.term_type.coeff_type)
    source = terms_type.from_iterable([("X0", coeff), ("Z0", coeff)], 1)
    return request.param.from_terms(source)


def test_sign_term_set_issue_262():
    terms = pauli.SignTermSet.from_str("(+1, X0), (+1, Z0)")
    collected = list(terms)
    assert [str(term) for term in collected] == ["(+1, X0)", "(+1, Z0)"]
    assert collected[0] is not collected[1]


def test_collected_terms_retain_distinct_values(terms):
    expected = [str(term) for term in terms]
    collected = list(terms)

    assert [str(term) for term in collected] == expected
    assert [str(term.string) for term in collected] == ["X0", "Z0"]
    assert collected[0] is not collected[1]
    assert list(type(terms)(1)) == []
    single = list(type(terms).from_terms(terms.to_terms()[:1]))
    assert len(single) == 1
    assert str(single[0].string) == "X0"


def test_advancing_iterator_preserves_previous_term(terms):
    iterator = iter(terms)
    first = next(iterator)
    saved = str(first)
    second = next(iterator)

    assert str(first) == saved
    assert str(first.string) == "X0"
    assert str(second.string) == "Z0"
    with pytest.raises(StopIteration):
        next(iterator)
    assert str(first) == saved


def test_yielded_terms_are_distinct_views(terms):
    first, second = list(terms)
    saved_second = str(second)

    assert first._impl is terms._impl
    assert second._impl is terms._impl
    assert not first.is_owning()
    assert not second.is_owning()
    first.coeff = -first.coeff

    assert str(second) == saved_second
    assert terms["X0"] == first.coeff
    terms["Z0"] = -second.coeff
    assert second.coeff == terms["Z0"]
    assert str(second) != saved_second
    assert terms.lookup_index("X0") == 0
    assert terms.lookup_index("Z0") == 1


def test_filter_map_still_operates_on_copies(terms):
    expected = [str(term) for term in terms]

    def mutate(term):
        term.string.set("Y0")
        term.coeff = -term.coeff
        return True

    collected = list(terms.iter_filter_map(mutate))

    assert collected[0] is not collected[1]
    assert [str(term.string) for term in collected] == ["Y0", "Y0"]
    assert all(term.coeff == -unit(terms.coeff_type) for term in collected)
    assert [str(term) for term in terms] == expected
    assert terms.lookup_index("X0") == 0
    assert terms.lookup_index("Z0") == 1
    assert terms.lookup_index("Y0") is None
