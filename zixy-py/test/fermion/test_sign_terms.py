"""Exact sign coefficients in fermionic state and operator families."""

import pytest

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
from zixy.container.coeffs import ComplexSign, Sign
from zixy.container.serialization import deserialize, serialize


@pytest.mark.parametrize(("family", "word"), [(zfog, "F0"), (zfon, "F0"), (zfs, "[0]")])
@pytest.mark.parametrize("coeff", [Sign.from_scalar(-1), ComplexSign.from_scalar(1j)])
def test_exact_sign_containers(family, word, coeff):
    term = family.String(1, word) * coeff
    prefix = type(coeff).__name__
    assert type(term) is getattr(family, prefix + "Term")
    assert term.coeff == coeff
    terms = getattr(family, prefix + "Terms").from_iterable([term], 1)
    term_set = getattr(family, prefix + "TermSet").from_terms(terms)
    for obj in (term, terms, term_set):
        restored = deserialize(serialize(obj))
        assert type(restored) is type(obj)
        assert restored == obj


@pytest.mark.parametrize("family", [zfog, zfon])
def test_complex_sign_adjoint(family):
    term = family.ComplexSignTerm(1, ("F0", ComplexSign.from_scalar(1j)))
    assert term.daggered() == family.ComplexSignTerm(1, ("F0^", ComplexSign.from_scalar(-1j)))


def test_exact_sign_products():
    phase = ComplexSign.from_scalar(1j)
    sign = Sign.from_scalar(-1)
    raw = zfog.ComplexSignTerm(1, ("F0", phase)) * zfog.SignTerm(1, ("F0^", sign))
    assert raw == zfog.ComplexSignTerm(1, ("F0 F0^", -phase))
    normal = zfon.ComplexSignTerm(1, ("F0", phase)) * zfon.SignTerm(1, ("F0^", sign))
    assert isinstance(normal, zfon.ComplexTermSum)
    assert normal == zfon.ComplexTermSum.from_iterable([("", -1j), ("F0^ F0", 1j)], 1)
