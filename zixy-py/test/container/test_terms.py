from __future__ import annotations

import pytest
from mock_cmpnts import Strings, StringSet, StringsImplArray

from zixy.container.coeffs import ComplexSign, RealCoeffs, Sign, SignCoeffs
from zixy.container.data import TermData
from zixy.container.terms import NumericTerms, Term, Terms, TermSet


def test_sign_terms():
    class MockTerm(Term[StringsImplArray, str, Sign]):
        cmpnts_type = Strings
        coeff_type = Sign

    class MockTerms(Terms[StringsImplArray, str, Sign]):
        term_type = MockTerm

    class MockTermSet(TermSet[StringsImplArray, str, Sign]):
        terms_type = MockTerms
        _set_type = StringSet

    terms = MockTerms(TermData(Strings(0), SignCoeffs.from_size(0)))
    assert (len(terms), len(terms.cmpnts), len(terms.coeffs)) == (0,) * 3
    assert str(terms) == ""
    assert len(MockTermSet(terms)) == 0

    terms = MockTerms(TermData(Strings(1), SignCoeffs.from_scalar(Sign(False), 1)))
    assert (len(terms), len(terms.cmpnts), len(terms.coeffs)) == (1,) * 3
    assert str(terms) == "(+1, )"
    assert len(MockTermSet(terms)) == 1

    terms = MockTerms(TermData(Strings(6), SignCoeffs.from_scalar(Sign(False), 6)))
    assert (len(terms), len(terms.cmpnts), len(terms.coeffs)) == (6,) * 3
    assert str(terms) == "(+1, ), (+1, ), (+1, ), (+1, ), (+1, ), (+1, )"

    assert terms[2].aliases(terms[2])
    assert not terms[2].aliases(terms[3])
    assert terms[2].cmpnt.aliases(terms.cmpnts[2])
    assert not terms[2].cmpnt.aliases(terms.cmpnts[3])
    assert not terms[2].cmpnt.aliases(terms.cmpnts[2].clone())

    assert not terms[2].aliases(terms[2].clone())
    assert not terms[2].aliases(terms[2].clone())
    assert not terms[2].aliases(terms[2].clone())

    terms.coeffs[4] = Sign(True)
    assert str(terms) == "(+1, ), (+1, ), (+1, ), (+1, ), (-1, ), (+1, )"

    terms[2].coeff = Sign(True)
    assert str(terms) == "(+1, ), (+1, ), (-1, ), (+1, ), (-1, ), (+1, )"

    terms.cmpnts[2].set("hello")
    assert str(terms) == "(+1, ), (+1, ), (-1, hello), (+1, ), (-1, ), (+1, )"

    terms[3].cmpnt.set("world!")
    assert str(terms) == "(+1, ), (+1, ), (-1, hello), (+1, world!), (-1, ), (+1, )"

    terms[1] = "Pauli", Sign(True)
    assert str(terms) == "(+1, ), (-1, Pauli), (-1, hello), (+1, world!), (-1, ), (+1, )"

    terms[1] = "Fermi"
    assert str(terms) == "(+1, ), (+1, Fermi), (-1, hello), (+1, world!), (-1, ), (+1, )"

    terms[1] = None, Sign(True)
    assert str(terms) == "(+1, ), (-1, ), (-1, hello), (+1, world!), (-1, ), (+1, )"

    terms[1] = terms[3].cmpnt, Sign(False)
    assert str(terms) == "(+1, ), (+1, world!), (-1, hello), (+1, world!), (-1, ), (+1, )"

    terms[1] = terms[2].cmpnt, Sign(True)
    assert str(terms) == "(+1, ), (-1, hello), (-1, hello), (+1, world!), (-1, ), (+1, )"

    terms.cmpnts[1] = "Pauli"
    terms[-3] = terms[1]
    assert str(terms) == "(+1, ), (-1, Pauli), (-1, hello), (-1, Pauli), (-1, ), (+1, )"

    terms[::-2].cmpnts[1] = "Fermi"
    assert str(terms) == "(+1, ), (-1, Pauli), (-1, hello), (-1, Fermi), (-1, ), (+1, )"

    terms[::-2].cmpnts[0] = "Fermi"
    assert str(terms) == "(+1, ), (-1, Pauli), (-1, hello), (-1, Fermi), (-1, ), (+1, Fermi)"

    terms[::-2].cmpnts[-1] = "Fermi"
    assert str(terms) == "(+1, ), (-1, Fermi), (-1, hello), (-1, Fermi), (-1, ), (+1, Fermi)"

    terms.resize(3)
    assert str(terms) == "(+1, ), (-1, Fermi), (-1, hello)"
    terms.resize(5)
    assert str(terms) == "(+1, ), (-1, Fermi), (-1, hello), (+1, ), (+1, )"
    terms.append(None)
    assert str(terms) == "(+1, ), (-1, Fermi), (-1, hello), (+1, ), (+1, ), (+1, )"
    assert str(terms.cmpnts[2::-1]) == "hello, Fermi, "

    owning = terms.clone()
    assert owning == terms
    assert owning is not terms
    assert all(a._impl is not b._impl for a, b in zip(terms, owning, strict=False))
    assert owning.is_owning()

    view = terms[:]
    assert view == terms
    assert view is not terms
    assert all(a._impl is b._impl for a, b in zip(terms, view, strict=False))
    assert not view.is_owning()
    with pytest.raises(ValueError):
        view.resize(0)

    view = terms[1:4]
    owning = view.clone()
    assert view == owning
    assert view is not owning
    assert all(a._impl is not b._impl for a, b in zip(view, owning, strict=False))

    for i, el in enumerate(terms):
        assert el == terms[i]
        assert el._impl is terms[i]._impl

    assert len(MockTermSet(terms)) == 3
    assert str(MockTermSet(terms)) == "(+1, ), (-1, Fermi), (-1, hello)"
    assert str(MockTermSet(terms).clone()) == "(+1, ), (-1, Fermi), (-1, hello)"


def test_real_terms():
    class MockTerm(Term[StringsImplArray, str, float]):
        cmpnts_type = Strings
        coeff_type = float

    class MockTerms(NumericTerms[StringsImplArray, str, float]):
        term_type = MockTerm

    terms = MockTerms(TermData(Strings(0), RealCoeffs.from_size(0)))
    assert (len(terms), len(terms.cmpnts), len(terms.coeffs)) == (0,) * 3
    assert str(terms) == ""

    terms = MockTerms(TermData(Strings(1), RealCoeffs.from_scalar(1.0, 1)))
    assert (len(terms), len(terms.cmpnts), len(terms.coeffs)) == (1,) * 3
    assert str(terms) == "(1.0, )"

    terms = MockTerms(TermData(Strings(6), RealCoeffs.from_scalar(1.0, 6)))
    assert (len(terms), len(terms.cmpnts), len(terms.coeffs)) == (6,) * 3
    assert str(terms) == "(1.0, ), (1.0, ), (1.0, ), (1.0, ), (1.0, ), (1.0, )"

    assert terms[2].aliases(terms[2])
    assert not terms[2].aliases(terms[3])
    assert terms[2].cmpnt.aliases(terms.cmpnts[2])
    assert not terms[2].cmpnt.aliases(terms.cmpnts[3])
    assert not terms[2].cmpnt.aliases(terms.cmpnts[2].clone())

    assert not terms[2].aliases(terms[2].clone())
    assert not terms[2].aliases(terms[2].clone())
    assert not terms[2].aliases(terms[2].clone())

    terms.coeffs[4] = Sign(True)
    assert str(terms) == "(1.0, ), (1.0, ), (1.0, ), (1.0, ), (-1.0, ), (1.0, )"

    terms[2].coeff = Sign(True)
    assert str(terms) == "(1.0, ), (1.0, ), (-1.0, ), (1.0, ), (-1.0, ), (1.0, )"

    terms[1].coeff = 1.234
    assert str(terms) == "(1.0, ), (1.234, ), (-1.0, ), (1.0, ), (-1.0, ), (1.0, )"

    terms.cmpnts[2].set("hello")
    assert str(terms) == "(1.0, ), (1.234, ), (-1.0, hello), (1.0, ), (-1.0, ), (1.0, )"

    terms[3].cmpnt.set("world!")
    assert str(terms) == "(1.0, ), (1.234, ), (-1.0, hello), (1.0, world!), (-1.0, ), (1.0, )"

    terms[1] = "Pauli", 4.567
    assert str(terms) == "(1.0, ), (4.567, Pauli), (-1.0, hello), (1.0, world!), (-1.0, ), (1.0, )"

    terms[1] = "Fermi"
    assert str(terms) == "(1.0, ), (1.0, Fermi), (-1.0, hello), (1.0, world!), (-1.0, ), (1.0, )"

    terms[1] = None, ComplexSign(2)
    assert str(terms) == "(1.0, ), (-1.0, ), (-1.0, hello), (1.0, world!), (-1.0, ), (1.0, )"

    terms[1] = terms[3].cmpnt, 9
    assert str(terms) == "(1.0, ), (9.0, world!), (-1.0, hello), (1.0, world!), (-1.0, ), (1.0, )"

    terms_1 = MockTerms(TermData(Strings(6), RealCoeffs.from_scalar(1.0, 6)))
    terms_2 = MockTerms(TermData(Strings(6), RealCoeffs.from_scalar(1.0, 6)))

    assert terms_1 == terms_2
    assert terms_1.allclose(terms_2)

    terms_2.coeffs[1] += 1e-6
    assert terms_1 != terms_2
    assert terms_1.allclose(terms_2)

    terms_2.coeffs.fill(1 + 1e-6)
    assert terms_1 != terms_2
    assert terms_1.allclose(terms_2)

    terms_1.coeffs.set(range(6))
    terms_2.coeffs.set(range(6))

    assert terms_1 == terms_2
    assert terms_1.allclose(terms_2)

    assert terms_1[4::-1] == terms_2[4::-1]
    assert terms_1[4::-1].allclose(terms_2[4::-1])
