from __future__ import annotations

import pytest
from sympy import Expr, sympify
from typing_extensions import Self

from zixy.container.coeffs import (
    ComplexCoeffs,
    ComplexSign,
    RealCoeffs,
    Sign,
    SignCoeffs,
    SymbolicCoeffs,
)
from zixy.container.data import TermData
from zixy.container.terms import NumericTerms, NumericTermSum, Term, Terms, TermSet, TermSum

from .mock_cmpnts import String, Strings, StringSet, StringsImplArray


def _mock_term_from_str(cls: type[Term[StringsImplArray, str, object]], source: str) -> object:
    if not source.startswith("(") or not source.endswith(")"):
        raise ValueError(f"String {source} is not a valid representation of a term.")
    coeff_str, cmpnt_str = source[1:-1].split(",", 1)
    coeff_str = coeff_str.strip()
    cmpnt_str = cmpnt_str.strip()
    if cls.coeff_type is Sign:
        coeff = Sign.from_str(coeff_str)
    elif cls.coeff_type is float:
        coeff = float(coeff_str)
    elif cls.coeff_type is complex:
        coeff = complex(coeff_str)
    else:
        raise TypeError(f"Unsupported mock coefficient type {cls.coeff_type}.")
    return cls.from_cmpnt_coeff(String.from_str(cmpnt_str), coeff)


class RealMockTerm(Term[StringsImplArray, str, float]):
    cmpnts_type = Strings
    coeff_type = float

    @classmethod
    def from_str(cls, source: str) -> Self:
        return _mock_term_from_str(cls, source)


class RealMockTerms(NumericTerms[StringsImplArray, str, float]):
    term_type = RealMockTerm

    def __init__(self, data: TermData[StringsImplArray, str, float] | None = None):
        super().__init__(TermData(Strings(0), RealCoeffs.from_size(0)) if data is None else data)


class RealMockTermSet(TermSet[StringsImplArray, str, float]):
    terms_type = RealMockTerms


class RealMockTermSum(NumericTermSum[StringsImplArray, str, float]):
    terms_type = RealMockTerms

    def __init__(self):
        super().__init__(RealMockTerms())


class ComplexMockTerm(Term[StringsImplArray, str, complex]):
    cmpnts_type = Strings
    coeff_type = complex

    @classmethod
    def from_str(cls, source: str) -> Self:
        return _mock_term_from_str(cls, source)


class ComplexMockTerms(NumericTerms[StringsImplArray, str, complex]):
    term_type = ComplexMockTerm

    def __init__(self, data: TermData[StringsImplArray, str, complex] | None = None):
        super().__init__(TermData(Strings(0), ComplexCoeffs.from_size(0)) if data is None else data)


class ComplexMockTermSet(TermSet[StringsImplArray, str, complex]):
    terms_type = ComplexMockTerms


class ComplexMockTermSum(NumericTermSum[StringsImplArray, str, complex]):
    terms_type = ComplexMockTerms

    def __init__(self):
        super().__init__(ComplexMockTerms())


class SymbolicMockTerm(Term[StringsImplArray, str, Expr]):
    cmpnts_type = Strings
    coeff_type = Expr

    @classmethod
    def from_str(cls, source: str) -> Self:
        return _mock_term_from_str(cls, source)


class SymbolicMockTerms(Terms[StringsImplArray, str, Expr]):
    term_type = SymbolicMockTerm

    def __init__(self, data: TermData[StringsImplArray, str, Expr] | None = None):
        super().__init__(
            TermData(Strings(0), SymbolicCoeffs.from_size(0)) if data is None else data
        )


class SymbolicMockTermSum(TermSum[StringsImplArray, str, Expr]):
    terms_type = SymbolicMockTerms

    def __init__(self):
        super().__init__(SymbolicMockTerms())


def test_sign_terms():
    class MockTerm(Term[StringsImplArray, str, Sign]):
        cmpnts_type = Strings
        coeff_type = Sign

        @classmethod
        def from_str(cls, source: str) -> Self:
            return _mock_term_from_str(cls, source)

    class MockTerms(Terms[StringsImplArray, str, Sign]):
        term_type = MockTerm

        def __init__(self, data: TermData[StringsImplArray, str, Sign] | None = None):
            super().__init__(
                TermData(Strings(0), SignCoeffs.from_size(0)) if data is None else data
            )

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

    terms[1] = "", Sign(True)
    assert str(terms) == "(+1, ), (-1, ), (-1, hello), (+1, world!), (-1, ), (+1, )"
    with pytest.raises(TypeError):
        terms[1] = None, Sign(True)

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
    terms.append()
    assert str(terms) == "(+1, ), (-1, Fermi), (-1, hello), (+1, ), (+1, ), (+1, )"
    with pytest.raises(TypeError):
        terms.append(None)
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
    assert MockTerms.from_str(str(terms)) == terms
    assert MockTermSet.from_str(str(MockTermSet(terms))) == MockTermSet(terms)


def test_real_terms():
    class MockTerm(Term[StringsImplArray, str, float]):
        cmpnts_type = Strings
        coeff_type = float

        @classmethod
        def from_str(cls, source: str) -> Self:
            return _mock_term_from_str(cls, source)

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

    terms[1] = "", ComplexSign(2)
    assert str(terms) == "(1.0, ), (-1.0, ), (-1.0, hello), (1.0, world!), (-1.0, ), (1.0, )"
    with pytest.raises(TypeError):
        terms[1] = None, ComplexSign(2)

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

    term_sum = RealMockTermSum.from_iterable((("alpha", 1.0), ("beta", -2.0), ("alpha", 3.0)))
    assert RealMockTermSet.from_str(str(RealMockTermSet.from_terms(term_sum.to_terms()))) == (
        RealMockTermSet.from_terms(term_sum.to_terms())
    )
    assert RealMockTermSum.from_str(str(term_sum)) == term_sum


def test_termsum_vectorised_coeff_multiplication():
    term_sum = RealMockTermSum.from_iterable((("alpha", 1.0), ("beta", -2.0), ("alpha", 3.0)))
    coeffs = RealCoeffs.from_sequence((2.0, 0.5))
    assert tuple(term_sum.to_terms().coeffs) == (4.0, -2.0)
    term_sum_omul = term_sum * coeffs
    term_sum_rmul = coeffs * term_sum
    term_sum *= coeffs
    # (4.0, -2.0).(2.0, 0.5) = (8.0, -1.0)
    assert tuple(term_sum.to_terms().coeffs) == (8.0, -1.0)
    assert tuple(term_sum_omul.to_terms().coeffs) == (8.0, -1.0)
    assert tuple(term_sum_rmul.to_terms().coeffs) == (8.0, -1.0)

    with pytest.raises(ValueError):
        term_sum_2 = RealMockTermSum.from_iterable(
            (("alpha", 1.0), ("beta", -2.0), ("alpha", 3.0), ("gamma", 4.0))
        )
        term_sum_2 *= coeffs

    term_sum *= term_sum.to_terms().coeffs[::-1]
    # (4.0, -2.0).(-2.0, 4.0) = (-8.0, -8.0)
    assert tuple(term_sum.to_terms().coeffs) == (-8.0, -8.0)


def test_terms_vectorised_coeff_multiplication():
    class MockTerm(Term[StringsImplArray, str, float]):
        cmpnts_type = Strings
        coeff_type = float

        @classmethod
        def from_str(cls, source: str) -> Self:
            return _mock_term_from_str(cls, source)

    class MockTerms(NumericTerms[StringsImplArray, str, float]):
        term_type = MockTerm

    terms = MockTerms(TermData(Strings(3), RealCoeffs.from_sequence((1.0, -3.0, 0.5))))
    coeffs = RealCoeffs.from_sequence((2.0, 0.5, -1.0))
    term_rmul = coeffs * terms
    terms_omul = terms * coeffs
    terms *= coeffs
    # (1.0, -3.0, 0.5).(2.0, 0.5, -1.0) = (2.0, -1.5, -0.5)
    assert tuple(terms.coeffs) == (2.0, -1.5, -0.5)
    assert tuple(terms_omul.coeffs) == (2.0, -1.5, -0.5)
    assert tuple(term_rmul.coeffs) == (2.0, -1.5, -0.5)

    with pytest.raises(ValueError):
        terms_2 = MockTerms(TermData(Strings(4), RealCoeffs.from_scalar(1.0, 4)))
        terms_2 *= coeffs

    terms *= terms.coeffs[::-1]
    # (2.0, -1.5, -0.5).(-0.5, -1.5, 2.0) = (-1.0, 2.25, -1.0)
    assert tuple(terms.coeffs) == (-1.0, 2.25, -1.0)


def test_sign_terms_vectorised_coeff_multiplication():
    class MockTerm(Term[StringsImplArray, str, Sign]):
        cmpnts_type = Strings
        coeff_type = Sign

        @classmethod
        def from_str(cls, source: str) -> Self:
            return _mock_term_from_str(cls, source)

    class MockTerms(Terms[StringsImplArray, str, Sign]):
        term_type = MockTerm

        def __init__(self, data: TermData[StringsImplArray, str, Sign] | None = None):
            super().__init__(
                TermData(Strings(0), SignCoeffs.from_size(0)) if data is None else data
            )

    terms = MockTerms(
        TermData(
            Strings(3),
            SignCoeffs.from_sequence((Sign(True), Sign(False), Sign(True))),
        )
    )
    coeffs = SignCoeffs.from_sequence((Sign(False), Sign(True), Sign(False)))
    terms_omul = terms * coeffs
    terms_rmul = coeffs * terms
    terms *= coeffs
    # False represents +1 and True represents -1.
    # (-1, +1, -1).(+1, -1, +1) = (-1, -1, -1)
    assert tuple(term.phase for term in terms.coeffs) == (True, True, True)
    assert tuple(term.phase for term in terms_omul.coeffs) == (True, True, True)
    assert tuple(term.phase for term in terms_rmul.coeffs) == (True, True, True)

    with pytest.raises(ValueError):
        terms_2 = MockTerms(TermData(Strings(4), SignCoeffs.from_scalar(Sign(False), 4)))
        terms_2 *= coeffs

    terms *= terms.coeffs[::-1]
    # (-1, -1, -1).(-1, -1, -1) = (+1, +1, +1)
    assert tuple(term.phase for term in terms.coeffs) == (False, False, False)


def test_complex_termsum_vectorised_coeff_multiplication():
    term_sum = ComplexMockTermSum.from_iterable(
        (("alpha", 1 + 2j), ("beta", -2 + 1j), ("alpha", 3 - 1j))
    )
    coeffs = ComplexCoeffs.from_sequence((1j, 2 - 1j))
    assert tuple(term_sum.to_terms().coeffs) == (4 + 1j, -2 + 1j)
    # (4 + 1j, -2 + 1j).(1j, 2 - 1j) = (-1 + 4j, -3 + 4j)
    term_sum_omul = term_sum * coeffs
    term_sum_rmul = coeffs * term_sum
    term_sum *= coeffs
    assert tuple(term_sum.to_terms().coeffs) == (-1 + 4j, -3 + 4j)
    assert tuple(term_sum_omul.to_terms().coeffs) == (-1 + 4j, -3 + 4j)
    assert tuple(term_sum_rmul.to_terms().coeffs) == (-1 + 4j, -3 + 4j)

    with pytest.raises(ValueError):
        term_sum_2 = ComplexMockTermSum.from_iterable(
            (("alpha", 1 + 2j), ("beta", -2 + 1j), ("gamma", 0.5 - 1j))
        )
        term_sum_2 *= coeffs

    term_sum *= term_sum.to_terms().coeffs[::-1]
    # (-1 + 4j, -3 + 4j).(-3 + 4j, -1 + 4j) = (-13 - 16j, -13 - 16j)
    assert tuple(term_sum.to_terms().coeffs) == (-13 - 16j, -13 - 16j)


def test_symbolic_terms_vectorised_coeff_multiplication():
    terms = SymbolicMockTerms(
        TermData(
            Strings(3),
            SymbolicCoeffs.from_sequence((sympify("x"), sympify("y"), sympify("x + y"))),
        )
    )
    coeffs = SymbolicCoeffs.from_sequence((sympify(2), sympify("x"), sympify(-1)))
    terms_omul = terms * coeffs
    terms_rmul = coeffs * terms
    terms *= coeffs
    # (x, y, x + y).(2, x, -1) = (2*x, x*y, -x - y)
    assert tuple(terms.coeffs) == (sympify("2*x"), sympify("x*y"), sympify("-x - y"))
    assert tuple(terms_omul.coeffs) == (sympify("2*x"), sympify("x*y"), sympify("-x - y"))
    assert tuple(terms_rmul.coeffs) == (sympify("2*x"), sympify("x*y"), sympify("-x - y"))

    with pytest.raises(ValueError):
        terms_2 = SymbolicMockTerms(
            TermData(Strings(4), SymbolicCoeffs.from_sequence((sympify(1),) * 4))
        )
        terms_2 *= coeffs

    terms *= terms.coeffs[::-1]
    # (2*x, x*y, -x - y).(-x - y, x*y, 2*x) = (2*x*(-x - y), x**2*y**2, 2*x*(-x-y))
    assert tuple(terms.coeffs) == (
        sympify("2*x*(-x - y)"),
        sympify("x**2*y**2"),
        sympify("2*x*(-x-y)"),
    )


def test_symbolic_termsum_vectorised_coeff_multiplication():
    term_sum = SymbolicMockTermSum.from_iterable(
        (("alpha", sympify("x")), ("beta", sympify("y")), ("alpha", sympify(2)))
    )
    coeffs = SymbolicCoeffs.from_sequence((sympify("z"), sympify(-1)))
    assert tuple(term_sum.to_terms().coeffs) == (sympify("x + 2"), sympify("y"))
    term_sum_rmul = coeffs * term_sum
    term_sum_omul = term_sum * coeffs
    term_sum *= coeffs
    assert tuple(term_sum.to_terms().coeffs) == (sympify("z*(x + 2)"), sympify("-y"))
    assert tuple(term_sum_omul.to_terms().coeffs) == (sympify("z*(x + 2)"), sympify("-y"))
    assert tuple(term_sum_rmul.to_terms().coeffs) == (sympify("z*(x + 2)"), sympify("-y"))

    with pytest.raises(ValueError):
        term_sum_2 = SymbolicMockTermSum.from_iterable(
            (("alpha", sympify("x")), ("beta", sympify("y")), ("gamma", sympify("z")))
        )
        term_sum_2 *= coeffs

    term_sum *= term_sum.to_terms().coeffs[::-1]
    assert tuple(term_sum.to_terms().coeffs) == (
        sympify("-y*z*(x + 2)"),
        sympify("-y*z*(x + 2)"),
    )


def test_str():
    empty_terms = RealMockTerms(TermData(Strings(0), RealCoeffs.from_size(0)))
    assert empty_terms.to_str() == ""
    assert RealMockTerms.from_str(empty_terms.to_str()) == empty_terms

    empty_term_set = RealMockTermSet.from_terms(empty_terms)
    assert empty_term_set.to_str() == ""
    assert RealMockTermSet.from_str(empty_term_set.to_str()) == empty_term_set


def test_container_into_shape_conversions():
    cmpnts = Strings.from_iterable(("alpha", "beta", "alpha"))
    terms = cmpnts.into(RealMockTerms)
    assert type(terms) is RealMockTerms
    assert terms.cmpnts == cmpnts
    assert tuple(terms.coeffs) == (1.0, 1.0, 1.0)
    assert terms.cmpnts is not cmpnts

    cmpnt_set = StringSet.from_cmpnts(cmpnts)
    assert type(cmpnt_set) is StringSet
    assert len(cmpnt_set) == 2
    assert str(cmpnt_set.to_cmpnts()) == "alpha, beta"

    term_set = RealMockTermSet.from_terms(terms)
    assert type(term_set) is RealMockTermSet
    assert len(term_set) == 2
    assert term_set["alpha"] == 1.0
    assert term_set["beta"] == 1.0

    complex_term_set = term_set.into(ComplexMockTermSet)
    assert type(complex_term_set) is ComplexMockTermSet
    assert len(complex_term_set) == 2
    assert complex_term_set["alpha"] == 1.0 + 0j
    assert complex_term_set["beta"] == 1.0 + 0j

    round_trip_cmpnts = term_set.to_terms().into(Strings)
    assert type(round_trip_cmpnts) is Strings
    assert str(round_trip_cmpnts) == "alpha, beta"
    round_trip_cmpnts[0] = "changed"
    assert str(term_set.to_terms().into(Strings)) == "alpha, beta"


def test_single_container_into_shape_conversions():
    cmpnt = String("alpha")
    term = cmpnt.into(ComplexMockTerm)
    assert type(term) is ComplexMockTerm
    assert term.cmpnt == cmpnt
    assert term.coeff == 1 + 0j
    assert not term.cmpnt.aliases(cmpnt)

    converted_cmpnt = term.into(String)
    assert type(converted_cmpnt) is String
    assert converted_cmpnt == cmpnt
    converted_cmpnt.set("changed")
    assert term.cmpnt == cmpnt


def test_terms_into_converts_coefficients_and_clones_once_at_surface():
    terms = RealMockTerms(
        TermData(
            Strings.from_iterable(("alpha", "beta")),
            RealCoeffs.from_sequence((1.5, -2.0)),
        )
    )
    complex_terms = terms.into(ComplexMockTerms)
    assert type(complex_terms) is ComplexMockTerms
    assert complex_terms.cmpnts == terms.cmpnts
    assert tuple(complex_terms.coeffs) == (1.5 + 0j, -2.0 + 0j)
    assert complex_terms.cmpnts is not terms.cmpnts

    complex_terms.coeffs[0] = 1.5j
    with pytest.raises(ValueError):
        complex_terms.into(RealMockTerms)


def test_into_rejects_unsupported_targets():
    cmpnts = Strings.from_iterable(("alpha", "beta"))
    with pytest.raises(TypeError):
        cmpnts.into(dict)
    with pytest.raises(TypeError):
        cmpnts.into(StringSet)
    assert not hasattr(StringSet, "into")
