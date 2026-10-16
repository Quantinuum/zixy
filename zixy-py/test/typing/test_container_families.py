from typing import assert_type

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs


def test_qubit_operator_relationships() -> None:
    strings = zqo.Strings(2, 1)
    terms = zqo.RealTerms(2, 1)
    total = zqo.RealTermSum.from_terms(terms)
    string_set = zqo.StringSet.from_strings(strings)

    assert_type(strings[0], zqo.String)
    assert_type(strings[:], zqo.Strings)
    assert_type(next(iter(strings)), zqo.String)
    assert_type(terms[0], zqo.RealTerm)
    assert_type(terms[:], zqo.RealTerms)
    assert_type(next(iter(terms)), zqo.RealTerm)
    assert_type(terms[0].string, zqo.String)
    assert_type(terms[0].string_type, type[zqo.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, zqo.Strings)
    assert_type(terms.strings_type, type[zqo.Strings])
    assert_type(terms.string_type, type[zqo.String])
    assert_type(total.strings_type, type[zqo.Strings])
    assert_type(total.string_type, type[zqo.String])
    assert_type(zqo.ComplexTerms(2, 1)[0], zqo.ComplexTerm)
    assert_type(zqo.SymbolicTerms(2, 1)[0], zqo.SymbolicTerm)
    assert_type(total.strings, zqo.Strings)
    assert_type(next(iter(total)), zqo.RealTerm)
    assert_type(string_set.to_strings(), zqo.Strings)
    assert_type(next(iter(string_set)), zqo.String)

    assert isinstance(next(iter(strings)), zqo.String)
    assert isinstance(next(iter(terms)), zqo.RealTerm)
    assert isinstance(next(iter(total)), zqo.RealTerm)
    assert isinstance(next(iter(string_set)), zqo.String)
    assert isinstance(terms.strings, zqo.Strings)
    assert isinstance(total.strings, zqo.Strings)
    assert isinstance(string_set.to_strings(), zqo.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])


def test_qubit_state_relationships() -> None:
    strings = zqs.Strings(2, 1)
    terms = zqs.RealTerms(2, 1)
    total = zqs.RealTermSum.from_terms(terms)
    string_set = zqs.StringSet.from_strings(strings)

    assert_type(strings[0], zqs.String)
    assert_type(strings[:], zqs.Strings)
    assert_type(next(iter(strings)), zqs.String)
    assert_type(terms[0], zqs.RealTerm)
    assert_type(terms[:], zqs.RealTerms)
    assert_type(next(iter(terms)), zqs.RealTerm)
    assert_type(terms[0].string, zqs.String)
    assert_type(terms[0].string_type, type[zqs.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, zqs.Strings)
    assert_type(terms.strings_type, type[zqs.Strings])
    assert_type(terms.string_type, type[zqs.String])
    assert_type(total.strings_type, type[zqs.Strings])
    assert_type(total.string_type, type[zqs.String])
    assert_type(zqs.ComplexTerms(2, 1)[0], zqs.ComplexTerm)
    assert_type(zqs.SymbolicTerms(2, 1)[0], zqs.SymbolicTerm)
    assert_type(total.strings, zqs.Strings)
    assert_type(next(iter(total)), zqs.RealTerm)
    assert_type(string_set.to_strings(), zqs.Strings)
    assert_type(next(iter(string_set)), zqs.String)

    assert isinstance(next(iter(strings)), zqs.String)
    assert isinstance(next(iter(terms)), zqs.RealTerm)
    assert isinstance(next(iter(total)), zqs.RealTerm)
    assert isinstance(next(iter(string_set)), zqs.String)
    assert isinstance(terms.strings, zqs.Strings)
    assert isinstance(total.strings, zqs.Strings)
    assert isinstance(string_set.to_strings(), zqs.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_general_relationships() -> None:
    strings = zfog.Strings(2, 1)
    terms = zfog.RealTerms(2, 1)
    total = zfog.RealTermSum.from_terms(terms)
    string_set = zfog.StringSet.from_strings(strings)

    assert_type(strings[0], zfog.String)
    assert_type(strings[:], zfog.Strings)
    assert_type(next(iter(strings)), zfog.String)
    assert_type(terms[0], zfog.RealTerm)
    assert_type(terms[:], zfog.RealTerms)
    assert_type(next(iter(terms)), zfog.RealTerm)
    assert_type(terms[0].string, zfog.String)
    assert_type(terms[0].string_type, type[zfog.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, zfog.Strings)
    assert_type(terms.strings_type, type[zfog.Strings])
    assert_type(terms.string_type, type[zfog.String])
    assert_type(total.strings_type, type[zfog.Strings])
    assert_type(total.string_type, type[zfog.String])
    assert_type(zfog.ComplexTerms(2, 1)[0], zfog.ComplexTerm)
    assert_type(zfog.SymbolicTerms(2, 1)[0], zfog.SymbolicTerm)
    assert_type(total.strings, zfog.Strings)
    assert_type(next(iter(total)), zfog.RealTerm)
    assert_type(string_set.to_strings(), zfog.Strings)
    assert_type(next(iter(string_set)), zfog.String)

    assert isinstance(next(iter(strings)), zfog.String)
    assert isinstance(next(iter(terms)), zfog.RealTerm)
    assert isinstance(next(iter(total)), zfog.RealTerm)
    assert isinstance(next(iter(string_set)), zfog.String)
    assert isinstance(terms.strings, zfog.Strings)
    assert isinstance(total.strings, zfog.Strings)
    assert isinstance(string_set.to_strings(), zfog.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_normal_relationships() -> None:
    strings = zfon.Strings(2, 1)
    terms = zfon.RealTerms(2, 1)
    total = zfon.RealTermSum.from_terms(terms)
    string_set = zfon.StringSet.from_strings(strings)

    assert_type(strings[0], zfon.String)
    assert_type(strings[:], zfon.Strings)
    assert_type(next(iter(strings)), zfon.String)
    assert_type(terms[0], zfon.RealTerm)
    assert_type(terms[:], zfon.RealTerms)
    assert_type(next(iter(terms)), zfon.RealTerm)
    assert_type(terms[0].string, zfon.String)
    assert_type(terms[0].string_type, type[zfon.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, zfon.Strings)
    assert_type(terms.strings_type, type[zfon.Strings])
    assert_type(terms.string_type, type[zfon.String])
    assert_type(total.strings_type, type[zfon.Strings])
    assert_type(total.string_type, type[zfon.String])
    assert_type(zfon.ComplexTerms(2, 1)[0], zfon.ComplexTerm)
    assert_type(zfon.SymbolicTerms(2, 1)[0], zfon.SymbolicTerm)
    assert_type(total.strings, zfon.Strings)
    assert_type(next(iter(total)), zfon.RealTerm)
    assert_type(string_set.to_strings(), zfon.Strings)
    assert_type(next(iter(string_set)), zfon.String)

    assert isinstance(next(iter(strings)), zfon.String)
    assert isinstance(next(iter(terms)), zfon.RealTerm)
    assert isinstance(next(iter(total)), zfon.RealTerm)
    assert isinstance(next(iter(string_set)), zfon.String)
    assert isinstance(terms.strings, zfon.Strings)
    assert isinstance(total.strings, zfon.Strings)
    assert isinstance(string_set.to_strings(), zfon.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_state_relationships() -> None:
    strings = zfs.Strings(2, 1)
    terms = zfs.RealTerms(2, 1)
    total = zfs.RealTermSum.from_terms(terms)
    string_set = zfs.StringSet.from_strings(strings)

    assert_type(strings[0], zfs.String)
    assert_type(strings[:], zfs.Strings)
    assert_type(next(iter(strings)), zfs.String)
    assert_type(terms[0], zfs.RealTerm)
    assert_type(terms[:], zfs.RealTerms)
    assert_type(next(iter(terms)), zfs.RealTerm)
    assert_type(terms[0].string, zfs.String)
    assert_type(terms[0].string_type, type[zfs.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, zfs.Strings)
    assert_type(terms.strings_type, type[zfs.Strings])
    assert_type(terms.string_type, type[zfs.String])
    assert_type(total.strings_type, type[zfs.Strings])
    assert_type(total.string_type, type[zfs.String])
    assert_type(zfs.ComplexTerms(2, 1)[0], zfs.ComplexTerm)
    assert_type(zfs.SymbolicTerms(2, 1)[0], zfs.SymbolicTerm)
    assert_type(total.strings, zfs.Strings)
    assert_type(next(iter(total)), zfs.RealTerm)
    assert_type(string_set.to_strings(), zfs.Strings)
    assert_type(next(iter(string_set)), zfs.String)

    assert isinstance(next(iter(strings)), zfs.String)
    assert isinstance(next(iter(terms)), zfs.RealTerm)
    assert isinstance(next(iter(total)), zfs.RealTerm)
    assert isinstance(next(iter(string_set)), zfs.String)
    assert isinstance(terms.strings, zfs.Strings)
    assert isinstance(total.strings, zfs.Strings)
    assert isinstance(string_set.to_strings(), zfs.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])
