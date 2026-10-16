from typing import assert_type

import zixy.fermion.operator.general as fermion_general
import zixy.fermion.operator.normal as fermion_normal
import zixy.fermion.state as fermion_state
import zixy.qubit.operator as qubit_operator
import zixy.qubit.state as qubit_state


def test_qubit_operator_relationships() -> None:
    strings = qubit_operator.Strings(2, 1)
    terms = qubit_operator.RealTerms(2, 1)
    total = qubit_operator.RealTermSum.from_terms(terms)
    string_set = qubit_operator.StringSet.from_strings(strings)

    assert_type(strings[0], qubit_operator.String)
    assert_type(strings[:], qubit_operator.Strings)
    assert_type(next(iter(strings)), qubit_operator.String)
    assert_type(terms[0], qubit_operator.RealTerm)
    assert_type(terms[:], qubit_operator.RealTerms)
    assert_type(next(iter(terms)), qubit_operator.RealTerm)
    assert_type(terms[0].string, qubit_operator.String)
    assert_type(terms[0].string_type, type[qubit_operator.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, qubit_operator.Strings)
    assert_type(terms.strings_type, type[qubit_operator.Strings])
    assert_type(terms.string_type, type[qubit_operator.String])
    assert_type(total.strings_type, type[qubit_operator.Strings])
    assert_type(total.string_type, type[qubit_operator.String])
    assert_type(qubit_operator.ComplexTerms(2, 1)[0], qubit_operator.ComplexTerm)
    assert_type(qubit_operator.SymbolicTerms(2, 1)[0], qubit_operator.SymbolicTerm)
    assert_type(total.strings, qubit_operator.Strings)
    assert_type(next(iter(total)), qubit_operator.RealTerm)
    assert_type(string_set.to_strings(), qubit_operator.Strings)
    assert_type(next(iter(string_set)), qubit_operator.String)

    assert isinstance(next(iter(strings)), qubit_operator.String)
    assert isinstance(next(iter(terms)), qubit_operator.RealTerm)
    assert isinstance(next(iter(total)), qubit_operator.RealTerm)
    assert isinstance(next(iter(string_set)), qubit_operator.String)
    assert isinstance(terms.strings, qubit_operator.Strings)
    assert isinstance(total.strings, qubit_operator.Strings)
    assert isinstance(string_set.to_strings(), qubit_operator.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])


def test_qubit_state_relationships() -> None:
    strings = qubit_state.Strings(2, 1)
    terms = qubit_state.RealTerms(2, 1)
    total = qubit_state.RealTermSum.from_terms(terms)
    string_set = qubit_state.StringSet.from_strings(strings)

    assert_type(strings[0], qubit_state.String)
    assert_type(strings[:], qubit_state.Strings)
    assert_type(next(iter(strings)), qubit_state.String)
    assert_type(terms[0], qubit_state.RealTerm)
    assert_type(terms[:], qubit_state.RealTerms)
    assert_type(next(iter(terms)), qubit_state.RealTerm)
    assert_type(terms[0].string, qubit_state.String)
    assert_type(terms[0].string_type, type[qubit_state.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, qubit_state.Strings)
    assert_type(terms.strings_type, type[qubit_state.Strings])
    assert_type(terms.string_type, type[qubit_state.String])
    assert_type(total.strings_type, type[qubit_state.Strings])
    assert_type(total.string_type, type[qubit_state.String])
    assert_type(qubit_state.ComplexTerms(2, 1)[0], qubit_state.ComplexTerm)
    assert_type(qubit_state.SymbolicTerms(2, 1)[0], qubit_state.SymbolicTerm)
    assert_type(total.strings, qubit_state.Strings)
    assert_type(next(iter(total)), qubit_state.RealTerm)
    assert_type(string_set.to_strings(), qubit_state.Strings)
    assert_type(next(iter(string_set)), qubit_state.String)

    assert isinstance(next(iter(strings)), qubit_state.String)
    assert isinstance(next(iter(terms)), qubit_state.RealTerm)
    assert isinstance(next(iter(total)), qubit_state.RealTerm)
    assert isinstance(next(iter(string_set)), qubit_state.String)
    assert isinstance(terms.strings, qubit_state.Strings)
    assert isinstance(total.strings, qubit_state.Strings)
    assert isinstance(string_set.to_strings(), qubit_state.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_general_relationships() -> None:
    strings = fermion_general.Strings(2, 1)
    terms = fermion_general.RealTerms(2, 1)
    total = fermion_general.RealTermSum.from_terms(terms)
    string_set = fermion_general.StringSet.from_strings(strings)

    assert_type(strings[0], fermion_general.String)
    assert_type(strings[:], fermion_general.Strings)
    assert_type(next(iter(strings)), fermion_general.String)
    assert_type(terms[0], fermion_general.RealTerm)
    assert_type(terms[:], fermion_general.RealTerms)
    assert_type(next(iter(terms)), fermion_general.RealTerm)
    assert_type(terms[0].string, fermion_general.String)
    assert_type(terms[0].string_type, type[fermion_general.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, fermion_general.Strings)
    assert_type(terms.strings_type, type[fermion_general.Strings])
    assert_type(terms.string_type, type[fermion_general.String])
    assert_type(total.strings_type, type[fermion_general.Strings])
    assert_type(total.string_type, type[fermion_general.String])
    assert_type(fermion_general.ComplexTerms(2, 1)[0], fermion_general.ComplexTerm)
    assert_type(fermion_general.SymbolicTerms(2, 1)[0], fermion_general.SymbolicTerm)
    assert_type(total.strings, fermion_general.Strings)
    assert_type(next(iter(total)), fermion_general.RealTerm)
    assert_type(string_set.to_strings(), fermion_general.Strings)
    assert_type(next(iter(string_set)), fermion_general.String)

    assert isinstance(next(iter(strings)), fermion_general.String)
    assert isinstance(next(iter(terms)), fermion_general.RealTerm)
    assert isinstance(next(iter(total)), fermion_general.RealTerm)
    assert isinstance(next(iter(string_set)), fermion_general.String)
    assert isinstance(terms.strings, fermion_general.Strings)
    assert isinstance(total.strings, fermion_general.Strings)
    assert isinstance(string_set.to_strings(), fermion_general.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_normal_relationships() -> None:
    strings = fermion_normal.Strings(2, 1)
    terms = fermion_normal.RealTerms(2, 1)
    total = fermion_normal.RealTermSum.from_terms(terms)
    string_set = fermion_normal.StringSet.from_strings(strings)

    assert_type(strings[0], fermion_normal.String)
    assert_type(strings[:], fermion_normal.Strings)
    assert_type(next(iter(strings)), fermion_normal.String)
    assert_type(terms[0], fermion_normal.RealTerm)
    assert_type(terms[:], fermion_normal.RealTerms)
    assert_type(next(iter(terms)), fermion_normal.RealTerm)
    assert_type(terms[0].string, fermion_normal.String)
    assert_type(terms[0].string_type, type[fermion_normal.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, fermion_normal.Strings)
    assert_type(terms.strings_type, type[fermion_normal.Strings])
    assert_type(terms.string_type, type[fermion_normal.String])
    assert_type(total.strings_type, type[fermion_normal.Strings])
    assert_type(total.string_type, type[fermion_normal.String])
    assert_type(fermion_normal.ComplexTerms(2, 1)[0], fermion_normal.ComplexTerm)
    assert_type(fermion_normal.SymbolicTerms(2, 1)[0], fermion_normal.SymbolicTerm)
    assert_type(total.strings, fermion_normal.Strings)
    assert_type(next(iter(total)), fermion_normal.RealTerm)
    assert_type(string_set.to_strings(), fermion_normal.Strings)
    assert_type(next(iter(string_set)), fermion_normal.String)

    assert isinstance(next(iter(strings)), fermion_normal.String)
    assert isinstance(next(iter(terms)), fermion_normal.RealTerm)
    assert isinstance(next(iter(total)), fermion_normal.RealTerm)
    assert isinstance(next(iter(string_set)), fermion_normal.String)
    assert isinstance(terms.strings, fermion_normal.Strings)
    assert isinstance(total.strings, fermion_normal.Strings)
    assert isinstance(string_set.to_strings(), fermion_normal.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_state_relationships() -> None:
    strings = fermion_state.Strings(2, 1)
    terms = fermion_state.RealTerms(2, 1)
    total = fermion_state.RealTermSum.from_terms(terms)
    string_set = fermion_state.StringSet.from_strings(strings)

    assert_type(strings[0], fermion_state.String)
    assert_type(strings[:], fermion_state.Strings)
    assert_type(next(iter(strings)), fermion_state.String)
    assert_type(terms[0], fermion_state.RealTerm)
    assert_type(terms[:], fermion_state.RealTerms)
    assert_type(next(iter(terms)), fermion_state.RealTerm)
    assert_type(terms[0].string, fermion_state.String)
    assert_type(terms[0].string_type, type[fermion_state.String])
    assert_type(terms[0].coeff, float)
    assert_type(terms.strings, fermion_state.Strings)
    assert_type(terms.strings_type, type[fermion_state.Strings])
    assert_type(terms.string_type, type[fermion_state.String])
    assert_type(total.strings_type, type[fermion_state.Strings])
    assert_type(total.string_type, type[fermion_state.String])
    assert_type(fermion_state.ComplexTerms(2, 1)[0], fermion_state.ComplexTerm)
    assert_type(fermion_state.SymbolicTerms(2, 1)[0], fermion_state.SymbolicTerm)
    assert_type(total.strings, fermion_state.Strings)
    assert_type(next(iter(total)), fermion_state.RealTerm)
    assert_type(string_set.to_strings(), fermion_state.Strings)
    assert_type(next(iter(string_set)), fermion_state.String)

    assert isinstance(next(iter(strings)), fermion_state.String)
    assert isinstance(next(iter(terms)), fermion_state.RealTerm)
    assert isinstance(next(iter(total)), fermion_state.RealTerm)
    assert isinstance(next(iter(string_set)), fermion_state.String)
    assert isinstance(terms.strings, fermion_state.Strings)
    assert isinstance(total.strings, fermion_state.Strings)
    assert isinstance(string_set.to_strings(), fermion_state.Strings)

    # Accessors must preserve views of the original data, including sliced term arrays.
    terms[:][0].coeff = 2.0
    assert terms[0].coeff == 2.0
    assert terms[0].string.aliases(terms.strings[0])
