from __future__ import annotations

from typing import assert_type

import zixy.fermion.operator.general as fermion_general
import zixy.fermion.operator.normal as fermion_normal
import zixy.fermion.state as fermion_state
import zixy.qubit.operator as qubit_operator
import zixy.qubit.state as qubit_state


def test_qubit_operator_inherited_factories() -> None:
    class DerivedTerms(qubit_operator.RealTerms):
        pass

    class DerivedSum(qubit_operator.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], qubit_operator.RealTerm)
    assert_type(next(iter(terms)), qubit_operator.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, qubit_operator.String)
    assert_type(terms.strings, qubit_operator.Strings)
    assert_type(terms.strings[0], qubit_operator.String)
    assert_type(total.strings, qubit_operator.Strings)
    assert_type(next(iter(total)), qubit_operator.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is qubit_operator.RealTerm
    assert type(total.strings) is qubit_operator.Strings


def test_qubit_operator_overridden_factories() -> None:
    class CustomString(qubit_operator.String):
        pass

    class CustomStrings(qubit_operator.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

    class CustomStringSet(qubit_operator.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(qubit_operator.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(qubit_operator.RealTerms):
        term_type = CustomTerm

    class CustomSum(qubit_operator.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    strings = CustomStrings(2, 1)
    string_set = CustomStringSet.from_strings(strings)
    terms = DerivedTerms(2, 1)
    total = CustomSum.from_terms(terms)

    assert_type(strings[0], CustomString)
    assert_type(strings[:], CustomStrings)
    assert_type(next(iter(strings)), CustomString)
    assert_type(string_set.to_strings(), CustomStrings)
    assert_type(next(iter(string_set)), CustomString)
    assert_type(terms[0], CustomTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(next(iter(terms)), CustomTerm)
    assert_type(terms[0].string, CustomString)
    assert_type(terms[0].string_type, type[CustomString])
    assert_type(terms.strings, CustomStrings)
    assert_type(terms.strings[0], CustomString)
    assert_type(terms.strings_type, type[CustomStrings])
    assert_type(terms.string_type, type[CustomString])
    assert_type(total.strings, CustomStrings)
    assert_type(next(iter(total)), CustomTerm)
    assert_type(total.strings_type, type[CustomStrings])
    assert_type(total.string_type, type[CustomString])

    assert type(strings[0]) is CustomString
    assert type(strings[:]) is CustomStrings
    assert type(next(iter(strings))) is CustomString
    assert type(string_set.to_strings()) is CustomStrings
    assert type(next(iter(string_set))) is CustomString
    assert type(terms[0]) is CustomTerm
    assert type(terms[:]) is DerivedTerms
    assert type(next(iter(terms))) is CustomTerm
    assert type(terms[0].string) is CustomString
    assert type(terms.strings) is CustomStrings
    assert type(total.strings) is CustomStrings
    assert type(next(iter(total))) is CustomTerm

    terms[:][0].coeff = 3.0
    assert terms[0].coeff == 3.0
    assert terms[0].string.aliases(terms.strings[0])


def test_qubit_state_inherited_factories() -> None:
    class DerivedTerms(qubit_state.RealTerms):
        pass

    class DerivedSum(qubit_state.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], qubit_state.RealTerm)
    assert_type(next(iter(terms)), qubit_state.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, qubit_state.String)
    assert_type(terms.strings, qubit_state.Strings)
    assert_type(terms.strings[0], qubit_state.String)
    assert_type(total.strings, qubit_state.Strings)
    assert_type(next(iter(total)), qubit_state.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is qubit_state.RealTerm
    assert type(total.strings) is qubit_state.Strings


def test_qubit_state_overridden_factories() -> None:
    class CustomString(qubit_state.String):
        pass

    class CustomStrings(qubit_state.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

    class CustomStringSet(qubit_state.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(qubit_state.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(qubit_state.RealTerms):
        term_type = CustomTerm

    class CustomSum(qubit_state.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    strings = CustomStrings(2, 1)
    string_set = CustomStringSet.from_strings(strings)
    terms = DerivedTerms(2, 1)
    total = CustomSum.from_terms(terms)

    assert_type(strings[0], CustomString)
    assert_type(strings[:], CustomStrings)
    assert_type(next(iter(strings)), CustomString)
    assert_type(string_set.to_strings(), CustomStrings)
    assert_type(next(iter(string_set)), CustomString)
    assert_type(terms[0], CustomTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(next(iter(terms)), CustomTerm)
    assert_type(terms[0].string, CustomString)
    assert_type(terms[0].string_type, type[CustomString])
    assert_type(terms.strings, CustomStrings)
    assert_type(terms.strings[0], CustomString)
    assert_type(terms.strings_type, type[CustomStrings])
    assert_type(terms.string_type, type[CustomString])
    assert_type(total.strings, CustomStrings)
    assert_type(next(iter(total)), CustomTerm)
    assert_type(total.strings_type, type[CustomStrings])
    assert_type(total.string_type, type[CustomString])

    assert type(strings[0]) is CustomString
    assert type(strings[:]) is CustomStrings
    assert type(next(iter(strings))) is CustomString
    assert type(string_set.to_strings()) is CustomStrings
    assert type(next(iter(string_set))) is CustomString
    assert type(terms[0]) is CustomTerm
    assert type(terms[:]) is DerivedTerms
    assert type(next(iter(terms))) is CustomTerm
    assert type(terms[0].string) is CustomString
    assert type(terms.strings) is CustomStrings
    assert type(total.strings) is CustomStrings
    assert type(next(iter(total))) is CustomTerm

    terms[:][0].coeff = 3.0
    assert terms[0].coeff == 3.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_general_inherited_factories() -> None:
    class DerivedTerms(fermion_general.RealTerms):
        pass

    class DerivedSum(fermion_general.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], fermion_general.RealTerm)
    assert_type(next(iter(terms)), fermion_general.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, fermion_general.String)
    assert_type(terms.strings, fermion_general.Strings)
    assert_type(terms.strings[0], fermion_general.String)
    assert_type(total.strings, fermion_general.Strings)
    assert_type(next(iter(total)), fermion_general.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is fermion_general.RealTerm
    assert type(total.strings) is fermion_general.Strings


def test_fermion_general_overridden_factories() -> None:
    class CustomString(fermion_general.String):
        pass

    class CustomStrings(fermion_general.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

    class CustomStringSet(fermion_general.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(fermion_general.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(fermion_general.RealTerms):
        term_type = CustomTerm

    class CustomSum(fermion_general.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    strings = CustomStrings(2, 1)
    string_set = CustomStringSet.from_strings(strings)
    terms = DerivedTerms(2, 1)
    total = CustomSum.from_terms(terms)

    assert_type(strings[0], CustomString)
    assert_type(strings[:], CustomStrings)
    assert_type(next(iter(strings)), CustomString)
    assert_type(string_set.to_strings(), CustomStrings)
    assert_type(next(iter(string_set)), CustomString)
    assert_type(terms[0], CustomTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(next(iter(terms)), CustomTerm)
    assert_type(terms[0].string, CustomString)
    assert_type(terms[0].string_type, type[CustomString])
    assert_type(terms.strings, CustomStrings)
    assert_type(terms.strings[0], CustomString)
    assert_type(terms.strings_type, type[CustomStrings])
    assert_type(terms.string_type, type[CustomString])
    assert_type(total.strings, CustomStrings)
    assert_type(next(iter(total)), CustomTerm)
    assert_type(total.strings_type, type[CustomStrings])
    assert_type(total.string_type, type[CustomString])

    assert type(strings[0]) is CustomString
    assert type(strings[:]) is CustomStrings
    assert type(next(iter(strings))) is CustomString
    assert type(string_set.to_strings()) is CustomStrings
    assert type(next(iter(string_set))) is CustomString
    assert type(terms[0]) is CustomTerm
    assert type(terms[:]) is DerivedTerms
    assert type(next(iter(terms))) is CustomTerm
    assert type(terms[0].string) is CustomString
    assert type(terms.strings) is CustomStrings
    assert type(total.strings) is CustomStrings
    assert type(next(iter(total))) is CustomTerm

    terms[:][0].coeff = 3.0
    assert terms[0].coeff == 3.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_normal_inherited_factories() -> None:
    class DerivedTerms(fermion_normal.RealTerms):
        pass

    class DerivedSum(fermion_normal.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], fermion_normal.RealTerm)
    assert_type(next(iter(terms)), fermion_normal.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, fermion_normal.String)
    assert_type(terms.strings, fermion_normal.Strings)
    assert_type(terms.strings[0], fermion_normal.String)
    assert_type(total.strings, fermion_normal.Strings)
    assert_type(next(iter(total)), fermion_normal.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is fermion_normal.RealTerm
    assert type(total.strings) is fermion_normal.Strings


def test_fermion_normal_overridden_factories() -> None:
    class CustomString(fermion_normal.String):
        pass

    class CustomStrings(fermion_normal.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

    class CustomStringSet(fermion_normal.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(fermion_normal.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(fermion_normal.RealTerms):
        term_type = CustomTerm

    class CustomSum(fermion_normal.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    strings = CustomStrings(2, 1)
    string_set = CustomStringSet.from_strings(strings)
    terms = DerivedTerms(2, 1)
    total = CustomSum.from_terms(terms)

    assert_type(strings[0], CustomString)
    assert_type(strings[:], CustomStrings)
    assert_type(next(iter(strings)), CustomString)
    assert_type(string_set.to_strings(), CustomStrings)
    assert_type(next(iter(string_set)), CustomString)
    assert_type(terms[0], CustomTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(next(iter(terms)), CustomTerm)
    assert_type(terms[0].string, CustomString)
    assert_type(terms[0].string_type, type[CustomString])
    assert_type(terms.strings, CustomStrings)
    assert_type(terms.strings[0], CustomString)
    assert_type(terms.strings_type, type[CustomStrings])
    assert_type(terms.string_type, type[CustomString])
    assert_type(total.strings, CustomStrings)
    assert_type(next(iter(total)), CustomTerm)
    assert_type(total.strings_type, type[CustomStrings])
    assert_type(total.string_type, type[CustomString])

    assert type(strings[0]) is CustomString
    assert type(strings[:]) is CustomStrings
    assert type(next(iter(strings))) is CustomString
    assert type(string_set.to_strings()) is CustomStrings
    assert type(next(iter(string_set))) is CustomString
    assert type(terms[0]) is CustomTerm
    assert type(terms[:]) is DerivedTerms
    assert type(next(iter(terms))) is CustomTerm
    assert type(terms[0].string) is CustomString
    assert type(terms.strings) is CustomStrings
    assert type(total.strings) is CustomStrings
    assert type(next(iter(total))) is CustomTerm

    terms[:][0].coeff = 3.0
    assert terms[0].coeff == 3.0
    assert terms[0].string.aliases(terms.strings[0])


def test_fermion_state_inherited_factories() -> None:
    class DerivedTerms(fermion_state.RealTerms):
        pass

    class DerivedSum(fermion_state.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], fermion_state.RealTerm)
    assert_type(next(iter(terms)), fermion_state.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, fermion_state.String)
    assert_type(terms.strings, fermion_state.Strings)
    assert_type(terms.strings[0], fermion_state.String)
    assert_type(total.strings, fermion_state.Strings)
    assert_type(next(iter(total)), fermion_state.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is fermion_state.RealTerm
    assert type(total.strings) is fermion_state.Strings


def test_fermion_state_overridden_factories() -> None:
    class CustomString(fermion_state.String):
        pass

    class CustomStrings(fermion_state.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

    class CustomStringSet(fermion_state.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(fermion_state.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(fermion_state.RealTerms):
        term_type = CustomTerm

    class CustomSum(fermion_state.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    strings = CustomStrings(2, 1)
    string_set = CustomStringSet.from_strings(strings)
    terms = DerivedTerms(2, 1)
    total = CustomSum.from_terms(terms)

    assert_type(strings[0], CustomString)
    assert_type(strings[:], CustomStrings)
    assert_type(next(iter(strings)), CustomString)
    assert_type(string_set.to_strings(), CustomStrings)
    assert_type(next(iter(string_set)), CustomString)
    assert_type(terms[0], CustomTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(next(iter(terms)), CustomTerm)
    assert_type(terms[0].string, CustomString)
    assert_type(terms[0].string_type, type[CustomString])
    assert_type(terms.strings, CustomStrings)
    assert_type(terms.strings[0], CustomString)
    assert_type(terms.strings_type, type[CustomStrings])
    assert_type(terms.string_type, type[CustomString])
    assert_type(total.strings, CustomStrings)
    assert_type(next(iter(total)), CustomTerm)
    assert_type(total.strings_type, type[CustomStrings])
    assert_type(total.string_type, type[CustomString])

    assert type(strings[0]) is CustomString
    assert type(strings[:]) is CustomStrings
    assert type(next(iter(strings))) is CustomString
    assert type(string_set.to_strings()) is CustomStrings
    assert type(next(iter(string_set))) is CustomString
    assert type(terms[0]) is CustomTerm
    assert type(terms[:]) is DerivedTerms
    assert type(next(iter(terms))) is CustomTerm
    assert type(terms[0].string) is CustomString
    assert type(terms.strings) is CustomStrings
    assert type(total.strings) is CustomStrings
    assert type(next(iter(total))) is CustomTerm

    terms[:][0].coeff = 3.0
    assert terms[0].coeff == 3.0
    assert terms[0].string.aliases(terms.strings[0])
