from __future__ import annotations

from collections.abc import Iterator
from typing import Self, assert_type

import pytest

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs


def test_qubit_operator_inherited_factories() -> None:
    class DerivedTerms(zqo.RealTerms):
        pass

    class DerivedSum(zqo.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], zqo.RealTerm)
    assert_type(next(iter(terms)), zqo.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, zqo.String)
    assert_type(terms.strings, zqo.Strings)
    assert_type(terms.strings[0], zqo.String)
    assert_type(total.strings, zqo.Strings)
    assert_type(next(iter(total)), zqo.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is zqo.RealTerm
    assert type(total.strings) is zqo.Strings


def test_qubit_operator_overridden_factories() -> None:
    class CustomString(zqo.String):
        pass

    class CustomStrings(zqo.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

        def custom_marker(self) -> str:
            return "custom strings"

    class CustomStringSet(zqo.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(zqo.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(zqo.RealTerms):
        term_type = CustomTerm

    class CustomSum(zqo.RealTermSum):
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

    def keep(string: CustomString) -> bool:
        return string == strings[0]

    assert_type(strings.new_clear_string(), CustomString)
    assert_type(strings.iter_filter_non_identity(), Iterator[CustomString])
    assert_type(strings.filter_non_identity(), CustomStrings)
    assert_type(strings.iter_filter_populated(), Iterator[CustomString])
    assert_type(strings.iter_filter_map(keep), Iterator[CustomString])
    assert_type(strings.filter_map(keep), CustomStrings)
    assert_type(strings.filter_unique(), CustomStrings)
    assert_type(strings.filter_populated(), CustomStrings)
    assert_type(string_set.iter_filter_map(keep), Iterator[CustomString])
    assert_type(string_set.filter_map(keep), CustomStringSet)
    strings.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    string_set.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    assert type(strings.new_clear_string()) is CustomString
    assert type(next(strings.iter_filter_map(keep))) is CustomString
    assert type(strings.filter_map(keep)) is CustomStrings
    assert type(strings.filter_unique()) is CustomStrings
    assert type(string_set.filter_map(keep)) is CustomStringSet

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

    # The inferred custom API must exist at runtime for correctly configured storage.
    assert_type(total.strings.custom_marker(), str)
    assert total.strings.custom_marker() == "custom strings"
    adopted = CustomSum._create(total._impl, total._string_set._map)
    assert adopted.strings.custom_marker() == "custom strings"

    class CustomSet(zqo.RealTermSet):
        terms_type = CustomTerms

    custom_set = CustomSet.from_terms(terms)
    assert_type(custom_set.strings, CustomStrings)
    assert custom_set.strings.custom_marker() == "custom strings"

    # Compatible native data is insufficient when its Python wrapper lacks the promised API.
    ordinary = zqo.RealTerms(2, 1)
    ordinary_sum = zqo.RealTermSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomTerms._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSet.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary_sum._impl, ordinary_sum._string_set._map)
    assert len(ordinary) == 1

    # Specialised factories accept compatible parent-class inputs, too.
    parent = zqo.String(2, "X0")
    copied = CustomStrings.from_string(parent)
    assert_type(copied, CustomStrings)
    assert type(copied[0]) is CustomString
    assert copied[0] == parent
    copied.append(parent)
    assert copied[1] == parent
    copied[0].set(parent)
    parent_term = zqo.RealTerm(2, (parent, 2.0))
    custom_term = CustomTerm.from_string_coeff(parent, 2.0)
    assert_type(custom_term, CustomTerm)
    assert custom_term == parent_term
    custom_term.set(parent_term)
    terms.append(parent_term)
    total.insert(parent_term)
    assert total.lookup_coeff(parent) == 2.0
    assert parent == zqo.String(2, "X0")


def test_qubit_state_inherited_factories() -> None:
    class DerivedTerms(zqs.RealTerms):
        pass

    class DerivedSum(zqs.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], zqs.RealTerm)
    assert_type(next(iter(terms)), zqs.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, zqs.String)
    assert_type(terms.strings, zqs.Strings)
    assert_type(terms.strings[0], zqs.String)
    assert_type(total.strings, zqs.Strings)
    assert_type(next(iter(total)), zqs.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is zqs.RealTerm
    assert type(total.strings) is zqs.Strings


def test_qubit_state_overridden_factories() -> None:
    class CustomString(zqs.String):
        pass

    class CustomStrings(zqs.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

        def custom_marker(self) -> str:
            return "custom strings"

    class CustomStringSet(zqs.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(zqs.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(zqs.RealTerms):
        term_type = CustomTerm

    class CustomSum(zqs.RealTermSum):
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

    def keep(string: CustomString) -> bool:
        return string == strings[0]

    assert_type(strings.new_clear_string(), CustomString)
    assert_type(strings.iter_filter_populated(), Iterator[CustomString])
    assert_type(strings.iter_filter_map(keep), Iterator[CustomString])
    assert_type(strings.filter_map(keep), CustomStrings)
    assert_type(strings.filter_unique(), CustomStrings)
    assert_type(strings.filter_populated(), CustomStrings)
    assert_type(string_set.iter_filter_map(keep), Iterator[CustomString])
    assert_type(string_set.filter_map(keep), CustomStringSet)
    strings.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    string_set.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    assert type(strings.new_clear_string()) is CustomString
    assert type(next(strings.iter_filter_map(keep))) is CustomString
    assert type(strings.filter_map(keep)) is CustomStrings
    assert type(strings.filter_unique()) is CustomStrings
    assert type(string_set.filter_map(keep)) is CustomStringSet

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

    # The inferred custom API must exist at runtime for correctly configured storage.
    assert_type(total.strings.custom_marker(), str)
    assert total.strings.custom_marker() == "custom strings"
    adopted = CustomSum._create(total._impl, total._string_set._map)
    assert adopted.strings.custom_marker() == "custom strings"

    class CustomSet(zqs.RealTermSet):
        terms_type = CustomTerms

    custom_set = CustomSet.from_terms(terms)
    assert_type(custom_set.strings, CustomStrings)
    assert custom_set.strings.custom_marker() == "custom strings"

    # Compatible native data is insufficient when its Python wrapper lacks the promised API.
    ordinary = zqs.RealTerms(2, 1)
    ordinary_sum = zqs.RealTermSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomTerms._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSet.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary_sum._impl, ordinary_sum._string_set._map)
    assert len(ordinary) == 1

    # Specialised factories accept compatible parent-class inputs, too.
    parent = zqs.String(2, "[1, 0]")
    copied = CustomStrings.from_string(parent)
    assert_type(copied, CustomStrings)
    assert type(copied[0]) is CustomString
    assert copied[0] == parent
    copied.append(parent)
    assert copied[1] == parent
    copied[0].set(parent)
    parent_term = zqs.RealTerm(2, (parent, 2.0))
    custom_term = CustomTerm.from_string_coeff(parent, 2.0)
    assert_type(custom_term, CustomTerm)
    assert custom_term == parent_term
    custom_term.set(parent_term)
    terms.append(parent_term)
    total.insert(parent_term)
    assert total.lookup_coeff(parent) == 2.0
    assert parent == zqs.String(2, "[1, 0]")


def test_fermion_general_inherited_factories() -> None:
    class DerivedTerms(zfog.RealTerms):
        pass

    class DerivedSum(zfog.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], zfog.RealTerm)
    assert_type(next(iter(terms)), zfog.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, zfog.String)
    assert_type(terms.strings, zfog.Strings)
    assert_type(terms.strings[0], zfog.String)
    assert_type(total.strings, zfog.Strings)
    assert_type(next(iter(total)), zfog.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is zfog.RealTerm
    assert type(total.strings) is zfog.Strings


def test_fermion_general_overridden_factories() -> None:
    class CustomString(zfog.String):
        pass

    class CustomStrings(zfog.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

        def custom_marker(self) -> str:
            return "custom strings"

    class CustomStringSet(zfog.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(zfog.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(zfog.RealTerms):
        term_type = CustomTerm

    class CustomSum(zfog.RealTermSum):
        terms_type = CustomTerms

    class DerivedTerms(CustomTerms):
        pass

    strings = CustomStrings(2, 1, max_string_len=1)
    string_set = CustomStringSet.from_strings(strings)
    terms = DerivedTerms(2, 1, max_string_len=1)
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

    def keep(string: CustomString) -> bool:
        return string == strings[0]

    assert_type(strings.new_clear_string(), CustomString)
    assert_type(strings.iter_filter_populated(), Iterator[CustomString])
    assert_type(strings.iter_filter_map(keep), Iterator[CustomString])
    assert_type(strings.filter_map(keep), CustomStrings)
    assert_type(strings.filter_unique(), CustomStrings)
    assert_type(strings.filter_populated(), CustomStrings)
    assert_type(string_set.iter_filter_map(keep), Iterator[CustomString])
    assert_type(string_set.filter_map(keep), CustomStringSet)
    strings.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    string_set.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    assert type(strings.new_clear_string()) is CustomString
    assert type(next(strings.iter_filter_map(keep))) is CustomString
    assert type(strings.filter_map(keep)) is CustomStrings
    assert type(strings.filter_unique()) is CustomStrings
    assert type(string_set.filter_map(keep)) is CustomStringSet

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

    # The inferred custom API must exist at runtime for correctly configured storage.
    assert_type(total.strings.custom_marker(), str)
    assert total.strings.custom_marker() == "custom strings"
    adopted = CustomSum._create(total._impl, total._string_set._map)
    assert adopted.strings.custom_marker() == "custom strings"

    class CustomSet(zfog.RealTermSet):
        terms_type = CustomTerms

    custom_set = CustomSet.from_terms(terms)
    assert_type(custom_set.strings, CustomStrings)
    assert custom_set.strings.custom_marker() == "custom strings"

    # Compatible native data is insufficient when its Python wrapper lacks the promised API.
    ordinary = zfog.RealTerms(2, 1)
    ordinary_sum = zfog.RealTermSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomTerms._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSet.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary_sum._impl, ordinary_sum._string_set._map)
    assert len(ordinary) == 1

    # Specialised factories accept compatible parent-class inputs, too.
    parent = zfog.String(2, "F0^")
    copied = CustomStrings.from_string(parent)
    assert_type(copied, CustomStrings)
    assert type(copied[0]) is CustomString
    assert copied[0] == parent
    copied.append(parent)
    assert copied[1] == parent
    copied[0].set(parent)
    parent_term = zfog.RealTerm(2, (parent, 2.0))
    custom_term = CustomTerm.from_string_coeff(parent, 2.0)
    assert_type(custom_term, CustomTerm)
    assert custom_term == parent_term
    custom_term.set(parent_term)
    terms.append(parent_term)
    total.insert(parent_term)
    assert total.lookup_coeff(parent) == 2.0
    assert parent == zfog.String(2, "F0^")


def test_fermion_normal_inherited_factories() -> None:
    class DerivedTerms(zfon.RealTerms):
        pass

    class DerivedSum(zfon.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], zfon.RealTerm)
    assert_type(next(iter(terms)), zfon.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, zfon.String)
    assert_type(terms.strings, zfon.Strings)
    assert_type(terms.strings[0], zfon.String)
    assert_type(total.strings, zfon.Strings)
    assert_type(next(iter(total)), zfon.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is zfon.RealTerm
    assert type(total.strings) is zfon.Strings


def test_fermion_normal_overridden_factories() -> None:
    class CustomString(zfon.String):
        pass

    class CustomStrings(zfon.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

        def custom_marker(self) -> str:
            return "custom strings"

    class CustomStringSet(zfon.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(zfon.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(zfon.RealTerms):
        term_type = CustomTerm

    class CustomSum(zfon.RealTermSum):
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

    def keep(string: CustomString) -> bool:
        return string == strings[0]

    assert_type(strings.new_clear_string(), CustomString)
    assert_type(strings.iter_filter_populated(), Iterator[CustomString])
    assert_type(strings.iter_filter_map(keep), Iterator[CustomString])
    assert_type(strings.filter_map(keep), CustomStrings)
    assert_type(strings.filter_unique(), CustomStrings)
    assert_type(strings.filter_populated(), CustomStrings)
    assert_type(string_set.iter_filter_map(keep), Iterator[CustomString])
    assert_type(string_set.filter_map(keep), CustomStringSet)
    strings.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    string_set.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    assert type(strings.new_clear_string()) is CustomString
    assert type(next(strings.iter_filter_map(keep))) is CustomString
    assert type(strings.filter_map(keep)) is CustomStrings
    assert type(strings.filter_unique()) is CustomStrings
    assert type(string_set.filter_map(keep)) is CustomStringSet

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

    # The inferred custom API must exist at runtime for correctly configured storage.
    assert_type(total.strings.custom_marker(), str)
    assert total.strings.custom_marker() == "custom strings"
    adopted = CustomSum._create(total._impl, total._string_set._map)
    assert adopted.strings.custom_marker() == "custom strings"

    class CustomSet(zfon.RealTermSet):
        terms_type = CustomTerms

    custom_set = CustomSet.from_terms(terms)
    assert_type(custom_set.strings, CustomStrings)
    assert custom_set.strings.custom_marker() == "custom strings"

    # Compatible native data is insufficient when its Python wrapper lacks the promised API.
    ordinary = zfon.RealTerms(2, 1)
    ordinary_sum = zfon.RealTermSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomTerms._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSet.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary_sum._impl, ordinary_sum._string_set._map)
    assert len(ordinary) == 1

    # Specialised factories accept compatible parent-class inputs, too.
    parent = zfon.String(2, "F0^")
    copied = CustomStrings.from_string(parent)
    assert_type(copied, CustomStrings)
    assert type(copied[0]) is CustomString
    assert copied[0] == parent
    copied.append(parent)
    assert copied[1] == parent
    copied[0].set(parent)
    parent_term = zfon.RealTerm(2, (parent, 2.0))
    custom_term = CustomTerm.from_string_coeff(parent, 2.0)
    assert_type(custom_term, CustomTerm)
    assert custom_term == parent_term
    custom_term.set(parent_term)
    terms.append(parent_term)
    total.insert(parent_term)
    assert total.lookup_coeff(parent) == 2.0
    assert parent == zfon.String(2, "F0^")


def test_fermion_state_inherited_factories() -> None:
    class DerivedTerms(zfs.RealTerms):
        pass

    class DerivedSum(zfs.RealTermSum):
        pass

    terms = DerivedTerms(2, 1)
    total = DerivedSum.from_terms(terms)

    assert_type(terms[0], zfs.RealTerm)
    assert_type(next(iter(terms)), zfs.RealTerm)
    assert_type(terms[:], DerivedTerms)
    assert_type(terms[0].string, zfs.String)
    assert_type(terms.strings, zfs.Strings)
    assert_type(terms.strings[0], zfs.String)
    assert_type(total.strings, zfs.Strings)
    assert_type(next(iter(total)), zfs.RealTerm)
    assert type(terms[:]) is DerivedTerms
    assert type(terms[0]) is zfs.RealTerm
    assert type(total.strings) is zfs.Strings


def test_fermion_state_overridden_factories() -> None:
    class CustomString(zfs.String):
        pass

    class CustomStrings(zfs.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]

        def custom_marker(self) -> str:
            return "custom strings"

    class CustomStringSet(zfs.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet

    class CustomTerm(zfs.RealTerm):
        strings_type = CustomStrings

    class CustomTerms(zfs.RealTerms):
        term_type = CustomTerm

    class CustomSum(zfs.RealTermSum):
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

    def keep(string: CustomString) -> bool:
        return string == strings[0]

    assert_type(strings.new_clear_string(), CustomString)
    assert_type(strings.iter_filter_populated(), Iterator[CustomString])
    assert_type(strings.iter_filter_map(keep), Iterator[CustomString])
    assert_type(strings.filter_map(keep), CustomStrings)
    assert_type(strings.filter_unique(), CustomStrings)
    assert_type(strings.filter_populated(), CustomStrings)
    assert_type(string_set.iter_filter_map(keep), Iterator[CustomString])
    assert_type(string_set.filter_map(keep), CustomStringSet)
    strings.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    string_set.filter_map(lambda item: assert_type(item, CustomString) == strings[0])
    assert type(strings.new_clear_string()) is CustomString
    assert type(next(strings.iter_filter_map(keep))) is CustomString
    assert type(strings.filter_map(keep)) is CustomStrings
    assert type(strings.filter_unique()) is CustomStrings
    assert type(string_set.filter_map(keep)) is CustomStringSet

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

    # The inferred custom API must exist at runtime for correctly configured storage.
    assert_type(total.strings.custom_marker(), str)
    assert total.strings.custom_marker() == "custom strings"
    adopted = CustomSum._create(total._impl, total._string_set._map)
    assert adopted.strings.custom_marker() == "custom strings"

    class CustomSet(zfs.RealTermSet):
        terms_type = CustomTerms

    custom_set = CustomSet.from_terms(terms)
    assert_type(custom_set.strings, CustomStrings)
    assert custom_set.strings.custom_marker() == "custom strings"

    # Compatible native data is insufficient when its Python wrapper lacks the promised API.
    ordinary = zfs.RealTerms(2, 1)
    ordinary_sum = zfs.RealTermSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomTerms._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSet.from_terms(ordinary)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary._impl)
    with pytest.raises(TypeError, match="String storage does not match its factory"):
        CustomSum._create(ordinary_sum._impl, ordinary_sum._string_set._map)
    assert len(ordinary) == 1

    # Specialised factories accept compatible parent-class inputs, too.
    parent = zfs.String(2, "[1, 0]")
    copied = CustomStrings.from_string(parent)
    assert_type(copied, CustomStrings)
    assert type(copied[0]) is CustomString
    assert copied[0] == parent
    copied.append(parent)
    assert copied[1] == parent
    copied[0].set(parent)
    parent_term = zfs.RealTerm(2, (parent, 2.0))
    custom_term = CustomTerm.from_string_coeff(parent, 2.0)
    assert_type(custom_term, CustomTerm)
    assert custom_term == parent_term
    custom_term.set(parent_term)
    terms.append(parent_term)
    total.insert(parent_term)
    assert total.lookup_coeff(parent) == 2.0
    assert parent == zfs.String(2, "[1, 0]")


def test_non_identity_filters_dispatch_to_overrides() -> None:
    class CustomString(zqo.String):
        pass

    class CustomStrings(zqo.Strings):
        string_type = CustomString
        _set_type: type[CustomStringSet]
        lazy_calls = 0
        eager_calls = 0

        def iter_filter_populated(self) -> Iterator[CustomString]:
            self.lazy_calls += 1
            return iter((self[1],))

        def filter_populated(self) -> Self:
            self.eager_calls += 1
            return self[2:].clone()

    class CustomStringSet(zqo.StringSet):
        strings_type = CustomStrings

    CustomStrings._set_type = CustomStringSet
    strings = CustomStrings.from_iterable(["X0", "Y0", "Z0"], 1)
    lazy = strings.iter_filter_non_identity()
    eager = strings.filter_non_identity()

    assert_type(lazy, Iterator[CustomString])
    assert_type(eager, CustomStrings)
    assert strings.lazy_calls == 1
    assert strings.eager_calls == 1
    collected = list(lazy)
    assert [str(string) for string in collected] == ["Y0"]
    assert [str(string) for string in eager] == ["Z0"]
    assert type(collected[0]) is CustomString
    assert type(eager) is CustomStrings

    # Forwarding preserves the lazy override's views and the eager override's copied storage.
    collected[0].set("X0")
    assert str(strings[1]) == "X0"
    eager[0].set("X0")
    assert str(strings[2]) == "Z0"
