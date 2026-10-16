from collections.abc import Iterator
from typing import Any, assert_type

import zixy.fermion.operator.general as fermion_general
import zixy.fermion.operator.normal as fermion_normal
import zixy.fermion.state as fermion_state
import zixy.qubit.operator as qubit_operator
import zixy.qubit.state as qubit_state
from zixy.container.strings import String as BaseString


def _keep_base(string: BaseString[Any, Any]) -> bool:
    return True


def test_qubit_operator_string_helpers() -> None:
    strings = qubit_operator.Strings.from_iterable(["X0", "", "Z1", "X0"], 2)
    string_set = qubit_operator.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: qubit_operator.String) -> bool:
        return string != clear

    assert_type(clear, qubit_operator.String)
    assert_type(strings.iter_filter_non_identity(), Iterator[qubit_operator.String])
    assert_type(strings.filter_non_identity(), qubit_operator.Strings)
    assert strings.filter_non_identity() == strings.filter_populated()
    assert_type(strings.iter_filter_populated(), Iterator[qubit_operator.String])
    assert_type(strings.iter_filter_map(keep), Iterator[qubit_operator.String])
    assert_type(strings.filter_map(keep), qubit_operator.Strings)
    assert_type(strings.filter_unique(), qubit_operator.Strings)
    assert_type(strings.filter_populated(), qubit_operator.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[qubit_operator.String])
    assert_type(string_set.filter_map(keep), qubit_operator.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[qubit_operator.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[qubit_operator.String])
    assert_type(strings.filter_map(_keep_base), qubit_operator.Strings)
    assert_type(string_set.filter_map(_keep_base), qubit_operator.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, qubit_operator.String) != clear)
    strings.filter_map(lambda item: assert_type(item, qubit_operator.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, qubit_operator.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, qubit_operator.String) != clear)

    assert type(clear) is qubit_operator.String
    assert clear == qubit_operator.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is qubit_operator.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: qubit_operator.String) -> bool:
        string.clear()
        return True

    assert all(item == clear for item in strings.iter_filter_map(clear_copy))
    mapped = strings.filter_map(clear_copy)
    assert len(mapped) == len(strings)
    assert all(item == clear for item in mapped)
    assert len(string_set.filter_map(clear_copy)) == 1
    assert all(item == clear for item in string_set.iter_filter_map(clear_copy))
    assert str(strings) == before
    assert str(string_set) == set_before

    def unexpected_callback(string: qubit_operator.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is qubit_operator.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = qubit_operator.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0


def test_qubit_state_string_helpers() -> None:
    strings = qubit_state.Strings.from_iterable(["[1, 0]", "", "[0, 1]", "[1, 0]"], 2)
    string_set = qubit_state.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: qubit_state.String) -> bool:
        return string != clear

    assert_type(clear, qubit_state.String)
    assert_type(strings.iter_filter_populated(), Iterator[qubit_state.String])
    assert_type(strings.iter_filter_map(keep), Iterator[qubit_state.String])
    assert_type(strings.filter_map(keep), qubit_state.Strings)
    assert_type(strings.filter_unique(), qubit_state.Strings)
    assert_type(strings.filter_populated(), qubit_state.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[qubit_state.String])
    assert_type(string_set.filter_map(keep), qubit_state.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[qubit_state.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[qubit_state.String])
    assert_type(strings.filter_map(_keep_base), qubit_state.Strings)
    assert_type(string_set.filter_map(_keep_base), qubit_state.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, qubit_state.String) != clear)
    strings.filter_map(lambda item: assert_type(item, qubit_state.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, qubit_state.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, qubit_state.String) != clear)

    assert type(clear) is qubit_state.String
    assert clear == qubit_state.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is qubit_state.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: qubit_state.String) -> bool:
        string.clear()
        return True

    assert all(item == clear for item in strings.iter_filter_map(clear_copy))
    mapped = strings.filter_map(clear_copy)
    assert len(mapped) == len(strings)
    assert all(item == clear for item in mapped)
    assert len(string_set.filter_map(clear_copy)) == 1
    assert all(item == clear for item in string_set.iter_filter_map(clear_copy))
    assert str(strings) == before
    assert str(string_set) == set_before

    def unexpected_callback(string: qubit_state.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is qubit_state.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = qubit_state.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0


def test_fermion_general_string_helpers() -> None:
    strings = fermion_general.Strings.from_iterable(["F0^", "", "F1", "F0^"], 2)
    string_set = fermion_general.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: fermion_general.String) -> bool:
        return string != clear

    assert_type(clear, fermion_general.String)
    assert_type(strings.iter_filter_populated(), Iterator[fermion_general.String])
    assert_type(strings.iter_filter_map(keep), Iterator[fermion_general.String])
    assert_type(strings.filter_map(keep), fermion_general.Strings)
    assert_type(strings.filter_unique(), fermion_general.Strings)
    assert_type(strings.filter_populated(), fermion_general.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[fermion_general.String])
    assert_type(string_set.filter_map(keep), fermion_general.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[fermion_general.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[fermion_general.String])
    assert_type(strings.filter_map(_keep_base), fermion_general.Strings)
    assert_type(string_set.filter_map(_keep_base), fermion_general.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, fermion_general.String) != clear)
    strings.filter_map(lambda item: assert_type(item, fermion_general.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, fermion_general.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, fermion_general.String) != clear)

    assert type(clear) is fermion_general.String
    assert clear == fermion_general.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is fermion_general.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: fermion_general.String) -> bool:
        string.clear()
        return True

    assert all(item == clear for item in strings.iter_filter_map(clear_copy))
    mapped = strings.filter_map(clear_copy)
    assert len(mapped) == len(strings)
    assert all(item == clear for item in mapped)
    assert len(string_set.filter_map(clear_copy)) == 1
    assert all(item == clear for item in string_set.iter_filter_map(clear_copy))
    assert str(strings) == before
    assert str(string_set) == set_before

    def unexpected_callback(string: fermion_general.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is fermion_general.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = fermion_general.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0


def test_fermion_normal_string_helpers() -> None:
    strings = fermion_normal.Strings.from_iterable(["F0^", "", "F1", "F0^"], 2)
    string_set = fermion_normal.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: fermion_normal.String) -> bool:
        return string != clear

    assert_type(clear, fermion_normal.String)
    assert_type(strings.iter_filter_populated(), Iterator[fermion_normal.String])
    assert_type(strings.iter_filter_map(keep), Iterator[fermion_normal.String])
    assert_type(strings.filter_map(keep), fermion_normal.Strings)
    assert_type(strings.filter_unique(), fermion_normal.Strings)
    assert_type(strings.filter_populated(), fermion_normal.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[fermion_normal.String])
    assert_type(string_set.filter_map(keep), fermion_normal.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[fermion_normal.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[fermion_normal.String])
    assert_type(strings.filter_map(_keep_base), fermion_normal.Strings)
    assert_type(string_set.filter_map(_keep_base), fermion_normal.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, fermion_normal.String) != clear)
    strings.filter_map(lambda item: assert_type(item, fermion_normal.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, fermion_normal.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, fermion_normal.String) != clear)

    assert type(clear) is fermion_normal.String
    assert clear == fermion_normal.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is fermion_normal.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: fermion_normal.String) -> bool:
        string.clear()
        return True

    assert all(item == clear for item in strings.iter_filter_map(clear_copy))
    mapped = strings.filter_map(clear_copy)
    assert len(mapped) == len(strings)
    assert all(item == clear for item in mapped)
    assert len(string_set.filter_map(clear_copy)) == 1
    assert all(item == clear for item in string_set.iter_filter_map(clear_copy))
    assert str(strings) == before
    assert str(string_set) == set_before

    def unexpected_callback(string: fermion_normal.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is fermion_normal.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = fermion_normal.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0


def test_fermion_state_string_helpers() -> None:
    strings = fermion_state.Strings.from_iterable(["[1, 0]", "", "[0, 1]", "[1, 0]"], 2)
    string_set = fermion_state.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: fermion_state.String) -> bool:
        return string != clear

    assert_type(clear, fermion_state.String)
    assert_type(strings.iter_filter_populated(), Iterator[fermion_state.String])
    assert_type(strings.iter_filter_map(keep), Iterator[fermion_state.String])
    assert_type(strings.filter_map(keep), fermion_state.Strings)
    assert_type(strings.filter_unique(), fermion_state.Strings)
    assert_type(strings.filter_populated(), fermion_state.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[fermion_state.String])
    assert_type(string_set.filter_map(keep), fermion_state.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[fermion_state.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[fermion_state.String])
    assert_type(strings.filter_map(_keep_base), fermion_state.Strings)
    assert_type(string_set.filter_map(_keep_base), fermion_state.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, fermion_state.String) != clear)
    strings.filter_map(lambda item: assert_type(item, fermion_state.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, fermion_state.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, fermion_state.String) != clear)

    assert type(clear) is fermion_state.String
    assert clear == fermion_state.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is fermion_state.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: fermion_state.String) -> bool:
        string.clear()
        return True

    assert all(item == clear for item in strings.iter_filter_map(clear_copy))
    mapped = strings.filter_map(clear_copy)
    assert len(mapped) == len(strings)
    assert all(item == clear for item in mapped)
    assert len(string_set.filter_map(clear_copy)) == 1
    assert all(item == clear for item in string_set.iter_filter_map(clear_copy))
    assert str(strings) == before
    assert str(string_set) == set_before

    def unexpected_callback(string: fermion_state.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is fermion_state.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = fermion_state.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0
