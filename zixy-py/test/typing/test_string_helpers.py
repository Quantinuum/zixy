from collections.abc import Iterator
from typing import Any, assert_type

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy.container.strings import String as BaseString


def _keep_base(string: BaseString[Any, Any]) -> bool:
    return True


def test_qubit_operator_string_helpers() -> None:
    strings = zqo.Strings.from_iterable(["X0", "", "Z1", "X0"], 2)
    string_set = zqo.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: zqo.String) -> bool:
        return string != clear

    assert_type(clear, zqo.String)
    assert_type(strings.iter_filter_non_identity(), Iterator[zqo.String])
    assert_type(strings.filter_non_identity(), zqo.Strings)
    assert strings.filter_non_identity() == strings.filter_populated()
    assert_type(strings.iter_filter_populated(), Iterator[zqo.String])
    assert_type(strings.iter_filter_map(keep), Iterator[zqo.String])
    assert_type(strings.filter_map(keep), zqo.Strings)
    assert_type(strings.filter_unique(), zqo.Strings)
    assert_type(strings.filter_populated(), zqo.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[zqo.String])
    assert_type(string_set.filter_map(keep), zqo.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[zqo.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[zqo.String])
    assert_type(strings.filter_map(_keep_base), zqo.Strings)
    assert_type(string_set.filter_map(_keep_base), zqo.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, zqo.String) != clear)
    strings.filter_map(lambda item: assert_type(item, zqo.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, zqo.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, zqo.String) != clear)

    assert type(clear) is zqo.String
    assert clear == zqo.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is zqo.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: zqo.String) -> bool:
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

    def unexpected_callback(string: zqo.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is zqo.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = zqo.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0


def test_qubit_state_string_helpers() -> None:
    strings = zqs.Strings.from_iterable(["[1, 0]", "", "[0, 1]", "[1, 0]"], 2)
    string_set = zqs.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: zqs.String) -> bool:
        return string != clear

    assert_type(clear, zqs.String)
    assert_type(strings.iter_filter_populated(), Iterator[zqs.String])
    assert_type(strings.iter_filter_map(keep), Iterator[zqs.String])
    assert_type(strings.filter_map(keep), zqs.Strings)
    assert_type(strings.filter_unique(), zqs.Strings)
    assert_type(strings.filter_populated(), zqs.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[zqs.String])
    assert_type(string_set.filter_map(keep), zqs.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[zqs.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[zqs.String])
    assert_type(strings.filter_map(_keep_base), zqs.Strings)
    assert_type(string_set.filter_map(_keep_base), zqs.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, zqs.String) != clear)
    strings.filter_map(lambda item: assert_type(item, zqs.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, zqs.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, zqs.String) != clear)

    assert type(clear) is zqs.String
    assert clear == zqs.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is zqs.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: zqs.String) -> bool:
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

    def unexpected_callback(string: zqs.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is zqs.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = zqs.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0


def test_fermion_general_string_helpers() -> None:
    strings = zfog.Strings.from_iterable(["F0^", "", "F1", "F0^"], 2)
    string_set = zfog.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: zfog.String) -> bool:
        return string != clear

    assert_type(clear, zfog.String)
    assert_type(strings.iter_filter_populated(), Iterator[zfog.String])
    assert_type(strings.iter_filter_map(keep), Iterator[zfog.String])
    assert_type(strings.filter_map(keep), zfog.Strings)
    assert_type(strings.filter_unique(), zfog.Strings)
    assert_type(strings.filter_populated(), zfog.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[zfog.String])
    assert_type(string_set.filter_map(keep), zfog.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[zfog.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[zfog.String])
    assert_type(strings.filter_map(_keep_base), zfog.Strings)
    assert_type(string_set.filter_map(_keep_base), zfog.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, zfog.String) != clear)
    strings.filter_map(lambda item: assert_type(item, zfog.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, zfog.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, zfog.String) != clear)

    assert type(clear) is zfog.String
    assert clear == zfog.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is zfog.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: zfog.String) -> bool:
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

    def unexpected_callback(string: zfog.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is zfog.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = zfog.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0


def test_fermion_normal_string_helpers() -> None:
    strings = zfon.Strings.from_iterable(["F0^", "", "F1", "F0^"], 2)
    string_set = zfon.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: zfon.String) -> bool:
        return string != clear

    assert_type(clear, zfon.String)
    assert_type(strings.iter_filter_populated(), Iterator[zfon.String])
    assert_type(strings.iter_filter_map(keep), Iterator[zfon.String])
    assert_type(strings.filter_map(keep), zfon.Strings)
    assert_type(strings.filter_unique(), zfon.Strings)
    assert_type(strings.filter_populated(), zfon.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[zfon.String])
    assert_type(string_set.filter_map(keep), zfon.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[zfon.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[zfon.String])
    assert_type(strings.filter_map(_keep_base), zfon.Strings)
    assert_type(string_set.filter_map(_keep_base), zfon.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, zfon.String) != clear)
    strings.filter_map(lambda item: assert_type(item, zfon.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, zfon.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, zfon.String) != clear)

    assert type(clear) is zfon.String
    assert clear == zfon.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is zfon.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: zfon.String) -> bool:
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

    def unexpected_callback(string: zfon.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is zfon.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = zfon.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0


def test_fermion_state_string_helpers() -> None:
    strings = zfs.Strings.from_iterable(["[1, 0]", "", "[0, 1]", "[1, 0]"], 2)
    string_set = zfs.StringSet.from_strings(strings)
    clear = strings.new_clear_string()
    before = str(strings)
    set_before = str(string_set)

    def keep(string: zfs.String) -> bool:
        return string != clear

    assert_type(clear, zfs.String)
    assert_type(strings.iter_filter_populated(), Iterator[zfs.String])
    assert_type(strings.iter_filter_map(keep), Iterator[zfs.String])
    assert_type(strings.filter_map(keep), zfs.Strings)
    assert_type(strings.filter_unique(), zfs.Strings)
    assert_type(strings.filter_populated(), zfs.Strings)
    assert_type(string_set.iter_filter_map(keep), Iterator[zfs.String])
    assert_type(string_set.filter_map(keep), zfs.StringSet)
    assert_type(strings.iter_filter_map(_keep_base), Iterator[zfs.String])
    assert_type(string_set.iter_filter_map(_keep_base), Iterator[zfs.String])
    assert_type(strings.filter_map(_keep_base), zfs.Strings)
    assert_type(string_set.filter_map(_keep_base), zfs.StringSet)

    # Check contextual inference as well as accepting callbacks with explicit annotations.
    strings.iter_filter_map(lambda item: assert_type(item, zfs.String) != clear)
    strings.filter_map(lambda item: assert_type(item, zfs.String) != clear)
    string_set.iter_filter_map(lambda item: assert_type(item, zfs.String) != clear)
    string_set.filter_map(lambda item: assert_type(item, zfs.String) != clear)

    assert type(clear) is zfs.String
    assert clear == zfs.String(2)
    assert [str(item) for item in strings.iter_filter_populated()] == [
        str(strings[0]),
        str(strings[2]),
        str(strings[3]),
    ]
    assert strings.filter_map(keep) == strings.filter_populated()
    assert string_set.filter_map(keep).to_strings() == strings.filter_populated().filter_unique()
    assert type(strings.filter_unique()) is zfs.Strings
    assert len(strings.filter_unique()) == 3

    # Deduplication must use only the selected view, including reversed and strided slices.
    assert strings[1:3].filter_unique() == strings[1:3]
    assert strings[::-2].filter_unique() == strings[::-2]
    assert strings[:0].filter_unique() == strings[:0]
    unique = strings.filter_unique()
    unique[0].clear()
    assert str(strings) == before

    def clear_copy(string: zfs.String) -> bool:
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

    def unexpected_callback(string: zfs.String) -> bool:
        raise AssertionError("Empty collections must not invoke callbacks")

    empty = strings[:0]
    assert type(empty.new_clear_string()) is zfs.String
    assert list(empty.iter_filter_map(unexpected_callback)) == []
    assert list(empty.iter_filter_populated()) == []
    assert len(empty.filter_map(unexpected_callback)) == 0
    empty_set = zfs.StringSet(2)
    assert list(empty_set.iter_filter_map(unexpected_callback)) == []
    assert len(empty_set.filter_map(unexpected_callback)) == 0
