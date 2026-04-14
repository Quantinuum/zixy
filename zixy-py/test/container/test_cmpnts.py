from __future__ import annotations

import pytest
from mock_cmpnts import String, Strings, StringSet, StringsImplArray


def test_cmpnt():
    s = String("hello")
    assert str(s) == "hello"
    s.set("world")
    assert str(s) == "world"
    s.set(String("hello"))
    assert str(s) == "hello"
    assert str(s.copy()) == "hello"
    s.set(None)
    assert str(s) == ""
    with pytest.raises(TypeError):
        s.set(1)


def test_cmpnt_array():
    impl = StringsImplArray()
    impl.resize(10)
    strings = Strings._create(impl)
    assert len(strings) == 10
    for i, c in enumerate("abcdefghij"):
        strings[i] = c
    assert str(strings) == "a, b, c, d, e, f, g, h, i, j"
    assert str(strings[:]) == "a, b, c, d, e, f, g, h, i, j"
    assert str(strings[0]) == "a"
    assert strings[::-1].slice == slice(9, None, -1)
    assert str(strings[::-1]) == "j, i, h, g, f, e, d, c, b, a"
    assert str(strings[::2]) == "a, c, e, g, i"
    assert str(strings[::-2]) == "j, h, f, d, b"
    assert str(strings[::-2][::-1]) == "b, d, f, h, j"
    assert str(strings.clone()) == "a, b, c, d, e, f, g, h, i, j"
    assert str(strings[:].clone()) == "a, b, c, d, e, f, g, h, i, j"
    assert str(strings[0].clone()) == "a"
    assert strings[::-1].clone().slice == slice(None)
    assert str(strings[::-1].clone()) == "j, i, h, g, f, e, d, c, b, a"
    assert len(Strings(5)) == 5
    assert str(Strings(5)) == ", , , , "
    owning = strings[:].clone()
    assert type(owning) is Strings
    assert owning is not None
    assert owning == strings
    owning = strings[::-2].clone()
    assert type(owning) is Strings
    assert str(owning) == "j, h, f, d, b"
    owning[:] = owning[::-1]
    assert str(owning) == "b, d, f, h, j"
    owning[::-1] = owning
    assert str(owning) == "j, h, f, d, b"
    assert str(owning[1:3]) == "h, f"
    with pytest.raises(ValueError) as err:
        owning[:3] = owning[:2]
    assert str(err.value) == "Length of source (2) does not match length of destination (3)"
    view = owning[1:3]
    assert owning.is_owning()
    assert not view.is_owning()
    with pytest.raises(ValueError) as err:
        view.resize(1)


def test_cmpnt_array_filter_map():
    s = Strings.from_iterable("what a time to be alive".split())
    assert len(s) == 6
    assert str(s) == "what, a, time, to, be, alive"

    def f(cmpnt) -> bool:
        if str(cmpnt)[0] != "t":
            cmpnt.set(str(cmpnt) + "?!")
        if "v" in str(cmpnt):
            return False
        return True

    s.filter_map(f)
    # filter does not modify the source
    assert str(s) == "what, a, time, to, be, alive"
    s = s.filter_map(f)
    assert len(s) == 5
    assert str(s) == "what?!, a?!, time, to, be?!"
    assert str(s.clone()) == "what?!, a?!, time, to, be?!"


def test_cmpnt_set():
    s = StringSet()
    assert len(s) == 0
    assert not s.contains("hello")
    assert s.insert("hello") == 0
    assert s.insert("hello") == 0
    assert len(s) == 1
    assert s.contains("hello")
    assert s.insert("world") == 1
    assert len(s) == 2
    assert s._map == {"hello": 0, "world": 1}
    s = StringSet.from_iterable("what a a time to to be to time alive time".split())
    assert len(s) == 6
    assert str(s) == "what, a, time, to, be, alive"
    assert s.lookup("be") == 4
    assert s.lookup("was") is None

    def f(cmpnt) -> bool:
        if str(cmpnt)[0] != "t":
            cmpnt.set(str(cmpnt) + "?!")
        if "v" in str(cmpnt):
            return False
        return True

    s_new = tuple(str(item) for item in s.iter_filter_map(f)) == (
        "what?!",
        "a?!",
        "time",
        "to",
        "be?!",
    )
    s_new = s.filter_map(f)
    assert len(s_new) == 5
    assert str(s_new) == "what?!, a?!, time, to, be?!"
    assert str(s_new.clone()) == "what?!, a?!, time, to, be?!"
