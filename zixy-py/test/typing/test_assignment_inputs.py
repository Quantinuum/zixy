from collections import UserList
from typing import assert_type

import pytest

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs


def test_qubit_operator_assignment_contract() -> None:
    class DerivedString(zqo.String):
        pass

    class DerivedTerm(zqo.RealTerm):
        pass

    class DerivedTerms(zqo.RealTerms):
        term_type = DerivedTerm

    class DerivedTermSet(zqo.RealTermSet):
        terms_type = DerivedTerms

    specs: list[zqo.StringSpec] = [
        "X0",
        (zqo.X, zqo.I),
        [zqo.X, zqo.I],
        {0: zqo.X},
        UserList([zqo.X, zqo.I]),
    ]
    expected = zqo.String(2, specs[0])
    source = zqo.RealTerm(2, (expected, 2.0))

    # The same domain specifications work through construction and every mutation path.
    for spec in specs:
        string = DerivedString(2, spec)
        assert string == expected
        string.clear()
        string.set(spec)
        assert string == expected
        strings = zqo.Strings(2)
        strings.append(spec)
        strings[0] = spec
        strings[:] = spec
        assert strings[0] == expected
        string_set = zqo.StringSet(2)
        assert string_set.insert(spec) == 0
        assert string_set.contains(expected)
        term = DerivedTerm(2, spec)
        assert term.string == expected and term.coeff == 1.0
        term.set((spec, 2.0))
        assert term == source
        term.set((spec, None))
        assert term.coeff == 1.0
        term.set(spec)
        terms = DerivedTerms(2)
        terms.append(spec)
        terms[0] = (spec, 2.0)
        terms[:] = (spec, 2.0)
        terms[:] = terms[:]
        assert terms[0] == source
        total = DerivedTermSet(2)
        total.insert((spec, 2.0))
        assert next(iter(total)) == source

    internal = zqo.Strings.from_iterable([specs[0], ""], 2)
    internal[1].set(internal[0])
    assert internal[0] == internal[1] == expected
    internal[0].clear()
    assert internal[1] == expected

    # Parent-to-subclass copying is supported, including inference of the source's mode space.
    string = DerivedString(source=expected)
    assert_type(string, DerivedString)
    assert string == expected
    string.set(expected)
    copied_back = zqo.String(source=string)
    assert copied_back == expected
    copied_back.clear()
    assert string == expected
    term = DerivedTerm(2, source)
    term.set(source)
    assert_type(term, DerivedTerm)
    assert term == source
    terms = DerivedTerms(2)
    terms.append(source)
    terms[0] = source
    terms[:] = source
    assert terms[0] == source
    total = DerivedTermSet(2)
    total.insert(source)
    assert next(iter(total)) == source

    # All term input paths use checked coefficient conversion, independent of source subtype.
    complex_source = zqo.ComplexTerm(2, (expected, 2 + 0j))
    term.set(complex_source)
    assert_type(term.coeff, float)
    assert term == source
    assert zqo.RealTerm(2, complex_source) == source
    terms.append(complex_source)
    terms[0] = complex_source
    terms[:] = complex_source
    total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0
    promoted = zqo.ComplexTerm(2, source)
    assert_type(promoted.coeff, complex)
    assert promoted.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    saved = term.clone()
    with pytest.raises(ValueError):
        term.set(complex_source)
    assert term == saved
    with pytest.raises(ValueError):
        total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0

    # Copying must reject incompatible spaces and families without changing existing values.
    wrong_space = zqo.String(3, specs[0])
    with pytest.raises(ValueError):
        string.set(wrong_space)
    assert string == expected
    wrong_term = zqo.RealTerm(3, (specs[0], 3.0))
    with pytest.raises(ValueError):
        term.set(wrong_term)
    assert term == saved
    wrong_string = zqs.String(2)
    with pytest.raises(TypeError):
        string.set(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DerivedString(2, wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        strings.append(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        string_set.insert(wrong_string)  # type: ignore[arg-type]
    foreign_term = zqs.RealTerm(2)
    with pytest.raises(TypeError):
        zqo.RealTerm(2, foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        terms.append(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        term.set(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        total.insert(foreign_term)  # type: ignore[arg-type]
    assert term == saved
    assert source.coeff == 2.0 and source.string == expected


def test_qubit_state_assignment_contract() -> None:
    class DerivedString(zqs.String):
        pass

    class DerivedTerm(zqs.RealTerm):
        pass

    class DerivedTerms(zqs.RealTerms):
        term_type = DerivedTerm

    class DerivedTermSet(zqs.RealTermSet):
        terms_type = DerivedTerms

    specs: list[zqs.StringSpec] = [
        "[1, 0]",
        (True, False),
        [True, False],
        {0},
        UserList([True, False]),
    ]
    expected = zqs.String(2, specs[0])
    source = zqs.RealTerm(2, (expected, 2.0))

    # The same domain specifications work through construction and every mutation path.
    for spec in specs:
        string = DerivedString(2, spec)
        assert string == expected
        string.clear()
        string.set(spec)
        assert string == expected
        strings = zqs.Strings(2)
        strings.append(spec)
        strings[0] = spec
        strings[:] = spec
        assert strings[0] == expected
        string_set = zqs.StringSet(2)
        assert string_set.insert(spec) == 0
        assert string_set.contains(expected)
        term = DerivedTerm(2, spec)
        assert term.string == expected and term.coeff == 1.0
        term.set((spec, 2.0))
        assert term == source
        term.set((spec, None))
        assert term.coeff == 1.0
        term.set(spec)
        terms = DerivedTerms(2)
        terms.append(spec)
        terms[0] = (spec, 2.0)
        terms[:] = (spec, 2.0)
        terms[:] = terms[:]
        assert terms[0] == source
        total = DerivedTermSet(2)
        total.insert((spec, 2.0))
        assert next(iter(total)) == source

    internal = zqs.Strings.from_iterable([specs[0], ""], 2)
    internal[1].set(internal[0])
    assert internal[0] == internal[1] == expected
    internal[0].clear()
    assert internal[1] == expected

    # Parent-to-subclass copying is supported, including inference of the source's mode space.
    string = DerivedString(source=expected)
    assert_type(string, DerivedString)
    assert string == expected
    string.set(expected)
    copied_back = zqs.String(source=string)
    assert copied_back == expected
    copied_back.clear()
    assert string == expected
    term = DerivedTerm(2, source)
    term.set(source)
    assert_type(term, DerivedTerm)
    assert term == source
    terms = DerivedTerms(2)
    terms.append(source)
    terms[0] = source
    terms[:] = source
    assert terms[0] == source
    total = DerivedTermSet(2)
    total.insert(source)
    assert next(iter(total)) == source

    # All term input paths use checked coefficient conversion, independent of source subtype.
    complex_source = zqs.ComplexTerm(2, (expected, 2 + 0j))
    term.set(complex_source)
    assert_type(term.coeff, float)
    assert term == source
    assert zqs.RealTerm(2, complex_source) == source
    terms.append(complex_source)
    terms[0] = complex_source
    terms[:] = complex_source
    total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0
    promoted = zqs.ComplexTerm(2, source)
    assert_type(promoted.coeff, complex)
    assert promoted.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    saved = term.clone()
    with pytest.raises(ValueError):
        term.set(complex_source)
    assert term == saved
    with pytest.raises(ValueError):
        total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0

    # Copying must reject incompatible spaces and families without changing existing values.
    wrong_space = zqs.String(3, specs[0])
    with pytest.raises(ValueError):
        string.set(wrong_space)
    assert string == expected
    wrong_term = zqs.RealTerm(3, (specs[0], 3.0))
    with pytest.raises(ValueError):
        term.set(wrong_term)
    assert term == saved
    wrong_string = zfs.String(2)
    with pytest.raises(TypeError):
        string.set(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DerivedString(2, wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        strings.append(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        string_set.insert(wrong_string)  # type: ignore[arg-type]
    foreign_term = zfs.RealTerm(2)
    with pytest.raises(TypeError):
        zqs.RealTerm(2, foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        terms.append(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        term.set(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        total.insert(foreign_term)  # type: ignore[arg-type]
    assert term == saved
    assert source.coeff == 2.0 and source.string == expected


def test_fermion_general_assignment_contract() -> None:
    class DerivedString(zfog.String):
        pass

    class DerivedTerm(zfog.RealTerm):
        pass

    class DerivedTerms(zfog.RealTerms):
        term_type = DerivedTerm

    class DerivedTermSet(zfog.RealTermSet):
        terms_type = DerivedTerms

    specs: list[zfog.StringSpec] = [
        "F0^ F1",
        [(0, True), (1, False)],
        ((0, True), (1, False)),
        UserList([(0, True), (1, False)]),
    ]
    expected = zfog.String(2, specs[0])
    source = zfog.RealTerm(2, (expected, 2.0))

    # The same domain specifications work through construction and every mutation path.
    for spec in specs:
        string = DerivedString(2, spec)
        assert string == expected
        string.clear()
        string.set(spec)
        assert string == expected
        strings = zfog.Strings(2, max_string_len=2)
        strings.append(spec)
        strings[0] = spec
        strings[:] = spec
        assert strings[0] == expected
        string_set = zfog.StringSet(2, max_string_len=2)
        assert string_set.insert(spec) == 0
        assert string_set.contains(expected)
        term = DerivedTerm(2, spec)
        assert term.string == expected and term.coeff == 1.0
        term.set((spec, 2.0))
        assert term == source
        term.set((spec, None))
        assert term.coeff == 1.0
        term.set(spec)
        terms = DerivedTerms(2, max_string_len=2)
        terms.append(spec)
        terms[0] = (spec, 2.0)
        terms[:] = (spec, 2.0)
        terms[:] = terms[:]
        assert terms[0] == source
        total = DerivedTermSet(2, max_string_len=2)
        total.insert((spec, 2.0))
        assert next(iter(total)) == source

    internal = zfog.Strings.from_iterable([specs[0], ""], 2)
    internal[1].set(internal[0])
    assert internal[0] == internal[1] == expected
    internal[0].clear()
    assert internal[1] == expected

    # Parent-to-subclass copying is supported, including inference of the source's mode space.
    string = DerivedString(source=expected)
    assert_type(string, DerivedString)
    assert string == expected
    string.set(expected)
    copied_back = zfog.String(source=string)
    assert copied_back == expected
    copied_back.clear()
    assert string == expected
    term = DerivedTerm(2, source)
    term.set(source)
    assert_type(term, DerivedTerm)
    assert term == source
    terms = DerivedTerms(2, max_string_len=2)
    terms.append(source)
    terms[0] = source
    terms[:] = source
    assert terms[0] == source
    total = DerivedTermSet(2, max_string_len=2)
    total.insert(source)
    assert next(iter(total)) == source

    # All term input paths use checked coefficient conversion, independent of source subtype.
    complex_source = zfog.ComplexTerm(2, (expected, 2 + 0j))
    term.set(complex_source)
    assert_type(term.coeff, float)
    assert term == source
    assert zfog.RealTerm(2, complex_source) == source
    terms.append(complex_source)
    terms[0] = complex_source
    terms[:] = complex_source
    total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0
    promoted = zfog.ComplexTerm(2, source)
    assert_type(promoted.coeff, complex)
    assert promoted.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    saved = term.clone()
    with pytest.raises(ValueError):
        term.set(complex_source)
    assert term == saved
    with pytest.raises(ValueError):
        total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0

    # Copying must reject incompatible spaces and families without changing existing values.
    wrong_space = zfog.String(3, specs[0])
    with pytest.raises(ValueError):
        string.set(wrong_space)
    assert string == expected
    wrong_term = zfog.RealTerm(3, (specs[0], 3.0))
    with pytest.raises(ValueError):
        term.set(wrong_term)
    assert term == saved
    wrong_string = zfon.String(2)
    with pytest.raises(TypeError):
        string.set(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DerivedString(2, wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        strings.append(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        string_set.insert(wrong_string)  # type: ignore[arg-type]
    foreign_term = zfon.RealTerm(2)
    with pytest.raises(TypeError):
        zfog.RealTerm(2, foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        terms.append(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        term.set(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        total.insert(foreign_term)  # type: ignore[arg-type]
    assert term == saved
    assert source.coeff == 2.0 and source.string == expected


def test_fermion_normal_assignment_contract() -> None:
    class DerivedString(zfon.String):
        pass

    class DerivedTerm(zfon.RealTerm):
        pass

    class DerivedTerms(zfon.RealTerms):
        term_type = DerivedTerm

    class DerivedTermSet(zfon.RealTermSet):
        terms_type = DerivedTerms

    specs: list[zfon.StringSpec] = ["F0^ F1", ([0], [1]), ((True, False), (False, True))]
    expected = zfon.String(2, specs[0])
    source = zfon.RealTerm(2, (expected, 2.0))

    # The same domain specifications work through construction and every mutation path.
    for spec in specs:
        string = DerivedString(2, spec)
        assert string == expected
        string.clear()
        string.set(spec)
        assert string == expected
        strings = zfon.Strings(2)
        strings.append(spec)
        strings[0] = spec
        strings[:] = spec
        assert strings[0] == expected
        string_set = zfon.StringSet(2)
        assert string_set.insert(spec) == 0
        assert string_set.contains(expected)
        term = DerivedTerm(2, spec)
        assert term.string == expected and term.coeff == 1.0
        term.set((spec, 2.0))
        assert term == source
        term.set((spec, None))
        assert term.coeff == 1.0
        term.set(spec)
        terms = DerivedTerms(2)
        terms.append(spec)
        terms[0] = (spec, 2.0)
        terms[:] = (spec, 2.0)
        terms[:] = terms[:]
        assert terms[0] == source
        total = DerivedTermSet(2)
        total.insert((spec, 2.0))
        assert next(iter(total)) == source

    internal = zfon.Strings.from_iterable([specs[0], ""], 2)
    internal[1].set(internal[0])
    assert internal[0] == internal[1] == expected
    internal[0].clear()
    assert internal[1] == expected

    # Parent-to-subclass copying is supported, including inference of the source's mode space.
    string = DerivedString(source=expected)
    assert_type(string, DerivedString)
    assert string == expected
    string.set(expected)
    copied_back = zfon.String(source=string)
    assert copied_back == expected
    copied_back.clear()
    assert string == expected
    term = DerivedTerm(2, source)
    term.set(source)
    assert_type(term, DerivedTerm)
    assert term == source
    terms = DerivedTerms(2)
    terms.append(source)
    terms[0] = source
    terms[:] = source
    assert terms[0] == source
    total = DerivedTermSet(2)
    total.insert(source)
    assert next(iter(total)) == source

    # All term input paths use checked coefficient conversion, independent of source subtype.
    complex_source = zfon.ComplexTerm(2, (expected, 2 + 0j))
    term.set(complex_source)
    assert_type(term.coeff, float)
    assert term == source
    assert zfon.RealTerm(2, complex_source) == source
    terms.append(complex_source)
    terms[0] = complex_source
    terms[:] = complex_source
    total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0
    promoted = zfon.ComplexTerm(2, source)
    assert_type(promoted.coeff, complex)
    assert promoted.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    saved = term.clone()
    with pytest.raises(ValueError):
        term.set(complex_source)
    assert term == saved
    with pytest.raises(ValueError):
        total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0

    # Copying must reject incompatible spaces and families without changing existing values.
    wrong_space = zfon.String(3, specs[0])
    with pytest.raises(ValueError):
        string.set(wrong_space)
    assert string == expected
    wrong_term = zfon.RealTerm(3, (specs[0], 3.0))
    with pytest.raises(ValueError):
        term.set(wrong_term)
    assert term == saved
    wrong_string = zfog.String(2)
    with pytest.raises(TypeError):
        string.set(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DerivedString(2, wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        strings.append(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        string_set.insert(wrong_string)  # type: ignore[arg-type]
    foreign_term = zfog.RealTerm(2)
    with pytest.raises(TypeError):
        zfon.RealTerm(2, foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        terms.append(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        term.set(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        total.insert(foreign_term)  # type: ignore[arg-type]
    assert term == saved
    assert source.coeff == 2.0 and source.string == expected


def test_fermion_state_assignment_contract() -> None:
    class DerivedString(zfs.String):
        pass

    class DerivedTerm(zfs.RealTerm):
        pass

    class DerivedTerms(zfs.RealTerms):
        term_type = DerivedTerm

    class DerivedTermSet(zfs.RealTermSet):
        terms_type = DerivedTerms

    specs: list[zfs.StringSpec] = [
        "[1, 0]",
        (True, False),
        [True, False],
        {0},
        UserList([True, False]),
    ]
    expected = zfs.String(2, specs[0])
    source = zfs.RealTerm(2, (expected, 2.0))

    # The same domain specifications work through construction and every mutation path.
    for spec in specs:
        string = DerivedString(2, spec)
        assert string == expected
        string.clear()
        string.set(spec)
        assert string == expected
        strings = zfs.Strings(2)
        strings.append(spec)
        strings[0] = spec
        strings[:] = spec
        assert strings[0] == expected
        string_set = zfs.StringSet(2)
        assert string_set.insert(spec) == 0
        assert string_set.contains(expected)
        term = DerivedTerm(2, spec)
        assert term.string == expected and term.coeff == 1.0
        term.set((spec, 2.0))
        assert term == source
        term.set((spec, None))
        assert term.coeff == 1.0
        term.set(spec)
        terms = DerivedTerms(2)
        terms.append(spec)
        terms[0] = (spec, 2.0)
        terms[:] = (spec, 2.0)
        terms[:] = terms[:]
        assert terms[0] == source
        total = DerivedTermSet(2)
        total.insert((spec, 2.0))
        assert next(iter(total)) == source

    internal = zfs.Strings.from_iterable([specs[0], ""], 2)
    internal[1].set(internal[0])
    assert internal[0] == internal[1] == expected
    internal[0].clear()
    assert internal[1] == expected

    # Parent-to-subclass copying is supported, including inference of the source's mode space.
    string = DerivedString(source=expected)
    assert_type(string, DerivedString)
    assert string == expected
    string.set(expected)
    copied_back = zfs.String(source=string)
    assert copied_back == expected
    copied_back.clear()
    assert string == expected
    term = DerivedTerm(2, source)
    term.set(source)
    assert_type(term, DerivedTerm)
    assert term == source
    terms = DerivedTerms(2)
    terms.append(source)
    terms[0] = source
    terms[:] = source
    assert terms[0] == source
    total = DerivedTermSet(2)
    total.insert(source)
    assert next(iter(total)) == source

    # All term input paths use checked coefficient conversion, independent of source subtype.
    complex_source = zfs.ComplexTerm(2, (expected, 2 + 0j))
    term.set(complex_source)
    assert_type(term.coeff, float)
    assert term == source
    assert zfs.RealTerm(2, complex_source) == source
    terms.append(complex_source)
    terms[0] = complex_source
    terms[:] = complex_source
    total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0
    promoted = zfs.ComplexTerm(2, source)
    assert_type(promoted.coeff, complex)
    assert promoted.coeff == 2 + 0j
    complex_source.coeff = 2 + 1j
    saved = term.clone()
    with pytest.raises(ValueError):
        term.set(complex_source)
    assert term == saved
    with pytest.raises(ValueError):
        total.insert(complex_source)
    assert next(iter(total)).coeff == 2.0

    # Copying must reject incompatible spaces and families without changing existing values.
    wrong_space = zfs.String(3, specs[0])
    with pytest.raises(ValueError):
        string.set(wrong_space)
    assert string == expected
    wrong_term = zfs.RealTerm(3, (specs[0], 3.0))
    with pytest.raises(ValueError):
        term.set(wrong_term)
    assert term == saved
    wrong_string = zqs.String(2)
    with pytest.raises(TypeError):
        string.set(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DerivedString(2, wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        strings.append(wrong_string)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        string_set.insert(wrong_string)  # type: ignore[arg-type]
    foreign_term = zqs.RealTerm(2)
    with pytest.raises(TypeError):
        zfs.RealTerm(2, foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        terms.append(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        term.set(foreign_term)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        total.insert(foreign_term)  # type: ignore[arg-type]
    assert term == saved
    assert source.coeff == 2.0 and source.string == expected
