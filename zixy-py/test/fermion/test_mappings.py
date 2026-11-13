"""Tests for fermionic-string mappers."""

from typing import get_type_hints

import pytest
from sympy import Symbol

import zixy.fermion.operator.general as zfog
import zixy.fermion.operator.normal as zfon
import zixy.fermion.state as zfs
import zixy.qubit.operator as zqo
import zixy.qubit.state as zqs
from zixy._zixy import Qubits
from zixy.mappings import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    Mapper,
    ParaparticularMapper,
    ParityMapper,
)

MAPPER_TYPES = (
    JordanWignerMapper,
    BravyiKitaevMapper,
    ParityMapper,
    ParaparticularMapper,
)


@pytest.mark.parametrize(
    ("mapper_type", "qubits", "mode_ordering", "string_type", "source", "expected"),
    (
        (
            JordanWignerMapper,
            2,
            None,
            zfon.String,
            "F0^",
            "((0.5+0j), X0), (-0.5j, Y0)",
        ),
        (
            BravyiKitaevMapper,
            2,
            None,
            zfon.String,
            "F0^",
            "((0.5+0j), X0 X1), (-0.5j, Y0 X1)",
        ),
        (
            ParityMapper,
            2,
            None,
            zfon.String,
            "F0^",
            "((0.5+0j), X0 X1), (-0.5j, Y0 X1)",
        ),
        (
            ParaparticularMapper,
            2,
            None,
            zfon.String,
            "F0^",
            "((0.5+0j), X0), (-0.5j, Y0)",
        ),
        (
            JordanWignerMapper,
            4,
            None,
            zfon.String,
            "F1^",
            "((0.5+0j), Z0 X1), (-0.5j, Z0 Y1)",
        ),
        (
            BravyiKitaevMapper,
            4,
            None,
            zfon.String,
            "F1^",
            "((0.5+0j), Z0 X1 X3), (-0.5j, Y1 X3)",
        ),
        (
            ParityMapper,
            4,
            None,
            zfon.String,
            "F1^",
            "((0.5+0j), Z0 X1 X2 X3), (-0.5j, Y1 X2 X3)",
        ),
        (
            ParaparticularMapper,
            4,
            None,
            zfon.String,
            "F1^",
            "((0.5+0j), X1), (-0.5j, Y1)",
        ),
        (
            JordanWignerMapper,
            2,
            [1, 0],
            zfon.String,
            "F0^",
            "((0.5+0j), X1), (-0.5j, Y1)",
        ),
        (
            BravyiKitaevMapper,
            2,
            [1, 0],
            zfon.String,
            "F0^",
            "((0.5+0j), X0 X1), (-0.5j, X0 Y1)",
        ),
        (
            ParityMapper,
            2,
            [1, 0],
            zfon.String,
            "F0^",
            "((0.5+0j), X0 X1), (-0.5j, X0 Y1)",
        ),
        (
            ParaparticularMapper,
            2,
            [1, 0],
            zfon.String,
            "F0^",
            "((0.5+0j), X1), (-0.5j, Y1)",
        ),
        (
            JordanWignerMapper,
            2,
            None,
            zfog.String,
            "F0 F1^",
            "(-0.25j, Y0 X1), ((-0.25+0j), X0 X1), ((-0.25+0j), Y0 Y1), (0.25j, X0 Y1)",
        ),
    ),
)
def test_apply_strings(mapper_type, qubits, mode_ordering, string_type, source, expected):
    mapper = mapper_type(qubits, mode_ordering=mode_ordering)

    terms = mapper.apply(string_type(qubits, source))

    assert isinstance(terms, zqo.ComplexTermSum)
    assert str(terms) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "F2^ F1",
            "(0.25j, Y1 X2), ((0.25+0j), Y1 Y2), ((0.25+0j), X1 X2), (-0.25j, X1 Y2)",
        ),
        (
            "F0^ F0 F1^ F1",
            "((0.25+0j), ), ((-0.25+0j), Z0), ((-0.25+0j), Z1), ((0.25+0j), Z0 Z1)",
        ),
        (
            "F0^ F1 F2^ F3",
            "((-0.0625+0j), Y0 X1 Y2 X3), (-0.0625j, X0 X1 Y2 X3), "
            "(-0.0625j, Y0 Y1 Y2 X3), ((0.0625+0j), X0 Y1 Y2 X3), "
            "(-0.0625j, Y0 X1 X2 X3), ((0.0625+0j), X0 X1 X2 X3), "
            "((0.0625+0j), Y0 Y1 X2 X3), (0.0625j, X0 Y1 X2 X3), "
            "(-0.0625j, Y0 X1 Y2 Y3), ((0.0625+0j), X0 X1 Y2 Y3), "
            "((0.0625+0j), Y0 Y1 Y2 Y3), (0.0625j, X0 Y1 Y2 Y3), "
            "((0.0625+0j), Y0 X1 X2 Y3), (0.0625j, X0 X1 X2 Y3), "
            "(0.0625j, Y0 Y1 X2 Y3), ((-0.0625+0j), X0 Y1 X2 Y3)",
        ),
    ),
)
def test_apply_products(source, expected):
    mapper = JordanWignerMapper(4)

    terms = mapper.apply(zfog.String(4, source))

    assert isinstance(terms, zqo.ComplexTermSum)
    assert str(terms) == expected


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_apply_adopts_native_map_without_python_reinsertion(mapper_type, monkeypatch):
    def fail(*_args):
        raise AssertionError("mapped terms were reinserted in Python")

    monkeypatch.setattr(zqo.ComplexTermSum, "insert_iterable", fail)

    terms = mapper_type(2).apply(zfog.String(2, "F0 F1^"))

    assert len(terms) == 4
    assert terms.lookup_index(terms.strings[0]) == 0


def test_apply_products_anticommute():
    mapper = JordanWignerMapper(4)
    caca = mapper.apply(zfog.String(4, "F0^ F1 F2^ F3"))
    ccaa = mapper.apply(zfog.String(4, "F0^ F2^ F1 F3"))

    assert caca == -ccaa


def test_apply_scaled_sums():
    mapper = JordanWignerMapper(2)
    terms = zqo.ComplexTermSum(mapper.qubits)

    terms += 2.0 * mapper.apply(zfog.String(2, "F0^ F0"))
    terms += -mapper.apply(zfog.String(2, "F1^ F1"))

    assert str(terms) == "((0.5+0j), ), ((-1+0j), Z0), ((0.5+0j), Z1)"


def test_to_qubit_number_operator():
    mapper = JordanWignerMapper(2)
    fermion_terms = zfon.RealTermSum.from_str("F0^ F0", 2)

    qubit_terms = fermion_terms.to_qubit()

    assert str(qubit_terms) == "((0.5+0j), ), ((-0.5+0j), Z0)"
    assert qubit_terms == mapper.apply(zfog.String(2, "F0^ F0"))


@pytest.mark.parametrize(
    ("term_sum_type", "string_type", "source", "pieces"),
    (
        (
            zfon.RealTermSum,
            zfon.String,
            "(2, F0^ F0), (-1, F1^ F1)",
            (("F0^ F0", 2.0), ("F1^ F1", -1.0)),
        ),
        (
            zfog.RealTermSum,
            zfog.String,
            "(2, F0 F1^), (-3, F1 F0^)",
            (("F0 F1^", 2.0), ("F1 F0^", -3.0)),
        ),
    ),
)
@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_to_qubit_real_terms(mapper_type, term_sum_type, string_type, source, pieces):
    mapper = mapper_type(2)
    fermion_terms = term_sum_type.from_str(source, 2)

    via_to_qubit = fermion_terms.to_qubit(mapper=mapper_type)
    via_strings = zqo.ComplexTermSum(mapper.qubits)
    for string_source, coeff in pieces:
        via_strings += coeff * mapper.apply(string_type(2, string_source))

    assert isinstance(via_to_qubit, zqo.ComplexTermSum)
    assert via_to_qubit == via_strings


@pytest.mark.parametrize(
    ("term_sum_type", "string_type", "source", "pieces"),
    (
        (
            zfon.ComplexTermSum,
            zfon.String,
            "((2j), F0^), ((3), F1^ F1)",
            (("F0^", 2j), ("F1^ F1", 3.0)),
        ),
        (
            zfog.ComplexTermSum,
            zfog.String,
            "((2j), F0 F1^), ((3), F1 F0^)",
            (("F0 F1^", 2j), ("F1 F0^", 3.0)),
        ),
    ),
)
def test_to_qubit_complex_terms(term_sum_type, string_type, source, pieces):
    mapper = JordanWignerMapper(2)
    fermion_terms = term_sum_type.from_str(source, 2)

    via_to_qubit = fermion_terms.to_qubit()
    via_strings = zqo.ComplexTermSum(mapper.qubits)
    for string_source, coeff in pieces:
        via_strings += coeff * mapper.apply(string_type(2, string_source))

    assert isinstance(via_to_qubit, zqo.ComplexTermSum)
    assert via_to_qubit == via_strings


def test_mapper():
    class StringLength(Mapper):
        def apply(self, value: str, /) -> int:
            return len(value)

    assert StringLength().apply("abc") == 3


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_apply_real_fermion_operator(mapper_type):
    mapper = mapper_type(2)

    mapped = mapper.apply(zfon.String(2, "F0^ F1"), real=True)
    mapped += mapper.apply(zfon.String(2, "F1^ F0"), real=True)

    expected = mapper.apply(zfon.String(2, "F0^ F1"))
    expected += mapper.apply(zfon.String(2, "F1^ F0"))
    assert isinstance(mapped, zqo.RealTermSum)
    assert mapped.into(zqo.ComplexTermSum) == expected


def test_apply_state_accepts_real_none():
    mapped = JordanWignerMapper(2).apply(zfs.String(2), real=None)

    assert isinstance(mapped, zqs.String)


@pytest.mark.parametrize("real", (False, True))
def test_apply_state_rejects_non_none_real(real):
    with pytest.raises(TypeError, match="real must be None for fermionic state strings"):
        JordanWignerMapper(2).apply(zfs.String(2), real=real)


def test_to_qubit_type_hints():
    assert get_type_hints(zfon.RealTermSum.to_qubit)["return"] is zqo.ComplexTermSum


@pytest.mark.parametrize(
    "fermion_terms",
    (
        zfon.RealTermSum.from_str("F0^", 2),
        zfog.RealTermSum.from_str("F0^", 2),
    ),
)
def test_to_qubit_accepts_qubits(fermion_terms):
    qubits = Qubits.from_count(2)

    inferred = fermion_terms.to_qubit()
    from_scalar = fermion_terms.to_qubit(qubits=2)
    from_qubits = fermion_terms.to_qubit(qubits=qubits)

    assert from_scalar == inferred
    assert from_qubits == inferred


def test_to_qubit_rejects_mapper_instance():
    mapper = JordanWignerMapper(2)
    fermion_terms = zfon.RealTermSum.from_str("F0^", 2)

    with pytest.raises(TypeError, match="is not callable"):
        fermion_terms.to_qubit(mapper=mapper)


@pytest.mark.parametrize(
    ("mapper_type", "expected"),
    (
        (JordanWignerMapper, {0, 2}),
        (BravyiKitaevMapper, {0, 1, 2}),
        (ParityMapper, {0, 1}),
        (ParaparticularMapper, {0, 2}),
    ),
)
def test_apply_state_strings(mapper_type, expected):
    mapped = mapper_type(4).apply(zfs.String(4, {0, 2}))

    assert isinstance(mapped, zqs.String)
    assert mapped.get_set() == expected


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_apply_vacuum_state(mapper_type):
    assert mapper_type(4).apply(zfs.String(4)).is_vacuum()


def test_apply_operator_and_state_use_same_impl():
    mapper = JordanWignerMapper(2)
    impl = mapper._impl

    mapper.apply(zfon.String(2, "F0^"))
    mapper.apply(zfs.String(2, {0}))

    assert mapper._impl is impl


def test_apply_state_respects_mode_ordering():
    mapped = JordanWignerMapper(4, mode_ordering=[3, 2, 1, 0]).apply(zfs.String(4, {0, 2}))

    assert mapped.get_set() == {1, 3}


def test_apply_state_rejects_different_mode_count():
    with pytest.raises(ValueError, match="mode count must equal qubit count"):
        JordanWignerMapper(4).apply(zfs.String(2, {0}))


def test_apply_rejects_unsupported_input():
    with pytest.raises(TypeError, match="Cannot map an instance of object"):
        JordanWignerMapper(2).apply(object())


def test_state_to_qubit_preserves_real_coefficients():
    state = zfs.RealTermSum.from_str("(2, [1, 0, 1, 0]), (-1, [0, 1, 0, 1])", 4)

    mapped = state.to_qubit(mapper=ParityMapper)

    assert isinstance(mapped, zqs.RealTermSum)
    assert str(mapped) == "(2.0, [1, 1, 0, 0]), (-1.0, [0, 1, 1, 0])"


def test_state_to_qubit_preserves_complex_coefficients():
    state = zfs.ComplexTermSum.from_str("((2j), [1, 0, 1, 0]), ((3), [0, 1, 0, 1])", 4)

    mapped = state.to_qubit(mapper=ParityMapper)

    assert isinstance(mapped, zqs.ComplexTermSum)
    assert str(mapped) == "(2j, [1, 1, 0, 0]), ((3+0j), [0, 1, 1, 0])"


def test_state_to_qubit_preserves_symbolic_coefficients():
    x = Symbol("x")
    terms = zfs.SymbolicTerms.from_iterable([({0, 2}, x), ({1, 3}, 2 * x)], 4)
    state = zfs.SymbolicTermSum.from_terms(terms)

    mapped = state.to_qubit(mapper=ParityMapper)

    assert isinstance(mapped, zqs.SymbolicTermSum)
    assert str(mapped) == "(x, [1, 1, 0, 0]), (2*x, [0, 1, 1, 0])"


def test_state_to_qubit_defaults_to_jordan_wigner():
    state = zfs.RealTermSum.from_str("(2, [1, 0])", 2)

    inferred = state.to_qubit()
    from_scalar = state.to_qubit(qubits=2)
    from_qubits = state.to_qubit(qubits=Qubits.from_count(2))

    assert inferred == zfs.RealTermSum.from_str("(2, [1, 0])", 2).to_qubit(
        mapper=JordanWignerMapper
    )
    assert from_scalar == inferred
    assert from_qubits == inferred


def test_state_to_qubit_rejects_mapper_instance():
    state = zfs.RealTermSum.from_str("(2, [1, 0])", 2)

    with pytest.raises(TypeError, match="is not callable"):
        state.to_qubit(mapper=JordanWignerMapper(2))


def test_state_to_qubit_rejects_different_mode_count():
    with pytest.raises(ValueError, match="mode count must equal qubit count"):
        zfs.RealTermSum(2).to_qubit(qubits=4)


def test_state_to_qubit_type_hints():
    assert get_type_hints(zfs.RealTermSum.to_qubit)["return"] is zqs.RealTermSum
    assert get_type_hints(zfs.ComplexTermSum.to_qubit)["return"] is zqs.ComplexTermSum
    assert get_type_hints(zfs.SymbolicTermSum.to_qubit)["return"] is zqs.SymbolicTermSum


_NUMERIC_OPERATOR_TERMS = [
    pytest.param(getattr(family, prefix + "Term"), coefficient, id=f"{family.__name__}.{prefix}")
    for family in (zfog, zfon)
    for prefix, coefficient in (
        ("Sign", -1),
        ("ComplexSign", 1j),
        ("Real", 2.5),
        ("Complex", 1.0 + 2.0j),
    )
    if hasattr(family, prefix + "Term")
]


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
@pytest.mark.parametrize("term_type,coefficient", _NUMERIC_OPERATOR_TERMS)
@pytest.mark.parametrize("real", [False, True])
def test_apply_operator_term(mapper_type, term_type, coefficient, real):
    coefficient = term_type.coeffs_type.coeff_type(coefficient)
    source = term_type(2, ("F0^ F1", coefficient))
    before = source.to_dict()
    mapper = mapper_type(2)
    expected = mapper.apply(source.string) * complex(coefficient)

    mapped = mapper.apply(source, real=real)

    assert type(mapped) is (zqo.RealTermSum if real else zqo.ComplexTermSum)
    assert mapped == (expected.real_part if real else expected)
    assert source.to_dict() == before
    assert mapped.strings.is_owning()
    assert mapped.coeffs.is_owning()


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
@pytest.mark.parametrize("family", [zfog, zfon])
@pytest.mark.parametrize("prefix", ["Real", "Complex"])
@pytest.mark.parametrize("real", [False, True])
def test_apply_operator_sum(mapper_type, family, prefix, real):
    cls = getattr(family, prefix + "TermSum")
    values = [2.0, -1.0, 0.25] if prefix == "Real" else [2.0 + 1j, -1j, 0.25 - 0.5j]
    kwargs = {"max_string_len": 2} if family is zfog else {}
    source = cls.from_iterable(zip(["F0^", "F1", "F0^ F1"], values, strict=True), 2, **kwargs)
    before = source.to_dict()
    mapper = mapper_type(2)
    expected = zqo.ComplexTermSum(2)
    for term in source:
        expected += mapper.apply(term.string) * complex(term.coeff)

    mapped = mapper.apply(source, real=real)

    assert type(mapped) is (zqo.RealTermSum if real else zqo.ComplexTermSum)
    assert mapped == (expected.real_part if real else expected)
    assert source.to_dict() == before


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_apply_complex_term_projects_after_scaling(mapper_type):
    source = zfon.ComplexTerm(2, ("F0^", 1j))
    mapper = mapper_type(2)
    expected = -mapper.apply(source.string).imag_part

    mapped = mapper.apply(source, real=True)

    assert mapped == expected
    assert any(coeff != 0 for coeff in mapped.coeffs)


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_apply_sum_accumulates_cancellations(mapper_type):
    source = zfog.RealTermSum.from_iterable(
        [("F0 F0^", 1.0), ("F0^ F0", 1.0), ("", -1.0)], 2, max_string_len=2
    )

    mapped = mapper_type(2).apply(source)

    assert len(mapped.filter_nonzero()) == 0
    assert len(source) == 3


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
@pytest.mark.parametrize("family", [zfog, zfon])
@pytest.mark.parametrize("prefix", ["Real", "Complex"])
@pytest.mark.parametrize("real", [False, True])
def test_apply_empty_operator_sum(mapper_type, family, prefix, real):
    source = getattr(family, prefix + "TermSum")(2)
    qubits = Qubits.from_inds([5, 2])
    expected_type = (
        zqo.SymbolicTermSum
        if prefix == "Symbolic"
        else zqo.RealTermSum
        if real
        else zqo.ComplexTermSum
    )

    mapped = mapper_type(qubits).apply(source, real=real)

    assert type(mapped) is expected_type
    assert len(mapped) == 0
    assert mapped.qubits == qubits


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
@pytest.mark.parametrize("family", [zfog, zfon])
def test_apply_operator_term_view_and_mode_ordering(mapper_type, family):
    kwargs = {"max_string_len": 1} if family is zfog else {}
    terms = family.RealTerms.from_iterable([("F0^", 2.0), ("F1", -3.0)], 2, **kwargs)
    source = terms[::-1][0]
    before = terms.to_dict()
    qubits = Qubits.from_inds([5, 2])
    mapper = mapper_type(qubits, mode_ordering=[1, 0])
    expected = mapper.apply(source.string) * source.coeff

    mapped = mapper.apply(source)

    assert mapped == expected
    assert mapped.qubits == qubits
    mapped.coeffs[0] = 10.0
    assert terms.to_dict() == before


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
@pytest.mark.parametrize("container", ["Term", "TermSum"])
def test_apply_operator_out_of_bounds(mapper_type, container):
    source = (
        zfon.RealTerm(3, ("F2^", 2.0))
        if container == "Term"
        else zfon.RealTermSum.from_str("(2.0, F2^)", 3)
    )
    with pytest.raises(ValueError, match="mode index is outside"):
        mapper_type(2).apply(source)


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_apply_operator_allows_larger_register(mapper_type):
    source = zfon.RealTermSum.from_str("(2.0, F0^)", 2)
    mapper = mapper_type(4)

    mapped = mapper.apply(source)

    assert mapped == mapper.apply(zfon.String(2, "F0^")) * 2.0
    assert len(mapped.qubits) == 4


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
@pytest.mark.parametrize("family", [zfog, zfon])
@pytest.mark.parametrize("prefix", ["Real", "Complex"])
def test_operator_to_qubit_matches_mapper(mapper_type, family, prefix):
    coefficient = 1.0
    kwargs = {"max_string_len": 1} if family is zfog else {}
    source = getattr(family, prefix + "TermSum").from_iterable([("F0^", coefficient)], 2, **kwargs)
    qubits = Qubits.from_inds([5, 2])

    mapped = source.to_qubit(mapper=mapper_type, qubits=qubits)

    assert mapped == mapper_type(qubits).apply(source)
    assert type(mapped) is zqo.ComplexTermSum


def test_operator_to_qubit_accepts_string_only_mapper():
    class StringOnlyMapper(Mapper):
        def __init__(self, qubits):
            self.mapper = JordanWignerMapper(qubits)

        def apply(self, value, /):
            assert isinstance(value, zfon.String)
            return self.mapper.apply(value)

    source = zfon.RealTermSum.from_str("(2.0, F0^), (-1.0, F1)", 2)
    assert source.to_qubit(mapper=StringOnlyMapper) == JordanWignerMapper(2).apply(source)


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
@pytest.mark.parametrize("family", [zfog, zfon])
@pytest.mark.parametrize("container", ["Term", "TermSum", "empty"])
@pytest.mark.parametrize("real", [False, True])
def test_apply_operator_rejects_symbolic_coefficients(mapper_type, family, container, real):
    if container == "Term":
        source = family.SymbolicTerm(2, ("F0^", Symbol("x")))
    elif container == "TermSum":
        kwargs = {"max_string_len": 1} if family is zfog else {}
        source = family.SymbolicTermSum.from_iterable([("F0^", Symbol("x"))], 2, **kwargs)
    else:
        source = family.SymbolicTermSum(2)
    with pytest.raises(TypeError, match="Only numeric fermionic operator coefficients"):
        mapper_type(2).apply(source, real=real)
