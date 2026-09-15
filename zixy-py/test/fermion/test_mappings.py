"""Tests for fermionic-string mappers."""

from typing import get_type_hints

import pytest
from sympy import Symbol

from zixy._zixy import Qubits
from zixy.fermion.operator.general import (
    ComplexTermSum as GeneralComplexTermSum,
    RealTermSum as GeneralRealTermSum,
    String as GeneralString,
)
from zixy.fermion.operator.normal import (
    ComplexTermSum as NormalComplexTermSum,
    RealTermSum as NormalRealTermSum,
    String as NormalString,
)
from zixy.fermion.state import (
    ComplexTermSum as FermionComplexState,
    RealTermSum as FermionRealState,
    String as FermionStateString,
    SymbolicTerms as FermionSymbolicStateTerms,
    SymbolicTermSum as FermionSymbolicState,
)
from zixy.mappings import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    Mapper,
    ParaparticularMapper,
    ParityMapper,
)
from zixy.qubit.pauli import ComplexTermSum as PauliComplexTermSum
from zixy.qubit.state import (
    ComplexTermSum as QubitComplexState,
    RealTermSum as QubitRealState,
    String as QubitStateString,
    SymbolicTermSum as QubitSymbolicState,
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
            NormalString,
            "F0^",
            "((0.5+0j), X0), (-0.5j, Y0)",
        ),
        (
            BravyiKitaevMapper,
            2,
            None,
            NormalString,
            "F0^",
            "((0.5+0j), X0 X1), (-0.5j, Y0 X1)",
        ),
        (
            ParityMapper,
            2,
            None,
            NormalString,
            "F0^",
            "((0.5+0j), X0 X1), (-0.5j, Y0 X1)",
        ),
        (
            ParaparticularMapper,
            2,
            None,
            NormalString,
            "F0^",
            "((0.5+0j), X0), (-0.5j, Y0)",
        ),
        (
            JordanWignerMapper,
            4,
            None,
            NormalString,
            "F1^",
            "((0.5+0j), Z0 X1), (-0.5j, Z0 Y1)",
        ),
        (
            BravyiKitaevMapper,
            4,
            None,
            NormalString,
            "F1^",
            "((0.5+0j), Z0 X1 X3), (-0.5j, Y1 X3)",
        ),
        (
            ParityMapper,
            4,
            None,
            NormalString,
            "F1^",
            "((0.5+0j), Z0 X1 X2 X3), (-0.5j, Y1 X2 X3)",
        ),
        (
            ParaparticularMapper,
            4,
            None,
            NormalString,
            "F1^",
            "((0.5+0j), X1), (-0.5j, Y1)",
        ),
        (
            JordanWignerMapper,
            2,
            [1, 0],
            NormalString,
            "F0^",
            "((0.5+0j), X1), (-0.5j, Y1)",
        ),
        (
            BravyiKitaevMapper,
            2,
            [1, 0],
            NormalString,
            "F0^",
            "((0.5+0j), X0 X1), (-0.5j, X0 Y1)",
        ),
        (
            ParityMapper,
            2,
            [1, 0],
            NormalString,
            "F0^",
            "((0.5+0j), X0 X1), (-0.5j, X0 Y1)",
        ),
        (
            ParaparticularMapper,
            2,
            [1, 0],
            NormalString,
            "F0^",
            "((0.5+0j), X1), (-0.5j, Y1)",
        ),
        (
            JordanWignerMapper,
            2,
            None,
            GeneralString,
            "F0 F1^",
            "(-0.25j, Y0 X1), ((-0.25+0j), X0 X1), " "((-0.25+0j), Y0 Y1), (0.25j, X0 Y1)",
        ),
    ),
)
def test_apply_strings(mapper_type, qubits, mode_ordering, string_type, source, expected):
    mapper = mapper_type(qubits, mode_ordering=mode_ordering)

    terms = mapper.apply(string_type(qubits, source))

    assert isinstance(terms, PauliComplexTermSum)
    assert str(terms) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "F2^ F1",
            "(0.25j, Y1 X2), ((0.25+0j), Y1 Y2), " "((0.25+0j), X1 X2), (-0.25j, X1 Y2)",
        ),
        (
            "F0^ F0 F1^ F1",
            "((0.25+0j), ), ((-0.25+0j), Z0), " "((-0.25+0j), Z1), ((0.25+0j), Z0 Z1)",
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

    terms = mapper.apply(GeneralString(4, source))

    assert isinstance(terms, PauliComplexTermSum)
    assert str(terms) == expected


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_apply_adopts_native_map_without_python_reinsertion(mapper_type, monkeypatch):
    def fail(*_args):
        raise AssertionError("mapped terms were reinserted in Python")

    monkeypatch.setattr(PauliComplexTermSum, "insert_iterable", fail)

    terms = mapper_type(2).apply(GeneralString(2, "F0 F1^"))

    assert len(terms) == 4
    found = terms.lookup(terms.strings[0])
    assert found is not None
    assert found[0] == 0


def test_apply_products_anticommute():
    mapper = JordanWignerMapper(4)
    caca = mapper.apply(GeneralString(4, "F0^ F1 F2^ F3"))
    ccaa = mapper.apply(GeneralString(4, "F0^ F2^ F1 F3"))

    assert caca == -ccaa


def test_apply_scaled_sums():
    mapper = JordanWignerMapper(2)
    terms = PauliComplexTermSum(mapper.qubits)

    terms += 2.0 * mapper.apply(GeneralString(2, "F0^ F0"))
    terms += -mapper.apply(GeneralString(2, "F1^ F1"))

    assert str(terms) == "((0.5+0j), ), ((-1+0j), Z0), ((0.5+0j), Z1)"


def test_to_qubit_number_operator():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0^ F0", 2)

    qubit_terms = fermion_terms.to_qubit()

    assert str(qubit_terms) == "((0.5+0j), ), ((-0.5+0j), Z0)"
    assert qubit_terms == mapper.apply(GeneralString(2, "F0^ F0"))


@pytest.mark.parametrize(
    ("term_sum_type", "string_type", "source", "pieces"),
    (
        (
            NormalRealTermSum,
            NormalString,
            "(2, F0^ F0), (-1, F1^ F1)",
            (("F0^ F0", 2.0), ("F1^ F1", -1.0)),
        ),
        (
            GeneralRealTermSum,
            GeneralString,
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
    via_strings = PauliComplexTermSum(mapper.qubits)
    for string_source, coeff in pieces:
        via_strings += coeff * mapper.apply(string_type(2, string_source))

    assert isinstance(via_to_qubit, PauliComplexTermSum)
    assert via_to_qubit == via_strings


@pytest.mark.parametrize(
    ("term_sum_type", "string_type", "source", "pieces"),
    (
        (
            NormalComplexTermSum,
            NormalString,
            "((2j), F0^), ((3), F1^ F1)",
            (("F0^", 2j), ("F1^ F1", 3.0)),
        ),
        (
            GeneralComplexTermSum,
            GeneralString,
            "((2j), F0 F1^), ((3), F1 F0^)",
            (("F0 F1^", 2j), ("F1 F0^", 3.0)),
        ),
    ),
)
def test_to_qubit_complex_terms(term_sum_type, string_type, source, pieces):
    mapper = JordanWignerMapper(2)
    fermion_terms = term_sum_type.from_str(source, 2)

    via_to_qubit = fermion_terms.to_qubit()
    via_strings = PauliComplexTermSum(mapper.qubits)
    for string_source, coeff in pieces:
        via_strings += coeff * mapper.apply(string_type(2, string_source))

    assert isinstance(via_to_qubit, PauliComplexTermSum)
    assert via_to_qubit == via_strings


def test_mapper():
    class StringLength(Mapper):
        def apply(self, value: str, /) -> int:
            return len(value)

    assert StringLength().apply("abc") == 3


def test_to_qubit_type_hints():
    assert get_type_hints(NormalRealTermSum.to_qubit)["return"] is PauliComplexTermSum


@pytest.mark.parametrize(
    "fermion_terms",
    (
        NormalRealTermSum.from_str("F0^", 2),
        GeneralRealTermSum.from_str("F0^", 2),
    ),
)
def test_to_qubit_accepts_qubits(fermion_terms):
    qubits = Qubits.from_count(2)

    inferred = fermion_terms.to_qubit()
    from_int = fermion_terms.to_qubit(qubits=2)
    from_qubits = fermion_terms.to_qubit(qubits=qubits)

    assert from_int == inferred
    assert from_qubits == inferred


def test_to_qubit_rejects_mapper_instance():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0^", 2)

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
    mapped = mapper_type(4).apply(FermionStateString(4, {0, 2}))

    assert isinstance(mapped, QubitStateString)
    assert mapped.get_set() == expected


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_apply_vacuum_state(mapper_type):
    assert mapper_type(4).apply(FermionStateString(4)).is_vacuum()


def test_apply_operator_and_state_use_same_impl():
    mapper = JordanWignerMapper(2)
    impl = mapper._impl

    mapper.apply(NormalString(2, "F0^"))
    mapper.apply(FermionStateString(2, {0}))

    assert mapper._impl is impl


def test_apply_state_respects_mode_ordering():
    mapped = JordanWignerMapper(4, mode_ordering=[3, 2, 1, 0]).apply(FermionStateString(4, {0, 2}))

    assert mapped.get_set() == {1, 3}


def test_apply_state_rejects_different_mode_count():
    with pytest.raises(ValueError, match="mode count must equal qubit count"):
        JordanWignerMapper(4).apply(FermionStateString(2, {0}))


def test_apply_rejects_unsupported_input():
    with pytest.raises(TypeError, match="Cannot map an instance of object"):
        JordanWignerMapper(2).apply(object())


def test_state_to_qubit_preserves_real_coefficients():
    state = FermionRealState.from_str("(2, [1, 0, 1, 0]), (-1, [0, 1, 0, 1])", 4)

    mapped = state.to_qubit(mapper=ParityMapper)

    assert isinstance(mapped, QubitRealState)
    assert str(mapped) == "(2.0, [1, 1, 0, 0]), (-1.0, [0, 1, 1, 0])"


def test_state_to_qubit_preserves_complex_coefficients():
    state = FermionComplexState.from_str("((2j), [1, 0, 1, 0]), ((3), [0, 1, 0, 1])", 4)

    mapped = state.to_qubit(mapper=ParityMapper)

    assert isinstance(mapped, QubitComplexState)
    assert str(mapped) == "(2j, [1, 1, 0, 0]), ((3+0j), [0, 1, 1, 0])"


def test_state_to_qubit_preserves_symbolic_coefficients():
    x = Symbol("x")
    terms = FermionSymbolicStateTerms.from_iterable([({0, 2}, x), ({1, 3}, 2 * x)], 4)
    state = FermionSymbolicState.from_terms(terms)

    mapped = state.to_qubit(mapper=ParityMapper)

    assert isinstance(mapped, QubitSymbolicState)
    assert str(mapped) == "(x, [1, 1, 0, 0]), (2*x, [0, 1, 1, 0])"


def test_state_to_qubit_defaults_to_jordan_wigner():
    state = FermionRealState.from_str("(2, [1, 0])", 2)

    inferred = state.to_qubit()
    from_int = state.to_qubit(qubits=2)
    from_qubits = state.to_qubit(qubits=Qubits.from_count(2))

    assert inferred == FermionRealState.from_str("(2, [1, 0])", 2).to_qubit(
        mapper=JordanWignerMapper
    )
    assert from_int == inferred
    assert from_qubits == inferred


def test_state_to_qubit_rejects_mapper_instance():
    state = FermionRealState.from_str("(2, [1, 0])", 2)

    with pytest.raises(TypeError, match="is not callable"):
        state.to_qubit(mapper=JordanWignerMapper(2))


def test_state_to_qubit_rejects_different_mode_count():
    with pytest.raises(ValueError, match="mode count must equal qubit count"):
        FermionRealState(2).to_qubit(qubits=4)


def test_state_to_qubit_type_hints():
    assert get_type_hints(FermionRealState.to_qubit)["return"] is QubitRealState
    assert get_type_hints(FermionComplexState.to_qubit)["return"] is QubitComplexState
    assert get_type_hints(FermionSymbolicState.to_qubit)["return"] is QubitSymbolicState
