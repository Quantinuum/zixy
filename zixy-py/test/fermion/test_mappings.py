import pytest

from zixy._zixy import Qubits
from zixy.fermion.mappings import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParaparticularMapper,
    ParityMapper,
)
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
from zixy.qubit.pauli import (
    ComplexTermSum as PauliComplexTermSum,
    RealTermSum as PauliRealTermSum,
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
def test_encode_strings(mapper_type, qubits, mode_ordering, string_type, source, expected):
    mapper = mapper_type(qubits, mode_ordering=mode_ordering)

    terms = mapper.encode(string_type(qubits, source))

    assert isinstance(terms, PauliComplexTermSum)
    assert str(terms) == expected


def test_encode_products():
    mapper = JordanWignerMapper(4)

    assert str(mapper.encode(GeneralString(4, "F2^ F1"))) == (
        "(0.25j, Y1 X2), ((0.25+0j), Y1 Y2), " "((0.25+0j), X1 X2), (-0.25j, X1 Y2)"
    )

    product = mapper.encode(GeneralString(4, "F0^ F0 F1^ F1"))
    assert product == mapper.encode(GeneralString(4, "F0^ F0 F1^ F1"))

    caca = mapper.encode(GeneralString(4, "F0^ F1 F2^ F3"))
    ccaa = mapper.encode(GeneralString(4, "F0^ F2^ F1 F3"))
    assert isinstance(caca, PauliComplexTermSum)
    assert isinstance(ccaa, PauliComplexTermSum)
    assert len(caca) == 16
    assert len(ccaa) == 16


def test_encode_scaled_sums():
    mapper = JordanWignerMapper(2)
    terms = PauliComplexTermSum(mapper.qubits)

    terms += 2.0 * mapper.encode(GeneralString(2, "F0^ F0"))
    terms += -mapper.encode(GeneralString(2, "F1^ F1"))

    assert str(terms) == "((0.5+0j), ), ((-1+0j), Z0), ((0.5+0j), Z1)"


def test_to_qubit_identity_contraction():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0 F0^", 2)

    qubit_terms = fermion_terms.to_qubit()

    assert str(qubit_terms) == "((0.5+0j), ), ((0.5+0j), Z0)"
    assert qubit_terms == mapper.encode(GeneralString(2, "F0 F0^"))


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

    via_native = fermion_terms.to_qubit(mapper=mapper_type)
    via_strings = PauliComplexTermSum(mapper.qubits)
    for string_source, coeff in pieces:
        via_strings += mapper.encode_complex(string_type(2, string_source), coeff)

    assert isinstance(via_native, PauliComplexTermSum)
    assert via_native == via_strings


@pytest.mark.parametrize(
    ("term_sum_type", "string_type", "source", "pieces"),
    (
        (
            NormalComplexTermSum,
            NormalString,
            "(2j, F0^), (3, F1^ F1)",
            (("F0^", 2j), ("F1^ F1", 3.0)),
        ),
        (
            GeneralComplexTermSum,
            GeneralString,
            "(2j, F0 F1^), (3, F1 F0^)",
            (("F0 F1^", 2j), ("F1 F0^", 3.0)),
        ),
    ),
)
def test_to_qubit_complex_terms(term_sum_type, string_type, source, pieces):
    mapper = JordanWignerMapper(2)
    fermion_terms = term_sum_type.from_str(source, 2)

    via_native = fermion_terms.to_qubit()
    via_strings = PauliComplexTermSum(mapper.qubits)
    for string_source, coeff in pieces:
        via_strings += mapper.encode(string_type(2, string_source), coeff)

    assert isinstance(via_native, PauliComplexTermSum)
    assert via_native == via_strings


@pytest.mark.parametrize("mapper_type", MAPPER_TYPES)
def test_encode_real(mapper_type):
    mapper = mapper_type(2)

    terms = PauliRealTermSum(mapper.qubits)
    terms += mapper.encode_real(NormalString(2, "F0^ F1"), 1.0)
    terms += mapper.encode_real(NormalString(2, "F1^ F0"), 1.0)

    assert isinstance(terms, PauliRealTermSum)
    assert terms == (
        mapper.encode_real(NormalString(2, "F0^ F1"))
        + mapper.encode_real(NormalString(2, "F1^ F0"))
    )


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
