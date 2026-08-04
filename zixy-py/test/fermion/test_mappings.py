import pytest

from zixy._zixy import Qubits
from zixy.fermion.mappings import JordanWignerMapper
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


def test_real_term_sum_to_qubit_handles_identity_contractions():
    fermion_terms = NormalRealTermSum.from_str("F0 F0^", 2)

    qubit_terms = fermion_terms.to_qubit()

    assert str(qubit_terms) == "((0.5+0j), ), ((0.5+0j), Z0)"


def test_native_and_explicit_string_mapping_agree():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0 F0^", 2)

    via_native = fermion_terms.to_qubit()
    via_string = mapper.encode(GeneralString(2, "F0 F0^"))

    assert via_native == via_string


def test_mapper_accepts_native_normal_string():
    mapper = JordanWignerMapper(2)

    terms = mapper.encode(NormalString(2, "F0^"))

    assert isinstance(terms, PauliComplexTermSum)
    assert str(terms) == "((0.5+0j), X0), (-0.5j, Y0)"


def test_mapper_helper_methods_match_explicit_products():
    mapper = JordanWignerMapper(4)

    assert str(mapper.encode(GeneralString(4, "F2^ F1"))) == (
        "(0.25j, Y1 X2), ((0.25+0j), Y1 Y2), " "((0.25+0j), X1 X2), (-0.25j, X1 Y2)"
    )

    explicit = mapper.encode(GeneralString(4, "F0^ F0 F1^ F1"))
    helper = mapper.encode(GeneralString(4, "F0^ F0 F1^ F1"))
    assert helper == explicit

    caca = mapper.encode(GeneralString(4, "F0^ F1 F2^ F3"))
    ccaa = mapper.encode(GeneralString(4, "F0^ F2^ F1 F3"))
    assert isinstance(caca, PauliComplexTermSum)
    assert isinstance(ccaa, PauliComplexTermSum)
    assert len(caca) == 16
    assert len(ccaa) == 16


def test_scaled_encoded_term_sums_add():
    mapper = JordanWignerMapper(2)
    terms = PauliComplexTermSum(mapper.qubits)

    terms += 2.0 * mapper.encode(GeneralString(2, "F0^ F0"))
    terms += -mapper.encode(GeneralString(2, "F1^ F1"))

    assert str(terms) == "((0.5+0j), ), ((-1+0j), Z0), ((0.5+0j), Z1)"


def test_native_normal_term_sum_to_qubit_matches_helpers():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("(2, F0^ F0), (-1, F1^ F1)", 2)

    via_native = fermion_terms.to_qubit()
    via_helpers = PauliComplexTermSum(mapper.qubits)
    via_helpers += 2.0 * mapper.encode(GeneralString(2, "F0^ F0"))
    via_helpers += -mapper.encode(GeneralString(2, "F1^ F1"))

    assert via_native == via_helpers


def test_native_normal_real_term_sum_to_qubit_accepts_explicit_mapper():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("(2, F0^ F0), (-1, F1^ F1)", 2)

    via_native = fermion_terms.to_qubit(mapper=JordanWignerMapper)
    via_helpers = PauliComplexTermSum(mapper.qubits)
    via_helpers += mapper.encode_complex(NormalString(2, "F0^ F0"), 2.0)
    via_helpers += mapper.encode_complex(NormalString(2, "F1^ F1"), -1.0)

    assert isinstance(via_native, PauliComplexTermSum)
    assert via_native == via_helpers


def test_native_normal_real_term_sum_to_qubit_can_return_real_terms():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("(2, F0^ F0), (-1, F1^ F1)", 2)

    via_native = fermion_terms.to_qubit(real=True)
    via_helpers = PauliRealTermSum(mapper.qubits)
    via_helpers += mapper.encode_real(NormalString(2, "F0^ F0"), 2.0)
    via_helpers += mapper.encode_real(NormalString(2, "F1^ F1"), -1.0)

    assert isinstance(via_native, PauliRealTermSum)
    assert via_native == via_helpers


def test_native_normal_real_term_sum_to_qubit_rejects_nonhermitian_real_output():
    fermion_terms = NormalRealTermSum.from_str("F0^", 2)

    with pytest.raises(ValueError, match="non-Hermitian"):
        fermion_terms.to_qubit(real=True)


def test_native_normal_complex_term_sum_to_qubit_matches_string_mapping():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalComplexTermSum.from_str("(2j, F0^), (3, F1^ F1)", 2)

    via_native = fermion_terms.to_qubit()
    via_helpers = PauliComplexTermSum(mapper.qubits)
    via_helpers += mapper.encode_complex(NormalString(2, "F0^"), 2j)
    via_helpers += mapper.encode_complex(NormalString(2, "F1^ F1"), 3.0)

    assert isinstance(via_native, PauliComplexTermSum)
    assert via_native == via_helpers


def test_mapper_accepts_general_string():
    mapper = JordanWignerMapper(2)

    terms = mapper.encode(GeneralString(2, "F0 F1^"))

    assert isinstance(terms, PauliComplexTermSum)
    assert str(terms) == (
        "(-0.25j, Y0 X1), ((-0.25+0j), X0 X1), " "((-0.25+0j), Y0 Y1), (0.25j, X0 Y1)"
    )


def test_native_general_term_sum_to_qubit_matches_string_mapping():
    mapper = JordanWignerMapper(2)
    fermion_terms = GeneralRealTermSum.from_str("(2, F0 F1^)", 2)

    via_native = fermion_terms.to_qubit()
    via_string = mapper.encode(GeneralString(2, "F0 F1^")) * 2.0

    assert via_native == via_string


def test_native_general_real_term_sum_to_qubit_accepts_explicit_mapper_and_qubits():
    qubits = Qubits.from_count(2)
    mapper = JordanWignerMapper(qubits)
    fermion_terms = GeneralRealTermSum.from_str("(2, F0 F1^), (-3, F1 F0^)", 2)

    via_native = fermion_terms.to_qubit(mapper=JordanWignerMapper, qubits=qubits)
    via_helpers = PauliComplexTermSum(mapper.qubits)
    via_helpers += mapper.encode(GeneralString(2, "F0 F1^"), 2.0)
    via_helpers += mapper.encode(GeneralString(2, "F1 F0^"), -3.0)

    assert isinstance(via_native, PauliComplexTermSum)
    assert via_native == via_helpers


def test_native_general_complex_term_sum_to_qubit_matches_string_mapping():
    mapper = JordanWignerMapper(2)
    fermion_terms = GeneralComplexTermSum.from_str("(2j, F0 F1^), (3, F1 F0^)", 2)

    via_native = fermion_terms.to_qubit()
    via_helpers = PauliComplexTermSum(mapper.qubits)
    via_helpers += mapper.encode(GeneralString(2, "F0 F1^"), 2j)
    via_helpers += mapper.encode(GeneralString(2, "F1 F0^"), 3.0)

    assert isinstance(via_native, PauliComplexTermSum)
    assert via_native == via_helpers


def test_term_sum_to_qubit_accepts_explicit_qubits():
    qubits = Qubits.from_count(2)
    fermion_terms = NormalRealTermSum.from_str("F0^", 2)

    inferred = fermion_terms.to_qubit()
    from_int = fermion_terms.to_qubit(qubits=2)
    from_qubits = fermion_terms.to_qubit(qubits=qubits)

    assert from_int == inferred
    assert from_qubits == inferred


def test_term_sum_to_qubit_rejects_mapper_instance():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0^", 2)

    with pytest.raises(TypeError, match="is not callable"):
        fermion_terms.to_qubit(mapper=mapper)
