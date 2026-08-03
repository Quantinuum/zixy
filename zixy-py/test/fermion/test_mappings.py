from zixy.fermion.mappings import JordanWignerMapper
from zixy.fermion.operator.general import String as GeneralString
from zixy.fermion.operator.normal import (
    RealTermSum as NormalRealTermSum,
    String as NormalString,
)
from zixy.qubit.pauli import RealTermSum as PauliRealTermSum


def test_real_term_sum_to_qubit_handles_identity_contractions():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0 F0^", 2)

    qubit_terms = fermion_terms.to_qubit(mapper)

    assert str(qubit_terms) == "(0.5, ), (0.5, Z0)"


def test_existing_from_fermionic_sequence_remains_compatible():
    mapper = JordanWignerMapper(2)

    terms = PauliRealTermSum.from_fermionic(
        mapper.qubits,
        mapper,
        [([(0, False), (0, True)], 1.0)],
    )

    assert str(terms) == "(0.5, ), (0.5, Z0)"


def test_native_and_sequence_mapping_agree():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0 F0^", 2)

    via_native = fermion_terms.to_qubit(mapper)
    via_sequence = PauliRealTermSum.from_fermionic(
        mapper.qubits,
        mapper,
        [([(0, False), (0, True)], 1.0)],
    )

    assert via_native == via_sequence


def test_mapper_accepts_native_normal_string():
    mapper = JordanWignerMapper(2)

    contribution = mapper.encode(NormalString(2, "F0^"))

    assert contribution.coeff == 1


def test_mapper_helper_methods_match_explicit_products():
    mapper = JordanWignerMapper(4)

    assert mapper.encode_ca(2, 1).coeff == 1
    assert mapper.encode_ca(2, 1)._mapper is mapper

    explicit = mapper.encode(((0, True), (0, False), (1, True), (1, False)))
    helper = mapper.encode_nn(0, 1)
    assert helper.coeff == explicit.coeff
    assert helper._mapper is explicit._mapper

    caca = mapper.encode_caca(0, 1, 2, 3)
    ccaa = mapper.encode_ccaa(0, 2, 1, 3)
    assert caca.coeff == 1
    assert ccaa.coeff == 1


def test_scaled_contribution_adds_to_pauli_term_sum():
    mapper = JordanWignerMapper(2)
    terms = PauliRealTermSum(mapper.qubits)

    terms += 2.0 * mapper.encode_n(0)
    terms += -mapper.encode_n(1)

    assert str(terms) == "(0.5, ), (-1.0, Z0), (0.5, Z1)"


def test_native_normal_term_sum_to_qubit_matches_helpers():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("(2, F0^ F0), (-1, F1^ F1)", 2)

    via_native = fermion_terms.to_qubit(mapper)
    via_helpers = PauliRealTermSum(mapper.qubits)
    via_helpers += 2.0 * mapper.encode_n(0)
    via_helpers += -mapper.encode_n(1)

    assert via_native == via_helpers


def test_mapper_accepts_general_string():
    mapper = JordanWignerMapper(2)

    contribution = mapper.encode(GeneralString(2, "F0 F1^"))

    assert contribution.coeff == 1
