from zixy.fermion.mappings import JordanWignerMapper
from zixy.fermion.operator import normal
from zixy.qubit.pauli import RealTermSum as PauliRealTermSum


def test_real_term_sum_to_qubit_handles_identity_contractions():
    mapper = JordanWignerMapper(2)
    fermion_terms = normal.RealTermSum.from_str("F0 F0^", 2)

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
    fermion_terms = normal.RealTermSum.from_str("F0 F0^", 2)

    via_native = fermion_terms.to_qubit(mapper)
    via_sequence = PauliRealTermSum.from_fermionic(
        mapper.qubits,
        mapper,
        [([(0, False), (0, True)], 1.0)],
    )

    assert via_native == via_sequence


def test_mapper_accepts_native_normal_string():
    mapper = JordanWignerMapper(2)

    contribution = mapper.encode(normal.String(2, "F0^"))

    assert contribution.coeff == 1
