from zixy.fermion.mappings import JordanWignerMapper
from zixy.fermion.operator.general import String as GeneralString
from zixy.fermion.operator.normal import (
    RealTermSum as NormalRealTermSum,
    String as NormalString,
)
from zixy.qubit.pauli import ComplexTermSum as PauliComplexTermSum


def test_real_term_sum_to_qubit_handles_identity_contractions():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0 F0^", 2)

    qubit_terms = fermion_terms.to_qubit(mapper)

    assert str(qubit_terms) == "((0.5+0j), ), ((0.5+0j), Z0)"


def test_native_and_explicit_string_mapping_agree():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("F0 F0^", 2)

    via_native = fermion_terms.to_qubit(mapper)
    via_string = mapper.encode(GeneralString(2, "F0 F0^"))

    assert via_native == via_string


def test_mapper_accepts_native_normal_string():
    mapper = JordanWignerMapper(2)

    terms = mapper.encode(NormalString(2, "F0^"))

    assert isinstance(terms, PauliComplexTermSum)
    assert str(terms) == "((0.5+0j), X0), (-0.5j, Y0)"


def test_mapper_helper_methods_match_explicit_products():
    mapper = JordanWignerMapper(4)

    assert str(mapper.encode_ca(2, 1)) == (
        "(0.25j, Y1 X2), ((0.25+0j), Y1 Y2), " "((0.25+0j), X1 X2), (-0.25j, X1 Y2)"
    )

    explicit = mapper.encode(GeneralString(4, "F0^ F0 F1^ F1"))
    helper = mapper.encode_nn(0, 1)
    assert helper == explicit

    caca = mapper.encode_caca(0, 1, 2, 3)
    ccaa = mapper.encode_ccaa(0, 2, 1, 3)
    assert isinstance(caca, PauliComplexTermSum)
    assert isinstance(ccaa, PauliComplexTermSum)
    assert len(caca) == 16
    assert len(ccaa) == 16


def test_scaled_encoded_term_sums_add():
    mapper = JordanWignerMapper(2)
    terms = PauliComplexTermSum(mapper.qubits)

    terms += 2.0 * mapper.encode_n(0)
    terms += -mapper.encode_n(1)

    assert str(terms) == "((0.5+0j), ), ((-1+0j), Z0), ((0.5+0j), Z1)"


def test_native_normal_term_sum_to_qubit_matches_helpers():
    mapper = JordanWignerMapper(2)
    fermion_terms = NormalRealTermSum.from_str("(2, F0^ F0), (-1, F1^ F1)", 2)

    via_native = fermion_terms.to_qubit(mapper)
    via_helpers = PauliComplexTermSum(mapper.qubits)
    via_helpers += 2.0 * mapper.encode_n(0)
    via_helpers += -mapper.encode_n(1)

    assert via_native == via_helpers


def test_mapper_accepts_general_string():
    mapper = JordanWignerMapper(2)

    terms = mapper.encode(GeneralString(2, "F0 F1^"))

    assert isinstance(terms, PauliComplexTermSum)
    assert str(terms) == (
        "(-0.25j, Y0 X1), ((-0.25+0j), X0 X1), " "((-0.25+0j), Y0 Y1), (0.25j, X0 Y1)"
    )
