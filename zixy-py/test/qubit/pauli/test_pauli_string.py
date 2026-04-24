import pytest

from zixy._zixy import QubitPauliArray, PauliSprings
from zixy.qubit import Qubits
from zixy.qubit.pauli import I, String, X, Y, Z


def test_from_str():
    source = "X0 Y4 Z7"
    string = String(8, (X, I, I, I, Y, I, I, Z))
    assert str(string) == source
    string = String(8, source)
    assert str(string) == source
    with pytest.raises(ValueError) as err:
        string.set("Y0, X1 Y3")
    assert str(err.value) == "String input has more than one component."
    assert str(string) == source
    # infer the qubits value from the highest mode index in the source
    assert String(source=source) == string
    assert String(9, source=source) != string


def test_cmpnts_mul_pairwise():
    # Create a 2-qubit system
    qubits = Qubits.from_count(2)

    # Create first array with 3 Pauli strings: X0, Y0, Z0
    springs1 = PauliSprings("X0, Y0, Z0")
    arr1 = QubitPauliArray(qubits, springs1)

    # Create second array with 3 Pauli strings: Y0, Z0, X0
    springs2 = PauliSprings("Y0, Z0, X0")
    arr2 = QubitPauliArray(qubits, springs2)

    # Test the pairwise multiplication
    result, phases = QubitPauliArray.cmpnts_mul_pairwise(arr1, arr2)

    # Expected results:
    # X0 * Y0 = iZ0
    # Y0 * Z0 = iX0
    # Z0 * X0 = iY0
    expected_springs = PauliSprings("Z0, X0, Y0")
    expected = QubitPauliArray(qubits, expected_springs)

    # Compare using equality operator
    assert result == expected

    # Verify phases are returned (they represent the complex sign of each product)
    assert len(phases) == 3
    # Phases should be: i from X*Y, i from Y*Z, i from Z*X
    assert str(phases[0]) == "+i"
    assert str(phases[1]) == "+i"
    assert str(phases[2]) == "+i"


def test_cmpnts_mul_pairwise_different_lengths():
    # Test error handling for arrays of different lengths
    qubits = Qubits.from_count(2)

    arr1 = QubitPauliArray(qubits, PauliSprings("X0, Y0"))  # 2 strings
    arr2 = QubitPauliArray(qubits, PauliSprings("Z0"))  # 1 string

    with pytest.raises(ValueError, match="Input arrays must have the same length"):
        QubitPauliArray.cmpnts_mul_pairwise(arr1, arr2)
