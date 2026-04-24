import pytest

from zixy._zixy import QubitPauliArray
from zixy.qubit import Qubits
from zixy.qubit.pauli import I, String, X, Y, Z
from zixy.qubit.springs import PauliSprings


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
    
    assert str(result) == str(expected)
    
    # All phases should be i (represented as 1 in the phase encoding)
    assert phases[0] == 1  # i from X*Y
    assert phases[1] == 1  # i from Y*Z
    assert phases[2] == 1  # i from Z*X


def test_cmpnts_mul_pairwise_different_lengths():
    # Test error handling for arrays of different lengths
    qubits = Qubits.from_count(2)
    
    arr1 = QubitPauliArray(qubits, PauliSprings("X0, Y0"))  # 2 strings
    arr2 = QubitPauliArray(qubits, PauliSprings("Z0"))       # 1 string
    
    with pytest.raises(ValueError, match="Input arrays must have the same length"):
        QubitPauliArray.cmpnts_mul_pairwise(arr1, arr2)
