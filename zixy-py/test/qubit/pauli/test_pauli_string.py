import pytest

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
