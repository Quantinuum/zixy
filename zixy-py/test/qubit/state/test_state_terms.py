from zixy.container.coeffs import Sign
from zixy.qubit.state import RealTerm, RealTermSum, SignTerm


def test_term():
    term = SignTerm(6, None)
    assert term.coeff == Sign(False)
    term.string.set((1, 0) * 3)
    assert str(term) == "(+1, [1, 0, 1, 0, 1, 0])"
    term *= -1
    assert str(term) == "(-1, [1, 0, 1, 0, 1, 0])"
    assert str(term * Sign(True)) == "(+1, [1, 0, 1, 0, 1, 0])"


def test_term_real_sum():
    lc = RealTermSum(6)
    assert len(lc._impl._cmpnts._impl.qubits) == 6
    assert len(lc) == 0
    assert str(lc) == ""
    lc += RealTerm(6, ((1, 0, 0, 0, 0, 1), 3.0))
    assert len(lc) == 1
    assert str(lc) == "(3.0, [1, 0, 0, 0, 0, 1])"
    lc += RealTerm(6, ((1, 1, 0, 0, 1, 1), 4.0))
    assert len(lc) == 2
    assert str(lc) == "(3.0, [1, 0, 0, 0, 0, 1]), (4.0, [1, 1, 0, 0, 1, 1])"
    assert lc.l1_norm == 3 + 4
    assert lc.l2_norm_square == 3**2 + 4**2
