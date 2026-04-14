//! Defines a basic set of Clifford gates and some related utilities.

use std::fmt::Display;

use num_complex::Complex64;

use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::proj::Borrow;
use crate::container::traits::proj::BorrowMut;
use crate::container::utils::DistinctPair;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::cmpnt_major::cmpnt::PauliWord;
use crate::qubit::traits::{PauliWordMutRef, PauliWordRef};

/// Enum with a variant for each kind of Clifford gate implemented
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Gate {
    H(usize),                // Hadamard
    S(usize),                // Square root of Z
    CX(DistinctPair<usize>), // controlled not. 0: control bit, 1: target bit
}

pub fn h(i: usize) -> Gate {
    Gate::H(i)
}
pub fn s(i: usize) -> Gate {
    Gate::S(i)
}
pub fn cx(i_control: usize, i_target: usize) -> Option<Gate> {
    DistinctPair::new(i_control, i_target).map(Gate::CX)
}

impl Display for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Gate::H(i) => format!("H({i})"),
                Gate::S(i) => format!("S({i})"),
                Gate::CX(ij) => format!("CX({}, {})", ij.get().0, ij.get().1),
            }
        )
    }
}

/// Return the gate as a sparse matrix and an integer.
/// the sparse matrix returned multiplied by (the inverse sqrt of 2) to the power of the integer returned is
/// the unitary representation of the given gate.
fn to_sparse_and_n_isqrt2(
    gate: Gate,
    n_qubit: usize,
    big_endian: bool,
) -> (sprs::CsMat<Complex64>, usize) {
    use crate::qubit::mode::PauliMatrix::*;
    let mut tmp = PauliWord::new(Qubits::from_count(n_qubit));
    match gate {
        Gate::H(i) => {
            // H = (X + Z) / sqrt(2)
            tmp.borrow_mut().set_pauli_unchecked(i, X);
            let mat_x = tmp.borrow().to_sparse_matrix(big_endian);
            tmp.borrow_mut().set_pauli_unchecked(i, Z);
            let mat_z = tmp.borrow().to_sparse_matrix(big_endian);
            (&mat_x + &mat_z, 1)
        }
        Gate::S(i) => {
            // S = ((i+1)I + (1-i)Z) / 2
            let mut mat_i = tmp.borrow().to_sparse_matrix(big_endian);
            mat_i *= Complex64::new(1.0, 1.0);
            tmp.borrow_mut().set_pauli_unchecked(i, Z);
            let mut mat_z = tmp.borrow().to_sparse_matrix(big_endian);
            mat_z *= Complex64::new(1.0, -1.0);
            (&mat_i + &mat_z, 2)
        }
        Gate::CX(distinct_pair) => {
            // CX = (II + ZI - ZX + IX) / 2
            let (i_control, i_target) = distinct_pair.get();
            let mut mat = tmp.borrow().to_sparse_matrix(big_endian);
            tmp.borrow_mut().set_pauli_unchecked(i_control, Z);
            let mat_zi = tmp.borrow().to_sparse_matrix(big_endian);
            tmp.borrow_mut().set_pauli_unchecked(i_target, X);
            let mat_zx = tmp.borrow().to_sparse_matrix(big_endian);
            tmp.borrow_mut().set_pauli_unchecked(i_control, I);
            let mat_ix = tmp.borrow().to_sparse_matrix(big_endian);
            mat = &mat + &mat_zi;
            mat = &mat - &mat_zx;
            mat = &mat + &mat_ix;
            (mat, 2)
        }
    }
}

/// Return the unitary representation of the given gate as a sparse matrix with dimensionality given by the number of qubits.
fn to_sparse_matrix(gate: Gate, n_qubit: usize, big_endian: bool) -> sprs::CsMat<Complex64> {
    let (mut mat, n_isqrt2) = to_sparse_and_n_isqrt2(gate, n_qubit, big_endian);
    mat /= Complex64::new((2 * (n_isqrt2 / 2)) as f64, 0.0);
    if n_isqrt2 & 1 != 0 {
        mat *= Complex64::ISQRT2;
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubit::mode::{PauliMatrix, Qubits};
    use crate::qubit::sparse_matrix::conj;
    use rstest::rstest;
    use PauliMatrix::*;

    fn z(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn test_gate_matrices() {
        let (sparse, n_isqrt2) = to_sparse_and_n_isqrt2(h(0), 1, true);
        let dense = sparse.to_dense();
        let vec = dense
            .rows()
            .into_iter()
            .flat_map(|f| f.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(
            vec,
            vec![z(1.0, 0.0), z(1.0, 0.0), z(1.0, 0.0), z(-1.0, 0.0)]
        );
        assert_eq!(n_isqrt2, 1);

        let (sparse, n_isqrt2) = to_sparse_and_n_isqrt2(s(0), 1, true);
        let dense = sparse.to_dense();
        let vec = dense
            .rows()
            .into_iter()
            .flat_map(|f| f.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(
            vec,
            vec![z(2.0, 0.0), z(0.0, 0.0), z(0.0, 0.0), z(0.0, 2.0)]
        );
        assert_eq!(n_isqrt2, 2);

        let (sparse, n_isqrt2) = to_sparse_and_n_isqrt2(cx(0, 1).unwrap(), 2, true);
        let dense = sparse.to_dense();
        let vec = dense
            .rows()
            .into_iter()
            .flat_map(|f| f.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(
            vec,
            vec![
                z(2.0, 0.0),
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(2.0, 0.0),
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(2.0, 0.0),
                z(0.0, 0.0),
                z(0.0, 0.0),
                z(2.0, 0.0),
                z(0.0, 0.0),
            ]
        );
        assert_eq!(n_isqrt2, 2);
    }

    #[rstest]
    #[case(1, vec![X], vec![h(0)])]
    #[case(1, vec![Z], vec![h(0)])]
    #[case(1, vec![I], vec![s(0)])]
    #[case(1, vec![Z], vec![s(0)])]
    #[case(1, vec![X], vec![s(0)])]
    #[case(1, vec![Y], vec![s(0)])]
    #[case(1, vec![Y], vec![s(0), h(0)])]
    #[case(1, vec![Y], vec![h(0), s(0), h(0)])]
    #[case(1, vec![X], vec![h(0), s(0), h(0)])]
    #[case(1, vec![Z], vec![h(0), s(0), h(0)])]
    #[case(2, vec![Z, X], vec![h(0), s(1), cx(0, 1).unwrap()])]
    #[case(2, vec![Z, Y], vec![h(0), s(1), cx(1, 0).unwrap()])]
    #[case(2, vec![Z, Z], vec![cx(0, 1).unwrap()])]
    #[case(3, vec![I, I, I], vec![cx(0, 2).unwrap(), h(0), cx(1, 0).unwrap()])]
    #[case(3, vec![Z, Y, X], vec![cx(1, 2).unwrap(), h(0), s(1), cx(1, 0).unwrap()])]
    fn test_to_sparse(
        #[case] n_qubit: usize,
        #[case] paulis: Vec<PauliMatrix>,
        #[case] gates: Vec<Gate>,
    ) {
        for big_endian in [true, false] {
            use crate::{
                container::coeffs::{complex_sign::ComplexSign, sign::Sign, traits::NumRepr},
                qubit::{pauli::cmpnt_major::term::Term, traits::PauliWordRef},
            };

            let mut cmpnt =
                PauliWord::from_vec(Qubits::from_count(n_qubit), paulis.clone()).unwrap();
            let mut term = Term::<ComplexSign>::from_vec_unchecked(
                Qubits::from_count(n_qubit),
                paulis.clone(),
            );
            let mut sparse_work = cmpnt.borrow().to_sparse_matrix(big_endian);
            let mut n_isqrt2_tot = 0_usize;
            let mut phase = Sign(false);
            // build up the conjugation as a product of parse matrices in sparse_work
            // Gates Pauli Gates^dagger
            // for Pauli Gates^dagger part first
            for gate in gates.iter().rev() {
                let (mut gate_sparse, n_isqrt2) =
                    to_sparse_and_n_isqrt2(*gate, n_qubit, big_endian);
                conj(&mut gate_sparse);
                sparse_work = &sparse_work * &gate_sparse;
                n_isqrt2_tot += n_isqrt2;
                // perform each conjugation on the Pauli string
                phase *= cmpnt.borrow_mut().conj_clifford(*gate);
                term.borrow_mut().conj_clifford(*gate);
            }
            for gate in gates.iter().rev() {
                let (gate_sparse, n_isqrt2) = to_sparse_and_n_isqrt2(*gate, n_qubit, big_endian);
                sparse_work = &gate_sparse * &sparse_work;
                n_isqrt2_tot += n_isqrt2;
            }
            // normalizer should be a binary power, no sqrt2 part
            assert!(n_isqrt2_tot & 1 == 0);
            sparse_work /= Complex64::new((1 << (n_isqrt2_tot / 2)) as f64, 0.0);

            assert_eq!(
                cmpnt.borrow().to_sparse_matrix(big_endian).to_dense() * phase.to_complex(),
                sparse_work.to_dense()
            );
            assert_eq!(
                term.borrow().to_sparse_matrix(big_endian).to_dense(),
                sparse_work.to_dense()
            );

            let mut term = Term::<ComplexSign>::from_vec_unchecked(
                Qubits::from_count(n_qubit),
                paulis.clone(),
            );
            let gates_rev = gates.iter().rev().copied().collect::<Vec<_>>();
            term.borrow_mut().conj_clifford_vec(gates_rev);
            assert_eq!(
                term.borrow().to_sparse_matrix(big_endian).to_dense(),
                sparse_work.to_dense()
            );
        }
    }
}
