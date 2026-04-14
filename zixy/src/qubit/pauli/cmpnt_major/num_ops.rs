//! Factory functions for qubit number operators.

use std::collections::{HashMap, HashSet};

use crate::container::coeffs::traits::FieldElem;
use crate::container::errors::OutOfBounds;
use crate::container::traits::proj::BorrowMut;
use crate::qubit::mode::{PauliMatrix, Qubits};
use crate::qubit::pauli::cmpnt_major::lincomb::add_from_pauli_map;
use crate::qubit::pauli::cmpnt_major::term_set::TermSet;

/// Create a number operator over the given set of mode indices.
pub fn num_op_inds<C: FieldElem>(
    qubits: Qubits,
    inds: HashSet<usize>,
) -> Result<TermSet<C>, OutOfBounds> {
    let n = inds.len();
    let mut out = TermSet::<C>::new(qubits);
    // Z_i = |0><0| - |1><1| i.e. N_i = (I_i - Z_i)/2 = |1><1|
    // N = sum_i^n N_i = n/2 I - sum_i^n 1/2 Z_i
    type Map = HashMap<usize, PauliMatrix>;
    let coeff_n = n as f64;
    add_from_pauli_map(
        &mut out.borrow_mut(),
        Map::new(),
        C::HALF * C::from_real(coeff_n),
    )?;
    for i in inds.into_iter() {
        add_from_pauli_map(
            &mut out.borrow_mut(),
            Map::from([(i, PauliMatrix::Z)]),
            -C::HALF,
        )?;
    }
    Ok(out)
}

/// Create the full number operator over all modes.
pub fn num_op<C: FieldElem>(qubits: Qubits) -> TermSet<C> {
    num_op_inds::<C>(qubits.clone(), (0..qubits.n_qubit()).collect()).unwrap()
}

/// Create the number operator over the modes with even positions i.e. modes 0, 2, 4, ...
pub fn num_op_even_pos<C: FieldElem>(qubits: Qubits) -> TermSet<C> {
    num_op_inds::<C>(qubits.clone(), (0..qubits.n_qubit()).step_by(2).collect()).unwrap()
}

/// Create the number operator over the modes with odd positions i.e. modes 1, 3, 5, ...
pub fn num_op_odd_pos<C: FieldElem>(qubits: Qubits) -> TermSet<C> {
    num_op_inds::<C>(qubits.clone(), (1..qubits.n_qubit()).step_by(2).collect()).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::container::traits::proj::Borrow;
    use crate::container::word_iters::term_set::AsView;
    use crate::qubit::mode::{PauliMatrix, Qubits};
    use crate::qubit::pauli::cmpnt_major::lincomb::from_vec_coeff_pairs;
    use crate::qubit::pauli::cmpnt_major::num_ops::num_op;
    use num_complex::Complex64;

    #[test]
    fn test_number_op() {
        use crate::qubit::mode::PauliMatrix::*;

        let qubits = Qubits::from_count(6);
        let pairs: Vec<(Vec<PauliMatrix>, f64)> = vec![
            (vec![I, I, I, I, I, I], 3.0),
            (vec![Z, I, I, I, I, I], -0.5),
            (vec![I, Z, I, I, I, I], -0.5),
            (vec![I, I, Z, I, I, I], -0.5),
            (vec![I, I, I, Z, I, I], -0.5),
            (vec![I, I, I, I, Z, I], -0.5),
            (vec![I, I, I, I, I, Z], -0.5),
        ];
        assert!(num_op::<f64>(qubits.clone()).borrow().equal(
            &from_vec_coeff_pairs(qubits.clone(), pairs)
                .unwrap()
                .borrow()
        ));

        let qubits = Qubits::from_count(7);
        let pairs: Vec<(Vec<PauliMatrix>, f64)> = vec![
            (vec![I, I, I, I, I, I, I], 3.5),
            (vec![Z, I, I, I, I, I, I], -0.5),
            (vec![I, Z, I, I, I, I, I], -0.5),
            (vec![I, I, Z, I, I, I, I], -0.5),
            (vec![I, I, I, Z, I, I, I], -0.5),
            (vec![I, I, I, I, Z, I, I], -0.5),
            (vec![I, I, I, I, I, Z, I], -0.5),
            (vec![I, I, I, I, I, I, Z], -0.5),
        ];
        assert!(num_op::<f64>(qubits.clone()).borrow().equal(
            &from_vec_coeff_pairs(qubits.clone(), pairs)
                .unwrap()
                .borrow()
        ));

        let qubits = Qubits::from_count(3);
        let pairs: Vec<(Vec<PauliMatrix>, Complex64)> = vec![
            (vec![I, I, I], Complex64::new(1.5, 0.0)),
            (vec![Z, I, I], Complex64::new(-0.5, 0.0)),
            (vec![I, Z, I], Complex64::new(-0.5, 0.0)),
            (vec![I, I, Z], Complex64::new(-0.5, 0.0)),
        ];
        assert!(num_op::<Complex64>(qubits.clone()).borrow().equal(
            &from_vec_coeff_pairs(qubits.clone(), pairs)
                .unwrap()
                .borrow()
        ));

        let qubits = Qubits::from_count(4);
        let pairs: Vec<(Vec<PauliMatrix>, Complex64)> = vec![
            (vec![I, I, I, I], Complex64::new(2.0, 0.0)),
            (vec![Z, I, I, I], Complex64::new(-0.5, 0.0)),
            (vec![I, Z, I, I], Complex64::new(-0.5, 0.0)),
            (vec![I, I, Z, I], Complex64::new(-0.5, 0.0)),
            (vec![I, I, I, Z], Complex64::new(-0.5, 0.0)),
        ];
        assert!(num_op::<Complex64>(qubits.clone()).borrow().equal(
            &from_vec_coeff_pairs(qubits.clone(), pairs)
                .unwrap()
                .borrow()
        ));
    }
}
