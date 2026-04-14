//! Implements the evalutation of matrix elements of Pauli operators with respect to states.

use ndarray::Array2 as Matrix;
use num_complex::Complex64;
use std::collections::HashMap;

use crate::container::coeffs::traits::FieldElem;
use crate::container::coeffs::traits::NumRepr;
use crate::container::quicksort::LexicographicSort;
use crate::container::quicksort::QuickSort;
use crate::container::traits::proj;
use crate::container::traits::proj::Borrow;
use crate::container::traits::proj::BorrowMut;
use crate::container::traits::Elements;
use crate::container::traits::HasIndex;
use crate::container::traits::RefElements;
use crate::container::word_iters;
use crate::container::word_iters::term_set::AsView;
use crate::container::word_iters::terms::AsView as _;
use crate::container::word_iters::terms::AsViewMut;
use crate::container::word_iters::WordIters;
use crate::qubit::pauli::cmpnt_major as pauli;
use crate::qubit::pauli::cmpnt_major::lincomb;
use crate::qubit::pauli::cmpnt_major::products::mul_op_state_bits_u64;
use crate::qubit::pauli::cmpnt_major::products::mul_op_state_phase_u64;
use crate::qubit::state;
use crate::qubit::state::cmpnt_list::CmpntList as StateList;
use crate::qubit::state::cmpnt_list::CmpntRef as StateRef;
use crate::qubit::traits::QubitsBased;

/// Error returned when a state subspace basis contains repeated computational basis vectors or
/// in general when any subspace has a non-unit metric.
#[derive(Debug, PartialEq)]
pub struct SubspaceNonorthogonal {}

impl std::fmt::Display for SubspaceNonorthogonal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "The subspace is non-orthogonal (it contains repeated basis vectors)"
        )
    }
}
impl std::error::Error for SubspaceNonorthogonal {}

/// Lexicographically sort the terms of the operator so that the strings are in order of the value of the X part.
/// This assists matrix element evaluation, since it naively would entail a triple loop over the terms of op, bra,
/// and ket costing
///     O(N_op * N_bra * N_ket).
/// However, with a sort of the components of op, this can be decomposed into a loop over op, followed by a double
/// loop over bra and ket in which the contributing operator terms are looked up, which with binary search would cost
///     O(N_op log N_op) + O(N_bra * N_ket * log N_op)
/// Introducing a HashMap to the start of the X blocks eliminates the log scaling in this lookup at the expense of
/// O(sqrt(N_op)) additional space. This gives a final cost of:
///     O(N_op log N_op) + O(N_bra * N_ket)
///
/// Return the sorted list of terms along with a map that points to the start of each block of terms with the same X string value.
pub fn sort_and_map_pauli_op<C: FieldElem>(
    op: &mut impl pauli::terms::AsViewMut<C>,
) -> HashMap<u64, usize> {
    let op = op.view_mut();
    // extract the components and coeffs, then sort them by component value

    QuickSort::<pauli::cmpnt_list::CmpntList, C>::sort_with_coeffs(
        &LexicographicSort { ascending: true },
        op.word_iters,
        op.coeffs,
    );
    // LexicographicSort { ascending: true }.sort_with_coeffs(&mut out_cmpnts, &mut out_coeffs);
    // now the components are sorted in order of increasing X part

    // construct a map from an X string to the position in cmpnts at which it first occurs in out_cmpnts
    let mut start_inds_map = HashMap::<u64, usize>::default();
    {
        // keep track of the X string of the previous element.
        let mut prev_x: Option<u64> = None;
        for op_ref in op.word_iters.iter() {
            let x = op_ref.get_part_iters().0.next().unwrap();
            if prev_x.is_none_or(|prev_x| prev_x != x) {
                // x value should be increasing
                assert!(prev_x.is_none_or(|prev_x| x > prev_x));
                // insert a starting index for this new block of terms with the same X string
                start_inds_map.insert(x, op_ref.get_index());
            }
            prev_x = Some(x);
        }
    }
    start_inds_map
}

/// Get the matrix element of the operator with the given terms and X block map between the computational basis vectors `bra` and `ket`.
pub fn mat_elem_cmpnts_from_presorted<C: FieldElem>(
    presorted_op: &impl pauli::terms::AsView<C>,
    x_block_map: &HashMap<u64, usize>,
    bra: &StateRef,
    ket: &StateRef,
) -> C {
    let presorted_op = presorted_op.view();
    // for now, components should fit in a single u64 word.
    assert!(presorted_op.qubits().len() <= 64);

    let word_bra = bra.get_u64it().next().unwrap();
    let word_ket = ket.get_u64it().next().unwrap();
    // only one x term connects the bra to the ket:
    let op_x = word_bra ^ word_ket;
    let mut i = match x_block_map.get(&op_x) {
        Some(&x) => x,
        None => return C::ZERO,
    };
    assert_eq!(mul_op_state_bits_u64(op_x, word_ket), word_bra);
    let mut contrib_sum = C::ZERO;
    // sum over all the Z parts with the same X part
    while i < presorted_op.len() && presorted_op.word_iters.x_part()[i][0] == op_x {
        let op_z = presorted_op.word_iters.z_part()[i][0];
        let phase = mul_op_state_phase_u64(op_x, op_z, word_ket);
        if let Ok(phase) = C::try_represent(phase) {
            contrib_sum += presorted_op.coeffs[i] * phase;
        }
        i += 1;
    }
    contrib_sum
}

/// Get the matrix element of the operator with the given terms and X block map between the state linear combinations `bra` and `ket`.
pub fn mat_elem_from_presorted<C: FieldElem>(
    presorted_op: &impl pauli::terms::AsView<C>,
    x_block_map: &HashMap<u64, usize>,
    bra: &impl state::terms::AsView<C>,
    ket: &impl state::terms::AsView<C>,
) -> C {
    let presorted_op = presorted_op.view();
    let bra = bra.view();
    let ket = ket.view();
    let mut out = C::ZERO;
    for (bra_cmpnt, bra_coeff) in bra.word_iters.iter().zip(bra.coeffs.iter()) {
        for (ket_cmpnt, ket_coeff) in ket.word_iters.iter().zip(ket.coeffs.iter()) {
            let contrib_sum =
                mat_elem_cmpnts_from_presorted(&presorted_op, x_block_map, &bra_cmpnt, &ket_cmpnt);
            out += bra_coeff.complex_conj() * *ket_coeff * contrib_sum;
        }
    }
    out
}

/// Evaluate the matrix element of op between bra and ket by first sorting and X block mapping `op`.
pub fn mat_elem<C: FieldElem>(
    op: &impl pauli::terms::AsView<C>,
    bra: &impl state::terms::AsView<C>,
    ket: &impl state::terms::AsView<C>,
) -> C {
    let mut presorted_op = proj::ToOwned::to_owned(&op.view());
    let x_block_map = sort_and_map_pauli_op(&mut presorted_op);
    mat_elem_from_presorted(&presorted_op, &x_block_map, bra, ket)
}

/// Evaluate the matrix elements and overlaps between all elements of `subspace` and return the matrices
/// of op matrix elements and overlaps with a basis ordering consistent with `subspace`.
pub fn mat_nonortho_projected<C: FieldElem>(
    op: &pauli::terms::View<C>,
    subspace: Vec<&state::term_set::View<C>>,
) -> (Matrix<C>, Matrix<C>) {
    let dim = subspace.len();
    let shape = (dim, dim);
    let mut h_mat = Matrix::<C>::zeros(shape);
    let mut s_mat = Matrix::<C>::zeros(shape);
    let mut presorted_op = proj::ToOwned::to_owned(op);
    let x_block_map = sort_and_map_pauli_op(&mut presorted_op.borrow_mut());
    let hermitian = lincomb::is_hermitian(op, 0.0);
    for (i_bra, bra) in subspace.iter().enumerate() {
        for (i_ket, ket) in subspace.iter().enumerate() {
            if hermitian && i_ket > i_bra {
                continue;
            }
            let h_elem = mat_elem_from_presorted(
                &presorted_op.borrow(),
                &x_block_map,
                &bra.as_terms(),
                &ket.as_terms(),
            );
            h_mat[[i_bra, i_ket]] = h_elem;
            let s_elem = state::lincomb::vdot(*bra, &ket.as_terms());
            s_mat[[i_bra, i_ket]] = s_elem;
            if i_ket != i_bra {
                s_mat[[i_ket, i_bra]] = s_elem.conj();
                if hermitian {
                    h_mat[[i_ket, i_bra]] = h_elem.conj();
                }
            }
        }
    }
    (h_mat, s_mat)
}

/// Evaluate the matrix elements for the orthonormal basis defined by the components of `subspace`.
/// Do not check that the subspace is indeed orthonormal.
pub fn mat_ortho_projected_unchecked<C: FieldElem>(
    op: &pauli::terms::View<C>,
    subspace: StateList,
) -> Matrix<C> {
    let shape = (subspace.len(), subspace.len());
    let mut h_mat = Matrix::<C>::zeros(shape);
    let mut presorted_op = proj::ToOwned::to_owned(op);
    let x_block_map = sort_and_map_pauli_op(&mut presorted_op.borrow_mut());
    let hermitian = lincomb::is_hermitian(op, 0.0);
    for (i_bra, bra) in subspace.iter().enumerate() {
        for (i_ket, ket) in subspace.iter().enumerate() {
            if hermitian && i_ket > i_bra {
                continue;
            }
            let h_elem =
                mat_elem_cmpnts_from_presorted(&presorted_op.borrow(), &x_block_map, &bra, &ket);
            h_mat[[i_bra, i_ket]] = h_elem;
            if i_ket != i_bra && hermitian {
                h_mat[[i_ket, i_bra]] = h_elem.conj();
            }
        }
    }
    h_mat
}

/// Evaluate the matrix elements for the orthonormal basis defined by the components of `subspace`.
/// Check that the subspace is indeed orthonormal (i.e. no repeated basis states).
pub fn mat_ortho_projected<C: FieldElem>(
    op: &pauli::terms::View<C>,
    subspace: StateList,
) -> Result<Matrix<C>, SubspaceNonorthogonal> {
    if subspace.find_duplicates().count() == 0 {
        Ok(mat_ortho_projected_unchecked(op, subspace))
    } else {
        Err(SubspaceNonorthogonal {})
    }
}

/// If state is an eigenfunction of op, then op |state> = E * |state>, thus for any state trial
/// with a non-zero overlap <trial|state>, the energy can be extracted by the "mixed estimator"
/// E = <trial|op|state> / <trial|state>
pub fn expval_eigenfunction<C: FieldElem>(
    op: &impl pauli::terms::AsView<C>,
    state: &impl state::terms::AsView<C>,
) -> C {
    let state = state.view();
    // use the dominant term as the trial state
    match state.dominant_term() {
        Some((elem_ref, c)) => {
            let mut bra = state::terms::Terms::new(op.view().to_qubits());
            bra.push_elem_coeff(elem_ref, C::ONE / c);
            mat_elem(op, &bra, &state)
        }
        None => C::ZERO,
    }
}

pub fn apply<C: FieldElem>(
    op: &pauli::terms::View<C>,
    state: &state::terms::View<C>,
    out: &mut state::term_set::ViewMut<Complex64>,
) {
    use crate::qubit::state::cmpnt::BasisState;
    let mut tmp = BasisState::new(state.word_iters.to_qubits());
    for (op_cmpnt, op_coeff) in op.word_iters.iter().zip(op.coeffs.iter()) {
        for (state_cmpnt, state_coeff) in state.word_iters.iter().zip(state.coeffs.iter()) {
            let phase = tmp
                .borrow_mut()
                .assign_mul_by_op(op_cmpnt.clone(), state_cmpnt);
            let mut c = phase.to_complex();
            c *= op_coeff.to_complex();
            c *= state_coeff.to_complex();
            word_iters::lincomb::scaled_iadd_elem(out, tmp.borrow(), c);
        }
    }
}

#[cfg(test)]
mod tests {
    use bincode::config;

    use crate::utils::io::{file_path_in_crate, BinFileReader};

    use super::*;

    #[test]
    fn test_expval_mof_energy() {
        const E_EXACT: f64 = -2404.544973130422;
        let operator_path = file_path_in_crate("src/qubit/test_files/mof_cas/operator.bin");
        let reader = BinFileReader::new(operator_path).unwrap();
        let op: pauli::term_set::TermSet<f64> =
            bincode::serde::decode_from_reader(reader, config::standard()).unwrap();
        let state_path = file_path_in_crate("src/qubit/test_files/mof_cas/state.bin");
        let reader = BinFileReader::new(state_path).unwrap();
        let state: state::term_set::TermSet<f64> =
            bincode::serde::decode_from_reader(reader, config::standard()).unwrap();
        assert!(state::lincomb::l2_norm(&state.borrow().as_terms()).is_close_default(1.0));
        let e = mat_elem(
            &op.borrow().as_terms(),
            &state.borrow().as_terms(),
            &state.borrow().as_terms(),
        );
        assert!(e.is_close_default(E_EXACT));
        // the state is actually an eigenfunction
        let e = expval_eigenfunction(&op.borrow().as_terms(), &state.borrow().as_terms());
        assert!(e.is_close_default(E_EXACT));
    }
}
