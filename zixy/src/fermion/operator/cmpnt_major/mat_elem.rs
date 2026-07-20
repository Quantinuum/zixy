//! Implements the evaluation of matrix elements of fermion operators with respect to Slater-determinant states.

use ndarray::Array2 as Matrix;
use num_complex::Complex64;

use crate::container::coeffs::traits::FieldElem;
use crate::container::coeffs::traits::NumRepr;
use crate::container::traits::Elements;
use crate::container::traits::RefElements;
use crate::container::word_iters;
use crate::container::word_iters::term_set::AsView as TermSetAsView;
use crate::container::word_iters::terms::AsView;
use crate::container::word_iters::terms::AsViewMut;
use crate::container::word_iters::WordIters;
use crate::fermion::operator::cmpnt_major as fermion_op;
use crate::fermion::operator::cmpnt_major::lincomb;
use crate::fermion::operator::products::apply_op_ket_u64;
use crate::fermion::operator::products::ApplyResult;
use crate::fermion::state;
use crate::fermion::state::cmpnt::BasisState;
use crate::fermion::state::cmpnt_list::CmpntList as StateList;
use crate::fermion::state::cmpnt_list::CmpntRef as StateRef;
use crate::fermion::traits::ModesBased;

/// Error returned when a state subspace basis contains repeated Slater determinants, or in general when any
/// subspace has a non-unit metric.
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

/// Evaluate the matrix element of a normal-ordered fermion operator between the Slater determinants `bra` and
/// `ket`. Assumes the mode space fits within a single `u64` word.
pub fn mat_elem_cmpnts<C: FieldElem>(
    op: &impl fermion_op::terms::AsView<C>,
    bra: &StateRef,
    ket: &StateRef,
) -> C {
    let op = op.view();
    assert!(op.to_modes().len() <= 64);
    let bra_word = bra.get_u64it().next().expect("bra state has zero modes");
    let ket_word = ket.get_u64it().next().expect("ket state has zero modes");
    let mut contrib_sum = C::ZERO;
    for (op_cmpnt, op_coeff) in op.word_iters.iter().zip(op.coeffs.iter()) {
        let cre = op_cmpnt
            .get_cre_part()
            .get_u64it()
            .next()
            .expect("operator has zero creation operators");
        let ann = op_cmpnt
            .get_ann_part()
            .get_u64it()
            .next()
            .expect("operator has zero annihilation operators");
        match apply_op_ket_u64(cre, ann, ket_word) {
            ApplyResult::Applied(bits, sign) => {
                if bits != bra_word {
                    continue;
                }
                if let Ok(phase) = C::try_represent(sign) {
                    contrib_sum += *op_coeff * phase;
                }
            }
            ApplyResult::Zero => continue,
        }
    }
    contrib_sum
}

/// Evaluate the matrix element of `op` between the state linear combinations `bra` and `ket`.
pub fn mat_elem<C: FieldElem>(
    op: &impl fermion_op::terms::AsView<C>,
    bra: &impl state::terms::AsView<C>,
    ket: &impl state::terms::AsView<C>,
) -> C {
    let bra = bra.view();
    let ket = ket.view();
    assert_eq!(op.view().to_modes(), bra.to_modes());
    assert_eq!(op.view().to_modes(), ket.to_modes());
    let mut out = C::ZERO;
    for (bra_cmpnt, bra_coeff) in bra.word_iters.iter().zip(bra.coeffs.iter()) {
        for (ket_cmpnt, ket_coeff) in ket.word_iters.iter().zip(ket.coeffs.iter()) {
            let contrib_sum = mat_elem_cmpnts(op, &bra_cmpnt, &ket_cmpnt);
            out += bra_coeff.complex_conj() * *ket_coeff * contrib_sum;
        }
    }
    out
}

/// Evaluate the matrix elements and overlaps between all elements of `subspace` and return the matrices
/// of `op` matrix elements and overlaps with a basis ordering consistent with `subspace`.
/// Returns `(H, S)` where `H` is the matrix of operator matrix elements and `S` is the overlap matrix.
pub fn mat_nonortho_projected<C: FieldElem>(
    op: &fermion_op::terms::View<C>,
    subspace: Vec<&state::term_set::View<C>>,
) -> (Matrix<C>, Matrix<C>) {
    let dim = subspace.len();
    let shape = (dim, dim);
    let mut h_mat = Matrix::<C>::zeros(shape);
    let mut s_mat = Matrix::<C>::zeros(shape);
    let hermitian = lincomb::is_hermitian(op, 0.0);
    for (i_bra, bra) in subspace.iter().enumerate() {
        for (i_ket, ket) in subspace.iter().enumerate() {
            if hermitian && i_ket > i_bra {
                continue;
            }
            let h_elem = mat_elem(op, &bra.as_terms(), &ket.as_terms());
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
    op: &fermion_op::terms::View<C>,
    subspace: StateList,
) -> Matrix<C> {
    let shape = (subspace.len(), subspace.len());
    let mut h_mat = Matrix::<C>::zeros(shape);
    let hermitian = lincomb::is_hermitian(op, 0.0);
    for (i_bra, bra) in subspace.iter().enumerate() {
        for (i_ket, ket) in subspace.iter().enumerate() {
            if hermitian && i_ket > i_bra {
                continue;
            }
            let h_elem = mat_elem_cmpnts(op, &bra, &ket);
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
    op: &fermion_op::terms::View<C>,
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
///
/// In debug builds, sanity-checks that `state` is actually (close to) an eigenfunction of `op` by
/// separately computing the full Rayleigh quotient <state|op|state> / <state|state> and comparing
/// it to the mixed-estimator result; panics if they disagree beyond default tolerance. This check
/// is O(n^2) in the number of terms in `state` and is skipped in release builds.
pub fn expval_eigenfunction<C: FieldElem>(
    op: &impl fermion_op::terms::AsView<C>,
    state: &impl state::terms::AsView<C>,
) -> C {
    let state = state.view();
    let result = match state.dominant_term() {
        Some((elem_ref, c)) => {
            let mut bra = state::terms::Terms::new(op.view().to_modes());
            bra.push_elem_coeff(elem_ref, C::ONE / c);
            let result = mat_elem(op, &bra, &state);

            #[cfg(debug_assertions)]
            {
                use crate::utils::vector_ops::l2_norm_square;
                let norm_sq = l2_norm_square(&state);
                if norm_sq > C::ATOL_DEFAULT {
                    let full = mat_elem(op, &state, &state) / C::from_real(norm_sq);
                    debug_assert!(
                        result.is_close_default(full),
                        "expval_eigenfunction: mixed estimator ({result}) diverges from the full \
                         Rayleigh quotient ({full}) beyond default tolerance \
                         (rtol={}, atol={}) — `state` may not be an eigenfunction of `op`",
                        C::RTOL_DEFAULT,
                        C::ATOL_DEFAULT,
                    );
                }
            }

            result
        }
        None => C::ZERO,
    };
    result
}

/// Apply a fermion operator to a state linear combination, adding the result to `out`.
pub fn apply<C: FieldElem>(
    op: &fermion_op::terms::View<C>,
    state: &state::terms::View<C>,
    out: &mut state::term_set::ViewMut<Complex64>,
) {
    let mut tmp = BasisState::new(state.word_iters.to_modes());
    for (op_cmpnt, op_coeff) in op.word_iters.iter().zip(op.coeffs.iter()) {
        for (state_cmpnt, state_coeff) in state.word_iters.iter().zip(state.coeffs.iter()) {
            let Some(sign) = tmp
                .borrow_mut()
                .assign_mul_by_op(op_cmpnt.clone(), state_cmpnt)
            else {
                continue;
            };
            let mut c = sign.to_complex();
            c *= op_coeff.to_complex();
            c *= state_coeff.to_complex();
            word_iters::lincomb::scaled_iadd_elem(out, tmp.borrow(), c);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::bit_matrix::AsRowMutRef;
    use crate::container::coeffs::traits::HasCoeffs;
    use crate::container::traits::proj::Borrow;
    use crate::container::traits::proj::BorrowMut;
    use crate::container::traits::MutRefElements;
    use crate::container::word_iters::lincomb::scaled_iadd_elem;
    use crate::fermion::mode::Modes;
    use crate::fermion::operator::cmpnt::Cmpnt;
    use crate::fermion::operator::cmpnt_major::term_set::TermSet as OpTermSet;
    use crate::fermion::state::term_set::TermSet as StateTermSet;
    use crate::fermion::state::terms::AsViewMut;
    use std::collections::HashSet;

    /// Build a hopping-term operator `c_1^+ c_0 + c_0^+ c_1` on the given mode space.
    fn build_hopping_op(modes: Modes) -> OpTermSet<f64> {
        let mut op = OpTermSet::<f64>::new(modes.clone());
        let hop_01 =
            Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([1]), HashSet::from([0]));
        let hop_10 = Cmpnt::from_sets_unchecked(modes, HashSet::from([0]), HashSet::from([1]));
        scaled_iadd_elem(&mut op.borrow_mut(), hop_01.borrow(), 1.0);
        scaled_iadd_elem(&mut op.borrow_mut(), hop_10.borrow(), 1.0);
        op
    }

    fn build_basis_state<C: FieldElem>(modes: Modes, occupied: HashSet<usize>) -> StateTermSet<C> {
        let mut terms = state::terms::Terms::<C>::new(modes);
        terms.push_set_with_coeff(occupied, C::ONE).unwrap();
        terms.into()
    }

    #[test]
    fn test_mat_elem_hopping() {
        let modes = Modes::from_count(2);
        let op = build_hopping_op(modes.clone());
        let ket = build_basis_state(modes.clone(), HashSet::from([0]));
        let bra_hopped = build_basis_state(modes.clone(), HashSet::from([1]));
        let bra_diag = build_basis_state(modes, HashSet::from([0]));

        let e_hop = mat_elem(
            &op.borrow().as_terms(),
            &bra_hopped.borrow().as_terms(),
            &ket.borrow().as_terms(),
        );
        assert_eq!(e_hop, 1.0);

        let e_diag = mat_elem(
            &op.borrow().as_terms(),
            &bra_diag.borrow().as_terms(),
            &ket.borrow().as_terms(),
        );
        assert_eq!(e_diag, 0.0);
    }

    #[test]
    fn test_apply_hopping() {
        let modes = Modes::from_count(2);
        let op = build_hopping_op(modes.clone());
        let ket = build_basis_state(modes.clone(), HashSet::from([0]));

        let mut out = StateTermSet::<Complex64>::new(modes);
        apply(
            &op.borrow().as_terms(),
            &ket.borrow().as_terms(),
            &mut out.borrow_mut(),
        );

        assert_eq!(out.view().get_coeffs().len(), 1);
        let expected = build_basis_state(out.view().to_modes(), HashSet::from([1]));
        let overlap = state::lincomb::vdot(&out, &expected.borrow().as_terms());
        assert_eq!(overlap, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_expval_eigenfunction() {
        // The number operator n_0 = c_0^+ c_0 has |1_0> as an eigenfunction with eigenvalue 1.
        let modes = Modes::from_count(1);
        let mut op = OpTermSet::<f64>::new(modes.clone());
        let n0 = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([0]), HashSet::from([0]));
        scaled_iadd_elem(&mut op.borrow_mut(), n0.borrow(), 1.0);

        let state = build_basis_state(modes, HashSet::from([0]));
        let e = expval_eigenfunction(&op.borrow().as_terms(), &state.borrow().as_terms());
        assert_eq!(e, 1.0);
    }

    #[test]
    fn test_mat_ortho_projected() {
        let modes = Modes::from_count(2);
        let op = build_hopping_op(modes.clone());

        let mut subspace = StateList::new(modes);
        subspace.push_clear();
        subspace
            .get_elem_mut_ref(0)
            .assign_set_unchecked(HashSet::from([0]));
        subspace.push_clear();
        subspace
            .get_elem_mut_ref(1)
            .assign_set_unchecked(HashSet::from([1]));

        let h_mat = mat_ortho_projected(&op.borrow().as_terms(), subspace).unwrap();
        assert_eq!(h_mat[[0, 0]], 0.0);
        assert_eq!(h_mat[[0, 1]], 1.0);
        assert_eq!(h_mat[[1, 0]], 1.0);
        assert_eq!(h_mat[[1, 1]], 0.0);
    }

    #[test]
    fn test_mat_nonortho_projected() {
        let modes = Modes::from_count(2);
        let op = build_hopping_op(modes.clone());

        let state_a = build_basis_state::<f64>(modes.clone(), HashSet::from([0]));
        let state_b = build_basis_state::<f64>(modes, HashSet::from([1]));
        let view_a = state_a.view();
        let view_b = state_b.view();

        let (h_mat, s_mat) =
            mat_nonortho_projected(&op.borrow().as_terms(), vec![&view_a, &view_b]);

        assert_eq!(h_mat[[0, 0]], 0.0);
        assert_eq!(h_mat[[0, 1]], 1.0);
        assert_eq!(h_mat[[1, 0]], 1.0);
        assert_eq!(h_mat[[1, 1]], 0.0);

        assert_eq!(s_mat[[0, 0]], 1.0);
        assert_eq!(s_mat[[0, 1]], 0.0);
        assert_eq!(s_mat[[1, 0]], 0.0);
        assert_eq!(s_mat[[1, 1]], 1.0);
    }

    #[test]
    #[should_panic(expected = "may not be an eigenfunction")]
    fn test_expval_eigenfunction_guard_panics_on_misuse() {
        // The hopping operator c_1^+ c_0 + c_0^+ c_1 has matrix [[0,1],[1,0]] in the {|1_0>,|1_1>}
        // basis, with eigenvectors (1,1) and (1,-1). The unequal combination 2|1_0> + 1|1_1> is NOT
        // one of those eigenvectors, so the debug-mode sanity check should panic here.

        let modes = Modes::from_count(2);
        let op = build_hopping_op(modes.clone());
        let mut terms = state::terms::Terms::<f64>::new(modes);
        terms.push_set_with_coeff(HashSet::from([0]), 2.0).unwrap();
        terms.push_set_with_coeff(HashSet::from([1]), 1.0).unwrap();
        let state: StateTermSet<f64> = terms.into();
        expval_eigenfunction(&op.borrow().as_terms(), &state.borrow().as_terms());
    }
}
