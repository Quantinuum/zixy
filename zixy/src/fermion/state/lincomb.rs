//! Fermion state-specific linear combination utilities.

use crate::container::bit_matrix::AsBitMatrix;
use crate::container::coeffs::traits::FieldElem;
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::traits::proj::BorrowMut;
use crate::container::traits::RefElements;
use crate::container::word_iters;
use crate::container::word_iters::term_set::AsViewMut;
use crate::fermion::mode::Modes;
use crate::fermion::state::{term_set, terms};
use crate::utils::arith::invert_endian;
pub use crate::utils::vector_ops::{l2_norm, vdot};

/// Convert a fermion state linear combination to a dense vector representation.
/// If big_endian is true, the bit associated with mode 0 is the most significant in the index.
/// Else, the occupation flag mode n_mode - 1 is the most significant bit.
pub fn to_dense<C: FieldElem>(
    state: &impl terms::AsView<C>,
    big_endian: bool,
) -> Result<Vec<C>, OutOfBounds> {
    let state_ref = state.view();
    let n = state_ref.word_iters.n_bit();
    OutOfBounds::check(n, 64, Dimension::Mode)?;

    let mut out: Vec<C> = vec![C::ZERO; 1 << n];
    for term in state_ref.iter() {
        let ind = term.get_word_iter_ref().get_u64it().next().unwrap_or(0);
        let ind = if big_endian {
            invert_endian(ind, n)
        } else {
            ind
        };
        out[ind as usize] = term.get_coeff();
    }
    Ok(out)
}

/// Create a state linear combination from a dense array slice of coefficients.
pub fn assign_from_dense<C: FieldElem>(
    out: &mut term_set::ViewMut<C>,
    source: &[C],
    big_endian: bool,
) {
    let n = out.word_iters.n_bit();
    let n_take = source.len().min(1 << n);
    out.clear();
    for (i, c) in source.iter().take(n_take).enumerate() {
        if *c == C::ZERO {
            continue;
        }
        word_iters::lincomb::scaled_iadd_u64it(
            out,
            std::iter::once(if big_endian {
                invert_endian(i as u64, n)
            } else {
                i as u64
            }),
            *c,
        );
    }
}

/// Create a state linear combination from a dense array slice of coefficients.
pub fn from_dense<C: FieldElem>(
    modes: Modes,
    source: &[C],
    big_endian: bool,
) -> term_set::TermSet<C> {
    let mut out: term_set::TermSet<C> = term_set::TermSet::<C>::new(modes);
    assign_from_dense(&mut out.borrow_mut(), source, big_endian);
    out
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::container::bit_matrix::AsRowRef;
    use crate::container::word_iters::term_set::AsView;
    use crate::fermion::mode::Modes;
    use crate::fermion::state::terms::AsViewMut;
    use crate::fermion::state::terms::Terms;
    use num_complex::Complex64;
    use std::collections::HashSet;

    #[test]
    fn test_assign_from_dense_endian() {
        let modes = Modes::from_count(3);

        // Dense index 1 (001 in little-endian significance)
        let source = vec![0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // little-endian
        let mut le: term_set::TermSet<f64> = term_set::TermSet::new(modes.clone());
        assign_from_dense(&mut le.borrow_mut(), &source, false);
        assert_eq!(le.view().coeffs.len(), 1);
        assert_eq!(le.view().coeffs[0], 7.0);
        assert_eq!(
            le.view().get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, false], // |100> in this mode ordering
        );

        // big-endian: index 1 -> bit-reverse(001) = 100
        let mut be: term_set::TermSet<f64> = term_set::TermSet::new(modes);
        assign_from_dense(&mut be.borrow_mut(), &source, true);
        assert_eq!(be.view().coeffs.len(), 1);
        assert_eq!(be.view().coeffs[0], 7.0);
        assert_eq!(
            be.view().get_elem_ref(0).get_word_iter_ref().to_set(),
            HashSet::from([2]), // |001>
        );
    }

    #[test]
    fn test_from_dense() {
        let modes = Modes::from_count(2);
        let source = vec![0.0, 1.0, 2.0, 3.0];
        let state = from_dense(modes, &source, false);
        assert_eq!(state.view().coeffs.len(), 3);
        assert_eq!(state.view().coeffs[0], 1.0);
        assert_eq!(state.view().coeffs[1], 2.0);
        assert_eq!(state.view().coeffs[2], 3.0);
    }

    #[test]
    fn test_to_dense_out_of_bounds() {
        let modes = Modes::from_count(65);
        let terms = Terms::<Complex64>::new(modes);
        let result = to_dense(&terms, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_dense() {
        let modes = Modes::from_count(3);
        let mut terms = Terms::<f64>::new(modes);
        terms
            .push_set_with_coeff(HashSet::from([0, 2]), 5.0)
            .unwrap();
        let dense = to_dense(&terms, false).unwrap();
        assert_eq!(dense.len(), 8);
        for (i, &c) in dense.iter().enumerate() {
            let expected = if i == 5 { 5.0 } else { 0.0 };
            assert_eq!(c, expected);
        }
    }
}
