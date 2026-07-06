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

/// Sum of squares of the coefficients of the given fermion state.
pub fn l2_norm_square<C: FieldElem>(state: &impl terms::AsView<C>) -> f64 {
    state.view().coeffs.iter().map(|c| c.magnitude_sq()).sum()
}

pub fn l2_norm<C: FieldElem>(state: &impl terms::AsView<C>) -> f64 {
    l2_norm_square(state).sqrt()
}

/// Take the inner product of a basis state linear combination with another.
pub fn vdot<C: FieldElem>(lhs: &impl term_set::AsView<C>, rhs: &impl terms::AsView<C>) -> C {
    rhs.view()
        .iter()
        .map(
            |rhs| match lhs.lookup_coeff_elem_ref(rhs.get_word_iter_ref()) {
                Some(lhs) => lhs.complex_conj() * rhs.get_coeff(),
                None => C::ZERO,
            },
        )
        .sum()
}

/// Reverse the bit order of the lowest n bits of index.
/// Pure relabeling: no sign consequence for fermion states.
pub fn invert_endian(index: u64, n: usize) -> u64 {
    let mut out = 0u64;
    for i in 0..n {
        out |= ((index >> i) & 1) << (n - 1 - i);
    }
    out
}

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
    use crate::fermion::mode::Modes;
    use crate::fermion::state::terms::Terms;
    use num_complex::Complex64;
    use rstest::rstest;

    #[rstest]
    #[case(vec![], 0.0)]
    #[case(vec![2.0], 4.0)]
    #[case(vec![-3.0], 9.0)]
    #[case(vec![1.0, 2.0, 3.0], 14.0)]
    #[case(vec![0.0, 0.0, 0.0], 0.0)]
    #[case(vec![0.0, 2.0, 0.0], 4.0)]
    fn test_l2_norm_square(#[case] coeffs: Vec<f64>, #[case] expected: f64) {
        let modes = Modes::from_count(coeffs.len());
        let mut state: terms::Terms<f64> = Terms::new(modes);
        for c in coeffs {
            state.coeffs.push(c);
        }
        assert_eq!(l2_norm_square(&state), expected);
    }

    #[rstest]
    #[case(vec![Complex64::new(3.0, 4.0)], 25.0)]
    #[case(vec![Complex64::new(0.0, 5.0)], 25.0)]
    #[case(vec![Complex64::new(1.0, 1.0)], 2.0)]
    #[case(vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -2.0)], 10.0)]
    #[case(vec![Complex64::new(-3.0, 4.0)], 25.0)]
    #[case(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)], 1.0)]
    #[case(vec![Complex64::new(1e-9, 1e-9)], 2e-18)]
    fn test_l2_norm_square_complex(
        #[case] coeffs: Vec<num_complex::Complex<f64>>,
        #[case] expected: f64,
    ) {
        let modes = Modes::from_count(coeffs.len());
        let mut state: terms::Terms<num_complex::Complex<f64>> = Terms::new(modes);
        for c in coeffs {
            state.coeffs.push(c);
        }
        assert_eq!(l2_norm_square(&state), expected);
    }
}
