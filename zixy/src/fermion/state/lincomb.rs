//! Fermion state-specific linear combination utilities.

use crate::container::coeffs::traits::FieldElem;
//use crate::container::errors::{Dimension, OutOfBounds};
//use crate::container::traits::proj::BorrowMut;
use crate::container::traits::RefElements;
//use crate::container::word_iters;
//use crate::container::word_iters::term_set::AsViewMut;
//use crate::fermion::mode::Modes;
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
