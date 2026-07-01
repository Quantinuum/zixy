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
