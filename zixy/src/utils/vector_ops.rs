//! Vector operations on linear combinations of basis states.
use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::RefElements;
use crate::container::word_iters::{term_set, terms, WordIters};

/// Sum of squares of the coefficients of a basis state linear combination.
pub fn l2_norm_square<T: WordIters, C: FieldElem>(state: &impl terms::AsView<T, C>) -> f64 {
    state.view().coeffs.iter().map(|c| c.magnitude_sq()).sum()
}

/// L2 norm of the coefficients of a basis state linear combination.
pub fn l2_norm<T: WordIters, C: FieldElem>(state: &impl terms::AsView<T, C>) -> f64 {
    l2_norm_square(state).sqrt()
}

/// Take the inner product of a basis state linear combination with another.
pub fn vdot<T: WordIters, C: FieldElem>(
    lhs: &impl term_set::AsView<T, C>,
    rhs: &impl terms::AsView<T, C>,
) -> C {
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
