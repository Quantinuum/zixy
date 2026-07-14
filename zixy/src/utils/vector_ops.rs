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

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use rstest::rstest;

    use crate::container::traits::proj::{Borrow, BorrowMut};
    use crate::container::traits::EmptyFrom;
    use crate::container::word_iters::term_set::test_defs::StringCmpnts;
    use crate::container::word_iters::term_set::AsView;
    use crate::container::word_iters::HasWordIters;
    use crate::container::word_iters::{term_set, Elem};

    use super::*;

    const N_CHAR_MAX: usize = 8;

    fn add_from_str<C: FieldElem>(
        lhs: &mut term_set::ViewMut<'_, StringCmpnts, C>,
        key: &str,
        scalar: C,
    ) {
        let mut tmp = Elem::<StringCmpnts>::from(lhs.get_word_iters());
        tmp.borrow_mut().assign_from_str(key);
        crate::container::word_iters::lincomb::scaled_iadd_elem(lhs, tmp.borrow(), scalar);
    }

    fn build_term_set<C: FieldElem>(entries: &[(&str, C)]) -> term_set::TermSet<StringCmpnts, C> {
        let mut ts =
            term_set::TermSet::<StringCmpnts, C>::empty_from(&StringCmpnts::new(N_CHAR_MAX));
        for (key, coeff) in entries {
            add_from_str(&mut ts.borrow_mut(), key, *coeff);
        }
        ts
    }

    #[rstest]
    #[case(vec![], 0.0)]
    #[case(vec![2.0], 4.0)]
    #[case(vec![-3.0], 9.0)]
    #[case(vec![1.0, 2.0, 3.0], 14.0)]
    #[case(vec![0.0, 0.0, 0.0], 0.0)]
    #[case(vec![0.0, 2.0, 0.0], 4.0)]
    fn test_l2_norm_square(#[case] coeffs: Vec<f64>, #[case] expected: f64) {
        let labels = ["a", "b", "c"];
        let entries: Vec<(&str, f64)> = labels.into_iter().zip(coeffs).collect();
        let ts = build_term_set(&entries);
        assert_eq!(l2_norm_square(&ts.as_terms()), expected);
    }

    #[rstest]
    #[case(vec![Complex64::new(3.0, 4.0)], 25.0)]
    #[case(vec![Complex64::new(0.0, 5.0)], 25.0)]
    #[case(vec![Complex64::new(1.0, 1.0)], 2.0)]
    #[case(vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -2.0)], 10.0)]
    #[case(vec![Complex64::new(-3.0, 4.0)], 25.0)]
    #[case(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)], 1.0)]
    #[case(vec![Complex64::new(1e-9, 1e-9)], 2e-18)]
    fn test_l2_norm_square_complex(#[case] coeffs: Vec<Complex64>, #[case] expected: f64) {
        let labels = ["a", "b"];
        let entries: Vec<(&str, Complex64)> = labels.into_iter().zip(coeffs).collect();
        let ts = build_term_set(&entries);
        assert_eq!(l2_norm_square(&ts.as_terms()), expected);
    }

    #[rstest]
    // <v1|v2> where |v1> = 2|red> and |v2> = 3|red>, expected <v1|v2> = 6
    #[case(vec![("red", Complex64::new(2.0, 0.0))], vec![("red", Complex64::new(3.0, 0.0))], Complex64::new(6.0, 0.0))]
    // <v1|v2> where |v1> = null and |v2> = (3+4i)|blue>, expected <v1|v2> = 0
    #[case(vec![], vec![("blue", Complex64::new(3.0, 4.0))], Complex64::new(0.0, 0.0))]
    // <v1|v2> where |v1> = (1+2i)|red> and |v2> = null, expected <v1|v2> = 0
    #[case(vec![("red", Complex64::new(1.0, 2.0))], vec![], Complex64::new(0.0, 0.0))]
    // <v1|v2> where |v1> = (1+2i)|red> and |v2> = (3+4i)|red>, expected <v1|v2> = (1-2i)(3+4i) = 11-2i
    #[case(vec![("red", Complex64::new(1.0, 2.0))], vec![("red", Complex64::new(3.0, 4.0))], Complex64::new(11.0, -2.0))]
    // orthogonal basis states, expected <v1|v2> = 0
    #[case(vec![("red", Complex64::new(1.0, 2.0))], vec![("blue", Complex64::new(3.0, 4.0))], Complex64::new(0.0, 0.0))]
    // |v1> = (1+2i)|red> + (-1-2.5i)|green>, |v2> = (3+4i)|red> + i|blue>, expected = 11-2i
    #[case(vec![("red", Complex64::new(1.0, 2.0)), ("green", Complex64::new(-1.0, -2.5))], vec![("red", Complex64::new(3.0, 4.0)), ("blue", Complex64::new(0.0, 1.0))], Complex64::new(11.0, -2.0))]
    // |v1> = (1+2i)|red> + (-1-2.5i)|green>, |v2> = (3+4i)|red> + i|green>, expected = 8.5-3i
    #[case(vec![("red", Complex64::new(1.0, 2.0)), ("green", Complex64::new(-1.0, -2.5))], vec![("red", Complex64::new(3.0, 4.0)), ("green", Complex64::new(0.0, 1.0))], Complex64::new(8.5, -3.0))]
    fn test_vdot(
        #[case] lhs_entries: Vec<(&str, Complex64)>,
        #[case] rhs_entries: Vec<(&str, Complex64)>,
        #[case] expected: Complex64,
    ) {
        let lhs = build_term_set(&lhs_entries);
        let rhs = build_term_set(&rhs_entries);
        assert_eq!(vdot(&lhs, &rhs.as_terms()), expected);
    }
}
