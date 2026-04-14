//! Utilities for viewing term sets of real and complex coefficient types as linear combinations.

use crate::container::coeffs::traits::{FieldElem, HasCoeffsMut};
use crate::container::traits::proj::BorrowMut;
use crate::container::traits::{proj, RefElements};
use crate::container::word_iters::term_set::AsViewMut;
use crate::container::word_iters::{self, term_set, terms, WordIters};

/// Perform in-place multiply add, lhs += scalar * rhs, where rhs is an iterator over u64 words.
pub fn scaled_iadd_u64it<'a, T: WordIters, C: FieldElem>(
    lhs: &mut term_set::ViewMut<'a, T, C>,
    rhs: impl Iterator<Item = u64> + Clone,
    scalar: C,
) {
    let (index, inserted) = lhs.insert_u64it_or_get_index(rhs, scalar);
    let tmp = &mut lhs.get_coeffs_mut()[index];
    if !inserted {
        *tmp += scalar;
    }
    if *tmp == C::default() {
        // remove identical zeros
        lhs.drop_index(index);
    }
}

/// Perform in-place multiply add, lhs += scalar * rhs, where rhs is a reference to an element of a `WordIters`.
pub fn scaled_iadd_elem<'a, T: WordIters, C: FieldElem>(
    lhs: &mut term_set::ViewMut<'a, T, C>,
    rhs: word_iters::ElemRef<T>,
    scalar: C,
) {
    scaled_iadd_u64it(lhs, rhs.get_u64it(), scalar);
}

/// Perform in-place multiply add, lhs += scalar * rhs
pub fn scaled_iadd<'a, 'b, T: WordIters, C: FieldElem>(
    lhs: &mut term_set::ViewMut<'a, T, C>,
    rhs: &terms::View<'b, T, C>,
    scalar: C,
) {
    rhs.iter()
        .for_each(|term| scaled_iadd_u64it(lhs, term.get_u64it(), scalar * term.get_coeff()));
}

/// Perform in-place add, lhs += rhs
pub fn iadd<'a, 'b, T: WordIters, C: FieldElem>(
    lhs: &mut term_set::ViewMut<'a, T, C>,
    rhs: &terms::View<'b, T, C>,
) {
    scaled_iadd(lhs, rhs, C::ONE)
}

/// Perform in-place subtract, lhs -= rhs
pub fn isub<'a, 'b, T: WordIters, C: FieldElem>(
    lhs: &mut term_set::ViewMut<'a, T, C>,
    rhs: &terms::View<'b, T, C>,
) {
    scaled_iadd(lhs, rhs, -C::ONE)
}

/// Perform in-place multiplication by a scalar, i.e. lhs *= scalar
pub fn scale<'a, T: WordIters, C: FieldElem>(lhs: &mut term_set::ViewMut<'a, T, C>, scalar: C) {
    lhs.get_coeffs_mut().iter_mut().for_each(|c| *c *= scalar);
}

/// Perform out-of-place multiply-add, lhs + c * rhs
pub fn scaled_sum<'a, 'b, T: WordIters, C: FieldElem>(
    lhs: &terms::View<'a, T, C>,
    rhs: &terms::View<'b, T, C>,
    scalar: C,
) -> term_set::TermSet<T, C> {
    let out = proj::ToOwned::to_owned(lhs);
    let mut out = term_set::TermSet::from(out);
    scaled_iadd(&mut out.borrow_mut(), rhs, scalar);
    out
}

/// Perform out-of-place add, lhs + rhs
pub fn sum<'a, 'b, T: WordIters, C: FieldElem>(
    lhs: &terms::View<'a, T, C>,
    rhs: &terms::View<'b, T, C>,
) -> term_set::TermSet<T, C> {
    scaled_sum(lhs, rhs, C::ONE)
}

/// Perform out-of-place substract, lhs - rhs
pub fn diff<'a, 'b, T: WordIters, C: FieldElem>(
    lhs: &terms::View<'a, T, C>,
    rhs: &terms::View<'b, T, C>,
) -> term_set::TermSet<T, C> {
    scaled_sum(lhs, rhs, -C::ONE)
}

/// Perform in-place multiplication by a scalar, i.e. lhs *= scalar
pub fn scaled<'a, T: WordIters, C: FieldElem>(
    lhs: &terms::View<'a, T, C>,
    scalar: C,
) -> term_set::TermSet<T, C> {
    let out = proj::ToOwned::to_owned(lhs);
    let mut out = term_set::TermSet::from(out);
    scale(&mut out.borrow_mut(), scalar);
    out
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::container::coeffs::traits::HasCoeffs;
    use crate::container::traits::proj::{Borrow, BorrowMut};
    use crate::container::traits::{Elements, EmptyFrom};
    use crate::container::word_iters::_word_iters::HasWordIters;
    use crate::container::word_iters::term_set::test_defs::StringCmpnts;
    use crate::container::word_iters::term_set::AsView;
    use crate::container::word_iters::{term_set, Elem};

    use super::*;

    fn add_from_str<'a, C: FieldElem>(
        lhs: &mut term_set::ViewMut<'a, StringCmpnts, C>,
        key: &str,
        scalar: C,
    ) {
        let mut tmp = Elem::<StringCmpnts>::from(lhs.get_word_iters());
        tmp.borrow_mut().assign_from_str(key);
        crate::container::word_iters::lincomb::scaled_iadd_elem(lhs, tmp.borrow(), scalar);
    }

    impl<'a, C: FieldElem> term_set::View<'a, StringCmpnts, C> {
        /// Return coeff associated with `key` if `key` is to be found in `self`, otherwise `None`.
        fn get_coeff_of_str(&self, key: &str) -> Option<C> {
            let mut tmp = Elem::<StringCmpnts>::from(self.get_word_iters());
            tmp.borrow_mut().assign_from_str(key);
            self.lookup_coeff_elem_ref(tmp.borrow())
        }
    }

    #[test]
    fn test_real_lin_comb() {
        const N_CHAR_MAX: usize = 20;
        let mut lc = term_set::TermSet::<_, f64>::empty_from(&StringCmpnts::new(N_CHAR_MAX));
        add_from_str(&mut lc.borrow_mut(), "red", 0.0);
        // adding zero should not result in a new entry
        assert_eq!(lc.len(), 0);
        add_from_str(&mut lc.borrow_mut(), "red", 0.3);
        add_from_str(&mut lc.borrow_mut(), "red", 0.2);
        assert_eq!(lc.len(), 1);
        assert_eq!(lc.get_coeffs().len(), 1);
        assert_eq!(lc.borrow().get_coeff_of_str("red"), Some(0.5));
        add_from_str(&mut lc.borrow_mut(), "blue", 0.2);
        assert_eq!(lc.len(), 2);
        assert_eq!(lc.get_coeffs().len(), 2);
        add_from_str(&mut lc.borrow_mut(), "green", 0.6);
        assert_eq!(lc.len(), 3);
        assert_eq!(lc.get_coeffs().len(), 3);
        assert_eq!(lc.to_string(), "(0.5, red), (0.2, blue), (0.6, green)");
        scale(&mut lc.borrow_mut(), 2.0);
        assert_eq!(lc.to_string(), "(1, red), (0.4, blue), (1.2, green)");
    }

    #[test]
    fn test_complex_lin_comb() {
        const N_CHAR_MAX: usize = 20;
        let mut lc = term_set::TermSet::<_, Complex64>::empty_from(&StringCmpnts::new(N_CHAR_MAX));
        add_from_str(&mut lc.borrow_mut(), "red", Complex64::new(1.0, 1.0));
        add_from_str(&mut lc.borrow_mut(), "red", Complex64::new(0.5, -0.5));
        assert_eq!(lc.len(), 1);
        assert_eq!(lc.get_coeffs().len(), 1);
        assert_eq!(
            lc.borrow().get_coeff_of_str("red"),
            Some(Complex64::new(1.5, 0.5))
        );
        add_from_str(&mut lc.borrow_mut(), "blue", Complex64::new(1.0, 2.0));
        assert_eq!(lc.get_coeffs().len(), 2);
        add_from_str(&mut lc.borrow_mut(), "green", Complex64::new(3.0, 1.0));
        assert_eq!(lc.get_coeffs().len(), 3);
        assert_eq!(
            lc.to_string(),
            "(1.5+0.5i, red), (1+2i, blue), (3+1i, green)"
        );
        let lc_re: term_set::TermSet<StringCmpnts, f64> = lc.real_part();
        assert_eq!(lc_re.to_string(), "(1.5, red), (1, blue), (3, green)");
        let lc_im: term_set::TermSet<StringCmpnts, f64> = lc.imag_part();
        assert_eq!(lc_im.to_string(), "(0.5, red), (2, blue), (1, green)");
    }
}
