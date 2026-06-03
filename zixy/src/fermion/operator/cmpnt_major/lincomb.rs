//! Fermion operator in linear combination utilities.

use crate::container::word_iters::lincomb::{iadd,isub, scaled_iadd};
use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::fermion::operator::cmpnt_major::term_set::{self, TermSet};
use crate::fermion::operator::cmpnt_major::terms;
use crate::fermion::errors::DifferentModes;

pub fn add<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> TermSet<C> {
    let mut out = TermSet::from(lhs);
    iadd(&mut out.borrow_mut(), rhs);
    out
}

pub fn sub<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> TermSet<C> {
    let mut out = TermSet::from(lhs);
    isub(&mut out.borrow_mut(), rhs);
    out
}

pub fn scaled_add<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
    scale: C,
) -> TermSet<C> {
    let mut out = TermSet::from(lhs);
    scaled_iadd(&mut out.borrow_mut(), rhs, scale);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fermion::operator::cmpnt_major::terms::Terms;
    use crate::fermion::mode::Modes;

    #[test]
    fn test_add_sub_scaled_add() {
        let mut a = Terms::<f64>::new(Modes::from_count(4));
        a.push_clear();
        a.get_elem_mut_ref(0).set_coeff(1.0);

        let mut b = Terms::<f64>::new(Modes::from_count(4));
        b.push_clear();
        b.get_elem_mut_ref(0).set_coeff(1.0);

        let sum = add(&a.borrow(), &b.borrow());
        assert_eq!(sum.len(), 1);
        assert_eq!(sum.view().get_coeffs()[0], 2.0);

        let diff = sub(&a.borrow(), &b.borrow());
        assert_eq!(diff.len(), 0);

        let scaled = scaled_add(&a.borrow(), &b.borrow(), 0.5);
        assert_eq!(scaled.len(), 1);
        assert_eq!(scaled.view().get_coeffs()[0], 1.5);
    }
}