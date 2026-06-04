//! Fermion operator in linear combination utilities.

use crate::container::word_iters::lincomb::{iadd,isub, scaled_iadd, scaled_iadd_elem};
use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::fermion::traits::ModesBased;
use crate::fermion::operator::products::mul_cmpnts;
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

pub fn assign_from_add<C: FieldElem>(
    out: &mut term_set::ViewMut<Complex64>,
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<(), DifferentModes> {
    DifferentModes::check(out.word_oters, lhs.word_iters, rhs.word_iters)?;
    out.clear();
    let n_lhs = lhs.word_iters.len().min(lhs.coeffs.len());
    let n_rhs = rhs.word_iters.len().min(rhs.coeffs.len());
    for (i_lhs, lhs_coeff) in lhs.coeffs.iter().take(n_lhs).enumerate() {
        let lhs_cmpnt = lhs.word_iters.get_elem_ref(i_lhs);
        for (i_rhs, rhs_coeff) in rhs.coeffs.iter().take(n_rhs).enumerate() {
            let rhs_cmpnt = rhs.word_iters.get_elem_ref(i_rhs);
            let (result_cmpnts, result_signs) = mul_cmpnts(lhs_cmpnt, rhs_cmpnt);
            for (i_res, sign) in result_signs.iter().enumerate() {
                let result_cmpnt = result_cmpnts.get_elem_ref(i_res);
                let c = sign.to_complex();
                let c = lhs_coeff.scaled_complex(c);
                let c = rhs_coeff.scaled_complex(c);
                scaled_iadd_elem(out, result_cmpnt, c);
            }
            
        }
    }
    Ok(())
}

pub fn mul<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<TermSet<Complex64>, DifferentModes> {
    let mut out = TermSet::<Complex64>::new(lhs.to_modes());
    assign_from_mul(&mut out.borrow_mut(), lhs, rhs)?;
    Ok(out)
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