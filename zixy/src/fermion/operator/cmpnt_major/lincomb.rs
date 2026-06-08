//! Fermion operator in linear combination utilities.

use crate::container::coeffs::traits::{FieldElem, FieldElemVec, HasCoeffs, NumRepr};
use crate::container::traits::proj::{Borrow, BorrowMut, ToOwned};
use crate::container::traits::Elements;
use crate::container::traits::RefElements;
use crate::container::word_iters::lincomb::{iadd, isub, scaled_iadd, scaled_iadd_elem};
use crate::container::word_iters::term_set::AsViewMut;
use crate::fermion::operator::cmpnt_major::term_set::{self, TermSet};
use crate::fermion::operator::cmpnt_major::terms;
use crate::fermion::operator::products::mul_cmpnts;
use crate::fermion::traits::{DifferentSpaces, ModesBased};
use num_complex::Complex64;

pub fn add<C: FieldElem>(lhs: &terms::View<C>, rhs: &terms::View<C>) -> TermSet<C> {
    let mut out = TermSet::from(lhs.to_owned());
    assign_from_add(&mut out.borrow_mut(), lhs, rhs);
    out
}

pub fn sub<C: FieldElem>(lhs: &terms::View<C>, rhs: &terms::View<C>) -> TermSet<C> {
    let mut out = TermSet::from(lhs.to_owned());
    assign_from_sub(&mut out.borrow_mut(), lhs, rhs);
    out
}

pub fn scaled_add<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
    scale: C,
) -> TermSet<C> {
    let mut out = TermSet::from(lhs.to_owned());
    scaled_iadd(&mut out.borrow_mut(), rhs, scale);
    out
}

// Assign lhs + rhs to out, normal-ordering each component product.
pub fn assign_from_add<C: FieldElem>(
    out: &mut term_set::ViewMut<Complex64>,
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<(), DifferentSpaces> {
    DifferentSpaces::check_transitive(out, lhs, rhs)?;
    out.clear();
    let n_lhs = lhs.word_iters.len().min(lhs.coeffs.len());
    let n_rhs = rhs.word_iters.len().min(rhs.coeffs.len());
    for (i_lhs, lhs_coeff) in lhs.coeffs.iter().take(n_lhs).enumerate() {
        let lhs_cmpnt = lhs.word_iters.get_elem_ref(i_lhs);
        for (i_rhs, rhs_coeff) in rhs.coeffs.iter().take(n_rhs).enumerate() {
            let rhs_cmpnt = rhs.word_iters.get_elem_ref(i_rhs);
            let (result_cmpnts, result_signs) = mul_cmpnts(&lhs_cmpnt, &rhs_cmpnt);
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

// Assign lhs * rhs to out, normal-ordering each component product.
pub fn assign_from_mul<C: FieldElem>(
    out: &mut term_set::ViewMut<Complex64>,
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<(), DifferentSpaces> {
    DifferentSpaces::check_transitive(out, lhs, rhs)?;
    out.clear();
    let n_lhs = lhs.word_iters.len().min(lhs.coeffs.len());
    let n_rhs = rhs.word_iters.len().min(rhs.coeffs.len());
    for (i_lhs, lhs_coeff) in lhs.coeffs.iter().take(n_lhs).enumerate() {
        let lhs_cmpnt = lhs.word_iters.get_elem_ref(i_lhs);
        for (i_rhs, rhs_coeff) in rhs.coeffs.iter().take(n_rhs).enumerate() {
            let rhs_cmpnt = rhs.word_iters.get_elem_ref(i_rhs);
            let (result_cmpnts, result_signs) = mul_cmpnts(&lhs_cmpnt, &rhs_cmpnt);
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
) -> Result<TermSet<Complex64>, DifferentSpaces> {
    let mut out = TermSet::<Complex64>::new(lhs.to_modes());
    assign_from_mul(&mut out.borrow_mut(), lhs, rhs)?;
    Ok(out)
}

// Assign the commutator [lhs, rhs] = lhs * rhs - rhs * lhs to out.
pub fn assign_from_commutator<C: FieldElem>(
    out: &mut term_set::ViewMut<Complex64>,
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<(), DifferentSpaces> {
    assign_from_mul(out, lhs, rhs)?;
    let tmp = mul(rhs, lhs)?.terms;
    isub(out, &tmp.borrow());
    Ok(())
}

// Assign the anticommutator {lhs, rhs} = lhs * rhs + rhs * lhs to out.
pub fn assign_from_anticommutator<C: FieldElem>(
    out: &mut term_set::ViewMut<Complex64>,
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<(), DifferentSpaces> {
    assign_from_mul(out, lhs, rhs)?;
    let tmp = mul(rhs, lhs)?.terms;
    iadd(out, &tmp.borrow());
    Ok(())
}

pub fn commutator<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<TermSet<Complex64>, DifferentSpaces> {
    let mut out = TermSet::<Complex64>::new(lhs.to_modes());
    assign_from_commutator(&mut out.borrow_mut(), lhs, rhs)?;
    Ok(out)
}

pub fn anticommutator<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<TermSet<Complex64>, DifferentSpaces> {
    let mut out = TermSet::<Complex64>::new(lhs.to_modes());
    assign_from_anticommutator(&mut out.borrow_mut(), lhs, rhs)?;
    Ok(out)
}

// Check if the commutator [lhs, rhs] is zero within the given tolerance.
pub fn commute<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
    atol: f64,
) -> Result<bool, DifferentSpaces> {
    Ok(commutator(lhs, rhs)?.get_coeffs().all_insignificant(atol))
}

// Check if the anticommutator {lhs, rhs} is zero within the given tolerance.
pub fn anticommute<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
    atol: f64,
) -> Result<bool, DifferentSpaces> {
    Ok(anticommutator(lhs, rhs)?
        .get_coeffs()
        .all_insignificant(atol))
}

// Check if the commutator [lhs, rhs] is zero within the default tolerance.
pub fn commute_default<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<bool, DifferentSpaces> {
    commute(lhs, rhs, C::COMMUTES_ATOL_DEFAULT)
}

// Check if the anticommutator {lhs, rhs} is zero within the default tolerance.
pub fn anticommute_default<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<bool, DifferentSpaces> {
    anticommute(lhs, rhs, C::COMMUTES_ATOL_DEFAULT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::traits::proj::Borrow;
    use crate::container::traits::{Elements, MutRefElements};
    use crate::container::word_iters::lincomb::scaled_iadd_elem;
    use crate::container::word_iters::term_set::AsView;
    use crate::container::word_iters::terms::AsViewMut;
    use crate::fermion::mode::Modes;
    use crate::fermion::operator::cmpnt::Cmpnt;
    use crate::fermion::operator::cmpnt_major::terms::Terms;
    use std::collections::HashSet;

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

    #[test]
    fn test_mul_bifurcation() {
        let modes = Modes::from_count(2);

        // lhs = a_0 annihilate mode 0
        let mut lhs = TermSet::<f64>::new(modes.clone());
        let a0 = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::new(), HashSet::from([0]));
        scaled_iadd_elem(&mut lhs.borrow_mut(), a0.borrow(), 1.0);

        // rhs = a_0^+ create mode 0
        let mut rhs = TermSet::<f64>::new(modes.clone());
        let a0_dag = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([0]), HashSet::new());
        scaled_iadd_elem(&mut rhs.borrow_mut(), a0_dag.borrow(), 1.0);

        //a_0 * a_0^+ = 1 - a_0^+ * a_0 -> two terms in the result
        let result = mul(&lhs.borrow().as_terms(), &rhs.borrow().as_terms()).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_commute_anticommute() {
        let modes = Modes::from_count(2);

        // lhs = a_0 annihilate mode 0
        let mut lhs = TermSet::<f64>::new(modes.clone());
        let a0 = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::new(), HashSet::from([0]));
        scaled_iadd_elem(&mut lhs.borrow_mut(), a0.borrow(), 1.0);

        // rhs = a_0^+ create mode 0
        let mut rhs = TermSet::<f64>::new(modes.clone());
        let a0_dag = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([0]), HashSet::new());
        scaled_iadd_elem(&mut rhs.borrow_mut(), a0_dag.borrow(), 1.0);

        // a_0 and a_0^+ neither commute nor anticommute to zero
        assert!(!commute(&lhs.borrow().as_terms(), &rhs.borrow().as_terms(), 1e-10).unwrap());
        assert!(!anticommute(&lhs.borrow().as_terms(), &rhs.borrow().as_terms(), 1e-10).unwrap());
    }

    #[test]
    fn test_mul_with_coefficients() {
        let modes = Modes::from_count(2);

        // lhs = a_0 annihilate mode 0
        let mut lhs = TermSet::<f64>::new(modes.clone());
        let a0 = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::new(), HashSet::from([0]));
        scaled_iadd_elem(&mut lhs.borrow_mut(), a0.borrow(), 2.0);

        // rhs = a_0^+ create mode 0
        let mut rhs = TermSet::<f64>::new(modes.clone());
        let a0_dag = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([0]), HashSet::new());
        scaled_iadd_elem(&mut rhs.borrow_mut(), a0_dag.borrow(), 3.0);

        let result = mul(&lhs.borrow().as_terms(), &rhs.borrow().as_terms()).unwrap();
        let coeffs: Vec<Complex64> = result.get_coeffs().to_vec();
        assert_eq!(coeffs.len(), 2);
        assert!(coeffs.contains(&Complex64::new(6.0, 0.0)));
        assert!(coeffs.contains(&Complex64::new(-6.0, 0.0)));
    }
}
