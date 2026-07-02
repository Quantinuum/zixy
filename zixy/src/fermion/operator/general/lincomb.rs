//! Fermion operator in linear combination utilities.

use crate::container::bit_matrix::AsRowRef;
use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::container::traits::{Elements, RefElements};
use crate::container::word_iters::lincomb::scaled_iadd;
use crate::container::word_iters::term_set::{
    AsView as TermSetAsView, AsViewMut as TermSetAsViewMut,
};
use crate::fermion::operator::general::term_set::{
    self as general_term_set, TermSet as GeneralTermSet,
};
use crate::fermion::operator::normal::cmpnt::Cmpnt;
use crate::fermion::operator::normal::cmpnt_major::lincomb::mul as normal_mul;
use crate::fermion::operator::normal::cmpnt_major::term_set::{
    self as normal_term_set, TermSet as NormalTermSet,
};
use crate::fermion::traits::{DifferentSpaces, ModesBased};
use itertools::Itertools;
use num_complex::Complex64;
use std::collections::HashSet;

/// Multiply two raw term sets without normal ordering, returning a raw term set.
pub fn rmul<C: FieldElem>(
    lhs: &general_term_set::View<C>,
    rhs: &general_term_set::View<C>,
) -> Result<GeneralTermSet<Complex64>, DifferentSpaces> {
    let max_len = lhs.word_iters.max_len + rhs.word_iters.max_len;
    let mut out = GeneralTermSet::<Complex64>::new(max_len, lhs.to_modes());
    DifferentSpaces::check_transitive(lhs, rhs, &out)?;
    let n_lhs = lhs.word_iters.len().min(lhs.coeffs.len());
    let n_rhs = rhs.word_iters.len().min(rhs.coeffs.len());
    for (i_lhs, lhs_coeff) in lhs.coeffs.iter().take(n_lhs).enumerate() {
        let (lhs_modes, lhs_adj) = lhs.word_iters.get(i_lhs);
        for (i_rhs, rhs_coeff) in rhs.coeffs.iter().take(n_rhs).enumerate() {
            let (rhs_modes, rhs_adj) = rhs.word_iters.get(i_rhs);
            let modes: Vec<usize> = lhs_modes.iter().chain(rhs_modes.iter()).copied().collect();
            let adj: Vec<bool> = lhs_adj.iter().chain(rhs_adj.iter()).copied().collect();
            let c = lhs_coeff.to_complex() * rhs_coeff.to_complex();
            out.push_term(&modes, &adj, c);
        }
    }
    Ok(out)
}

/// Convert a raw term set to a normal-ordered term set by applying fermionic anticommutation relations.
pub fn normalise<C: FieldElem>(terms: &general_term_set::View<C>) -> NormalTermSet<Complex64> {
    let mut out = NormalTermSet::<Complex64>::new(terms.to_modes().clone());
    let n_terms = terms.word_iters.len().min(terms.coeffs.len());
    for (index, coeff) in terms.coeffs.iter().take(n_terms).enumerate() {
        let (modes, adj) = terms.word_iters.get(index);
        let modes_space = terms.word_iters.modes().clone();
        let mut current = NormalTermSet::<Complex64>::new(modes_space.clone());
        let identity = Cmpnt::new(modes_space.clone());
        let _ = current
            .borrow_mut()
            .insert_elem_ref_or_update(identity.borrow(), Complex64::new(1.0, 0.0));

        for (mode, is_cre) in modes.iter().zip(adj.iter()) {
            let mut cre_set = HashSet::new();
            let mut ann_set = HashSet::new();
            if *is_cre {
                cre_set.insert(*mode);
            } else {
                ann_set.insert(*mode);
            }
            let rhs = Cmpnt::from_sets_unchecked(modes_space.clone(), cre_set, ann_set);
            let mut rhs_set = NormalTermSet::<Complex64>::new(modes_space.clone());
            let _ = rhs_set
                .borrow_mut()
                .insert_elem_ref_or_update(rhs.borrow(), Complex64::new(1.0, 0.0));
            let product = normal_mul(&current.borrow().as_terms(), &rhs_set.borrow().as_terms())
                .expect("normalise: mode spaces should be compatible");
            current = product;
        }

        scaled_iadd(
            &mut out.borrow_mut(),
            &current.borrow().as_terms(),
            coeff.to_complex(),
        );
    }
    out
}

/// Convert a normal-ordered term set to a raw term set.
pub fn generalise<C: FieldElem>(terms: &normal_term_set::View<C>) -> GeneralTermSet<Complex64> {
    let max_len = 2 * terms.word_iters.modes().len();
    let mut out = GeneralTermSet::<Complex64>::new(max_len, terms.to_modes().clone());
    let n_terms = terms.word_iters.len().min(terms.coeffs.len());
    for (i, coeff) in terms.coeffs.iter().take(n_terms).enumerate() {
        let cmpnt = terms.word_iters.get_elem_ref(i);
        let mut modes = Vec::new();
        let mut adj = Vec::new();
        for mode in cmpnt.get_cre_part().to_set().into_iter().sorted() {
            modes.push(mode);
            adj.push(true);
        }
        for mode in cmpnt.get_ann_part().to_set().into_iter().sorted() {
            modes.push(mode);
            adj.push(false);
        }
        out.push_term(&modes, &adj, coeff.to_complex());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::traits::proj::Borrow;
    use crate::container::traits::Elements;
    use crate::container::word_iters::lincomb::scaled_iadd_elem;
    use crate::container::word_iters::term_set::AsView;
    use crate::fermion::mode::Modes;
    use crate::fermion::operator::general::term_set::TermSet as GeneralTermSet;
    use crate::fermion::operator::normal::cmpnt::Cmpnt;
    use crate::fermion::operator::normal::cmpnt_major::term_set::TermSet as NormalTermSet;
    use std::collections::HashSet;

    fn make_raw(n_modes: usize, modes: &[usize], adj: &[bool]) -> GeneralTermSet<f64> {
        let modes_space = Modes::from_count(n_modes);
        let mut raw = GeneralTermSet::<f64>::new(modes.len(), modes_space);
        raw.push_term(modes, adj, 1.0_f64);
        raw
    }

    fn check_term(
        result: &NormalTermSet<Complex64>,
        n_modes: usize,
        cre: HashSet<usize>,
        ann: HashSet<usize>,
        coeff: Complex64,
    ) {
        let modes_space = Modes::from_count(n_modes);
        let expected_cmpnt = Cmpnt::from_sets_unchecked(modes_space, cre.clone(), ann.clone());
        let found = result.as_terms().iter().any(|t| {
            t.get_word_iter_ref() == expected_cmpnt.borrow()
                && (t.get_coeff().re - coeff.re).abs() < 1e-10
                && (t.get_coeff().im - coeff.im).abs() < 1e-10
        });
        assert!(
            found,
            "term cre={cre:?} ann={ann:?} coeff={coeff} not found"
        );
    }

    #[test]
    fn test_generalise_preserves_order_and_coeffs() {
        let modes = Modes::from_count(4);
        let mut terms = NormalTermSet::<f64>::new(modes.clone());
        let cmpnt =
            Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([2]), HashSet::from([0, 1]));
        scaled_iadd_elem(&mut terms.borrow_mut(), cmpnt.borrow(), 2.0);

        let result = generalise(&terms.borrow());
        assert_eq!(result.terms.word_iters.len(), 1);
        let (raw_modes, raw_adj) = result.terms.word_iters.get(0);
        assert_eq!(raw_modes, vec![2, 0, 1]);
        assert_eq!(raw_adj, vec![true, false, false]);
        assert_eq!(result.terms.coeffs[0], Complex64::new(2.0, 0.0));
    }

    #[test]
    fn test_normalise_a0_a0dag() {
        // a_0 a_0^+ -> 1 - a_0^+ a_0
        let raw = make_raw(4, &[0, 0], &[false, true]);
        let result = normalise(&raw.as_terms());
        assert_eq!(result.len(), 2);
        check_term(
            &result,
            4,
            HashSet::new(),
            HashSet::new(),
            Complex64::new(1.0, 0.0),
        );
        check_term(
            &result,
            4,
            HashSet::from([0]),
            HashSet::from([0]),
            Complex64::new(-1.0, 0.0),
        );
    }

    #[test]
    fn test_normalise_a0_a1dag() {
        // a_0 a_1^+ -> -a_1^+ a_0
        let raw = make_raw(4, &[0, 1], &[false, true]);
        let result = normalise(&raw.as_terms());
        assert_eq!(result.len(), 1);
        check_term(
            &result,
            4,
            HashSet::from([1]),
            HashSet::from([0]),
            Complex64::new(-1.0, 0.0),
        );
    }

    #[test]
    fn test_normalise_a0dag_a1_a0() {
        // a_0^+ a_1 a_0 -> -a_0^+ a_0 a_1
        let raw = make_raw(4, &[0, 1, 0], &[true, false, false]);
        let result = normalise(&raw.as_terms());
        assert_eq!(result.len(), 1);
        check_term(
            &result,
            4,
            HashSet::from([0]),
            HashSet::from([0, 1]),
            Complex64::new(-1.0, 0.0),
        );
    }

    #[test]
    fn test_raw_mul() {
        let modes_space = Modes::from_count(2);

        // lhs = a_0^+ + 2*a_1^+
        let mut lhs = GeneralTermSet::<f64>::new(1, modes_space.clone());
        lhs.push_term(&[0], &[true], 1.0_f64);
        lhs.push_term(&[1], &[true], 2.0_f64);

        // rhs = 3*a_0
        let mut rhs = GeneralTermSet::<f64>::new(1, modes_space.clone());
        rhs.push_term(&[0], &[false], 3.0_f64);

        let result = rmul(&lhs.as_terms(), &rhs.as_terms()).unwrap();

        let check = |modes: &[usize], adj: &[bool], coeff: Complex64| {
            let n = result.terms.word_iters.len();
            let found = (0..n).any(|i| {
                let (m, a) = result.terms.word_iters.get(i);
                let c = result.terms.coeffs[i];
                m == modes
                    && a == adj
                    && (c.re - coeff.re).abs() < 1e-10
                    && (c.im - coeff.im).abs() < 1e-10
            });
            assert!(
                found,
                "Non-normal-ordered term {modes:?}/{adj:?} coeff {coeff} not found"
            );
        };

        assert_eq!(result.terms.word_iters.len(), 2);
        check(&[0, 0], &[true, false], Complex64::new(3.0, 0.0));
        check(&[1, 0], &[true, false], Complex64::new(6.0, 0.0));
    }
}
