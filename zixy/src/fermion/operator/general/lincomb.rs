//! Fermion operator in linear combination utilities.

use crate::container::bit_matrix::AsRowRef;
use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::container::traits::Elements;
use crate::container::traits::RefElements;
use crate::container::word_iters::lincomb::{scaled_iadd, scaled_iadd_elem};
use crate::container::word_iters::term_set::AsView;
use crate::fermion::operator::general::raw_term_set;
use crate::fermion::operator::general::raw_term_set::RawTermSet;
use crate::fermion::operator::normal::cmpnt::Cmpnt;
use crate::fermion::operator::normal::cmpnt_major::lincomb::mul;
use crate::fermion::operator::normal::cmpnt_major::term_set::{self, TermSet};
use crate::fermion::traits::{DifferentSpaces, ModesBased};
use num_complex::Complex64;
use std::collections::HashSet;

/// Multiply two `RawTermSet` without normal ordering, returning a `RawTermSet`.
/// Use `normalise` to convert the result to a normal-ordered `TermSet`.
pub fn raw_mul<C: FieldElem>(
    lhs: &raw_term_set::View<C>,
    rhs: &raw_term_set::View<C>,
) -> Result<RawTermSet<Complex64>, DifferentSpaces> {
    let max_len = lhs.word_iters.max_len + rhs.word_iters.max_len;
    let mut out = RawTermSet::<Complex64>::new(max_len, lhs.to_modes());
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

/// Convert a `RawTermSet` to a normal-ordered `TermSet` by applying fermionic anticommutation relations.
pub fn normalise<C: FieldElem>(raw_terms: &raw_term_set::View<C>) -> TermSet<Complex64> {
    let mut out = TermSet::<Complex64>::new(raw_terms.to_modes());
    let n_terms = raw_terms.word_iters.len().min(raw_terms.coeffs.len());
    for (index, coeff) in raw_terms.coeffs.iter().take(n_terms).enumerate() {
        let (modes, adj) = raw_terms.word_iters.get(index);
        // start with the first operator as initial Termset
        let modes_space = raw_terms.word_iters.modes().clone();
        let mut cre_set = HashSet::<usize>::new();
        let mut ann_set = HashSet::<usize>::new();
        let mut acc = TermSet::<Complex64>::new(modes_space.clone());
        if !modes.is_empty() {
            if adj[0] {
                cre_set.insert(modes[0]);
            } else {
                ann_set.insert(modes[0]);
            }
        }
        scaled_iadd_elem(
            &mut acc.borrow_mut(),
            Cmpnt::from_sets_unchecked(modes_space.clone(), cre_set.clone(), ann_set.clone())
                .borrow(),
            Complex64::new(1.0, 0.0),
        );
        // multiply with each remaing operator
        for (mode, is_cre) in modes
            .get(1..)
            .unwrap_or(&[])
            .iter()
            .zip(adj.get(1..).unwrap_or(&[]).iter())
        {
            let mut cre_set = HashSet::<usize>::new();
            let mut ann_set = HashSet::<usize>::new();
            if *is_cre {
                cre_set.insert(*mode);
            } else {
                ann_set.insert(*mode);
            };

            let rhs = Cmpnt::from_sets_unchecked(modes_space.clone(), cre_set, ann_set);
            let mut rhs_set = TermSet::<Complex64>::new(modes_space.clone());
            scaled_iadd_elem(
                &mut rhs_set.borrow_mut(),
                rhs.borrow(),
                Complex64::new(1.0, 0.0),
            );
            acc = mul(&acc.borrow().as_terms(), &rhs_set.borrow().as_terms())
                .expect("normalise: acc and lhs_set should always have the same mode space")
        }
        // add acc into out with the raw coefficient
        scaled_iadd(
            &mut out.borrow_mut(),
            &acc.borrow_mut().as_terms(),
            coeff.to_complex(),
        );
    }
    out
}

/// Convert a normal-ordered `TermSet` to a `RawTermSet`.
pub fn generalise<C: FieldElem>(terms: &term_set::View<C>) -> RawTermSet<Complex64> {
    let max_len = 2 * terms.word_iters.modes().len();
    let mut out = RawTermSet::<Complex64>::new(max_len, terms.to_modes());
    let n_terms = terms.word_iters.len().min(terms.coeffs.len());
    for (i, coeff) in terms.coeffs.iter().take(n_terms).enumerate() {
        let cmpnt = terms.word_iters.get_elem_ref(i);
        let mut modes = Vec::new();
        let mut adj = Vec::new();
        // add creation modes first (normal order)
        for mode in cmpnt.get_cre_part().iter_set_bits_flat() {
            modes.push(mode);
            adj.push(true);
        }
        // add annihilation modes after
        for mode in cmpnt.get_ann_part().iter_set_bits_flat() {
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
    use crate::fermion::operator::general::raw_term_set::RawTermSet;
    use crate::fermion::operator::normal::cmpnt::Cmpnt;
    use std::collections::HashSet;

    fn make_raw(n_modes: usize, modes: &[usize], adj: &[bool]) -> RawTermSet<f64> {
        let modes_space = Modes::from_count(n_modes);
        let mut raw = RawTermSet::<f64>::new(modes.len(), modes_space);
        raw.push_term(modes, adj, 1.0_f64);
        raw
    }

    fn check_term(
        result: &TermSet<Complex64>,
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
        let mut terms = TermSet::<f64>::new(modes.clone());
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
        let result = normalise(&raw.as_raw_terms());
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
        let result = normalise(&raw.as_raw_terms());
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
        let result = normalise(&raw.as_raw_terms());
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
        let mut lhs = RawTermSet::<f64>::new(1, modes_space.clone());
        lhs.push_term(&[0], &[true], 1.0_f64);
        lhs.push_term(&[1], &[true], 2.0_f64);

        // rhs = 3*a_0
        let mut rhs = RawTermSet::<f64>::new(1, modes_space.clone());
        rhs.push_term(&[0], &[false], 3.0_f64);

        let result = raw_mul(&lhs.as_raw_terms(), &rhs.as_raw_terms()).unwrap();

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
            assert!(found, "raw term {modes:?}/{adj:?} coeff {coeff} not found");
        };

        assert_eq!(result.terms.word_iters.len(), 2);
        check(&[0, 0], &[true, false], Complex64::new(3.0, 0.0)); // a_0^+ a_0, coeff 1*3 = 3
        check(&[1, 0], &[true, false], Complex64::new(6.0, 0.0)); // a_1^+ a_0, coeff 2*3 = 6
    }
}
