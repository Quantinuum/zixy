//! Fermion operator in linear combination utilities, including conversions between
//! raw and normal-ordered representations.

use crate::container::bit_matrix::AsRowRef;
use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::container::traits::{Elements, RefElements};
use crate::container::word_iters::lincomb::scaled_iadd;
use crate::container::word_iters::term_set::AsView as TermSetAsView;
use crate::container::word_iters::term_set::AsViewMut;
use crate::fermion::operator::general::term_set::{
    self as general_term_set, TermSet as GeneralTermSet,
};
use crate::fermion::operator::normal::cmpnt::Cmpnt;
use crate::fermion::operator::normal::cmpnt_major::lincomb::mul as normal_mul;
use crate::fermion::operator::normal::cmpnt_major::term_set::{
    self as normal_term_set, TermSet as NormalTermSet,
};
use crate::fermion::traits::ModesBased;
use itertools::Itertools;
use num_complex::Complex64;
use std::collections::HashSet;

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
    use rstest::rstest;
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
    fn test_normolise_repeated_annihilation_vanishes() {
        // a_0 a_0 -> 0
        let raw = make_raw(4, &[0, 0], &[false, false]);
        let result = normalise(&raw.as_terms());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_normalise_already_normal_ordered_unchanged() {
        // a_^0+ a_1 already normal-ordered, so no anticommutations are needed and the
        // coefficient should remain unchanged.
        let raw = make_raw(4, &[0, 1], &[true, false]);
        let result = normalise(&raw.as_terms());
        assert_eq!(result.len(), 1);
        check_term(
            &result,
            4,
            HashSet::from([0]),
            HashSet::from([1]),
            Complex64::new(1.0, 0.0),
        );
    }

    #[test]
    fn test_normalise_combines_multiple_raw_terms() {
        // a_0 a_0^+ + a_0^+ a_0 -> 1
        let modes_space = Modes::from_count(4);
        let mut raw = GeneralTermSet::<f64>::new(2, modes_space.clone());
        raw.push_term(&[0, 0], &[false, true], 1.0);
        raw.push_term(&[0, 0], &[true, false], 1.0);
        let result = normalise(&raw.as_terms());
        assert_eq!(result.len(), 1);
        check_term(
            &result,
            4,
            HashSet::new(),
            HashSet::new(),
            Complex64::new(1.0, 0.0),
        );
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

    #[rstest]
    // note inquanto reorders modes in cre/ann substrings:
    // e.g. str(op_5ann.normal_ordered()) == "(1.0, F4  F3  F2  F1  F0 )"
    // op_5ann = FermionOperator([(1.0, "F0 F1 F2 F3 F4")])
    #[case(5, vec![(vec![0, 1, 2, 3, 4], vec![false, false, false, false, false], 1.0)],
        vec![(vec![], vec![0, 1, 2, 3, 4], 1.0)])]
    // op_5cre = FermionOperator([(1.0, "F0^ F1^ F2^ F3^ F4^")])
    #[case(5, vec![(vec![0, 1, 2, 3, 4], vec![true, true, true, true, true], 1.0)],
        vec![(vec![0, 1, 2, 3, 4], vec![], 1.0)])]
    // op = FermionOperator([(1.0, "F0 F1 F2 F3 F4^ F5^ F6^ F7^")])
    #[case(8, vec![(vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![false, false, false, false, true, true, true, true], 1.0)],
        vec![(vec![4, 5, 6, 7], vec![0, 1, 2, 3], 1.0)])]
    // cases below are from inquanto/tests/operators/test_fermion_operator.py
    // term = FermionOperatorString.from_string("F0 F0^ F1 F0")
    // ordered_fo = FermionOperator._normal_ordered_ladder_term(term, 0.1)
    // assert str(ordered_fo) == "(0.1, F1  F0 )"
    // note that after normal ordering in zixy the coeff is -0.1 and string is "F0 F1"
    #[case(2, vec![(vec![0, 0, 1, 0], vec![false, true, false, false], 0.1)],
        vec![(vec![], vec![1, 0], -0.1)])]
    // each pair of cases below should have same normal ordered form for Ai and Bi:
    // assert operatorAi.normal_ordered().approx_equal_to(operatorBi.normal_ordered())
    // operatorA1 = FermionOperator([(1.0, "F0^ F1 F0 F1^")])
    #[case(2, vec![(vec![0, 1, 0, 1], vec![true, false, false, true], 1.0)],
        vec![(vec![0], vec![0], -1.0), (vec![0, 1], vec![0, 1], -1.0)])]
    // operatorB1 = FermionOperator([(-1.0, "F0^ F0 F1 F1^")])
    #[case(2, vec![(vec![0, 0, 1, 1], vec![true, false, false, true], -1.0)],
        vec![(vec![0], vec![0], -1.0), (vec![0, 1], vec![0, 1], -1.0)])]
    // operatorA2 = FermionOperator([(1.0, "F1^ F2 F2^ F1")])
    #[case(3, vec![(vec![1, 2, 2, 1], vec![true, false, true, false], 1.0)],
        vec![(vec![1], vec![1], 1.0), (vec![1, 2], vec![1, 2], 1.0)])]
    // operatorB2 = FermionOperator([(1.0, "F1^ F1"), (1.0, "F1^ F2^ F1 F2")])
    #[case(3, vec![(vec![1, 1], vec![true, false], 1.0),
        (vec![1, 2, 1, 2], vec![true, true, false, false], 1.0)],
        vec![(vec![1], vec![1], 1.0), (vec![1, 2], vec![1, 2], 1.0)])]
    // operatorA3 = FermionOperator([(1.0, ""), (-1.0, "F1^ F1"), (-1.0, "F2^ F2")])
    #[case(3, vec![(vec![], vec![], 1.0), (vec![1, 1], vec![true, false], -1.0),
        (vec![2, 2], vec![true, false], -1.0)],
        vec![(vec![], vec![], 1.0), (vec![1], vec![1], -1.0), (vec![2], vec![2], -1.0)])]
    // operatorB3 = FermionOperator([(1.0, "F1 F1^ F2 F2^"), (-1.0, "F2^ F1^ F1 F2")])
    #[case(3, vec![(vec![1, 1, 2, 2], vec![false, true, false, true], 1.0),
        (vec![2, 1, 1, 2], vec![true, true, false, false], -1.0)],
        vec![(vec![], vec![], 1.0), (vec![1], vec![1], -1.0), (vec![2], vec![2], -1.0)])]
    fn test_normalise_against_inquanto(
        #[case] n_modes: usize,
        #[case] input_term: Vec<(Vec<usize>, Vec<bool>, f64)>,
        #[case] inquanto_results: Vec<(Vec<usize>, Vec<usize>, f64)>,
    ) {
        let modes_space_spin_minor = Modes::from_pair_count_spin_minor(n_modes);
        let mut raw_spin_minor = GeneralTermSet::<f64>::new(n_modes, modes_space_spin_minor);
        for (modes, adj, coeff) in input_term {
            raw_spin_minor.push_term(modes.as_slice(), adj.as_slice(), coeff);
        }
        let result_sm = normalise(&raw_spin_minor.as_terms());
        assert_eq!(result_sm.len(), inquanto_results.len());
        for inq_result in inquanto_results {
            let (vec_cre, vec_ann, inq_coeff) = inq_result;
            let inq_cre = HashSet::from_iter(vec_cre.into_iter());
            let inq_ann = HashSet::from_iter(vec_ann.into_iter());
            check_term(
                &result_sm,
                n_modes,
                inq_cre,
                inq_ann,
                Complex64::new(inq_coeff, 0.0),
            );
        }
    }
}
