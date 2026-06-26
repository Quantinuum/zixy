//! Factory functions for fermion number operators.

use std::collections::HashSet;

use crate::container::coeffs::traits::FieldElem;
use crate::container::errors::OutOfBounds;
use crate::container::traits::proj::Borrow;
use crate::container::traits::Elements;
use crate::container::word_iters::term_set::AsViewMut;
use crate::container::word_iters::terms::AsViewMut as TermsAsViewMut;
use crate::fermion::mode::Modes;
use crate::fermion::operator::cmpnt::Cmpnt;
use crate::fermion::operator::cmpnt_major::term_set::TermSet;

/// Create a number operator over the given set of mode indices.
pub fn num_op_from_inds<C: FieldElem>(
    modes: Modes,
    inds: HashSet<usize>,
) -> Result<TermSet<C>, OutOfBounds> {
    let mut out = TermSet::<C>::new(modes.clone());
    for i in inds.into_iter() {
        let cmpnt = Cmpnt::from_sets(modes.clone(), HashSet::from([i]), HashSet::from([i]))?;
        out.as_terms_mut().push_elem_coeff(cmpnt.borrow(), C::ONE);
    }

    Ok(out)
}

/// Create the full number operator over all modes.
pub fn num_op<C: FieldElem>(modes: Modes) -> TermSet<C> {
    num_op_from_inds::<C>(modes.clone(), (0..modes.len()).collect())
        .expect("num_op indices are always in bounds")
}

/// Returns a list of number operators, one for each mode.
/// The i-th operator counts the number of particles in mode i.
pub fn orbital_num_ops<C: FieldElem>(modes: Modes) -> Vec<Result<TermSet<C>, OutOfBounds>> {
    let n_modes = modes.len();
    (0..n_modes)
        .map(|i| num_op_from_inds(modes.clone(), [i].into()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::traits::{Elements, RefElements};
    use crate::container::word_iters::term_set::AsView;
    use crate::fermion::mode::Modes;
    use crate::fermion::operator::cmpnt::Cmpnt;
    use std::collections::HashSet;

    #[test]
    fn test_num_op_from_inds() {
        let modes = Modes::from_count(4);
        let inds = HashSet::from([1, 3]);
        let num_op = num_op_from_inds::<f64>(modes.clone(), inds).unwrap();
        assert_eq!(num_op.as_terms().len(), 2);
        let expected_cmpnts = vec![
            Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([1]), HashSet::from([1])),
            Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([3]), HashSet::from([3])),
        ];
        for cmpnt in expected_cmpnts {
            assert!(num_op
                .as_terms()
                .iter()
                .any(|term_ref| { term_ref.get_word_iter_ref() == cmpnt.borrow() }));
        }
    }
}
