//! Stores raw (non-normal-ordered) fermion terms.

use crate::container::coeffs::traits::{NumRepr, NumReprVec};
use crate::container::map::Map;
use crate::container::traits::Elements;
use crate::container::word_iters;
use crate::container::word_iters::terms;
use crate::container::word_iters::WordIters;
use crate::fermion::mode::Modes;
use crate::fermion::operator::general::cmpnt_list::CmpntList;
use crate::fermion::traits::ModesBased;

pub type Terms<C> = terms::Terms<CmpntList, C>;
pub type TermSet<C> = word_iters::term_set::TermSet<CmpntList, C>;
pub type View<'a, C> = word_iters::term_set::View<'a, CmpntList, C>;
pub type ViewMut<'a, C> = word_iters::term_set::ViewMut<'a, CmpntList, C>;

pub trait AsView<C: NumRepr>: word_iters::term_set::AsView<CmpntList, C> {}
pub trait AsViewMut<C: NumRepr>: word_iters::term_set::AsViewMut<CmpntList, C> {}

impl<C: NumRepr> AsView<C> for TermSet<C> {}
impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}
impl<'a, C: NumRepr> AsView<C> for ViewMut<'a, C> {}
impl<C: NumRepr> AsViewMut<C> for TermSet<C> {}
impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}

impl<C: NumRepr> TermSet<C> {
    /// Create a new instance.
    pub fn new(max_len: usize, modes: Modes) -> Self {
        Self {
            terms: Terms::new(max_len, modes),
            map: Map::default(),
        }
    }
    pub fn push_term(&mut self, modes: &[usize], adj: &[bool], coeff: C) {
        let old_capacity = self.terms.word_iters.max_len;
        let i = self.terms.word_iters.len();
        self.terms.word_iters.push(modes, adj);
        if self.terms.word_iters.max_len != old_capacity {
            self.map.populate_from(&self.terms.word_iters);
        } else {
            let k = self.terms.word_iters.hash_at_index(i);
            self.map.insert(k, i);
        }
        self.terms.coeffs.push(coeff);
    }

    pub fn push_concat_term(
        &mut self,
        lhs_modes: &[usize],
        lhs_adj: &[bool],
        rhs_modes: &[usize],
        rhs_adj: &[bool],
        coeff: C,
    ) {
        let old_capacity = self.terms.word_iters.max_len;
        let i = self.terms.word_iters.len();
        self.terms
            .word_iters
            .push_concat(lhs_modes, lhs_adj, rhs_modes, rhs_adj);
        if self.terms.word_iters.max_len != old_capacity {
            self.map.populate_from(&self.terms.word_iters);
        } else {
            let k = self.terms.word_iters.hash_at_index(i);
            self.map.insert(k, i);
        }
        self.terms.coeffs.push(coeff);
    }
}

impl<C: NumRepr> ModesBased for TermSet<C> {
    fn modes(&self) -> &Modes {
        self.terms.word_iters.modes()
    }
}

impl<C: NumRepr> Terms<C> {
    pub fn new(max_len: usize, modes: Modes) -> Self {
        use crate::container::traits::EmptyFrom;
        Self::empty_from(&CmpntList::new(max_len, modes))
    }
}

impl<'a, C: NumRepr> ModesBased for View<'a, C> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
    }
}

impl<C: NumRepr> TermSet<C> {
    pub fn as_terms(&self) -> View<'_, C> {
        use crate::container::traits::proj::Borrow;
        self.borrow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::word_iters::set::{AsView, View as SetView};

    #[test]
    fn growth_rebuilds_lookup_keys() {
        let mut terms = TermSet::new(0, Modes::from_count(3));
        terms.push_term(&[], &[], 1.0);
        terms.push_term(&[0], &[false], 2.0);
        terms.push_concat_term(&[1; 64], &[true; 64], &[2], &[false], 3.0);
        assert_eq!(terms.terms.word_iters.max_len, 65);
        assert_eq!(terms.terms.coeffs, vec![1.0, 2.0, 3.0]);
        SetView {
            word_iters: &terms.terms.word_iters,
            map: &terms.map,
        }
        .consistency_check();
    }
}
