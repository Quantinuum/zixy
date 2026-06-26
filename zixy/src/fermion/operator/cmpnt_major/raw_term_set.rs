//! Stores raw (non-normal-ordered) fermion terms.

use crate::container::coeffs::traits::{NumRepr, NumReprVec};
use crate::container::map::Map;
use crate::container::traits::Elements;
use crate::container::word_iters;
use crate::container::word_iters::terms;
use crate::container::word_iters::WordIters;
use crate::fermion::mode::Modes;
use crate::fermion::operator::raw_cmpnt_list::RawCmpntList;
use crate::fermion::traits::ModesBased;

pub type RawTerms<C> = terms::Terms<RawCmpntList, C>;
pub type RawTermSet<C> = word_iters::term_set::TermSet<RawCmpntList, C>;
pub type View<'a, C> = word_iters::term_set::View<'a, RawCmpntList, C>;
pub type ViewMut<'a, C> = word_iters::term_set::ViewMut<'a, RawCmpntList, C>;

pub trait AsView<C: NumRepr>: word_iters::term_set::AsView<RawCmpntList, C> {}
pub trait AsViewMut<C: NumRepr>: word_iters::term_set::AsViewMut<RawCmpntList, C> {}

impl<C: NumRepr> AsView<C> for RawTermSet<C> {}
impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}
impl<'a, C: NumRepr> AsView<C> for ViewMut<'a, C> {}
impl<C: NumRepr> AsViewMut<C> for RawTermSet<C> {}
impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}

impl<C: NumRepr> RawTermSet<C> {
    /// Create a new instance.
    pub fn new(max_len: usize, modes: Modes) -> Self {
        Self {
            terms: RawTerms::new(max_len, modes),
            map: Map::default(),
        }
    }
    pub fn push_term(&mut self, modes: &[usize], adj: &[bool], coeff: C) {
        let i = self.terms.word_iters.len();
        self.terms.word_iters.push(modes, adj);
        let k = self.terms.word_iters.hash_at_index(i);
        self.map.insert(k, i);
        self.terms.coeffs.push(coeff);
    }
}

impl<C: NumRepr> ModesBased for RawTermSet<C> {
    fn modes(&self) -> &Modes {
        self.terms.word_iters.modes()
    }
}

impl<C: NumRepr> RawTerms<C> {
    pub fn new(max_len: usize, modes: Modes) -> Self {
        use crate::container::traits::EmptyFrom;
        Self::empty_from(&RawCmpntList::new(max_len, modes))
    }
}

impl<'a, C: NumRepr> ModesBased for View<'a, C> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
    }
}

impl<C: NumRepr> RawTermSet<C> {
    pub fn as_raw_terms(&self) -> View<'_, C> {
        use crate::container::traits::proj::Borrow;
        self.borrow()
    }
}
