//! Stores fermion terms with the help of a `Map` to ensure each string appears at most once.

use crate::container::coeffs::traits::NumRepr;
use crate::container::map::Map;
use crate::container::word_iters;
use crate::fermion::mode::Modes;
use crate::fermion::operator::cmpnt_list::CmpntList;
use crate::fermion::operator::term::Terms;
use crate::fermion::traits::ModesBased

pub type TermSet<C /*: NumRepr*/> = word_iters::term_set::TermSet<CmpntList, C>;
pub type View<'a, C /*: NumRepr*/> = word_iters::term_set::View<'a, CmpntList, C>;
pub type ViewMut<'a, C /*: NumRepr*/> = word_iters::term_set::ViewMut<'a, CmpntList, C>;

/// Trait for structs that immutably view a [`TermSet`].
pub trait AsView<C: NumRepr>: word_iters::term_set::AsView<CmpntList, C> {}

/// Trait for structs that mutably view a [`TermSet`].
pub trait AsViewMut<C: NumRepr>: word_iters::term_set::AsViewMut<CmpntList, C> {}

impl<C: NumRepr> AsView<C> for TermSet<C> {}
impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}
impl<'a, C: NumRepr> AsView<C> for ViewMut<'a, C> {}

impl<C: NumRepr> AsViewMut<C> for TermSet<C> {}
impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}

impl<C: NumRepr> TermSet<C> {
    /// Create a new instance.
    pub fn new(modes: Modes) -> Self {
        Self {
            terms: Terms::new(modes),
            map: Map::default(),
        }
    }
}

impl<C: NumRepr> ModesBased for TermSet<C> {
    fn modes(&self) -> &Modes {
        self.terms.word_iters.modes()
    }
}

impl<'a, C: NumRepr> ModesBased for View<'a, C> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
    }
}

impl<'a, C: NumRepr> ModesBased for ViewMut<'a, C> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::coeffs::unity::Unity;
    use crate::container::traits::Elements;

    #[test]
    fn test_empty() {
        let set = TermSet::<Unity>::new(Modes::from_count(4));
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }
}
