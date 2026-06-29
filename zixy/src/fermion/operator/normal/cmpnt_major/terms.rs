//! Extends  fermion `CmpntList` fermion with associated coefficients.

use crate::container::coeffs::traits::NumRepr;
use crate::container::word_iters::terms;
use crate::fermion::mode::Modes;
use crate::fermion::operator::normal::cmpnt_list::CmpntList;
use crate::fermion::traits::ModesBased;

/// A Fermion `CmpntList` paired with one coefficient per component, plus
// immutable and mutable views into one.
pub type Terms<C /*: NumRepr*/> = terms::Terms<CmpntList, C>;
pub type View<'a, C /*: NumRepr*/> = terms::View<'a, CmpntList, C>;
pub type ViewMut<'a, C /*: NumRepr*/> = terms::ViewMut<'a, CmpntList, C>;

//Borrowed handles to a single fermion term inside a `Terms` container.
pub type TermRef<'a, C /*: NumRepr*/> = terms::TermRef<'a, CmpntList, C>;
pub type TermMutRef<'a, C /*: NumRepr*/> = terms::TermMutRef<'a, CmpntList, C>;

/// Trait for structs that immutably view a fermion [`Terms`].
/// Currently unused, but may be useful for future extensions.
pub trait AsView<C: NumRepr>: terms::AsView<CmpntList, C> {}
pub trait AsViewMut<C: NumRepr>: terms::AsViewMut<CmpntList, C> {}

impl<C: NumRepr> AsView<C> for Terms<C> {}
impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}

impl<C: NumRepr> AsViewMut<C> for Terms<C> {}
impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}

impl<C: NumRepr> Terms<C> {
    /// Create a new  (empty) list of fermion terms on the given mode space.
    pub fn new(modes: Modes) -> Self {
        use crate::container::traits::EmptyFrom;
        Self::empty_from(&CmpntList::new(modes))
    }
}

impl<C: NumRepr> ModesBased for Terms<C> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
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

impl<'a, C: NumRepr> ModesBased for TermRef<'a, C> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
    }
}

impl<'a, C: NumRepr> ModesBased for TermMutRef<'a, C> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::coeffs::unity::Unity;
    use crate::container::traits::Elements;
    use num_complex::Complex64;

    #[test]
    fn test_empty() {
        let list_unity = Terms::<Unity>::new(Modes::from_count(3));
        let list_real = Terms::<f64>::new(Modes::from_count(3));
        let list_complex = Terms::<Complex64>::new(Modes::from_count(3));
        assert!(list_unity.is_empty());
        assert!(list_real.is_empty());
        assert!(list_complex.is_empty());
    }
}
