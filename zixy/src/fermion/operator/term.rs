//! Single-term fermion utilities.

use std::fmt::Display;

use crate::container::coeffs::traits::NumRepr;
use crate::container::traits::proj::Borrow;
use crate::container::word_iters::terms;
use crate::container::word_iters::terms::AsViewMut;
use crate::fermion::mode::Modes;
use crate::fermion::operator::cmpnt_list::CmpntList;
use crate::fermion::operator::terms::Terms;
use crate::fermion::traits::ModesBased;

/// A single fermion operator with a generically-typed coefficient.
pub type Term<C /*: NumRepr*/> = terms::Term<CmpntList, C>;

impl<C: NumRepr> Term<C> {
    /// Create a single fermion term with unit coefficient on the given mode space.
    pub fn new(modes: Modes) -> Self {
        let mut terms = Terms::<C>::new(modes);
        terms.push_clear();
        terms.coeffs.set_unchecked(0, C::one()); // Ensure unit coefficient.
        Self {
            word_iters: terms.word_iters,
            coeffs: terms.coeffs,
        }
    }
}

impl<C: NumRepr> Display for Term<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

impl<C: NumRepr> ModesBased for Term<C> {
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
    fn test_new_has_one_term() {
        let term = Term::<Unity>::new(Modes::from_count(4));
        assert_eq!(term.coeffs.len(), 1);
    }
}
