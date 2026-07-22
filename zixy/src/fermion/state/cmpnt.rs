//! Definitions for a single Slater determinant.

use std::collections::HashSet;
use std::fmt;

use crate::container::bit_matrix::AsRowMutRef;
use crate::container::errors::OutOfBounds;
use crate::container::word_iters::{Elem, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::state::cmpnt_list::{CmpntList, CmpntMutRef, CmpntRef};
use crate::fermion::traits::ModesBased;

/// A single Slater determinant.
/// All functionality which does not pertain to the instantiation of `Elem<CmpntList>` is to be accessed via the
/// borrow and borrow_mut functions.
impl Elem<CmpntList> {
    /// Borrow an immutable projected view.
    pub fn borrow(&self) -> CmpntRef<'_> {
        CmpntRef {
            word_iters: &self.0,
            index: 0,
        }
    }

    /// Borrow a mutable projected view.
    pub fn borrow_mut(&mut self) -> CmpntMutRef<'_> {
        CmpntMutRef {
            word_iters: &mut self.0,
            index: 0,
        }
    }

    /// Create a new instance.
    pub fn new(modes: Modes) -> Self {
        let mut this = Self(CmpntList::new(modes));
        this.0.push_clear();
        this
    }

    /// Create an instance from a set of occupied mode indices.
    pub fn from_set(modes: Modes, occupied: HashSet<usize>) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(modes);
        this.borrow_mut().assign_set(occupied)?;
        Ok(this)
    }

    /// Create an instance from a reference to a component.
    pub fn from_cmpnt_ref(cmpnt_ref: CmpntRef) -> Self {
        let mut this = Self::new(cmpnt_ref.to_modes());
        this.borrow_mut().assign(cmpnt_ref);
        this
    }
}

impl ModesBased for Elem<CmpntList> {
    fn modes(&self) -> &Modes {
        self.0.modes()
    }
}

impl fmt::Display for Elem<CmpntList> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.fmt_elem(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::bit_matrix::AsRowRef;
    use std::collections::HashSet;

    #[test]
    fn test_from_set() {
        let state =
            Elem::<CmpntList>::from_set(Modes::from_count(4), HashSet::from([1, 3])).unwrap();
        assert_eq!(state.borrow().to_set(), HashSet::from([1, 3]));
    }

    #[test]
    fn test_from_cmpnt_ref() {
        let cmpnt =
            Elem::<CmpntList>::from_set(Modes::from_count(4), HashSet::from([1, 3])).unwrap();
        let state = Elem::<CmpntList>::from_cmpnt_ref(cmpnt.borrow());
        assert_eq!(state.borrow().to_set(), HashSet::from([1, 3]));
    }
}
