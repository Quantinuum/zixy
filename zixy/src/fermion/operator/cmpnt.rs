//! Definitions for a single normal ordered product of creation and annihilation operators.

use std::collections::HashSet;
use std::fmt::Display;

use crate::container::bit_matrix::AsRowMutRef;
use crate::container::errors::OutOfBounds;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::container::word_iters::{Elem, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::operator::cmpnt_list::{CmpntList, CmpntRef};
use crate::fermion::traits::ModesBased;

/// A single normal-ordered pair of fermionic creation and annihilation operator strings.
impl Elem<CmpntList> {
    /// Create a new instance.
    pub fn new(modes: Modes) -> Self {
        let mut this = Self(CmpntList::new(modes));
        this.0.push_clear();
        this
    }

    /// Create an instance from creation and annihilation vectors, returning out of bounds if one or both are too long for `modes`.
    pub fn from_vecs(modes: Modes, cre: Vec<bool>, ann: Vec<bool>) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(modes);
        this.borrow_mut().get_cre_part().assign_vec(cre)?;
        this.borrow_mut().get_ann_part().assign_vec(ann)?;
        Ok(this)
    }

    /// Create an instance from creation and annihilation vectors ignoring bounds checking.
    pub fn from_vecs_unchecked(modes: Modes, cre: Vec<bool>, ann: Vec<bool>) -> Self {
        Self::from_vecs(modes, cre, ann).unwrap()
    }

    /// Create an instance from creation and annihilation sets, returning out of bounds if any mode index is beyond the max index of `modes`.
    pub fn from_sets(
        modes: Modes,
        cre: HashSet<usize>,
        ann: HashSet<usize>,
    ) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(modes);
        this.borrow_mut().get_cre_part().assign_set(cre)?;
        this.borrow_mut().get_ann_part().assign_set(ann)?;
        Ok(this)
    }

    /// Create an instance from creation and annihilation sets, ignoring bounds checking.
    pub fn from_sets_unchecked(modes: Modes, cre: HashSet<usize>, ann: HashSet<usize>) -> Self {
        Self::from_sets(modes, cre, ann).unwrap()
    }

    /// Create an instance from creation and annihilation vectors, assuming the smallest mode count.
    pub fn from_vecs_default(cre: Vec<bool>, ann: Vec<bool>) -> Self {
        let modes = Modes::from_count(cre.len().max(ann.len()));
        Self::from_vecs(modes, cre, ann).unwrap()
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

impl Display for Elem<CmpntList> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

/// Type alias for a single element of the CmpntList.
pub type Cmpnt = Elem<CmpntList>;
