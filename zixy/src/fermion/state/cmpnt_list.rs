//! Defines gapless, contiguous lists of Slater determinants.
//! The value of the bit at position i indicates a fermion occupying the corresponding mode in the string.
//!
//! Assumes a "little-endian" convention whereby the occupation flag the first
//! mode is stored in the least significant bit of the first u64.

use std::fmt::Display;
use std::hash::Hash;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::cmpnt::bitset_cmpnt_list::{self, AsCmpntList, AsCmpntMutRef, AsCmpntRef};
use crate::cmpnt::mode::ModeOutOfBounds;
use crate::cmpnt::springs::ModeSettings;
use crate::cmpnt::state_springs::BinarySprings;
use crate::container::table::Table;
use crate::container::traits::{Compatible, Elements, EmptyClone};
use crate::container::u64it_elems::{self, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::traits::ModesBased;

/// Contiguous and compact storage for Slater determinants
#[derive(Debug, Hash, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct CmpntList {
    /// Raw storage table for the Slater determinants as bitsets.
    bitsets: bitset_cmpnt_list::CmpntList,
    /// Space of fermionic modes on which the Slater determinants are defined.
    modes: Modes,
}

impl CmpntList {
    /// Create an empty `CmpntList` on the modes given.
    pub fn new(modes: Modes) -> Self {
        Self {
            bitsets: bitset_cmpnt_list::CmpntList::new(modes.n_mode()),
            modes,
        }
    }

    /// Create a new instance from given `BinarySprings`.
    pub fn from_springs(modes: Modes, springs: &BinarySprings) -> Result<Self, ModeOutOfBounds> {
        let mut this = Self::new(modes);
        this.set_from_springs(springs)?;
        Ok(this)
    }

    /// Create an instance from springs with inferred modes.
    pub fn from_springs_default(springs: &BinarySprings) -> Self {
        Self::from_springs(
            Modes::from_count(springs.get_mode_inds().default_n_mode()),
            springs,
        )
        .unwrap()
    }
}

impl AsCmpntList for CmpntList {
    fn get_bitsets(&self) -> &Table {
        self.bitsets.get_bitsets()
    }

    fn get_bitsets_mut(&mut self) -> &mut Table {
        self.bitsets.get_bitsets_mut()
    }

    fn n_mode(&self) -> usize {
        self.modes.n_mode()
    }
}

impl Compatible for CmpntList {
    fn compatible_with(&self, other: &Self) -> bool {
        self.modes == other.modes
    }
}

impl Elements for CmpntList {
    fn len(&self) -> usize {
        self.bitsets.len()
    }
}

impl ModesBased for CmpntList {
    fn modes(&self) -> &Modes {
        &self.modes
    }
}

impl EmptyClone for CmpntList {
    fn empty_clone(&self) -> Self {
        Self {
            bitsets: self.bitsets.empty_clone(),
            modes: self.modes.clone(),
        }
    }
}

impl WordIters for CmpntList {
    fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone {
        self.bitsets.elem_u64it(i)
    }

    fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
        self.bitsets.elem_u64it_mut(i)
    }

    fn u64it_size(&self) -> usize {
        self.bitsets.u64it_size()
    }

    fn pop_and_swap(&mut self, i_row: usize) {
        self.bitsets.pop_and_swap(i_row);
    }

    fn fmt_elem(&self, i: usize) -> String {
        self.bitsets.fmt_elem(i)
    }

    fn resize(&mut self, n: usize) {
        self.bitsets.resize(n);
    }
}

impl Display for CmpntList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            (0..self.len()).map(|i| self.fmt_elem(i)).join(", ")
        )
    }
}

pub type CmpntRef<'a> = u64it_elems::ElemRef<'a, CmpntList>;
pub type CmpntMutRef<'a> = u64it_elems::ElemMutRef<'a, CmpntList>;

impl<'a> ModesBased for CmpntRef<'a> {
    fn modes(&self) -> &Modes {
        &self.get_word_iters().modes
    }
}

impl<'a> ModesBased for CmpntMutRef<'a> {
    fn modes(&self) -> &Modes {
        &self.get_word_iters().modes
    }
}

impl<'a> AsCmpntRef for CmpntRef<'a> {
    fn cmpnt_list(&self) -> &bitset_cmpnt_list::CmpntList {
        &self.get_word_iters().bitsets
    }
}

impl<'a> AsCmpntMutRef for CmpntMutRef<'a> {
    fn cmpnt_list(&self) -> &bitset_cmpnt_list::CmpntList {
        &self.get_word_iters().bitsets
    }

    fn cmpnt_list_mut(&mut self) -> &mut bitset_cmpnt_list::CmpntList {
        &mut self.get_word_iters_mut().bitsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        {
            let v = CmpntList::new(Modes::from_count(4));
            assert!(v.bitsets.is_empty());
        }
    }
}
