//! Defines gapless, contiguous lists of Slater determinants.
//! The value of the bit at position i indicates a fermion occupying the corresponding mode in the string.
//!
//! Assumes a "little-endian" convention whereby the occupation flag the first
//! mode is stored in the least significant bit of the first u64.

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::container::bit_matrix;
use crate::container::traits::{Compatible, Elements, EmptyClone};
use crate::container::word_iters::{self, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::traits::ModesBased;

/// Contiguous and compact storage for Slater determinants
#[derive(Debug, Hash, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct CmpntList {
    /// Raw storage table for the Slater determinants as bitsets.
    bitsets: bit_matrix::BitMatrix,
    /// Space of fermionic modes on which the Slater determinants are defined.
    modes: Modes,
}

impl CmpntList {
    /// Create an empty `CmpntList` on the modes given.
    pub fn new(modes: Modes) -> Self {
        Self {
            bitsets: bit_matrix::BitMatrix::new(modes.len()),
            modes,
        }
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
        Self::new(self.modes.clone())
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

pub type CmpntRef<'a> = word_iters::ElemRef<'a, CmpntList>;
pub type CmpntMutRef<'a> = word_iters::ElemMutRef<'a, CmpntList>;

impl<'a> ModesBased for CmpntRef<'a> {
    fn modes(&self) -> &Modes {
        &self.word_iters.modes
    }
}

impl<'a> ModesBased for CmpntMutRef<'a> {
    fn modes(&self) -> &Modes {
        &self.word_iters.modes
    }
}

impl<'a> bit_matrix::AsRowRef for CmpntRef<'a> {
    fn bit_mat(&self) -> &impl bit_matrix::AsBitMatrix {
        &self.word_iters.bitsets
    }
}

impl<'a> bit_matrix::AsRowMutRef for CmpntMutRef<'a> {
    fn bit_mat(&self) -> &impl bit_matrix::AsBitMatrix {
        &self.word_iters.bitsets
    }

    fn bit_mat_mut(&mut self) -> &mut impl bit_matrix::AsBitMatrix {
        &mut self.word_iters.bitsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        {
            let v = CmpntList::new(Modes::from_count(4));
            assert_eq!(v.len(), 0);
            assert!(v.is_empty());
        }
    }
}
