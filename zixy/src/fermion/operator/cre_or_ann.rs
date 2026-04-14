//! Defines gapless, contiguous lists of fermionic creation or annihilation ladder operator strings.
//! The value of the bit at position i flags the presence of the corresponding ladder operator in the string.
//! A set bit at position i indicates the ladder operator is present in the string.
//!
//! Assumes a "little-endian" convention whereby the presence flag of the ladder operator acting on the first
//! mode is stored in the least significant bit of the first u64.

use std::fmt::Display;
use std::hash::Hash;

use itertools::{izip, Itertools};
use serde::{Deserialize, Serialize};

use crate::cmpnt::springs::ModeSettings;
use crate::cmpnt::state_springs::BinarySprings;
use crate::container::bit_matrix::{AsBitMatrix, AsRowMutRef, AsRowRef, BitMatrix};
use crate::container::coeffs::sign::{Sign, SignVec};
use crate::container::coeffs::traits::NumReprVec;
use crate::container::errors::OutOfBounds;
use crate::container::table::Table;
use crate::container::traits::{Compatible, Elements, EmptyClone, RefElements};
use crate::container::word_iters::{self, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::traits::ModesBased;

/// Contiguous and compact storage for fermionic creation or annihilation ladder operator string on a given space of modes.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct CmpntList {
    /// Raw storage table for the fermionic ladder operators as bitsets.
    bitsets: BitMatrix,
    /// Space of fermionic modes on which the ladder operators are defined.
    modes: Modes,
    /// Flag indicating whether this represents creation operator strings.
    is_cre: bool,
}

impl CmpntList {
    /// Create an empty `CmpntList` on the `Qubits` given.
    pub fn new(modes: Modes, is_cre: bool) -> Self {
        Self {
            bitsets: BitMatrix::new(modes.len()),
            modes,
            is_cre,
        }
    }

    /// Create a new instance from given `BinarySprings`.
    pub fn from_springs(
        modes: Modes,
        is_cre: bool,
        springs: &BinarySprings,
    ) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(modes, is_cre);
        this.set_from_springs(springs)?;
        Ok(this)
    }

    /// Create an instance from springs assuming a default space of modes.
    pub fn from_springs_default(is_cre: bool, springs: &BinarySprings) -> Self {
        Self::from_springs(
            Modes::from_count(springs.get_mode_inds().default_n_mode() as usize),
            is_cre,
            springs,
        )
        .unwrap()
    }

    /// Conjugate without storing the sign of the exchange.
    pub fn dagger_ignore_signs(&mut self) {
        self.is_cre ^= true;
    }

    /// Conjugate every component, and store the associated signs.
    pub fn dagger(&mut self, signs: &mut SignVec) {
        signs.resize(self.len());
        for (i, elem) in self.iter().enumerate() {
            signs.imul_elem_unchecked(i, elem.sign_of_dagger())
        }
        self.dagger_ignore_signs();
    }

    /// Get the antisymmetric phase signs associated with taking the hermitian conjugate of the entirety of `self`.
    pub fn dagger_get_signs(&mut self) -> SignVec {
        let mut out = SignVec::default();
        self.dagger(&mut out);
        out
    }
}

impl AsBitMatrix for CmpntList {
    fn get_table(&self) -> &Table {
        self.bitsets.get_table()
    }

    fn get_table_mut(&mut self) -> &mut Table {
        self.bitsets.get_table_mut()
    }

    fn n_bit(&self) -> usize {
        self.bitsets.n_bit()
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
            is_cre: self.is_cre,
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
        (0..self.modes.len())
            .map(|i_mode| self.get_bit_unchecked(i, i_mode))
            .enumerate()
            .filter_map(|(i_mode, present)| if present { Some(i_mode) } else { None })
            .map(|i| format!("F{}{}", i, if self.is_cre { "^" } else { "" }))
            .join(" ")
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

impl<'a> AsRowRef for CmpntRef<'a> {
    fn bit_mat(&self) -> &impl AsBitMatrix {
        self.word_iters
    }
}

impl<'a> AsRowMutRef for CmpntMutRef<'a> {
    fn bit_mat(&self) -> &impl AsBitMatrix {
        self.word_iters
    }

    fn bit_mat_mut(&mut self) -> &mut impl AsBitMatrix {
        self.word_iters
    }
}

impl<'a> CmpntRef<'a> {
    /// Conjugation involves a triangular number of antisymmetric exchanges.
    /// n
    /// 2: ab -> ba (1)
    /// 3: abc -> acb -> cab -> cba (3)
    /// 4: abcd (3 + 3 = 6)
    /// 5: abcde (6 + 4 = 10)
    /// The number of exchanges is n * (n-1) / 2
    /// If this is odd, the `Sign` is -1, else +1
    pub fn sign_of_dagger(&self) -> Sign {
        let n = self.hamming_weight();
        Sign((n * (n.saturating_sub(1))) & 2 == 2)
    }

    /// Compute the number of set bits in the bitwise AND of the viewed bitset with another, or its negation.
    pub fn pair_hamming_weight(&self, other: &CmpntRef, negate: bool) -> usize {
        let func = if negate {
            |(a, b): (u64, u64)| (a & !b).count_ones()
        } else {
            |(a, b): (u64, u64)| (a & b).count_ones()
        };
        self.get_u64it()
            .zip(other.get_u64it())
            .map(func)
            .sum::<u32>() as usize
    }

    /// Compute the number of set bits in the bitwise AND of the viewed bitset with another.
    pub fn intersection_hamming_weight(&self, other: &CmpntRef) -> usize {
        self.pair_hamming_weight(other, false)
    }

    /// Compute the number of set bits in the bitwise AND of the viewed bitset with the negation of another.
    pub fn difference_hamming_weight(&self, other: &CmpntRef) -> usize {
        self.pair_hamming_weight(other, true)
    }

    /// Compute the number of set bits in the bitwise AND of the viewed bitset with two others, with given negations.
    pub fn triple_hamming_weight(
        &self,
        first: &CmpntRef,
        negate_first: bool,
        second: &CmpntRef,
        negate_second: bool,
    ) -> usize {
        let func = if negate_first {
            if negate_second {
                |(a, b, c): (u64, u64, u64)| (a & !b & !c).count_ones()
            } else {
                |(a, b, c): (u64, u64, u64)| (a & !b & c).count_ones()
            }
        } else if negate_second {
            |(a, b, c): (u64, u64, u64)| (a & b & !c).count_ones()
        } else {
            |(a, b, c): (u64, u64, u64)| (a & b & c).count_ones()
        };
        izip!(self.get_u64it(), first.get_u64it(), second.get_u64it())
            .map(func)
            .sum::<u32>() as usize
    }
}

#[cfg(test)]
mod tests {
    use crate::container::traits::MutRefElements;

    use super::*;

    #[test]
    fn test_empty() {
        {
            let v = CmpntList::new(Modes::from_count(4), true);
            assert!(v.bitsets.is_empty());
        }
    }

    #[test]
    fn test_to_string() {
        {
            let mut v = CmpntList::new(Modes::from_count(10), true);
            v.push_clear();
            v.get_elem_mut_ref(0).assign_set_unchecked([].into());
            assert_eq!(v.to_string(), "[]");
            v.get_elem_mut_ref(0).assign_set_unchecked([3, 6, 7].into());
            assert_eq!(v.to_string(), "[F3^ F6^ F7^]");
            let mut v = CmpntList::new(Modes::from_count(10), false);
            v.push_clear();
            v.get_elem_mut_ref(0).assign_set_unchecked([].into());
            assert_eq!(v.to_string(), "[]");
            v.get_elem_mut_ref(0).assign_set_unchecked([3, 6, 7].into());
            assert_eq!(v.to_string(), "[F3 F6 F7]");
        }
    }
}
