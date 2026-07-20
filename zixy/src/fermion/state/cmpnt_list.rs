//! Defines gapless, contiguous lists of Slater determinants.
//! The value of the bit at position i indicates a fermion occupying the corresponding mode in the string.
//!
//! Assumes a "little-endian" convention whereby the occupation flag the first
//! mode is stored in the least significant bit of the first u64.

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

use crate::cmpnt::springs::ModeSettings;
use crate::cmpnt::state_springs::BinarySprings;
use crate::container::bit_matrix;
use crate::container::bit_matrix::AsBitMatrix;
use crate::container::coeffs::sign::Sign;
use crate::container::errors::OutOfBounds;
use crate::container::table::Table;
use crate::container::traits::{Compatible, Elements, EmptyClone};
use crate::container::word_iters::{self, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::operator::cmpnt_list as operator;
use crate::fermion::operator::products::apply_op_ket_u64;
use crate::fermion::operator::products::ApplyResult;
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

    /// Create a new instance from given `BinarySprings`.
    pub fn from_springs(modes: Modes, springs: &BinarySprings) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(modes);
        this.set_from_springs(springs)?;
        Ok(this)
    }

    /// Create an instance from springs with an inferred fermionic mode space.
    pub fn from_springs_default(springs: &BinarySprings) -> Self {
        Self::from_springs(
            Modes::from_count(springs.get_mode_inds().default_n_mode() as usize),
            springs,
        )
        .unwrap()
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

impl<'a> CmpntMutRef<'a> {
    /// Set the value of this referenced Slater determinant to the result of applying the given normal-ordered
    /// ladder operator component to `rhs`, returning the anti-symmetric exchange sign of the result, or `None`
    /// if the fermionic exclusion principle forbids the term (an annihilated mode is unoccupied in `rhs`, or a
    /// created mode is already occupied).
    /// Assumes the mode space fits in a single `u64` word.
    pub fn assign_mul_by_op(&mut self, op: operator::CmpntRef, rhs: CmpntRef) -> Option<Sign> {
        let cre = op.get_cre_part().get_u64it().next().unwrap_or(0);
        let ann = op.get_ann_part().get_u64it().next().unwrap_or(0);
        let ket = rhs.get_u64it().next().unwrap_or(0);
        let (result, sign) = match apply_op_ket_u64(cre, ann, ket) {
            ApplyResult::Applied(bits, sign) => (bits, sign),
            ApplyResult::Zero => return None,
        };
        *self.get_u64it_mut().next().unwrap() = result;
        Some(sign)
    }
}

impl bit_matrix::AsBitMatrix for CmpntList {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::bit_matrix::AsRowRef;
    use crate::container::traits::RefElements;
    use rstest::rstest;

    #[test]
    fn test_empty() {
        {
            let v = CmpntList::new(Modes::from_count(4));
            assert_eq!(v.len(), 0);
            assert!(v.is_empty());
        }
    }

    #[rstest]
    #[case("[], [0, 0, 0, 1]", 
        vec![vec![0, 0, 0, 0], vec![0, 0, 0, 1]])]
    #[case("[], [0, 0, 0, 1], []",
        vec![vec![0, 0, 0, 0], vec![0, 0, 0, 1], vec![0, 0, 0, 0]])]
    #[case(
        "[0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0]",
        vec![vec![0, 1, 0, 1, 1, 0], vec![1, 0, 1, 1, 0, 0]]
    )]
    fn test_from_springs(#[case] input: &str, #[case] output: Vec<Vec<i32>>) {
        let springs = BinarySprings::from_str(input);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        let cmpnts = CmpntList::from_springs_default(&springs);
        let vecs = (0..cmpnts.len())
            .map(|i| {
                cmpnts
                    .get_elem_ref(i)
                    .iter()
                    .map(|x| if x { 1 } else { 0 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(output, vecs);
    }

    #[test]
    fn test_assign_mul_by_op() {
        use crate::container::bit_matrix::AsRowMutRef;
        use crate::container::traits::proj::Borrow;
        use crate::container::traits::MutRefElements;
        use crate::fermion::operator::cmpnt::Cmpnt;
        use std::collections::HashSet;

        let modes = Modes::from_count(3);
        // c_2^+ c_0, applied to |1_0 1_1> should give -|1_1 1_2>.
        let op = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::from([2]), HashSet::from([0]));

        let mut ket = CmpntList::new(modes.clone());
        ket.push_clear();
        ket.get_elem_mut_ref(0)
            .assign_set_unchecked(HashSet::from([0, 1]));

        let mut out = CmpntList::new(modes);
        out.push_clear();
        let sign = out
            .get_elem_mut_ref(0)
            .assign_mul_by_op(op.borrow(), ket.get_elem_ref(0));

        assert_eq!(sign, Some(Sign(true)));
        assert_eq!(out.get_elem_ref(0).to_set(), HashSet::from([1, 2]));
    }

    #[test]
    fn test_assign_mul_by_op_forbidden() {
        use crate::container::bit_matrix::AsRowMutRef;
        use crate::container::traits::proj::Borrow;
        use crate::container::traits::MutRefElements;
        use crate::fermion::operator::cmpnt::Cmpnt;
        use std::collections::HashSet;

        let modes = Modes::from_count(3);
        // c_0 annihilating an unoccupied mode 0 is forbidden.
        let op = Cmpnt::from_sets_unchecked(modes.clone(), HashSet::new(), HashSet::from([0]));

        let mut ket = CmpntList::new(modes.clone());
        ket.push_clear();
        ket.get_elem_mut_ref(0)
            .assign_set_unchecked(HashSet::from([1]));

        let mut out = CmpntList::new(modes);
        out.push_clear();
        let sign = out
            .get_elem_mut_ref(0)
            .assign_mul_by_op(op.borrow(), ket.get_elem_ref(0));

        assert_eq!(sign, None);
    }
}
