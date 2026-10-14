//! Defines gapless, contiguous lists of computational basis vectors using a string-major (mode-minor) bitset representation.
//! Assumes a "little-endian" convention whereby the local state of the first qubit is stored in the least
//! significant bits of the first u64.

use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::container::bit_matrix::{self, AsBitMatrix, AsRowMutRef, AsRowRef};
use crate::container::coeffs::complex_sign::ComplexSign;
use crate::container::errors::OutOfBounds;
use crate::container::table::Table;
use crate::container::traits::{Compatible, Elements, EmptyClone, MutRefElements, RefElements};
use crate::container::word_iters::{self, WordIters};
use crate::qubit::mode::Qubits;
use crate::qubit::operator::string_major;
use crate::qubit::operator::string_major::products::{imul_op_state, mul_op_state};
use crate::qubit::traits::{QubitsBased, QubitsRelabel, QubitsStandardize, QubitsStandardized};
use crate::string::springs::ModeSettings;
use crate::string::state_springs::BinarySprings;

/// Contiguous and compact storage for computational basis states defined on a given basis of `Qubits`.
#[derive(Debug, Hash, PartialEq, Clone, Serialize, Deserialize)]
pub struct StringList {
    /// Raw storage table for the computational basis states as bitsets.
    bitsets: bit_matrix::BitMatrix,
    /// Space or number of qubits on which the computational basis is defined.
    qubits: Qubits,
}

impl StringList {
    /// Create an empty `StringList` on the `Qubits` given.
    pub fn new(qubits: Qubits) -> Self {
        Self {
            bitsets: bit_matrix::BitMatrix::new(qubits.len()),
            qubits,
        }
    }

    /// Create a new instance from given `BinarySprings`.
    pub fn from_springs(qubits: Qubits, springs: &BinarySprings) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(qubits);
        this.set_from_springs(springs)?;
        Ok(this)
    }

    /// Create an instance from springs with an inferred qubit space.
    pub fn from_springs_default(springs: &BinarySprings) -> Self {
        Self::from_springs(
            Qubits::from_count(springs.get_mode_inds().default_n_mode() as usize),
            springs,
        )
        .unwrap()
    }
}

impl bit_matrix::AsBitMatrix for StringList {
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

impl Compatible for StringList {
    fn compatible_with(&self, other: &Self) -> bool {
        self.qubits == other.qubits
    }
}

impl Elements for StringList {
    fn len(&self) -> usize {
        self.bitsets.len()
    }
}

impl QubitsBased for StringList {
    fn qubits(&self) -> &Qubits {
        &self.qubits
    }
}

impl QubitsStandardized for StringList {
    fn general_standardized(&self, n_qubit: usize) -> Self::OwnedType {
        let mut out = Self::new(Qubits::from_count(n_qubit));
        out.resize(self.len());
        for (i, old_ref) in self.iter().enumerate() {
            let mut new_ref = out.get_elem_mut_ref(i);
            for (i_old, i_new) in self.qubits.iter().enumerate() {
                new_ref.set_bit_unchecked(i_new, old_ref.get_bit_unchecked(i_old));
            }
        }
        out
    }
}

impl QubitsStandardize for StringList {
    fn general_standardize(&mut self, n_qubit: usize) {
        *self = self.general_standardized(n_qubit)
    }

    fn resize_standardize(&mut self, n_qubit: usize) {
        self.bitsets.reformat(n_qubit);
        self.qubits = Qubits::from_count(n_qubit)
    }
}

impl QubitsRelabel for StringList {
    fn qubits_mut(&mut self) -> &mut Qubits {
        &mut self.qubits
    }
}

impl EmptyClone for StringList {
    fn empty_clone(&self) -> Self {
        Self {
            bitsets: self.bitsets.empty_clone(),
            qubits: self.qubits.clone(),
        }
    }
}

impl WordIters for StringList {
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

    fn fmt_elem(&self, i: usize) -> std::string::String {
        self.bitsets.fmt_elem(i)
    }

    fn resize(&mut self, n: usize) {
        self.bitsets.resize(n);
    }
}

pub type StringRef<'a> = word_iters::ElemRef<'a, StringList>;
pub type StringMutRef<'a> = word_iters::ElemMutRef<'a, StringList>;

impl<'a> QubitsBased for StringRef<'a> {
    fn qubits(&self) -> &Qubits {
        &self.word_iters.qubits
    }
}

impl<'a> QubitsBased for StringMutRef<'a> {
    fn qubits(&self) -> &Qubits {
        &self.word_iters.qubits
    }
}

impl<'a> bit_matrix::AsRowRef for StringRef<'a> {
    fn bit_mat(&self) -> &impl bit_matrix::AsBitMatrix {
        &self.word_iters.bitsets
    }
}

impl<'a> bit_matrix::AsRowMutRef for StringMutRef<'a> {
    fn bit_mat(&self) -> &impl bit_matrix::AsBitMatrix {
        &self.word_iters.bitsets
    }

    fn bit_mat_mut(&mut self) -> &mut impl bit_matrix::AsBitMatrix {
        &mut self.word_iters.bitsets
    }
}

impl<'a> StringMutRef<'a> {
    /// `Set` the value of this referenced basis state to the left-multiplication of another by a Pauli operator,
    /// returning the phase of the product.
    pub fn assign_mul_by_op(
        &mut self,
        lhs: string_major::string_list::StringRef,
        rhs: StringRef,
    ) -> ComplexSign {
        let (lx, lz) = lhs.get_part_iters();
        mul_op_state(lx, lz, rhs.get_u64it(), self.get_u64it_mut())
    }

    /// `Set` the value of this referenced basis state to its left-multiplication by a Pauli operator,
    /// returning the phase of the product.
    pub fn imul_by_op(&mut self, op: string_major::string_list::StringRef) -> ComplexSign {
        let (lx, lz) = op.get_part_iters();
        imul_op_state(lx, lz, self.get_u64it_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubit::state::string::BasisState;
    use rstest::rstest;

    #[test]
    fn test_empty() {
        {
            let v = StringList::new(Qubits::from_count(4));
            assert!(v.bitsets.is_empty());
        }
        {
            let v = StringList::new(Qubits::from_count(4));
            assert!(v.bitsets.is_empty());
        }
    }

    #[test]
    fn test_get_set() {
        let mut v = BasisState::new(Qubits::from_count(10));
        for i in 0..v.qubits().len() {
            assert!(!v.borrow().get_bit_unchecked(i));
            v.borrow_mut().set_bit_unchecked(i, true);
            assert!(v.borrow().get_bit_unchecked(i));
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
        let strings = StringList::from_springs_default(&springs);
        let vecs = (0..strings.len())
            .map(|i| {
                strings
                    .get_elem_ref(i)
                    .iter()
                    .map(|x| if x { 1 } else { 0 })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(output, vecs);
    }

    #[test]
    fn test_to_string() {
        let mut v = BasisState::new(Qubits::from_count(10));

        v.borrow_mut().assign_vec_unchecked(
            vec![1, 0, 1, 1, 0, 1, 1, 1]
                .into_iter()
                .map(|x| x != 0)
                .collect(),
        );
        assert_eq!(v.to_string(), "[1, 0, 1, 1, 0, 1, 1, 1, 0, 0]");
    }
}
