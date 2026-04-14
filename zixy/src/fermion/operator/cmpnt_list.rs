//! Defines gapless, contiguous lists of normal-ordered fermion ladder operator products.

use std::fmt::Display;

use crate::container::bit_matrix::AsBitMatrix;
use crate::container::coeffs::sign::SignVec;
use crate::container::coeffs::traits::NumReprVec;
use crate::container::errors::OutOfBounds;
use crate::container::traits::{
    Compatible, Elements, EmptyClone, HasIndex, MutRefElements, RefElements,
};
use crate::container::word_iters::{self, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::operator::cre_or_ann;
use crate::fermion::traits::ModesBased;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// Contiguous and compact storage for vectors of Pauli words.
#[derive(Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CmpntList {
    /// creation part of the normal ordered fermion operator list
    pub cre_part: cre_or_ann::CmpntList,
    /// annihilation part of the normal ordered fermion operator list
    pub ann_part: cre_or_ann::CmpntList,
}

impl CmpntList {
    /// Create an empty `CmpntList` on the fermion modes given.
    pub fn new(modes: Modes) -> Self {
        Self {
            cre_part: cre_or_ann::CmpntList::new(modes.clone(), true),
            ann_part: cre_or_ann::CmpntList::new(modes, false),
        }
    }

    /*
    /// Push the sparse string with index i at the back of this vector.
    /// Repeated mode indices with differing Pauli matrix settings in general result in a phase
    /// factor. This factor is returned.
    /// Mode bounds are unchecked.
    pub fn push_spring_unchecked(&mut self, springs: &Springs, i: usize) -> ComplexSign {
        let i_cmpnt: usize = self.len();
        self.push_pauli_identity();
        self.get_elem_mut_ref(i_cmpnt)
            .set_spring_unchecked(springs, i)
    }

    /// Append spring.
    pub fn push_spring(&mut self, springs: &Springs, i: usize) -> Result<ComplexSign, ParseError> {
        let i_cmpnt: usize = self.len();
        self.push_pauli_identity();
        self.get_elem_mut_ref(i_cmpnt).set_spring(springs, i)
    }

    /// Create a new instance from given springs, returning the associated phases as a `TwoBitVec`.
    pub fn from_springs(
        qubits: Qubits,
        springs: &Springs,
    ) -> Result<(Self, ComplexSignVec), ParseError> {
        let mut this = Self::new(qubits);
        let mut phases = ComplexSignVec::default();
        for i in 0..springs.len() {
            let phase = this.push_spring(springs, i)?;
            phases.push(phase);
        }
        Ok((this, phases))
    }

    /// Create a new instance from given springs, but infer the `Qubits` with `Qubits::from_count` from the max
    /// mode index of the sparse strings.
    pub fn from_springs_default(springs: &Springs) -> (Self, ComplexSignVec) {
        let n_qubit = springs.get_mode_inds().default_n_mode();
        // from_springs can only result in
        Self::from_springs(Qubits::from_count(n_qubit), springs).unwrap()
    }
    */

    /// A mode is redundant with respect to a component list if it is not flagged as present in either part of
    /// any component of the list.
    pub fn mode_redundant(&self, i_mode: usize) -> Result<bool, OutOfBounds> {
        Ok(self.cre_part.bit_redundant(i_mode)? && self.ann_part.bit_redundant(i_mode)?)
    }

    /// Conjugate without storing the sign of the exchange.
    pub fn dagger_ignore_signs(&mut self) {
        std::mem::swap(&mut self.cre_part, &mut self.ann_part);
        self.cre_part.dagger_ignore_signs();
        self.ann_part.dagger_ignore_signs();
    }

    /// Conjugate every component, and store the associated signs.
    pub fn dagger(&mut self, signs: &mut SignVec) {
        self.dagger_ignore_signs();
        signs.resize(self.len());
        for (i, elem) in self.iter().enumerate() {
            signs.imul_elem_unchecked(i, elem.get_cre_part().sign_of_dagger());
            signs.imul_elem_unchecked(i, elem.get_ann_part().sign_of_dagger());
        }
    }

    /// Return dagger get signs.
    pub fn dagger_get_signs(&mut self) -> SignVec {
        let mut out = SignVec::default();
        self.dagger(&mut out);
        out
    }
}

impl Compatible for CmpntList {
    fn compatible_with(&self, other: &Self) -> bool {
        self.cre_part.compatible_with(&other.cre_part)
    }
}

impl Elements for CmpntList {
    fn len(&self) -> usize {
        self.cre_part.len()
    }
}

impl EmptyClone for CmpntList {
    fn empty_clone(&self) -> Self {
        Self {
            cre_part: self.cre_part.empty_clone(),
            ann_part: self.ann_part.empty_clone(),
        }
    }
}

impl ModesBased for CmpntList {
    fn modes(&self) -> &Modes {
        self.cre_part.modes()
    }
}

impl Display for CmpntList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]",
            (0..self.len())
                .map(|i| self.get_elem_ref(i).to_string())
                .join(", ")
        )
    }
}

impl Clone for CmpntList {
    fn clone(&self) -> Self {
        Self {
            cre_part: self.cre_part.clone(),
            ann_part: self.ann_part.clone(),
        }
    }
}

impl WordIters for CmpntList {
    fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone {
        self.cre_part
            .elem_u64it(i)
            .chain(self.ann_part.elem_u64it(i))
    }

    fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
        self.cre_part
            .elem_u64it_mut(i)
            .chain(self.ann_part.elem_u64it_mut(i))
    }

    fn u64it_size(&self) -> usize {
        self.cre_part.u64it_size() + self.ann_part.u64it_size()
    }

    fn pop_and_swap(&mut self, i_row: usize) {
        self.cre_part.pop_and_swap(i_row);
        self.ann_part.pop_and_swap(i_row);
    }

    fn resize(&mut self, n: usize) {
        self.cre_part.resize(n);
        self.ann_part.resize(n);
    }

    fn fmt_elem(&self, i: usize) -> String {
        let mut out = self.cre_part.get_elem_ref(i).to_string();
        if !out.is_empty() {
            out.push(' ');
        }
        out += self.ann_part.get_elem_ref(i).to_string().as_str();
        out
    }
}

/// Borrowed immutable handle to a single normal-ordered fermion operator component in a [`CmpntList`].
pub type CmpntRef<'a> = word_iters::ElemRef<'a, CmpntList>;
/// Borrowed mutable handle to a single normal-ordered fermion operator component in a [`CmpntList`].
pub type CmpntMutRef<'a> = word_iters::ElemMutRef<'a, CmpntList>;

impl<'a> CmpntRef<'a> {
    /// Return cre part.
    pub fn get_cre_part(&self) -> cre_or_ann::CmpntRef<'_> {
        self.word_iters.cre_part.get_elem_ref(self.get_index())
    }

    /// Return ann part.
    pub fn get_ann_part(&self) -> cre_or_ann::CmpntRef<'_> {
        self.word_iters.ann_part.get_elem_ref(self.get_index())
    }
}

impl<'a> ModesBased for CmpntRef<'a> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
    }
}

impl<'a> CmpntMutRef<'a> {
    /// Return cre part.
    pub fn get_cre_part(&mut self) -> cre_or_ann::CmpntMutRef<'_> {
        let index = self.get_index();
        self.word_iters.cre_part.get_elem_mut_ref(index)
    }

    /// Return ann part.
    pub fn get_ann_part(&mut self) -> cre_or_ann::CmpntMutRef<'_> {
        let index = self.get_index();
        self.word_iters.ann_part.get_elem_mut_ref(index)
    }
}

impl<'a> ModesBased for CmpntMutRef<'a> {
    fn modes(&self) -> &Modes {
        self.word_iters.modes()
    }
}

#[cfg(test)]
mod tests {
    use crate::container::{bit_matrix::AsRowMutRef, coeffs::sign::Sign};

    use super::*;

    #[test]
    fn test_empty() {
        {
            let list = CmpntList::new(Modes::from_pair_count_spin_major(5));
            assert!(list.is_empty());
        }
    }

    #[test]
    fn test_to_string() {
        let mut op = CmpntList::new(Modes::from_pair_count_spin_major(5));
        op.push_clear();
        let mut op_ref = op.get_elem_mut_ref(0);
        op_ref.get_cre_part().assign_set_unchecked([3, 4, 6].into());
        op_ref.get_ann_part().assign_set_unchecked([0, 5, 9].into());

        assert_eq!(op.to_string(), "[F3^ F4^ F6^ F0 F5 F9]");
        let signs = op.dagger_get_signs();
        assert_eq!(signs.get_unchecked(0), Sign(false));
        assert_eq!(op.to_string(), "[F0^ F5^ F9^ F3 F4 F6]");

        let mut op_ref = op.get_elem_mut_ref(0);
        op_ref.get_cre_part().assign_set_unchecked([3, 4, 6].into());
        op_ref
            .get_ann_part()
            .assign_set_unchecked([0, 1, 5, 9].into());

        assert_eq!(op.to_string(), "[F3^ F4^ F6^ F0 F1 F5 F9]");
        let signs = op.dagger_get_signs();
        // negative parity to reverse 3 ops, positive to reverse 2.
        assert_eq!(signs.get_unchecked(0), Sign(true));
        assert_eq!(op.to_string(), "[F0^ F1^ F5^ F9^ F3 F4 F6]");
    }
}
