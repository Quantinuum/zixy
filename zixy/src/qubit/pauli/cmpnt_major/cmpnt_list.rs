//! Defines gapless, contiguous lists of Pauli words using a cmpnt-major (mode-minor) symplectic representation.
//! Assumes a "little-endian" convention whereby the Pauli matrix acting on the first qubit is stored in the least
//! significant bits of the first u64.

use std::collections::HashMap;
use std::fmt::Display;

use super::products::*;
use crate::cmpnt::parse::ParseError;
use crate::cmpnt::springs::ModeSettings;
use crate::container::bit_matrix::{AsBitMatrix, BitMatrix};
use crate::container::coeffs::complex_sign::{ComplexSign, ComplexSignVec};
use crate::container::coeffs::traits::NumReprVec;
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::traits::{
    Compatible, Elements, EmptyClone, HasIndex, MutRefElements, RefElements,
};
use crate::container::word_iters::{self, WordIters};
use crate::qubit::mode::{PauliMatrix, Qubits};
use crate::qubit::pauli::cmpnt_major::encoding;
use crate::qubit::pauli::mode_major;
use crate::qubit::pauli::mode_major::array::Array;
use crate::qubit::pauli::springs::Springs;
use crate::qubit::traits::{
    DifferentQubits, PauliWordMutRef, PauliWordRef, PushPaulis, QubitsBased, QubitsRelabel,
    QubitsStandardize, QubitsStandardized,
};
use crate::utils::arith::divceil;
use itertools::{izip, Itertools};
use serde::{Deserialize, Serialize};

/// Contiguous and compact storage for vectors of Pauli words.
#[derive(Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CmpntList {
    /// X part of the phaseless Pauli tableau
    x_part: BitMatrix,
    /// Z part of the phaseless Pauli tableau
    z_part: BitMatrix,
    /// `Qubits` on which the Pauli words are defined
    qubits: Qubits,
}

impl CmpntList {
    /// Create an empty `CmpntList` on the `Qubits` given.
    pub fn new(qubits: Qubits) -> Self {
        Self {
            x_part: BitMatrix::new(qubits.len()),
            z_part: BitMatrix::new(qubits.len()),
            qubits,
        }
    }

    /// Get a reference to the X-part of the symplectic representation.
    pub fn x_part(&self) -> &BitMatrix {
        &self.x_part
    }

    /// Get a reference to the Z-part of the symplectic representation.
    pub fn z_part(&self) -> &BitMatrix {
        &self.z_part
    }

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

    /// Append spring entry `i`, checking mode bounds and returning any accumulated phase factor.
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
        let n_qubit = springs.get_mode_inds().default_n_mode() as usize;
        // from_springs can only result in
        Self::from_springs(Qubits::from_count(n_qubit), springs).unwrap()
    }

    /// Resize the list to `n` components.
    pub fn resize(&mut self, n: usize) {
        self.x_part.resize(n);
        self.z_part.resize(n);
    }

    /// A qubit is redundant with respect to a component list if it holds the identity matrix across all
    /// components of the list.
    pub fn qubit_redundant(&self, i_qubit: usize) -> Result<bool, OutOfBounds> {
        Ok(self.x_part.bit_redundant(i_qubit)? && self.z_part.bit_redundant(i_qubit)?)
    }

    /// Get the compatibility matrix over the list of Pauli strings, with 1 representing an anti-commuting string pair, and 0 a commuting pair.
    pub fn compatibility_matrix(&self) -> BitMatrix {
        let mut out = BitMatrix::new_square(self.len());
        for i_row in 0..self.len() {
            let row_ref = self.get_elem_ref(i_row);
            for i_col in 0..=i_row {
                let col_ref = self.get_elem_ref(i_col);
                if row_ref.commutes_with(col_ref) {
                    out.set_bit_unchecked(i_row, i_col, true);
                    out.set_bit_unchecked(i_col, i_row, true);
                }
            }
        }
        out
    }

    /// Return whether the indexed component is an element of the centralizer set of this list.
    /// i.e. whether the component commutes with all others in the list.
    pub fn belongs_to_centralizer(&self, i_cmpnt: usize) -> bool {
        let row_ref = self.get_elem_ref(i_cmpnt);
        for i_row in 0..self.len() {
            if row_ref.anticommutes_with(self.get_elem_ref(i_row)) {
                return false;
            }
        }
        true
    }

    /// Get an iterator over the elements of the centralizer.
    pub fn centralizer_members(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).filter(|i| self.belongs_to_centralizer(*i))
    }
}

impl Compatible for CmpntList {
    fn compatible_with(&self, other: &Self) -> bool {
        self.qubits == other.qubits
    }
}

impl Elements for CmpntList {
    fn len(&self) -> usize {
        self.x_part.len()
    }
}

impl EmptyClone for CmpntList {
    fn empty_clone(&self) -> Self {
        Self {
            x_part: self.x_part().empty_clone(),
            z_part: self.z_part().empty_clone(),
            qubits: self.qubits.clone(),
        }
    }
}

impl QubitsBased for CmpntList {
    fn qubits(&self) -> &Qubits {
        &self.qubits
    }
}

impl QubitsStandardized for CmpntList {
    fn general_standardized(&self, n_qubit: usize) -> Self::OwnedType {
        let mut out = Self::new(Qubits::from_count(n_qubit));
        out.resize(self.len());
        for (i, old_ref) in self.iter().enumerate() {
            let mut new_ref = out.get_elem_mut_ref(i);
            for (i_old, i_new) in self.qubits.iter().enumerate() {
                new_ref.set_pauli_unchecked(i_new, old_ref.get_pauli_unchecked(i_old));
            }
        }
        out
    }
}

impl QubitsStandardize for CmpntList {
    fn general_standardize(&mut self, n_qubit: usize) {
        *self = self.general_standardized(n_qubit)
    }

    fn resize_standardize(&mut self, n_qubit: usize) {
        self.x_part.reformat(n_qubit);
        self.z_part.reformat(n_qubit);
        self.qubits = Qubits::from_count(n_qubit)
    }
}

impl QubitsRelabel for CmpntList {
    fn qubits_mut(&mut self) -> &mut Qubits {
        &mut self.qubits
    }
}

impl PushPaulis for CmpntList {
    fn push_pauli_iter(
        &mut self,
        iter: impl Iterator<Item = (usize, PauliMatrix)>,
    ) -> Result<(), OutOfBounds> {
        self.x_part.push_clear();
        self.z_part.push_clear();
        for (i, p) in iter {
            if let Err(x) = OutOfBounds::check(i, self.qubits.len(), Dimension::Cmpnt) {
                self.x_part.pop();
                self.z_part.pop();
                return Err(x);
            }
            let index = self.len().saturating_sub(1);
            CmpntMutRef {
                word_iters: self,
                index,
            }
            .set_pauli_unchecked(i, p)
        }
        Ok(())
    }
}

impl Display for CmpntList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            (0..self.len())
                .map(|i| self.get_elem_ref(i).to_string())
                .join(", ")
        )
    }
}

impl Clone for CmpntList {
    fn clone(&self) -> Self {
        Self {
            x_part: self.x_part.clone(),
            z_part: self.z_part.clone(),
            qubits: self.qubits.clone(),
        }
    }
}

impl From<mode_major::array::Array> for CmpntList {
    fn from(value: Array) -> Self {
        Self {
            x_part: value.x_part().transpose(),
            z_part: value.z_part().transpose(),
            qubits: value.to_qubits(),
        }
    }
}

impl WordIters for CmpntList {
    fn elem_u64it(&self, i: usize) -> impl Iterator<Item = u64> + Clone {
        self.x_part[i]
            .iter()
            .copied()
            .chain(self.z_part[i].iter().copied())
    }

    fn elem_u64it_mut(&mut self, i: usize) -> impl Iterator<Item = &mut u64> {
        self.x_part[i].iter_mut().chain(self.z_part[i].iter_mut())
    }

    fn u64it_size(&self) -> usize {
        self.x_part.u64it_size() + self.z_part.u64it_size()
    }

    fn pop_and_swap(&mut self, i_row: usize) {
        self.x_part.pop_and_swap(i_row);
        self.z_part.pop_and_swap(i_row);
    }

    fn resize(&mut self, n: usize) {
        self.x_part.resize(n);
        self.z_part.resize(n);
    }

    fn fmt_elem(&self, i: usize) -> String {
        (0..self.qubits().len())
            .filter_map(|i_qubit| {
                let pauli = self.get_elem_ref(i).get_pauli_unchecked(i_qubit);
                match pauli {
                    PauliMatrix::I => None,
                    PauliMatrix::X => Some(format!("X{i_qubit}").to_string()),
                    PauliMatrix::Y => Some(format!("Y{i_qubit}").to_string()),
                    PauliMatrix::Z => Some(format!("Z{i_qubit}").to_string()),
                }
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
}

pub type CmpntRef<'a> = word_iters::ElemRef<'a, CmpntList>;
pub type CmpntMutRef<'a> = word_iters::ElemMutRef<'a, CmpntList>;

impl<'a> CmpntRef<'a> {
    /// Return the raw `u64` words of the referenced component's X part.
    pub fn get_x_part_slice(&self) -> &[u64] {
        &self.word_iters.x_part[self.get_index()]
    }

    /// Return the raw `u64` words of the referenced component's Z part.
    pub fn get_z_part_slice(&self) -> &[u64] {
        &self.word_iters.z_part[self.get_index()]
    }

    /// Get the X and Z parts of the referenced component as separate iterators over u64 words.
    pub fn get_part_iters(
        &self,
    ) -> (
        impl Iterator<Item = u64> + Clone + use<'_>,
        impl Iterator<Item = u64> + Clone + use<'_>,
    ) {
        (
            self.get_x_part_slice().iter().copied(),
            self.get_z_part_slice().iter().copied(),
        )
    }

    /// Get the phase that would result from multiplying this referenced component by another.
    pub fn phase_of_mul(&self, other: Self) -> ComplexSign {
        let (lx, lz) = self.get_part_iters();
        let (rx, rz) = other.get_part_iters();
        mul_op_op_phase(lx, lz, rx, rz)
    }

    /// Return whether this referenced Pauli word commutes with another.
    pub fn commutes_with(&self, other: Self) -> bool {
        self.phase_of_mul(other.clone()) == other.phase_of_mul(self.clone())
    }

    /// Return whether this referenced Pauli word anticommutes with another.
    pub fn anticommutes_with(&self, other: Self) -> bool {
        !self.commutes_with(other)
    }

    /// Return the result of right-multiplication by a single-u64 word state
    pub fn mul_state_u64(&self, ket: u64) -> (u64, ComplexSign) {
        mul_op_state_u64(self.get_x_part_slice()[0], self.get_z_part_slice()[0], ket)
    }
}

impl<'a> QubitsBased for CmpntRef<'a> {
    fn qubits(&self) -> &Qubits {
        &self.word_iters.qubits
    }
}

impl<'a> PauliWordRef for CmpntRef<'a> {
    type T = CmpntList;

    fn get_container(&self) -> &Self::T {
        self.word_iters
    }

    fn get_pauli_unchecked(&self, i_mode: usize) -> PauliMatrix {
        encoding::get_qubit(self.get_x_part_slice(), self.get_z_part_slice(), i_mode)
    }

    fn count(&self, pauli: PauliMatrix) -> usize {
        let (x_iter, z_iter) = self.get_part_iters();
        let n = izip!(x_iter, z_iter)
            .map(|(x, z)| {
                match pauli {
                    PauliMatrix::I => !x & !z,
                    PauliMatrix::X => x & !z,
                    PauliMatrix::Y => x & z,
                    PauliMatrix::Z => !x & z,
                }
                .count_ones() as usize
            })
            .sum::<usize>();
        match pauli {
            PauliMatrix::I => {
                // the I case is overcounted in general due to the clear bits at the end of the string.
                let n_extra = 64 * divceil(self.qubits().len(), 64) - self.qubits().len();
                n.saturating_sub(n_extra)
            }
            _ => n,
        }
    }
}

impl<'a> CmpntMutRef<'a> {
    /// Return mutable access to the raw `u64` words of the referenced component's X part.
    pub fn get_x_part_slice_mut(&mut self) -> &mut [u64] {
        let index = self.get_index();
        &mut self.word_iters.x_part[index]
    }

    /// Return mutable access to the raw `u64` words of the referenced component's Z part.
    pub fn get_z_part_slice_mut(&mut self) -> &mut [u64] {
        let index = self.get_index();
        &mut self.word_iters.z_part[index]
    }

    /// Get the X and Z parts of the referenced component as a separate slices over u64 words.
    pub fn elem_part_slices_mut(&mut self) -> (&mut [u64], &mut [u64]) {
        let index = self.get_index();
        let elems = &mut self.word_iters;
        (&mut elems.x_part[index], &mut elems.z_part[index])
    }

    /// Get the X and Z parts of the referenced component as a separate iterators over u64 words.
    pub fn get_part_iters_mut(
        &mut self,
    ) -> (
        impl Iterator<Item = &mut u64> + use<'_>,
        impl Iterator<Item = &mut u64> + use<'_>,
    ) {
        let index = self.get_index();
        let elems = &mut self.word_iters;
        (
            elems.x_part[index].iter_mut(),
            elems.z_part[index].iter_mut(),
        )
    }

    /// Assign values to many qubits from a vector of Pauli matrices.
    pub fn assign_vec_unchecked(&mut self, paulis: Vec<PauliMatrix>) {
        self.clear();
        for (i_qubit, pauli) in paulis.iter().enumerate() {
            self.set_pauli_unchecked(i_qubit, *pauli);
        }
    }

    /// Assign values to many qubits from a vector of Pauli matrices.
    pub fn assign_vec(&mut self, paulis: Vec<PauliMatrix>) -> Result<(), OutOfBounds> {
        OutOfBounds::check(
            paulis.len().saturating_sub(1),
            self.qubits().len(),
            Dimension::Mode,
        )?;
        self.assign_vec_unchecked(paulis);
        Ok(())
    }

    /// Assign values to many qubits from a hashmap from mode indices to Pauli matrices.
    pub fn assign_map_unchecked(&mut self, paulis: HashMap<usize, PauliMatrix>) {
        self.clear();
        for (i_qubit, pauli) in paulis.into_iter() {
            self.set_pauli_unchecked(i_qubit, pauli);
        }
    }

    /// Assign values to many qubits from a hashmap from mode indices to Pauli matrices.
    pub fn assign_map(&mut self, paulis: HashMap<usize, PauliMatrix>) -> Result<(), OutOfBounds> {
        if let Some((&i, _)) = paulis.iter().max_by_key(|(&i, _)| i) {
            OutOfBounds::check(i, self.qubits().len(), Dimension::Mode)?;
        }
        self.assign_map_unchecked(paulis);
        Ok(())
    }

    /// `Set` the value of this referenced Pauli word from the multiplication of two others, ignoring phase.
    pub fn assign_mul_matrices_unchecked(&mut self, lhs: CmpntRef, rhs: CmpntRef) {
        let (lx, lz) = lhs.get_part_iters();
        let (rx, rz) = rhs.get_part_iters();
        let (ox, oz) = self.get_part_iters_mut();
        mul_op_op_matrices(lx, lz, rx, rz, ox, oz)
    }

    /// `Set` the value of this referenced Pauli word from the multiplication of two others, ignoring phase.
    /// `DifferentQubits` error is returned if the `lhs` or `rhs` is based on different qubits from those of `self`.
    pub fn assign_mul_matrices(
        &mut self,
        lhs: CmpntRef,
        rhs: CmpntRef,
    ) -> Result<(), DifferentQubits> {
        DifferentQubits::check_transitive(self, &lhs, &rhs)?;
        self.assign_mul_matrices_unchecked(lhs, rhs);
        Ok(())
    }

    /// `Set` the value of this referenced Pauli word from the multiplication of two others, and return the associated phase.
    pub fn assign_mul_unchecked(&mut self, lhs: CmpntRef, rhs: CmpntRef) -> ComplexSign {
        self.assign_mul_matrices_unchecked(lhs.clone(), rhs.clone());
        lhs.phase_of_mul(rhs)
    }

    /// `Set` the value of this referenced Pauli word from the multiplication of two others, and return the associated phase.
    /// `DifferentQubits` error is returned if the `lhs` or `rhs` is based on different qubits from those of `self`.
    pub fn assign_mul(
        &mut self,
        lhs: CmpntRef,
        rhs: CmpntRef,
    ) -> Result<ComplexSign, DifferentQubits> {
        DifferentQubits::check_transitive(self, &lhs, &rhs)?;
        Ok(self.assign_mul_unchecked(lhs, rhs))
    }

    /// Multiply by rhs in-place, ignoring phase.
    pub fn imul_by_cmpnt_ref_matrices_unchecked(&mut self, rhs: CmpntRef) {
        let (lx, lz) = self.get_part_iters_mut();
        let (rx, rz) = rhs.get_part_iters();
        imul_op_op_matrices(lx, lz, rx, rz)
    }

    /// Multiply by rhs in-place, ignoring phase.
    /// `DifferentQubits` error is returned if the `rhs` is based on different qubits from those of `self`.
    pub fn imul_by_cmpnt_ref_matrices(&mut self, rhs: CmpntRef) -> Result<(), DifferentQubits> {
        DifferentQubits::check(self, &rhs)?;
        self.imul_by_cmpnt_ref_matrices_unchecked(rhs);
        Ok(())
    }

    /// Multiply by rhs in-place, returning phase.
    pub fn imul_by_cmpnt_ref_unchecked(&mut self, rhs: CmpntRef) -> ComplexSign {
        let (lx, lz) = self.get_part_iters_mut();
        let (rx, rz) = rhs.get_part_iters();
        imul_op_op(lx, lz, rx, rz)
    }

    /// Multiply by rhs in-place, returning phase.
    /// `DifferentQubits` error is returned if the `rhs` is based on different qubits from those of `self`.
    pub fn imul_by_cmpnt_ref(&mut self, rhs: CmpntRef) -> Result<ComplexSign, DifferentQubits> {
        DifferentQubits::check(self, &rhs)?;
        Ok(self.imul_by_cmpnt_ref_unchecked(rhs))
    }
}

impl<'a> QubitsBased for CmpntMutRef<'a> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

impl<'a> PauliWordMutRef for CmpntMutRef<'a> {
    fn set_pauli_unchecked(&mut self, i_mode: usize, pauli: PauliMatrix) {
        let parts = self.elem_part_slices_mut();
        encoding::set_qubit(parts.0, parts.1, i_mode, pauli);
    }

    fn get_pauli_unchecked(&self, i_mode: usize) -> PauliMatrix {
        self.as_ref().get_pauli_unchecked(i_mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::traits::proj::{Borrow, BorrowMut};
    use crate::qubit::mode::pauli_matrix_product as pmp;
    use crate::qubit::mode::{PauliMatrix, Qubits};
    use crate::qubit::pauli::cmpnt_major::cmpnt::PauliWord;
    use crate::qubit::test::HEHP_STO3G_HAM_JW_INPUT;
    use bincode::config;
    use num_complex::Complex64;
    use rstest::rstest;
    use PauliMatrix::*;

    #[test]
    fn test_empty() {
        {
            let list = CmpntList::new(Qubits::from_count(4));
            assert!(list.is_empty());
        }
    }

    #[test]
    fn test_to_string() {
        let mut op = PauliWord::new(Qubits::from_count(10));

        op.borrow_mut().assign_vec_unchecked(vec![X, Y, I, Z]);
        assert_eq!(op.borrow().to_string(), "X0 Y1 Z3");

        op.borrow_mut()
            .assign_vec_unchecked(vec![X, Y, I, I, X, Z, X, Y, Y, Z]);
        assert_eq!(op.borrow().to_string(), "X0 Y1 X4 Z5 X6 Y7 Y8 Z9");

        op.borrow_mut()
            .assign_vec_unchecked(vec![X, Y, I, I, I, Z, X, Y, Y, Z]);
        assert_eq!(op.borrow().to_string(), "X0 Y1 Z5 X6 Y7 Y8 Z9");

        op.borrow_mut()
            .assign_vec_unchecked(vec![X, I, I, I, I, Z, X, Y, Y, Z]);
        assert_eq!(op.borrow().to_string(), "X0 Z5 X6 Y7 Y8 Z9");

        op.borrow_mut()
            .assign_vec_unchecked(vec![X, I, I, I, I, Z, X, Y, Y, I]);
        assert_eq!(op.borrow().to_string(), "X0 Z5 X6 Y7 Y8");

        op.borrow_mut().assign_vec_unchecked(vec![I]);
        assert_eq!(op.borrow().to_string(), "");

        op.borrow_mut().assign_vec_unchecked(vec![]);
        assert_eq!(op.borrow().to_string(), "");
    }

    #[test]
    fn test_to_string_long() {
        let mut op = PauliWord::new(Qubits::from_count(500));

        op.borrow_mut().set_pauli_unchecked(0, X);
        op.borrow_mut().set_pauli_unchecked(499, Z);
        assert_eq!(op.borrow().to_string(), "X0 Z499");
    }

    #[rstest]
    #[case(vec![I], vec![I], vec![pmp(I, I).0], pmp(I, I).1)]
    #[case(vec![I], vec![X], vec![pmp(I, X).0], pmp(I, X).1)]
    #[case(vec![I], vec![Y], vec![pmp(I, Y).0], pmp(I, Y).1)]
    #[case(vec![I], vec![Z], vec![pmp(I, Z).0], pmp(I, Z).1)]
    #[case(vec![X], vec![I], vec![pmp(X, I).0], pmp(X, I).1)]
    #[case(vec![X], vec![X], vec![pmp(X, X).0], pmp(X, X).1)]
    #[case(vec![X], vec![Y], vec![pmp(X, Y).0], pmp(X, Y).1)]
    #[case(vec![X], vec![Z], vec![pmp(X, Z).0], pmp(X, Z).1)]
    #[case(vec![Y], vec![I], vec![pmp(Y, I).0], pmp(Y, I).1)]
    #[case(vec![Y], vec![X], vec![pmp(Y, X).0], pmp(Y, X).1)]
    #[case(vec![Y], vec![Y], vec![pmp(Y, Y).0], pmp(Y, Y).1)]
    #[case(vec![Y], vec![Z], vec![pmp(Y, Z).0], pmp(Y, Z).1)]
    #[case(vec![Z], vec![I], vec![pmp(Z, I).0], pmp(Z, I).1)]
    #[case(vec![Z], vec![X], vec![pmp(Z, X).0], pmp(Z, X).1)]
    #[case(vec![Z], vec![Y], vec![pmp(Z, Y).0], pmp(Z, Y).1)]
    #[case(vec![Z], vec![Z], vec![pmp(Z, Z).0], pmp(Z, Z).1)]
    #[case(vec![X, Y, I, Z], vec![I, Z, I, Z], vec![X, X, I, I], ComplexSign(1))]
    #[case(vec![X, Y, I, Y], vec![I, Z, I, Z], vec![X, X, I, X], ComplexSign(2))]
    #[case(
        vec![I, Y, X, I, I, Y, Z, I],
        vec![Y, I, Y, Z, X, Y, Z, Z],
        vec![Y, Y, Z, Z, X, I, I, Z],
        ComplexSign(1)
    )]
    #[case(
        vec![Y, X, I, Y, Z, X, Z, X, I, X, Z, X, Y, Y, X, X, I, X, I, Y, Y, I, I, Z, X, Z, X, X, I, Z, X, X],
        vec![X, Z, Y, Z, Y, X, Z, Z, X, X, Z, X, Y, Y, Y, Z, Z, Y, I, Z, I, X, Z, Y, X, Z, Y, I, X, X, I, I],
        vec![Z, Y, Y, X, X, I, I, Y, X, I, I, I, I, I, Z, Y, Z, Z, I, X, Y, X, Z, X, I, I, Z, X, X, Y, X, X],
        ComplexSign(0)
    )]
    #[case(
        vec![Y, X, X, X, X, X, Z, X, Z, I, I, I, Z, Z, I, I, Z, Y, X, Y, I, I, X, Y, X, Y, I, X, Z, Y, X, I, Y, Z, I, I],
        vec![Z, Y, Y, Y, Y, Y, Z, I, Y, Y, I, I, Z, Z, Z, I, Z, X, Y, X, Y, Y, Y, I, Y, X, I, Y, X, X, Y, I, X, Y, X, X],
        vec![X, Z, Z, Z, Z, Z, I, X, X, Y, I, I, I, I, Z, I, I, Z, Z, Z, Y, Y, Z, Y, Z, Z, I, Z, Y, Z, Z, I, Z, X, X, X],
        ComplexSign(1)
    )]
    fn test_mult(
        #[case] lv: Vec<PauliMatrix>,
        #[case] rv: Vec<PauliMatrix>,
        #[case] ov: Vec<PauliMatrix>,
        #[case] phase: ComplexSign,
    ) {
        let lhs = PauliWord::from_vec_default(lv);
        let rhs = PauliWord::from_vec_default(rv);
        let mut out = PauliWord::new(Qubits::from_count(lhs.qubits().len()));
        assert_eq!(
            out.borrow_mut()
                .assign_mul_unchecked(lhs.borrow(), rhs.borrow()),
            phase
        );
        assert_eq!(out.borrow().get_pauli_vec(), ov);
    }

    /*
    #[rstest]
    #[case(vec![X, Y, I, Z], vec![0, 0, 0, 0], vec![1, 1, 0, 0], ComplexSign(1))]
    #[case(vec![X, Y, I, Z], vec![0, 1, 0, 0], vec![1, 0, 0, 0], ComplexSign(3))]
    #[case(vec![X, Y, I, Z], vec![0, 1, 1, 0], vec![1, 0, 1, 0], ComplexSign(3))]
    #[case(vec![X, Y, I, Y], vec![0, 1, 1, 0], vec![1, 0, 1, 1], ComplexSign(0))]
    fn test_mult_ket(
        #[case] lv: Vec<PauliMatrix>,
        #[case] rv: Vec<u8>,
        #[case] ov: Vec<u8>,
        #[case] phase: ComplexSign,
    ) {
        use crate::{cmpnt::bitset_cmpnt_list::AsCmpntRef, qubit::state::cmpnt::BasisState};

        let lhs = PauliWord::from_vec_default(lv);
        let rhs = BasisState::from_vec_default(rv.iter().map(|x| *x != 0).collect());
        let mut out = BasisState::new(lhs.to_qubits());
        assert_eq!(
            out.borrow_mut()
                .assign_mul_by_op(lhs.borrow(), rhs.borrow()),
            phase
        );
        let ov = ov.iter().map(|x| *x != 0).collect::<Vec<_>>();
        assert_eq!(out.borrow().to_vec(), ov);
    }
    */

    #[rstest]
    #[case("X1 X3 Y7", "X1 X3 Y7", vec![0])]
    #[case("X1 Y1 Y7", "Z1 Y7", vec![1])]
    #[case("X1 Y1000 Y1000", "X1", vec![0])]
    #[case("X1 Z1000 Y1000", "X1 X1000", vec![3])]
    #[case("X1 Z1000 Y1000", "X1 X1000", vec![3])]
    #[case("X1 Y1, Z0 Y5 X1", "Z1, Z0 X1 Y5", vec![1, 0])]
    #[case("X1 Y1, Z0 Y5 X0", "Z1, Y0 Y5", vec![1, 1])]
    fn test_from_springs(#[case] input: &str, #[case] output: &str, #[case] chk_phases: Vec<u8>) {
        let springs = Springs::from_str(input);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        let (op_vec, phases) = CmpntList::from_springs_default(&springs);
        assert_eq!(op_vec.len(), phases.len());
        assert_eq!(format!("{op_vec}"), output);
        assert_eq!(phases.0.as_bytes().collect_vec(), chk_phases);
    }

    const _0: Complex64 = Complex64::new(0.0, 0.0);
    const P1: Complex64 = Complex64::new(1.0, 0.0);
    const PJ: Complex64 = Complex64::new(0.0, 1.0);
    const M1: Complex64 = Complex64::new(-1.0, 0.0);
    const MJ: Complex64 = Complex64::new(0.0, -1.0);
    #[rstest]
    #[case(
        vec![I],
        vec![P1, _0,
             _0, P1])]
    #[case(
        vec![X],
        vec![_0, P1,
             P1, _0])]
    #[case(
        vec![Y],
        vec![_0, MJ,
             PJ, _0])]
    #[case(
        vec![Z],
        vec![P1, _0,
             _0, M1])]
    #[case(
        vec![I, I],
        vec![P1, _0, _0, _0,
             _0, P1, _0, _0,
             _0, _0, P1, _0,
             _0, _0, _0, P1,])]
    #[case(
        vec![I, X],
        vec![_0, P1, _0, _0,
              P1, _0, _0, _0,
              _0, _0, _0, P1,
              _0, _0, P1, _0,])]
    #[case(
        vec![I, Y],
        vec![_0, MJ, _0, _0,
             PJ, _0, _0, _0,
             _0, _0, _0, MJ,
             _0, _0, PJ, _0,])]
    #[case(
        vec![I, Z],
        vec![P1, _0, _0, _0,
             _0, M1, _0, _0,
             _0, _0, P1, _0,
             _0, _0, _0, M1,])]
    #[case(
        vec![X, I],
        vec![_0, _0, P1, _0,
             _0, _0, _0, P1,
             P1, _0, _0, _0,
             _0, P1, _0, _0,])]
    #[case(
        vec![X, X],
        vec![_0, _0, _0, P1,
             _0, _0, P1, _0,
             _0, P1, _0, _0,
             P1, _0, _0, _0,])]
    #[case(
        vec![X, Y],
        vec![_0, _0, _0, MJ,
             _0, _0, PJ, _0,
             _0, MJ, _0, _0,
             PJ, _0, _0, _0,])]
    #[case(
        vec![X, Z],
        vec![_0, _0, P1, _0,
             _0, _0, _0, M1,
             P1, _0, _0, _0,
             _0, M1, _0, _0,])]
    #[case(
        vec![Y, X, Z],
        vec![_0, _0, _0, _0, _0, _0, MJ, _0,
             _0, _0, _0, _0, _0, _0, _0, PJ,
             _0, _0, _0, _0, MJ, _0, _0, _0,
             _0, _0, _0, _0, _0, PJ, _0, _0,
             _0, _0, PJ, _0, _0, _0, _0, _0,
             _0, _0, _0, MJ, _0, _0, _0, _0,
             PJ, _0, _0, _0, _0, _0, _0, _0,
             _0, MJ, _0, _0, _0, _0, _0, _0,])]
    #[case(
        vec![Y, Z, I, X],
        vec![_0, _0, _0, _0, _0, _0, _0, _0, _0, MJ, _0, _0, _0, _0, _0, _0,
             _0, _0, _0, _0, _0, _0, _0, _0, MJ, _0, _0, _0, _0, _0, _0, _0,
             _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, MJ, _0, _0, _0, _0,
             _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, MJ, _0, _0, _0, _0, _0,
             _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, PJ, _0, _0,
             _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, PJ, _0, _0, _0,
             _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, PJ,
             _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, PJ, _0,
             _0, PJ, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0,
             PJ, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0,
             _0, _0, _0, PJ, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0,
             _0, _0, PJ, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0,
             _0, _0, _0, _0, _0, MJ, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0,
             _0, _0, _0, _0, MJ, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0, _0,
             _0, _0, _0, _0, _0, _0, _0, MJ, _0, _0, _0, _0, _0, _0, _0, _0,
             _0, _0, _0, _0, _0, _0, MJ, _0, _0, _0, _0, _0, _0, _0, _0, _0,])]

    fn test_to_sparse(#[case] paulis: Vec<PauliMatrix>, #[case] flat_mat: Vec<Complex64>) {
        let op = PauliWord::from_vec_default(paulis);
        assert_eq!(
            op.borrow()
                .to_sparse_matrix(true)
                .to_dense()
                .flatten()
                .to_vec(),
            flat_mat
        );
    }

    #[test]
    fn test_to_from_binary() {
        let springs = Springs::from_str(HEHP_STO3G_HAM_JW_INPUT);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        let op_vec = CmpntList::from_springs_default(&springs).0;
        let encoded: Vec<u8> = bincode::serde::encode_to_vec(&op_vec, config::standard()).unwrap();
        let (decoded, len): (CmpntList, usize) =
            bincode::serde::decode_from_slice(&encoded[..], config::standard()).unwrap();
        assert_eq!(len, encoded.len());
        assert_eq!(op_vec, decoded);
    }
}
