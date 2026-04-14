//! Functionality common among collections of Pauli words.

use std::collections::HashMap;
use std::fmt::Display;

use num_complex::Complex64;

use crate::cmpnt::parse::ParseError;
use crate::container::coeffs::complex_sign::ComplexSign;
use crate::container::coeffs::sign::Sign;
use crate::container::coeffs::traits::NumRepr;
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::traits::{proj, Elements};
use crate::qubit::clifford;
use crate::qubit::mode::{
    pauli_matrix_product, BasisError, DifferentModeCounts, PauliMatrix, Qubits,
};
use crate::qubit::pauli::springs::Springs;

/// Trait for containers that can append Pauli words from sparse or dense specifications of Pauli matrix content.
pub trait PushPaulis: Elements {
    /// Push a Pauli word component given by an iterator.
    fn push_pauli_iter(
        &mut self,
        iter: impl Iterator<Item = (usize, PauliMatrix)>,
    ) -> Result<(), OutOfBounds>;

    /// Push a clear component of identity matrices.
    fn push_pauli_identity(&mut self) {
        let _ = self.push_pauli_iter(std::iter::empty());
    }

    /// Push a new component at the back of the vector encoded with the given Pauli word
    /// expressed as a vector of PauliMatrices no longer than self.qubits
    fn push_pauli_vec(&mut self, paulis: Vec<PauliMatrix>) -> Result<(), OutOfBounds> {
        self.push_pauli_iter(paulis.into_iter().enumerate())
    }

    /// Push a new component at the back of the vector encoded with the given Pauli word.
    /// expressed as a map from mode indices to PauliMatrices with keys less than self.`len()`
    fn push_pauli_map(&mut self, paulis: HashMap<usize, PauliMatrix>) -> Result<(), OutOfBounds> {
        self.push_pauli_iter(paulis.into_iter())
    }
}

/// Error returned when operands are defined on different qubit registers.
#[derive(Debug, PartialEq)]
pub struct DifferentQubits {}

impl DifferentQubits {
    /// Check whether the `QubitsBased` inputs are based on the same qubit space.
    pub fn check<L: QubitsBased, R: QubitsBased>(lhs: &L, rhs: &R) -> Result<(), DifferentQubits> {
        if lhs.same_qubits(rhs) {
            Ok(())
        } else {
            Err(DifferentQubits {})
        }
    }

    /// Check whether the three `QubitsBased` inputs are all based on the same qubit space.
    pub fn check_transitive<L: QubitsBased, M: QubitsBased, R: QubitsBased>(
        lhs: &L,
        mid: &M,
        rhs: &R,
    ) -> Result<(), DifferentQubits> {
        Self::check(lhs, rhs)?;
        Self::check(mid, rhs)
    }
}

impl std::fmt::Display for DifferentQubits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Qubits-based objects are based on different qubit spaces."
        )
    }
}
impl std::error::Error for DifferentQubits {}

/// Any object that is based on a space of many qubits.
pub trait QubitsBased {
    /// Get a reference to the `Qubits` instance on which this object is defined.
    fn qubits(&self) -> &Qubits;

    /// Get a clone of the `Qubits` instance on which this object is defined.
    fn to_qubits(&self) -> Qubits {
        self.qubits().clone()
    }

    /// Return whether `self` is based on the same qubits as `other`.
    fn same_qubits<T: QubitsBased>(&self, other: &T) -> bool {
        self.qubits() == other.qubits()
    }
}

/// Functionality for reordering the bits in qubit-based objects such that the qubit register can be replaced with a Count-based qubit register
pub trait QubitsStandardize: QubitsBased {
    /// Rewrite the object onto the standard `0..n_qubit` register, and relabel with a count-type qubit space.
    fn general_standardize(&mut self, n_qubit: usize);

    /// Separate standardize function for resizes, which can often be done more efficiently than generic nontrivial.
    fn resize_standardize(&mut self, n_qubit: usize) {
        self.general_standardize(n_qubit)
    }

    /// Standardize onto a qubit register with one more qubit than the original register.
    fn push_standardize(&mut self) {
        self.resize_standardize(self.qubits().len() + 1);
    }

    /// Standardize onto `Qubits::from_count(n_qubit)`, using the cheaper resize path when possible.
    fn standardize(&mut self, n_qubit: usize) {
        if self.to_qubits() != Qubits::from_count(n_qubit) {
            if self.to_qubits() == Qubits::from_count(self.qubits().len()) {
                self.resize_standardize(n_qubit)
            } else {
                self.general_standardize(n_qubit)
            }
        }
    }
}

/// Convenience trait for producing standardized owned copies of qubit-based values.
pub trait QubitsStandardized: proj::ToOwned
where
    Self::OwnedType: QubitsStandardize,
{
    /// Return an owned copy after applying `general_standardize(n_qubit)`.
    fn general_standardized(&self, n_qubit: usize) -> Self::OwnedType {
        let mut tmp = self.to_owned();
        tmp.general_standardize(n_qubit);
        tmp
    }

    /// Separate function for resizes, which can often be done more efficiently than generic nontrivial.
    fn resize_standardized(&self, n_qubit: usize) -> Self::OwnedType {
        let mut tmp = self.to_owned();
        tmp.resize_standardize(n_qubit);
        tmp
    }

    /// Return an owned copy standardized after appending one qubit to the register size.
    fn push_standardized(&self) -> Self::OwnedType {
        let mut tmp = self.to_owned();
        tmp.push_standardize();
        tmp
    }

    /// Return an owned copy standardized onto `Qubits::from_count(n_qubit)`.
    fn standardized(&self, n_qubit: usize) -> Self::OwnedType {
        let mut tmp = self.to_owned();
        tmp.standardize(n_qubit);
        tmp
    }
}

/// Trait for qubit-based values whose register labels can be replaced without changing their physical layout.
pub trait QubitsRelabel: QubitsStandardize {
    /// Get a mut reference to the `Qubits` instance on which this object is defined.
    fn qubits_mut(&mut self) -> &mut Qubits;

    /// Replace the current qubit labels with `qubits` without changing the number of qubits or the physical layout of the qubits-based object.
    fn relabel(&mut self, qubits: Qubits) -> Result<(), BasisError> {
        DifferentModeCounts::check(qubits.len(), self.qubits().len())?;
        *self.qubits_mut() = qubits;
        Ok(())
    }
}

/// Convenience trait for producing owned copies with relabelled qubit registers.
pub trait QubitsRelabelled: proj::ToOwned
where
    Self::OwnedType: QubitsRelabel,
{
    /// Return an owned copy relabelled onto `qubits`.
    fn relabelled(&self, qubits: Qubits) -> Result<Self::OwnedType, BasisError> {
        let mut out = proj::ToOwned::to_owned(self);
        out.relabel(qubits)?;
        Ok(out)
    }
}

/// Immutable view trait for a single Pauli word.
pub trait PauliWordRef: QubitsBased + Display {
    type T: QubitsBased;
    /// Return the container which `self` views.
    fn get_container(&self) -> &Self::T;
    /// Return the Pauli matrix at `i_mode` without bounds checking.
    fn get_pauli_unchecked(&self, i_mode: usize) -> PauliMatrix;
    /// Return the Pauli matrix at `i_mode`, or `OutOfBounds` if `i_mode` is invalid.
    fn get_pauli(&self, i_mode: usize) -> Result<PauliMatrix, OutOfBounds> {
        OutOfBounds::check(
            i_mode,
            self.get_container().qubits().len(),
            Dimension::Cmpnt,
        )?;
        Ok(self.get_pauli_unchecked(i_mode))
    }

    /// Get the pauli matrices at each mode and collect into a vector.
    fn get_pauli_vec(&self) -> Vec<PauliMatrix> {
        (0..self.qubits().len())
            .map(|i| self.get_pauli_unchecked(i))
            .collect()
    }

    /// Get a map from mode indices to pauli matrices, ignoring trivial (I matrix) qubits.
    fn get_pauli_map(&self) -> HashMap<usize, PauliMatrix> {
        (0..self.qubits().len())
            .filter_map(|i| {
                let pauli = self.get_pauli_unchecked(i);
                if pauli == PauliMatrix::I {
                    None
                } else {
                    Some((i, pauli))
                }
            })
            .collect()
    }

    /// Count how many qubit positions currently store matrix `pauli`.
    fn count(&self, pauli: PauliMatrix) -> usize;

    /// Return whether the cmpnt is purely diagonal (all I and Z matrices).
    fn is_diagonal(&self) -> bool {
        self.count(PauliMatrix::X) + self.count(PauliMatrix::Y) == 0
    }

    /// Return an iterator over the Pauli matrices in this word.
    fn iter(&self) -> impl Iterator<Item = PauliMatrix> {
        (0..self.qubits().len()).map(|i| self.get_pauli_unchecked(i))
    }

    /// Return whether the cmpnt is identity (all I matrices).
    fn is_identity(&self) -> bool {
        self.count(PauliMatrix::I) == self.qubits().len()
    }

    /// Following the algorithm in `https://arxiv.org/pdf/2301.00560`, build a sparse matrix as a tensor product of the
    /// Pauli matrices in the referenced Pauli word.
    fn to_sparse_matrix(&self, big_endian: bool) -> sprs::CsMat<Complex64> {
        use PauliMatrix::*;
        let n = self.get_container().qubits().len();
        let dim = 1_usize << n;
        let mut indices: Vec<usize> = vec![0; dim];
        let mut data: Vec<Complex64> = vec![Complex64::ZERO; dim];

        let ny = self.count(Y);
        let ind_fn = |i| if big_endian { n - 1 - i } else { i };

        if ny != 0 || self.count(X) != 0 {
            /*
             * not a diagonal string, so follow Algo 1 in the paper
             */
            let i_col = (0..n)
                .map(|i| {
                    let p = self.get_pauli_unchecked(i);
                    if p == X || p == Y {
                        1 << ind_fn(i)
                    } else {
                        0
                    }
                })
                .sum();

            indices[0] = i_col;
            // (-i) ^ ny
            data[0] = ComplexSign::from(ny).conj().to_complex();
            for i_mode in 0..n {
                let exp_i_mode = 1_usize << i_mode;
                let mat = self.get_pauli_unchecked(ind_fn(i_mode));
                let is_nd_mat = mat == X || mat == Y;
                let parity = mat == Y || mat == Z;
                for j in 0..exp_i_mode {
                    let i = j + exp_i_mode;
                    let i_col = if is_nd_mat {
                        indices[j].saturating_sub(exp_i_mode)
                    } else {
                        indices[j] + exp_i_mode
                    };
                    indices[i] = i_col;
                    data[i] = if parity { -data[j] } else { data[j] };
                }
            }
        } else {
            /*
             * no X or Y matrices means the output is diagonal, so follow Algo 2.
             */
            indices[0] = 0;
            data[0] = Complex64::ONE;
            for i_mode in 0..n {
                let exp_i_mode = 1_usize << i_mode;
                let mat = self.get_pauli_unchecked(ind_fn(i_mode));
                let parity = mat == Z;
                for j in 0..exp_i_mode {
                    let i = j + exp_i_mode;
                    indices[i] = i;
                    data[i] = if parity { -data[j] } else { data[j] };
                }
            }
        }
        sprs::CsMat::new((dim, dim), (0..dim + 1).collect(), indices, data)
    }
}

/// Mutable view trait for a single Pauli word whose per-qubit Pauli matrices can be edited in place.
pub trait PauliWordMutRef: QubitsBased {
    /// Set every qubit position to the identity matrix.
    fn clear(&mut self) {
        (0..self.qubits().len()).for_each(|i| self.set_pauli_unchecked(i, PauliMatrix::I));
    }

    /// Set the Pauli matrix at `i_mode` without bounds checking.
    fn set_pauli_unchecked(&mut self, i_mode: usize, pauli: PauliMatrix);
    /// Set the Pauli matrix at `i_mode`, or return `OutOfBounds` if `i_mode` is invalid.
    fn set_pauli(&mut self, i_mode: usize, pauli: PauliMatrix) -> Result<(), OutOfBounds> {
        OutOfBounds::check(i_mode, self.qubits().len(), Dimension::Mode)?;
        self.set_pauli_unchecked(i_mode, pauli);
        Ok(())
    }

    /// Assign from a vector of Pauli matrices without bounds checking.
    fn set_pauli_vec_unchecked(&mut self, paulis: Vec<PauliMatrix>) {
        for (i_mode, pauli) in paulis.iter().enumerate() {
            self.set_pauli_unchecked(i_mode, *pauli);
        }
    }

    /// Assign from a vector of Pauli matrices, with bounds checking on the highest index used.
    fn set_pauli_vec(&mut self, paulis: Vec<PauliMatrix>) -> Result<(), OutOfBounds> {
        OutOfBounds::check(
            paulis.len().saturating_sub(1),
            self.qubits().len(),
            Dimension::Mode,
        )?;
        self.set_pauli_vec_unchecked(paulis);
        Ok(())
    }

    /// Assign from a map of qubit positions to Pauli matrices, without bounds checking.
    fn set_pauli_map_unchecked(&mut self, paulis: HashMap<usize, PauliMatrix>) {
        for (i_mode, pauli) in paulis.into_iter() {
            self.set_pauli_unchecked(i_mode, pauli);
        }
    }

    /// Assign from a map of qubit positions to Pauli matrices, with bounds checking on the highest position index used.
    fn set_pauli_map(&mut self, paulis: HashMap<usize, PauliMatrix>) -> Result<(), OutOfBounds> {
        for (i_mode, pauli) in paulis.into_iter() {
            self.set_pauli(i_mode, pauli)?
        }
        Ok(())
    }

    /// Return the Pauli matrix at `i_mode` without bounds checking.
    fn get_pauli_unchecked(&self, i_mode: usize) -> PauliMatrix;

    /// Conjugate the referenced Pauli word by the given clifford operation and return the phase.
    /// Conjugation means compute C P C^\dagger where P is this Pauli word, and C is the Clifford operation.
    fn conj_clifford(&mut self, gate: clifford::Gate) -> Sign {
        use PauliMatrix::*;
        let get_pauli = |i_mode: usize| self.get_pauli_unchecked(i_mode);

        Sign(match gate {
            clifford::Gate::H(i_qubit) => match get_pauli(i_qubit) {
                I => false,
                X => {
                    self.set_pauli_unchecked(i_qubit, Z);
                    false
                }
                Y => true,
                Z => {
                    self.set_pauli_unchecked(i_qubit, X);
                    false
                }
            },
            clifford::Gate::S(i_qubit) => match get_pauli(i_qubit) {
                I => false,
                X => {
                    self.set_pauli_unchecked(i_qubit, Y);
                    false
                }
                Y => {
                    self.set_pauli_unchecked(i_qubit, X);
                    true
                }
                Z => false,
            },
            clifford::Gate::CX(i_qubits) => {
                match get_pauli(i_qubits.get().0) {
                    I => match get_pauli(i_qubits.get().1) {
                        // CX IY CX = ZY
                        Y => {
                            self.set_pauli_unchecked(i_qubits.get().0, Z);
                            false
                        }
                        // CX IZ CX = ZZ
                        Z => {
                            self.set_pauli_unchecked(i_qubits.get().0, Z);
                            false
                        }
                        // CX II CX = II
                        // CX IX CX = IX
                        _ => false,
                    },
                    X => match get_pauli(i_qubits.get().1) {
                        // CX XI CX = XX
                        I => {
                            self.set_pauli_unchecked(i_qubits.get().1, X);
                            false
                        }
                        // CX XX CX = XI
                        X => {
                            self.set_pauli_unchecked(i_qubits.get().1, I);
                            false
                        }
                        // CX XY CX = YZ
                        Y => {
                            self.set_pauli_unchecked(i_qubits.get().0, Y);
                            self.set_pauli_unchecked(i_qubits.get().1, Z);
                            false
                        }
                        // CX XZ CX = YY * -1
                        Z => {
                            self.set_pauli_unchecked(i_qubits.get().0, Y);
                            self.set_pauli_unchecked(i_qubits.get().1, Y);
                            true
                        }
                    },
                    Y => match get_pauli(i_qubits.get().1) {
                        // CX YI CX = YX
                        I => {
                            self.set_pauli_unchecked(i_qubits.get().1, X);
                            false
                        }
                        // CX YX CX = YI
                        X => {
                            self.set_pauli_unchecked(i_qubits.get().1, I);
                            false
                        }
                        // CX YY CX = XZ * -1
                        Y => {
                            self.set_pauli_unchecked(i_qubits.get().0, X);
                            self.set_pauli_unchecked(i_qubits.get().1, Z);
                            true
                        }
                        // CX YZ CX = XY
                        Z => {
                            self.set_pauli_unchecked(i_qubits.get().0, X);
                            self.set_pauli_unchecked(i_qubits.get().1, Y);
                            false
                        }
                    },
                    Z => match get_pauli(i_qubits.get().1) {
                        // CX ZY CX = IY
                        Y => {
                            self.set_pauli_unchecked(i_qubits.get().0, I);
                            false
                        }
                        // CX ZZ CX = IZ
                        Z => {
                            self.set_pauli_unchecked(i_qubits.get().0, I);
                            false
                        }
                        // CX ZI CX = ZI
                        // CX ZX CX = ZX
                        _ => false,
                    },
                }
            }
        })
    }

    /// Conjugate the referenced Pauli operator by each gate in turn and return the overall phase.
    fn conj_clifford_vec(&mut self, gates: &[clifford::Gate]) -> Sign {
        Sign(
            gates
                .iter()
                .map(|gate| self.conj_clifford(*gate).0 as u64)
                .sum::<u64>()
                & 1
                != 0,
        )
    }

    /// Overwrite this Pauli word from spring entry `i`, accumulating and returning the resulting phase.
    fn set_spring_unchecked(&mut self, springs: &Springs, i: usize) -> ComplexSign {
        self.clear();
        let mut tot_phase: ComplexSign = ComplexSign(0);
        // multiply-in each term from the right
        for (pauli, i_mode) in springs.get_pauli_iter(i) {
            let i_mode = i_mode as usize;
            let (new_pauli, phase) = pauli_matrix_product(self.get_pauli_unchecked(i_mode), pauli);
            tot_phase *= phase;
            self.set_pauli_unchecked(i_mode, new_pauli);
        }
        tot_phase
    }

    /// Overwrite this Pauli word from spring entry `i`, returning `ParseError` if any mode index is out of bounds.
    fn set_spring(&mut self, springs: &Springs, i: usize) -> Result<ComplexSign, ParseError> {
        if let Some(i_mode) = springs.get_pauli_iter(i).map(|(_, i)| i as usize).max() {
            OutOfBounds::check(i_mode, self.qubits().len(), Dimension::Mode)?;
            Ok(self.set_spring_unchecked(springs, i))
        } else {
            Ok(ComplexSign(0))
        }
    }
}
