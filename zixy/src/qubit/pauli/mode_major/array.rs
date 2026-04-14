//! Defines gapless, contiguous lists of Pauli words using a cmpnt-major (mode-minor) symplectic representation.
//! Assumes a "little-endian" convention whereby the Pauli matrix acting on the first qubit is stored in the least
//! significant bits of the first u64.

use crate::container::bit_matrix::BitMatrix;
use crate::container::traits::{Compatible, Elements, EmptyClone};
use crate::container::word_iters::WordIters;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::cmpnt_major;
use crate::qubit::traits::QubitsBased;
use serde::{Deserialize, Serialize};

/// Contiguous and compact storage for vectors of Pauli words.
#[derive(Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Array {
    /// X part of the phaseless Pauli tableau transpose
    x_part: BitMatrix,
    /// Z part of the phaseless Pauli tableau transpose
    z_part: BitMatrix,
    /// Number of components (row size of the binary matrices)
    n_cmpnt: usize,
    /// `Qubits` on which the Pauli words are defined (len is the number of rows in the binary matrices)
    qubits: Qubits,
}

impl Array {
    /// Create an empty list on the cmpnts and `Qubits` given.
    pub fn new(qubits: Qubits, n_mode: usize) -> Self {
        Self {
            x_part: BitMatrix::new(n_mode),
            z_part: BitMatrix::new(n_mode),
            n_cmpnt: n_mode,
            qubits,
        }
    }

    /// Get a reference to the X-part of the transposed symplectic representation.
    pub fn x_part(&self) -> &BitMatrix {
        &self.x_part
    }

    /// Get a reference to the Z-part of the transposed symplectic representation.
    pub fn z_part(&self) -> &BitMatrix {
        &self.z_part
    }

    /// Resize the list to `n` components.
    pub fn resize(&mut self, n: usize) {
        self.x_part.resize(n);
        self.z_part.resize(n);
    }
}

impl Compatible for Array {
    fn compatible_with(&self, other: &Self) -> bool {
        self.qubits == other.qubits
    }
}

impl Elements for Array {
    fn len(&self) -> usize {
        self.x_part.len()
    }
}

impl EmptyClone for Array {
    fn empty_clone(&self) -> Self {
        Self {
            x_part: self.x_part.empty_clone(),
            z_part: self.z_part.empty_clone(),
            n_cmpnt: self.n_cmpnt,
            qubits: self.qubits.clone(),
        }
    }
}

impl QubitsBased for Array {
    fn qubits(&self) -> &Qubits {
        &self.qubits
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        Self {
            x_part: self.x_part.clone(),
            z_part: self.z_part.clone(),
            n_cmpnt: self.n_cmpnt,
            qubits: self.qubits.clone(),
        }
    }
}

impl From<cmpnt_major::cmpnt_list::CmpntList> for Array {
    fn from(value: cmpnt_major::cmpnt_list::CmpntList) -> Self {
        Self {
            x_part: value.x_part().transpose(),
            z_part: value.z_part().transpose(),
            n_cmpnt: value.len(),
            qubits: value.to_qubits(),
        }
    }
}

impl WordIters for Array {
    fn elem_u64it(&self, index: usize) -> impl Iterator<Item = u64> + Clone {
        self.x_part
            .elem_u64it(index)
            .chain(self.z_part.elem_u64it(index))
    }

    fn elem_u64it_mut(&mut self, index: usize) -> impl Iterator<Item = &mut u64> {
        self.x_part
            .elem_u64it_mut(index)
            .chain(self.z_part.elem_u64it_mut(index))
    }

    fn u64it_size(&self) -> usize {
        self.x_part.u64it_size() + self.z_part.u64it_size()
    }

    fn resize(&mut self, n: usize) {
        self.x_part.resize(n);
        self.z_part.resize(n);
    }

    fn pop_and_swap(&mut self, index: usize) {
        self.x_part.pop_and_swap(index);
        self.z_part.pop_and_swap(index);
    }
}

#[cfg(test)]
mod tests {
    use crate::qubit::mode::PauliMatrix::*;
    use crate::qubit::mode::Qubits;
    use crate::qubit::pauli::cmpnt_major;
    use crate::qubit::pauli::mode_major::array::Array;
    use crate::qubit::traits::PushPaulis;

    #[test]
    fn test_from_standard() {
        let qubits = Qubits::from_count(6);
        let mut standard_strings = cmpnt_major::cmpnt_list::CmpntList::new(qubits);
        let _ = standard_strings.push_pauli_vec([X, Y, Y, X, I, Z].into());
        let _ = standard_strings.push_pauli_vec([I, X, X, X, Z, Z].into());
        let _ = standard_strings.push_pauli_vec([X, Z, Y, Z, I, I].into());
        let transposed_strings: Array = standard_strings.into();
        assert_eq!(transposed_strings.n_cmpnt, 3);
    }
}
