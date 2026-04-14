//! Definitions for a single computational basis state.

use std::collections::HashSet;
use std::fmt;

use crate::container::bit_matrix::AsRowMutRef;
use crate::container::errors::OutOfBounds;
use crate::container::word_iters::{Elem, WordIters};
use crate::qubit::mode::Qubits;
use crate::qubit::state::cmpnt_list::{CmpntList, CmpntMutRef, CmpntRef};
use crate::qubit::traits::{
    QubitsBased, QubitsRelabel, QubitsRelabelled, QubitsStandardize, QubitsStandardized,
};

/// A single computational basis vector.
/// All functionality which does not pertain to the instantiation of State is to be accessed via the borrow
/// and borrow_mut functions.
impl Elem<CmpntList> {
    /// Borrow an immutable projected view.
    pub fn borrow(&self) -> CmpntRef<'_> {
        CmpntRef {
            word_iters: &self.0,
            index: 0,
        }
    }
    /// Borrow a mutable projected view.
    pub fn borrow_mut(&mut self) -> CmpntMutRef<'_> {
        CmpntMutRef {
            word_iters: &mut self.0,
            index: 0,
        }
    }

    /// Create a new instance.
    pub fn new(qubits: Qubits) -> Self {
        let mut this = Self(CmpntList::new(qubits));
        this.0.push_clear();
        this
    }

    /// Create an instance from a vector of bit settings.
    pub fn from_vec(qubits: Qubits, values: Vec<bool>) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(qubits);
        this.borrow_mut().assign_vec(values)?;
        Ok(this)
    }

    /// Create an instance from a set of positions of set bits.
    pub fn from_set(qubits: Qubits, i_qubits: HashSet<usize>) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(qubits);
        this.borrow_mut().assign_set(i_qubits)?;
        Ok(this)
    }

    /// Create an instance from a vector of bit settings inferring the qubit space from the length of the vector.
    pub fn from_vec_default(values: Vec<bool>) -> Self {
        Self::from_vec(Qubits::from_count(values.len()), values).unwrap()
    }

    /// Create an instance from a reference to a component.
    pub fn from_cmpnt_ref(cmpnt_ref: CmpntRef) -> Self {
        let mut this = Self::new(cmpnt_ref.to_qubits());
        this.borrow_mut().assign(cmpnt_ref);
        this
    }
}

impl QubitsBased for Elem<CmpntList> {
    fn qubits(&self) -> &Qubits {
        self.0.qubits()
    }
}

impl QubitsStandardize for Elem<CmpntList> {
    fn general_standardize(&mut self, n_qubit: usize) {
        self.0.general_standardize(n_qubit);
    }
}

impl QubitsStandardized for Elem<CmpntList> {}

impl QubitsRelabel for Elem<CmpntList> {
    fn qubits_mut(&mut self) -> &mut Qubits {
        self.0.qubits_mut()
    }
}

impl QubitsRelabelled for Elem<CmpntList> {}

impl fmt::Display for Elem<CmpntList> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.fmt_elem(0))
    }
}

pub type BasisState = Elem<CmpntList>;
