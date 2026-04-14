//! Definitions for a single component.

use crate::cmpnt::parse::ParseError;
use crate::cmpnt::springs::ModeSettings;
use crate::container::errors::OutOfBounds;
use crate::container::traits::proj::BorrowMut;
use crate::container::word_iters::{Elem, WordIters};
use crate::qubit::mode::{PauliMatrix, Qubits};
use crate::qubit::pauli::cmpnt_major::cmpnt_list::{CmpntList, CmpntRef};
use crate::qubit::pauli::springs::Springs;
use crate::qubit::traits::{
    QubitsBased, QubitsRelabel, QubitsRelabelled, QubitsStandardize, QubitsStandardized,
};
use std::collections::HashMap;
use std::fmt;

/// A single Pauli word. Note that this should never be used in types like `Vec<Cmpnt>`, which makes a separate
/// dynamic allocation for each Pauli word. This would forfeit the main advantage of `CmpntList` which stores
/// many Pauli words under a single dynamic allocation.
/// All functionality which does not pertain to the instantiation of `Cmpnt` is to be accessed via the borrow
/// and borrow_mut functions.
impl Elem<CmpntList> {
    /// Create a single identity Pauli word on the given qubit space.
    pub fn new(qubits: Qubits) -> Self {
        let mut this = Self(CmpntList::new(qubits));
        this.0.push_clear();
        this
    }

    /// Create a Pauli word from a dense vector of Pauli matrices on the given qubit space.
    pub fn from_vec(qubits: Qubits, paulis: Vec<PauliMatrix>) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(qubits);
        this.borrow_mut().assign_vec(paulis)?;
        Ok(this)
    }

    /// Create a Pauli word from a sparse map of qubit indices to Pauli matrices on the given qubit space.
    pub fn from_map(
        qubits: Qubits,
        paulis: HashMap<usize, PauliMatrix>,
    ) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(qubits);
        this.borrow_mut().assign_map(paulis)?;
        Ok(this)
    }

    /// Create a Pauli word from a dense vector, inferring the qubit space from the length of `paulis`.
    pub fn from_vec_default(paulis: Vec<PauliMatrix>) -> Self {
        let n_mode = paulis.len();
        // can't be Mode OoB error, since the qubit space was chosen to avoid it.
        Self::from_vec(Qubits::from_count(n_mode), paulis).unwrap()
    }

    /// Create a Pauli word from a sparse map, inferring the qubit space from the highest index in `paulis`.
    pub fn from_map_default(paulis: HashMap<usize, PauliMatrix>) -> Self {
        let n_mode = paulis
            .iter()
            .max_by_key(|(&i, _)| i)
            .map(|(&i, _)| i + 1)
            .unwrap_or_default();
        // can't be Mode OoB error, since the qubit space was chosen to avoid it.
        Self::from_map(Qubits::from_count(n_mode), paulis).unwrap()
    }

    /// Create an owned single-word value by copying the contents of `cmpnt_ref`.
    pub fn from_cmpnt_ref(cmpnt_ref: CmpntRef) -> Self {
        let mut this = Self::new(cmpnt_ref.to_qubits());
        this.borrow_mut().assign(cmpnt_ref);
        this
    }

    /// Create a Pauli word by parsing spring entry `i` on the given qubit space.
    pub fn from_spring(qubits: Qubits, springs: &Springs, i: usize) -> Result<Self, ParseError> {
        let mut list = CmpntList::new(qubits);
        list.push_spring(springs, i)?;
        Ok(Self(list))
    }

    /// Create a Pauli word from spring entry `i`, inferring the smallest count-based qubit space that fits it.
    pub fn from_spring_default(springs: &Springs, i: usize) -> Result<Self, ParseError> {
        Self::from_spring(
            Qubits::from_count(springs.get_mode_inds().default_n_mode() as usize),
            springs,
            i,
        )
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

pub type PauliWord = Elem<CmpntList>;
