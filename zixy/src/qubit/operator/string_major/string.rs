//! Definitions for a single Pauli string.

use crate::container::errors::OutOfBounds;
use crate::container::traits::proj::BorrowMut;
use crate::container::word_iters::{Elem, WordIters};
use crate::qubit::mode::{PauliMatrix, Qubits};
use crate::qubit::operator::springs::Springs;
use crate::qubit::operator::string_major::string_list::{StringList, StringRef};
use crate::qubit::traits::{
    QubitsBased, QubitsRelabel, QubitsRelabelled, QubitsStandardize, QubitsStandardized,
};
use crate::string::parse::ParseError;
use crate::string::springs::ModeSettings;
use std::collections::HashMap;
use std::fmt;

/// A single Pauli word. Note that this should never be used in types like `Vec<String>`, which makes a separate
/// dynamic allocation for each Pauli word. This would forfeit the main advantage of `StringList` which stores
/// many Pauli words under a single dynamic allocation.
/// All functionality which does not pertain to the instantiation of `String` is to be accessed via the borrow
/// and borrow_mut functions.
impl Elem<StringList> {
    /// Create a single identity Pauli word on the given qubit space.
    pub fn new(qubits: Qubits) -> Self {
        let mut this = Self(StringList::new(qubits));
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

    /// Create an owned single-word value by copying the contents of `string_ref`.
    pub fn from_string_ref(string_ref: StringRef) -> Self {
        let mut this = Self::new(string_ref.to_qubits());
        this.borrow_mut().assign(string_ref);
        this
    }

    /// Create a Pauli word by parsing spring entry `i` on the given qubit space.
    pub fn from_spring(qubits: Qubits, springs: &Springs, i: usize) -> Result<Self, ParseError> {
        let mut list = StringList::new(qubits);
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

impl QubitsBased for Elem<StringList> {
    fn qubits(&self) -> &Qubits {
        self.0.qubits()
    }
}

impl QubitsStandardize for Elem<StringList> {
    fn general_standardize(&mut self, n_qubit: usize) {
        self.0.general_standardize(n_qubit);
    }
}

impl QubitsStandardized for Elem<StringList> {}

impl QubitsRelabel for Elem<StringList> {
    fn qubits_mut(&mut self) -> &mut Qubits {
        self.0.qubits_mut()
    }
}

impl QubitsRelabelled for Elem<StringList> {}

impl fmt::Display for Elem<StringList> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.fmt_elem(0))
    }
}

pub type PauliWord = Elem<StringList>;
