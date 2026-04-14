//! Single-term utilities.

use std::collections::HashMap;
use std::fmt::Display;

use crate::container::coeffs::traits::NumRepr;
use crate::container::errors::OutOfBounds;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::container::word_iters::terms::{self, AsViewMut};
use crate::qubit::mode::{PauliMatrix, Qubits};
use crate::qubit::pauli::cmpnt_major::cmpnt_list::CmpntList;
use crate::qubit::pauli::cmpnt_major::terms::Terms;
use crate::qubit::traits::{
    QubitsBased, QubitsRelabel, QubitsRelabelled, QubitsStandardize, QubitsStandardized,
};

/// A single Pauli word with a generically-typed coefficient.
pub type Term<C /*: NumRepr*/> = terms::Term<CmpntList, C>;

impl<C: NumRepr> Term<C> {
    /// Create a single Pauli term with unit coefficient on the given qubit space.
    pub fn new(qubits: Qubits) -> Self {
        let mut terms = Terms::<C>::new(qubits);
        terms.push_clear();
        Self {
            word_iters: terms.word_iters,
            coeffs: terms.coeffs,
        }
    }

    /// Create a term from a dense Pauli vector and a given coefficient on the given qubit space.
    pub fn from_vec_and_coeff(
        qubits: Qubits,
        paulis: Vec<PauliMatrix>,
        c: C,
    ) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(qubits);
        this.borrow_mut()
            .get_word_iter_mut_ref()
            .assign_vec(paulis)?;
        this.borrow_mut().set_coeff(c);
        Ok(this)
    }

    /// Create a term from a sparse Pauli map and a given coefficient on the given qubit space.
    pub fn from_map_and_coeff(
        qubits: Qubits,
        paulis: HashMap<usize, PauliMatrix>,
        c: C,
    ) -> Result<Self, OutOfBounds> {
        let mut this = Self::new(qubits);
        this.borrow_mut()
            .get_word_iter_mut_ref()
            .assign_map(paulis)?;
        this.borrow_mut().set_coeff(c);
        Ok(this)
    }

    /// Create a term from a dense Pauli vector, inferring a count-based qubit space and assigning coefficient `c`.
    pub fn from_vec_and_coeff_default(paulis: Vec<PauliMatrix>, c: C) -> Result<Self, OutOfBounds> {
        Self::from_vec_and_coeff(Qubits::from_count(paulis.len()), paulis, c)
    }

    /// Create a term from a sparse Pauli map, inferring a count-based qubit space and assigning coefficient `c`.
    pub fn from_map_and_coeff_default(paulis: HashMap<usize, PauliMatrix>, c: C) -> Self {
        Self::from_map_and_coeff(Qubits::from_count(paulis.len()), paulis, c).unwrap()
    }

    /// Create a term from a dense Pauli vector with unit coefficient on the given qubit space.
    pub fn from_vec(qubits: Qubits, paulis: Vec<PauliMatrix>) -> Result<Self, OutOfBounds> {
        Self::from_vec_and_coeff(qubits, paulis, C::default())
    }

    /// Create a term from a dense Pauli vector and panic if the data is out of bounds for `qubits`.
    pub fn from_vec_unchecked(qubits: Qubits, paulis: Vec<PauliMatrix>) -> Self {
        Self::from_vec(qubits, paulis).unwrap()
    }

    /// Create a term from a sparse Pauli map with unit coefficient on the given qubit space with bounds checking.
    pub fn from_map(
        qubits: Qubits,
        paulis: HashMap<usize, PauliMatrix>,
    ) -> Result<Self, OutOfBounds> {
        Self::from_map_and_coeff(qubits, paulis, C::default())
    }

    /// Create a term from a sparse Pauli map with unit coefficient on the given qubit space without bounds checking.
    pub fn from_map_unchecked(qubits: Qubits, paulis: HashMap<usize, PauliMatrix>) -> Self {
        Self::from_map(qubits, paulis).unwrap()
    }

    /// Create a term from a sparse Pauli map, inferring a count-based qubit space and using unit coefficient with bounds checking.
    pub fn from_vec_default(paulis: Vec<PauliMatrix>) -> Result<Self, OutOfBounds> {
        Self::from_vec_and_coeff_default(paulis, C::default())
    }

    /// Create a term from a sparse Pauli map with unit coefficient on an inferred qubit space without bounds checking.
    pub fn from_map_default(paulis: HashMap<usize, PauliMatrix>) -> Self {
        Self::from_map_and_coeff_default(paulis, C::default())
    }
}

impl<C: NumRepr> Display for Term<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.borrow())
    }
}

impl<C: NumRepr> QubitsBased for Term<C> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

impl<C: NumRepr> QubitsStandardize for Term<C> {
    fn general_standardize(&mut self, n_qubit: usize) {
        self.word_iters.general_standardize(n_qubit)
    }

    fn resize_standardize(&mut self, n_qubit: usize) {
        self.word_iters.resize_standardize(n_qubit)
    }
}
impl<C: NumRepr> QubitsStandardized for Term<C> {}

impl<C: NumRepr> QubitsRelabel for Term<C> {
    fn qubits_mut(&mut self) -> &mut Qubits {
        self.word_iters.qubits_mut()
    }
}
impl<C: NumRepr> QubitsRelabelled for Term<C> {}
