//! Components are representations of fixed-width primitive terms
//! e.g. a qubit pauli operator string or a computational basis state.
//! for efficiency these are implemented as contiguous lists of such terms
//! Component is abbreviated as "cmpnt" throughout.

#![allow(dead_code)]
pub mod mode;
pub mod parse;
pub mod springs;
pub mod state_springs;
