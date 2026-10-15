//! Algebraic strings represent coefficient-free primitive terms, such as a Pauli
//! operator string or a computational basis state. For efficiency, their storage
//! groups strings into contiguous lists.

#![allow(dead_code)]
pub mod mode;
pub mod parse;
pub mod springs;
pub mod state_springs;
