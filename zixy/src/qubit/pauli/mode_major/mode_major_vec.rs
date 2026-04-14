//! Qubit pauli mode major mode major vec module.

// //! Defines vectors of Pauli words, stored as contiguous vectors of Pauli matrices associated with each mode.
// use std::collections::HashMap;
// use std::fmt::Display;

// use super::encoding;
// use super::mode::{PauliMatrix, Qubits};
// use super::mode_settings::OpModeSettings;
// use crate::cmpnt::mode_settings::ModeSettings;
// use crate::container::iterable_elements::IterableElements;
// use crate::container::table::Table;
// use crate::container::two_bit_vec::{self, TwoBitVec};
// use crate::container::traits::{ElementMutRef, ElementRef};
// use crate::qubit::clifford;
// use crate::qubit::encoding::minus_i_to_power_phase;
// use crate::qubit::mode::{n_qubit_from_qubits, pauli_matrix_product};
// use crate::qubit::products::*;
// use crate::qubit::traits::{Collection, PushPaulis};
// use crate::utils::arith::divceil;
// use bincode::{Decode, Encode};
// use itertools::{izip, Itertools};
// use num_complex::Complex64;
// use num_traits::One;

// /// Contiguous and compact storage for vectors of mode-major Pauli words.
// #[derive(Debug, Hash, PartialEq, Eq, Encode, Decode)]
// pub struct CmpntVec {
//     /// Each element is the vector of bit-packed Pauli matrices associated with each mode.
//     modes: Vec<TwoBitVec>,
//     /// Qubits on which the Pauli words are defined
//     qubits: Qubits,
// }

// impl CmpntVec {
//     /// Create an empty CmpntVec on the Qubits given.
//     pub fn new(qubits: Qubits) -> Self {
//         Self {
//             modes: vec![TwoBitVec::default(); n_qubit_from_qubits(&qubits) as usize],
//             qubits,
//         }
//     }

//     /// Push the mode settings and index i at the back of this vector.
//     /// Repeated mode indices with differing Pauli matrix settings in general result in a phase
//     /// factor. This factor is returned as a power of the imag unit.
//     pub fn push_mode_settings(&mut self, mode_settings: &OpModeSettings, i: usize) -> u8 {
//         let i_cmpnt: usize = self.len();
//         self.push_clear();
//         let mut tot_phase: usize = 0;
//         // multiply-in each term from the right
//         for (pauli, i_mode) in mode_settings.get_pauli_iter(i) {
//             let (new_pauli, phase) =
//                 pauli_matrix_product(self.elem_ref(i_cmpnt).get_qubit(i_mode), pauli);
//             tot_phase += phase as usize;
//             OpMutRef(self, i_cmpnt).set_qubit(i_mode, new_pauli);
//         }
//         (tot_phase & 3) as u8
//     }

//     /// Create a new instance from given OpModeSettings, returning the associated phases as a TwoBitVec.
//     pub fn from_mode_settings(
//         qubits: Qubits,
//         mode_settings: &OpModeSettings,
//     ) -> (Self, two_bit_vec::TwoBitVec) {
//         let mut this = Self::new(qubits);
//         let mut phases = two_bit_vec::TwoBitVec::default();
//         for i in 0..mode_settings.len() {
//             phases.push(this.push_mode_settings(mode_settings, i));
//         }
//         (this, phases)
//     }

//     /// Create a new instance from given OpModeSettings, but infer the Qubits as a Qubits::from_count from the max
//     /// mode index of the mode_settings.
//     pub fn from_mode_settings_default(
//         mode_settings: &OpModeSettings,
//     ) -> (Self, two_bit_vec::TwoBitVec) {
//         let n_qubit = mode_settings.get_mode_inds().default_n_mode();
//         Self::from_mode_settings(Qubits::from_count(n_qubit), mode_settings)
//     }

//     /// Get an iterator over the components in this collection.
//     pub fn iter(&self) -> impl Iterator<Item = OpRef> {
//         (0..self.len()).map(|x| self.elem_ref(x))
//     }
// }

// impl Clone for CmpntVec {
//     fn clone(&self) -> Self {
//         Self {
//             modes: self.modes.clone(),
//             qubits: self.qubits.clone(),
//         }
//     }
// }

// impl Collection for CmpntVec {
//     fn len(&self) -> usize {
//         if let Some(v) = self.modes.first() {v.len()}
//         else {0}
//     }

//     fn qubits(&self) -> &Qubits {
//         todo!()
//     }
// }

// impl PushPaulis for CmpntVec {

//     fn push_clear(&mut self) {
//         for v in self.modes.iter_mut() {
//             v.push(0);
//         }
//     }

//     fn push_pauli_iter(&mut self, iter: impl Iterator<Item = (usize, PauliMatrix)>) -> Result<(), super::traits::ModeOutOfBounds> {
//         self.push_clear();
//         let n = self.len();
//         for (i, p) in iter {
//             if let Some(mode) = self.modes.get_mut(i as usize) {
//                 mode.set(mode.len().saturating_sub(1) as usize, p as u8);
//             }
//             else {
//                 for v in self.modes.iter_mut() {
//                     v.resize(v.len().saturating_sub(1));
//                 }
//                 return Err(ModeOutOfBounds{mode_ind: i, n_mode: n});
//             }
//         }
//         Ok(())
//     }
// }
