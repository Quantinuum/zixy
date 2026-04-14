//! Utilities for mapping from single ladder operator strings to Pauli linear combinations.

use num_complex::Complex64;

use crate::container::coeffs::complex_sign::ComplexSign;
use crate::container::coeffs::traits::{HasCoeffsMut, NumRepr, NumReprVec, Represent};
use crate::container::traits::{Elements, MutRefElements, RefElements};
use crate::container::word_iters::lincomb;
use crate::container::word_iters::terms::AsViewMut;
use crate::fermion::mappings::traits::UpdateParityRho;
use crate::qubit::mode::PauliMatrix::{X, Y, Z};
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::cmpnt_major::cmpnt_list::CmpntList;
use crate::qubit::pauli::cmpnt_major::{term_set, terms};
use crate::qubit::traits::{PauliWordMutRef, QubitsBased};

/// Workspace that caches mapped real and imaginary Pauli strings for encoding fermionic ladder operator products.
#[derive(Clone)]
pub struct Operators {
    re_strings: CmpntList,
    im_strings: CmpntList,
    work: terms::Terms<ComplexSign>,
}

/// Operator stored as a mode index and a creation flag (annihilation if false)
pub type Op = (usize, bool);

impl Operators {
    /// Create a new instance.
    pub fn new<T: UpdateParityRho>(qubits: Qubits, mode_ordering: Option<Vec<usize>>) -> Self {
        let mode_map_fn = |i: usize| {
            if let Some(mode_ordering) = &mode_ordering {
                mode_ordering[i]
            } else {
                i
            }
        };

        let mut out = Self {
            re_strings: CmpntList::new(qubits.clone()),
            im_strings: CmpntList::new(qubits.clone()),
            work: terms::Terms::<ComplexSign>::new(qubits.clone()),
        };
        out.re_strings.resize(qubits.len());
        out.im_strings.resize(qubits.len());
        for i_mode in 0..qubits.len() {
            for i in T::update_set(i_mode, qubits.len()) {
                let i_qubit = mode_map_fn(i);
                out.re_strings
                    .get_elem_mut_ref(i_mode)
                    .set_pauli_unchecked(i_qubit, X);
                out.im_strings
                    .get_elem_mut_ref(i_mode)
                    .set_pauli_unchecked(i_qubit, X);
            }
            for i in T::parity_set(i_mode, qubits.len()) {
                let i_qubit = mode_map_fn(i);
                out.re_strings
                    .get_elem_mut_ref(i_mode)
                    .set_pauli_unchecked(i_qubit, Z);
            }
            for i in T::rho_set(i_mode, qubits.len()) {
                let i_qubit = mode_map_fn(i);
                out.im_strings
                    .get_elem_mut_ref(i_mode)
                    .set_pauli_unchecked(i_qubit, Z);
            }
            let i_qubit = mode_map_fn(i_mode);
            out.re_strings
                .get_elem_mut_ref(i_mode)
                .set_pauli_unchecked(i_qubit, X);
            out.im_strings
                .get_elem_mut_ref(i_mode)
                .set_pauli_unchecked(i_qubit, Y);
        }
        out
    }

    /// Set the internal state of `self` to the product of the fermionic operators in `ops`.
    pub fn load_product(&mut self, ops: &[Op]) {
        self.work.clear();
        let mut op_iter = ops.iter();
        if let Some(op) = op_iter.next() {
            self.work.resize(2);
            // seed with the first operator
            let i = op.0;
            let mut re_part = self.work.word_iters.get_elem_mut_ref(0);
            re_part.assign(self.re_strings.get_elem_ref(i));
            let mut im_part = self.work.word_iters.get_elem_mut_ref(1);
            im_part.assign(self.im_strings.get_elem_ref(i));
            self.work
                .get_coeffs_mut()
                .imul_elem_unchecked(1, ComplexSign(if op.1 { 3 } else { 1 }));
        } else {
            // no ops to work on.
            return;
        }
        for op in op_iter {
            self.work.self_append();
            for i in 0..(self.work.len() / 2) {
                let mut re_part = self.work.word_iters.get_elem_mut_ref(i);
                let phase = re_part.imul_by_cmpnt_ref_unchecked(self.re_strings.get_elem_ref(op.0));
                self.work.get_coeffs_mut().imul_elem_unchecked(i, phase);
            }
            for i in (self.work.len() >> 1)..self.work.len() {
                let mut im_part = self.work.word_iters.get_elem_mut_ref(i);
                let phase = im_part.imul_by_cmpnt_ref_unchecked(self.im_strings.get_elem_ref(op.0));
                self.work
                    .get_coeffs_mut()
                    .imul_elem_unchecked(i, phase * ComplexSign(if op.1 { 3 } else { 1 }));
            }
        }
        assert_eq!(self.work.len(), 1 << ops.len());
    }

    /// Contribute the loaded contents of `self` to real coefficient `op` scaled by `scalar`.
    pub fn contribute_real(&mut self, op: &mut term_set::ViewMut<f64>, scalar: f64) {
        let scalar = scalar / f64::from(self.work.len() as u32);
        for term in self.work.iter() {
            if let Ok(s) = f64::try_represent(term.get_coeff()) {
                lincomb::scaled_iadd_elem(op, term.get_word_iter_ref(), s * scalar);
            }
        }
    }

    /// Given a slice of ops and their coefficients, contribute each to `op`.
    pub fn encode_real(&mut self, fops: &[(Vec<Op>, f64)], op: &mut term_set::ViewMut<f64>) {
        for (ops, coeff) in fops.iter() {
            self.load_product(ops);
            self.contribute_real(op, *coeff);
        }
    }

    /// Contribute the loaded contents of `self` to complex coefficient `op` scaled by `scalar`.
    pub fn contribute_complex(&mut self, op: &mut term_set::ViewMut<Complex64>, scalar: Complex64) {
        let scalar = scalar / f64::from(self.work.len() as u32);
        for term in self.work.iter() {
            lincomb::scaled_iadd_elem(
                op,
                term.get_word_iter_ref(),
                Complex64::represent(term.get_coeff()) * scalar,
            );
        }
    }
}

impl QubitsBased for Operators {
    fn qubits(&self) -> &Qubits {
        self.re_strings.qubits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fermion::mappings::{
        bk::BravyiKitaevMapper, jw::JordanWignerMapper, parity::ParityMapper,
    };

    #[test]
    fn test_jw() {
        let ops = Operators::new::<JordanWignerMapper>(Qubits::from_count(4), None);
        assert_eq!(
            ops.re_strings.to_string(),
            "X0, Z0 X1, Z0 Z1 X2, Z0 Z1 Z2 X3"
        );
        assert_eq!(
            ops.im_strings.to_string(),
            "Y0, Z0 Y1, Z0 Z1 Y2, Z0 Z1 Z2 Y3"
        );
        let mut ops = Operators::new::<JordanWignerMapper>(Qubits::from_count(6), None);
        ops.load_product(Vec::from([(0, true)]).as_slice());
        assert_eq!(ops.work.to_string(), "(+1, X0), (-i, Y0)");
        ops.load_product(Vec::from([(0, false)]).as_slice());
        assert_eq!(ops.work.to_string(), "(+1, X0), (+i, Y0)");
        ops.load_product(Vec::from([(0, true), (0, false)]).as_slice());
        assert_eq!(ops.work.to_string(), "(+1, ), (-1, Z0), (-1, Z0), (+1, )");
        ops.load_product(Vec::from([(0, true), (1, false)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(-i, Y0 X1), (+1, X0 X1), (+1, Y0 Y1), (+i, X0 Y1)"
        );
        ops.load_product(Vec::from([(4, true), (5, false)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(-i, Y4 X5), (+1, X4 X5), (+1, Y4 Y5), (+i, X4 Y5)"
        );
    }

    #[test]
    fn test_bk() {
        let ops = Operators::new::<BravyiKitaevMapper>(Qubits::from_count(4), None);
        assert_eq!(
            ops.re_strings.to_string(),
            "X0 X1 X3, Z0 X1 X3, Z1 X2 X3, Z1 Z2 X3"
        );
        assert_eq!(ops.im_strings.to_string(), "Y0 X1 X3, Y1 X3, Z1 Y2 X3, Y3");

        let mut ops = Operators::new::<BravyiKitaevMapper>(Qubits::from_count(16), None);
        ops.load_product(Vec::from([(2, true), (15, false)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(-i, Z1 X2 X3 Y7 Z11 Z13 Z14), (-1, Z1 Y2 X3 Y7 Z11 Z13 Z14), (-1, Z1 X2 X3 X7 Z15), (+i, Z1 Y2 X3 X7 Z15)"
        );

        ops.load_product(Vec::from([(15, true), (2, false)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(+i, Z1 X2 X3 Y7 Z11 Z13 Z14), (-1, Z1 X2 X3 X7 Z15), (-1, Z1 Y2 X3 Y7 Z11 Z13 Z14), (-i, Z1 Y2 X3 X7 Z15)"
        );

        ops.load_product(Vec::from([(2, false), (15, true)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(-i, Z1 X2 X3 Y7 Z11 Z13 Z14), (+1, Z1 Y2 X3 Y7 Z11 Z13 Z14), (+1, Z1 X2 X3 X7 Z15), (+i, Z1 Y2 X3 X7 Z15)"
        );

        ops.load_product(Vec::from([(15, false), (2, true)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(+i, Z1 X2 X3 Y7 Z11 Z13 Z14), (+1, Z1 X2 X3 X7 Z15), (+1, Z1 Y2 X3 Y7 Z11 Z13 Z14), (-i, Z1 Y2 X3 X7 Z15)"
        );
    }

    #[test]
    fn test_parity() {
        let ops = Operators::new::<ParityMapper>(Qubits::from_count(4), None);
        assert_eq!(
            ops.re_strings.to_string(),
            "X0 X1 X2 X3, Z0 X1 X2 X3, Z1 X2 X3, Z2 X3"
        );
        assert_eq!(
            ops.im_strings.to_string(),
            "Y0 X1 X2 X3, Y1 X2 X3, Y2 X3, Y3"
        );

        let mut ops = Operators::new::<ParityMapper>(Qubits::from_count(6), None);
        ops.load_product(Vec::from([(0, false)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(+1, X0 X1 X2 X3 X4 X5), (+i, Y0 X1 X2 X3 X4 X5)"
        );

        ops.load_product(Vec::from([(0, true), (1, false)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(-i, Y0), (+1, X0), (-1, X0 Z1), (+i, Y0 Z1)"
        );

        ops.load_product(Vec::from([(4, true), (5, false)]).as_slice());
        assert_eq!(
            ops.work.to_string(),
            "(-i, Z3 Y4), (+1, X4), (-1, Z3 X4 Z5), (+i, Y4 Z5)"
        );
    }
}
