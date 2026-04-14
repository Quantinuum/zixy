//! Extends `CmpntList` with a vector of associated coefficients.

use num_complex::Complex64;

use crate::cmpnt::parse::ParseError;
use crate::cmpnt::springs::ModeSettings;
use crate::container::coeffs::traits::{
    ComplexSigned, HasCoeffsMut, IMulResult, NumRepr, NumReprVec, Signed,
};
use crate::container::traits::proj::{self, AsRef};
use crate::container::traits::{Elements, EmptyFrom, MutRefElements};
use crate::container::utils::DistinctPair;
use crate::container::word_iters::{terms, WordIters};
use crate::qubit::clifford;
use crate::qubit::mode::{pauli_matrix_product, PauliMatrix, Qubits};
use crate::qubit::pauli::cmpnt_major::cmpnt_list::{CmpntList, CmpntRef};
use crate::qubit::pauli::springs::Springs;
use crate::qubit::traits::{
    DifferentQubits, PauliWordMutRef, PauliWordRef, PushPaulis, QubitsBased, QubitsRelabel,
    QubitsStandardize, QubitsStandardized,
};

/// Stores one coeff for each component of a `CmpntList`.
pub type Terms<C /*: NumRepr*/> = terms::Terms<CmpntList, C>;
pub type View<'a, C /*: NumRepr*/> = terms::View<'a, CmpntList, C>;
pub type ViewMut<'a, C /*: NumRepr*/> = terms::ViewMut<'a, CmpntList, C>;

/// Trait for structs that immutably view a [`Terms`].
pub trait AsView<C: NumRepr>: terms::AsView<CmpntList, C> + proj::ToOwned {
    /// Split `self` into the centralizer set and the remainder of the terms.
    fn centralizer_and_remainder(&self) -> (Terms<C>, Terms<C>) {
        self.bipartition(self.view().word_iters.centralizer_members())
    }
}

/// Trait for structs that mutably view a [`Terms`].
pub trait AsViewMut<C: NumRepr>: terms::AsViewMut<CmpntList, C> {
    /// Multiply the element at i_lhs in-place by the one at i_rhs.
    fn imul(&mut self, i_lhs: usize, i_rhs: usize)
    where
        C: ComplexSigned,
    {
        let mut self_mut_ref = self.view_mut();
        match DistinctPair::new(i_lhs, i_rhs) {
            Some(inds) => {
                let (mut lhs, rhs) = self_mut_ref.get_semi_mut_refs(inds);
                lhs.imul_unchecked(rhs);
            }
            None => {
                // square the i_lhs element in-place i.e. clear the matrices to identity and double the phase
                self_mut_ref.word_iters.clear_elem(i_lhs);
                self_mut_ref.get_coeffs_mut().square_unchecked(i_lhs);
            }
        }
    }

    /// Conjugate self by gate, updating phase accordingly
    fn conj_clifford(&mut self, gate: clifford::Gate)
    where
        C: Signed,
    {
        let self_mut_ref = self.view_mut();
        for i in 0..self_mut_ref.len() {
            self_mut_ref
                .word_iters
                .get_elem_mut_ref(i)
                .conj_clifford(gate);
        }
    }

    /// Conjugate self by each gate in turn, updating phase accordingly
    /// Return whether there were any factors of i that could not be absorbed into the phase container element.
    fn conj_clifford_vec(&mut self, gates: Vec<clifford::Gate>)
    where
        C: Signed,
    {
        gates.into_iter().for_each(|gate| self.conj_clifford(gate));
    }
}

impl<C: NumRepr> AsView<C> for Terms<C> {}
impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}

impl<C: NumRepr> AsViewMut<C> for Terms<C> {}
impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}

impl<C: NumRepr> Terms<C> {
    /// Create a new list of Pauli strings on the given space of qubits.
    pub fn new(qubits: Qubits) -> Self {
        Self::empty_from(&CmpntList::new(qubits))
    }

    /// Create from the given sparse strings, absorbing any phases into the coefficient if representable, else return error.
    pub fn from_springs(qubits: Qubits, springs: &Springs) -> Result<Self, ParseError> {
        let pair = CmpntList::from_springs(qubits, springs)?;
        let coeffs = C::Vector::try_represent(&pair.1)?;
        Ok(Self::from((pair.0, coeffs)))
    }

    /// Create from the given sparse strings, absorbing any phases into the coefficient if representable, else return error.
    /// Infer a Count-type qubit space from the springs object.
    pub fn from_springs_default(springs: &Springs) -> Result<Self, ParseError> {
        let n_qubit = springs.get_mode_inds().default_n_mode();
        Self::from_springs(Qubits::from_count(n_qubit as usize), springs)
    }

    /// Create from the given sparse strings and coeff vector, absorbing any phases into the coefficient if representable, else return error.
    /// If `coeffs` is shorter than `springs`, it is padded to the length of `springs`` before attempting to absorb phases.
    /// Else if `springs` is shorter than `coeffs`, it is padded to the length of `coeffs` with empty strings.
    pub fn from_springs_coeffs(
        qubits: Qubits,
        mut springs: Springs,
        mut coeffs: C::Vector,
    ) -> Result<Self, ParseError> {
        if coeffs.len() < springs.len() {
            coeffs.resize_with_units(springs.len());
        }
        springs.append_empty(springs.len().saturating_sub(coeffs.len()));
        let pair = CmpntList::from_springs(qubits, &springs)?;
        let phase_coeffs = C::Vector::try_represent(&pair.1)?;
        Ok(Self::from((pair.0, coeffs.mul_elemwise(&phase_coeffs))))
    }

    /// Create from the given sparse strings and coeff vector, absorbing any phases into the coefficient if representable, else return error.
    /// Infer a Count-type qubit space from the springs object.
    pub fn from_springs_coeffs_default(
        springs: Springs,
        coeffs: C::Vector,
    ) -> Result<Self, ParseError> {
        let n_qubit = springs.get_mode_inds().default_n_mode();
        Self::from_springs_coeffs(Qubits::from_count(n_qubit as usize), springs, coeffs)
    }
}

pub type TermRef<'a, C /*: NumRepr*/> = terms::TermRef<'a, CmpntList, C>;
pub type TermMutRef<'a, C /*: NumRepr*/> = terms::TermMutRef<'a, CmpntList, C>;

impl<'a, C: NumRepr> TermRef<'a, C> {
    /// Call the to_sparse method of the `PauliWordRef`, and return the result scaled by the coeff of self.
    pub fn to_sparse_matrix(&self, big_endian: bool) -> sprs::CsMat<Complex64> {
        let mut matrix = self.get_word_iter_ref().to_sparse_matrix(big_endian);
        matrix.scale(self.get_coeff().to_complex());
        matrix
    }
}

impl<'a, C: NumRepr> QubitsBased for TermRef<'a, C> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

impl<'a, C: NumRepr> TermMutRef<'a, C> {
    /// Try to set the value of this Pauli term from the multiplication of two Pauli strings.
    /// If the phase could not be absorbed, it is returned in the Failure arm of `IMulResult`.
    pub fn try_assign_mul_cmpnt_refs_unchecked(
        &mut self,
        lhs: CmpntRef,
        rhs: CmpntRef,
    ) -> IMulResult {
        self.get_word_iter_mut_ref()
            .assign_mul_matrices_unchecked(lhs.clone(), rhs.clone());
        self.try_imul_coeff(lhs.phase_of_mul(rhs))
    }

    /// Try to set the value of this Pauli term from the multiplication of two Pauli strings.
    /// If the phase could not be absorbed, it is returned in the Failure arm of `IMulResult`.
    /// If the qubits on which `self`, `lhs`, and `rhs` are based are not all the same, return `DifferentQubits` error.
    pub fn try_assign_mul_cmpnt_refs(
        &mut self,
        lhs: CmpntRef,
        rhs: CmpntRef,
    ) -> Result<IMulResult, DifferentQubits> {
        DifferentQubits::check_transitive(self, &lhs, &rhs)?;
        Ok(self.try_assign_mul_cmpnt_refs_unchecked(lhs, rhs))
    }

    /// `Set` the value of this Pauli term from the multiplication of two Pauli strings.
    /// This is restricted to coefficient types that can definitely absorb factors of the imag unit.
    /// For other coeff types, try_assign_mul_cmpnt_refs should be used instead.
    pub fn assign_mul_cmpnt_refs_unchecked(&mut self, lhs: CmpntRef, rhs: CmpntRef)
    where
        C: ComplexSigned,
    {
        self.get_word_iter_mut_ref()
            .assign_mul_matrices_unchecked(lhs.clone(), rhs.clone());
        self.imul_complex_sign(lhs.phase_of_mul(rhs))
    }

    /// `Set` the value of this Pauli term from the multiplication of two Pauli strings.
    /// This is restricted to coefficient types that can definitely absorb factors of the imag unit.
    /// For other coeff types, try_assign_mul_cmpnt_refs should be used instead.
    /// If the qubits on which `self`, `lhs`, and `rhs` are based are not all the same, return `DifferentQubits` error.
    pub fn assign_mul_cmpnt_refs(
        &mut self,
        lhs: CmpntRef,
        rhs: CmpntRef,
    ) -> Result<(), DifferentQubits>
    where
        C: ComplexSigned,
    {
        DifferentQubits::check_transitive(self, &lhs, &rhs)?;
        self.assign_mul_cmpnt_refs_unchecked(lhs, rhs);
        Ok(())
    }

    /// Try to set the value of this Pauli term from the multiplication of two others.
    /// The coeff of self is first set to the product of lhs and rhs coeffs since all NumReprs are closed under mul.
    /// If the phase cannot be absorbed in the coeff, it is returned in the Failure arm of `IMulResult`.
    pub fn try_assign_mul_unchecked(&mut self, lhs: TermRef<C>, rhs: TermRef<C>) -> IMulResult {
        self.get_word_iter_mut_ref().assign_mul_matrices_unchecked(
            lhs.get_word_iter_ref().clone(),
            rhs.get_word_iter_ref().clone(),
        );
        self.set_coeff(lhs.get_coeff() * rhs.get_coeff());
        self.try_imul_coeff(
            lhs.get_word_iter_ref()
                .phase_of_mul(rhs.get_word_iter_ref()),
        )
    }

    /// Try to set the value of this Pauli term from the multiplication of two others.
    /// The coeff of self is first set to the product of lhs and rhs coeffs since all NumReprs are closed under mul.
    /// If the phase cannot be absorbed in the coeff, it is returned in the Failure arm of `IMulResult`.
    /// If the qubits on which `self`, `lhs`, and `rhs` are based are not all the same, return `DifferentQubits` error.
    pub fn try_assign_mul(
        &mut self,
        lhs: TermRef<C>,
        rhs: TermRef<C>,
    ) -> Result<IMulResult, DifferentQubits> {
        DifferentQubits::check_transitive(self, &lhs, &rhs)?;
        Ok(self.try_assign_mul_unchecked(lhs, rhs))
    }

    /// `Set` the value of this Pauli term from the multiplication of two others.
    /// The coeff of self is set to the product of lhs and rhs coeffs multiplied by the phase factor.
    pub fn assign_mul_unchecked(&mut self, lhs: TermRef<C>, rhs: TermRef<C>)
    where
        C: ComplexSigned,
    {
        self.get_word_iter_mut_ref().assign_mul_matrices_unchecked(
            lhs.get_word_iter_ref().clone(),
            rhs.get_word_iter_ref().clone(),
        );
        self.set_coeff(lhs.get_coeff() * rhs.get_coeff());
        self.imul_complex_sign(
            lhs.get_word_iter_ref()
                .phase_of_mul(rhs.get_word_iter_ref()),
        );
    }

    /// `Set` the value of this Pauli term from the multiplication of two others.
    /// The coeff of self is set to the product of lhs and rhs coeffs multiplied by the phase factor.
    /// If the qubits on which `self`, `lhs`, and `rhs` are based are not all the same, return `DifferentQubits` error.
    pub fn assign_mul(&mut self, lhs: TermRef<C>, rhs: TermRef<C>) -> Result<(), DifferentQubits>
    where
        C: ComplexSigned,
    {
        DifferentQubits::check_transitive(self, &lhs, &rhs)?;
        self.assign_mul_unchecked(lhs, rhs);
        Ok(())
    }

    /// Try to right-multiply self by a single pauli matrix.
    /// If the phase could not be absorbed, it is returned in the Failure arm of `IMulResult`.
    pub fn try_imul_by_single_pauli(&mut self, i_qubit: usize, p: PauliMatrix) -> IMulResult {
        let (p, phase) = pauli_matrix_product(
            self.as_ref()
                .get_word_iter_ref()
                .get_pauli_unchecked(i_qubit),
            p,
        );
        self.get_word_iter_mut_ref().set_pauli_unchecked(i_qubit, p);
        self.try_imul_coeff(phase)
    }

    /// Right-multiply self by a single pauli matrix.
    pub fn imul_by_single_pauli(&mut self, i_qubit: usize, p: PauliMatrix) {
        let _ = self.try_imul_by_single_pauli(i_qubit, p);
    }

    /// Try to right-multiply self by a Pauli string.
    pub fn try_imul_by_cmpnt_ref_unchecked(&mut self, rhs: CmpntRef) -> IMulResult {
        let phase = self.as_ref().get_word_iter_ref().phase_of_mul(rhs.clone());
        self.get_word_iter_mut_ref()
            .imul_by_cmpnt_ref_matrices_unchecked(rhs);
        self.try_imul_coeff(phase)
    }

    /// Try to right-multiply self by a Pauli string.
    /// If the qubits on which `self` and `rhs` are based are not the same, return `DifferentQubits` error.
    pub fn try_imul_by_cmpnt_ref(&mut self, rhs: CmpntRef) -> Result<IMulResult, DifferentQubits> {
        DifferentQubits::check(self, &rhs)?;
        Ok(self.try_imul_by_cmpnt_ref_unchecked(rhs))
    }

    /// Right-multiply self by a Pauli string.
    pub fn imul_by_cmpnt_ref_unchecked(&mut self, rhs: CmpntRef) {
        let _ = self.try_imul_by_cmpnt_ref(rhs);
    }

    /// Right-multiply self by a Pauli string.
    /// If the qubits on which `self` and `rhs` are based are not the same, return `DifferentQubits` error.
    pub fn imul_by_cmpnt_ref(&mut self, rhs: CmpntRef) -> Result<(), DifferentQubits> {
        DifferentQubits::check(self, &rhs)?;
        self.imul_by_cmpnt_ref_unchecked(rhs);
        Ok(())
    }

    /// Try to right-multiply self by another term of the same type.
    pub fn try_imul_unchecked(&mut self, rhs: TermRef<C>) -> IMulResult {
        self.imul_coeff(rhs.get_coeff());
        self.try_imul_by_cmpnt_ref_unchecked(rhs.get_word_iter_ref())
    }

    /// Try to right-multiply self by another term of the same type.
    /// If the qubits on which `self` and `rhs` are based are not the same, return `DifferentQubits` error.
    pub fn try_imul(&mut self, rhs: TermRef<C>) -> Result<IMulResult, DifferentQubits> {
        DifferentQubits::check(self, &rhs)?;
        Ok(self.try_imul_unchecked(rhs))
    }

    /// Right-multiply self by another term of the same type.
    pub fn imul_unchecked(&mut self, rhs: TermRef<C>) {
        self.imul_coeff(rhs.get_coeff());
        self.imul_by_cmpnt_ref_unchecked(rhs.get_word_iter_ref());
    }

    /// Right-multiply self by another term of the same type.
    /// If the qubits on which `self` and `rhs` are based are not the same, return `DifferentQubits` error.
    pub fn imul(&mut self, rhs: TermRef<C>) -> Result<(), DifferentQubits> {
        DifferentQubits::check(self, &rhs)?;
        self.imul_unchecked(rhs);
        Ok(())
    }

    /// Conjugate self by gate, updating phase accordingly
    pub fn conj_clifford(&mut self, gate: clifford::Gate)
    where
        C: Signed,
    {
        let phase = self.get_word_iter_mut_ref().conj_clifford(gate);
        let c = self.as_ref().get_coeff();
        self.set_coeff(if phase.0 { -c } else { c })
    }

    /// Conjugate self by each gate in turn, updating phase accordingly
    /// Return whether there were any factors of i that could not be absorbed into the phase container element.
    pub fn conj_clifford_vec(&mut self, gates: Vec<clifford::Gate>)
    where
        C: Signed,
    {
        gates.into_iter().for_each(|gate| self.conj_clifford(gate));
    }
}

impl<'a, C: NumRepr> QubitsBased for TermMutRef<'a, C> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

impl<'a, C: NumRepr> QubitsBased for View<'a, C> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

impl<C: NumRepr> QubitsBased for Terms<C> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

impl<C: NumRepr> QubitsStandardize for Terms<C> {
    fn general_standardize(&mut self, n_qubit: usize) {
        self.word_iters.general_standardize(n_qubit);
    }
}

impl<'a, C: NumRepr> QubitsStandardized for View<'a, C> {}

impl<'a, C: NumRepr> QubitsBased for ViewMut<'a, C> {
    fn qubits(&self) -> &Qubits {
        self.word_iters.qubits()
    }
}

impl<'a, C: NumRepr> QubitsStandardize for ViewMut<'a, C> {
    fn general_standardize(&mut self, n_qubit: usize) {
        self.word_iters.general_standardize(n_qubit);
    }
}

impl<'a, C: NumRepr> QubitsStandardized for ViewMut<'a, C> {}

impl<'a, C: NumRepr> QubitsRelabel for ViewMut<'a, C> {
    fn qubits_mut(&mut self) -> &mut Qubits {
        self.word_iters.qubits_mut()
    }
}

impl<C: NumRepr> PushPaulis for Terms<C> {
    fn push_pauli_iter(
        &mut self,
        iter: impl Iterator<Item = (usize, PauliMatrix)>,
    ) -> Result<(), crate::container::errors::OutOfBounds> {
        self.word_iters.push_pauli_iter(iter)?;
        self.get_coeffs_mut().push_default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::container::coeffs::complex_sign::ComplexSign;
    use crate::container::coeffs::sign::Sign;
    use crate::container::coeffs::unity::Unity;
    use crate::container::traits::Elements;
    use crate::qubit::mode::{PauliMatrix, Qubits};
    use crate::qubit::pauli::cmpnt_major::cmpnt::PauliWord;
    use rstest::rstest;
    use PauliMatrix::*;

    #[test]
    fn test_empty() {
        {
            let list = Terms::<Unity>::new(Qubits::from_count(4));
            assert!(list.is_empty());
        }
    }

    #[test]
    fn test_to_string() {
        let mut list = Terms::<Unity>::new(Qubits::from_count(6));
        list.push_pauli_identity();
        assert_eq!(list.to_string(), "");
        list.push_pauli_identity();
        assert_eq!(list.to_string(), ", ");
        assert!(list.push_pauli_vec(vec![X, Y, Z, Z, Y, X]).is_ok());
        assert_eq!(list.to_string(), ", , X0 Y1 Z2 Z3 Y4 X5");
        let mut list = Terms::<Sign>::new(Qubits::from_count(6));
        list.push_pauli_identity();
        assert_eq!(list.to_string(), "(+1, )");
        list.push_pauli_identity();
        assert_eq!(list.to_string(), "(+1, ), (+1, )");
        assert!(list.push_pauli_vec(vec![X, Y, Z, Z, Y, X]).is_ok());
        assert_eq!(list.to_string(), "(+1, ), (+1, ), (+1, X0 Y1 Z2 Z3 Y4 X5)");
        list.get_coeffs_mut().imul_elem_unchecked(1, Sign(true));
        assert_eq!(list.to_string(), "(+1, ), (-1, ), (+1, X0 Y1 Z2 Z3 Y4 X5)");
        let mut list = Terms::<ComplexSign>::new(Qubits::from_count(6));
        assert!(list.push_pauli_map(HashMap::from([(0, X)])).is_ok());
        assert_eq!(list.to_string(), "(+1, X0)");
        list.get_coeffs_mut().imul_elem_unchecked(0, ComplexSign(1));
        assert_eq!(list.to_string(), "(+i, X0)");
        list.get_coeffs_mut().imul_elem_unchecked(0, ComplexSign(1));
        assert_eq!(list.to_string(), "(-1, X0)");
        list.get_coeffs_mut().imul_elem_unchecked(0, ComplexSign(1));
        assert_eq!(list.to_string(), "(-i, X0)");
        list.get_coeffs_mut().imul_elem_unchecked(0, ComplexSign(1));
        assert_eq!(list.to_string(), "(+1, X0)");
    }

    #[rstest]
    #[case(vec![X, Y, I, Z], vec![I, Z, I, Z], vec![X, X, I, I], ComplexSign(1))]
    #[case(vec![X, Y, I, Y], vec![I, Z, I, Z], vec![X, X, I, X], ComplexSign(2))]
    #[case(
        vec![I, Y, X, I, I, Y, Z, I],
        vec![Y, I, Y, Z, X, Y, Z, Z],
        vec![Y, Y, Z, Z, X, I, I, Z],
        ComplexSign(1)
    )]
    #[case(
        vec![Y, X, I, Y, Z, X, Z, X, I, X, Z, X, Y, Y, X, X, I, X, I, Y, Y, I, I, Z, X, Z, X, X, I, Z, X, X],
        vec![X, Z, Y, Z, Y, X, Z, Z, X, X, Z, X, Y, Y, Y, Z, Z, Y, I, Z, I, X, Z, Y, X, Z, Y, I, X, X, I, I],
        vec![Z, Y, Y, X, X, I, I, Y, X, I, I, I, I, I, Z, Y, Z, Z, I, X, Y, X, Z, X, I, I, Z, X, X, Y, X, X],
        ComplexSign(0)
    )]
    #[case(
        vec![Y, X, X, X, X, X, Z, X, Z, I, I, I, Z, Z, I, I, Z, Y, X, Y, I, I, X, Y, X, Y, I, X, Z, Y, X, I, Y, Z, I, I],
        vec![Z, Y, Y, Y, Y, Y, Z, I, Y, Y, I, I, Z, Z, Z, I, Z, X, Y, X, Y, Y, Y, I, Y, X, I, Y, X, X, Y, I, X, Y, X, X],
        vec![X, Z, Z, Z, Z, Z, I, X, X, Y, I, I, I, I, Z, I, I, Z, Z, Z, Y, Y, Z, Y, Z, Z, I, Z, Y, Z, Z, I, Z, X, X, X],
        ComplexSign(1)
    )]
    fn test_mul(
        #[case] lv: Vec<PauliMatrix>,
        #[case] rv: Vec<PauliMatrix>,
        #[case] ov: Vec<PauliMatrix>,
        #[case] phase: ComplexSign,
    ) {
        use crate::container::traits::{proj::Borrow, RefElements};

        let lhs = PauliWord::from_vec_default(lv);
        assert_eq!(lhs.borrow().get_x_part_slice().len(), 1);
        assert_eq!(lhs.borrow().get_z_part_slice().len(), 1);
        let rhs = PauliWord::from_vec_default(rv);
        assert_eq!(rhs.borrow().get_x_part_slice().len(), 1);
        assert_eq!(rhs.borrow().get_z_part_slice().len(), 1);
        let mut out = Terms::<ComplexSign>::new(lhs.to_qubits());
        out.push_pauli_identity();
        assert_eq!(out.word_iters.u64it_size(), 2);
        assert_eq!(out.len(), 1);
        assert_eq!(out.word_iters.elem_u64it(0).count(), 2);
        out.get_elem_mut_ref(0)
            .assign_mul_cmpnt_refs_unchecked(lhs.borrow(), rhs.borrow());
        assert_eq!(out.get_elem_ref(0).get_word_iter_ref().get_pauli_vec(), ov);
        assert_eq!(out.get_elem_ref(0).get_coeff(), phase);
    }
}
