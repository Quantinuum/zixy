//! Qubit state-specific linear combination utilities.

use crate::container::bit_matrix::AsBitMatrix;
use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::proj::BorrowMut;
use crate::container::traits::RefElements;
use crate::container::word_iters;
use crate::container::word_iters::term_set::AsViewMut;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::cmpnt_major::encoding::invert_endian;
use crate::qubit::state::{term_set, terms};

pub fn l2_norm_square<C: FieldElem>(state: &impl terms::AsView<C>) -> f64 {
    state.view().coeffs.iter().map(|c| c.magnitude_sq()).sum()
}

pub fn l2_norm<C: FieldElem>(state: &impl terms::AsView<C>) -> f64 {
    l2_norm_square(state).sqrt()
}

/// Take the inner product of a basis state linear combination with another.
pub fn vdot<C: FieldElem>(lhs: &impl term_set::AsView<C>, rhs: &impl terms::AsView<C>) -> C {
    rhs.view()
        .iter()
        .map(
            |rhs| match lhs.lookup_coeff_elem_ref(rhs.get_word_iter_ref()) {
                Some(lhs) => lhs.complex_conj() * rhs.get_coeff(),
                None => C::ZERO,
            },
        )
        .sum()
}

/// If big_endian is true, the bit associated with mode 0 is the most significant in the index integer
/// Else, the bit associated with mode n_qubit - 1 is the most significant
pub fn to_dense<C: FieldElem>(state: &impl terms::AsView<C>, big_endian: bool) -> Vec<C> {
    let state_ref = state.view();
    let n = state_ref.word_iters.n_bit();
    if n >= 64 {
        panic!("too many qubits to convert to sparse.");
    }
    let mut out: Vec<C> = vec![C::ZERO; 1 << n];
    for term in state_ref.iter() {
        let ind = term.get_word_iter_ref().get_u64it().next().unwrap_or(0);
        let ind = if big_endian {
            invert_endian(ind, n)
        } else {
            ind
        };
        out[ind as usize] = term.get_coeff();
    }
    out
}

/// Create a state linear combination from a dense array slice of coefficients.
pub fn assign_from_dense<C: FieldElem>(
    out: &mut term_set::ViewMut<C>,
    source: &[C],
    big_endian: bool,
) {
    let n = out.word_iters.n_bit();
    let n_take = source.len().min(1 << n);
    out.clear();
    for (i, c) in source.iter().take(n_take).enumerate() {
        if *c == C::ZERO {
            continue;
        }
        word_iters::lincomb::scaled_iadd_u64it(
            out,
            std::iter::once(if big_endian {
                invert_endian(i as u64, n)
            } else {
                i as u64
            }),
            *c,
        );
    }
}

/// Create a state linear combination from a dense array slice of coefficients.
pub fn from_dense<C: FieldElem>(
    qubits: Qubits,
    source: &[C],
    big_endian: bool,
) -> term_set::TermSet<C> {
    let mut out: word_iters::term_set::TermSet<super::cmpnt_list::CmpntList, C> =
        term_set::TermSet::<C>::new(qubits);
    assign_from_dense(&mut out.borrow_mut(), source, big_endian);
    out
}

/*
impl<C: FieldElem> Sum<C> {
    /// Create an empty set of states on the given space of qubits.
    pub fn new(qubits: Qubits) -> Self {
        Self::empty_from(&CmpntList::new(qubits))
    }

    /// Add a contribution of c * basis vector to the linear combination by converting the vector of bits to a temporary `BasisState`.
    pub fn add_from_vec(&mut self, bits: Vec<bool>, c: C) -> Result<(), ModeOutOfBounds> {
        self.insert_or_add_elem_ref(BasisState::from_vec(self.to_qubits(), bits)?.borrow(), c);
        Ok(())
    }

    /// Add a contribution of c * basis vector to the linear combination by converting the hashset of set bits to a temporary `BasisState`.
    pub fn add_from_set(
        &mut self,
        set_bits: HashSet<usize>,
        c: C,
    ) -> Result<(), ModeOutOfBounds> {
        self.insert_or_add_elem_ref(
            BasisState::from_set(self.to_qubits(), set_bits)?.borrow(),
            c,
        );
        Ok(())
    }

    /// Get the coefficient of a computational basis vector in the linear combination by converting the vector of bit values to a temporary `BasisState`.
    pub fn lookup_coeff_vec(&self, bits: Vec<bool>) -> Result<Option<C>, ModeOutOfBounds> {
        Ok(self.lookup_coeff_elem_ref(BasisState::from_vec(self.to_qubits(), bits)?.borrow()))
    }

    /// Create a new linear combination on a given qubit space from some springs and coefficients.
    pub fn from_springs_coeffs(
        qubits: Qubits,
        springs: &BinarySprings,
        coeffs: &[C],
    ) -> Result<Self, ModeOutOfBounds> {
        let mut this = Self::new(qubits);
        let mut work = BasisState::new(this.to_qubits());
        let n = coeffs.len().min(springs.len());
        for (i, c) in coeffs.iter().enumerate().take(n) {
            work.borrow_mut().clear();
            for (setting, i_mode) in springs.get_iter(i) {
                work.borrow_mut().set_mode(i_mode, setting != 0)?;
            }
            this.insert_or_add_elem_ref(work.borrow(), *c);
        }
        Ok(this)
    }

    /// Create a new linear combination from some springs and coefficients, inferring a default qubit space.
    pub fn from_springs_coeffs_default(springs: &BinarySprings, coeffs: &[C]) -> Self {
        let qubits = Qubits::from_count(springs.get_mode_inds().default_n_mode());
        Self::from_springs_coeffs(qubits, springs, coeffs).unwrap()
    }

    /// Create a new linear combination from some pairs of vectors of bit values and coefficients.
    pub fn from_vec_coeff_pairs(
        qubits: Qubits,
        pairs: Vec<(Vec<bool>, C)>,
    ) -> Result<Self, ModeOutOfBounds> {
        let mut this = Self::new(qubits);
        for (bits, c) in pairs {
            this.add_from_vec(bits, c)?;
        }
        Ok(this)
    }

    /// Create a new linear combination from some pairs of vectors of bit values and coefficients.
    /// assuming an inferred default qubit space.
    pub fn from_vec_coeff_pairs_default(pairs: Vec<(Vec<bool>, C)>) -> Self {
        let n_qubit = pairs.iter().map(|(v, _)| v.len()).max().unwrap_or_default();
        Self::from_vec_coeff_pairs(Qubits::from_count(n_qubit), pairs).unwrap()
    }

    /// Create a new linear combination from some pairs of hashsets of set bit positions and coefficients.
    pub fn from_set_coeff_pairs(
        qubits: Qubits,
        pairs: Vec<(HashSet<usize>, C)>,
    ) -> Result<Self, ModeOutOfBounds> {
        let mut this = Self::new(qubits);
        for (bits, c) in pairs {
            this.add_from_set(bits, c)?;
        }
        Ok(this)
    }

    /// Create a new linear combination from some pairs of hashsets of set bit positions and coefficients,
    /// assuming an inferred default qubit space.
    pub fn from_set_coeff_pairs_default(pairs: Vec<(HashSet<usize>, C)>) -> Self {
        let i_qubit_max = pairs.iter().map(|(map, _)| map.iter().max()).max();
        let n_qubit = match i_qubit_max {
            Some(Some(x)) => *x,
            _ => 0,
        };
        Self::from_set_coeff_pairs(Qubits::from_count(n_qubit), pairs).unwrap()
    }

    /// Create a new linear combination of a single computational basis vector with a unit coefficient.
    pub fn from_cmpnt_ref(cmpnt_ref: CmpntRef) -> Self {
        let mut this = Self::new(cmpnt_ref.to_qubits());
        this.insert_or_add_elem_ref(cmpnt_ref, C::ONE);
        this
    }

    /// Take the inner product of this basis state linear combination with another.
    pub fn vdot(&self, other: &Self) -> C {
        sum_ops::vdot(&Refs::from(&self.0), &Refs::from(&other.0))
    }

    /// Return Some with the Hamming weight if all terms have the same Hamming weight, else return None
    pub fn hamming_weight(&self) -> Option<usize> {
        self.get_word_iters().hamming_weight()
    }

    /// If big_endian is true, the bit associated with mode 0 is the most significant in the index integer
    /// Else, the bit associated with mode n_qubit - 1 is the most significant
    pub fn to_dense(&self, big_endian: bool) -> Vec<C> {
        sum_ops::to_dense(&Refs::from(&self.0), big_endian)
    }

    /// Create a state linear combination from a dense array slice of coefficients.
    pub fn from_dense(qubits: Qubits, coeffs: &[C], big_endian: bool) -> Self {
        let mut out = Self::new(qubits);
        let mut mut_refs = MutRefs::from(&mut out.0);
        sum_ops::from_dense(&mut mut_refs, coeffs, big_endian);
        out
    }

    /// Create a state linear combination from a dense array slice of coefficients. Inferring the
    /// number of qubits from the dimension of the coefficient slice.
    pub fn from_dense_default(coeffs: &[C], big_endian: bool) -> Self {
        Self::from_dense(
            Qubits::from_hilbert_space_dim(coeffs.len()),
            coeffs,
            big_endian,
        )
    }

    /// Get the coefficient and reference to the component with the largest coefficient magnitude.
    pub fn dominant_term(&self) -> Option<TermRef<'_, C>> {
        let tmp = sum_ops::dominant_term::<C>(&self.0 .0 .0, &self.0 .0 .1);
        tmp.map(|(elem_ref, _)| self.get_elem_ref(elem_ref.get_index()))
    }

    /// Get the square of the L2 norm, i.e. sum of the squares of all coefficients.
    pub fn l2_norm_sq(&self) -> f64 {
        self.get_coeffs().iter().map(C::magnitude_sq).sum::<f64>()
    }

    /// Get the L2 norm.
    pub fn l2_norm(&self) -> f64 {
        self.l2_norm_sq().sqrt()
    }

    /// Scale the state sum such that the L2 norm is 1.
    pub fn l2_normalize(&mut self) {
        let norm = C::from_real(1.0 / self.l2_norm());
        self.scale(norm);
    }
}

impl<C: FieldElem> QubitsBased for Sum<C> {
    fn qubits(&self) -> &Qubits {
        self.get_word_iters().qubits()
    }
}

impl<C: FieldElem> QubitsRelabel for Sum<C> {
    fn qubits_mut(&mut self) -> &mut Qubits {
        self.get_word_iters_mut().qubits_mut()
    }

    fn general_standardized(&self, n_qubit: usize) -> Self {
        Self::from(self.0 .0.general_standardized(n_qubit))
    }
}
*/
