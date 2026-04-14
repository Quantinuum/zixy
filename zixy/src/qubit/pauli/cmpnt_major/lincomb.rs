//! Pauli operator-specifict Linear-combination utilities.

use std::collections::HashMap;

use indexmap::IndexSet;
use num_complex::Complex64;

use crate::cmpnt::parse::ParseError;
use crate::cmpnt::springs::ModeSettings;
use crate::container::coeffs::complex_sign::ComplexSign;
use crate::container::coeffs::traits::{
    ComplexSigned, FieldElem, FieldElemVec, HasCoeffs, NumRepr,
};
use crate::container::errors::OutOfBounds;
use crate::container::traits::proj::{Borrow, BorrowMut};
use crate::container::traits::{Elements, RefElements};
use crate::container::two_bit_vec::ODD_BIT_MASK;
use crate::container::word_iters::lincomb::{iadd, isub, scaled_iadd_elem};
use crate::container::word_iters::term_set::AsViewMut;
use crate::qubit::mode::{PauliMatrix, Qubits};
use crate::qubit::pauli::cmpnt_major::cmpnt::PauliWord;
use crate::qubit::pauli::cmpnt_major::encoding::invert_endian;
use crate::qubit::pauli::cmpnt_major::num_ops::{num_op, num_op_odd_pos};
use crate::qubit::pauli::cmpnt_major::term_set::{self, TermSet};
use crate::qubit::pauli::cmpnt_major::terms::{self, Terms};
use crate::qubit::pauli::springs::Springs;
use crate::qubit::traits::{DifferentQubits, PauliWordMutRef, QubitsBased};

pub fn assign_from_mul<C: FieldElem>(
    out: &mut term_set::ViewMut<Complex64>,
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<(), DifferentQubits> {
    DifferentQubits::check_transitive(out.word_iters, lhs.word_iters, rhs.word_iters)?;
    out.clear();
    let qubits = lhs.word_iters.to_qubits();
    let mut work = PauliWord::new(qubits.clone());
    let n_lhs = lhs.word_iters.len().min(lhs.coeffs.len());
    let n_rhs = rhs.word_iters.len().min(rhs.coeffs.len());
    for (i_lhs, lhs_coeff) in lhs.coeffs.iter().take(n_lhs).enumerate() {
        let lhs_cmpnt = lhs.word_iters.get_elem_ref(i_lhs);
        for (i_rhs, rhs_coeff) in rhs.coeffs.iter().take(n_rhs).enumerate() {
            let rhs_cmpnt = rhs.word_iters.get_elem_ref(i_rhs);
            let phase = work
                .borrow_mut()
                .assign_mul_unchecked(lhs_cmpnt.clone(), rhs_cmpnt);
            let c = phase.to_complex();
            let c = lhs_coeff.scaled_complex(c);
            let c = rhs_coeff.scaled_complex(c);
            scaled_iadd_elem(out, work.borrow(), c);
        }
    }
    Ok(())
}

pub fn assign_from_commutator<C: FieldElem>(
    out: &mut term_set::ViewMut<Complex64>,
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<(), DifferentQubits> {
    assign_from_mul(out, lhs, rhs)?;
    let tmp = mul(rhs, lhs)?.terms;
    isub(out, &tmp.borrow());
    Ok(())
}

pub fn assign_from_anticommutator<C: FieldElem>(
    out: &mut term_set::ViewMut<Complex64>,
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<(), DifferentQubits> {
    assign_from_mul(out, lhs, rhs)?;
    let tmp = mul(rhs, lhs)?.terms;
    iadd(out, &tmp.borrow());
    Ok(())
}

pub fn mul<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<TermSet<Complex64>, DifferentQubits> {
    let mut out = TermSet::<Complex64>::new(lhs.to_qubits());
    assign_from_mul(&mut out.borrow_mut(), lhs, rhs)?;
    Ok(out)
}

pub fn commutator<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<TermSet<Complex64>, DifferentQubits> {
    let mut out = TermSet::<Complex64>::new(lhs.to_qubits());
    assign_from_commutator(&mut out.borrow_mut(), lhs, rhs)?;
    Ok(out)
}

pub fn anticommutator<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<TermSet<Complex64>, DifferentQubits> {
    let mut out = TermSet::<Complex64>::new(lhs.to_qubits());
    assign_from_anticommutator(&mut out.borrow_mut(), lhs, rhs)?;
    Ok(out)
}

pub fn commute<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
    atol: f64,
) -> Result<bool, DifferentQubits> {
    Ok(commutator(lhs, rhs)?.get_coeffs().all_insignificant(atol))
}

pub fn anticommute<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
    atol: f64,
) -> Result<bool, DifferentQubits> {
    Ok(anticommutator(lhs, rhs)?
        .get_coeffs()
        .all_insignificant(atol))
}

pub fn commute_default<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<bool, DifferentQubits> {
    commute(lhs, rhs, C::COMMUTES_ATOL_DEFAULT)
}

pub fn anticommute_default<C: FieldElem>(
    lhs: &terms::View<C>,
    rhs: &terms::View<C>,
) -> Result<bool, DifferentQubits> {
    anticommute(lhs, rhs, C::COMMUTES_ATOL_DEFAULT)
}

pub fn conserves_hamming_weight<C: FieldElem>(terms: &terms::View<C>, atol: f64) -> bool {
    let nop = num_op::<C>(terms.word_iters.to_qubits()).terms;
    commute(terms, &nop.borrow(), atol).unwrap()
}

pub fn conserves_hamming_weight_default<C: FieldElem>(terms: &terms::View<C>) -> bool {
    conserves_hamming_weight(terms, C::COMMUTES_ATOL_DEFAULT)
}

pub fn conserves_odd_pos_hamming_weight<C: FieldElem>(terms: &terms::View<C>, atol: f64) -> bool {
    let nop = num_op_odd_pos::<C>(terms.word_iters.to_qubits()).terms;
    commute(terms, &nop.borrow(), atol).unwrap()
}

pub fn conserves_odd_pos_hamming_weight_default<C: FieldElem>(terms: &terms::View<C>) -> bool {
    conserves_odd_pos_hamming_weight(terms, C::COMMUTES_ATOL_DEFAULT)
}

pub fn is_hermitian<C: FieldElem>(terms: &terms::View<C>, atol: f64) -> bool {
    !terms.coeffs.imag_part_is_significant(atol)
}

pub fn is_hermitian_default<C: FieldElem>(terms: &terms::View<C>) -> bool {
    !terms
        .coeffs
        .imag_part_is_significant(C::COMMUTES_ATOL_DEFAULT)
}

/// Variants for specifying the size, ordering, and content of a sparse representation
pub enum SparseBasis {
    Full,
    Partial(Vec<u64>),
}

/// Return this linear combination of Pauli words as a sparse matrix.
/// If basis is Full, the entire Hilbert space will be used.
/// Else (basis has Partial value), the returned sparse matrix will be the sum represented by `refs` projected into the given basis,
/// If sym_atol has a value, that value will be used as the commute tolerance to decide whether hamming weights are conserved.
/// Then if symmetries are detected, they will be used to elide exact cancellations.
/// Else (sym_atol is None), no attempt will be made to detect or exploit symmetries.
/// If big_endian is true, the bit associated with mode 0 is the most significant in the index integer
/// Else, the bit associated with mode n_qubit - 1 is the most significant
/// Note there is interaction between the basis and big_endian args:
/// The big_endian value indicates which storage order is assumed in the partial basis
pub fn to_sparse_matrix<C: FieldElem>(
    terms: &terms::View<C>,
    basis: SparseBasis,
    sym_atol: Option<f64>,
    big_endian: bool,
) -> sprs::CsMat<Complex64> {
    let n_qubit = terms.word_iters.qubits().len();
    if n_qubit >= 64 {
        panic!("too many qubits to convert to sparse.");
    }

    let dim = 1_usize << n_qubit;
    let basis_size = if let SparseBasis::Partial(v) = &basis {
        v.len()
    } else {
        dim
    };
    // if Hamming weight conservation is to be exploited, find whether it holds within the supplied tolerance.
    let hw_consrv = if let Some(atol) = sym_atol {
        conserves_hamming_weight(terms, atol)
    } else {
        false
    };
    // if odd bit Hamming weight conservation is to be exploited, find whether it holds within the supplied tolerance.
    let odd_hw_consrv = if let Some(atol) = sym_atol {
        conserves_odd_pos_hamming_weight(terms, atol)
    } else {
        false
    };
    let basis_set: Option<IndexSet<u64>> = if let SparseBasis::Partial(v) = &basis {
        Some(v.iter().copied().collect::<IndexSet<_>>())
    } else {
        None
    };

    use sprs::TriMat;
    let mut trips = TriMat::<Complex64>::new((basis_size, basis_size));
    for i_ket in 0..basis_size {
        let mut row_entries = HashMap::<usize, Complex64>::default();
        let ket: u64 = if let SparseBasis::Partial(v) = &basis {
            v[i_ket]
        } else {
            i_ket as u64
        };
        let ket = if big_endian {
            invert_endian(ket, n_qubit)
        } else {
            ket
        };
        // ket is in little-endian
        for (cmpnt, coeff) in terms.word_iters.iter().zip(terms.coeffs.iter()) {
            let (bra, phase) = cmpnt.mul_state_u64(ket);
            // bra is in little-endian
            if hw_consrv && ket.count_ones() != bra.count_ones() {
                // self conserves Hamming weight, but this contribution does not, so skip.
                continue;
            }
            if odd_hw_consrv
                && (ket & ODD_BIT_MASK).count_ones() != (bra & ODD_BIT_MASK).count_ones()
            {
                // self conserves odd position Hamming weight, but this contribution does not, so skip.
                continue;
            }
            let bra = if big_endian {
                invert_endian(bra, n_qubit)
            } else {
                bra
            };
            // bra is in specified endianness
            let mut i_bra = bra as usize;
            if let Some(set) = &basis_set {
                if let Some(x) = &set.get_index_of(&bra) {
                    i_bra = *x;
                } else {
                    continue;
                }
            }

            match row_entries.get_mut(&i_bra) {
                Some(x) => {
                    *x += coeff.scaled_complex(phase.to_complex());
                    // remove entry if exact cancellation
                    if *x == Complex64::ZERO {
                        row_entries.remove(&i_bra);
                    }
                }
                None => {
                    row_entries.insert(i_bra, coeff.scaled_complex(phase.to_complex()));
                }
            }
        }
        row_entries
            .into_iter()
            .for_each(|(i_col, c)| trips.add_triplet(i_ket, i_col, c));
    }
    trips.to_csc()
}

/// Recursively add a matrix with a single non-zero element to terms.
/// i_qubit is the current qubit index.
/// i_row and i_col are the current row and column indices.
/// coeff is the value of the matrix element.
/// phase is the power of i that multiplies the current branch.
/// work is the pauli operator string accumulating the components to add to `self`.
fn add_mat_elem_recursive<C: FieldElem>(
    term_set: &mut term_set::ViewMut<C>,
    i_qubit: usize,
    i_row: usize,
    i_col: usize,
    c: C,
    phase: ComplexSign,
    work: &mut PauliWord,
) {
    assert_eq!(term_set.qubits(), work.borrow().qubits());
    let n_qubit = term_set.qubits().len();
    if i_qubit == n_qubit {
        let mut coeff = FieldElem::scaled_complex(&c, phase.conj().to_complex());
        // incorporate all factors of 1/2.
        coeff /= (1 << n_qubit) as f64;
        scaled_iadd_elem(term_set, work.borrow(), C::complex_part(coeff));
        return;
    }
    let dim = 1_usize << (n_qubit - i_qubit);
    /*
     * dimension of each of the 4 quadrants of the current sub-matrix
     */
    let half = dim >> 1;
    if i_row < half {
        if i_col < half {
            // top left quadrant
            // unit matrix e00 = [[1, 0], [0, 0]] -> (I + Z) / 2
            work.borrow_mut()
                .set_pauli_unchecked(i_qubit, PauliMatrix::I);
            add_mat_elem_recursive(term_set, i_qubit + 1, i_row, i_col, c, phase, work);
            work.borrow_mut()
                .set_pauli_unchecked(i_qubit, PauliMatrix::Z);
            add_mat_elem_recursive(term_set, i_qubit + 1, i_row, i_col, c, phase, work);
        } else {
            // top right quadrant
            // unit matrix e01 = [[0, 1], [0, 0]] -> (X + iY) / 2
            work.borrow_mut()
                .set_pauli_unchecked(i_qubit, PauliMatrix::X);
            add_mat_elem_recursive(term_set, i_qubit + 1, i_row, i_col - half, c, phase, work);
            work.borrow_mut()
                .set_pauli_unchecked(i_qubit, PauliMatrix::Y);
            add_mat_elem_recursive(
                term_set,
                i_qubit + 1,
                i_row,
                i_col - half,
                c,
                phase.mul_by_i(),
                work,
            );
        }
    } else if i_col < half {
        // bottom left quadrant
        // unit matrix e10 = [[0, 0], [1, 0]] -> (X - iY) / 2
        work.borrow_mut()
            .set_pauli_unchecked(i_qubit, PauliMatrix::X);
        add_mat_elem_recursive(term_set, i_qubit + 1, i_row - half, i_col, c, phase, work);
        work.borrow_mut()
            .set_pauli_unchecked(i_qubit, PauliMatrix::Y);
        add_mat_elem_recursive(
            term_set,
            i_qubit + 1,
            i_row - half,
            i_col,
            c,
            phase.div_by_i(),
            work,
        );
    } else {
        // bottom right quadrant
        // unit matrix e11 = [[0, 0], [0, 1]] -> (I - Z) / 2
        work.borrow_mut()
            .set_pauli_unchecked(i_qubit, PauliMatrix::I);
        add_mat_elem_recursive(
            term_set,
            i_qubit + 1,
            i_row - half,
            i_col - half,
            c,
            phase,
            work,
        );
        work.borrow_mut()
            .set_pauli_unchecked(i_qubit, PauliMatrix::Z);
        add_mat_elem_recursive(
            term_set,
            i_qubit + 1,
            i_row - half,
            i_col - half,
            c,
            -phase,
            work,
        );
    }
}

pub fn add_mat_elem<C: FieldElem>(
    term_set: &mut term_set::ViewMut<C>,
    i_row: usize,
    i_col: usize,
    c: C,
) {
    let mut work = PauliWord::new(term_set.to_qubits());
    add_mat_elem_recursive(term_set, 0, i_row, i_col, c, ComplexSign(0), &mut work);
}

/// Add a contribution of c * paulis to the linear combination by converting the vector of matrices to a temporary `PauliWord`.
pub fn add_from_pauli_vec<C: FieldElem>(
    term_set: &mut term_set::ViewMut<C>,
    paulis: Vec<PauliMatrix>,
    c: C,
) -> Result<(), OutOfBounds> {
    scaled_iadd_elem(
        term_set,
        PauliWord::from_vec(term_set.to_qubits(), paulis)?.borrow(),
        c,
    );
    Ok(())
}

/// Add a contribution of c * paulis to the linear combination by converting the hashmap of matrices to a temporary `PauliWord`.
pub fn add_from_pauli_map<C: FieldElem>(
    term_set: &mut term_set::ViewMut<C>,
    paulis: HashMap<usize, PauliMatrix>,
    c: C,
) -> Result<(), OutOfBounds> {
    scaled_iadd_elem(
        term_set,
        PauliWord::from_map(term_set.to_qubits(), paulis)?.borrow(),
        c,
    );
    Ok(())
}

/// Create from the given coeff vector and mode settings, absorbing any phases into the coefficient if representable, else return error.
pub fn from_springs_coeffs<C: FieldElem>(
    qubits: Qubits,
    springs: Springs,
    coeffs: C::Vector,
) -> Result<TermSet<C>, ParseError> {
    let terms = Terms::<C>::from_springs_coeffs(qubits, springs, coeffs)?;
    Ok(TermSet::<C>::from(terms))
}

/// Create from the given springs and coeff vector, absorbing any phases into the coefficient if representable, else return error.
/// Infers a default qubit space.
pub fn from_springs_coeffs_default<C: FieldElem>(
    springs: Springs,
    coeffs: C::Vector,
) -> Result<TermSet<C>, ParseError> {
    let qubits = Qubits::from_count(springs.get_mode_inds().default_n_mode() as usize);
    from_springs_coeffs(qubits, springs, coeffs)
}

/// Create a new linear combination from some pairs of vectors of Pauli matrices and coefficients.
pub fn from_vec_coeff_pairs<C: FieldElem>(
    qubits: Qubits,
    pairs: Vec<(Vec<PauliMatrix>, C)>,
) -> Result<TermSet<C>, OutOfBounds> {
    let mut this = TermSet::<C>::new(qubits);
    for (paulis, c) in pairs {
        add_from_pauli_vec(&mut this.borrow_mut(), paulis, c)?;
    }
    Ok(this)
}

/// Create a new linear combination from some pairs of vectors of Pauli matrices and coefficients.
/// assuming an inferred default qubit space.
pub fn from_vec_coeff_pairs_default<C: FieldElem>(pairs: Vec<(Vec<PauliMatrix>, C)>) -> TermSet<C> {
    let n_qubit = pairs.iter().map(|(v, _)| v.len()).max().unwrap_or_default();
    // can't possibly have ModeOutOfBounds, since the space was created to hold the input.
    from_vec_coeff_pairs(Qubits::from_count(n_qubit), pairs).unwrap()
}

/// Create a new linear combination from some pairs of hashmaps from mode indices to Pauli matrices and coefficients.
pub fn from_map_coeff_pairs<C: FieldElem>(
    qubits: Qubits,
    pairs: Vec<(HashMap<usize, PauliMatrix>, C)>,
) -> Result<TermSet<C>, OutOfBounds> {
    let mut this = TermSet::<C>::new(qubits);
    for (paulis, c) in pairs {
        add_from_pauli_map(&mut this.borrow_mut(), paulis, c)?;
    }
    Ok(this)
}

/// Create a new linear combination from some pairs of hashmaps from mode indices to Pauli matrices and coefficients,
/// assuming an inferred default qubit space.
pub fn from_map_coeff_pairs_default<C: FieldElem>(
    pairs: Vec<(HashMap<usize, PauliMatrix>, C)>,
) -> TermSet<C> {
    let i_qubit_max = pairs.iter().map(|(map, _)| map.keys().max()).max();
    let n_qubit = match i_qubit_max {
        Some(Some(x)) => *x,
        _ => 0,
    };
    from_map_coeff_pairs(Qubits::from_count(n_qubit), pairs).unwrap()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::cmpnt::springs::ModeSettings;
    use crate::container::coeffs::traits::NumReprVec;
    use crate::container::traits::{Elements, RefElements};
    use crate::container::word_iters::lincomb::{scaled, sum};
    use crate::container::word_iters::term_set::AsView;
    use crate::container::word_iters::terms::Terms;
    use crate::qubit::mode::{PauliMatrix, Qubits};
    use crate::qubit::pauli::cmpnt_major;
    use crate::qubit::pauli::cmpnt_major::cmpnt_list::CmpntList;
    use crate::qubit::pauli::cmpnt_major::term_set::AsView as _;
    use crate::qubit::pauli::springs::Springs;
    use crate::qubit::test::{HEHP_STO3G_HAM_JW_INPUT, HEHP_STO3G_HF_ENERGY};
    use crate::qubit::traits::PauliWordRef;
    use bincode::config;
    use num_complex::Complex64;
    use PauliMatrix::*;

    #[test]
    fn test_contains() {
        let qubits = Qubits::from_count(4);
        let make_cmpnt = |v: Vec<PauliMatrix>| PauliWord::from_vec(qubits.clone(), v).unwrap();
        let mut sum = TermSet::<f64>::new(qubits.clone());
        assert_eq!(sum.len(), 0);
        assert!(sum.is_empty());
        scaled_iadd_elem(
            &mut sum.borrow_mut(),
            make_cmpnt(vec![I, X, Y, Z]).borrow(),
            PI,
        );
        assert_eq!(sum.len(), 1);
        assert!(sum
            .borrow()
            .lookup_coeff_elem_ref(make_cmpnt(vec![I, X, Y, Z]).borrow())
            .is_some_and(|x| x == PI));
    }

    #[test]
    fn test_to_from_binary() {
        let input = HEHP_STO3G_HAM_JW_INPUT;
        let springs = Springs::from_str(input);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        let coeffs = Vec::<f64>::try_parse(input);
        assert!(coeffs.is_ok());
        let terms = Terms {
            word_iters: CmpntList::from_springs_default(&springs).0,
            coeffs: coeffs.unwrap(),
        };
        let sum = TermSet::<f64>::from(terms);
        let encoded = bincode::serde::encode_to_vec(&sum, config::standard()).unwrap();
        let (decoded, len): (TermSet<f64>, usize) =
            bincode::serde::decode_from_slice(&encoded[..], config::standard()).unwrap();
        assert_eq!(len, encoded.len());
        assert!(sum.borrow() == decoded.borrow());
    }

    #[test]
    fn test_real() {
        const N_QUBIT: usize = 6;
        let mut lc = TermSet::<f64>::new(Qubits::from_count(N_QUBIT));
        assert_eq!(lc.len(), 0);
        scaled_iadd_elem(
            &mut lc.borrow_mut(),
            PauliWord::from_vec_default(vec![Y, Z, X, X, I, Z]).borrow(),
            1.5,
        );
        assert_eq!(lc.len(), 1);
        scaled_iadd_elem(
            &mut lc.borrow_mut(),
            PauliWord::from_vec_default(vec![I, Z, Y, X, I, Z]).borrow(),
            2.3,
        );
        assert_eq!(lc.len(), 2);
        scaled_iadd_elem(
            &mut lc.borrow_mut(),
            PauliWord::from_vec_default(vec![Y, Z, X, X, I, Z]).borrow(),
            -1.0,
        );
        assert_eq!(lc.len(), 2);
        assert_eq!(lc.to_string(), "(0.5, Y0 Z1 X2 X3 Z5), (2.3, Z1 Y2 X3 Z5)");
    }

    #[test]
    fn test_complex() {
        const N_QUBIT: usize = 6;
        let mut lc = TermSet::<Complex64>::new(Qubits::from_count(N_QUBIT));
        assert_eq!(lc.len(), 0);
        scaled_iadd_elem(
            &mut lc.borrow_mut(),
            PauliWord::from_vec_default(vec![Y, Z, X, X, I, Z]).borrow(),
            Complex64::new(1.2, 2.2),
        );
        assert_eq!(lc.len(), 1);
        scaled_iadd_elem(
            &mut lc.borrow_mut(),
            PauliWord::from_vec_default(vec![Y, Z, X, Y, I, Z]).borrow(),
            Complex64::new(-1.2, 9.2),
        );
        assert_eq!(lc.len(), 2);
        scaled_iadd_elem(
            &mut lc.borrow_mut(),
            PauliWord::from_vec_default(vec![Y, Z, X, Y, I, Z]).borrow(),
            Complex64::new(1.2, -9.2),
        );
        // exact cancellation
        assert_eq!(lc.len(), 1);
        assert_eq!(lc.to_string(), "(1.2+2.2i, Y0 Z1 X2 X3 Z5)");
    }

    #[test]
    fn test_basic_chem_hamiltonian() {
        let input = HEHP_STO3G_HAM_JW_INPUT;
        let coeffs = Vec::<f64>::try_parse(input);
        assert!(coeffs.is_ok());
        let coeffs = coeffs.unwrap();
        let springs = Springs::from_str(input);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        assert_eq!(springs.len(), 27);
        let ham =
            cmpnt_major::terms::Terms::<f64>::from_springs_coeffs_default(springs, coeffs).unwrap();
        let ham = TermSet::from(ham);
        // check selected components
        assert!(ham
            .borrow()
            .lookup_coeff_pauli_vec(vec![I, I, I, I])
            .is_ok_and(|x| x.is_some_and(|x| x.is_close_default(-1.541975952896969_f64))));
        assert!(ham
            .borrow()
            .lookup_coeff_pauli_vec(vec![Z, Y, Z, Y])
            .is_ok_and(|x| x.is_some_and(|x| x.is_close_default(0.0432299714640452_f64))));
        assert!(ham
            .borrow()
            .lookup_coeff_pauli_vec(vec![I, I, Z, Z])
            .is_ok_and(|x| x.is_some_and(|x| x.is_close_default(0.18815905542064587_f64))));
        let ham_plus_ham = sum(&ham.borrow().as_terms(), &ham.borrow().as_terms());
        let ham_times_2 = scaled(&ham.borrow().as_terms(), 2.0);
        assert!(ham_plus_ham
            .borrow()
            .all_close_default(&ham_times_2.borrow()));
        // hamiltonian should commute with the number operator
        assert!(conserves_hamming_weight(&ham.borrow().as_terms(), 1e-5));
        // and the odd pos number operator (i.e. N_beta in JW encoding)
        assert!(conserves_odd_pos_hamming_weight(
            &ham.borrow().as_terms(),
            1e-12
        ));
    }

    #[test]
    fn test_to_sparse() {
        let input = HEHP_STO3G_HAM_JW_INPUT;
        let coeffs = Vec::<f64>::try_parse(input);
        assert!(coeffs.is_ok());
        let coeffs = coeffs.unwrap();
        let springs = Springs::from_str(input);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        let ham =
            cmpnt_major::terms::Terms::<f64>::from_springs_coeffs_default(springs, coeffs).unwrap();
        for big_endian in [true, false] {
            // use the reference configuration as the sole element of the basis set
            let ref_state = SparseBasis::Partial(vec![if big_endian { 0b1100 } else { 0b0011 }]);
            let sparse = to_sparse_matrix(&ham.borrow(), ref_state, None, big_endian);
            assert_eq!(sparse.nnz(), 1);
            assert!(<f64 as FieldElem>::is_close_default(
                &HEHP_STO3G_HF_ENERGY,
                sparse.data().iter().next().unwrap().re
            ));
            assert_eq!(sparse.data().iter().next().unwrap().im, 0.0);
            // use the full computational basis as the basis set
            let sparse = to_sparse_matrix(&ham.borrow(), SparseBasis::Full, None, big_endian);
            // this should be equivalent to summing over the to_sparse results of the individual Pauli words
            let mut sparse_chk = sprs::CsMat::<Complex64>::zero(sparse.shape());
            for term in ham.iter() {
                let (cmpnt, coeff) = term.unpack();
                let mut tmp = cmpnt.to_sparse_matrix(big_endian);
                tmp.scale(Complex64::new(coeff, 0.0));
                sparse_chk = &sparse_chk + &tmp;
            }
            assert!(sparse.to_dense() == sparse_chk.to_dense());
        }
    }

    #[test]
    fn test_sum_to_from_binary() {
        let input = HEHP_STO3G_HAM_JW_INPUT;
        let coeffs = Vec::<f64>::try_parse(input);
        assert!(coeffs.is_ok());
        let coeffs = coeffs.unwrap();
        let springs = Springs::from_str(input);
        assert!(springs.is_ok());
        let springs = springs.unwrap();
        assert_eq!(springs.len(), 27);
        let ham =
            cmpnt_major::terms::Terms::<f64>::from_springs_coeffs_default(springs, coeffs).unwrap();

        let encoded = bincode::serde::encode_to_vec(&ham, config::standard()).unwrap();
        assert_eq!(encoded.len(), 281);
        let (decoded, len): (TermSet<f64>, usize) =
            bincode::serde::decode_from_slice(&encoded[..], config::standard()).unwrap();
        assert_eq!(len, encoded.len());
        assert!(TermSet::from(ham).borrow() == decoded.borrow());
    }
}
