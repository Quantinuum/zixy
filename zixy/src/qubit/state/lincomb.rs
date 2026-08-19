//! Qubit state-specific linear combination utilities.

use crate::container::bit_matrix::AsBitMatrix;
use crate::container::coeffs::traits::FieldElem;
use crate::container::errors::{Dimension, OutOfBounds};
use crate::container::traits::proj::BorrowMut;
use crate::container::traits::RefElements;
use crate::container::word_iters;
use crate::container::word_iters::term_set::AsViewMut;
use crate::qubit::mode::Qubits;
use crate::qubit::state::{term_set, terms};
use crate::utils::arith::invert_endian;
pub use crate::utils::vector_ops::{l2_norm, vdot};

/// If big_endian is true, the bit associated with mode 0 is the most significant in the index integer
/// Else, the bit associated with mode n_qubit - 1 is the most significant
pub fn to_dense<C: FieldElem>(
    state: &impl terms::AsView<C>,
    big_endian: bool,
) -> Result<Vec<C>, OutOfBounds> {
    let state_ref = state.view();
    let n = state_ref.word_iters.n_bit();
    OutOfBounds::check(n, 64, Dimension::Element)?;

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
    Ok(out)
}

/// Populate an existing state linear combination in place from a dense array slice of coefficients.
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

/// Create and return a  new state linear combination from a dense array slice of coefficients.
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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::cmpnt::springs::ModeSettings;
    use crate::cmpnt::state_springs::BinarySprings;
    use crate::container::bit_matrix::AsRowRef;
    use crate::container::traits::RefElements;
    use crate::container::word_iters::term_set::AsView;
    use crate::qubit::state::terms::Terms;

    #[test]
    fn test_assign_from_dense_endian() {
        let qubits = Qubits::from_count(3);

        // Dense index 1 (001 in little-endian significance)
        let source = vec![0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // little-endian
        let mut le: term_set::TermSet<f64> = term_set::TermSet::new(qubits.clone());
        assign_from_dense(&mut le.borrow_mut(), &source, false);
        assert_eq!(le.view().coeffs.len(), 1);
        assert_eq!(le.view().coeffs[0], 7.0);
        assert_eq!(
            le.view().get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![true, false, false], // |100> in this mode ordering
        );

        // big-endian: index 1 -> bit-reverse(001) = 100
        let mut be: term_set::TermSet<f64> = term_set::TermSet::new(qubits);
        assign_from_dense(&mut be.borrow_mut(), &source, true);
        assert_eq!(be.view().coeffs.len(), 1);
        assert_eq!(be.view().coeffs[0], 7.0);
        assert_eq!(
            be.view().get_elem_ref(0).get_word_iter_ref().to_vec(),
            vec![false, false, true], // |001>
        );
    }

    #[test]
    fn test_from_dense() {
        let qubits = Qubits::from_count(2);
        let source = vec![0.0, 1.0, 2.0, 3.0];
        let state = from_dense(qubits, &source, false);
        assert_eq!(state.view().coeffs.len(), 3);
        assert_eq!(state.view().coeffs[0], 1.0);
        assert_eq!(state.view().coeffs[1], 2.0);
        assert_eq!(state.view().coeffs[2], 3.0);
    }

    #[test]
    fn test_to_dense_out_of_bounds() {
        let qubits = Qubits::from_count(65);
        let lhs = Terms::<num_complex::Complex<f64>>::new(qubits);
        let result = to_dense(&lhs, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_to_dense() -> Result<(), Box<dyn std::error::Error>> {
        let qubits = Qubits::from_count(3);
        let springs = BinarySprings::from_str("[1, 0, 1]")?;
        let terms = Terms::<f64>::from_springs(qubits, &springs)?;
        let dense = to_dense(&terms, false)?;
        assert_eq!(dense.len(), 8);
        assert_eq!(dense[0], 0.0); // |000>
        assert_eq!(dense[1], 0.0); // |001>
        assert_eq!(dense[2], 0.0); // |010>
        assert_eq!(dense[3], 0.0); // |011>
        assert_eq!(dense[4], 0.0); // |100>
        assert_eq!(dense[5], 1.0); // |101>
        assert_eq!(dense[6], 0.0); // |110>
        assert_eq!(dense[7], 0.0); // |111>
        Ok(())
    }
}
