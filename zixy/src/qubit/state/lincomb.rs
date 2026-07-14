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
pub use crate::utils::vector_ops::{l2_norm, l2_norm_square, vdot};

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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::cmpnt::springs::ModeSettings;
    use crate::cmpnt::state_springs::BinarySprings;
    use crate::container::bit_matrix::AsRowRef;
    use crate::container::traits::RefElements;
    use crate::container::word_iters::term_set::AsView;
    use crate::qubit::state::terms::Terms;
    use num_complex::Complex64;
    use rstest::rstest;

    #[rstest]
    #[case(vec![], 0.0)]
    #[case(vec![2.0], 4.0)]
    #[case(vec![-3.0], 9.0)]
    #[case(vec![1.0, 2.0, 3.0], 14.0)]
    #[case(vec![0.0, 0.0, 0.0], 0.0)]
    #[case(vec![0.0, 2.0, 0.0], 4.0)]
    fn test_l2_norm_square(#[case] coeffs: Vec<f64>, #[case] expected: f64) {
        let qubits = Qubits::from_count(coeffs.len());
        let mut state: terms::Terms<f64> = Terms::new(qubits);
        for c in coeffs {
            state.coeffs.push(c);
        }
        assert_eq!(l2_norm_square(&state), expected);
    }

    #[rstest]
    #[case(vec![Complex64::new(3.0, 4.0)], 25.0)]
    #[case(vec![Complex64::new(0.0, 5.0)], 25.0)]
    #[case(vec![Complex64::new(1.0, 1.0)], 2.0)]
    #[case(vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -2.0)], 10.0)]
    #[case(vec![Complex64::new(-3.0, 4.0)], 25.0)]
    #[case(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)], 1.0)]
    #[case(vec![Complex64::new(1e-9, 1e-9)], 2e-18)]
    fn test_l2_norm_square_complex(
        #[case] coeffs: Vec<num_complex::Complex<f64>>,
        #[case] expected: f64,
    ) {
        let qubits = Qubits::from_count(coeffs.len());
        let mut state: terms::Terms<num_complex::Complex<f64>> = Terms::new(qubits);
        for c in coeffs {
            state.coeffs.push(c);
        }
        assert_eq!(l2_norm_square(&state), expected);
    }

    #[rstest]
    // Test case: <v1|v2> where |v1> = 2|0> and |v2> = 3|0>, expected <v1|v2> = 6
    #[case(vec![Complex64::new(2.0, 0.0)], "[0]", vec![Complex64::new(3.0, 0.0)], "[0]",  Complex64::new(6.0, 0.0))]
    // Test case: <v1|v2> where |v1> = null and |v2> = (3+4i)|1>, expected <v1|v2> = 0
    #[case(vec![], "", vec![Complex64::new(3.0, 4.0)], "[0]", Complex64::new(0.0, 0.0))]
    // Test case: <v1|v2> where |v1> = (1+2i)|0> and |v2> = null, expected <v1|v2> = 0
    #[case(vec![Complex64::new(1.0, 2.0)], "[0]", vec![], "", Complex64::new(0.0, 0.0))]
    // Test case: <v1|v2> where |v1> = (1+2i)|0> and |v2> = (3+4i)|0>, expected <v1|v2> = (1-2i)(3+4i) = 11-2i
    #[case(vec![Complex64::new(1.0, 2.0)], "[0]", vec![Complex64::new(3.0, 4.0)], "[0]", Complex64::new(11.0, -2.0))]
    // Test case: <v1|v2> where |v1> = (1+2i)|1> and |v2> = (3+4i)|0>, expected <v1|v2> = 0 since they are orthogonal
    #[case(vec![Complex64::new(1.0, 2.0)], "[1]", vec![Complex64::new(3.0, 4.0)], "[0]", Complex64::new(0.0, 0.0))]
    // Test case: <v1|v2> where |v1> = (1+2i)|0> and |v2> = (3+4i)|1>, expected <v1|v2> = 0 since they are orthogonal
    #[case(vec![Complex64::new(1.0, 2.0)], "[0]", vec![Complex64::new(3.0, 4.0)], "[1]", Complex64::new(0.0, 0.0))]
    // Test case: <v1|v2> where |v1> = (1+2i)|01> + (-1+-2.5i)|11> and |v2> = (3+4i)|01> + i|00>, expected <v1|v2> = (1-2i)(3+4i) = 11-2i
    #[case(vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, -2.5)], "[0, 1], [1,1]", vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 1.0)], "[0,1], [0,0]", Complex64::new(11.0, -2.0))]
    // Test case: <v1|v2> where |v1> = (1+2i)|01> + (-1-2.5i)|11> and |v2> = (3+4i)|01> + i|11>, expected <v1|v2> = (1-2i)(3+4i) + (-1+2.5i)(i) = 8.5-3i
    #[case(vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, -2.5)], "[0, 1], [1,1]", vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 1.0)], "[0,1], [1,1]", Complex64::new(8.5, -3.0))]
    fn test_vdot(
        #[case] lhs_coeffs: Vec<num_complex::Complex<f64>>,
        #[case] lhs_qubit_labels: &str,
        #[case] rhs_coeffs: Vec<num_complex::Complex<f64>>,
        #[case] rhs_qubit_labels: &str,
        #[case] expected: num_complex::Complex<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let qubits_lhs = Qubits::from_count(lhs_coeffs.len());
        let qubits_rhs = Qubits::from_count(rhs_coeffs.len());
        let springs_rhs = BinarySprings::from_str(rhs_qubit_labels)?;
        let springs_lhs = BinarySprings::from_str(lhs_qubit_labels)?;
        let rhs_terms = Terms::<num_complex::Complex<f64>>::from_springs_coeffs(
            qubits_rhs,
            springs_rhs,
            rhs_coeffs,
        )?;
        let lhs_terms = Terms::<num_complex::Complex<f64>>::from_springs_coeffs(
            qubits_lhs,
            springs_lhs,
            lhs_coeffs,
        )?;

        let lhs_term_set: term_set::TermSet<num_complex::Complex<f64>> = lhs_terms.into();
        assert_eq!(vdot(&lhs_term_set, &rhs_terms), expected);
        Ok(())
    }

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
