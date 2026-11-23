//! Fermion operator in linear combination utilities.

use crate::container::coeffs::traits::FieldElem;
use crate::container::traits::Elements;
use crate::container::word_iters::term_set::AsViewMut;
use crate::container::word_iters::WordIters;
use crate::fermion::operator::general::string_list::StringList;
use crate::fermion::operator::general::term_set::{
    self as general_term_set, TermSet as GeneralTermSet,
};
use crate::fermion::traits::{DifferentSpaces, ModesBased};

/// Multiply two raw term sets without normal ordering, returning a raw term set.
pub fn rmul<C: FieldElem>(
    lhs: &general_term_set::View<C>,
    rhs: &general_term_set::View<C>,
) -> Result<GeneralTermSet<C>, DifferentSpaces> {
    let max_string_len = lhs.word_iters.max_string_len + rhs.word_iters.max_string_len;
    let mut out = GeneralTermSet::<C>::new(max_string_len, lhs.to_modes());
    DifferentSpaces::check_transitive(lhs, rhs, &out)?;
    let mut product = StringList::new(max_string_len, lhs.to_modes());
    let n_lhs = lhs.word_iters.len().min(lhs.coeffs.len());
    let n_rhs = rhs.word_iters.len().min(rhs.coeffs.len());
    for (i_lhs, lhs_coeff) in lhs.coeffs.iter().take(n_lhs).enumerate() {
        let (lhs_modes, lhs_adj) = lhs.word_iters.get(i_lhs);
        for (i_rhs, rhs_coeff) in rhs.coeffs.iter().take(n_rhs).enumerate() {
            let (rhs_modes, rhs_adj) = rhs.word_iters.get(i_rhs);
            let c = *lhs_coeff * *rhs_coeff;
            product.push_concat(&lhs_modes, &lhs_adj, &rhs_modes, &rhs_adj);
            let (index, inserted) = out.insert_u64it_or_get_index(product.elem_u64it(0), c);
            if !inserted {
                out.terms.coeffs[index] += c;
            }
            product.resize(0);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::traits::Elements;
    use crate::fermion::mode::Modes;
    use crate::fermion::operator::general::term_set::TermSet as GeneralTermSet;

    #[test]
    fn test_mul() {
        let modes_space = Modes::from_count(2);

        // lhs = a_0^+ + 2*a_1^+
        let mut lhs = GeneralTermSet::<f64>::new(1, modes_space.clone());
        lhs.push_term(&[0], &[true], 1.0_f64);
        lhs.push_term(&[1], &[true], 2.0_f64);

        // rhs = 3*a_0
        let mut rhs = GeneralTermSet::<f64>::new(1, modes_space.clone());
        rhs.push_term(&[0], &[false], 3.0_f64);

        let result = rmul(&lhs.as_terms(), &rhs.as_terms()).unwrap();

        let check = |modes: &[usize], adj: &[bool], coeff: f64| {
            let n = result.terms.word_iters.len();
            let found = (0..n).any(|i| {
                let (m, a) = result.terms.word_iters.get(i);
                let c = result.terms.coeffs[i];
                m == modes && a == adj && c == coeff
            });
            assert!(
                found,
                "Non-normal-ordered term {modes:?}/{adj:?} coeff {coeff} not found"
            );
        };

        assert_eq!(result.terms.word_iters.len(), 2);
        check(&[0, 0], &[true, false], 3.0);
        check(&[1, 0], &[true, false], 6.0);
    }
}
