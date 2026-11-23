//! Fermion operator products module.

use crate::container::bit_matrix::{AsRowMutRef, AsRowRef};
use crate::container::coeffs::sign::{Sign, SignVec};
use crate::container::coeffs::traits::NumReprVec;
use crate::container::traits::{Elements, MutRefElements, RefElements};
use crate::container::word_iters::{ElemMutRef, ElemRef, WordIters};
use crate::fermion::mode::Modes;
use crate::fermion::operator::normal::cre_or_ann;
use crate::fermion::operator::normal::string_list::{StringList, StringRef};
use crate::fermion::traits::ModesBased;

/// Stores variables to support the recursive computation of normal-ordered fermion operator products.
struct ProductHelper {
    strings: StringList,
    signs: SignVec,
    n_lhs_ann: usize,
    n_rhs_ann: usize,
    n_lhs_cre: usize,
    n_rhs_cre: usize,
    n_bif: usize,
}

/// - `branch`: branch index, created by setting bits where the high-rank branch is selected and clearing bits
///   where the low-rank branch is selected.
/// - `bif`: bifurcation index; the number of bifurcations that have occurred before this call.
/// - `exchange`: number of anti-symmetric exchanges that have taken place to bring the product to its current
///   state.
/// - `lhs_ann`: number of set modes in `LHS_` that have already been processed.
/// - `rhs_cre`: number of set modes in `RHS^` that have already been processed.

#[derive(Clone, Copy, Default)]
struct Indices {
    bif: usize,
    exchange: usize,
    lhs_ann: usize,
    rhs_cre: usize,
}

impl Indices {
    /// Return a copy with both processed common-mode counters advanced by one.
    fn incremented(&self) -> Self {
        let mut out = *self;
        out.lhs_ann += 1;
        out.rhs_cre += 1;
        out
    }
}

impl ProductHelper {
    /// Create an empty helper for building products of operators on the given fermionic mode space.
    pub fn new() -> Self {
        Self {
            strings: StringList::new(Modes::from_count(0)),
            signs: SignVec::default(),
            n_lhs_ann: 0,
            n_rhs_ann: 0,
            n_lhs_cre: 0,
            n_rhs_cre: 0,
            n_bif: 0,
        }
    }

    /// Return whether the product `lhs * rhs` is annihilated by fermionic anticommutation rules.
    fn destroys(lhs: &StringRef, rhs: &StringRef) -> bool {
        /*
         * Fermion operators obey the following anticommutation relations
         *  {i^, j^} = 0   {i_, j_} = 0   {i^, j_} = delta_ij
         * The component product is zero if two like operators appear next to each other, since
         *  {i^, i^} = i^i^ + i^i^ = 0 => i^i^ = 0
         * This occurs when either:
         *  - the same index appears in the LHS and RHS creations but not in the LHS annihilations
         *  - the same index appears in the LHS and RHS annihilations but not in the RHS creations
         */
        lhs.get_cre_part().triple_hamming_weight(
            &rhs.get_cre_part(),
            false,
            &lhs.get_ann_part(),
            true,
        ) != 0
            || lhs.get_ann_part().triple_hamming_weight(
                &rhs.get_ann_part(),
                false,
                &rhs.get_cre_part(),
                true,
            ) != 0
    }

    /// Return whether a common mode must contract to avoid a repeated outer operator.
    fn must_contract(lhs: &StringRef, rhs: &StringRef, mode: usize) -> bool {
        lhs.get_cre_part().get_bit_unchecked(mode) || rhs.get_ann_part().get_bit_unchecked(mode)
    }

    /// Count common modes with both contracted and uncontracted results.
    fn count_branch_points(lhs: &StringRef, rhs: &StringRef) -> usize {
        lhs.get_ann_part()
            .iter_set_bits_flat()
            .filter(|&mode| {
                rhs.get_cre_part().get_bit_unchecked(mode) && !Self::must_contract(lhs, rhs, mode)
            })
            .count()
    }

    /// Compute and store every branch of the product `lhs * rhs`, including their signs.
    pub fn set(&mut self, lhs: &StringRef, rhs: &StringRef) {
        self.strings = StringList::new(lhs.modes().clone());
        self.signs.clear();
        if Self::destroys(lhs, rhs) {
            return;
        }
        self.n_bif = Self::count_branch_points(lhs, rhs);
        self.strings.resize(1 << self.n_bif);
        self.signs.resize(self.strings.len());
        self.n_lhs_ann = lhs.get_ann_part().hamming_weight();
        self.n_rhs_ann = rhs.get_ann_part().hamming_weight();
        self.n_lhs_cre = lhs.get_cre_part().hamming_weight();
        self.n_rhs_cre = rhs.get_cre_part().hamming_weight();
        // copy the LHS^ and RHS_ strings to the last element of the result
        let i_branch_last = self.strings.len().saturating_sub(1);
        // these are the parts that are already in normal order
        self.res_cre_mut(i_branch_last).assign(lhs.get_cre_part());
        self.res_ann_mut(i_branch_last).assign(rhs.get_ann_part());
        self.do_branch(
            self.strings.len().saturating_sub(1),
            Indices::default(),
            lhs,
            rhs,
            0,
            0,
        );
    }

    /// Return the annihilation-part result component for branch `branch`.
    fn res_ann(&self, branch: usize) -> ElemRef<'_, cre_or_ann::StringList> {
        self.strings.ann_part.get_elem_ref(branch)
    }

    /// Return mutable access to the annihilation-part result component for branch `branch`.
    fn res_ann_mut(&mut self, branch: usize) -> ElemMutRef<'_, cre_or_ann::StringList> {
        self.strings.ann_part.get_elem_mut_ref(branch)
    }

    /// Return the creation-part result component for branch `branch`.
    fn res_cre(&self, branch: usize) -> ElemRef<'_, cre_or_ann::StringList> {
        self.strings.cre_part.get_elem_ref(branch)
    }

    /// Return mutable access to the creation-part result component for branch `branch`.
    fn res_cre_mut(&mut self, branch: usize) -> ElemMutRef<'_, cre_or_ann::StringList> {
        self.strings.cre_part.get_elem_mut_ref(branch)
    }

    /// Insert a creator from the right of its sorted block and return the number of exchanges.
    fn insert_creation(&mut self, branch: usize, mode: usize) -> usize {
        let exchanges = self.res_cre(branch).count_set_bits_after(mode);
        self.res_cre_mut(branch).set_bit_unchecked(mode, true);
        exchanges
    }

    /// Insert an annihilator from the left of its sorted block and return the number of exchanges.
    fn insert_annihilation(&mut self, branch: usize, mode: usize) -> usize {
        let exchanges = self.res_ann(branch).count_set_bits_before(mode);
        self.res_ann_mut(branch).set_bit_unchecked(mode, true);
        exchanges
    }

    /// Handle a branch (recursively) of the normal-ordering product operation.
    ///
    /// Creation and annihilation blocks are each ordered by increasing mode index.
    ///
    /// The generic situation at input is:
    /// `L^(0) L^(1) ... L^(n_lc-1) L_(0) L_(1) ... L_(n_la-1) R^(0) R^(1) ... R^(n_rc-1) R_(0) R_(1) ... R_(n_ra)`.
    ///
    /// Where:
    /// - `n_lc` is the number of modes in the LHS creation operator string (the number of modes in `LHS^` initially
    ///   plus the number of modes already processed from `RHS^`).
    /// - `n_la` is the number of modes in the LHS annihilation operator string still to be processed.
    /// - `n_rc` is the number of modes in the RHS creation operator string still to be processed.
    /// - `n_ra` is the number of modes in the RHS annihilation operator string (the number of modes in `RHS_`
    ///   initially plus the number of modes already processed from `LHS_`).
    ///
    /// In comments, this product is abbreviated as:
    /// `LHS^ L_(0) L_(1) ... L_(n_la-1) R^(0) R^(1) ... R^(n_rc-1) RHS_`.
    fn do_branch(
        &mut self,
        i_branch: usize,
        mut inds: Indices,
        lhs: &StringRef,
        rhs: &StringRef,
        i_lhs: usize,
        i_rhs: usize,
    ) {
        let mut i_lhs_maybe = lhs.get_ann_part().lowest_set_bit_not_before(i_lhs);
        let mut i_rhs_maybe = rhs.get_cre_part().lowest_set_bit_not_before(i_rhs);
        while let (Some(i_lhs), Some(i_rhs)) = (i_lhs_maybe, i_rhs_maybe) {
            if i_lhs < i_rhs {
                // Move the annihilator right across the remaining LHS annihilators
                // and RHS creators, then insert it from the left of the output block.
                inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann + 1);
                inds.exchange += self.n_rhs_cre.saturating_sub(inds.rhs_cre);
                inds.exchange += self.insert_annihilation(i_branch, i_lhs);
                i_lhs_maybe = lhs.get_ann_part().lowest_set_bit_after(i_lhs);
                inds.lhs_ann += 1;
            } else if i_lhs > i_rhs {
                // Move the creator left across the remaining LHS annihilators,
                // then insert it from the right of the output creation block.
                inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann);
                inds.exchange += self.insert_creation(i_branch, i_rhs);
                i_rhs_maybe = rhs.get_cre_part().lowest_set_bit_after(i_rhs);
                inds.rhs_cre += 1;
            } else {
                // Bring the equal-mode operators together and apply a_i a_i^dagger
                // = 1 - a_i^dagger a_i. Recurse on the contracted contribution first.
                assert_eq!(i_lhs, i_rhs);
                let i = i_lhs;
                // Match the allocation count: a forced contraction has no high-rank branch.
                let is_bif = !Self::must_contract(lhs, rhs, i);
                let mut i_branch_next = i_branch;
                if is_bif {
                    i_branch_next &= !(1 << self.n_bif.saturating_sub(1 + inds.bif));
                    assert!(i_branch_next < self.strings.len());
                    inds.bif += 1;
                    /*
                     * copy the LHS^ and RHS_ strings to the next branch of the result if the views exist
                     */
                    self.strings.copy(i_branch_next, i_branch);
                }

                // Move the creator left to the matching annihilator.
                inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann + 1);
                self.do_branch(
                    i_branch_next,
                    inds.incremented(),
                    lhs,
                    rhs,
                    i_lhs + 1,
                    i_rhs + 1,
                );
                i_lhs_maybe = lhs.get_ann_part().lowest_set_bit_after(i_lhs);
                i_rhs_maybe = rhs.get_cre_part().lowest_set_bit_after(i_rhs);
                /*
                 * high-rank branch only from here, so first exit if the index is not bifurcating (i.e. only the low-
                 * rank branch is continued)
                 */
                if !is_bif {
                    return;
                }

                // The uncontracted contribution has the minus sign from anticommutation.
                inds.exchange += 1;
                inds.exchange += self.insert_creation(i_branch, i);
                // Move the annihilator right through the remaining middle blocks.
                inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann + 1);
                inds.exchange += self.n_rhs_cre.saturating_sub(inds.rhs_cre + 1);
                inds.exchange += self.insert_annihilation(i_branch, i);

                inds.lhs_ann += 1;
                inds.rhs_cre += 1;
            }
        }
        while let Some(i_lhs) = i_lhs_maybe {
            // No RHS creators remain; move past the remaining LHS annihilators.
            inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann + 1);
            inds.exchange += self.insert_annihilation(i_branch, i_lhs);
            inds.lhs_ann += 1;
            i_lhs_maybe = lhs.get_ann_part().lowest_set_bit_after(i_lhs);
        }
        while let Some(i_rhs) = i_rhs_maybe {
            // No LHS annihilators remain; insert the remaining creators directly.
            inds.exchange += self.insert_creation(i_branch, i_rhs);
            inds.rhs_cre += 1;
            i_rhs_maybe = rhs.get_cre_part().lowest_set_bit_after(i_rhs);
        }

        assert_eq!(inds.lhs_ann, self.n_lhs_ann); //should have no LHS_ modes left to handle
        assert_eq!(inds.rhs_cre, self.n_rhs_cre); //should have no RHS^ modes left to handle
        self.signs
            .set_unchecked(i_branch, Sign(inds.exchange & 1 == 1));
    }
}

// Multiply two fermion components, returning the normal-ordered result terms and their signs.
pub fn mul_strings(lhs: &StringRef, rhs: &StringRef) -> (StringList, SignVec) {
    let mut helper = ProductHelper::new();
    helper.set(lhs, rhs);
    (helper.strings, helper.signs)
}

/// Compute the number of set bits of `word` at positions strictly below each set bit of `at`, summed together.
/// This is the anti-symmetric exchange count picked up by inserting or removing, one at a time in descending
/// mode-index order, the ladder operators flagged in `at` against the fixed occupation pattern `word`.
fn count_before_each_set_bit(word: u64, mut at: u64) -> u32 {
    let mut count = 0;
    while at != 0 {
        let lowest = at & at.wrapping_neg();
        count += (word & (lowest - 1)).count_ones();
        at &= at - 1;
    }
    count
}

/// Result of applying a single normal-ordered ladder operator component to a Slater determinant ket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyResult {
    Applied(u64, Sign),
    /// The fermionic exclusion principle forbids the term, i.e. if `ann` is not a subset of `ket`,
    /// or if any mode flagged in `cre` is already occupied after removing the modes in `ann`.
    Zero,
}

// Apply a single normal-ordered ladder operator component, given as its creation and annihilation mode bitsets,
/// to a Slater determinant ket occupation bitset.
/// Returns `ApplyResult::Zero` if the fermionic exclusion principle forbids the term, i.e. if `ann` is not a subset of `ket`,
/// or if any mode flagged in `cre` is already occupied after removing the modes in `ann`. Otherwise, returns the
/// resulting occupation bitset together with the anti-symmetric exchange sign picked up by normal-ordering the
/// application, assuming both `cre` and `ann` list their modes in ascending order.
pub fn apply_op_ket_u64(cre: u64, ann: u64, ket: u64) -> ApplyResult {
    if ket & ann != ann {
        return ApplyResult::Zero;
    }
    let ket_after_ann = ket & !ann;
    if cre & ket_after_ann != 0 {
        return ApplyResult::Zero;
    }
    let exponent =
        count_before_each_set_bit(ket, ann) + count_before_each_set_bit(ket_after_ann, cre);
    ApplyResult::Applied(ket_after_ann | cre, Sign(exponent & 1 == 1))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use rstest::rstest;

    // Construct the action on occupation states directly, without normal ordering.
    fn occupation_matrix(cre: &[usize], ann: &[usize], n_mode: usize) -> Vec<i32> {
        let dimension = 1 << n_mode;
        let mut matrix = vec![0; dimension * dimension];
        for column in 0..dimension {
            let mut occupied: Vec<bool> =
                (0..n_mode).map(|mode| column & (1 << mode) != 0).collect();
            let mut amplitude = 1;
            // The rightmost annihilator acts first, followed by the creators in reverse order.
            for (mode, creation) in ann
                .iter()
                .rev()
                .map(|&mode| (mode, false))
                .chain(cre.iter().rev().map(|&mode| (mode, true)))
            {
                if occupied[mode] == creation {
                    amplitude = 0;
                    break;
                }
                for &lower_occupied in &occupied[..mode] {
                    if lower_occupied {
                        amplitude = -amplitude;
                    }
                }
                occupied[mode] = creation;
            }
            let row: usize = occupied
                .iter()
                .enumerate()
                .filter(|(_, present)| **present)
                .map(|(mode, _)| 1 << mode)
                .sum();
            matrix[row * dimension + column] = amplitude;
        }
        matrix
    }

    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    fn products_match_occupation_matrices(#[case] n_mode: usize) {
        use crate::container::traits::proj::Borrow;
        use crate::fermion::operator::normal::string::String;

        let dimension = 1 << n_mode;
        let mut operators = Vec::new();
        for cre in 0..dimension {
            for ann in 0..dimension {
                let creations: Vec<_> = (0..n_mode).filter(|mode| cre & (1 << mode) != 0).collect();
                let annihilations: Vec<_> =
                    (0..n_mode).filter(|mode| ann & (1 << mode) != 0).collect();
                let matrix = occupation_matrix(&creations, &annihilations, n_mode);
                let operator = String::from_sets_unchecked(
                    Modes::from_count(n_mode),
                    creations.into_iter().collect(),
                    annihilations.into_iter().collect(),
                );
                operators.push((operator, matrix));
            }
        }

        for (lhs, lhs_matrix) in &operators {
            for (rhs, rhs_matrix) in &operators {
                let mut product = ProductHelper::new();
                product.set(&lhs.borrow(), &rhs.borrow());
                let mut actual = vec![0; dimension * dimension];
                for term in 0..product.strings.len() {
                    let creations: Vec<_> = product.res_cre(term).iter_set_bits_flat().collect();
                    let annihilations: Vec<_> =
                        product.res_ann(term).iter_set_bits_flat().collect();
                    let contribution = occupation_matrix(&creations, &annihilations, n_mode);
                    let sign = if product.signs.get_unchecked(term).0 {
                        -1
                    } else {
                        1
                    };
                    for (entry, value) in actual.iter_mut().zip(contribution) {
                        *entry += sign * value;
                    }
                }
                for row in 0..dimension {
                    for column in 0..dimension {
                        let expected: i32 = (0..dimension)
                            .map(|inner| {
                                lhs_matrix[row * dimension + inner]
                                    * rhs_matrix[inner * dimension + column]
                            })
                            .sum();
                        assert_eq!(
                            actual[row * dimension + column],
                            expected,
                            "{lhs} * {rhs}, matrix entry ({row}, {column})"
                        );
                    }
                }
            }
        }
    }

    #[rstest]
    // Distinct number operators commute, including across storage words.
    #[case(66, (vec![0], vec![0]), (vec![65], vec![65]),
        vec![vec![0, 65]], vec![vec![0, 65]], vec![true])]
    #[case(66, (vec![65], vec![65]), (vec![0], vec![0]),
        vec![vec![0, 65]], vec![vec![0, 65]], vec![true])]
    // Forced contractions must not leave unused identity rows.
    #[case(2, (vec![0], vec![0]), (vec![0], vec![]),
        vec![vec![0]], vec![vec![]], vec![false])]
    #[case(2, (vec![], vec![0]), (vec![0], vec![0]),
        vec![vec![]], vec![vec![0]], vec![false])]
    // Number operators are projectors: n_0^2 = n_0.
    #[case(2, (vec![0], vec![0]), (vec![0], vec![0]),
        vec![vec![0]], vec![vec![0]], vec![false])]
    // T = a_0^dagger a_1^dagger a_0 a_1 = -n_0 n_1, so T^2 = -T.
    #[case(2, (vec![0, 1], vec![0, 1]), (vec![0, 1], vec![0, 1]),
        vec![vec![0, 1]], vec![vec![0, 1]], vec![true])]
    // A forced contraction followed by a genuine split produces exactly two rows.
    #[case(2, (vec![0], vec![0, 1]), (vec![0, 1], vec![]),
        vec![vec![0], vec![0, 1]], vec![vec![], vec![1]], vec![true, false])]
    // Count forced contractions across storage words.
    #[case(66, (vec![0, 65], vec![0, 65]), (vec![0, 65], vec![0, 65]),
        vec![vec![0, 65]], vec![vec![0, 65]], vec![true])]
    // Seventy forced contractions need one row, not a shift by seventy bits.
    #[case(70, ((0..70).collect(), (0..70).collect()),
        ((0..70).collect(), (0..70).collect()),
        vec![(0..70).collect()], vec![(0..70).collect()], vec![true])]
    // Empty operator products remain identity on either side.
    #[case(2, (vec![], vec![]), (vec![0], vec![1]),
        vec![vec![0]], vec![vec![1]], vec![false])]
    #[case(2, (vec![0], vec![1]), (vec![], vec![]),
        vec![vec![0]], vec![vec![1]], vec![false])]
    // creation ops only
    #[case(6, (vec![0, 1], vec![]), (vec![2, 3], vec![]),
        vec![vec![0, 1, 2, 3]], vec![vec![]], vec![false])]
    #[case(6, (vec![0, 2], vec![]), (vec![1, 3], vec![]),
        vec![vec![0, 1, 2, 3]], vec![vec![]], vec![true])]
    #[case(6, (vec![0, 3], vec![]), (vec![1, 2], vec![]),
        vec![vec![0, 1, 2, 3]], vec![vec![]], vec![false])]
    #[case(6, (vec![2, 3], vec![]), (vec![0, 1], vec![]),
        vec![vec![0, 1, 2, 3]], vec![vec![]], vec![false])]
    // annihilation ops only
    #[case(6, (vec![], vec![0, 1]), (vec![], vec![2, 3]),
        vec![vec![]], vec![vec![0, 1, 2, 3]], vec![false])]
    #[case(6, (vec![], vec![0, 2]), (vec![], vec![1, 3]),
        vec![vec![]], vec![vec![0, 1, 2, 3]], vec![true])]
    #[case(6, (vec![], vec![0, 3]), (vec![], vec![1, 2]),
        vec![vec![]], vec![vec![0, 1, 2, 3]], vec![false])]
    #[case(6, (vec![], vec![2, 3]), (vec![], vec![0, 1]),
        vec![vec![]], vec![vec![0, 1, 2, 3]], vec![false])]
    // coincident out-of-order indices
    #[case(6, (vec![], vec![0]), (vec![0], vec![]),
        vec![vec![], vec![0]],
        vec![vec![], vec![0]],
        vec![false, true])]
    #[case(6, (vec![], vec![0, 1]), (vec![0, 1], vec![]),
        vec![vec![], vec![1], vec![0], vec![0, 1]],
        vec![vec![], vec![1], vec![0], vec![0, 1]],
        vec![true, false, false, false])]
    #[case(6, (vec![], vec![0, 1, 2]), (vec![0, 1, 2], vec![]),
        vec![vec![], vec![2], vec![1], vec![1, 2], vec![0], vec![0, 2], vec![0, 1], vec![0, 1, 2]],
        vec![vec![], vec![2], vec![1], vec![1, 2], vec![0], vec![0, 2], vec![0, 1], vec![0, 1, 2]],
        vec![true, false, false, false, false, false, false, true])]
    #[case(6, (vec![0], vec![1, 2]), (vec![1, 2], vec![]),
        vec![vec![0], vec![0, 2], vec![0, 1], vec![0, 1, 2]],
        vec![vec![], vec![2], vec![1], vec![1, 2]],
        vec![true, false, false, false])]
    #[case(6, (vec![0, 3, 4], vec![1, 2]), (vec![1, 2], vec![]),
        vec![vec![0, 3, 4], vec![0, 2, 3, 4], vec![0, 1, 3, 4], vec![0, 1, 2, 3, 4]],
        vec![vec![], vec![2], vec![1], vec![1, 2]],
        vec![true, false, false, false])]
    #[case(6, (vec![3], vec![1, 2]), (vec![1, 2], vec![]),
        vec![vec![3], vec![2, 3], vec![1, 3], vec![1, 2, 3]],
        vec![vec![], vec![2], vec![1], vec![1, 2]],
        vec![true, true, true, false])]
    #[case(6, (vec![], vec![0, 1]), (vec![0, 1], vec![2]),
        vec![vec![], vec![1], vec![0], vec![0, 1]],
        vec![vec![2], vec![1, 2], vec![0, 2], vec![0, 1, 2]],
        vec![true, false, false, false])]
    #[case(6, (vec![], vec![0, 1, 2]), (vec![0, 1], vec![]),
        vec![vec![], vec![1], vec![0], vec![0, 1]],
        vec![vec![2], vec![1, 2], vec![0, 2], vec![0, 1, 2]],
        vec![true, false, false, false])]
    #[case(6, (vec![2], vec![0, 1]), (vec![0, 1], vec![3]),
        vec![vec![2], vec![1, 2], vec![0, 2], vec![0, 1, 2]],
        vec![vec![3], vec![1, 3], vec![0, 3], vec![0, 1, 3]],
        vec![true, true, true, false])]
    #[case(6, (vec![], vec![0, 3]), (vec![0, 1, 3], vec![]),
        vec![vec![1], vec![1, 3], vec![0, 1], vec![0, 1, 3]],
        vec![vec![], vec![3], vec![0], vec![0, 3]],
        vec![false, true, false, false])]
    #[case(6, (vec![], vec![2, 3]), (vec![], vec![2, 1]),
        vec![], vec![], vec![])] // destruction criterion met with F2 F2 product
    #[case(10, (vec![4, 7], vec![0, 3, 4, 5]), (vec![0, 1, 3], vec![2, 6]),
        vec![vec![1, 4, 7], vec![1, 3, 4, 7], vec![0, 1, 4, 7], vec![0, 1, 3, 4, 7]],
        vec![vec![2, 4, 5, 6], vec![2, 3, 4, 5, 6], vec![0, 2, 4, 5, 6], vec![0, 2, 3, 4, 5, 6]],
        vec![false, false, false, true])]
    fn test_fermion_products(
        #[case] n_mode: usize,
        #[case] lhs: (Vec<usize>, Vec<usize>),
        #[case] rhs: (Vec<usize>, Vec<usize>),
        #[case] result_cres: Vec<Vec<usize>>,
        #[case] result_anns: Vec<Vec<usize>>,
        #[case] result_signs: Vec<bool>,
    ) {
        use crate::container::traits::proj::Borrow;
        use crate::container::traits::EmptyClone;
        use crate::fermion::operator::normal::string::String;

        let modes = Modes::from_count(n_mode);
        let (lhs_cre, lhs_ann) = lhs;
        let (rhs_cre, rhs_ann) = rhs;
        let lhs_cre = HashSet::from_iter(lhs_cre.into_iter());
        let lhs_ann = HashSet::from_iter(lhs_ann.into_iter());
        let rhs_cre = HashSet::from_iter(rhs_cre.into_iter());
        let rhs_ann = HashSet::from_iter(rhs_ann.into_iter());
        let mut helper = ProductHelper::new();
        let lhs = String::from_sets_unchecked(modes.clone(), lhs_cre, lhs_ann);
        let rhs = String::from_sets_unchecked(modes.clone(), rhs_cre, rhs_ann);
        helper.set(&lhs.borrow(), &rhs.borrow());
        let mut result_strings = helper.strings.empty_clone();
        for (cre, ann) in result_cres.into_iter().zip(result_anns.into_iter()) {
            let index = result_strings.len();
            let cre = HashSet::from_iter(cre.into_iter());
            let ann = HashSet::from_iter(ann.into_iter());
            result_strings.push_clear();
            result_strings
                .cre_part
                .get_elem_mut_ref(index)
                .assign_set_unchecked(cre);
            result_strings
                .ann_part
                .get_elem_mut_ref(index)
                .assign_set_unchecked(ann);
        }
        let result_signs = SignVec::from_phases(&result_signs);
        assert_eq!(helper.strings, result_strings);
        assert_eq!(helper.signs, result_signs);
    }

    #[rstest]
    // c_0^+ |vac> = |0>
    #[case(0b1, 0b0, 0b0, Some((0b1, false)))]
    // c_1^+ c_0 |1_0> = |1_1>, no modes in between to pick up a sign.
    #[case(0b10, 0b1, 0b1, Some((0b10, false)))]
    // c_2^+ c_0 |1_0 1_1> = -|1_1 1_2>, one occupied mode (1) sits strictly between 0 and 2.
    #[case(0b100, 0b1, 0b011, Some((0b110, true)))]
    // c_0^+ c_0 |1_0> = |1_0>, the number operator on an occupied mode leaves the ket unchanged.
    #[case(0b1, 0b1, 0b1, Some((0b1, false)))]
    // annihilating an unoccupied mode is forbidden.
    #[case(0b0, 0b1, 0b0, None)]
    // creating into an already occupied mode is forbidden.
    #[case(0b1, 0b0, 0b1, None)]
    fn test_apply_op_ket_u64(
        #[case] cre: u64,
        #[case] ann: u64,
        #[case] ket: u64,
        #[case] expected: Option<(u64, bool)>,
    ) {
        let expected = match expected {
            Some((bits, sign)) => ApplyResult::Applied(bits, Sign(sign)),
            None => ApplyResult::Zero,
        };
        let result = apply_op_ket_u64(cre, ann, ket);
        assert_eq!(result, expected);
    }
}
