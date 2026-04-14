//! Fermion operator products module.

#[cfg(test)]
use crate::container::bit_matrix::{AsRowMutRef, AsRowRef};
#[cfg(test)]
use crate::container::coeffs::sign::{Sign, SignVec};
#[cfg(test)]
use crate::container::coeffs::traits::NumReprVec;
#[cfg(test)]
use crate::container::traits::{Elements, MutRefElements, RefElements};
#[cfg(test)]
use crate::container::word_iters::{ElemMutRef, ElemRef, WordIters};
#[cfg(test)]
use crate::fermion::mode::Modes;
#[cfg(test)]
use crate::fermion::operator::cmpnt_list::{CmpntList, CmpntRef};
#[cfg(test)]
use crate::fermion::operator::cre_or_ann;

/// Stores variables to support the recursive computation of normal-ordered fermion operator products.
#[cfg(test)]
struct ProductHelper {
    cmpnts: CmpntList,
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
#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct Indices {
    bif: usize,
    exchange: usize,
    lhs_ann: usize,
    rhs_cre: usize,
}

#[cfg(test)]
impl Indices {
    /// Return a copy with both processed common-mode counters advanced by one.
    fn incremented(&self) -> Self {
        let mut out = *self;
        out.lhs_ann += 1;
        out.rhs_cre += 1;
        out
    }
}

#[cfg(test)]
impl ProductHelper {
    /// Create an empty helper for building products of operators on the given fermionic mode space.
    pub fn new(modes: Modes) -> Self {
        Self {
            cmpnts: CmpntList::new(modes),
            signs: SignVec::default(),
            n_lhs_ann: 0,
            n_rhs_ann: 0,
            n_lhs_cre: 0,
            n_rhs_cre: 0,
            n_bif: 0,
        }
    }

    /// Return whether the product `lhs * rhs` is annihilated by fermionic anticommutation rules.
    fn destroys(lhs: &CmpntRef, rhs: &CmpntRef) -> bool {
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

    /// Compute and store every branch of the product `lhs * rhs`, including their signs.
    pub fn set(&mut self, lhs: &CmpntRef, rhs: &CmpntRef) {
        self.cmpnts.clear();
        self.signs.clear();
        if Self::destroys(lhs, rhs) {
            return;
        }
        let n_bif = lhs
            .get_ann_part()
            .intersection_hamming_weight(&rhs.get_cre_part());
        self.cmpnts.resize(1 << n_bif);
        self.signs.resize(self.cmpnts.len());
        self.n_lhs_ann = lhs.get_ann_part().hamming_weight();
        self.n_rhs_ann = rhs.get_ann_part().hamming_weight();
        self.n_lhs_cre = lhs.get_cre_part().hamming_weight();
        self.n_rhs_cre = rhs.get_cre_part().hamming_weight();
        self.n_bif = rhs
            .get_cre_part()
            .intersection_hamming_weight(&lhs.get_ann_part());
        // copy the LHS^ and RHS_ strings to the last element of the result
        let i_branch_last = self.cmpnts.len().saturating_sub(1);
        // these are the parts that are already in normal order
        self.res_cre_mut(i_branch_last).assign(lhs.get_cre_part());
        self.res_ann_mut(i_branch_last).assign(rhs.get_ann_part());
        self.do_branch(
            self.cmpnts.len().saturating_sub(1),
            Indices::default(),
            lhs,
            rhs,
            0,
            0,
        );
    }

    /// Return the annihilation-part result component for branch `branch`.
    fn res_ann(&self, branch: usize) -> ElemRef<'_, cre_or_ann::CmpntList> {
        self.cmpnts.ann_part.get_elem_ref(branch)
    }

    /// Return mutable access to the annihilation-part result component for branch `branch`.
    fn res_ann_mut(&mut self, branch: usize) -> ElemMutRef<'_, cre_or_ann::CmpntList> {
        self.cmpnts.ann_part.get_elem_mut_ref(branch)
    }

    /// Return the creation-part result component for branch `branch`.
    fn res_cre(&self, branch: usize) -> ElemRef<'_, cre_or_ann::CmpntList> {
        self.cmpnts.cre_part.get_elem_ref(branch)
    }

    /// Return mutable access to the creation-part result component for branch `branch`.
    fn res_cre_mut(&mut self, branch: usize) -> ElemMutRef<'_, cre_or_ann::CmpntList> {
        self.cmpnts.cre_part.get_elem_mut_ref(branch)
    }

    /// Handle a branch (recursively) of the normal-ordering product operation.
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
        lhs: &CmpntRef,
        rhs: &CmpntRef,
        i_lhs: usize,
        i_rhs: usize,
    ) {
        let mut i_lhs_maybe = lhs.get_ann_part().lowest_set_bit_not_before(i_lhs);
        let mut i_rhs_maybe = rhs.get_cre_part().lowest_set_bit_not_before(i_rhs);
        while let (Some(i_lhs), Some(i_rhs)) = (i_lhs_maybe, i_rhs_maybe) {
            if i_lhs < i_rhs {
                /*
                 * LHS_ index gets moved to RHS_, not a common mode index. The relevant portion of the current state of
                 * the product is:
                 *  L_(0) L_(1) ... L_(n_la-1) R^(0) R^(1) ... R^(n_rc-1) RHS_
                 * which can be thought of as being manipulated in 3 steps:
                 * moving L_(0) to:
                 *                            A
                 * makes n_la-1 exchanges,
                 * moving L_(0) to
                 *                                                        B
                 * makes a further n_rc exchanges,
                 * and moving L_(0) to its position within RHS_
                 *                                                           C
                 * makes a further number of exchanges equal to the number of set bits after L_(0) in RHS_
                 */
                // anti-commute to A
                inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann + 1);
                // anti-commute to B
                inds.exchange += self.n_rhs_cre.saturating_sub(inds.rhs_cre);
                // anti-commute to C
                inds.exchange += self.res_ann(i_branch).count_set_bits_after(i_lhs);
                self.res_ann_mut(i_branch).set_bit_unchecked(i_lhs, true);
                i_lhs_maybe = lhs.get_ann_part().lowest_set_bit_after(i_lhs);
                inds.lhs_ann += 1;
            } else if i_lhs > i_rhs {
                /*
                 * RHS^ index gets moved to LHS^, not a common mode index. The relevant portion of the current state of
                 * the product is:
                 *  LHS^ L_(0) L_(1) ... L_(n_la-1) R^(0) R^(1) ... R^(n_rc-1)
                 * which can be thought of as being manipulated in 2 steps:
                 * moving R^(0) to
                 *      A
                 * makes n_la exchanges,
                 * and moving R^(0) to its position within LHS^
                 *  B?
                 * makes a further number of exchanges equal to the number of set bits before R^(0) in LHS^
                 */
                // anti-commute to A
                inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann);
                // anti-commute to B
                inds.exchange += self.res_cre(i_branch).count_set_bits_before(i_rhs);
                self.res_cre_mut(i_branch).set_bit_unchecked(i_rhs, true);
                i_rhs_maybe = rhs.get_cre_part().lowest_set_bit_after(i_rhs);
                inds.rhs_cre += 1;
            } else {
                /*
                 * Common mode index.
                 * The current state of the product is (with L_(0) same mode as R^(0)):
                 *  LHS^ L_(0) L_(1) ... L_(n_la-1) R^(0) R^(1) ... R^(n_rc-1) RHS_
                 * which can be thought of as being manipulated in 6 steps:
                 * moving R^(0) to:
                 *            A
                 * makes n_la-1 exchanges,
                 * and moving R^(0) to:
                 *      B
                 * makes 1 exchange in the high-rank branch, and 0 exchanges in the low-rank branch
                 * at this point, call perform for the low-rank branch, with (n_la-1) exchanges
                 * moving R^(0) to its position within LHS^
                 *  C?
                 * makes a further number of exchanges equal to the number of set bits before R^(0) in LHS^
                 * moving L_(0) to:
                 *                                 D
                 * makes n_la-1 exchanges,
                 * moving L_(0) to:
                 *                                                             E
                 * makes n_rc-1 exchanges,
                 *
                 * and moving L_(0) to its position within RHS_
                 *                                                               F?
                 * makes a further number of exchanges equal to the number of set bits after L_(0) in RHS_
                 */
                assert_eq!(i_lhs, i_rhs);
                let i = i_lhs;
                /*
                 * only bifurcate if both LHS^ and RHS_ are given, and the mode index is not set in LHS^ or RHS_,
                 * otherwise, the high-rank branch is destroyed and only the low-rank branch survives
                 */
                let is_bif: bool = !lhs.get_cre_part().get_bit_unchecked(i)
                    && !rhs.get_ann_part().get_bit_unchecked(i);
                let mut i_branch_next = i_branch;
                if is_bif {
                    i_branch_next &= !(1 << self.n_bif.saturating_sub(1 + inds.bif));
                    assert!(i_branch_next < self.cmpnts.len());
                    inds.bif += 1;
                    /*
                     * copy the LHS^ and RHS_ strings to the next branch of the result if the views exist
                     */
                    self.cmpnts.copy(i_branch_next, i_branch);
                }

                // anti-commute to A
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

                // anti-commute to B
                inds.exchange += 1;
                // anti-commute to C
                inds.exchange += self.res_cre(i_branch).count_set_bits_after(i);
                // anti-commute to D
                inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann + 1);
                // anti-commute to E
                inds.exchange += self.n_rhs_cre.saturating_sub(inds.rhs_cre + 1);
                // anti-commute to F
                inds.exchange += self.res_ann(i_branch).count_set_bits_before(i);

                self.res_ann_mut(i_branch).set_bit_unchecked(i, true);
                self.res_cre_mut(i_branch).set_bit_unchecked(i, true);

                inds.lhs_ann += 1;
                inds.rhs_cre += 1;
            }
        }
        while let Some(i_lhs) = i_lhs_maybe {
            /*
             * all RHS^ modes have been processed, now finish off LHS_
             * just like the *lhs_ann_iter > *rhs_cre_iter case but with no modes in RHS^
             */
            inds.exchange += self.n_lhs_ann.saturating_sub(inds.lhs_ann + 1);
            inds.exchange += self.res_ann(i_branch).count_set_bits_before(i_lhs);
            self.res_ann_mut(i_branch).set_bit_unchecked(i_lhs, true);
            inds.lhs_ann += 1;
            i_lhs_maybe = lhs.get_ann_part().lowest_set_bit_after(i_lhs);
        }
        while let Some(i_rhs) = i_rhs_maybe {
            /*
             * all LHS_ modes have been processed, now finish off RHS^
             * just like the *lhs_ann_iter < *rhs_cre_iter case but with no modes in LHS_
             */
            inds.exchange += self.res_cre(i_branch).count_set_bits_after(i_rhs);
            self.res_cre_mut(i_branch).set_bit_unchecked(i_rhs, true);
            inds.rhs_cre += 1;
            i_rhs_maybe = rhs.get_cre_part().lowest_set_bit_after(i_rhs);
        }

        assert_eq!(inds.lhs_ann, self.n_lhs_ann); //should have no LHS_ modes left to handle
        assert_eq!(inds.rhs_cre, self.n_rhs_cre); //should have no RHS^ modes left to handle
        self.signs
            .set_unchecked(i_branch, Sign(inds.exchange & 1 == 1));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use rstest::rstest;

    #[rstest]
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
        vec![false, true, true, true])]
    #[case(6, (vec![], vec![2, 3]), (vec![], vec![2, 1]),
        vec![], vec![], vec![])] // destruction criterion met with F2 F2 product
    #[case(10, (vec![4, 7], vec![0, 3, 4, 5]), (vec![0, 1, 3], vec![2, 6]),
        vec![vec![1, 4, 7], vec![1, 3, 4, 7], vec![0, 1, 4, 7], vec![0, 1, 3, 4, 7]],
        vec![vec![2, 4, 5, 6], vec![2, 3, 4, 5, 6], vec![0, 2, 4, 5, 6], vec![0, 2, 3, 4, 5, 6]],
        vec![false, false, true, false])]
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
        use crate::fermion::operator::cmpnt::Cmpnt;

        let modes = Modes::from_count(n_mode);
        let (lhs_cre, lhs_ann) = lhs;
        let (rhs_cre, rhs_ann) = rhs;
        let lhs_cre = HashSet::from_iter(lhs_cre.into_iter());
        let lhs_ann = HashSet::from_iter(lhs_ann.into_iter());
        let rhs_cre = HashSet::from_iter(rhs_cre.into_iter());
        let rhs_ann = HashSet::from_iter(rhs_ann.into_iter());
        let mut helper = ProductHelper::new(modes.clone());
        let lhs = Cmpnt::from_sets_unchecked(modes.clone(), lhs_cre, lhs_ann);
        let rhs = Cmpnt::from_sets_unchecked(modes.clone(), rhs_cre, rhs_ann);
        helper.set(&lhs.borrow(), &rhs.borrow());
        let mut result_cmpnts = helper.cmpnts.empty_clone();
        for (cre, ann) in result_cres.into_iter().zip(result_anns.into_iter()) {
            let index = result_cmpnts.len();
            let cre = HashSet::from_iter(cre.into_iter());
            let ann = HashSet::from_iter(ann.into_iter());
            result_cmpnts.push_clear();
            result_cmpnts
                .cre_part
                .get_elem_mut_ref(index)
                .assign_set_unchecked(cre);
            result_cmpnts
                .ann_part
                .get_elem_mut_ref(index)
                .assign_set_unchecked(ann);
        }
        let result_signs = SignVec::from_phases(&result_signs);
        assert_eq!(helper.cmpnts, result_cmpnts);
        assert_eq!(helper.signs, result_signs);
    }
}
