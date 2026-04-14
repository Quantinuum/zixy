//! Sorting utilities for word iterators including those with associated coefficients.

use crate::container::coeffs::traits::{FieldElem, NumRepr, NumReprVec};
use crate::container::coeffs::unity::{Unity, UnityVec};
use crate::container::traits::{proj, NewWithLen, RefElements};
use crate::container::word_iters::{terms, ElemRef, WordIters};

/// In-place quicksort strategy for [`WordIters`] containers and a simultaneously-permuted coefficient array.
pub trait QuickSort<T: WordIters, C: NumRepr> {
    /// Compare two `(element, coeff)` pairs according to this sorter’s ordering criterion.
    fn cmp(&self, lhs: (ElemRef<T>, C), rhs: (ElemRef<T>, C)) -> std::cmp::Ordering;

    /// Partition `elements[lo..=hi]` around the pivot at `hi`, swapping `coeffs` simultaneously.
    fn partition(&self, elements: &mut T, coeffs: &mut C::Vector, lo: usize, hi: usize) -> usize {
        let mut i = lo;
        for j in lo..hi {
            let ordering = {
                let elements = &*elements;
                let lhs = (elements.get_elem_ref(j), coeffs.get_unchecked(j));
                let rhs = (elements.get_elem_ref(hi), coeffs.get_unchecked(hi));
                self.cmp(lhs, rhs)
            };
            if ordering.is_le() {
                elements.swap(i, j);
                coeffs.swap_unchecked(i, j);
                i += 1;
            }
        }
        elements.swap(i, hi);
        coeffs.swap_unchecked(i, hi);
        i
    }

    /// Quicksort the inclusive range `lo..=hi`, swapping `coeffs` simultaneously.
    fn quicksort_range(&self, elements: &mut T, coeffs: &mut C::Vector, lo: usize, hi: usize) {
        if (lo + 1) > (hi + 1) {
            return;
        }
        let pi = self.partition(elements, coeffs, lo, hi);
        if pi != 0 {
            self.quicksort_range(elements, coeffs, lo, pi - 1);
        }
        self.quicksort_range(elements, coeffs, pi + 1, hi);
    }

    /// Sort `elements` in place and apply the same permutation to `coeffs`.
    fn sort_with_coeffs(&self, elements: &mut T, coeffs: &mut C::Vector) {
        if !elements.is_empty() {
            self.quicksort_range(elements, coeffs, 0, elements.len().saturating_sub(1));
        }
    }

    /// Sort a terms-like view in place using this sorter.
    fn sort_list_with_coeffs(&self, list: &mut impl terms::AsViewMut<T, C>) {
        let list = list.view_mut();
        self.sort_with_coeffs(list.word_iters, list.coeffs);
    }

    /// Return a sorted owned copy of `list`.
    fn sorted_list_with_coeffs(&self, list: &impl terms::AsView<T, C>) -> terms::Terms<T, C> {
        use proj::EmptyOwned;
        let list = list.view();
        let mut out = list.empty_owned();
        self.sort_list_with_coeffs(&mut out);
        out
    }
}

/// Convenience extension for quicksort strategies that can operate [`WordIters`] data without associated coefficients.
pub trait QuickSortNoCoeffs<T: WordIters>: QuickSort<T, Unity> {
    /// Sort `elements` with no associated coefficient vector.
    fn sort(&self, elements: &mut T) {
        let mut dummy_coeffs = UnityVec::new_with_len(elements.len());
        self.sort_with_coeffs(elements, &mut dummy_coeffs);
    }
}

/// Sorter that orders words lexicographically, either ascending or descending.
pub struct LexicographicSort {
    pub ascending: bool,
}

impl<T: WordIters, C: NumRepr> QuickSort<T, C> for LexicographicSort {
    fn cmp(&self, lhs: (ElemRef<T>, C), rhs: (ElemRef<T>, C)) -> std::cmp::Ordering {
        if self.ascending {
            lhs.0.cmp(&rhs.0)
        } else {
            rhs.0.cmp(&lhs.0)
        }
    }
}

impl<T: WordIters> QuickSortNoCoeffs<T> for LexicographicSort {}

/// Sorter for real-valued coefficients, optionally comparing by absolute value instead of signed value.
pub struct RealSort {
    pub ascending: bool,
    pub by_magnitude: bool,
}

impl<T: WordIters> QuickSort<T, f64> for RealSort {
    fn cmp(&self, lhs: (ElemRef<T>, f64), rhs: (ElemRef<T>, f64)) -> std::cmp::Ordering {
        if self.ascending {
            if self.by_magnitude {
                PartialOrd::partial_cmp(&lhs.1.abs(), &rhs.1.abs()).unwrap()
            } else {
                PartialOrd::partial_cmp(&lhs.1, &rhs.1).unwrap()
            }
        } else if self.by_magnitude {
            PartialOrd::partial_cmp(&rhs.1.abs(), &lhs.1.abs()).unwrap()
        } else {
            PartialOrd::partial_cmp(&rhs.1, &lhs.1).unwrap()
        }
    }
}
/// Sorter that orders coefficients by their magnitude.
pub struct MagnitudeSort {
    pub ascending: bool,
}

impl<T: WordIters, C: FieldElem> QuickSort<T, C> for MagnitudeSort {
    fn cmp(&self, lhs: (ElemRef<T>, C), rhs: (ElemRef<T>, C)) -> std::cmp::Ordering {
        if self.ascending {
            PartialOrd::partial_cmp(&lhs.1.magnitude(), &rhs.1.magnitude()).unwrap()
        } else {
            PartialOrd::partial_cmp(&rhs.1.magnitude(), &lhs.1.magnitude()).unwrap()
        }
    }
}
