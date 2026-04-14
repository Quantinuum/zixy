//! Trait definitions for `WordIters`-implementing types.

use num_complex::Complex64;

use crate::container::coeffs::traits::{HasCoeffs, NumRepr, NumReprVec};
use crate::container::traits::EmptyFrom;
use crate::container::word_iters::{HasWordIters, HasWordItersMut, WordIters};

/// Exposes a common interface for all types that can have iters over u64s inserted into them.
pub trait InsertU64It<C: NumRepr> {
    /// Insert an iterator over `u64` ints with an associated coefficient.
    fn insert_u64it(&mut self, iter: impl Iterator<Item = u64> + Clone, c: C) -> Option<usize>;

    /// Insert an iterator over `u64` ints with a default coefficient.
    fn insert_u64it_default(&mut self, iter: impl Iterator<Item = u64> + Clone) -> Option<usize> {
        self.insert_u64it(iter, C::default())
    }

    /// Insertion that only sets the coefficient if the insertion is successful - does not overwrite if the insertion found an existing record.
    fn soft_insert_u64it(
        &mut self,
        iter: impl Iterator<Item = u64> + Clone,
        c: C,
    ) -> Option<(usize, bool)> {
        self.insert_u64it(iter, c).map(|i| (i, true))
    }
}

/// Transform all coefficients out-of-place, including changes of type.
pub trait TransformCoeffs<T: WordIters, InpC: NumRepr, OutC: NumRepr>:
    HasWordIters<T> + HasCoeffs<InpC>
{
    /// Output container type produced after re-encoding the same word iterators with coefficient type `OutC`.
    type Output: InsertU64It<OutC> + EmptyFrom<T> + HasWordItersMut<T>;

    /// Get a new instance with the same IterableElements contents, but with coeffs transformed by function f.
    fn transformed_coeffs<F>(&self, f: F) -> Self::Output
    where
        F: Fn(InpC) -> OutC,
    {
        let inp_iter_elems = self.get_word_iters();
        let inp_coeffs = self.get_coeffs();
        let mut out = Self::Output::empty_from(inp_iter_elems);
        for i in 0..self.len() {
            out.insert_u64it(inp_iter_elems.elem_u64it(i), f(inp_coeffs.get_unchecked(i)));
        }
        out
    }

    /// Transform complex valued coeffs by taking only the real part and copying into a real valued coeff container.
    fn real_part(&self) -> Self::Output
    where
        InpC: Into<Complex64>,
        OutC: From<f64>,
    {
        self.transformed_coeffs(|x| OutC::from(Into::<Complex64>::into(x).re))
    }

    /// Transform complex valued coeffs by taking only the imag part and copying into a real valued coeff container.
    fn imag_part(&self) -> Self::Output
    where
        InpC: Into<Complex64>,
        OutC: From<f64>,
    {
        self.transformed_coeffs(|x| OutC::from(Into::<Complex64>::into(x).im))
    }

    /// Represent coeffs as complex numbers.
    fn to_complex(&self) -> Self::Output
    where
        OutC: From<Complex64>,
    {
        self.transformed_coeffs(|x| OutC::from(x.to_complex()))
    }
}
