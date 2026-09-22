//! Bulk numeric term collection shared by the Python component arrays.

use pyo3::{exceptions::PyValueError, PyResult};
use zixy::container::coeffs::traits::FieldElem;
use zixy::container::word_iters::set::AsView;
use zixy::container::word_iters::{set, WordIters};

use super::map::Map;

/// Accumulate numeric terms without removing zeros or changing insertion order.
pub fn scaled_iadd<T: WordIters, C: FieldElem>(
    lhs: &mut T,
    coeffs: &mut Vec<C>,
    map: &mut Map,
    rhs: &T,
    rhs_coeffs: &[C],
    scalar: C,
    push: impl Fn(&mut T, &T, usize) -> PyResult<()>,
) -> PyResult<()> {
    if lhs.len() != coeffs.len() || rhs.len() != rhs_coeffs.len() {
        return Err(PyValueError::new_err(
            "Component and coefficient counts must match.",
        ));
    }
    if !lhs.compatible_with(rhs) {
        return Err(PyValueError::new_err("Incompatible component spaces."));
    }
    // Normalize the source to the destination layout and validate capacity before
    // mutating the destination. General fermion arrays may have different max_string_len.
    let mut source = lhs.empty_clone();
    for i in 0..rhs.len() {
        push(&mut source, rhs, i)?;
    }
    for (i, &coeff) in rhs_coeffs.iter().enumerate() {
        let found = (set::View {
            word_iters: lhs,
            map: &map.0,
        })
        .lookup(source.elem_u64it(i));
        if let Some(index) = found {
            coeffs[index] += scalar * coeff;
        } else {
            let index = coeffs.len();
            push(lhs, &source, i)?;
            map.0.insert(lhs.hash_at_index(index), index);
            coeffs.push(scalar * coeff);
        }
    }
    Ok(())
}

/// Copy or collect terms while preserving first-occurrence order.
pub fn collect<T: WordIters, C: FieldElem>(
    source: &T,
    coeffs: &[C],
    nonzero: bool,
    combine: Option<bool>,
    push: impl Fn(&mut T, &T, usize) -> PyResult<()>,
) -> PyResult<(T, Map, Vec<C>)> {
    if source.len() != coeffs.len() {
        return Err(PyValueError::new_err(
            "Component and coefficient counts must match.",
        ));
    }
    let mut out = source.empty_clone();
    let mut map = Map(Default::default());
    let mut values = Vec::with_capacity(coeffs.len());
    for (i, &value) in coeffs.iter().enumerate() {
        if nonzero && value == C::default() {
            continue;
        }
        if let Some(sum) = combine {
            if let Some(index) = (set::View {
                word_iters: &out,
                map: &map.0,
            })
            .lookup(source.elem_u64it(i))
            {
                if sum {
                    values[index] += value;
                } else {
                    values[index] = value;
                }
                continue;
            }
        }
        push(&mut out, source, i)?;
        let index = values.len();
        map.0.insert(out.hash_at_index(index), index);
        values.push(value);
    }
    Ok((out, map, values))
}

/// Bind bulk copying and duplicate collection for numeric component arrays.
macro_rules! numeric_terms {
    ($array:ty, $push:expr) => {
        #[pyo3::pymethods]
        impl $array {
            /// Add scaled real terms in-place, retaining zeros until cleanup.
            fn scaled_iadd_real(
                &mut self,
                coeffs: &mut crate::container::coeffs::RealVec,
                map: &mut crate::container::map::Map,
                rhs: &Self,
                rhs_coeffs: &crate::container::coeffs::RealVec,
                scalar: f64,
            ) -> pyo3::PyResult<()> {
                crate::container::terms::scaled_iadd(
                    &mut self.0,
                    &mut coeffs.0,
                    map,
                    &rhs.0,
                    &rhs_coeffs.0,
                    scalar,
                    $push,
                )
            }

            /// Add scaled complex terms in-place, retaining zeros until cleanup.
            fn scaled_iadd_complex(
                &mut self,
                coeffs: &mut crate::container::coeffs::ComplexVec,
                map: &mut crate::container::map::Map,
                rhs: &Self,
                rhs_coeffs: &crate::container::coeffs::ComplexVec,
                scalar: num_complex::Complex64,
            ) -> pyo3::PyResult<()> {
                crate::container::terms::scaled_iadd(
                    &mut self.0,
                    &mut coeffs.0,
                    map,
                    &rhs.0,
                    &rhs_coeffs.0,
                    scalar,
                    $push,
                )
            }

            /// Copy real terms, optionally omitting exact zeros.
            fn copy_terms_real(
                &self,
                coeffs: &crate::container::coeffs::RealVec,
                nonzero: bool,
            ) -> pyo3::PyResult<(
                Self,
                crate::container::map::Map,
                crate::container::coeffs::RealVec,
            )> {
                let (array, map, values) =
                    crate::container::terms::collect(&self.0, &coeffs.0, nonzero, None, $push)?;
                Ok((Self(array), map, crate::container::coeffs::RealVec(values)))
            }
            /// Copy complex terms, optionally omitting exact zeros.
            fn copy_terms_complex(
                &self,
                coeffs: &crate::container::coeffs::ComplexVec,
                nonzero: bool,
            ) -> pyo3::PyResult<(
                Self,
                crate::container::map::Map,
                crate::container::coeffs::ComplexVec,
            )> {
                let (array, map, values) =
                    crate::container::terms::collect(&self.0, &coeffs.0, nonzero, None, $push)?;
                Ok((
                    Self(array),
                    map,
                    crate::container::coeffs::ComplexVec(values),
                ))
            }
            /// Collect real terms, adding duplicates or keeping their last coefficient.
            fn collect_terms_real(
                &self,
                coeffs: &crate::container::coeffs::RealVec,
                sum: bool,
            ) -> pyo3::PyResult<(
                Self,
                crate::container::map::Map,
                crate::container::coeffs::RealVec,
            )> {
                let (array, map, values) =
                    crate::container::terms::collect(&self.0, &coeffs.0, false, Some(sum), $push)?;
                Ok((Self(array), map, crate::container::coeffs::RealVec(values)))
            }
            /// Collect complex terms, adding duplicates or keeping their last coefficient.
            fn collect_terms_complex(
                &self,
                coeffs: &crate::container::coeffs::ComplexVec,
                sum: bool,
            ) -> pyo3::PyResult<(
                Self,
                crate::container::map::Map,
                crate::container::coeffs::ComplexVec,
            )> {
                let (array, map, values) =
                    crate::container::terms::collect(&self.0, &coeffs.0, false, Some(sum), $push)?;
                Ok((
                    Self(array),
                    map,
                    crate::container::coeffs::ComplexVec(values),
                ))
            }
        }
    };
}

pub(crate) use numeric_terms;
