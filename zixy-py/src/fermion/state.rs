//! Array of fermionic occupation-number state strings.

use std::collections::HashSet;

use num_complex::Complex64;
use pyo3::{pyclass, pymethods, PyResult};
use zixy::container::bit_matrix::{AsRowMutRef, AsRowRef};
use zixy::container::traits::{Compatible, Elements, EmptyClone, MutRefElements, RefElements};
use zixy::container::utils::DistinctPair;
use zixy::container::word_iters::set::{AsView as _, AsViewMut as _};
use zixy::container::word_iters::{self, WordIters};
use zixy::fermion::state;
use zixy::fermion::state::cmpnt_list::CmpntList;
use zixy::fermion::traits::{DifferentSpaces, ModesBased};

use crate::cmpnt::state_springs::BinarySprings;
use crate::container::coeffs::{ComplexVec, RealVec};
use crate::container::map::Map;
use crate::fermion::mode::Modes;
use crate::utils::{cmpnt_to_string, try_py_index, try_py_indices, ToPyResult};

/// A list of fermionic occupation-number basis state strings.
#[pyclass(subclass)]
#[pyo3(name = "FermionStateArray")]
#[repr(transparent)]
#[derive(Clone)]
pub struct Array(pub CmpntList);

impl Elements for Array {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl Default for Array {
    fn default() -> Self {
        Self::new(None, None).unwrap()
    }
}

impl EmptyClone for Array {
    fn empty_clone(&self) -> Self {
        Self(self.0.empty_clone())
    }
}

impl Array {
    /// Constructor.
    pub fn new(modes: Option<Modes>, springs: Option<BinarySprings>) -> PyResult<Self> {
        let springs = springs.unwrap_or_default();
        Ok(Self(match modes {
            Some(modes) => CmpntList::from_springs(modes.0, &springs.0).to_py_result()?,
            None => CmpntList::from_springs_default(&springs.0),
        }))
    }
}

#[pymethods]
impl Array {
    /// Constructor.
    #[new]
    #[pyo3(signature = (modes=None, springs=None))]
    pub fn __init__(modes: Option<Modes>, springs: Option<BinarySprings>) -> PyResult<Self> {
        Self::new(modes, springs)
    }

    /// Get the number of occupation-number basis strings in this array.
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Determine whether two arrays are equal.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    /// Return whether the two instances are the same.
    pub fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    /// Set the number of occupation-number basis strings in `self`.
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }

    /// Get the mode space on which this array is based.
    #[getter]
    pub fn get_modes(&self) -> Modes {
        Modes(self.0.to_modes())
    }

    /// Append a vacuum state string to the end of `self`.
    pub fn append_clear(&mut self) {
        self.0.push_clear();
    }

    /// Set the indexed component to the vacuum state string.
    pub fn cmpnt_clear(&mut self, i: isize) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).clear();
        Ok(())
    }

    /// Set the occupation on a single mode.
    pub fn cmpnt_set_bit(&mut self, i_cmpnt: isize, i_mode: isize, bit: bool) -> PyResult<()> {
        let n_mode = self.get_modes().0.len();
        self.0
            .get_elem_mut_ref(try_py_index(i_cmpnt, self.len())?)
            .set_bit(try_py_index(i_mode, n_mode)?, bit)
            .to_py_result()
    }

    /// Get the occupation on a single mode.
    pub fn cmpnt_get_bit(&mut self, i_cmpnt: isize, i_mode: isize) -> PyResult<bool> {
        let n_mode = self.get_modes().0.len();
        self.0
            .get_elem_ref(try_py_index(i_cmpnt, self.len())?)
            .get(try_py_index(i_mode, n_mode)?)
            .to_py_result()
    }

    /// Set the indexed component to the given sequence of occupation flags.
    pub fn cmpnt_set_from_list(&mut self, i: isize, bits: Vec<bool>) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).assign_vec(bits).to_py_result()
    }

    /// Set the indexed component from the set of occupied mode indices.
    pub fn cmpnt_set_from_set(&mut self, i: isize, bits: HashSet<usize>) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).assign_set(bits).to_py_result()
    }

    /// Set the indexed component with the indexed parsed spring.
    pub fn cmpnt_set_from_spring(
        &mut self,
        i: isize,
        src: &BinarySprings,
        i_src: usize,
    ) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0
            .get_elem_mut_ref(i)
            .set_spring(&src.0, i_src)
            .to_py_result()?;
        Ok(())
    }

    /// Get the indexed component as a list of occupation flags.
    pub fn cmpnt_get_list(&self, i: isize) -> PyResult<Vec<bool>> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).to_vec())
    }

    /// Get the indexed component as a set of occupied mode indices.
    pub fn cmpnt_get_set(&self, i: isize) -> PyResult<HashSet<usize>> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).to_set())
    }

    /// Copy a component within `self`.
    pub fn cmpnt_copy_internal(&mut self, i_dst: isize, i_src: isize) -> PyResult<()> {
        let i_dst = try_py_index(i_dst, self.len())?;
        let i_src = try_py_index(i_src, self.len())?;
        if let Some(inds) = DistinctPair::new(i_dst, i_src) {
            let (mut dst, src) = self.0.get_semi_mut_refs(inds);
            dst.assign(src);
        }
        Ok(())
    }

    /// Copy a component from another array.
    pub fn cmpnt_copy_external(&mut self, i_dst: isize, src: &Self, i_src: isize) -> PyResult<()> {
        DifferentSpaces::check(&self.0, &src.0).to_py_result()?;
        let i_dst = try_py_index(i_dst, self.len())?;
        let i_src = try_py_index(i_src, src.len())?;
        let mut dst = self.0.get_elem_mut_ref(i_dst);
        let src = src.0.get_elem_ref(i_src);
        dst.assign(src);
        Ok(())
    }

    /// Return whether the two referenced components are equal.
    pub fn cmpnt_equal(&self, i_lhs: usize, rhs: &Self, i_rhs: usize) -> bool {
        self.0.compatible_with(&rhs.0) && self.0.get_elem_ref(i_lhs) == rhs.0.get_elem_ref(i_rhs)
    }

    /// Get the number of occurrences of `bit` in the indexed component.
    pub fn cmpnt_count(&self, i: isize, bit: bool) -> PyResult<usize> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).count(bit))
    }

    /// Get the string representation of the indexed component.
    pub fn cmpnt_to_string(&self, i: isize) -> PyResult<String> {
        cmpnt_to_string(&self.0, i)
    }

    /// Clone selected components into a new array.
    pub fn cmpnts_clone(&self, indices: Option<Vec<isize>>) -> PyResult<Self> {
        Ok(match indices {
            Some(indices) => {
                let indices = try_py_indices(indices, self.len())?;
                let mut out = self.0.empty_clone();
                for i in indices.into_iter() {
                    out.push_elem_ref(self.0.get_elem_ref(i));
                }
                Self(out)
            }
            None => self.clone(),
        })
    }

    /// Determine equality of two slices.
    pub fn cmpnts_eq(
        &self,
        indices: Option<Vec<isize>>,
        other: &Self,
        other_indices: Option<Vec<isize>>,
    ) -> PyResult<bool> {
        if !self.0.compatible_with(&other.0) {
            return Ok(false);
        }
        let n = indices.as_ref().map_or(self.len(), Vec::len);
        if other_indices.as_ref().map_or(other.len(), Vec::len) != n {
            return Ok(false);
        }
        for i in 0..n {
            let l = self.0.get_elem_ref(match &indices {
                Some(v) => try_py_index(v[i], self.len())?,
                None => i,
            });
            let r = other.0.get_elem_ref(match &other_indices {
                Some(v) => try_py_index(v[i], other.len())?,
                None => i,
            });
            if l != r {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Clear and repopulate the map to reflect the current contents of `self`.
    pub fn refresh_map(&self, map: &mut Map) {
        map.0.populate_from(&self.0);
    }

    /// Insert into `self` using the map to ensure uniqueness.
    pub fn mapped_insert(&mut self, map: &mut Map, other: &Self, index: isize) -> PyResult<usize> {
        let index = try_py_index(index, other.len())?;
        let mut tmp = word_iters::set::ViewMut {
            word_iters: &mut self.0,
            map: &mut map.0,
        };
        Ok(tmp.insert_or_get_index(other.0.get_elem_ref(index)).0)
    }

    /// Find the index in `self` corresponding to the component indexed in `other`.
    pub fn mapped_lookup(&self, map: &Map, other: &Self, index: isize) -> PyResult<Option<usize>> {
        let index = try_py_index(index, other.len())?;
        let tmp = word_iters::set::View {
            word_iters: &self.0,
            map: &map.0,
        };
        Ok(tmp.lookup(other.0.get_elem_ref(index).get_u64it()))
    }

    /// Remove the component indexed in `other` from `self`, if present.
    pub fn mapped_remove(
        &mut self,
        map: &mut Map,
        other: &Self,
        index: isize,
    ) -> PyResult<Option<usize>> {
        let index = try_py_index(index, other.len())?;
        let out = word_iters::set::View {
            word_iters: &self.0,
            map: &map.0,
        }
        .lookup(other.0.elem_u64it(index));
        if let Some(i) = out {
            word_iters::set::ViewMut {
                word_iters: &mut self.0,
                map: &mut map.0,
            }
            .drop(i);
        }
        Ok(out)
    }

    /// Determine equality of `self` with another, ignoring order of contents.
    pub fn mapped_equal(&self, map: &Map, other: &Self) -> PyResult<bool> {
        DifferentSpaces::check(&self.0, &other.0).to_py_result()?;
        if self.len() != other.len() {
            return Ok(false);
        }
        for elem_ref in other.0.iter() {
            if (word_iters::set::View {
                word_iters: &self.0,
                map: &map.0,
            })
            .lookup(elem_ref.get_u64it())
            .is_none()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Convert to a real dense vector.
    pub fn to_dense_real(&self, coeffs: RealVec, big_endian: bool) -> PyResult<Vec<f64>> {
        let refs = state::terms::View {
            word_iters: &self.0,
            coeffs: &coeffs.0,
        };
        state::lincomb::to_dense(&refs, big_endian).to_py_result()
    }

    /// Convert to a complex dense vector.
    pub fn to_dense_complex(
        &self,
        coeffs: ComplexVec,
        big_endian: bool,
    ) -> PyResult<Vec<Complex64>> {
        let refs = state::terms::View {
            word_iters: &self.0,
            coeffs: &coeffs.0,
        };
        state::lincomb::to_dense(&refs, big_endian).to_py_result()
    }

    /// Convert from a real dense vector.
    pub fn from_dense_real(
        &mut self,
        map: &mut Map,
        coeffs: &mut RealVec,
        source: Vec<f64>,
        big_endian: bool,
    ) {
        let mut mut_refs = state::term_set::ViewMut {
            word_iters: &mut self.0,
            map: &mut map.0,
            coeffs: &mut coeffs.0,
        };
        state::lincomb::assign_from_dense(&mut mut_refs, source.as_slice(), big_endian);
    }

    /// Convert from a complex dense vector.
    pub fn from_dense_complex(
        &mut self,
        map: &mut Map,
        coeffs: &mut ComplexVec,
        source: Vec<Complex64>,
        big_endian: bool,
    ) {
        let mut mut_refs = state::term_set::ViewMut {
            word_iters: &mut self.0,
            map: &mut map.0,
            coeffs: &mut coeffs.0,
        };
        state::lincomb::assign_from_dense(&mut mut_refs, source.as_slice(), big_endian);
    }

    /// Compute the real inner product.
    pub fn vdot_real(
        &self,
        map: &Map,
        coeffs: &RealVec,
        rhs_cmpnts: &Self,
        rhs_coeffs: &RealVec,
    ) -> f64 {
        let lhs = state::term_set::View {
            word_iters: &self.0,
            map: &map.0,
            coeffs: &coeffs.0,
        };
        let rhs = state::terms::View {
            word_iters: &rhs_cmpnts.0,
            coeffs: &rhs_coeffs.0,
        };
        state::lincomb::vdot(&lhs, &rhs)
    }

    /// Compute the complex inner product.
    pub fn vdot_complex(
        &self,
        map: &Map,
        coeffs: &ComplexVec,
        rhs_cmpnts: &Self,
        rhs_coeffs: &ComplexVec,
    ) -> Complex64 {
        let lhs = state::term_set::View {
            word_iters: &self.0,
            map: &map.0,
            coeffs: &coeffs.0,
        };
        let rhs = state::terms::View {
            word_iters: &rhs_cmpnts.0,
            coeffs: &rhs_coeffs.0,
        };
        state::lincomb::vdot(&lhs, &rhs)
    }
}
