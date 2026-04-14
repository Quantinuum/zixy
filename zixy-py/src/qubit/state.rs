//! Array of computational basis state strings.
use std::collections::HashSet;

use num_complex::Complex64;
use pyo3::{pyclass, pymethods, PyResult};
use zixy::container::bit_matrix::{AsRowMutRef, AsRowRef};
use zixy::container::traits::{Compatible, Elements, EmptyClone, MutRefElements, RefElements};
use zixy::container::utils::DistinctPair;
use zixy::container::word_iters::set::{AsView as _, AsViewMut as _};
use zixy::container::word_iters::{self, WordIters};
use zixy::qubit::state;
use zixy::qubit::state::cmpnt_list::CmpntList;
use zixy::qubit::traits::{DifferentQubits, QubitsBased, QubitsRelabel, QubitsStandardized};

use crate::cmpnt::state_springs::BinarySprings;
use crate::container::coeffs::{ComplexSign, ComplexVec, RealVec};
use crate::container::map::Map;
use crate::qubit::mode::Qubits;
use crate::qubit::pauli::Array as PauliArray;
use crate::utils::{try_py_index, try_py_indices, ToPyResult};

/// A list of computational basis state strings
#[pyclass(subclass)]
#[pyo3(name = "QubitStateArray")]
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
    pub fn new(qubits: Option<Qubits>, springs: Option<BinarySprings>) -> PyResult<Self> {
        let springs = springs.unwrap_or_default();
        Ok(Self(match qubits {
            Some(qubits) => CmpntList::from_springs(qubits.0, &springs.0).to_py_result()?,
            None => CmpntList::from_springs_default(&springs.0),
        }))
    }
}

#[pymethods]
impl Array {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits=None, springs=None))]
    pub fn __init__(qubits: Option<Qubits>, springs: Option<BinarySprings>) -> PyResult<Self> {
        Self::new(qubits, springs)
    }

    /// Redefine the virtual register on which `self` is based.
    pub fn relabel(&mut self, qubits: Qubits) -> PyResult<()> {
        self.0.relabel(qubits.0).to_py_result()
    }

    /// Physically reorder contents so that the qubits on which `self` is based are a simple count from 0 in the standard register.
    pub fn standardize(&mut self, n_qubit: usize) {
        self.0 = self.0.standardized(n_qubit);
    }

    /// Get the number of computational basis strings in this array.
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Determine whether two Arrays are equal
    pub fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    /// Return whether the two instances are the same.
    pub fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    /// Set the number of computational basis strings in `self`.
    /// Erases elements if the new size is less than the current size.
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }

    /// Get the qubit space on which this term is based.
    #[getter]
    pub fn get_qubits(&self) -> Qubits {
        Qubits(self.0.to_qubits())
    }

    /// Append an identity string to the end of `self`
    pub fn append_clear(&mut self) {
        self.0.push_clear();
    }

    /// Set the indexed cmpnt to the identity string.
    pub fn cmpnt_clear(&mut self, i: isize) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).clear();
        Ok(())
    }

    /// Set the Pauli matrix on a single qubit.
    pub fn cmpnt_set_bit(&mut self, i_cmpnt: isize, i_qubit: isize, bit: bool) -> PyResult<()> {
        let n_qubit = self.get_qubits().0.len();
        self.0
            .get_elem_mut_ref(try_py_index(i_cmpnt, self.len())?)
            .set_bit(try_py_index(i_qubit, n_qubit)?, bit)
            .to_py_result()
    }

    /// Get the Pauli matrix on a single qubit.
    pub fn cmpnt_get_bit(&mut self, i_cmpnt: isize, i_qubit: isize) -> PyResult<bool> {
        let n_qubit = self.get_qubits().0.len();
        self.0
            .get_elem_ref(try_py_index(i_cmpnt, self.len())?)
            .get(try_py_index(i_qubit, n_qubit)?)
            .to_py_result()
    }

    /// Set the indexed cmpnt to the given sequence of bools.
    /// Errors if length of `bits` is greater than the number of qubits,
    /// and pads the upper indices with 0 in the case that the length of `bits` is smaller.
    pub fn cmpnt_set_from_list(&mut self, i: isize, bits: Vec<bool>) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).assign_vec(bits).to_py_result()
    }

    /// Set the indexed cmpnt with the given dict mapping qubit indices to pauli matrices.
    /// All qubit indices which do not appear in the key set of `bits` is set to 0.
    pub fn cmpnt_set_from_set(&mut self, i: isize, bits: HashSet<usize>) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).assign_set(bits).to_py_result()
    }

    /// Set the indexed cmpnt with the indexed spring.
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

    /// Get the indexed cmpnt as a list of bits.
    pub fn cmpnt_get_list(&self, i: isize) -> PyResult<Vec<bool>> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).to_vec())
    }

    /// Get the indexed cmpnt as a dict from qubit indices to set mode indices matrices.
    pub fn cmpnt_get_set(&self, i: isize) -> PyResult<HashSet<usize>> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).to_set())
    }

    /// Perform a copy from the `i_src` cmpnt into the `i_dst` cmpnt of `self`.
    pub fn cmpnt_copy_internal(&mut self, i_dst: isize, i_src: isize) -> PyResult<()> {
        let i_dst = try_py_index(i_dst, self.len())?;
        let i_src = try_py_index(i_src, self.len())?;
        if let Some(inds) = DistinctPair::new(i_dst, i_src) {
            let (mut dst, src) = self.0.get_semi_mut_refs(inds);
            dst.assign(src);
        }
        Ok(())
    }

    /// Perform a copy from the `i_src` cmpnt of `src` into the `i_dst` cmpnt of `self`.
    pub fn cmpnt_copy_external(&mut self, i_dst: isize, src: &Self, i_src: isize) -> PyResult<()> {
        DifferentQubits::check(&self.0, &src.0).to_py_result()?;
        let i_dst = try_py_index(i_dst, self.len())?;
        let i_src = try_py_index(i_src, src.len())?;
        let mut dst = self.0.get_elem_mut_ref(i_dst);
        let src = src.0.get_elem_ref(i_src);
        dst.assign(src);
        Ok(())
    }

    /// Multiply in-place by a Pauli string and return the associated ComplexSign phase.
    pub fn cmpnt_pauli_string_imul(
        &mut self,
        index: isize,
        paulis: &PauliArray,
        i_pauli: isize,
    ) -> PyResult<ComplexSign> {
        let index = try_py_index(index, self.len())?;
        let i_pauli = try_py_index(i_pauli, paulis.len())?;
        DifferentQubits::check(&self.0, &paulis.0).to_py_result()?;
        let mut state = self.0.get_elem_mut_ref(index);
        let pauli = paulis.0.get_elem_ref(i_pauli);
        Ok(state.imul_by_op(pauli).into())
    }

    /// Return whether the two referenced cmpnts are equal.
    pub fn cmpnt_equal(&self, i_lhs: usize, rhs: &Self, i_rhs: usize) -> bool {
        self.0.compatible_with(&rhs.0) && self.0.get_elem_ref(i_lhs) == rhs.0.get_elem_ref(i_rhs)
    }

    /// Get the number of occurrences of `bit` in the indexed cmpnt.
    pub fn cmpnt_count(&self, i: isize, bit: bool) -> PyResult<usize> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).count(bit))
    }

    /// Get the string representation of the indexed cmpnt.
    pub fn cmpnt_to_string(&self, i: isize) -> PyResult<String> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).to_string())
    }

    /// Take the indexed cmpnts of `self` and write them contiguously to a new instance.
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

    /// Determine equality of the two slices.
    pub fn cmpnts_eq(
        &self,
        indices: Option<Vec<isize>>,
        other: &Self,
        other_indices: Option<Vec<isize>>,
    ) -> PyResult<bool> {
        if !self.0.compatible_with(&other.0) {
            return Ok(false);
        }
        let n = match &indices {
            Some(v) => v.len(),
            None => self.len(),
        };
        if match &other_indices {
            Some(v) => v.len(),
            None => self.len(),
        } != n
        {
            // lengths don't match
            return Ok(false);
        };
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

    /// Find the index in `self` corresponding to the cmpnt indexed in `other` if it exists, else return None.
    pub fn mapped_lookup(&self, map: &Map, other: &Self, index: isize) -> PyResult<Option<usize>> {
        let index = try_py_index(index, other.len())?;
        let tmp = word_iters::set::View {
            word_iters: &self.0,
            map: &map.0,
        };
        Ok(tmp.lookup(other.0.get_elem_ref(index).get_u64it()))
    }

    /// Find the element indexed in `other` in `self` and if it exists, remove it by swap remove.
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

    /// Determine equality of `self` with another, order of contents insignificant.
    pub fn mapped_equal(&self, map: &Map, other: &Self) -> PyResult<bool> {
        DifferentQubits::check(&self.0, &other.0).to_py_result()?;
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

    /// Convert to a real dense vector
    pub fn to_dense_real(&self, coeffs: RealVec, big_endian: bool) -> Vec<f64> {
        let refs = state::terms::View {
            word_iters: &self.0,
            coeffs: &coeffs.0,
        };
        state::lincomb::to_dense(&refs, big_endian)
    }

    /// Convert to a complex dense vector
    pub fn to_dense_complex(&self, coeffs: ComplexVec, big_endian: bool) -> Vec<Complex64> {
        let refs = state::terms::View {
            word_iters: &self.0,
            coeffs: &coeffs.0,
        };
        state::lincomb::to_dense(&refs, big_endian)
    }

    /// Convert from a real dense vector
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

    /// Convert from a complex dense vector
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

    /// Compute the inner product
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

    /// Compute the inner product
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
