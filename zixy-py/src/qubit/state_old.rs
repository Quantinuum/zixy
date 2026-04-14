//! List of computational basis vectors.
use std::collections::HashSet;

use pyo3::{pyclass, pymethods, PyResult};
use zixy::cmpnt::springs::ModeInd;
use zixy::container::traits::{Compatible, Elements, EmptyClone, MutRefElements, RefElements};
use zixy::container::utils::DistinctPair;
use zixy::container::u64it_elems::U64ItElems;
use zixy::qubit::mode::Qubits as Qubits_;
use zixy::qubit::state::cmpnt_list::CmpntList;
use zixy::qubit::traits::{DifferentQubits, QubitsBased};

use crate::qubit::mode::Qubits;
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
        Self(CmpntList::new(Qubits_::Count(0)))
    }
}

impl EmptyClone for Array {
    fn empty_clone(&self) -> Self {
        Self(self.0.empty_clone())
    }
}

#[pymethods]
impl Array {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits=None))]
    pub fn __init__(qubits: Option<Qubits>) -> PyResult<Self> {
        let qubits = if let Some(qubits) = qubits {
            qubits.0
        } else {
            Qubits_::Count(0)
        };
        Ok(Self(CmpntList::new(qubits)))
    }

    /// Get the number of computational basis state strings in this array.
    pub fn __len__(&self) -> usize {
        self.len()
    }

    /// Return whether the two instances are the same.
    pub fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    /// Set the number of computational basis state strings in `self`.
    /// Erases elements if the new size is less than the current size.
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }

    /// Get the qubit space on which this state is based.
    #[getter]
    pub fn get_qubits(&self) -> Qubits {
        Qubits(self.0.to_qubits())
    }

    /// Append a vacuum state string to the end of `self`
    pub fn append_clear(&mut self) {
        self.0.push_clear();
    }

    /// Set the indexed cmpnt to the vacuum state string.
    pub fn cmpnt_clear(&mut self, i: isize) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).clear();
        Ok(())
    }

    /// Set the indexed cmpnt to the given sequence of booleans.
    /// Errors if length of `bits` is greater than the number of qubits,
    /// and pads the upper indices with 0 in the case that the length of `bits` is smaller.
    pub fn cmpnt_set_from_list(&mut self, i: isize, levels: Vec<bool>) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).assign_vec(levels).to_py_result()
    }

    /// Set the indexed cmpnt with the given set of excited qubits.
    /// All qubit indices which do not appear in the key set of `paulis` is set to 0.
    pub fn cmpnt_set_from_set(&mut self, i: isize, inds: HashSet<ModeInd>) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).assign_set(inds).to_py_result()
    }

    /// Set the excitation level of the indexed mode.
    pub fn cmpnt_set_level(&mut self, i_cmpnt: isize, i_qubit: isize, level: bool) -> PyResult<()> {
        let i_cmpnt = try_py_index(i_cmpnt, self.len())?;
        let i_qubit = try_py_index(i_qubit, self.get_qubits().len())?;
        self.0
            .get_elem_mut_ref(i_cmpnt)
            .set_qubit(i_qubit as ModeInd, level)
            .to_py_result()
    }

    // /// Set the indexed cmpnt with the indexed spring.
    // pub fn cmpnt_set_from_spring(&mut self, i: isize, src: &PauliSprings, i_src: usize) -> PyResult<ComplexSign> {
    //     let i = try_py_index(i, self.len())?;
    //     let phase = self.0.get_elem_mut_ref(i).set_spring(&src.0, i_src).to_py_result()?;
    //     Ok(ComplexSign(phase))
    // }

    /// Get the indexed cmpnt as a list of pauli matrices.
    pub fn cmpnt_get_list(&self, i: isize) -> PyResult<Vec<bool>> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).to_vec())
    }

    /// Get the indexed cmpnt as a set of excited state qubit indices.
    pub fn cmpnt_get_set(&self, i: isize) -> PyResult<HashSet<ModeInd>> {
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

    /// Return whether the two referenced cmpnts are equal.
    pub fn cmpnt_equal(&self, i_lhs: usize, rhs: &Self, i_rhs: usize) -> bool {
        self.0.compatible_with(&rhs.0) && self.0.get_elem_ref(i_lhs) == rhs.0.get_elem_ref(i_rhs)
    }

    // /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `rhs` and return the resulting cmpnt as
    // /// a single element array along with the complex sign phase of the product.
    // pub fn cmpnt_mul(&self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<(Self, ComplexSign)> {
    //     let i_lhs = try_py_index(i_lhs, self.len())?;
    //     let i_rhs = try_py_index(i_rhs, rhs.len())?;
    //     let lhs = self.0.get_elem_ref(i_lhs);
    //     let rhs = rhs.0.get_elem_ref(i_rhs);
    //     let mut out = self.empty_clone();
    //     out.resize(1);
    //     let phase = out.0.get_elem_mut_ref(0).assign_mul(lhs, rhs).to_py_result()?;
    //     Ok((out, ComplexSign(phase)))
    // }

    // /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `self` and store the resulting cmpnt in
    // /// cmpnt `i_lhs` of `self`
    // pub fn cmpnt_matrices_imul_internal(&mut self, i_lhs: isize, i_rhs: isize) -> PyResult<()> {
    //     let i_lhs = try_py_index(i_lhs, self.len())?;
    //     let i_rhs = try_py_index(i_rhs, self.len())?;
    //     if let Some(inds) = DistinctPair::new(i_lhs, i_rhs) {
    //         let (mut lhs, rhs) = self.0.get_semi_mut_refs(inds);
    //         lhs.imul_by_cmpnt_ref_matrices(rhs).to_py_result()
    //     }
    //     else {
    //         // squaring any string sets it to the identity string
    //         self.0.get_elem_mut_ref(i_lhs).clear();
    //         Ok(())
    //     }
    // }

    // /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `rhs` and store the resulting cmpnt in
    // /// cmpnt `i_lhs` of `self`
    // pub fn cmpnt_matrices_imul_external(&mut self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<()> {
    //     let i_lhs = try_py_index(i_lhs, self.len())?;
    //     let i_rhs = try_py_index(i_rhs, rhs.len())?;
    //     DifferentQubits::check(&self.0, &rhs.0).to_py_result()?;
    //     let mut lhs = self.0.get_elem_mut_ref(i_lhs);
    //     let rhs = rhs.0.get_elem_ref(i_rhs);
    //     lhs.imul_by_cmpnt_ref_matrices(rhs).to_py_result()
    // }

    // /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `self` and store the resulting cmpnt in
    // /// cmpnt `i_lhs` of `self`, returning only the complex sign phase of the product.
    // pub fn cmpnt_imul_internal(&mut self, i_lhs: isize, i_rhs: isize) -> PyResult<ComplexSign> {
    //     let i_lhs = try_py_index(i_lhs, self.len())?;
    //     let i_rhs = try_py_index(i_rhs, self.len())?;
    //     Ok(if let Some(inds) = DistinctPair::new(i_lhs, i_rhs) {
    //         let (mut lhs, rhs) = self.0.get_semi_mut_refs(inds);
    //         lhs.imul_by_cmpnt_ref(rhs).to_py_result()?.into()
    //     }
    //     else {
    //         // squaring any string sets it to the identity string
    //         self.0.get_elem_mut_ref(i_lhs).clear();
    //         ComplexSign::default()
    //     })
    // }

    // /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `rhs` and store the resulting cmpnt in
    // /// cmpnt `i_lhs` of `self`, returning only the complex sign phase of the product.
    // pub fn cmpnt_imul_external(&mut self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<ComplexSign> {
    //     let i_lhs = try_py_index(i_lhs, self.len())?;
    //     let i_rhs = try_py_index(i_rhs, rhs.len())?;
    //     DifferentQubits::check(&self.0, &rhs.0).to_py_result()?;
    //     let mut lhs = self.0.get_elem_mut_ref(i_lhs);
    //     let rhs = rhs.0.get_elem_ref(i_rhs);
    //     Ok(lhs.imul_by_cmpnt_ref(rhs).to_py_result()?.into())
    // }

    // /// Get the complex sign phase of the cmpnt product.
    // pub fn cmpnt_phase_of_mul(&self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<ComplexSign> {
    //     let i_lhs = try_py_index(i_lhs, self.len())?;
    //     let i_rhs = try_py_index(i_rhs, rhs.len())?;
    //     DifferentQubits::check(&self.0, &rhs.0).to_py_result()?;
    //     let lhs = self.0.get_elem_ref(i_lhs);
    //     let rhs = rhs.0.get_elem_ref(i_rhs);
    //     Ok(lhs.phase_of_mul(rhs).into())
    // }

    // /// Determine whether the `i_lhs` cmpnt of `self` commutes with the `i_rhs` cmpnt of `rhs`.
    // pub fn cmpnt_commutes_with(&self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<bool> {
    //     DifferentQubits::check(&self.0, &rhs.0).to_py_result()?;
    //     let i_lhs = try_py_index(i_lhs, self.len())?;
    //     let i_rhs = try_py_index(i_rhs, rhs.len())?;
    //     let lhs = self.0.get_elem_ref(i_lhs);
    //     let rhs = rhs.0.get_elem_ref(i_rhs);
    //     Ok(lhs.commutes_with(rhs))
    // }

    /// Get the hamming weight of the indexed cmpnt.
    pub fn cmpnt_count(&self, i: isize) -> PyResult<usize> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).count(true))
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

    /// Return whether the two indexed vectors of cmpnts are elementwise equal.
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
                Some(v) => try_py_index(v[i], self.len())?,
                None => i,
            });
            if l != r {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
