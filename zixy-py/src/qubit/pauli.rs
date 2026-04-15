//! Array of Pauli strings.
use std::collections::HashMap;
use std::path::PathBuf;

use bincode::config;
use itertools::izip;
use num_complex::Complex64;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult, Python};
use zixy::cmpnt::springs::ModeSettings;
use zixy::container::coeffs::traits::{NewUnitsWithLen, NumReprVec};
use zixy::container::coeffs::unity::UnityVec;
use zixy::container::quicksort::LexicographicSort;
use zixy::container::traits::{Compatible, Elements, EmptyClone, MutRefElements, RefElements};
use zixy::container::utils::DistinctPair;
use zixy::container::word_iters::set::{AsView as _, AsViewMut as _};
use zixy::container::word_iters::{self, WordIters};
use zixy::qubit::mode::Qubits as Qubits_;
use zixy::qubit::pauli::cmpnt_major as pauli;
use zixy::qubit::pauli::cmpnt_major::cmpnt_list::CmpntList;
use zixy::qubit::state;
use zixy::qubit::traits::{
    DifferentQubits, PauliWordMutRef, PauliWordRef, QubitsBased, QubitsRelabel, QubitsStandardized,
};
use zixy::utils::io::{BinFileReader, BinFileWriter};

use crate::container::coeffs::{ComplexSign, ComplexSignVec, ComplexVec, RealVec, SignVec};
use crate::container::map::Map;
use crate::qubit::clifford::CliffordGateList;
use crate::qubit::mode::{PauliMatrix, Qubits};
use crate::qubit::springs::PauliSprings;
use crate::qubit::state::Array as StateArray;
use crate::utils::{
    to_numpy_dense_matrix, to_scipy_sparse, try_py_index, try_py_indices, ToPyResult,
};

/// A list of Pauli strings
#[pyclass(subclass)]
#[pyo3(name = "QubitPauliArray")]
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
    /// Constructor with phases.
    pub fn new_with_phases(
        qubits: Option<Qubits>,
        springs: Option<PauliSprings>,
    ) -> PyResult<(Self, ComplexSignVec)> {
        let springs = springs.unwrap_or_default();
        let (list, phases) = if let Some(qubits) = qubits {
            CmpntList::from_springs(qubits.0, &springs.0).to_py_result()?
        } else {
            CmpntList::from_springs_default(&springs.0)
        };
        Ok((Self(list), ComplexSignVec(phases)))
    }

    /// Constructor that checks for non-unit phases.
    pub fn new(qubits: Option<Qubits>, springs: Option<PauliSprings>) -> PyResult<Self> {
        let (list, phases) = Array::new_with_phases(qubits, springs)?;
        UnityVec::try_represent(&phases.0).to_py_result()?;
        Ok(list)
    }
}

#[pymethods]
impl Array {
    /// Constructor.
    #[new]
    #[pyo3(signature = (qubits=None, springs=None))]
    pub fn __init__(qubits: Option<Qubits>, springs: Option<PauliSprings>) -> PyResult<Self> {
        Self::new(qubits, springs)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("qubits", self.0.qubits().inds()).unwrap();
        dict.set_item("cmpnts", self.0.to_string()).unwrap();
        dict
    }

    #[staticmethod]
    fn from_dict<'py>(dict: pyo3::Bound<'py, PyDict>) -> PyResult<Self> {
        let qubits: Vec<usize> = dict.get_item("qubits")?.unwrap().extract()?;
        let qubits = Qubits(Qubits_::from_inds(qubits).to_py_result()?);
        let cmpnts: String = dict.get_item("cmpnts")?.unwrap().extract()?;
        let cmpnts = zixy::qubit::pauli::springs::Springs::from_str(&cmpnts).to_py_result()?;
        Ok(Self(
            CmpntList::from_springs(qubits.0, &cmpnts).to_py_result()?.0,
        ))
    }

    /// Get a new instance and the phases associated with coincidentally-indexed Paulis in the input springs.
    #[staticmethod]
    pub fn with_phases(
        qubits: Option<Qubits>,
        springs: Option<PauliSprings>,
    ) -> PyResult<(Self, ComplexSignVec)> {
        Self::new_with_phases(qubits, springs)
    }

    /// Redefine the virtual register on which `self` is based.
    pub fn relabel(&mut self, qubits: Qubits) -> PyResult<()> {
        self.0.relabel(qubits.0).to_py_result()
    }

    /// Physically reorder contents so that the qubits on which `self` is based are a simple count from 0 in the standard register.
    pub fn standardize(&mut self, n_qubit: usize) {
        self.0 = self.0.standardized(n_qubit);
    }

    /// Physically reorder contents so that the qubits on which `self` is based are a simple count from 0 in the standard register.
    pub fn standardized(&self, n_qubit: usize) -> Self {
        Self(self.0.standardized(n_qubit))
    }

    /// Get the number of Pauli strings in this array.
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

    /// Set the number of Pauli strings in `self`.
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
    pub fn cmpnt_set_pauli(
        &mut self,
        i_cmpnt: isize,
        i_qubit: isize,
        pauli: PauliMatrix,
    ) -> PyResult<()> {
        let n_qubit = self.get_qubits().0.len();
        self.0
            .get_elem_mut_ref(try_py_index(i_cmpnt, self.len())?)
            .set_pauli(try_py_index(i_qubit, n_qubit)? as usize, pauli.into())
            .to_py_result()
    }

    /// Get the Pauli matrix on a single qubit.
    pub fn cmpnt_get_pauli(&mut self, i_cmpnt: isize, i_qubit: isize) -> PyResult<PauliMatrix> {
        let n_qubit = self.get_qubits().0.len();
        self.0
            .get_elem_ref(try_py_index(i_cmpnt, self.len())?)
            .get_pauli(try_py_index(i_qubit, n_qubit)? as usize)
            .map(|p| p.into())
            .to_py_result()
    }

    /// Set the indexed cmpnt to the given sequence of pauli matrices.
    /// Errors if length of `paulis` is greater than the number of qubits,
    /// and pads the upper indices with I in the case that the length of `paulis` is smaller.
    pub fn cmpnt_set_from_list(&mut self, i: isize, paulis: Vec<PauliMatrix>) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        let paulis = paulis.into_iter().map(|p| p.into()).collect();
        self.0.get_elem_mut_ref(i).assign_vec(paulis).to_py_result()
    }

    /// Set the indexed cmpnt with the given dict mapping qubit indices to pauli matrices.
    /// All qubit indices which do not appear in the key set of `paulis` is set to I.
    pub fn cmpnt_set_from_dict(
        &mut self,
        i: isize,
        paulis: HashMap<usize, PauliMatrix>,
    ) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        let paulis = paulis.into_iter().map(|(i, p)| (i, p.into())).collect();
        self.0.get_elem_mut_ref(i).assign_map(paulis).to_py_result()
    }

    /// Set the indexed cmpnt with the indexed spring.
    pub fn cmpnt_set_from_spring(
        &mut self,
        i: isize,
        src: &PauliSprings,
        i_src: usize,
    ) -> PyResult<ComplexSign> {
        let i = try_py_index(i, self.len())?;
        let phase = self
            .0
            .get_elem_mut_ref(i)
            .set_spring(&src.0, i_src)
            .to_py_result()?;
        Ok(ComplexSign(phase))
    }

    /// Get the indexed cmpnt as a list of pauli matrices.
    pub fn cmpnt_get_list(&self, i: isize) -> PyResult<Vec<PauliMatrix>> {
        let i = try_py_index(i, self.len())?;
        let paulis = self.0.get_elem_ref(i).get_pauli_vec();
        Ok(paulis.into_iter().map(|p| p.into()).collect())
    }

    /// Get the indexed cmpnt as a dict from qubit indices to pauli matrices.
    pub fn cmpnt_get_dict(&self, i: isize) -> PyResult<HashMap<usize, PauliMatrix>> {
        let i = try_py_index(i, self.len())?;
        let paulis = self.0.get_elem_ref(i).get_pauli_map();
        Ok(paulis.into_iter().map(|(i, p)| (i, p.into())).collect())
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

    /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `rhs` and return the resulting cmpnt as
    /// a single element array along with the complex sign phase of the product.
    pub fn cmpnt_mul(
        &self,
        i_lhs: isize,
        rhs: &Self,
        i_rhs: isize,
    ) -> PyResult<(Self, ComplexSign)> {
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, rhs.len())?;
        let lhs = self.0.get_elem_ref(i_lhs);
        let rhs = rhs.0.get_elem_ref(i_rhs);
        let mut out = self.empty_clone();
        out.resize(1);
        let phase = out
            .0
            .get_elem_mut_ref(0)
            .assign_mul(lhs, rhs)
            .to_py_result()?;
        Ok((out, ComplexSign(phase)))
    }

    /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `self` and store the resulting cmpnt in
    /// cmpnt `i_lhs` of `self`
    pub fn cmpnt_matrices_imul_internal(&mut self, i_lhs: isize, i_rhs: isize) -> PyResult<()> {
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, self.len())?;
        if let Some(inds) = DistinctPair::new(i_lhs, i_rhs) {
            let (mut lhs, rhs) = self.0.get_semi_mut_refs(inds);
            lhs.imul_by_cmpnt_ref_matrices(rhs).to_py_result()
        } else {
            // squaring any string sets it to the identity string
            self.0.get_elem_mut_ref(i_lhs).clear();
            Ok(())
        }
    }

    /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `rhs` and store the resulting cmpnt in
    /// cmpnt `i_lhs` of `self`
    pub fn cmpnt_matrices_imul_external(
        &mut self,
        i_lhs: isize,
        rhs: &Self,
        i_rhs: isize,
    ) -> PyResult<()> {
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, rhs.len())?;
        DifferentQubits::check(&self.0, &rhs.0).to_py_result()?;
        let mut lhs = self.0.get_elem_mut_ref(i_lhs);
        let rhs = rhs.0.get_elem_ref(i_rhs);
        lhs.imul_by_cmpnt_ref_matrices(rhs).to_py_result()
    }

    /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `self` and store the resulting cmpnt in
    /// cmpnt `i_lhs` of `self`, returning only the complex sign phase of the product.
    pub fn cmpnt_imul_internal(&mut self, i_lhs: isize, i_rhs: isize) -> PyResult<ComplexSign> {
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, self.len())?;
        Ok(if let Some(inds) = DistinctPair::new(i_lhs, i_rhs) {
            let (mut lhs, rhs) = self.0.get_semi_mut_refs(inds);
            lhs.imul_by_cmpnt_ref(rhs).to_py_result()?.into()
        } else {
            // squaring any string sets it to the identity string
            self.0.get_elem_mut_ref(i_lhs).clear();
            ComplexSign::default()
        })
    }

    /// Multiply the `i_lhs` cmpnt of `self` by the `i_rhs` cmpnt of `rhs` and store the resulting cmpnt in
    /// cmpnt `i_lhs` of `self`, returning only the complex sign phase of the product.
    pub fn cmpnt_imul_external(
        &mut self,
        i_lhs: isize,
        rhs: &Self,
        i_rhs: isize,
    ) -> PyResult<ComplexSign> {
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, rhs.len())?;
        DifferentQubits::check(&self.0, &rhs.0).to_py_result()?;
        let mut lhs = self.0.get_elem_mut_ref(i_lhs);
        let rhs = rhs.0.get_elem_ref(i_rhs);
        Ok(lhs.imul_by_cmpnt_ref(rhs).to_py_result()?.into())
    }

    /// Get the complex sign phase of the cmpnt product.
    pub fn cmpnt_phase_of_mul(
        &self,
        i_lhs: isize,
        rhs: &Self,
        i_rhs: isize,
    ) -> PyResult<ComplexSign> {
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, rhs.len())?;
        DifferentQubits::check(&self.0, &rhs.0).to_py_result()?;
        let lhs = self.0.get_elem_ref(i_lhs);
        let rhs = rhs.0.get_elem_ref(i_rhs);
        Ok(lhs.phase_of_mul(rhs).into())
    }

    /// Determine whether the `i_lhs` cmpnt of `self` commutes with the `i_rhs` cmpnt of `rhs`.
    pub fn cmpnt_commutes_with(&self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<bool> {
        DifferentQubits::check(&self.0, &rhs.0).to_py_result()?;
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, rhs.len())?;
        let lhs = self.0.get_elem_ref(i_lhs);
        let rhs = rhs.0.get_elem_ref(i_rhs);
        Ok(lhs.commutes_with(rhs))
    }

    /// Get the number of occurrences of `pauli` in the indexed cmpnt.
    pub fn cmpnt_count(&self, i: isize, pauli: PauliMatrix) -> PyResult<usize> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).count(pauli.into()))
    }

    /// Get the string representation of the indexed cmpnt.
    pub fn cmpnt_to_string(&self, i: isize) -> PyResult<String> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).to_string())
    }

    /// Get the indexed cmpnt as a complex sparse matrix.
    pub fn cmpnt_to_sparse_matrix(&self, i: isize, big_endian: bool) -> PyResult<Py<PyAny>> {
        let i = try_py_index(i, self.len())?;
        to_scipy_sparse(self.0.get_elem_ref(i).to_sparse_matrix(big_endian))
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

    /// Get a matrix of compatibility of the terms in this list. If the row and column strings commute, the matrix
    /// element is 1, else it's 0.
    pub fn compatibility_matrix(&self) -> Py<numpy::PyArray2<u8>> {
        use crate::utils::to_numpy_dense_matrix;
        to_numpy_dense_matrix(self.0.compatibility_matrix().to_matrix())
    }

    /// Get the centralizer and remainder of this array of pauli strings.
    pub fn centralizer_and_remainder(&self) -> (Self, Self) {
        let (c, r) = self.0.bipartition(self.0.centralizer_members());
        (Self(c), Self(r))
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

    /// Mul lhs by rhs and update out with the result
    #[staticmethod]
    pub fn lincomb_mul_real(
        lhs_impl: &Self,
        lhs_coeffs: &RealVec,
        rhs_impl: &Self,
        rhs_coeffs: &RealVec,
        out_impl: &mut Self,
        out_map: &mut Map,
        out_coeffs: &mut ComplexVec,
    ) -> PyResult<()> {
        let mut out = pauli::term_set::ViewMut {
            word_iters: &mut out_impl.0,
            coeffs: &mut out_coeffs.0,
            map: &mut out_map.0,
        };
        let lhs = pauli::terms::View::<f64> {
            word_iters: &lhs_impl.0,
            coeffs: &lhs_coeffs.0,
        };
        let rhs = pauli::terms::View::<f64> {
            word_iters: &rhs_impl.0,
            coeffs: &rhs_coeffs.0,
        };
        pauli::lincomb::assign_from_mul(&mut out, &lhs, &rhs).to_py_result()
    }

    /// Mul lhs by rhs and update out with the result
    #[staticmethod]
    pub fn lincomb_mul_complex(
        lhs_impl: &Self,
        lhs_coeffs: &ComplexVec,
        rhs_impl: &Self,
        rhs_coeffs: &ComplexVec,
        out_impl: &mut Self,
        out_map: &mut Map,
        out_coeffs: &mut ComplexVec,
    ) -> PyResult<()> {
        let mut out = pauli::term_set::ViewMut {
            word_iters: &mut out_impl.0,
            coeffs: &mut out_coeffs.0,
            map: &mut out_map.0,
        };
        let lhs = pauli::terms::View::<Complex64> {
            word_iters: &lhs_impl.0,
            coeffs: &lhs_coeffs.0,
        };
        let rhs = pauli::terms::View::<Complex64> {
            word_iters: &rhs_impl.0,
            coeffs: &rhs_coeffs.0,
        };
        pauli::lincomb::assign_from_mul(&mut out, &lhs, &rhs).to_py_result()
    }

    /// Get a real-valued operator as a complex sparse matrix.
    #[staticmethod]
    pub fn lincomb_to_sparse_real(
        cmpnts: &Self,
        coeffs: &RealVec,
        big_endian: bool,
    ) -> PyResult<Py<PyAny>> {
        use zixy::qubit::pauli::cmpnt_major::lincomb::SparseBasis;
        let lc = pauli::terms::View::<f64> {
            word_iters: &cmpnts.0,
            coeffs: &coeffs.0,
        };
        let sparse = pauli::lincomb::to_sparse_matrix(&lc, SparseBasis::Full, None, big_endian);
        to_scipy_sparse(sparse)
    }

    /// Get a complex-valued operator as a complex sparse matrix.
    #[staticmethod]
    pub fn lincomb_to_sparse_complex(
        cmpnts: &Self,
        coeffs: &ComplexVec,
        big_endian: bool,
    ) -> PyResult<Py<PyAny>> {
        use zixy::qubit::pauli::cmpnt_major::lincomb::SparseBasis;
        let lc = pauli::terms::View::<Complex64> {
            word_iters: &cmpnts.0,
            coeffs: &coeffs.0,
        };
        let sparse = pauli::lincomb::to_sparse_matrix(&lc, SparseBasis::Full, None, big_endian);
        to_scipy_sparse(sparse)
    }

    /// Get the projected operator for real-valued operator in a given subspace
    #[staticmethod]
    pub fn lincomb_project_into_ortho_subspace_real(
        cmpnts: &Self,
        coeffs: &RealVec,
        subspace: &StateArray,
    ) -> PyResult<Py<numpy::PyArray2<f64>>> {
        use zixy::qubit::pauli::cmpnt_major::mat_elem::mat_ortho_projected;
        let mat = mat_ortho_projected(
            &pauli::terms::View::<f64> {
                word_iters: &cmpnts.0,
                coeffs: &coeffs.0,
            },
            subspace.0.clone(),
        );
        let mat = mat.to_py_result()?;
        Ok(to_numpy_dense_matrix(mat))
    }

    /// Get the projected operator for complex-valued operator in a given subspace
    #[staticmethod]
    pub fn lincomb_project_into_ortho_subspace_complex(
        cmpnts: &Self,
        coeffs: &ComplexVec,
        subspace: &StateArray,
    ) -> PyResult<Py<numpy::PyArray2<Complex64>>> {
        use zixy::qubit::pauli::cmpnt_major::mat_elem::mat_ortho_projected;
        let mat = mat_ortho_projected(
            &pauli::terms::View::<Complex64> {
                word_iters: &cmpnts.0,
                coeffs: &coeffs.0,
            },
            subspace.0.clone(),
        );
        let mat = mat.to_py_result()?;
        Ok(to_numpy_dense_matrix(mat))
    }

    /// Get the projected operator and the overlap matrix for real-valued operator and subspace
    #[staticmethod]
    pub fn lincomb_project_into_nonortho_subspace_real(
        cmpnts: &Self,
        coeffs: &RealVec,
        subspace_cmpnts: Vec<StateArray>,
        subspace_maps: Vec<Map>,
        subspace_coeffs: Vec<RealVec>,
    ) -> (Py<numpy::PyArray2<f64>>, Py<numpy::PyArray2<f64>>) {
        use zixy::qubit::pauli::cmpnt_major::mat_elem::mat_nonortho_projected;
        let subspace = izip!(
            subspace_cmpnts.iter(),
            subspace_maps.iter(),
            subspace_coeffs.iter()
        )
        .map(|(cmpnts, map, coeffs)| state::term_set::View {
            word_iters: &cmpnts.0,
            map: &map.0,
            coeffs: &coeffs.0,
        })
        .collect::<Vec<_>>();
        let (mat, ovlp) = mat_nonortho_projected(
            &pauli::terms::View {
                word_iters: &cmpnts.0,
                coeffs: &coeffs.0,
            },
            subspace.iter().collect(),
        );
        (to_numpy_dense_matrix(mat), to_numpy_dense_matrix(ovlp))
    }

    /// Get the projected operator and the overlap matrix for complex-valued operator and subspace
    #[staticmethod]
    pub fn lincomb_project_into_nonortho_subspace_complex(
        cmpnts: &Self,
        coeffs: &ComplexVec,
        subspace_cmpnts: Vec<StateArray>,
        subspace_maps: Vec<Map>,
        subspace_coeffs: Vec<ComplexVec>,
    ) -> (
        Py<numpy::PyArray2<Complex64>>,
        Py<numpy::PyArray2<Complex64>>,
    ) {
        use zixy::qubit::pauli::cmpnt_major::mat_elem::mat_nonortho_projected;
        let subspace = izip!(
            subspace_cmpnts.iter(),
            subspace_maps.iter(),
            subspace_coeffs.iter()
        )
        .map(|(cmpnts, map, coeffs)| state::term_set::View {
            word_iters: &cmpnts.0,
            map: &map.0,
            coeffs: &coeffs.0,
        })
        .collect::<Vec<_>>();
        let (mat, ovlp) = mat_nonortho_projected(
            &pauli::terms::View {
                word_iters: &cmpnts.0,
                coeffs: &coeffs.0,
            },
            subspace.iter().collect(),
        );
        (to_numpy_dense_matrix(mat), to_numpy_dense_matrix(ovlp))
    }

    /// Sort in lexicographic order
    pub fn lexicographic_sort(&mut self, ascending: bool) {
        use zixy::container::quicksort::QuickSortNoCoeffs;
        LexicographicSort { ascending }.sort(&mut self.0);
    }

    /// Sort in lexicographic order
    pub fn lexicographic_sort_with_sign_vec(&mut self, coeffs: &mut SignVec, ascending: bool) {
        use zixy::container::coeffs::sign::Sign;
        use zixy::container::quicksort::QuickSort;
        let sorter = LexicographicSort { ascending };
        QuickSort::<CmpntList, Sign>::sort_with_coeffs(&sorter, &mut self.0, &mut coeffs.0);
    }
    /// Sort in lexicographic order
    pub fn lexicographic_sort_with_complex_sign_vec(
        &mut self,
        coeffs: &mut ComplexSignVec,
        ascending: bool,
    ) {
        use zixy::container::coeffs::complex_sign::ComplexSign;
        use zixy::container::quicksort::QuickSort;
        let sorter = LexicographicSort { ascending };
        QuickSort::<CmpntList, ComplexSign>::sort_with_coeffs(&sorter, &mut self.0, &mut coeffs.0);
    }
    /// Sort in lexicographic order
    pub fn lexicographic_sort_with_real_vec(&mut self, coeffs: &mut RealVec, ascending: bool) {
        use zixy::container::quicksort::QuickSort;
        let sorter = LexicographicSort { ascending };
        QuickSort::<CmpntList, f64>::sort_with_coeffs(&sorter, &mut self.0, &mut coeffs.0);
    }
    /// Sort in lexicographic order
    pub fn lexicographic_sort_with_complex_vec(
        &mut self,
        coeffs: &mut ComplexVec,
        ascending: bool,
    ) {
        use zixy::container::quicksort::QuickSort;
        let sorter = LexicographicSort { ascending };
        QuickSort::<CmpntList, Complex64>::sort_with_coeffs(&sorter, &mut self.0, &mut coeffs.0);
    }

    /// Apply this array as an operator to state.
    pub fn apply_to_state_real(
        &self,
        coeffs: &RealVec,
        state_cmpnts: &StateArray,
        state_coeffs: &RealVec,
        out_cmpnts: &mut StateArray,
        out_map: &mut Map,
        out_coeffs: &mut ComplexVec,
    ) {
        use zixy::qubit::pauli::cmpnt_major::mat_elem::apply;
        let op = pauli::terms::View {
            word_iters: &self.0,
            coeffs: &coeffs.0,
        };
        let state = state::terms::View {
            word_iters: &state_cmpnts.0,
            coeffs: &state_coeffs.0,
        };
        let mut out = state::term_set::ViewMut {
            word_iters: &mut out_cmpnts.0,
            map: &mut out_map.0,
            coeffs: &mut out_coeffs.0,
        };
        apply::<f64>(&op, &state, &mut out);
    }

    /// Apply this array as an operator to state.
    pub fn apply_to_state_complex(
        &self,
        coeffs: &ComplexVec,
        state_cmpnts: &StateArray,
        state_coeffs: &ComplexVec,
        out_cmpnts: &mut StateArray,
        out_map: &mut Map,
        out_coeffs: &mut ComplexVec,
    ) {
        use zixy::qubit::pauli::cmpnt_major::mat_elem::apply;
        let op = pauli::terms::View {
            word_iters: &self.0,
            coeffs: &coeffs.0,
        };
        let state = state::terms::View {
            word_iters: &state_cmpnts.0,
            coeffs: &state_coeffs.0,
        };
        let mut out = state::term_set::ViewMut {
            word_iters: &mut out_cmpnts.0,
            map: &mut out_map.0,
            coeffs: &mut out_coeffs.0,
        };
        apply::<Complex64>(&op, &state, &mut out);
    }

    /// Get matrix element of real operator
    pub fn mat_elem_real(
        &self,
        coeffs: &RealVec,
        bra_cmpnts: &StateArray,
        bra_coeffs: &RealVec,
        ket_cmpnts: &StateArray,
        ket_coeffs: &RealVec,
    ) -> f64 {
        use zixy::qubit::pauli::cmpnt_major::mat_elem::mat_elem;
        let op = pauli::terms::View {
            word_iters: &self.0,
            coeffs: &coeffs.0,
        };
        let bra = state::terms::View {
            word_iters: &bra_cmpnts.0,
            coeffs: &bra_coeffs.0,
        };
        let ket = state::terms::View {
            word_iters: &ket_cmpnts.0,
            coeffs: &ket_coeffs.0,
        };
        mat_elem(&op, &bra, &ket)
    }

    /// Get matrix element of complex operator
    pub fn mat_elem_complex(
        &self,
        coeffs: &ComplexVec,
        bra_cmpnts: &StateArray,
        bra_coeffs: &ComplexVec,
        ket_cmpnts: &StateArray,
        ket_coeffs: &ComplexVec,
    ) -> Complex64 {
        use zixy::qubit::pauli::cmpnt_major::mat_elem::mat_elem;
        let op = pauli::terms::View {
            word_iters: &self.0,
            coeffs: &coeffs.0,
        };
        let bra = state::terms::View {
            word_iters: &bra_cmpnts.0,
            coeffs: &bra_coeffs.0,
        };
        let ket = state::terms::View {
            word_iters: &ket_cmpnts.0,
            coeffs: &ket_coeffs.0,
        };
        mat_elem(&op, &bra, &ket)
    }

    /// Conjugate `self` by a list of clifford gates and return the associated signs
    pub fn conj_clifford_list(&mut self, gates: &CliffordGateList) -> SignVec {
        use zixy::container::coeffs::sign::SignVec as SignVec_;
        let mut out = SignVec_::new_units_with_len(self.0.len());
        for i in 0..self.0.len() {
            let sign = self.0.get_elem_mut_ref(i).conj_clifford_vec(&gates.0);
            out.set_unchecked(i, sign);
        }
        SignVec(out)
    }

    /// Save in binary format
    pub fn save_to_file(&self, path: String) {
        bincode::serde::encode_into_writer(
            &self.0,
            BinFileWriter::new(PathBuf::from(path)).unwrap(),
            config::standard(),
        )
        .unwrap();
    }

    /// Load from binary format file
    #[staticmethod]
    pub fn load_from_file(path: String) -> Self {
        let reader = BinFileReader::new(PathBuf::from(path)).unwrap();
        let cmpnt_list: CmpntList =
            bincode::serde::decode_from_reader(reader, config::standard()).unwrap();
        Self(cmpnt_list)
    }
}
