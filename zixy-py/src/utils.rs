//! Utilities useful across many binding modules.
use std::fmt::Display;

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::types::PyDict;
use pyo3::{prelude::*, IntoPyObjectExt};
use pyo3::{
    types::{PyAnyMethods, PyModule, PyModuleMethods, PySequence},
    Bound, PyAny, PyResult, Python,
};
use zixy::cmpnt::parse::ParseError;
use zixy::cmpnt::springs::BadParse;
use zixy::container::coeffs::traits::Unrepresentable;
use zixy::container::errors::IndistinctError;
use zixy::container::errors::OutOfBounds;
use zixy::container::word_iters::WordIters;
use zixy::qubit::mode::{BasisError, CoincidentIndex, DifferentModeCounts};
use zixy::qubit::pauli::cmpnt_major::mat_elem::SubspaceNonorthogonal;
use zixy::qubit::traits::DifferentQubits;

#[allow(dead_code)]
pub(crate) fn add_submodule(
    py: Python,
    parent: &Bound<PyModule>,
    submodule: Bound<PyModule>,
) -> PyResult<()> {
    parent.add_submodule(&submodule)?;

    // Add submodule to sys.modules.
    // This is required to be able to do `from parent.submodule import ...`.
    //
    // See [https://github.com/PyO3/pyo3/issues/759]
    let parent_name = parent.name()?;
    let submodule_name = submodule.name()?;
    let modules = py.import("sys")?.getattr("modules")?;
    modules.set_item(format!("{parent_name}.{submodule_name}"), submodule)?;
    Ok(())
}

#[allow(dead_code)]
pub(crate) fn len_seq(src: &Bound<PyAny>) -> PyResult<usize> {
    let seq = src.cast::<PySequence>()?;
    seq.len()
}

#[allow(dead_code)]
pub(crate) fn extract_from_seq<T>(src: &Bound<PyAny>, index: usize) -> PyResult<T>
where
    T: for<'py, 'a> FromPyObject<'a, 'py> + for<'py> FromPyObjectOwned<'py>,
{
    let seq = src.cast::<PySequence>()?;
    let item = seq.get_item(index)?;
    let item = item.extract::<T>();
    item.map_err(Into::into)
}

/// Convert a CSR sprs instance to the scipy equivalent.
pub fn to_scipy_sparse<C: numpy::Element>(src: sprs::CsMat<C>) -> PyResult<Py<PyAny>> {
    // Convert internal data to numpy arrays and then construct the scipy sparse mat
    Python::attach(|py| {
        let py_indptr = PyArray1::from_slice(py, src.indptr().as_slice().unwrap());
        let py_indices = PyArray1::from_slice(py, src.indices());
        let py_data = PyArray1::from_slice(py, src.data());
        let scipy_sparse: Bound<'_, pyo3::prelude::PyModule> = py.import("scipy.sparse")?;
        let csr_matrix_class = scipy_sparse.getattr("csr_matrix")?;
        let shape = (src.rows(), src.cols());
        let args = pyo3::types::PyTuple::new(
            py,
            &[
                py_data.into_any(),
                py_indices.into_any(),
                py_indptr.into_any(),
            ],
        )?;
        let scipy_matrix = csr_matrix_class.as_any().call1((args, shape))?;
        Ok(scipy_matrix.into())
    })
}

/// Convert a `ndarray::Array2<C>` to the numpy equivalent.
pub fn to_numpy_dense_matrix<C: numpy::Element>(src: ndarray::Array2<C>) -> Py<numpy::PyArray2<C>> {
    Python::attach(|py| PyArray2::from_owned_array(py, src).unbind())
}

/// Supported types for pandas DataFrame columns
pub enum DataFrameColumn {
    /// String column.
    Strings(Vec<String>),
    /// Integer column.
    Ints(Vec<isize>),
    /// Real number column.
    RealNums(Vec<f64>),
    /// Complex number column.
    ComplexNums(Vec<num_complex::Complex64>),
}

/// Helper for the creation of pandas DataFrame representations of objects.
pub trait CreateDataFrame {
    /// Get a raw component list as a vector of strings variant within the DataFrameColumn enum.
    fn u64it_elems_to_column<T: WordIters>(elems: &T) -> DataFrameColumn {
        use zixy::container::traits::RefElements;
        DataFrameColumn::Strings(elems.iter().map(|e| e.to_string()).collect::<Vec<_>>())
    }

    /// Get the raw data for the pandas DataFrame columns.
    fn get_dataframe_columns(&self) -> Vec<(String, DataFrameColumn)>;

    /// Convert a HashMap to a pandas DataFrame
    fn to_dataframe(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let pandas = py.import("pandas")?;
            // Convert to a Python dict
            let dict = PyDict::new(py);
            for (key, vec) in self.get_dataframe_columns().into_iter() {
                match vec {
                    DataFrameColumn::Strings(items) => dict.set_item(key, items)?,
                    DataFrameColumn::Ints(items) => dict.set_item(key, items)?,
                    DataFrameColumn::RealNums(items) => dict.set_item(key, items)?,
                    DataFrameColumn::ComplexNums(items) => dict.set_item(key, items)?,
                }
            }
            pandas.call_method1("DataFrame", (dict,))?.into_py_any(py)
        })
    }
}

/// Support Python-style negative indexing
fn py_index(index: isize, len: usize) -> isize {
    if index < 0 {
        len as isize + index
    } else {
        index
    }
}

/// Try to extract a rust-compatible unsigned index from a python-style signed index.
pub fn try_py_index(index: isize, len: usize) -> PyResult<usize> {
    let index = py_index(index, len);
    if index < 0 || index >= len as isize {
        Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "Element index {index} is out-of-bounds for container of length {len}.",
        )))
    } else {
        Ok(index as usize)
    }
}

/// Convert a list of a signed integers to unsigned indices referring to the elements of
/// a `len` element sequence. Returns Err if the index is out of bounds.
pub fn try_py_indices(indices: Vec<isize>, len: usize) -> PyResult<Vec<usize>> {
    let mut out = Vec::<usize>::with_capacity(indices.len());
    for i in indices.into_iter() {
        out.push(try_py_index(i, len)?);
    }
    Ok(out)
}

/// Implement this to provide access to a Rust object within a wrapper object.
pub trait AccessImplementation {
    /// Type of the the wrapped object.
    type Output;
    /// Get the Rust implementation within a wrapper object.
    fn access(&self) -> &Self::Output;
}

/// __str__ method for a type that wraps a rust object.
#[macro_export]
macro_rules! wrapped_str {
    ($ty: ty) => {
        #[pymethods]
        impl $ty {
            /// Get string representation.
            pub fn __str__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }
    };
}

/// Standard special methods for a type that wraps a rust object.
#[macro_export]
macro_rules! standard_dunders {
    ($ty: ty) => {
        #[pymethods]
        impl $ty {
            /// Get string representation.
            pub fn __repr__(&self) -> PyResult<String> {
                self.__str__()
            }

            /// Return shallow copy.
            pub fn __copy__(&self) -> PyResult<Self> {
                Ok(self.clone())
            }

            /// Return deep copy.
            pub fn __deepcopy__(&self, _memo: Bound<PyAny>) -> PyResult<Self> {
                Ok(self.clone())
            }
        }
    };
}

/// Maps a Rust Error to a Python exception.
pub trait ErrorToException: Display {
    /// Convert this Rust error to a python exception.
    fn get_exception(&self) -> PyErr;
}

impl ErrorToException for BadParse {
    fn get_exception(&self) -> PyErr {
        PyErr::new::<PyValueError, _>(self.to_string())
    }
}
impl ErrorToException for SubspaceNonorthogonal {
    fn get_exception(&self) -> PyErr {
        PyErr::new::<PyValueError, _>(self.to_string())
    }
}
impl ErrorToException for OutOfBounds {
    fn get_exception(&self) -> PyErr {
        PyErr::new::<PyIndexError, _>(self.to_string())
    }
}
impl ErrorToException for DifferentModeCounts {
    fn get_exception(&self) -> PyErr {
        PyErr::new::<PyIndexError, _>(self.to_string())
    }
}
impl ErrorToException for CoincidentIndex {
    fn get_exception(&self) -> PyErr {
        PyErr::new::<PyIndexError, _>(self.to_string())
    }
}
impl ErrorToException for DifferentQubits {
    fn get_exception(&self) -> PyErr {
        PyErr::new::<PyValueError, _>(self.to_string())
    }
}
impl ErrorToException for Unrepresentable {
    fn get_exception(&self) -> PyErr {
        PyErr::new::<PyValueError, _>(self.to_string())
    }
}
impl<T: Copy + PartialEq + Display> ErrorToException for IndistinctError<T> {
    fn get_exception(&self) -> PyErr {
        PyErr::new::<PyValueError, _>(self.to_string())
    }
}

/// Error occurring the in-place multiplication of terms.
#[derive(Debug)]
pub enum TermIMulError {
    /// Tried to multiply terms based on different qubit spaces.
    Incompatible(DifferentQubits),
    /// Could not absorb the phase factor in the coefficient type.
    Unrepresentable(Unrepresentable),
}

impl std::fmt::Display for TermIMulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TermIMulError::Incompatible(x) => x.fmt(f),
            TermIMulError::Unrepresentable(x) => x.fmt(f),
        }
    }
}
impl std::error::Error for TermIMulError {}

impl ErrorToException for TermIMulError {
    fn get_exception(&self) -> PyErr {
        match self {
            TermIMulError::Incompatible(x) => x.get_exception(),
            TermIMulError::Unrepresentable(x) => x.get_exception(),
        }
    }
}

impl From<Unrepresentable> for TermIMulError {
    fn from(val: Unrepresentable) -> Self {
        TermIMulError::Unrepresentable(val)
    }
}

impl From<DifferentQubits> for TermIMulError {
    fn from(val: DifferentQubits) -> Self {
        TermIMulError::Incompatible(val)
    }
}

impl ErrorToException for ParseError {
    fn get_exception(&self) -> PyErr {
        match self {
            ParseError::BadParse(x) => x.get_exception(),
            ParseError::ModeBounds(x) => x.get_exception(),
            ParseError::CoeffUnrepresentable(x) => x.get_exception(),
        }
    }
}

impl ErrorToException for BasisError {
    fn get_exception(&self) -> PyErr {
        match self {
            BasisError::Bounds(x) => x.get_exception(),
            BasisError::Counts(x) => x.get_exception(),
            BasisError::Coincident(x) => x.get_exception(),
        }
    }
}

/// Convert to PyResult
pub trait ToPyResult {
    /// Type to return if Ok.
    type OkType;
    /// Convert `self` into a Python-compatible result (with exception)
    fn to_py_result(self) -> PyResult<Self::OkType>;
}

impl<T, E: ErrorToException> ToPyResult for Result<T, E> {
    type OkType = T;

    fn to_py_result(self) -> PyResult<Self::OkType> {
        self.map_err(|e| e.get_exception())
    }
}
