//! Bindings for coefficient types.
use std::ops::Neg;

use numpy::{Complex64, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::{pyclass, pymethods, Bound, PyAny, PyResult};
use zixy::container::coeffs::traits::{FieldElem, NewUnitsWithLen, NumRepr};
use zixy::container::{
    coeffs::traits::NumReprVec,
    traits::{Elements, NewWithLen},
};

use zixy::container::coeffs::{
    complex_sign::ComplexSign as ComplexSign_, complex_sign::ComplexSignVec as ComplexSignVec_,
};
use zixy::container::coeffs::{sign::Sign as Sign_, sign::SignVec as SignVec_};
use zixy::container::coeffs::{unity::Unity as Unity_, unity::UnityVec as UnityVec_};

use crate::utils::ToPyResult;
use crate::{
    standard_dunders,
    utils::{try_py_index, AccessImplementation},
    wrapped_str,
};

/// A Unity wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "Unity")]
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Unity(pub Unity_);

#[pymethods]
impl Unity {
    /// Constructor.
    #[new]
    pub fn __init__() -> PyResult<Self> {
        Ok(Self(Unity_ {}))
    }

    /// Equality
    pub fn __eq__(&mut self, _other: Unity) -> bool {
        true
    }

    /// In-place multiplication
    pub fn __imul__(&mut self, _other: Unity) {}

    /// Out-of-place multiplication
    pub fn __mul__(&self, _other: Unity) -> PyResult<Self> {
        Ok(Self(Unity_ {}))
    }
}

wrapped_str!(Unity);
standard_dunders!(Unity);

impl From<Unity_> for Unity {
    fn from(val: Unity_) -> Self {
        Unity(val)
    }
}

impl From<Unity> for Unity_ {
    fn from(val: Unity) -> Self {
        val.0
    }
}

/// A UnityVec wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "UnityVec")]
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct UnityVec(UnityVec_);

#[pymethods]
impl UnityVec {
    /// Constructor.
    #[new]
    #[pyo3(signature = (n_element = 0))]
    pub fn __init__(n_element: Option<usize>) -> PyResult<Self> {
        Ok(Self::new_with_len(n_element.unwrap_or_default()))
    }
}
wrapped_str!(UnityVec);
standard_dunders!(UnityVec);

impl Elements for UnityVec {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl NewWithLen for UnityVec {
    fn new_with_len(n_element: usize) -> Self {
        Self(UnityVec_::new_with_len(n_element))
    }
}

impl AccessImplementation for UnityVec {
    type Output = UnityVec_;

    fn access(&self) -> &Self::Output {
        &self.0
    }
}

/// A Sign wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "Sign")]
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone, Default)]
pub struct Sign(pub Sign_);

#[pymethods]
impl Sign {
    /// Constructor.
    #[new]
    #[pyo3(signature = (is_minus_one,))]
    pub fn __init__(is_minus_one: bool) -> PyResult<Self> {
        Ok(Self(Sign_(is_minus_one)))
    }

    /// Equality
    pub fn __eq__(&mut self, other: Sign) -> bool {
        self.0 == other.0
    }

    /// In-place multiplication
    pub fn __imul__(&mut self, other: Sign) {
        self.0 *= other.0;
    }

    /// Out-of-place multiplication
    pub fn __mul__(&self, other: Sign) -> PyResult<Self> {
        Ok(Self(self.0 * other.0))
    }

    /// Get the raw phase bool.
    pub fn to_phase(&self) -> bool {
        self.0 .0
    }
}
impl Neg for Sign {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}
wrapped_str!(Sign);
standard_dunders!(Sign);

impl From<Sign_> for Sign {
    fn from(val: Sign_) -> Self {
        Sign(val)
    }
}

impl From<Sign> for Sign_ {
    fn from(val: Sign) -> Self {
        val.0
    }
}

/// A SignVec wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "SignVec")]
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct SignVec(pub SignVec_);

#[pymethods]
impl SignVec {
    /// Constructor.
    #[new]
    #[pyo3(signature = (n_element = 0))]
    pub fn __init__(n_element: Option<usize>) -> PyResult<Self> {
        Ok(Self(SignVec_::new_with_len(n_element.unwrap_or_default())))
    }

    /// Return whether the two instances are the same.
    pub fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    /// Return whether the two instances are equal.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.same_as(other) || self.0 == other.0
    }

    fn __getitem__(&self, index: isize) -> PyResult<Sign> {
        let index = try_py_index(index, self.0.len())?;
        Ok(Sign(self.0.get_unchecked(index)))
    }

    fn __setitem__(&mut self, index: isize, value: Sign) -> PyResult<()> {
        let index = try_py_index(index, self.0.len())?;
        self.0.set_unchecked(index, value.0);
        Ok(())
    }

    /// Multiply in-place by `other` at `index`.
    pub fn imul_elem(&mut self, index: isize, other: Sign) -> PyResult<()> {
        let index = try_py_index(index, self.len())?;
        self.0 .0.iadd(index, other.0 .0);
        Ok(())
    }

    /// Append `value` to `self`.
    pub fn append(&mut self, value: Sign) {
        self.0.push(value.0);
    }

    /// Extend `self` with `value` where `value` is different to `self`.
    pub fn extend_external(&mut self, value: &Self) {
        self.0.append(&value.0)
    }

    /// Extend `self` with `self`.
    pub fn extend_internal(&mut self) {
        self.0.append_self()
    }

    /// Set the number of elements.
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }

    /// Return number of elements.
    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.len())
    }

    /// Get a new instance from a python array of phases.
    #[staticmethod]
    pub fn from_phases(source: PyReadonlyArray1<bool>) -> PyResult<Self> {
        let mut out = Self::new_with_len(source.len());
        for i in 0..out.len() {
            if let Some(item) = source.get_owned(i) {
                out.0.set_unchecked(i, Sign_(item));
            }
        }
        Ok(out)
    }

    /// Try to read from a string
    #[staticmethod]
    pub fn parse(source: &str) -> PyResult<Self> {
        Ok(Self(SignVec_::try_parse(source).to_py_result()?))
    }
}
wrapped_str!(SignVec);
standard_dunders!(SignVec);

impl Elements for SignVec {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl NewWithLen for SignVec {
    fn new_with_len(n_element: usize) -> Self {
        Self(SignVec_::new_with_len(n_element))
    }
}

impl AccessImplementation for SignVec {
    type Output = SignVec_;

    fn access(&self) -> &Self::Output {
        &self.0
    }
}

/// A ComplexSign wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "ComplexSign")]
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone, Default)]
pub struct ComplexSign(pub ComplexSign_);

#[pymethods]
impl ComplexSign {
    /// Constructor.
    #[new]
    #[pyo3(signature = (power_of_i,))]
    pub fn __init__(power_of_i: u8) -> PyResult<Self> {
        Ok(Self(ComplexSign_(power_of_i)))
    }

    /// Equality
    pub fn __eq__(&mut self, other: ComplexSign) -> bool {
        self.0 == other.0
    }

    /// Raise self to an integer exponent
    pub fn __pow__(&self, exp: i32, _modulo: Option<i32>) -> PyResult<Self> {
        Ok(Self(self.0.pow(exp)))
    }

    /// In-place multiplication.
    pub fn __imul__(&mut self, other: ComplexSign) {
        self.0 *= other.0;
    }

    /// Out-of-place multiplication.
    pub fn __mul__(&self, other: ComplexSign) -> PyResult<Self> {
        Ok(Self(self.0 * other.0))
    }

    /// Get the raw phase byte.
    pub fn to_phase(&self) -> u8 {
        self.0 .0
    }
}
wrapped_str!(ComplexSign);
standard_dunders!(ComplexSign);

impl Neg for ComplexSign {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl From<ComplexSign_> for ComplexSign {
    fn from(val: ComplexSign_) -> Self {
        ComplexSign(val)
    }
}

impl From<ComplexSign> for ComplexSign_ {
    fn from(val: ComplexSign) -> Self {
        val.0
    }
}

/// A ComplexSignVec wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "ComplexSignVec")]
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct ComplexSignVec(pub ComplexSignVec_);

#[pymethods]
impl ComplexSignVec {
    /// Constructor.
    #[new]
    #[pyo3(signature = (n_element = 0))]
    pub fn __init__(n_element: Option<usize>) -> PyResult<Self> {
        Ok(Self(ComplexSignVec_::new_with_len(
            n_element.unwrap_or_default(),
        )))
    }

    /// Return whether the two instances are the same.
    pub fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    /// Return whether the two instances are equal.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.same_as(other) || self.0 == other.0
    }

    fn __getitem__(&self, index: isize) -> PyResult<ComplexSign> {
        let index = try_py_index(index, self.0.len())?;
        Ok(ComplexSign(self.0.get_unchecked(index)))
    }

    fn __setitem__(&mut self, index: isize, value: ComplexSign) -> PyResult<()> {
        let index = try_py_index(index, self.0.len())?;
        self.0.set_unchecked(index, value.0);
        Ok(())
    }

    /// Multiply in-place by `other` at `index`.
    pub fn imul_elem(&mut self, index: isize, other: ComplexSign) -> PyResult<()> {
        let index = try_py_index(index, self.len())?;
        self.0.imul_elem_unchecked(index, other.0);
        Ok(())
    }

    /// Append `value` to `self`.
    pub fn append(&mut self, value: ComplexSign) {
        self.0.push(value.0);
    }

    /// Extend `self` with `value` where `value` is different to `self`.
    pub fn extend_external(&mut self, value: &Self) {
        self.0.append(&value.0)
    }

    /// Extend `self` with `self`.
    pub fn extend_internal(&mut self) {
        self.0.append_self()
    }

    /// Set the number of elements.
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }

    /// Return number of elements.
    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.len())
    }

    /// Get a new instance from a python array of phases.
    #[staticmethod]
    pub fn from_phases(source: PyReadonlyArray1<u8>) -> PyResult<Self> {
        let mut out = Self::new_with_len(source.len());
        for i in 0..out.len() {
            if let Some(item) = source.get_owned(i) {
                out.0.set_unchecked(i, item.into());
            }
        }
        Ok(out)
    }

    /// Try to read from a string
    #[staticmethod]
    pub fn parse(source: &str) -> PyResult<Self> {
        Ok(Self(ComplexSignVec_::try_parse(source).to_py_result()?))
    }
}
wrapped_str!(ComplexSignVec);
standard_dunders!(ComplexSignVec);

impl Elements for ComplexSignVec {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl NewWithLen for ComplexSignVec {
    fn new_with_len(n_element: usize) -> Self {
        Self(ComplexSignVec_::new_with_len(n_element))
    }
}

impl AccessImplementation for ComplexSignVec {
    type Output = ComplexSignVec_;

    fn access(&self) -> &Self::Output {
        &self.0
    }
}

/// A RealVec wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "RealVec")]
#[repr(transparent)]
#[derive(PartialEq, Clone)]
pub struct RealVec(pub Vec<f64>);

#[pymethods]
impl RealVec {
    /// Constructor.
    #[new]
    #[pyo3(signature = (n_element = 0))]
    pub fn __init__(n_element: Option<usize>) -> PyResult<Self> {
        Ok(Self(Vec::<f64>::new_units_with_len(
            n_element.unwrap_or_default(),
        )))
    }

    /// Return whether the two instances are the same.
    pub fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    /// Return whether the two instances are equal.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.same_as(other) || self.0 == other.0
    }

    fn __getitem__(&self, index: isize) -> PyResult<f64> {
        let index = try_py_index(index, self.0.len())?;
        Ok(self.0.get_unchecked(index))
    }

    fn __setitem__(&mut self, index: isize, value: f64) -> PyResult<()> {
        let index = try_py_index(index, self.0.len())?;
        self.0.set_unchecked(index, value);
        Ok(())
    }

    /// Multiply in-place by `other` at `index`.
    pub fn imul_elem(&mut self, index: isize, other: f64) -> PyResult<()> {
        let index = try_py_index(index, self.len())?;
        self.0[index] *= other;
        Ok(())
    }

    /// Append `value` to `self`.
    pub fn append(&mut self, value: f64) {
        self.0.push(value);
    }

    /// Extend `self` with `value` where `value` is different to `self`.
    pub fn extend_external(&mut self, value: &Self) {
        NumReprVec::append(&mut self.0, &value.0);
    }

    /// Extend `self` with `self`.
    pub fn extend_internal(&mut self) {
        self.0.append_self()
    }

    /// Set the number of elements.
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n, 1.0);
    }

    /// Return number of elements.
    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.len())
    }

    /// Get a new instance from a python array of phases.
    #[staticmethod]
    pub fn from_array(source: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let mut out = Self::new_with_len(source.len());
        for i in 0..out.len() {
            if let Some(item) = source.get_owned(i) {
                out.0.set_unchecked(i, item);
            }
        }
        Ok(out)
    }

    /// Get `self` as a Python list.
    pub fn to_array(&self) -> Vec<f64> {
        self.0.clone()
    }

    /// Determine whether this vector is nearly zero in all elements.
    pub fn all_insignificant(&self, atol: f64) -> bool {
        !self.0.iter().any(|x| x.is_significant(atol))
    }

    /// Try to read from a string
    #[staticmethod]
    pub fn parse(source: &str) -> PyResult<Self> {
        Ok(Self(Vec::<f64>::try_parse(source).to_py_result()?))
    }
}

impl Elements for RealVec {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl NewWithLen for RealVec {
    fn new_with_len(n_element: usize) -> Self {
        Self(Vec::<f64>::new_with_len(n_element))
    }
}

impl AccessImplementation for RealVec {
    type Output = Vec<f64>;

    fn access(&self) -> &Self::Output {
        &self.0
    }
}

/// A ComplexVec wrapper for Python.
#[pyclass(subclass)]
#[pyo3(name = "ComplexVec")]
#[repr(transparent)]
#[derive(PartialEq, Clone)]
pub struct ComplexVec(pub Vec<Complex64>);

#[pymethods]
impl ComplexVec {
    /// Constructor.
    #[new]
    #[pyo3(signature = (n_element = 0))]
    pub fn __init__(n_element: Option<usize>) -> PyResult<Self> {
        Ok(Self(Vec::<Complex64>::new_units_with_len(
            n_element.unwrap_or_default(),
        )))
    }

    /// Return whether the two instances are the same.
    pub fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    /// Return whether the two instances are equal.
    pub fn __eq__(&self, other: &Self) -> bool {
        self.same_as(other) || self.0 == other.0
    }

    fn __getitem__(&self, index: isize) -> PyResult<Complex64> {
        let index = try_py_index(index, self.0.len())?;
        Ok(self.0.get_unchecked(index))
    }

    fn __setitem__(&mut self, index: isize, value: Complex64) -> PyResult<()> {
        let index = try_py_index(index, self.0.len())?;
        self.0.set_unchecked(index, value);
        Ok(())
    }

    /// Multiply in-place by `other` at `index`.
    pub fn imul_elem(&mut self, index: isize, other: Complex64) -> PyResult<()> {
        let index = try_py_index(index, self.len())?;
        self.0[index] *= other;
        Ok(())
    }

    /// Append `value` to `self`.
    pub fn append(&mut self, value: Complex64) {
        self.0.push(value);
    }

    /// Extend `self` with `value` where `value` is different to `self`.
    pub fn extend_external(&mut self, value: &Self) {
        NumReprVec::append(&mut self.0, &value.0);
    }

    /// Extend `self` with `self`.
    pub fn extend_internal(&mut self) {
        self.0.append_self()
    }

    /// Set the number of elements.
    pub fn resize(&mut self, n: usize) {
        self.0.resize(n, Complex64::ONE);
    }

    /// Return number of elements.
    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.0.len())
    }

    /// Get a new instance from a python array of phases.
    #[staticmethod]
    pub fn from_array(source: PyReadonlyArray1<Complex64>) -> PyResult<Self> {
        let mut out = Self::new_with_len(source.len());
        for i in 0..out.len() {
            if let Some(item) = source.get_owned(i) {
                out.0.set_unchecked(i, item);
            }
        }
        Ok(out)
    }

    /// Get `self` as a Python list.
    pub fn to_array(&self) -> Vec<Complex64> {
        self.0.clone()
    }

    /// Determine whether this vector is nearly zero in all elements.
    pub fn all_insignificant(&self, atol: f64) -> bool {
        !self.0.iter().any(|x| x.is_significant(atol))
    }

    /// Try to read from a string
    #[staticmethod]
    pub fn parse(source: &str) -> PyResult<Self> {
        Ok(Self(Vec::<Complex64>::try_parse(source).to_py_result()?))
    }
}
// standard_dunders!(ComplexVec);

impl Elements for ComplexVec {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl NewWithLen for ComplexVec {
    fn new_with_len(n_element: usize) -> Self {
        Self(Vec::<Complex64>::new_with_len(n_element))
    }
}

impl AccessImplementation for ComplexVec {
    type Output = Vec<Complex64>;

    fn access(&self) -> &Self::Output {
        &self.0
    }
}

impl AccessImplementation for Vec<f64> {
    type Output = Self;

    fn access(&self) -> &Self::Output {
        self
    }
}

impl AccessImplementation for Vec<Complex64> {
    type Output = Self;

    fn access(&self) -> &Self::Output {
        self
    }
}
