//! Arrays of fermion ladder-operator products.

use std::collections::HashSet;

use itertools::Itertools;
use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, PyErr, PyResult};
use zixy::cmpnt::springs::ModeSettings;
use zixy::container::bit_matrix::{AsRowMutRef, AsRowRef};
use zixy::container::map::Map as CoreMap;
use zixy::container::traits::{Compatible, Elements, EmptyClone, MutRefElements, RefElements};
use zixy::container::utils::DistinctPair;
use zixy::container::word_iters::set::{AsView as _, AsViewMut as _};
use zixy::container::word_iters::{self, WordIters};
use zixy::fermion::mode::Modes as Modes_;
use zixy::fermion::operator::general;
use zixy::fermion::operator::normal;
use zixy::fermion::traits::{DifferentSpaces, ModesBased};

use crate::container::coeffs::{ComplexVec, RealVec, SignVec};
use crate::container::map::Map;
use crate::fermion::mode::Modes;
use crate::fermion::springs::FermionSprings;
use crate::fermion::state::Array as StateArray;
use crate::utils::{try_py_index, try_py_indices, ToPyResult};

/// A list of normal-ordered fermion ladder-operator products.
#[pyclass(subclass)]
#[pyo3(name = "NormalFermionOperatorArray")]
#[repr(transparent)]
#[derive(Clone)]
pub struct NormalArray(pub normal::cmpnt_list::CmpntList);

impl Elements for NormalArray {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl EmptyClone for NormalArray {
    fn empty_clone(&self) -> Self {
        Self(self.0.empty_clone())
    }
}

impl Default for NormalArray {
    fn default() -> Self {
        Self(normal::cmpnt_list::CmpntList::new(Modes_::from_count(0)))
    }
}

fn normal_terms_real<'a>(
    cmpnts: &'a NormalArray,
    coeffs: &'a RealVec,
) -> normal::cmpnt_major::terms::View<'a, f64> {
    normal::cmpnt_major::terms::View {
        word_iters: &cmpnts.0,
        coeffs: &coeffs.0,
    }
}

fn normal_terms_complex<'a>(
    cmpnts: &'a NormalArray,
    coeffs: &'a ComplexVec,
) -> normal::cmpnt_major::terms::View<'a, Complex64> {
    normal::cmpnt_major::terms::View {
        word_iters: &cmpnts.0,
        coeffs: &coeffs.0,
    }
}

fn normal_set_real<'a>(
    cmpnts: &'a NormalArray,
    map: &'a Map,
    coeffs: &'a RealVec,
) -> normal::cmpnt_major::term_set::View<'a, f64> {
    normal::cmpnt_major::term_set::View {
        word_iters: &cmpnts.0,
        coeffs: &coeffs.0,
        map: &map.0,
    }
}

fn normal_set_complex<'a>(
    cmpnts: &'a NormalArray,
    map: &'a Map,
    coeffs: &'a ComplexVec,
) -> normal::cmpnt_major::term_set::View<'a, Complex64> {
    normal::cmpnt_major::term_set::View {
        word_iters: &cmpnts.0,
        coeffs: &coeffs.0,
        map: &map.0,
    }
}

fn normal_set_to_py_complex(
    term_set: normal::cmpnt_major::term_set::TermSet<Complex64>,
) -> (NormalArray, ComplexVec) {
    (
        NormalArray(term_set.terms.word_iters),
        ComplexVec(clean_complex_coeffs(term_set.terms.coeffs)),
    )
}

fn normal_set_to_py_real(
    term_set: normal::cmpnt_major::term_set::TermSet<f64>,
) -> (NormalArray, RealVec) {
    (
        NormalArray(term_set.terms.word_iters),
        RealVec(term_set.terms.coeffs),
    )
}

fn normal_terms_to_py_real(
    terms: normal::cmpnt_major::terms::Terms<f64>,
) -> (NormalArray, RealVec) {
    (NormalArray(terms.word_iters), RealVec(terms.coeffs))
}

fn normal_terms_to_py_complex(
    terms: normal::cmpnt_major::terms::Terms<Complex64>,
) -> (NormalArray, ComplexVec) {
    (
        NormalArray(terms.word_iters),
        ComplexVec(clean_complex_coeffs(terms.coeffs)),
    )
}

fn normal_order_product(
    modes: Modes_,
    ops: Vec<(usize, bool)>,
) -> PyResult<(NormalArray, RealVec)> {
    let max_len = ops.len();
    let (mode_inds, adj): (Vec<_>, Vec<_>) = ops.into_iter().unzip();
    check_general_ops(&modes, &mode_inds, &adj, max_len)?;
    let mut general_set = general::term_set::TermSet::<f64>::new(max_len, modes);
    general_set.push_term(&mode_inds, &adj, 1.0);
    let out = zixy::fermion::operator::lincomb::to_normal_order(&general_set.as_terms());
    Ok(normal_set_to_py_real(out))
}

fn check_general_ops(
    modes: &Modes_,
    mode_inds: &[usize],
    adj: &[bool],
    max_len: usize,
) -> PyResult<()> {
    if mode_inds.len() != adj.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "modes and adj must have the same length",
        ));
    }
    if mode_inds.len() > max_len {
        return Err(PyErr::new::<PyValueError, _>(
            "operator product is longer than max_len",
        ));
    }
    if mode_inds.iter().any(|mode| *mode >= modes.len()) {
        return Err(PyErr::new::<PyValueError, _>("mode index out of bounds"));
    }
    Ok(())
}

#[pymethods]
impl NormalArray {
    /// Constructor.
    #[new]
    #[pyo3(signature = (modes=None, springs=None))]
    fn __init__(modes: Option<Modes>, springs: Option<FermionSprings>) -> PyResult<Self> {
        let springs = springs.unwrap_or_default();
        let modes = modes.unwrap_or_else(|| {
            Modes(Modes_::from_count(
                springs.0.get_mode_inds().default_n_mode() as usize,
            ))
        });
        let mut out = Self(normal::cmpnt_list::CmpntList::new(modes.0.clone()));
        for i in 0..springs.0.len() {
            let ops = springs
                .0
                .get_ladder_op_iter(i)
                .map(|(is_cre, mode)| (usize::from(mode), is_cre))
                .collect();
            let (cmpnts, coeffs) = normal_order_product(modes.0.clone(), ops)?;
            if cmpnts.0.len() != 1 || coeffs.0.len() != 1 || coeffs.0[0] != 1.0 {
                return Err(PyErr::new::<PyValueError, _>(
                    "Fermion string does not normal-order to exactly one positive component",
                ));
            }
            out.0.push_elem_ref(cmpnts.0.get_elem_ref(0));
        }
        Ok(out)
    }

    /// Create from a single arbitrary ladder-operator product, returning normal-ordered terms.
    #[staticmethod]
    fn from_ladder_product(modes: Modes, ops: Vec<(usize, bool)>) -> PyResult<(Self, RealVec)> {
        normal_order_product(modes.0, ops)
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }

    #[getter]
    fn get_modes(&self) -> Modes {
        Modes(self.0.to_modes())
    }

    fn append_clear(&mut self) {
        self.0.push_clear();
    }

    fn cmpnt_clear(&mut self, i: isize) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        self.0.get_elem_mut_ref(i).clear();
        Ok(())
    }

    fn cmpnt_set_from_sets(
        &mut self,
        i: isize,
        cre: HashSet<usize>,
        ann: HashSet<usize>,
    ) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        let mut elem = self.0.get_elem_mut_ref(i);
        elem.clear();
        elem.get_cre_part().assign_set(cre).to_py_result()?;
        elem.get_ann_part().assign_set(ann).to_py_result()
    }

    fn cmpnt_set_from_lists(&mut self, i: isize, cre: Vec<bool>, ann: Vec<bool>) -> PyResult<()> {
        let i = try_py_index(i, self.len())?;
        let mut elem = self.0.get_elem_mut_ref(i);
        elem.clear();
        elem.get_cre_part().assign_vec(cre).to_py_result()?;
        elem.get_ann_part().assign_vec(ann).to_py_result()
    }

    fn cmpnt_get_sets(&self, i: isize) -> PyResult<(Vec<usize>, Vec<usize>)> {
        let i = try_py_index(i, self.len())?;
        let elem = self.0.get_elem_ref(i);
        Ok((
            elem.get_cre_part().to_set().into_iter().sorted().collect(),
            elem.get_ann_part().to_set().into_iter().sorted().collect(),
        ))
    }

    fn cmpnt_set_cre(&mut self, i_cmpnt: isize, i_mode: isize, value: bool) -> PyResult<()> {
        let i_cmpnt = try_py_index(i_cmpnt, self.len())?;
        let i_mode = try_py_index(i_mode, self.0.modes().len())?;
        self.0
            .get_elem_mut_ref(i_cmpnt)
            .get_cre_part()
            .set_bit(i_mode, value)
            .to_py_result()
    }

    fn cmpnt_set_ann(&mut self, i_cmpnt: isize, i_mode: isize, value: bool) -> PyResult<()> {
        let i_cmpnt = try_py_index(i_cmpnt, self.len())?;
        let i_mode = try_py_index(i_mode, self.0.modes().len())?;
        self.0
            .get_elem_mut_ref(i_cmpnt)
            .get_ann_part()
            .set_bit(i_mode, value)
            .to_py_result()
    }

    fn cmpnt_get_cre(&self, i_cmpnt: isize, i_mode: isize) -> PyResult<bool> {
        let i_cmpnt = try_py_index(i_cmpnt, self.len())?;
        let i_mode = try_py_index(i_mode, self.0.modes().len())?;
        Ok(self
            .0
            .get_elem_ref(i_cmpnt)
            .get_cre_part()
            .get_bit_unchecked(i_mode))
    }

    fn cmpnt_get_ann(&self, i_cmpnt: isize, i_mode: isize) -> PyResult<bool> {
        let i_cmpnt = try_py_index(i_cmpnt, self.len())?;
        let i_mode = try_py_index(i_mode, self.0.modes().len())?;
        Ok(self
            .0
            .get_elem_ref(i_cmpnt)
            .get_ann_part()
            .get_bit_unchecked(i_mode))
    }

    fn cmpnt_to_string(&self, i: isize) -> PyResult<String> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).to_string())
    }

    fn cmpnt_count_n_body(&self, i: isize) -> PyResult<usize> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).count_n_body())
    }

    fn cmpnt_particle_number_change(&self, i: isize) -> PyResult<isize> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get_elem_ref(i).particle_number_change())
    }

    fn dagger_get_signs(&mut self) -> SignVec {
        SignVec(self.0.dagger_get_signs())
    }

    fn cmpnt_copy_internal(&mut self, i_dst: isize, i_src: isize) -> PyResult<()> {
        let i_dst = try_py_index(i_dst, self.len())?;
        let i_src = try_py_index(i_src, self.len())?;
        if let Some(inds) = DistinctPair::new(i_dst, i_src) {
            let (mut dst, src) = self.0.get_semi_mut_refs(inds);
            dst.assign(src);
        }
        Ok(())
    }

    fn cmpnt_copy_external(&mut self, i_dst: isize, src: &Self, i_src: isize) -> PyResult<()> {
        DifferentSpaces::check(&self.0, &src.0).to_py_result()?;
        let i_dst = try_py_index(i_dst, self.len())?;
        let i_src = try_py_index(i_src, src.len())?;
        self.0
            .get_elem_mut_ref(i_dst)
            .assign(src.0.get_elem_ref(i_src));
        Ok(())
    }

    fn cmpnt_equal(&self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<bool> {
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, rhs.len())?;
        Ok(self.0.compatible_with(&rhs.0)
            && self.0.get_elem_ref(i_lhs) == rhs.0.get_elem_ref(i_rhs))
    }

    fn cmpnt_mul(&self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<(Self, SignVec)> {
        DifferentSpaces::check(&self.0, &rhs.0).to_py_result()?;
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, rhs.len())?;
        let (cmpnts, signs) =
            normal::products::mul_cmpnts(&self.0.get_elem_ref(i_lhs), &rhs.0.get_elem_ref(i_rhs));
        Ok((Self(cmpnts), SignVec(signs)))
    }

    fn cmpnts_clone(&self, indices: Option<Vec<isize>>) -> PyResult<Self> {
        Ok(match indices {
            Some(indices) => {
                let indices = try_py_indices(indices, self.len())?;
                let mut out = self.0.empty_clone();
                for i in indices {
                    out.push_elem_ref(self.0.get_elem_ref(i));
                }
                Self(out)
            }
            None => self.clone(),
        })
    }

    fn cmpnts_eq(
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

    fn refresh_map(&self, map: &mut Map) {
        map.0.populate_from(&self.0);
    }

    fn mapped_insert(&mut self, map: &mut Map, other: &Self, index: isize) -> PyResult<usize> {
        let index = try_py_index(index, other.len())?;
        let mut tmp = word_iters::set::ViewMut {
            word_iters: &mut self.0,
            map: &mut map.0,
        };
        Ok(tmp.insert_or_get_index(other.0.get_elem_ref(index)).0)
    }

    fn mapped_lookup(&self, map: &Map, other: &Self, index: isize) -> PyResult<Option<usize>> {
        let index = try_py_index(index, other.len())?;
        let tmp = word_iters::set::View {
            word_iters: &self.0,
            map: &map.0,
        };
        Ok(tmp.lookup(other.0.get_elem_ref(index).get_u64it()))
    }

    fn mapped_remove(
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

    fn mapped_equal(&self, map: &Map, other: &Self) -> PyResult<bool> {
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

    #[staticmethod]
    fn lincomb_mul_real(
        lhs_impl: &Self,
        lhs_coeffs: &RealVec,
        rhs_impl: &Self,
        rhs_coeffs: &RealVec,
    ) -> PyResult<(Self, RealVec)> {
        let out = normal::cmpnt_major::lincomb::mul(
            &normal_terms_real(lhs_impl, lhs_coeffs),
            &normal_terms_real(rhs_impl, rhs_coeffs),
        )
        .to_py_result()?;
        Ok(normal_set_to_py_real(out))
    }

    #[staticmethod]
    fn lincomb_mul_complex(
        lhs_impl: &Self,
        lhs_coeffs: &ComplexVec,
        rhs_impl: &Self,
        rhs_coeffs: &ComplexVec,
    ) -> PyResult<(Self, ComplexVec)> {
        let out = normal::cmpnt_major::lincomb::mul(
            &normal_terms_complex(lhs_impl, lhs_coeffs),
            &normal_terms_complex(rhs_impl, rhs_coeffs),
        )
        .to_py_result()?;
        Ok(normal_set_to_py_complex(out))
    }

    #[staticmethod]
    fn lincomb_commutator_real(
        lhs_impl: &Self,
        lhs_coeffs: &RealVec,
        rhs_impl: &Self,
        rhs_coeffs: &RealVec,
    ) -> PyResult<(Self, RealVec)> {
        let out = normal::cmpnt_major::lincomb::commutator(
            &normal_terms_real(lhs_impl, lhs_coeffs),
            &normal_terms_real(rhs_impl, rhs_coeffs),
        )
        .to_py_result()?;
        Ok(normal_set_to_py_real(out))
    }

    #[staticmethod]
    fn lincomb_anticommutator_real(
        lhs_impl: &Self,
        lhs_coeffs: &RealVec,
        rhs_impl: &Self,
        rhs_coeffs: &RealVec,
    ) -> PyResult<(Self, RealVec)> {
        let out = normal::cmpnt_major::lincomb::anticommutator(
            &normal_terms_real(lhs_impl, lhs_coeffs),
            &normal_terms_real(rhs_impl, rhs_coeffs),
        )
        .to_py_result()?;
        Ok(normal_set_to_py_real(out))
    }

    #[staticmethod]
    fn lincomb_commutator_complex(
        lhs_impl: &Self,
        lhs_coeffs: &ComplexVec,
        rhs_impl: &Self,
        rhs_coeffs: &ComplexVec,
    ) -> PyResult<(Self, ComplexVec)> {
        let out = normal::cmpnt_major::lincomb::commutator(
            &normal_terms_complex(lhs_impl, lhs_coeffs),
            &normal_terms_complex(rhs_impl, rhs_coeffs),
        )
        .to_py_result()?;
        Ok(normal_set_to_py_complex(out))
    }

    #[staticmethod]
    fn lincomb_anticommutator_complex(
        lhs_impl: &Self,
        lhs_coeffs: &ComplexVec,
        rhs_impl: &Self,
        rhs_coeffs: &ComplexVec,
    ) -> PyResult<(Self, ComplexVec)> {
        let out = normal::cmpnt_major::lincomb::anticommutator(
            &normal_terms_complex(lhs_impl, lhs_coeffs),
            &normal_terms_complex(rhs_impl, rhs_coeffs),
        )
        .to_py_result()?;
        Ok(normal_set_to_py_complex(out))
    }

    #[staticmethod]
    fn lincomb_adjoint_real(cmpnts: &Self, coeffs: &RealVec) -> (Self, RealVec) {
        normal_terms_to_py_real(normal::cmpnt_major::lincomb::adjoint(&normal_terms_real(
            cmpnts, coeffs,
        )))
    }

    #[staticmethod]
    fn lincomb_adjoint_complex(cmpnts: &Self, coeffs: &ComplexVec) -> (Self, ComplexVec) {
        normal_terms_to_py_complex(normal::cmpnt_major::lincomb::adjoint(
            &normal_terms_complex(cmpnts, coeffs),
        ))
    }

    #[staticmethod]
    fn lincomb_is_hermitian_real(cmpnts: &Self, coeffs: &RealVec, atol: f64) -> bool {
        normal::cmpnt_major::lincomb::is_hermitian(&normal_terms_real(cmpnts, coeffs), atol)
    }

    #[staticmethod]
    fn lincomb_is_hermitian_complex(cmpnts: &Self, coeffs: &ComplexVec, atol: f64) -> bool {
        normal::cmpnt_major::lincomb::is_hermitian(&normal_terms_complex(cmpnts, coeffs), atol)
    }

    #[staticmethod]
    fn lincomb_conserves_particle_number_real(cmpnts: &Self, coeffs: &RealVec, atol: f64) -> bool {
        normal::cmpnt_major::lincomb::conserves_particle_number(
            &normal_terms_real(cmpnts, coeffs),
            atol,
        )
    }

    #[staticmethod]
    fn lincomb_conserves_particle_number_complex(
        cmpnts: &Self,
        coeffs: &ComplexVec,
        atol: f64,
    ) -> bool {
        normal::cmpnt_major::lincomb::conserves_particle_number(
            &normal_terms_complex(cmpnts, coeffs),
            atol,
        )
    }

    #[staticmethod]
    fn lincomb_max_n_body_real(cmpnts: &Self, coeffs: &RealVec) -> usize {
        normal::cmpnt_major::lincomb::max_n_body(&normal_terms_real(cmpnts, coeffs))
    }

    #[staticmethod]
    fn lincomb_max_n_body_complex(cmpnts: &Self, coeffs: &ComplexVec) -> usize {
        normal::cmpnt_major::lincomb::max_n_body(&normal_terms_complex(cmpnts, coeffs))
    }

    #[staticmethod]
    fn lincomb_active_modes_real(cmpnts: &Self, coeffs: &RealVec) -> HashSet<usize> {
        normal::cmpnt_major::lincomb::active_modes(&normal_terms_real(cmpnts, coeffs))
    }

    #[staticmethod]
    fn lincomb_active_modes_complex(cmpnts: &Self, coeffs: &ComplexVec) -> HashSet<usize> {
        normal::cmpnt_major::lincomb::active_modes(&normal_terms_complex(cmpnts, coeffs))
    }

    #[staticmethod]
    fn lincomb_to_general_real(
        cmpnts: &Self,
        map: &Map,
        coeffs: &RealVec,
    ) -> (GeneralArray, RealVec) {
        let out =
            zixy::fermion::operator::lincomb::to_general(&normal_set_real(cmpnts, map, coeffs));
        general_set_to_py_real(out)
    }

    #[staticmethod]
    fn lincomb_to_general_complex(
        cmpnts: &Self,
        map: &Map,
        coeffs: &ComplexVec,
    ) -> (GeneralArray, ComplexVec) {
        let out =
            zixy::fermion::operator::lincomb::to_general(&normal_set_complex(cmpnts, map, coeffs));
        general_set_to_py_complex(out)
    }

    /// Apply this normal-ordered operator to a real state.
    fn apply_to_state_real(
        &self,
        coeffs: &RealVec,
        state_cmpnts: &StateArray,
        state_coeffs: &RealVec,
        out_cmpnts: &mut StateArray,
        out_map: &mut Map,
        out_coeffs: &mut RealVec,
    ) {
        use zixy::fermion::operator::normal::cmpnt_major::mat_elem::apply;
        use zixy::fermion::state;
        let op = normal::cmpnt_major::terms::View {
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

    /// Apply this normal-ordered operator to a complex state.
    fn apply_to_state_complex(
        &self,
        coeffs: &ComplexVec,
        state_cmpnts: &StateArray,
        state_coeffs: &ComplexVec,
        out_cmpnts: &mut StateArray,
        out_map: &mut Map,
        out_coeffs: &mut ComplexVec,
    ) {
        use zixy::fermion::operator::normal::cmpnt_major::mat_elem::apply;
        use zixy::fermion::state;
        let op = normal::cmpnt_major::terms::View {
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

    /// Evaluate a real matrix element between real states.
    fn mat_elem_real(
        &self,
        coeffs: &RealVec,
        bra_cmpnts: &StateArray,
        bra_coeffs: &RealVec,
        ket_cmpnts: &StateArray,
        ket_coeffs: &RealVec,
    ) -> f64 {
        use zixy::fermion::operator::normal::cmpnt_major::mat_elem::mat_elem;
        use zixy::fermion::state;
        let op = normal::cmpnt_major::terms::View {
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

    /// Evaluate a complex matrix element between complex states.
    fn mat_elem_complex(
        &self,
        coeffs: &ComplexVec,
        bra_cmpnts: &StateArray,
        bra_coeffs: &ComplexVec,
        ket_cmpnts: &StateArray,
        ket_coeffs: &ComplexVec,
    ) -> Complex64 {
        use zixy::fermion::operator::normal::cmpnt_major::mat_elem::mat_elem;
        use zixy::fermion::state;
        let op = normal::cmpnt_major::terms::View {
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
}

/// A list of raw, non-normal-ordered fermion ladder-operator products.
#[pyclass(subclass)]
#[pyo3(name = "GeneralFermionOperatorArray")]
#[repr(transparent)]
#[derive(Clone)]
pub struct GeneralArray(pub general::cmpnt_list::CmpntList);

impl Elements for GeneralArray {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl EmptyClone for GeneralArray {
    fn empty_clone(&self) -> Self {
        Self(self.0.empty_clone())
    }
}

fn general_terms_real<'a>(
    cmpnts: &'a GeneralArray,
    map: &'a CoreMap,
    coeffs: &'a RealVec,
) -> general::term_set::View<'a, f64> {
    general::term_set::View {
        word_iters: &cmpnts.0,
        coeffs: &coeffs.0,
        map,
    }
}

fn general_terms_complex<'a>(
    cmpnts: &'a GeneralArray,
    map: &'a CoreMap,
    coeffs: &'a ComplexVec,
) -> general::term_set::View<'a, Complex64> {
    general::term_set::View {
        word_iters: &cmpnts.0,
        coeffs: &coeffs.0,
        map,
    }
}

fn general_set_to_py_real(term_set: general::term_set::TermSet<f64>) -> (GeneralArray, RealVec) {
    (
        GeneralArray(term_set.terms.word_iters),
        RealVec(term_set.terms.coeffs),
    )
}

fn general_set_to_py_complex(
    term_set: general::term_set::TermSet<Complex64>,
) -> (GeneralArray, ComplexVec) {
    (
        GeneralArray(term_set.terms.word_iters),
        ComplexVec(clean_complex_coeffs(term_set.terms.coeffs)),
    )
}

fn clean_complex_coeffs(mut coeffs: Vec<Complex64>) -> Vec<Complex64> {
    for coeff in coeffs.iter_mut() {
        if coeff.re == 0.0 {
            coeff.re = 0.0;
        }
        if coeff.im == 0.0 {
            coeff.im = 0.0;
        }
    }
    coeffs
}

#[pymethods]
impl GeneralArray {
    #[new]
    #[pyo3(signature = (modes, max_len=0))]
    fn __init__(modes: Modes, max_len: usize) -> PyResult<Self> {
        Ok(Self(general::cmpnt_list::CmpntList::new(max_len, modes.0)))
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0.compatible_with(&other.0)
            && self.len() == other.len()
            && (0..self.len()).all(|i| self.0.get(i) == other.0.get(i))
    }

    fn same_as(&self, other: &Self) -> bool {
        std::ptr::eq(&self.0, &other.0)
    }

    fn resize(&mut self, n: usize) {
        self.0.resize(n);
    }

    #[getter]
    fn get_modes(&self) -> Modes {
        Modes(self.0.to_modes())
    }

    #[getter]
    fn get_max_len(&self) -> usize {
        self.0.max_len
    }

    fn append_clear(&mut self) {
        self.0.push(&[], &[]);
    }

    fn cmpnt_clear(&mut self, i: isize) -> PyResult<()> {
        self.cmpnt_set_from_ops(i, Vec::new(), Vec::new())
    }

    fn cmpnt_set_from_ops(&mut self, i: isize, modes: Vec<usize>, adj: Vec<bool>) -> PyResult<()> {
        check_general_ops(self.0.modes(), &modes, &adj, self.0.max_len)?;
        let i = try_py_index(i, self.len())?;
        let mut out = self.0.empty_clone();
        for j in 0..self.len() {
            if i == j {
                out.push(&modes, &adj);
            } else {
                let (row_modes, row_adj) = self.0.get(j);
                out.push(&row_modes, &row_adj);
            }
        }
        self.0 = out;
        Ok(())
    }

    fn cmpnt_get_ops(&self, i: isize) -> PyResult<(Vec<usize>, Vec<bool>)> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.get(i))
    }

    fn cmpnt_to_string(&self, i: isize) -> PyResult<String> {
        let i = try_py_index(i, self.len())?;
        Ok(self.0.fmt_elem(i))
    }

    fn cmpnt_copy_internal(&mut self, i_dst: isize, i_src: isize) -> PyResult<()> {
        let i_dst = try_py_index(i_dst, self.len())?;
        let i_src = try_py_index(i_src, self.len())?;
        let (modes, adj) = self.0.get(i_src);
        self.cmpnt_set_from_ops(i_dst as isize, modes, adj)
    }

    fn cmpnt_copy_external(&mut self, i_dst: isize, src: &Self, i_src: isize) -> PyResult<()> {
        DifferentSpaces::check(&self.0, &src.0).to_py_result()?;
        let i_dst = try_py_index(i_dst, self.len())?;
        let i_src = try_py_index(i_src, src.len())?;
        let (modes, adj) = src.0.get(i_src);
        self.cmpnt_set_from_ops(i_dst as isize, modes, adj)
    }

    fn cmpnt_equal(&self, i_lhs: isize, rhs: &Self, i_rhs: isize) -> PyResult<bool> {
        let i_lhs = try_py_index(i_lhs, self.len())?;
        let i_rhs = try_py_index(i_rhs, rhs.len())?;
        Ok(self.0.compatible_with(&rhs.0) && self.0.get(i_lhs) == rhs.0.get(i_rhs))
    }

    fn cmpnts_clone(&self, indices: Option<Vec<isize>>) -> PyResult<Self> {
        let mut out = self.0.empty_clone();
        match indices {
            Some(indices) => {
                for i in try_py_indices(indices, self.len())? {
                    let (modes, adj) = self.0.get(i);
                    out.push(&modes, &adj);
                }
            }
            None => {
                for i in 0..self.len() {
                    let (modes, adj) = self.0.get(i);
                    out.push(&modes, &adj);
                }
            }
        }
        Ok(Self(out))
    }

    fn cmpnts_eq(
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
            let l = match &indices {
                Some(v) => self.0.get(try_py_index(v[i], self.len())?),
                None => self.0.get(i),
            };
            let r = match &other_indices {
                Some(v) => other.0.get(try_py_index(v[i], other.len())?),
                None => other.0.get(i),
            };
            if l != r {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn refresh_map(&self, map: &mut Map) {
        map.0.populate_from(&self.0);
    }

    fn mapped_insert(&mut self, map: &mut Map, other: &Self, index: isize) -> PyResult<usize> {
        let index = try_py_index(index, other.len())?;
        if let Some(found) = (word_iters::set::View {
            word_iters: &self.0,
            map: &map.0,
        })
        .lookup(other.0.elem_u64it(index))
        {
            return Ok(found);
        }
        let (modes, adj) = other.0.get(index);
        let out = self.0.len();
        self.0.push(&modes, &adj);
        map.0.insert(self.0.hash_at_index(out), out);
        Ok(out)
    }

    fn mapped_lookup(&self, map: &Map, other: &Self, index: isize) -> PyResult<Option<usize>> {
        let index = try_py_index(index, other.len())?;
        let tmp = word_iters::set::View {
            word_iters: &self.0,
            map: &map.0,
        };
        Ok(tmp.lookup(other.0.elem_u64it(index)))
    }

    fn mapped_remove(
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

    fn mapped_equal(&self, map: &Map, other: &Self) -> PyResult<bool> {
        DifferentSpaces::check(&self.0, &other.0).to_py_result()?;
        if self.len() != other.len() {
            return Ok(false);
        }
        for i in 0..other.len() {
            if (word_iters::set::View {
                word_iters: &self.0,
                map: &map.0,
            })
            .lookup(other.0.elem_u64it(i))
            .is_none()
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[staticmethod]
    fn from_ladder_product(modes: Modes, ops: Vec<(usize, bool)>) -> PyResult<Self> {
        let (mode_inds, adj): (Vec<_>, Vec<_>) = ops.into_iter().unzip();
        check_general_ops(&modes.0, &mode_inds, &adj, mode_inds.len())?;
        let mut out = general::cmpnt_list::CmpntList::new(mode_inds.len(), modes.0);
        out.push(&mode_inds, &adj);
        Ok(Self(out))
    }

    #[staticmethod]
    fn lincomb_mul_real(
        lhs_impl: &Self,
        lhs_coeffs: &RealVec,
        rhs_impl: &Self,
        rhs_coeffs: &RealVec,
    ) -> PyResult<(Self, RealVec)> {
        let lhs_map = CoreMap::default();
        let rhs_map = CoreMap::default();
        let out = general::lincomb::rmul(
            &general_terms_real(lhs_impl, &lhs_map, lhs_coeffs),
            &general_terms_real(rhs_impl, &rhs_map, rhs_coeffs),
        )
        .to_py_result()?;
        Ok(general_set_to_py_real(out))
    }

    #[staticmethod]
    fn lincomb_mul_complex(
        lhs_impl: &Self,
        lhs_coeffs: &ComplexVec,
        rhs_impl: &Self,
        rhs_coeffs: &ComplexVec,
    ) -> PyResult<(Self, ComplexVec)> {
        let lhs_map = CoreMap::default();
        let rhs_map = CoreMap::default();
        let out = general::lincomb::rmul(
            &general_terms_complex(lhs_impl, &lhs_map, lhs_coeffs),
            &general_terms_complex(rhs_impl, &rhs_map, rhs_coeffs),
        )
        .to_py_result()?;
        Ok(general_set_to_py_complex(out))
    }

    #[staticmethod]
    fn lincomb_to_normal_order_real(cmpnts: &Self, coeffs: &RealVec) -> (NormalArray, RealVec) {
        let map = CoreMap::default();
        normal_set_to_py_real(zixy::fermion::operator::lincomb::to_normal_order(
            &general_terms_real(cmpnts, &map, coeffs),
        ))
    }

    #[staticmethod]
    fn lincomb_to_normal_order_complex(
        cmpnts: &Self,
        coeffs: &ComplexVec,
    ) -> (NormalArray, ComplexVec) {
        let map = CoreMap::default();
        normal_set_to_py_complex(zixy::fermion::operator::lincomb::to_normal_order(
            &general_terms_complex(cmpnts, &map, coeffs),
        ))
    }
}
