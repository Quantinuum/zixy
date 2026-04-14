//! Utils for sparse matrices, mainly needed because `sprs` does not provide complex number functionality.

use num_complex::Complex64;

/// Complex conjugate the sparse matrix in-place
pub fn conj(sparse: &mut sprs::CsMat<Complex64>) {
    sparse.transpose_mut();
    for data in sparse.data_mut() {
        *data = Complex64::new(data.re, -data.im);
    }
}

/// Extract and return either the real or imag part of a complex-valued sparse matrix.
pub fn get_part(sparse: &sprs::CsMat<Complex64>, real: bool) -> sprs::CsMat<f64> {
    let data = sparse
        .data()
        .iter()
        .map(|x| if real { x.re } else { x.im })
        .collect::<Vec<f64>>();
    sprs::CsMat::<f64>::new(
        sparse.shape(),
        sparse.proper_indptr().to_vec(),
        sparse.indices().to_vec(),
        data,
    )
}

/// Convert the sparse matrix into another which contains only the real or imag part of the input.
pub fn into_part(sparse: sprs::CsMat<Complex64>, real: bool) -> sprs::CsMat<f64> {
    get_part(&sparse, real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn z(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[rstest]
    #[case((1, 1), vec![(0, 0, z(1.0, 2.0))], vec![(0, 0, z(1.0, -2.0))])]
    #[case((1, 2), vec![(0, 1, z(1.0, 2.0))], vec![(1, 0, z(1.0, -2.0))])]
    #[case((2, 2), vec![(0, 1, z(1.0, 2.0))], vec![(1, 0, z(1.0, -2.0))])]
    #[case((2, 2), vec![(0, 1, z(1.0, 2.0)), (1, 1, z(1.0, -3.0))],
                   vec![(1, 0, z(1.0, -2.0)), (1, 1, z(1.0, 3.0))])]
    fn test_conj(
        #[case] shape: (usize, usize),
        #[case] trips_in: Vec<(usize, usize, Complex64)>,
        #[case] trips_out: Vec<(usize, usize, Complex64)>,
    ) {
        use sprs::TriMat;
        let mut a = TriMat::<Complex64>::new(shape);
        for trip in trips_in {
            a.add_triplet(trip.0, trip.1, trip.2);
        }
        let mut a_csc = a.to_csc::<usize>();
        conj(&mut a_csc);

        let shape_t = (shape.1, shape.0);
        let mut b = TriMat::<Complex64>::new(shape_t);
        for trip in trips_out {
            b.add_triplet(trip.0, trip.1, trip.2);
        }
        let b_csc = b.to_csc::<usize>();
        assert_eq!(a_csc.to_dense(), b_csc.to_dense());
    }
}
