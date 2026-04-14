use std::hint::black_box;

use criterion::{criterion_group, AxisScale, BenchmarkId, Criterion, PlotConfiguration};
use zixy::container::coeffs::complex_sign::ComplexSign;
use zixy::container::traits::proj::{Borrow, BorrowMut};
use zixy::container::traits::Elements;
use zixy::container::traits::MutRefElements;
use zixy::container::traits::RefElements;
use zixy::qubit::mode::PauliMatrix::*;
use zixy::qubit::mode::Qubits;
use zixy::qubit::pauli::cmpnt_major::term::Term;
use zixy::qubit::pauli::cmpnt_major::terms::{AsViewMut, Terms};
use zixy::qubit::traits::PushPaulis;

type PhasedOp = Term<ComplexSign>;
type PhasedOpList = Terms<ComplexSign>;

fn bench_pauli_mult(c: &mut Criterion) {
    let mut g = c.benchmark_group("multiply components within a Pauli tableau");
    let qubits = Qubits::from_count(20);
    let s0 = PhasedOp::from_vec(
        qubits.clone(),
        vec![I, Z, Z, Y, X, Z, X, Z, X, X, Z, Z, Y, Z, Y, Y, Z, Z, Z, I],
    )
    .unwrap();
    let s1 = PhasedOp::from_vec(
        qubits.clone(),
        vec![I, Y, X, I, I, X, Z, Y, Y, Y, X, Y, Y, Y, Z, X, Y, X, I, Y],
    )
    .unwrap();
    let s2 = PhasedOp::from_vec(
        qubits.clone(),
        vec![Y, X, I, Y, Y, Z, I, Y, I, I, Z, Z, I, Z, Z, Y, X, X, Y, Z],
    )
    .unwrap();
    let s3 = PhasedOp::from_vec(
        qubits.clone(),
        vec![Y, Y, I, I, I, I, Y, Z, Z, Z, X, I, Z, I, I, I, Z, Z, Y, Z],
    )
    .unwrap();

    let mut tab = PhasedOpList::new(qubits.clone());
    for i in 0..5 {
        tab.push_pauli_identity();
        tab.get_elem_mut_ref(4 * i).imul_unchecked(s0.borrow());
        tab.push_pauli_identity();
        tab.get_elem_mut_ref(4 * i + 1).imul_unchecked(s1.borrow());
        tab.push_pauli_identity();
        tab.get_elem_mut_ref(4 * i + 2).imul_unchecked(s2.borrow());
        tab.push_pauli_identity();
        tab.get_elem_mut_ref(4 * i + 3).imul_unchecked(s3.borrow());
    }

    g.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for n_iter in [1_000, 10_000, 100_000, 1_000_000] {
        g.bench_with_input(
            BenchmarkId::new("pauli_mult", n_iter),
            &n_iter,
            |b, n_iter| {
                b.iter(|| {
                    black_box({
                        for _ in 0..*n_iter {
                            for i in 0..5 {
                                tab.borrow_mut().imul(4 * i + 3, 4 * i);
                                tab.borrow_mut().imul(4 * i + 3, 4 * i + 1);
                                tab.borrow_mut().imul(4 * i + 2, 4 * i + 1);
                                tab.borrow_mut().imul(4 * i + 1, 4 * i);
                            }
                        }

                        let mut p0 = PhasedOp::from_vec(
                            qubits.clone(),
                            vec![Z, Y, I, X, Y, Z, I, I, I, Z, I, I, Z, X, I, Y, I, Z, Y, Y],
                        )
                        .unwrap();
                        p0.borrow_mut().set_coeff(ComplexSign::I);
                        for _ in 0..*n_iter {
                            for i_cmpnt in 0..tab.len() {
                                let tab_ref = tab.get_elem_ref(i_cmpnt);
                                if tab_ref
                                    .get_word_iter_ref()
                                    .anticommutes_with(p0.borrow().get_word_iter_ref())
                                {
                                    let _ = tab.get_elem_mut_ref(i_cmpnt).imul(p0.borrow());
                                }
                            }
                        }
                        0
                    })
                })
            },
        );
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_pauli_mult,
}
