//! Benchmarks for the zixy crate.

#[allow(dead_code)]
mod benchmarks;

use criterion::criterion_main;

criterion_main! {
    benchmarks::tableau::benches,
}
