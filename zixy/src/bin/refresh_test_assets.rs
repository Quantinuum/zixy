//! Binary for refreshing test assets.

use std::env;
use std::path::PathBuf;

use bincode::config;
use zixy::cmpnt::springs::ModeSettings;
use zixy::cmpnt::state_springs::BinarySprings;
use zixy::container::coeffs::traits::NumReprVec;
use zixy::container::traits::Elements;
use zixy::qubit::mode::Qubits;
use zixy::qubit::pauli::cmpnt_major::term_set::TermSet as PauliTermSet;
use zixy::qubit::pauli::cmpnt_major::terms::Terms as PauliTerms;
use zixy::qubit::pauli::springs::Springs;
use zixy::qubit::state::term_set::TermSet as StateTermSet;
use zixy::qubit::state::terms::Terms as StateTerms;
use zixy::utils::io::{file_path_in_crate, BinFileWriter};

fn main() {
    let arg = env::args().nth(1);
    match arg.as_deref().unwrap_or("mof_cas") {
        "mof_cas" => refresh_mof_cas(),
        other => panic!("unknown asset set: {other}"),
    }
}

fn refresh_mof_cas() {
    write_mof_cas_state();
    write_mof_cas_operator();
}

fn write_mof_cas_state() {
    let txt_path = file_path_in_crate("src/qubit/test_files/mof_cas/state.txt");
    let s = std::fs::read_to_string(&txt_path).unwrap();
    let coeffs = Vec::<f64>::try_parse(&s).unwrap();
    let springs = BinarySprings::all_parts_from_str(&s)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    println!(
        "refreshing mof_cas state: {} coeffs, {} basis states",
        coeffs.len(),
        springs.len()
    );
    let state = StateTermSet::<f64>::from(
        StateTerms::<f64>::from_springs_coeffs(Qubits::from_count(26), springs, coeffs).unwrap(),
    );
    encode_to_file(state, "src/qubit/test_files/mof_cas/state.bin");
}

fn write_mof_cas_operator() {
    let txt_path = file_path_in_crate("src/qubit/test_files/mof_cas/operator.txt");
    let s = std::fs::read_to_string(&txt_path).unwrap();
    let coeffs = Vec::<f64>::try_parse(&s).unwrap();
    let springs = Springs::all_parts_from_str(&s)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    println!(
        "refreshing mof_cas operator: {} coeffs, {} Pauli terms",
        coeffs.len(),
        springs.len()
    );
    let op = PauliTermSet::<f64>::from(
        PauliTerms::<f64>::from_springs_coeffs(Qubits::from_count(26), springs, coeffs).unwrap(),
    );
    encode_to_file(op, "src/qubit/test_files/mof_cas/operator.bin");
}

fn encode_to_file<T: serde::Serialize>(value: T, rel_path: &str) {
    let path: PathBuf = file_path_in_crate(rel_path);
    let writer = BinFileWriter::new(path).unwrap();
    bincode::serde::encode_into_writer(value, writer, config::standard()).unwrap();
}
