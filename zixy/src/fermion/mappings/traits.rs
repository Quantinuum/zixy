//! Traits for mapping from fermion operators to Paulis.

/// For the update/parity/rho set formalism of Seeley, Richard, and Love (arXiv:1208.5986). These are the
/// only three index arrays required to construct the fermion ladder operators in terms of qubit pauli operators
pub trait UpdateParityRho {
    /// Get the update set of qubit mode indices
    fn update_set(i: usize, n_mode: usize) -> Vec<usize>;

    /// Get the parity set of qubit mode indices
    fn parity_set(i: usize, n_mode: usize) -> Vec<usize>;

    /// Get the rho set of qubit mode indices
    fn rho_set(i: usize, n_mode: usize) -> Vec<usize>;
}
