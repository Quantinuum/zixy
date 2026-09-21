//! Generic and specialized mappers between concrete types.

pub mod bk;
pub mod jw;
mod operators;
pub mod paraparticular;
pub mod parity;
mod traits;

/// A transformation from an input type to an output type.
pub trait Mapper<Input, Output> {
    /// Apply the mapper to `input`.
    fn apply(&mut self, input: Input) -> Output;
}
