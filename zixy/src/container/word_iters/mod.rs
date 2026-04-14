//! Defines functionality for iterators over u64 words and containers generic over them.

pub use _word_iters::{
    test_defs, Elem, ElemMutRef, ElemRef, HasWordIters, HasWordItersMut, WordIters,
};
mod _word_iters;
pub mod lincomb;
pub mod set;
pub mod term_set;
pub mod terms;
pub mod traits;
