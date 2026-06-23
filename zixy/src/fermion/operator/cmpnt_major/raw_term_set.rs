//! Stores raw (non-normal-ordered) fermion terms.

// use crate::container::coeffs::traits::NumRepr;
// use crate::container::word_iters;
// use crate::fermion::operator::raw_cmpnt_list::RawCmpntList;

// pub type RawTermSet<C /*: NumRepr*/> = word_iters::term_set::TermSet<RawCmpntList, C>;
// pub type View<'a, C /*: NumRepr*/> = word_iters::term_set::View<'a, RawCmpntList, C>;
// pub type ViewMut<'a, C /*: NumRepr*/> = word_iters::term_set::ViewMut<'a, RawCmpntList, C>;

// pub trait AsView<C: NumRepr>: word_iters::term_set::AsView<RawCmpntList, C> {}
// pub trait AsViewMut<C: NumRepr>: word_iters::term_set::AsViewMut<RawCmpntList, C> {}

// impl<C: NumRepr> AsView<C> for RawTermSet<C> {}
// impl<'a, C: NumRepr> AsView<C> for View<'a, C> {}
// impl<'a, C: NumRepr> AsView<C> for ViewMut<'a, C> {}
// impl<C: NumRepr> AsViewMut<C> for RawTermSet<C> {}
// impl<'a, C: NumRepr> AsViewMut<C> for ViewMut<'a, C> {}
