//! Trait definitions relating to container types.

use std::fmt::Display;

use crate::container::utils::DistinctPair;

/// Traits relating to the "projected view API" pattern.
/// In this pattern, there are three types:
/// - the owner: exists to hold data and create instances
/// - the view: immutably views the data held by the owner and implements read-only functionality on it
/// - the mutable view: mutably views the data held by the owner and implements readwrite functionality on it
pub mod proj {

    /// Owner-side trait for borrowing an immutable projected view on the members of the owner.
    pub trait Borrow<'a> {
        type RefType;

        /// Borrow an immutable projected view of `self`.
        fn borrow(&'a self) -> Self::RefType;
    }

    /// View-side trait for deriving another immutable projected view from an existing handle.
    pub trait AsRef<'a> {
        type RefType;

        /// Return an immutable projected view derived from `self`.
        fn as_ref(&'a self) -> Self::RefType;
    }

    /// Owner-side trait for borrowing a mutable projected view on the members of the owner.
    pub trait BorrowMut<'a> {
        type MutRefType;

        /// Borrow a mutable projected view of `self`.
        fn borrow_mut(&'a mut self) -> Self::MutRefType;
    }

    /// Trait for turning a projected view into an owned container value.
    pub trait ToOwned {
        type OwnedType;

        /// Materialize an owned value from this projected view.
        fn to_owned(&self) -> Self::OwnedType;
    }

    /// Trait for creating an owned container with matching metadata but no stored elements.
    pub trait EmptyOwned: ToOwned {
        /// Materialize an owned value with matching metadata but clear of element data.
        fn empty_owned(&self) -> Self::OwnedType;
    }
}

/// For any type that is identified with an unsigned integer index.
pub trait HasIndex {
    /// Return the index associated with this view or element handle.
    fn get_index(&self) -> usize;
}

/// A reference trait for a single element in a bijective data structure.
/// i.e. distinct indices point to distinct addresses.
pub trait RefElements<'a>: Elements {
    type Output: Display + PartialEq + HasIndex + 'a;

    /// Return an immutable reference-like view of the element at `index`.
    fn get_elem_ref(&'a self, index: usize) -> Self::Output;

    /// Return immutable views of the elements at indices `i` and `j`.
    fn get_pair_refs(&'a self, i: usize, j: usize) -> (Self::Output, Self::Output) {
        (self.get_elem_ref(i), self.get_elem_ref(j))
    }

    /// Iterate over immutable element views in index order.
    fn iter(&'a self) -> impl Iterator<Item = Self::Output> {
        (0..self.len()).map(|i| self.get_elem_ref(i))
    }
}

/// A reference trait for a single mutable element in a bijective data structure.
/// i.e. distinct indices point to distinct addresses.
pub trait MutRefElements<'a>: RefElements<'a> {
    type Output;

    /// Return a mutable reference-like view of the element at `index`.
    fn get_elem_mut_ref(&'a mut self, index: usize) -> <Self as MutRefElements<'a>>::Output;

    /// Return one mutable and one immutable view for a distinct pair of indices.
    fn get_semi_mut_refs(
        &'a mut self,
        inds: DistinctPair,
    ) -> (
        <Self as MutRefElements<'a>>::Output,
        <Self as RefElements<'a>>::Output,
    ) {
        let this = self as *mut Self;
        // This is alias-free because we assume bijectivity and DistinctPair instances with 0 and 1 fields equal cannot exist.
        unsafe {
            (
                (*this).get_elem_mut_ref(inds.get().0),
                (*this).get_elem_ref(inds.get().1),
            )
        }
    }

    /// Return mutable views for both indices in a distinct pair.
    fn get_pair_mut_refs(
        &'a mut self,
        inds: DistinctPair,
    ) -> (
        <Self as MutRefElements<'a>>::Output,
        <Self as MutRefElements<'a>>::Output,
    ) {
        let this = self as *mut Self;
        // This is alias-free because we assume bijectivity and DistinctPair instances with 0 and 1 fields equal cannot exist.
        unsafe {
            (
                (*this).get_elem_mut_ref(inds.get().0),
                (*this).get_elem_mut_ref(inds.get().1),
            )
        }
    }
}

/// Minimal container trait for values that contain a finite collection of elements.
pub trait Elements {
    /// Get the number of elements in the collection.
    fn len(&self) -> usize;

    /// Return whether there are no elements in the collection.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Container trait for swapping elements by index.
pub trait SwapElements: Elements {
    /// Swap the elements at the two distinct indices in `inds`.
    fn swap_distinct(&mut self, inds: DistinctPair);

    /// Swap the elements at `i` and `j` when they differ, and do nothing otherwise.
    fn swap_elements(&mut self, i: usize, j: usize) {
        if let Some(inds) = DistinctPair::new(i, j) {
            self.swap_distinct(inds);
        }
    }
}

/// Containers comprise metadata and elements.
/// Two containers are compatible with each other if their metadata are equal, but not necessarily the contents
/// or ordering of their elements. Containers can be compatible without being equal, but they can only be
/// equal if they are compatible.
pub trait Compatible {
    /// Return whether `self` and `_other` have compatible metadata.
    fn compatible_with(&self, _other: &Self) -> bool {
        true
    }
}

/// Trait for constructing a container pre-sized to hold a requested number of default elements.
pub trait NewWithLen: Elements {
    /// Create a new instance of an `Elements` implementor with a given number of default-valued elements
    fn new_with_len(n_element: usize) -> Self;
}

/// Trait for cloning only the metadata of a container while dropping its contents.
pub trait EmptyClone: Clone {
    /// Create a metadata-only clone of self, i.e. one with zero elements of data.
    fn empty_clone(&self) -> Self;
}

/// Trait for constructing an empty container by copying metadata from another container type.
pub trait EmptyFrom<T: Elements> {
    /// Create an instance of self, using only the metadata of value.
    fn empty_from(value: &T) -> Self;
}

/// Marker trait for immutable element-view types associated with a particular container.
pub trait ElementRef<'a> {
    type Container: RefElements<'a>;
}

/// Marker trait for mutable element-view types that can be reborrowed immutably.
pub trait ElementMutRef<'a> {
    type Container: RefElements<'a>;
    type ElementRefType<'b>: ElementRef<'b, Container = Self::Container>
    where
        Self: 'b;

    /// Reborrow this mutable element view as the corresponding immutable view.
    fn as_ref<'b>(&'b self) -> Self::ElementRefType<'b>;
}
