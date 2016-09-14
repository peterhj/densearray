#![feature(zero_one)]

extern crate openblas;

use std::marker::{PhantomData};
use std::num::{Zero};

pub mod linalg;

pub trait ArrayIndex: Copy {
  fn least_stride(&self) -> Self;
  fn flat_len(&self) -> usize;
  fn offset(&self, stride: Self) -> usize;
  fn diff(&self, rhs: Self) -> Self;

  /*fn major_iter(self) -> MajorIter<Self> where Self: Default {
    MajorIter{
      idx:          Default::default(),
      upper_bound:  self,
    }
  }*/
}

impl ArrayIndex for usize {
  fn least_stride(&self) -> Self {
    1
  }

  fn flat_len(&self) -> usize {
    *self
  }

  fn offset(&self, stride: Self) -> usize {
    *self * stride
  }

  fn diff(&self, rhs: Self) -> Self {
    *self - rhs
  }
}

impl ArrayIndex for (usize, usize) {
  fn least_stride(&self) -> Self {
    (1, self.0)
  }

  fn flat_len(&self) -> usize {
    self.0 * self.1
  }

  fn offset(&self, stride: Self) -> usize {
    //stride.0 * (self.0 + stride.1 * self.1)
    stride.0 * self.0 + stride.1 * self.1
  }

  fn diff(&self, rhs: Self) -> Self {
    (self.0 - rhs.0, self.1 - rhs.1)
  }
}

impl ArrayIndex for (usize, usize, usize) {
  fn least_stride(&self) -> Self {
    (1, self.0, self.0 * self.1)
  }

  fn flat_len(&self) -> usize {
    self.0 * self.1 * self.2
  }

  fn offset(&self, stride: Self) -> usize {
    //stride.0 * (self.0 + stride.1 * (self.1 + stride.2 * self.2))
    stride.0 * self.0 + stride.1 * self.1 + stride.2 * self.2
  }

  fn diff(&self, rhs: Self) -> Self {
    (self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
  }
}

pub trait Reshape<'a, Idx, Target> where Idx: ArrayIndex {
  fn reshape(&'a self, dim: Idx) -> Target;
  fn reshape_stride(&'a self, dim: Idx, stride: Idx) -> Target { unimplemented!(); }
}

pub trait ReshapeMut<'a, Idx, Target> where Idx: ArrayIndex {
  fn reshape_mut(&'a mut self, dim: Idx) -> Target;
  fn reshape_mut_stride(&'a mut self, dim: Idx, stride: Idx) -> Target { unimplemented!(); }
}

pub trait View<'a, Idx, Target> where Idx: ArrayIndex {
  fn view(&'a self, lo: Idx, hi: Idx) -> Target;
}

pub trait ViewMut<'a, Idx, Target> where Idx: ArrayIndex {
  fn view_mut(&'a mut self, lo: Idx, hi: Idx) -> Target;
}

pub trait AsView<'a, Target> {
  fn as_view(&'a self) -> Target;
}

pub trait AsViewMut<'a, Target> {
  fn as_view_mut(&'a mut self) -> Target;
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for [T] where T: Copy {
  fn reshape(&'a self, dim: (usize, usize)) -> Array2dView<'a, T> {
    // Assume unit stride.
    Array2dView{
      data_buf: self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for [T] where T: Copy {
  fn reshape_mut(&'a mut self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    // Assume unit stride.
    Array2dViewMut{
      data_buf: self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

pub trait ArrayStorage<T>: Clone + AsRef<[T]> + AsMut<[T]> {}

impl<T> ArrayStorage<T> for Vec<T> where T: Copy {}

#[derive(Clone)]
pub struct Array1d<T, S=Vec<T>> where T: Copy, S: ArrayStorage<T> {
  data_buf: S,
  dim:      usize,
  stride:   usize,
  _marker:  PhantomData<T>,
}

impl<T> Array1d<T> where T: Copy + Zero {
  pub fn zeros(dim: usize) -> Array1d<T> {
    let len = dim.flat_len();
    let mut data = Vec::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }
    Array1d{
      data_buf: data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array1d<T, S> where T: Copy, S: ArrayStorage<T> {
}

#[derive(Clone)]
pub struct Array2d<T, S=Vec<T>> where T: Copy, S: ArrayStorage<T> {
  data_buf: S,
  dim:      (usize, usize),
  stride:   (usize, usize),
  _marker:  PhantomData<T>,
}

impl<T> Array2d<T> where T: Copy + Zero {
  pub fn zeros(dim: (usize, usize)) -> Array2d<T> {
    let len = dim.flat_len();
    let mut data = Vec::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }
    Array2d{
      data_buf: data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array2d<T, S> where T: Copy, S: ArrayStorage<T> {
  pub fn from_storage(dim: (usize, usize), data_buf: S) -> Array2d<T, S> {
    assert_eq!(dim.flat_len(), data_buf.as_ref().len());
    Array2d{
      data_buf: data_buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    self.data_buf.as_ref()
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.data_buf.as_mut()
  }
}

impl<'a, T, S> AsView<'a, Array2dView<'a, T>> for Array2d<T, S> where T: Copy, S: ArrayStorage<T> {
  fn as_view(&'a self) -> Array2dView<'a, T> {
    Array2dView{
      data_buf: self.data_buf.as_ref(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array2dViewMut<'a, T>> for Array2d<T, S> where T: Copy, S: ArrayStorage<T> {
  fn as_view_mut(&'a mut self) -> Array2dViewMut<'a, T> {
    Array2dViewMut{
      data_buf: self.data_buf.as_mut(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

pub struct Array2dView<'a, T> where T: 'a + Copy {
  data_buf: &'a [T],
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<'a> Array2dView<'a, f32> {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }
}

impl<'a, T> View<'a, (usize, usize), Array2dView<'a, T>> for Array2dView<'a, T> where T: 'a + Copy {
  fn view(&'a self, lo: (usize, usize), hi: (usize, usize)) -> Array2dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array2dView{
      data_buf: &self.data_buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

pub struct Array2dViewMut<'a, T> where T: 'a + Copy {
  data_buf: &'a mut [T],
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> ViewMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array2dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(&'a mut self, lo: (usize, usize), hi: (usize, usize)) -> Array2dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array2dViewMut{
      data_buf: &mut self.data_buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

impl<'a> Array2dViewMut<'a, f32> {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn set_constant(&'a mut self, c: f32) {
    if self.stride == self.dim.least_stride() {
      for i in 0 .. self.dim.flat_len() {
        self.data_buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }
}

#[derive(Clone)]
pub struct Array3d<T, S=Vec<T>> where T: Copy, S: ArrayStorage<T> {
  data_buf: S,
  dim:      (usize, usize, usize),
  stride:   (usize, usize, usize),
  _marker:  PhantomData<T>,
}

impl<T, S> Array3d<T, S> where T: Copy, S: ArrayStorage<T> {
  pub fn from_storage(dim: (usize, usize, usize), data_buf: S) -> Array3d<T, S> {
    assert_eq!(dim.flat_len(), data_buf.as_ref().len());
    Array3d{
      data_buf: data_buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  pub fn dim(&self) -> (usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    self.data_buf.as_ref()
  }

  pub fn as_mut_slice(&mut self) -> &mut [T] {
    self.data_buf.as_mut()
  }
}
