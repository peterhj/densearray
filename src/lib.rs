#![feature(specialization)]
#![feature(zero_one)]

//extern crate densearray_kernels;

extern crate byteorder;
extern crate cblas_ffi;
#[cfg(feature = "mkl_parallel")]
extern crate mkl_link;
extern crate openblas_ffi;

extern crate libc;

use kernels::*;

use std::marker::{PhantomData};
use std::mem::{size_of};
use std::num::{Zero};
use std::ops::{Deref, DerefMut};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub mod kernels;
pub mod linalg;
pub mod parallel_linalg;
pub mod prelude;
pub mod serial;

pub trait ArrayIndex: Copy {
  type Axes: Copy;

  fn least_stride(self) -> Self;
  fn flat_len(self) -> usize;
  fn offset(self, stride: Self) -> usize;
  fn diff(self, rhs: Self) -> Self;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Axes<T>(pub T);

impl ArrayIndex for usize {
  type Axes = Axes<usize>;

  fn least_stride(self) -> Self {
    1
  }

  fn flat_len(self) -> usize {
    self
  }

  fn offset(self, stride: Self) -> usize {
    self * stride
  }

  fn diff(self, rhs: Self) -> Self {
    self - rhs
  }
}

impl ArrayIndex for (usize, usize) {
  type Axes = Axes<(usize, usize)>;

  fn least_stride(self) -> Self {
    (1, self.0)
  }

  fn flat_len(self) -> usize {
    self.0 * self.1
  }

  fn offset(self, stride: Self) -> usize {
    stride.0 * self.0 + stride.1 * self.1
  }

  fn diff(self, rhs: Self) -> Self {
    (self.0 - rhs.0, self.1 - rhs.1)
  }
}

impl ArrayIndex for (usize, usize, usize) {
  type Axes = (usize, usize, usize);

  fn least_stride(self) -> Self {
    (1, self.0, self.0 * self.1)
  }

  fn flat_len(self) -> usize {
    self.0 * self.1 * self.2
  }

  fn offset(self, stride: Self) -> usize {
    stride.0 * self.0 + stride.1 * self.1 + stride.2 * self.2
  }

  fn diff(self, rhs: Self) -> Self {
    (self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
  }
}

impl ArrayIndex for (usize, usize, usize, usize) {
  type Axes = (usize, usize, usize, usize);

  fn least_stride(self) -> Self {
    (1, self.0, self.0 * self.1, self.0 * self.1 * self.2)
  }

  fn flat_len(self) -> usize {
    self.0 * self.1 * self.2 * self.3
  }

  fn offset(self, stride: Self) -> usize {
    stride.0 * self.0 + stride.1 * self.1 + stride.2 * self.2 + stride.3 * self.3
  }

  fn diff(self, rhs: Self) -> Self {
    (self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2, self.3 - rhs.3)
  }
}

pub trait Flatten<'a, Target> {
  fn flatten(self) -> Target;
}

pub trait FlattenMut<'a, Target> {
  fn flatten_mut(self) -> Target;
}

pub trait Reshape<'a, Idx, Target> where Idx: ArrayIndex {
  fn reshape(self, dim: Idx) -> Target;
  //fn reshape_stride(self, dim: Idx, stride: Idx) -> Target { unimplemented!(); }
}

pub trait ReshapeMut<'a, Idx, Target> where Idx: ArrayIndex {
  fn reshape_mut(self, dim: Idx) -> Target;
  //fn reshape_mut_stride(self, dim: Idx, stride: Idx) -> Target { unimplemented!(); }
}

pub trait AsView<'a, Target> {
  fn as_view(&'a self) -> Target;
}

pub trait AsViewMut<'a, Target> {
  fn as_view_mut(&'a mut self) -> Target;
}

pub trait View<'a, Idx, Target> where Idx: ArrayIndex {
  fn view(self, lo: Idx, hi: Idx) -> Target;
}

pub trait ViewMut<'a, Idx, Target> where Idx: ArrayIndex {
  fn view_mut(self, lo: Idx, hi: Idx) -> Target;
}

pub trait AliasBytes<'a, Target: ?Sized> {
  fn alias_bytes(self) -> Target;
}

pub trait AliasBytesMut<'a, Target: ?Sized> {
  fn alias_bytes_mut(self) -> Target;
}

pub trait SetConstant<'a, T> {
  fn set_constant(&'a mut self, c: T);
}

pub trait ParallelSetConstant<'a, T> {
  fn parallel_set_constant(&'a mut self, c: T);
}

// FIXME(20161104): this should be removed in favor of impl functions.
pub trait CastFrom<'a, Target: ?Sized> {
  fn cast_from(self, target: Target);
}

impl<'a> CastFrom<'a, &'a [u8]> for &'a mut [f32] {
  fn cast_from(self, xs: &'a [u8]) {
    for (x, y) in xs.iter().zip(self.iter_mut()) {
      *y = *x as f32;
    }
  }
}

impl<'a> AliasBytes<'a, &'a [f32]> for &'a [u8] {
  fn alias_bytes(self) -> &'a [f32] {
    let bytes_sz = self.len();
    let new_sz = bytes_sz / size_of::<f32>();
    assert_eq!(0, bytes_sz % size_of::<f32>());
    unsafe { from_raw_parts(self.as_ptr() as *const f32, new_sz) }
  }
}

impl<'a> AliasBytesMut<'a, &'a mut [f32]> for &'a mut [u8] {
  fn alias_bytes_mut(self) -> &'a mut [f32] {
    let bytes_sz = self.len();
    let new_sz = bytes_sz / size_of::<f32>();
    assert_eq!(0, bytes_sz % size_of::<f32>());
    unsafe { from_raw_parts_mut(self.as_mut_ptr() as *mut f32, new_sz) }
  }
}

impl<'a, T> Flatten<'a, Array1dView<'a, T>> for &'a [T] where T: Copy {
  fn flatten(self) -> Array1dView<'a, T> {
    let len = self.len();
    self.reshape(len)
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for &'a [T] where T: Copy {
  fn reshape(self, dim: usize) -> Array1dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim);
    Array1dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> FlattenMut<'a, Array1dViewMut<'a, T>> for &'a mut [T] where T: Copy {
  fn flatten_mut(self) -> Array1dViewMut<'a, T> {
    let len = self.len();
    self.reshape_mut(len)
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for &'a mut [T] where T: Copy {
  fn reshape_mut(self, dim: usize) -> Array1dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim);
    Array1dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for &'a [T] where T: Copy {
  fn reshape(self, dim: (usize, usize)) -> Array2dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array2dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for &'a mut [T] where T: Copy {
  fn reshape_mut(self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array2dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize, usize, usize), Array4dView<'a, T>> for &'a [T] where T: Copy {
  fn reshape(self, dim: (usize, usize, usize, usize)) -> Array4dView<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array4dView{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize, usize, usize), Array4dViewMut<'a, T>> for &'a mut [T] where T: Copy {
  fn reshape_mut(self, dim: (usize, usize, usize, usize)) -> Array4dViewMut<'a, T> {
    // Assume unit stride.
    assert!(self.len() >= dim.flat_len());
    Array4dViewMut{
      buf:      self,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for Array1dView<'a, T> where T: Copy {
  fn reshape(self, dim: (usize, usize)) -> Array2dView<'a, T> {
    assert!(dim == (self.dim, 1) || dim == (1, self.dim));
    if dim.1 == 1 {
      Array2dView{
        buf:      self.buf,
        dim:      dim,
        stride:   (self.stride, self.stride * self.dim),
      }
    } else if dim.0 == 1 {
      Array2dView{
        buf:      self.buf,
        dim:      dim,
        stride:   (1, self.stride),
      }
    } else {
      unreachable!();
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array1dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    assert!(dim == (self.dim, 1) || dim == (1, self.dim));
    if dim.1 == 1 {
      Array2dViewMut{
        buf:      self.buf,
        dim:      dim,
        stride:   (self.stride, self.stride * self.dim),
      }
    } else if dim.0 == 1 {
      Array2dViewMut{
        buf:      self.buf,
        dim:      dim,
        stride:   (1, self.stride),
      }
    } else {
      unreachable!();
    }
  }
}

impl<'a, T> Flatten<'a, Array1dView<'a, T>> for Array2dView<'a, T> where T: Copy {
  fn flatten(self) -> Array1dView<'a, T> {
    let len = self.dim.flat_len();
    self.reshape(len)
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for Array2dView<'a, T> where T: Copy {
  fn reshape(self, dim: usize) -> Array1dView<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dView{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> FlattenMut<'a, Array1dViewMut<'a, T>> for Array2dViewMut<'a, T> where T: Copy {
  fn flatten_mut(self) -> Array1dViewMut<'a, T> {
    let len = self.dim.flat_len();
    self.reshape_mut(len)
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for Array2dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: usize) -> Array1dViewMut<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> Flatten<'a, Array1dView<'a, T>> for Array3dView<'a, T> where T: Copy {
  fn flatten(self) -> Array1dView<'a, T> {
    let len = self.dim.flat_len();
    self.reshape(len)
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for Array3dView<'a, T> where T: Copy {
  fn reshape(self, dim: usize) -> Array1dView<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dView{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> FlattenMut<'a, Array1dViewMut<'a, T>> for Array3dViewMut<'a, T> where T: Copy {
  fn flatten_mut(self) -> Array1dViewMut<'a, T> {
    let len = self.dim.flat_len();
    self.reshape_mut(len)
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for Array3dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: usize) -> Array1dViewMut<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> Flatten<'a, Array1dView<'a, T>> for Array4dView<'a, T> where T: Copy {
  fn flatten(self) -> Array1dView<'a, T> {
    let len = self.dim.flat_len();
    self.reshape(len)
  }
}

impl<'a, T> Reshape<'a, usize, Array1dView<'a, T>> for Array4dView<'a, T> where T: Copy {
  fn reshape(self, dim: usize) -> Array1dView<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dView{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> FlattenMut<'a, Array1dViewMut<'a, T>> for Array4dViewMut<'a, T> where T: Copy {
  fn flatten_mut(self) -> Array1dViewMut<'a, T> {
    let len = self.dim.flat_len();
    self.reshape_mut(len)
  }
}

impl<'a, T> ReshapeMut<'a, usize, Array1dViewMut<'a, T>> for Array4dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: usize) -> Array1dViewMut<'a, T> {
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim);
    Array1dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   1,
    }
  }
}

impl<'a, T> Reshape<'a, (usize, usize), Array2dView<'a, T>> for Array4dView<'a, T> where T: Copy {
  fn reshape(self, dim: (usize, usize)) -> Array2dView<'a, T> {
    // FIXME(20161008): should do a stricter check, but this is barely sufficient.
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim.flat_len());
    Array2dView{
      buf:      self.buf,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

impl<'a, T> ReshapeMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array4dViewMut<'a, T> where T: Copy {
  fn reshape_mut(self, dim: (usize, usize)) -> Array2dViewMut<'a, T> {
    // FIXME(20161008): should do a stricter check, but this is barely sufficient.
    assert_eq!(self.dim.least_stride(), self.stride);
    assert_eq!(self.dim.flat_len(), dim.flat_len());
    Array2dViewMut{
      buf:      self.buf,
      dim:      dim,
      stride:   dim.least_stride(),
    }
  }
}

#[derive(Clone)]
pub struct Array1d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      usize,
  stride:   usize,
  _marker:  PhantomData<T>,
}

impl<T> Array1d<T> where T: Copy + Zero {
  pub fn zeros(dim: usize) -> Array1d<T> {
    let mut data = Vec::with_capacity(dim);
    data.resize(dim, T::zero());
    Array1d{
      buf:      data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array1d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: usize, buf: S) -> Array1d<T, S> {
    assert_eq!(dim.flat_len(), buf.len());
    Array1d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array1d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

impl<'a, T, S> AsView<'a, Array1dView<'a, T>> for Array1d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array1dView<'a, T> {
    Array1dView{
      buf:      self.buf.as_ref(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array1dViewMut<'a, T>> for Array1d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array1dViewMut<'a, T> {
    Array1dViewMut{
      buf:      self.buf.as_mut(),
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

#[derive(Clone, Copy)]
pub struct Array1dView<'a, T> where T: 'a + Copy {
  buf:      &'a [T],
  dim:      usize,
  stride:   usize,
}

impl<'a, T> Array1dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

impl<'a> Array1dView<'a, f32> {
}

impl<'a, T> View<'a, usize, Array1dView<'a, T>> for Array1dView<'a, T> where T: 'a + Copy {
  fn view(self, lo: usize, hi: usize) -> Array1dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array1dView{
      buf:      &self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

pub struct Array1dViewMut<'a, T> where T: 'a + Copy {
  buf:      &'a mut [T],
  dim:      usize,
  stride:   usize,
}

impl<'a, T> ViewMut<'a, usize, Array1dViewMut<'a, T>> for Array1dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(self, lo: usize, hi: usize) -> Array1dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array1dViewMut{
      buf:      &mut self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> Array1dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.buf.as_mut_ptr()
  }
}

impl<'a, T> SetConstant<'a, T> for Array1dViewMut<'a, T> where T: 'a + Copy {
  default fn set_constant(&'a mut self, c: T) {
    if self.stride == 1 {
      for i in 0 .. self.dim {
        self.buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }
}

impl<'a> SetConstant<'a, f32> for Array1dViewMut<'a, f32> {
  fn set_constant(&'a mut self, c: f32) {
    if self.stride == 1 {
      unsafe { densearray_set_scalar_f32(
          self.buf.as_mut_ptr(),
          self.dim,
          c,
      ) };
    } else {
      unimplemented!();
    }
  }
}

impl<'a> SetConstant<'a, i32> for Array1dViewMut<'a, i32> {
  fn set_constant(&'a mut self, c: i32) {
    if self.stride == 1 {
      unsafe { densearray_set_scalar_i32(
          self.buf.as_mut_ptr(),
          self.dim,
          c,
      ) };
    } else {
      unimplemented!();
    }
  }
}

impl<'a> Array1dViewMut<'a, f32> {
  pub fn parallel_set_constant(&'a mut self, c: f32) {
    if self.stride == 1 {
      unsafe { densearray_omp_set_scalar_f32(
          self.buf.as_mut_ptr(),
          self.dim,
          c,
      ) };
    } else {
      unimplemented!();
    }
  }
}

#[derive(Clone)]
pub struct Array2d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize),
  stride:   (usize, usize),
  _marker:  PhantomData<T>,
}

impl<T> Array2d<T> where T: Copy + Zero {
  pub fn zeros(dim: (usize, usize)) -> Array2d<T> {
    let len = dim.flat_len();
    let mut data = Vec::with_capacity(len);
    /*unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }*/
    data.resize(len, T::zero());
    Array2d{
      buf:      data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array2d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: (usize, usize), buf: S) -> Array2d<T, S> {
    assert_eq!(dim.flat_len(), buf.len());
    Array2d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  /*pub fn _upgrade_3d(self) -> Array3d<T, S> {
    Array3d{
      buf:      self.buf,
      dim:      (self.dim.0, self.dim.1, 1),
      stride:   (self.stride.0, self.stride.1, self.stride.flat_len()),
      _marker:  PhantomData,
    }
  }*/

  pub fn storage(&self) -> &S {
    &self.buf
  }

  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array2d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

impl<'a, T, S> AsView<'a, Array2dView<'a, T>> for Array2d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array2dView<'a, T> {
    Array2dView{
      buf:      &*self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array2dViewMut<'a, T>> for Array2d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array2dViewMut<'a, T> {
    Array2dViewMut{
      buf:      &mut *self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

#[derive(Clone, Copy)]
pub struct Array2dView<'a, T> where T: 'a + Copy {
  buf:      &'a [T],
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> Array2dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

impl<'a> Array2dView<'a, f32> {
}

impl<'a, T> View<'a, (usize, usize), Array2dView<'a, T>> for Array2dView<'a, T> where T: 'a + Copy {
  fn view(self, lo: (usize, usize), hi: (usize, usize)) -> Array2dView<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array2dView{
      buf:      &self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

pub struct Array2dViewMut<'a, T> where T: 'a + Copy {
  buf:      &'a mut [T],
  dim:      (usize, usize),
  stride:   (usize, usize),
}

impl<'a, T> ViewMut<'a, (usize, usize), Array2dViewMut<'a, T>> for Array2dViewMut<'a, T> where T: 'a + Copy {
  fn view_mut(self, lo: (usize, usize), hi: (usize, usize)) -> Array2dViewMut<'a, T> {
    let new_dim = hi.diff(lo);
    let new_offset = lo.offset(self.stride);
    let new_offset_end = new_offset + new_dim.flat_len();
    Array2dViewMut{
      buf:      &mut self.buf[new_offset .. new_offset_end],
      dim:      new_dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T> Array2dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize) {
    self.stride
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.buf.as_mut_ptr()
  }

  pub fn set_constant(&'a mut self, c: T) {
    if self.stride == self.dim.least_stride() {
      for i in 0 .. self.dim.flat_len() {
        self.buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }
}

impl<'a> Array2dViewMut<'a, f32> {
  pub fn parallel_set_constant(&'a mut self, c: f32) {
    if self.stride == self.dim.least_stride() {
      unsafe { densearray_omp_set_scalar_f32(
          self.buf.as_mut_ptr(),
          self.dim.flat_len(),
          c,
      ) };
    } else {
      unimplemented!();
    }
  }
}

#[derive(Clone)]
pub struct Array3d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize, usize),
  stride:   (usize, usize, usize),
  _marker:  PhantomData<T>,
}

impl<T> Array3d<T> where T: Copy + Zero {
  pub fn zeros(dim: (usize, usize, usize)) -> Array3d<T> {
    let len = dim.flat_len();
    let mut data = Vec::with_capacity(len);
    /*unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }*/
    data.resize(len, T::zero());
    Array3d{
      buf:      data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array3d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: (usize, usize, usize), buf: S) -> Array3d<T, S> {
    assert_eq!(dim.flat_len(), buf.as_ref().len());
    Array3d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  pub fn storage(&self) -> &S {
    &self.buf
  }

  pub fn into_storage(self) -> S {
    self.buf
  }

  pub fn dim(&self) -> (usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array3d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

#[derive(Clone, Copy)]
pub struct Array3dView<'a, T> where T: 'a + Copy {
  buf:      &'a [T],
  dim:      (usize, usize, usize),
  stride:   (usize, usize, usize),
}

impl<'a, T> Array3dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize) {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

pub struct Array3dViewMut<'a, T> where T: 'a + Copy {
  buf:      &'a mut [T],
  dim:      (usize, usize, usize),
  stride:   (usize, usize, usize),
}

impl<'a, T> Array3dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize) {
    self.stride
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.buf.as_mut_ptr()
  }

  pub fn set_constant(&'a mut self, c: T) {
    if self.stride == self.dim.least_stride() {
      for i in 0 .. self.dim.flat_len() {
        self.buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }
}

impl<'a, T, S> AsView<'a, Array3dView<'a, T>> for Array3d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array3dView<'a, T> {
    Array3dView{
      buf:      &*self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array3dViewMut<'a, T>> for Array3d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array3dViewMut<'a, T> {
    Array3dViewMut{
      buf:      &mut *self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

#[derive(Clone)]
pub struct Array4d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
  _marker:  PhantomData<T>,
}

impl<T> Array4d<T> where T: Copy + Zero {
  pub fn zeros(dim: (usize, usize, usize, usize)) -> Array4d<T> {
    let len = dim.flat_len();
    let mut data = Vec::with_capacity(len);
    /*unsafe { data.set_len(len) };
    for i in 0 .. len {
      data[i] = T::zero();
    }*/
    data.resize(len, T::zero());
    Array4d{
      buf:      data,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }
}

impl<T, S> Array4d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: (usize, usize, usize, usize), buf: S) -> Array4d<T, S> {
    assert_eq!(dim.flat_len(), buf.as_ref().len());
    Array4d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      _marker:  PhantomData,
    }
  }

  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_slice(&self) -> &[T] {
    &*self.buf
  }
}

impl<T, S> Array4d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    &mut *self.buf
  }
}

impl<'a, T, S> AsView<'a, Array4dView<'a, T>> for Array4d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array4dView<'a, T> {
    Array4dView{
      buf:      &*self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array4dViewMut<'a, T>> for Array4d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array4dViewMut<'a, T> {
    Array4dViewMut{
      buf:      &mut *self.buf,
      dim:      self.dim,
      stride:   self.stride,
    }
  }
}

#[derive(Clone, Copy)]
pub struct Array4dView<'a, T> where T: 'a + Copy {
  buf:      &'a [T],
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<'a, T> Array4dView<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_ptr(&self) -> *const T {
    self.buf.as_ptr()
  }
}

pub struct Array4dViewMut<'a, T> where T: 'a + Copy {
  buf:      &'a mut [T],
  dim:      (usize, usize, usize, usize),
  stride:   (usize, usize, usize, usize),
}

impl<'a, T> Array4dViewMut<'a, T> where T: 'a + Copy {
  pub fn dim(&self) -> (usize, usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize, usize) {
    self.stride
  }

  pub fn as_mut_ptr(&mut self) -> *mut T {
    self.buf.as_mut_ptr()
  }

  pub fn set_constant(&'a mut self, c: T) {
    if self.stride == self.dim.least_stride() {
      for i in 0 .. self.dim.flat_len() {
        self.buf[i] = c;
      }
    } else {
      unimplemented!();
    }
  }

  /*pub default fn parallel_set_constant(&'a mut self, c: T) {
    unimplemented!();
  }*/
}

impl<'a> Array4dViewMut<'a, f32> {
  pub fn parallel_set_constant(&'a mut self, c: f32) {
    if self.stride == self.dim.least_stride() {
      unsafe { densearray_omp_set_scalar_f32(
          self.buf.as_mut_ptr(),
          self.dim.flat_len(),
          c,
      ) };
    } else {
      unimplemented!();
    }
  }
}

pub struct Batch<A> {
  elems:    Vec<A>,
  batch_sz: usize,
}

impl<A> Batch<A> {
  pub fn new() -> Self {
    Batch{
      elems:    Vec::new(),
      batch_sz: 0,
    }
  }

  pub fn batch_size(&self) -> usize {
    self.batch_sz
  }
}

impl<A> Batch<A> where A: Clone {
  pub fn set_batch_size(&mut self, new_batch_sz: usize, default_elem: A) {
    if new_batch_sz > self.elems.len() {
      self.elems.resize(new_batch_sz, default_elem);
    }
    assert!(new_batch_sz <= self.elems.len());
    self.batch_sz = new_batch_sz;
  }
}

impl<A> Deref for Batch<A> {
  type Target = [A];

  fn deref(&self) -> &[A] {
    &self.elems[ .. self.batch_sz]
  }
}

impl<A> DerefMut for Batch<A> {
  fn deref_mut(&mut self) -> &mut [A] {
    &mut self.elems[ .. self.batch_sz]
  }
}

pub struct BatchArray1d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      usize,
  stride:   usize,
  max_batch_sz: usize,
  batch_sz:     usize,
  batch_stride: usize,
}

impl<T> BatchArray1d<T> where T: Copy + Zero {
  pub fn zeros(dim: usize, batch_cap: usize) -> BatchArray1d<T> {
    let len = dim.flat_len() * batch_cap;
    let mut data = Vec::with_capacity(len);
    data.resize(len, T::zero());
    Self::from_storage(dim, batch_cap, data)
  }
}

impl<T, S> BatchArray1d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: usize, batch_cap: usize, buf: S) -> BatchArray1d<T, S> {
    assert_eq!(dim.flat_len() * batch_cap, buf.len());
    BatchArray1d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      max_batch_sz: batch_cap,
      batch_sz:     batch_cap,
      batch_stride: dim.flat_len(),
    }
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn stride(&self) -> usize {
    self.stride
  }

  pub fn batch_size(&self) -> usize {
    self.batch_sz
  }

  pub fn set_batch_size(&mut self, new_batch_sz: usize) {
    assert!(new_batch_sz <= self.max_batch_sz);
    self.batch_sz = new_batch_sz;
  }
}

impl<'a, T, S> AsView<'a, Array2dView<'a, T>> for BatchArray1d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array2dView<'a, T> {
    Array2dView{
      buf:      &*self.buf,
      dim:      (self.dim, self.batch_sz),
      stride:   (self.stride, self.batch_stride),
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array2dViewMut<'a, T>> for BatchArray1d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array2dViewMut<'a, T> {
    Array2dViewMut{
      buf:      &mut *self.buf,
      dim:      (self.dim, self.batch_sz),
      stride:   (self.stride, self.batch_stride),
    }
  }
}

pub struct BatchArray2d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize),
  stride:   (usize, usize),
  max_batch_sz: usize,
  batch_sz:     usize,
  batch_stride: usize,
}

pub struct BatchArray3d<T, S=Vec<T>> where T: Copy, S: Deref<Target=[T]> {
  buf:      S,
  dim:      (usize, usize, usize),
  stride:   (usize, usize, usize),
  max_batch_sz: usize,
  batch_sz:     usize,
  batch_stride: usize,
}

impl<T, S> BatchArray3d<T, S> where T: Copy, S: Deref<Target=[T]> {
  pub fn from_storage(dim: (usize, usize, usize), batch_cap: usize, buf: S) -> BatchArray3d<T, S> {
    assert_eq!(dim.flat_len() * batch_cap, buf.len());
    BatchArray3d{
      buf:      buf,
      dim:      dim,
      stride:   dim.least_stride(),
      max_batch_sz: batch_cap,
      batch_sz:     batch_cap,
      batch_stride: dim.flat_len(),
    }
  }

  pub fn dim(&self) -> (usize, usize, usize) {
    self.dim
  }

  pub fn stride(&self) -> (usize, usize, usize) {
    self.stride
  }

  pub fn batch_size(&self) -> usize {
    self.batch_sz
  }

  pub fn set_batch_size(&mut self, new_batch_sz: usize) {
    assert!(new_batch_sz <= self.max_batch_sz);
    self.batch_sz = new_batch_sz;
  }
}

impl<'a, T, S> AsView<'a, Array4dView<'a, T>> for BatchArray3d<T, S> where T: Copy, S: Deref<Target=[T]> {
  fn as_view(&'a self) -> Array4dView<'a, T> {
    Array4dView{
      buf:      &*self.buf,
      dim:      (self.dim.0, self.dim.1, self.dim.2, self.batch_sz),
      stride:   (self.stride.0, self.stride.1, self.stride.2, self.batch_stride),
    }
  }
}

impl<'a, T, S> AsViewMut<'a, Array4dViewMut<'a, T>> for BatchArray3d<T, S> where T: Copy, S: DerefMut<Target=[T]> {
  fn as_view_mut(&'a mut self) -> Array4dViewMut<'a, T> {
    Array4dViewMut{
      buf:      &mut *self.buf,
      dim:      (self.dim.0, self.dim.1, self.dim.2, self.batch_sz),
      stride:   (self.stride.0, self.stride.1, self.stride.2, self.batch_stride),
    }
  }
}
