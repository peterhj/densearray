#![feature(zero_one)]

extern crate openblas;

use linalg::{Transpose};

use openblas::ffi::*;

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

pub trait ArrayStorage<T>: AsRef<[T]> + AsMut<[T]> {}

impl<T> ArrayStorage<T> for Vec<T> {}

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
    let lo_offset = lo.offset(self.stride);
    let hi_offset = hi.offset(self.stride);
    let new_dim = hi.diff(lo);
    Array2dView{
      data_buf: &self.data_buf[lo_offset .. hi_offset],
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
    let lo_offset = lo.offset(self.stride);
    let hi_offset = hi.offset(self.stride);
    let new_dim = hi.diff(lo);
    Array2dViewMut{
      data_buf: &mut self.data_buf[lo_offset .. hi_offset],
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

  pub fn set_constant(&'a mut self, alpha: f32) {
    for i in 0 .. self.dim.flat_len() {
      self.data_buf[i] = alpha;
    }
  }

  pub fn matrix_sum(&'a mut self, alpha: f32, x: Array2dView<'a, f32>) {
    let (x_m, x_n) = x.dim();
    let (y_m, y_n) = self.dim();
    assert_eq!(x_m, y_m);
    assert_eq!(x_n, y_n);
    let (incx, ldx) = x.stride();
    let (incy, ldy) = self.stride();
    if x_n == 1 {
      unsafe { cblas_saxpy(
          x_m as _,
          alpha,
          x.data_buf.as_ptr(),
          incx as _,
          self.data_buf.as_mut_ptr(),
          incy as _,
      ) };
    } else if x_m == 1 {
      unimplemented!();
    } else {
      unimplemented!();
    }
  }

  pub fn matrix_prod(&'a mut self, alpha: f32, a: Array2dView<'a, f32>, a_trans: Transpose, b: Array2dView<'a, f32>, b_trans: Transpose, beta: f32) {
    let (a_m, a_n) = a.dim();
    let (b_m, b_n) = b.dim();
    let (c_m, c_n) = self.dim();
    let k = match (a_trans, b_trans) {
      (Transpose::N, Transpose::N) => {
        assert_eq!(c_m, a_m);
        assert_eq!(c_n, b_n);
        assert_eq!(a_n, b_m);
        a_n
      }
      (Transpose::T, Transpose::N) => {
        assert_eq!(c_m, a_n);
        assert_eq!(c_n, b_n);
        assert_eq!(a_m, b_m);
        a_m
      }
      (Transpose::N, Transpose::T) => {
        assert_eq!(c_m, a_m);
        assert_eq!(c_n, b_m);
        assert_eq!(a_n, b_n);
        a_n
      }
      (Transpose::T, Transpose::T) => {
        assert_eq!(c_m, a_n);
        assert_eq!(c_n, b_m);
        assert_eq!(a_m, b_n);
        a_m
      }
    };
    let (a_s0, lda) = a.stride();
    let (b_s0, ldb) = b.stride();
    let (c_s0, ldc) = self.stride();
    assert_eq!(1, a_s0);
    assert_eq!(1, b_s0);
    assert_eq!(1, c_s0);
    unsafe { cblas_sgemm(
        CblasOrder::ColMajor,
        match a_trans {
          Transpose::N => CblasTranspose::NoTrans,
          Transpose::T => CblasTranspose::Trans,
        },
        match b_trans {
          Transpose::N => CblasTranspose::NoTrans,
          Transpose::T => CblasTranspose::Trans,
        },
        c_m as _, c_n as _, k as _,
        alpha,
        a.data_buf.as_ptr(),
        lda as _,
        b.data_buf.as_ptr(),
        ldb as _,
        beta,
        self.data_buf.as_mut_ptr(),
        ldc as _,
    ) };
  }
}

impl<'a> Array2dViewMut<'a, f64> {
  pub fn matrix_prod(&'a mut self) {
    unimplemented!();
  }
}
