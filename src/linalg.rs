use super::{Array1dView, Array1dViewMut, Array2dView, Array2dViewMut};

use openblas::ffi::*;

#[derive(Clone, Copy)]
pub enum Transpose {
  N,
  T,
}

impl<'a> Array1dView<'a, f32> {
  pub fn l2_norm(&'a self) -> f32 {
    let n = self.dim();
    let incx = self.stride();
    unsafe { openblas_sequential_cblas_snrm2(
        n as _,
        self.buf.as_ptr(),
        incx as _,
    ) }
  }

  pub fn elem_sum(&'a self) -> f32 {
    let x_n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    let mut x_sum = 0.0;
    for _ in 0 .. x_n {
      let x_i = self.buf[p];
      x_sum += x_i;
      p += incx;
    }
    x_sum
  }

  pub fn inner_prod(&'a self, alpha: f32, y: Array1dView<'a, f32>) -> f32 {
    let x_n = self.dim();
    let y_n = y.dim();
    assert_eq!(x_n, y_n);
    let incx = self.stride();
    let incy = y.stride();
    unsafe { openblas_sequential_cblas_sdot(
        x_n as _,
        alpha,
        self.buf.as_ptr(),
        incx as _,
        y.as_ptr(),
        incy as _,
    ) }
  }
}

impl<'a> Array1dViewMut<'a, f32> {
  pub fn add_scalar(&'a mut self, c: f32) {
    let n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    for _ in 0 .. n {
      self.buf[p] += c;
      p += incx;
    }
  }

  pub fn vector_add_scalar(&'a mut self, c: f32) {
    self.add_scalar(c);
  }

  pub fn square(&'a mut self) {
    let n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    for _ in 0 .. n {
      let x_i = self.buf[p];
      self.buf[p] = x_i * x_i;
      p += incx;
    }
  }

  pub fn vector_square(&'a mut self) {
    self.square();
  }

  pub fn sqrt(&'a mut self) {
    let n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    for _ in 0 .. n {
      let x_i = self.buf[p];
      self.buf[p] = x_i.sqrt();
      p += incx;
    }
  }

  pub fn vector_sqrt(&'a mut self) {
    self.sqrt();
  }

  pub fn reciprocal(&'a mut self) {
    let n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    for _ in 0 .. n {
      let x_i = self.buf[p];
      self.buf[p] = 1.0 / x_i;
      p += incx;
    }
  }

  pub fn vector_recip(&'a mut self) {
    self.reciprocal();
  }

  pub fn exp(&mut self) {
    let n = self.dim();
    let incx = self.stride();
    let mut p = 0;
    for _ in 0 .. n {
      let x_i = self.buf[p];
      self.buf[p] = x_i.exp();
      p += incx;
    }
  }

  pub fn scale(&'a mut self, alpha: f32) {
    let n = self.dim();
    let incx = self.stride();
    unsafe { openblas_sequential_cblas_sscal(
        n as _,
        alpha,
        self.buf.as_mut_ptr(),
        incx as _,
    ) }
  }

  pub fn vector_scale(&'a mut self, alpha: f32) {
    self.scale(alpha);
  }

  pub fn add(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    let x_n = x.dim();
    let y_n = self.dim();
    assert_eq!(x_n, y_n);
    let incx = x.stride();
    let incy = self.stride();
    unsafe { openblas_sequential_cblas_saxpy(
        x_n as _,
        alpha,
        x.buf.as_ptr(),
        incx as _,
        self.buf.as_mut_ptr(),
        incy as _,
    ) };
  }

  pub fn vector_add(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    self.add(alpha, x);
  }

  pub fn average(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    let x_n = x.dim();
    let y_n = self.dim();
    assert_eq!(x_n, y_n);
    let incx = x.stride();
    let incy = self.stride();
    let mut p = 0;
    let mut q = 0;
    for _ in 0 .. x_n {
      let x_i = x.buf[p];
      let y_i = self.buf[q];
      self.buf[q] = y_i + alpha * (x_i - y_i);
      p += incx;
      q += incy;
    }
  }

  pub fn elem_mult(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    let x_n = x.dim();
    let y_n = self.dim();
    assert_eq!(x_n, y_n);
    let incx = x.stride();
    let incy = self.stride();
    let mut p = 0;
    let mut q = 0;
    for _ in 0 .. x_n {
      let x_i = x.buf[p];
      let y_i = self.buf[q];
      self.buf[q] = alpha * x_i * y_i;
      p += incx;
      q += incy;
    }
  }

  pub fn vector_elem_mult(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    self.elem_mult(alpha, x);
  }

  pub fn elem_div(&'a mut self, alpha: f32, x: Array1dView<'a, f32>) {
    let x_n = x.dim();
    let y_n = self.dim();
    assert_eq!(x_n, y_n);
    let incx = x.stride();
    let incy = self.stride();
    let mut p = 0;
    let mut q = 0;
    for _ in 0 .. x_n {
      let x_i = x.buf[p];
      let y_i = self.buf[q];
      self.buf[q] = alpha * y_i / x_i;
      p += incx;
      q += incy;
    }
  }
}

impl<'a> Array2dView<'a, f32> {
  pub fn matrix_diag(&'a self, y: Array1dViewMut<'a, f32>) {
    let (a_m, a_n) = self.dim();
    let y_m = y.dim();
    assert_eq!(a_m, a_n);
    assert_eq!(a_m, y_m);
    let (a_inc, lda) = self.stride();
    let incy = y.stride();
    assert_eq!(1, a_inc);
    let mut p = 0;
    let mut q = 0;
    for _ in 0 .. y_m {
      y.buf[q] = self.buf[p];
      p += a_inc + lda;
      q += incy;
    }
  }
}

impl<'a> Array2dViewMut<'a, f32> {
  pub fn matrix_add(&'a mut self, alpha: f32, x: Array2dView<'a, f32>) {
    let (x_m, x_n) = x.dim();
    let (y_m, y_n) = self.dim();
    assert_eq!(x_m, y_m);
    assert_eq!(x_n, y_n);
    let (incx, ldx) = x.stride();
    let (incy, ldy) = self.stride();
    if x_n == 1 {
      unsafe { openblas_sequential_cblas_saxpy(
          x_m as _,
          alpha,
          x.buf.as_ptr(),
          incx as _,
          self.buf.as_mut_ptr(),
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
    let (at_m, at_n) = match a_trans {
      Transpose::N => (a_m, a_n),
      Transpose::T => (a_n, a_m),
    };
    let (bt_m, bt_n) = match b_trans {
      Transpose::N => (b_m, b_n),
      Transpose::T => (b_n, b_m),
    };
    assert_eq!(c_m, at_m);
    assert_eq!(c_n, bt_n);
    assert_eq!(at_n, bt_m);
    let k = at_n;
    let (a_inc, lda) = a.stride();
    let (b_inc, ldb) = b.stride();
    let (c_inc, ldc) = self.stride();
    assert_eq!(1, a_inc);
    assert_eq!(1, b_inc);
    assert_eq!(1, c_inc);
    unsafe { openblas_sequential_cblas_sgemm(
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
        a.buf.as_ptr(), lda as _,
        b.buf.as_ptr(), ldb as _,
        beta,
        self.buf.as_mut_ptr(), ldc as _,
    ) };
  }
}

impl<'a> Array2dViewMut<'a, f64> {
  pub fn matrix_prod(&'a mut self) {
    unimplemented!();
  }
}
