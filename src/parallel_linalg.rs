use super::{Array1dView, Array1dViewMut, Array2dView, Array2dViewMut};
use kernels::*;
use linalg::{Transpose};

use openblas::ffi::*;

impl<'a> Array1dView<'a, f32> {
  pub fn parallel_l2_norm(&'a self) -> f32 {
    let n = self.dim();
    let incx = self.stride();
    unsafe { openblas_parallel_cblas_snrm2(
        n as _,
        self.buf.as_ptr(),
        incx as _,
    ) }
  }

  pub fn parallel_inner_prod(&'a self, alpha: f32, y: Array1dView<'a, f32>) -> f32 {
    let x_n = self.dim();
    let y_n = y.dim();
    assert_eq!(x_n, y_n);
    let incx = self.stride();
    let incy = y.stride();
    unsafe { openblas_parallel_cblas_sdot(
        x_n as _,
        alpha,
        self.buf.as_ptr(),
        incx as _,
        y.as_ptr(),
        incy as _,
    ) }
  }
}

impl<'a> Array2dViewMut<'a, f32> {
  pub fn parallel_matrix_prod(&'a mut self, alpha: f32, a: Array2dView<'a, f32>, a_trans: Transpose, b: Array2dView<'a, f32>, b_trans: Transpose, beta: f32) {
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
    unsafe { openblas_parallel_cblas_sgemm(
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
