use super::{Array2dView, Array2dViewMut};

use openblas::ffi::*;

/*pub trait MatrixOps<T> {
  fn matrix_prod(&mut self, alpha: T, a: Array2dView, b: (), beta: T); 
}*/

#[derive(Clone, Copy)]
pub enum Transpose {
  N,
  T,
}

impl<'a> Array2dViewMut<'a, f32> {
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
