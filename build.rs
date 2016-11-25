extern crate gcc;
extern crate walkdir;

use walkdir::{WalkDir};

use std::env;

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  for entry in WalkDir::new("kernels") {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }
  let out_dir = env::var("OUT_DIR").unwrap();
  let cc = env::var("CC").unwrap_or("gcc".to_owned());
  gcc::Config::new()
    .compiler(&cc)
    .opt_level(3)
    .pic(true)
    .flag("-std=gnu99")
    .flag("-march=native")
    .flag("-fno-strict-aliasing")
    .flag("-Ikernels")
    .file("kernels/vector.c")
    .compile("libdensearray_kernels.a");
  if cfg!(not(feature = "knl")) {
    gcc::Config::new()
      .compiler(&cc)
      .opt_level(3)
      .pic(true)
      .flag("-std=gnu99")
      .flag("-march=native")
      .flag("-fno-strict-aliasing")
      .flag("-fopenmp")
      .flag("-Ikernels")
      .flag("-DDENSEARRAY_OMP")
      .file("kernels/vector.c")
      .compile("libdensearray_omp_kernels.a");
  } else {
    gcc::Config::new()
      .compiler("icc")
      .opt_level(2)
      .pic(true)
      .flag("-std=c99")
      //.flag("-march=native")
      .flag("-fno-strict-aliasing")
      .flag("-openmp")
      .flag("-no-offload")
      .flag("-xMIC-AVX512")
      .flag("-Ikernels")
      .flag("-DDENSEARRAY_OMP")
      .file("kernels/vector.c")
      .compile("libdensearray_omp_kernels.a");
  }
  println!("cargo:rustc-link-search=native={}", out_dir);
}
