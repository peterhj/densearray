extern crate gcc;
extern crate walkdir;

use walkdir::{WalkDir};

use std::env;
use std::path::{PathBuf};

fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let out_dir = env::var("OUT_DIR").unwrap();
  let mut kernels_src_dir = PathBuf::from(manifest_dir);
  kernels_src_dir.push("kernels");
  for entry in WalkDir::new(&kernels_src_dir.to_str().unwrap()) {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }
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
    /*if cfg!(feature = "mkl_parallel") {
      // XXX: For debugging to check that both features are enabled.
      println!("cargo:rerun-if-changed=mkl_parallel_dummy");
    }*/
    gcc::Config::new()
      .compiler("icc")
      .opt_level(2)
      .pic(true)
      .flag("-std=c99")
      //.flag("-march=native")
      .flag("-fno-strict-aliasing")
      .flag("-qopenmp")
      .flag("-qno-offload")
      .flag("-xMIC-AVX512")
      .flag("-Ikernels")
      .flag("-DDENSEARRAY_OMP")
      .file("kernels/vector.c")
      .compile("libdensearray_omp_kernels.a");
  }
  println!("cargo:rustc-link-search=native={}", out_dir);
}
