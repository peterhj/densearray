extern crate gcc;

use std::env;

fn main() {
  let out_dir = env::var("OUT_DIR").unwrap();
  let cc = env::var("CC").unwrap_or("gcc".to_owned());
  gcc::Config::new()
    .compiler(&cc)
    .opt_level(3)
    .pic(true)
    .flag("-march=native")
    .flag("--std=gnu99")
    .file("vector.c")
    .compile("libdensearray_kernels.a");
  println!("cargo:rustc-link-search=native={}", out_dir);
}
