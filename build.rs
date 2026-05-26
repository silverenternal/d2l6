fn main() {
    println!("cargo:rustc-link-search=native=/usr/lib");
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-lib=dylib=torch_hip");
    println!("cargo:rustc-link-lib=dylib=c10_hip");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
}
