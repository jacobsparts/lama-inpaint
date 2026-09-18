# lama-inpaint (Rust engine)

See the [repository README](../README.md) for the project overview, the
quick start, the weight files and the GPU requirements. This crate is the
standalone engine: `src/cuda.rs` (CUDA), `src/cpu.rs` (CPU), `src/model.rs`
(the plan) and `src/weights.rs` (the flat blob store). Build with
`cargo build --release`; kernels are precompiled into `src/kernels.fatbin`, so
nvcc is not needed unless you edit `src/kernels.cu`.
