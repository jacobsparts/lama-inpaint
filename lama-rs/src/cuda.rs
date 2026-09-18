//! Runtime CUDA backend.
//!
//! Everything CUDA is resolved through `dlopen` at run time: the binary builds
//! and runs on hosts with no CUDA and simply falls back to the CPU engine.  The
//! device kernels live in `kernels.cu`, are precompiled to `kernels.fatbin` and
//! embedded below, so neither nvcc nor the CUDA headers are needed to build.
//!
//! The embedded image is a fatbin holding SASS for several compute capabilities
//! (`src/build_kernels.sh` regenerates it, and `KERNEL_ARCHS` lists the targets
//! so a load failure on an unlisted GPU says so).  PTX is deliberately not
//! embedded: it would only help on architectures the image does not cover, and
//! a driver refuses PTX newer than the ISA it understands.
//!
//! Numerical contract: identical arithmetic to `cpu.rs`.  Convolutions are
//! im2col + cuBLAS SGEMM (single precision, no implicit conversions), the
//! Fourier unit uses batched cuFFT with the same `1/sqrt(h*w)` ortho scaling,
//! and the half-spectrum completion reproduces the verified 2-D Hermitian
//! reflection rule from `irfft2_ortho`.

use std::ffi::c_void;
use std::ptr;

use crate::model::{ConvRef, FfcBlock, Model, Step};
use crate::weights::WeightStore;

/// Precompiled SASS for the kernel set in `kernels.cu`, built with
/// `nvcc -arch=sm_61 -cubin`.  A cubin rather than PTX: the installed
/// 535.216.01 driver JITs PTX only up to ISA 8.2, while nvcc 12.4 emits 8.4,
/// and loading newer PTX fails with CUDA_ERROR_INVALID_DEVICE_FUNCTION (98).
/// SASS for the exact target bypasses JIT entirely.
const KERNELS_IMAGE: &[u8] = include_bytes!("kernels.fatbin");

const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_FORWARD_COMPATIBILITY: i32 = 804;

// cudaMemcpy kinds.
const MEMCPY_H2D: i32 = 1;
const MEMCPY_D2H: i32 = 2;

// cuBLAS operation codes.
const CUBLAS_OP_N: i32 = 0;

fn driver_version() -> Option<String> {
    let text = std::fs::read_to_string("/proc/driver/nvidia/version").ok()?;
    let first = text.lines().next()?;
    let parts: Vec<&str> = first.split_whitespace().collect();
    if parts.contains(&"Kernel") && parts.len() > 7 {
        Some(parts[7].to_string())
    } else {
        None
    }
}

/// Explain a `cuInit` failure in terms of the userspace/kernel driver split.
fn unavailable_reason(code: i32) -> String {
    let mut lines = vec![format!("CUDA is not available in this process (cuInit error {code}).")];
    if let Some(v) = driver_version() {
        lines.push(format!("  nvidia kernel module: {v}"));
    }
    if code == CUDA_ERROR_FORWARD_COMPATIBILITY {
        lines.push("  libcuda.so.1 does not match the loaded kernel module, so cuInit".to_string());
        lines.push("  fails with error 804.  Point LD_LIBRARY_PATH at the userspace".to_string());
        lines.push("  driver matching the module, e.g.".to_string());
        lines.push("    LD_LIBRARY_PATH=/home/jacob/nvidia-535.216.01".to_string());
    }
    lines.join("\n")
}

/// Compute capabilities present in the embedded kernel image.  `build.sh`
/// regenerates the image with exactly these `-gencode` targets; keep the two in
/// step, because this list is what the module-load error reports.
const KERNEL_ARCHS: &str = "6.1, 7.5, 8.0, 8.6, 8.9, 9.0";

// Device attributes used below (CUDA driver API).
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

/// Describe the current device after a module load failure.  Every query is
/// optional: a driver that cannot answer must still yield a usable message, so
/// failures are ignored rather than propagated.
fn device_note(driver: &Lib) -> String {
    // SAFETY: the signatures below are the documented ones for these symbols.
    unsafe {
        let cu_device_get: Result<unsafe extern "C" fn(*mut i32, i32) -> i32, String> =
            driver.sym("cuDeviceGet");
        let cu_device_get_name: Result<
            unsafe extern "C" fn(*mut std::ffi::c_char, i32, i32) -> i32,
            String,
        > = driver.sym("cuDeviceGetName");
        let cu_device_get_attribute: Result<unsafe extern "C" fn(*mut i32, i32, i32) -> i32, String> =
            driver.sym("cuDeviceGetAttribute");
        let (device_get, name_of, attribute) = match (cu_device_get, cu_device_get_name, cu_device_get_attribute) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => return format!(" (embedded kernels target compute {KERNEL_ARCHS})"),
        };
        let mut dev = 0i32;
        if device_get(&mut dev, 0) != 0 {
            return format!(" (embedded kernels target compute {KERNEL_ARCHS})");
        }
        let mut buf = [0 as std::ffi::c_char; 128];
        let name = if name_of(buf.as_mut_ptr(), buf.len() as i32, dev) == 0 {
            std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
        } else {
            "unknown device".to_string()
        };
        let (mut major, mut minor) = (0i32, 0i32);
        let cap = if attribute(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) == 0
            && attribute(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) == 0
        {
            format!(" (compute capability {major}.{minor})")
        } else {
            String::new()
        };
        format!(
            " on \"{name}\"{cap}; the embedded kernel image targets compute {KERNEL_ARCHS}"
        )
    }
}

struct Lib(*mut c_void);

// SAFETY: dlopen handles are process-wide and only read after loading.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

impl Lib {
    fn open(names: &[&str]) -> Result<Self, String> {
        let mut last = String::new();
        for name in names {
            let c = std::ffi::CString::new(*name).unwrap();
            // SAFETY: `c` is a valid NUL-terminated string.
            let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
            if !h.is_null() {
                return Ok(Lib(h));
            }
            last = format!("{name}: {}", dlerror_text());
        }
        Err(last)
    }

    /// # Safety
    /// Caller must know the real signature of `name`.
    unsafe fn sym<T: Copy>(&self, name: &str) -> Result<T, String> {
        let c = std::ffi::CString::new(name).unwrap();
        let p = libc::dlsym(self.0, c.as_ptr());
        if p.is_null() {
            return Err(format!("{name}: {}", dlerror_text()));
        }
        Ok(std::mem::transmute_copy(&p))
    }
}

fn dlerror_text() -> String {
    // SAFETY: dlerror returns a NUL-terminated string or null.
    unsafe {
        let e = libc::dlerror();
        if e.is_null() {
            "unknown dlopen/dlsym error".to_string()
        } else {
            std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
        }
    }
}

/// A device buffer of `len` f32.
struct Dev {
    ptr: *mut c_void,
    len: usize,
    /// False for windows into another buffer, which the owner frees.
    owned: bool,
}

impl Drop for Dev {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            // SAFETY: `ptr` came from cudaMalloc in this context.
            if !pool_release(self.ptr, self.len.max(1) * 4) {
                if let Some(c) = Cuda::get() {
                    unsafe { (c.free)(self.ptr) };
                }
            }
        }
    }
}

/// A size-bucketed device memory pool.
///
/// `cudaMalloc`/`cudaFree` cost about 100 us per pair on this driver and each
/// `cudaMalloc` implicitly synchronises the context, so a pass that makes a
/// thousand short-lived allocations loses a large fraction of its time to the
/// allocator alone.  Freed blocks are kept in a free list keyed by their exact
/// byte size (activations and patch matrices come in a handful of sizes and are
/// recycled in a tight loop, so the hit rate is near total).
///
/// The pool is deliberately simple: exact-size matching, a per-size list, and a
/// global byte cap.  Blocks are never handed back to the driver except when the
/// cap forces it, and the process exit reclaims whatever is left.
struct Pool {
    free: std::collections::HashMap<usize, Vec<*mut c_void>>,
    bytes: usize,
    hits: usize,
    misses: usize,
}

thread_local! {
    static POOL: std::cell::RefCell<Pool> = std::cell::RefCell::new(Pool {
        free: std::collections::HashMap::new(),
        bytes: 0,
        hits: 0,
        misses: 0,
    });
}

/// Keep at most this many bytes of freed device memory around.
const POOL_CAP: usize = 768 << 20;

fn pool_enabled() -> bool {
    // Checked once: the env var never changes mid-run.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAMA_NO_POOL").is_err())
}

/// A boolean environment flag, resolved once per process.
///
/// The hot paths used to call `std::env::var` (a getenv plus a heap allocation
/// and a string compare) on every convolution, every upsample and every
/// `op_scope`, i.e. tens of times per forward pass.  Env vars cannot change
/// mid-run, so each flag is cached in a `OnceLock` at first use.
fn flag(name: &'static str) -> bool {
    static FLAGS: std::sync::OnceLock<std::collections::HashMap<&'static str, bool>> =
        std::sync::OnceLock::new();
    let m = FLAGS.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        for n in [
            "LAMA_CONVT_GATHER",
            "LAMA_PROFILE_SUB",
            "LAMA_PROFILE_SYNC",
            "LAMA_FFT_HOST",
            "LAMA_PROFILE_OPS",
            "LAMA_DEBUG_CUFFT",
            "LAMA_PROFILE",
            "LAMA_NO_POOL",
        ] {
            m.insert(n, std::env::var(n).is_ok());
        }
        m
    });
    *m.get(name).unwrap_or(&false)
}

/// Take a block of exactly `bytes` from the pool, if one is free.
fn pool_acquire(bytes: usize) -> Option<*mut c_void> {
    if !pool_enabled() {
        return None;
    }
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let list = p.free.get_mut(&bytes)?;
        let ptr = list.pop()?;
        if list.is_empty() {
            p.free.remove(&bytes);
        }
        p.bytes -= bytes;
        p.hits += 1;
        Some(ptr)
    })
}

/// Return a block to the pool.  Returns false if the caller must free it.
fn pool_release(ptr: *mut c_void, bytes: usize) -> bool {
    if !pool_enabled() {
        return false;
    }
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.bytes + bytes > POOL_CAP {
            return false;
        }
        p.free.entry(bytes).or_default().push(ptr);
        p.bytes += bytes;
        true
    })
}

/// Resolved CUDA entry points.  Initialised once per process.
///
/// SAFETY: the handles are process-wide library/module/context pointers that are
/// only ever read after loading.  This one-shot worker uses them from a single
/// thread, and the buffers they name are owned by `Dev` values that are dropped
/// before the context goes away.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

pub struct Cuda {
    _driver: Lib,
    _cudart: Lib,
    _cublas: Lib,
    _cufft: Lib,
    // cudart
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> i32,
    free: unsafe extern "C" fn(*mut c_void) -> i32,
    memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32,
    stream_sync: unsafe extern "C" fn(*mut c_void) -> i32,
    get_error: unsafe extern "C" fn(i32) -> *const std::ffi::c_char,
    /// `cuLaunchKernel` from the driver API: the runtime's `cudaLaunchKernel`
    /// cannot launch functions that came from `cuModuleGetFunction`.
    launch: unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> i32,
    // driver
    module_get_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const std::ffi::c_char) -> i32,
    // cublas
    blas_set_stream: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    blas_sgemm: unsafe extern "C" fn(
        *mut c_void, i32, i32, i32, i32, i32, *const f32, *const c_void, i32, *const c_void, i32,
        *const f32, *mut c_void, i32,
    ) -> i32,
    // cufft
    /// `cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride,
    /// odist, type, batch)`.  The plan handle is an out-parameter, so it is a
    /// pointer-to-pointer rather than the opaque handle value itself.
    cufft_plan_many: unsafe extern "C" fn(
        *mut *mut c_void, i32, *mut i32, *mut i32, i32, i32, *mut i32, i32, i32, i32, i32,
    ) -> i32,
    cufft_set_stream: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    cufft_exec_r2c: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
    cufft_exec_c2r: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i32,
    // context
    blas: *mut c_void,
    stream: *mut c_void,
    module: *mut c_void,
    /// One device block holding a verbatim copy of the whole weight blob.
    ///
    /// The blob is a single contiguous mmap of 989 tensors with no padding, so
    /// one `cudaMemcpy` reproduces every tensor at `arena + info.offset`.  This
    /// replaces ~989 individual uploads, each of which cost about 111 us of
    /// pure API latency, with one copy of 204 MB that runs at PCIe speed.
    blob_arena: std::sync::OnceLock<std::rc::Rc<Dev>>,
    // kernel functions
    k_im2col: *mut c_void,
    k_conv_transpose: *mut c_void,
    k_bn_relu: *mut c_void,
    k_relu: *mut c_void,
    k_sigmoid: *mut c_void,
    k_add_inplace: *mut c_void,
    k_scale: *mut c_void,
    k_reflect_pad: *mut c_void,
    k_avgpool2x2: *mut c_void,
    k_copy_plane: *mut c_void,
    k_spec_pack: *mut c_void,
    k_spec_unpack: *mut c_void,
    k_cplx_real_scale: *mut c_void,
    k_convt_col: *mut c_void,
    k_convt_put: *mut c_void,
    k_bias_plane: *mut c_void,
    k_conv7x7: *mut c_void,
    k_bn_relu_add: *mut c_void,
}

static CUDA: std::sync::OnceLock<Result<Cuda, String>> = std::sync::OnceLock::new();

impl Cuda {
    fn get() -> Option<&'static Cuda> {
        match CUDA.get_or_init(Cuda::open) {
            Ok(c) => Some(c),
            Err(_) => None,
        }
    }

    fn open() -> Result<Cuda, String> {
        // The one-off setup is a large slice of a short run's wall time, so each
        // step is timed under `LAMA_PROFILE`: the four dlopens pull in the driver
        // plus the (very large) BLAS and FFT libraries, `cuInit` brings up the
        // device, `cublasCreate_v2` builds a handle, and only then is the cubin
        // loaded and its kernels resolved.
        let profile = flag("LAMA_PROFILE");
        let t = std::time::Instant::now();
        // The driver must be loaded globally so the runtime finds it.
        let driver = Lib::open(&["libcuda.so.1", "libcuda.so"])?;
        if profile {
            eprintln!("gpu ctx: dlopen libcuda {:.3}s", t.elapsed().as_secs_f32());
        }
        // SAFETY: cuInit takes an unsigned flags word.
        let init: unsafe extern "C" fn(u32) -> i32 = unsafe { driver.sym("cuInit")? };
        let t = std::time::Instant::now();
        let rc = unsafe { init(0) };
        if rc != CUDA_SUCCESS {
            return Err(unavailable_reason(rc));
        }
        if profile {
            eprintln!("gpu ctx: cuInit {:.3}s", t.elapsed().as_secs_f32());
        }

        let t = std::time::Instant::now();
        let cudart = Lib::open(&["libcudart.so.12", "libcudart.so"])?;
        let t_rt = t.elapsed().as_secs_f32();
        let t = std::time::Instant::now();
        let cublas = Lib::open(&["libcublas.so.12", "libcublas.so"])?;
        let t_blas = t.elapsed().as_secs_f32();
        let t = std::time::Instant::now();
        let cufft = Lib::open(&["libcufft.so.11", "libcufft.so"])?;
        let t_fft = t.elapsed().as_secs_f32();
        if profile {
            eprintln!(
                "gpu ctx: dlopen cudart {t_rt:.3}s cublas {t_blas:.3}s cufft {t_fft:.3}s"
            );
        }

        // SAFETY: every symbol is resolved under its documented signature.
        unsafe {
            let mut blas = ptr::null_mut();
            let blas_create: unsafe extern "C" fn(*mut *mut c_void) -> i32 = cublas.sym("cublasCreate_v2")?;
            let t = std::time::Instant::now();
            let r = blas_create(&mut blas);
            if r != 0 {
                return Err(format!("cublasCreate_v2 failed: {r}"));
            }
            if profile {
                eprintln!("gpu ctx: cublasCreate {:.3}s", t.elapsed().as_secs_f32());
            }
            let mut stream = ptr::null_mut();
            let stream_create: unsafe extern "C" fn(*mut *mut c_void) -> i32 = cudart.sym("cudaStreamCreate")?;
            let r = stream_create(&mut stream);
            if r != 0 {
                return Err(format!("cudaStreamCreate failed: {r}"));
            }

            let module_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32 =
                driver.sym("cuModuleLoadData")?;
            let mut module = ptr::null_mut();
            let t = std::time::Instant::now();
            let r = module_load(&mut module, KERNELS_IMAGE.as_ptr() as *const c_void);
            if r != 0 {
                // The blob carries SASS for several compute capabilities but not
                // for every GPU; name the device so an unsupported arch is
                // obvious rather than a bare error code.
                return Err(format!(
                    "cuModuleLoadData failed: {r}{}",
                    device_note(&driver)
                ));
            }
            if profile {
                eprintln!("gpu ctx: cuModuleLoadData {:.3}s", t.elapsed().as_secs_f32());
            }
            let module_get_function: unsafe extern "C" fn(
                *mut *mut c_void,
                *mut c_void,
                *const std::ffi::c_char,
            ) -> i32 = driver.sym("cuModuleGetFunction")?;

            let mut c = Cuda {
                malloc: cudart.sym("cudaMalloc")?,
                free: cudart.sym("cudaFree")?,
                memcpy: cudart.sym("cudaMemcpy")?,
                stream_sync: cudart.sym("cudaStreamSynchronize")?,
                get_error: cudart.sym("cudaGetErrorString")?,
                launch: driver.sym("cuLaunchKernel")?,
                module_get_function,
                blas_set_stream: cublas.sym("cublasSetStream_v2")?,
                blas_sgemm: cublas.sym("cublasSgemm_v2")?,
                cufft_plan_many: cufft.sym("cufftPlanMany")?,
                cufft_set_stream: cufft.sym("cufftSetStream")?,
                cufft_exec_r2c: cufft.sym("cufftExecR2C")?,
                cufft_exec_c2r: cufft.sym("cufftExecC2R")?,
                blas,
                stream,
                module,
                blob_arena: std::sync::OnceLock::new(),
                k_im2col: ptr::null_mut(),
                k_conv_transpose: ptr::null_mut(),
                k_bn_relu: ptr::null_mut(),
                k_relu: ptr::null_mut(),
                k_sigmoid: ptr::null_mut(),
                k_add_inplace: ptr::null_mut(),
                k_scale: ptr::null_mut(),
                k_reflect_pad: ptr::null_mut(),
                k_avgpool2x2: ptr::null_mut(),
                k_copy_plane: ptr::null_mut(),
                k_spec_pack: ptr::null_mut(),
                k_spec_unpack: ptr::null_mut(),
                k_cplx_real_scale: ptr::null_mut(),
                k_convt_col: ptr::null_mut(),
                k_convt_put: ptr::null_mut(),
                k_bias_plane: ptr::null_mut(),
                k_conv7x7: ptr::null_mut(),
                k_bn_relu_add: ptr::null_mut(),
                _driver: driver,
                _cudart: cudart,
                _cublas: cublas,
                _cufft: cufft,
            };
            (c.blas_set_stream)(c.blas, c.stream);
            (c.cufft_set_stream)(c.stream, c.stream);

            macro_rules! kernel {
                ($name:ident, $sym:expr) => {{
                    let sym = std::ffi::CString::new($sym).unwrap();
                    let mut f = ptr::null_mut();
                    let r = (c.module_get_function)(&mut f, c.module, sym.as_ptr());
                    if r != 0 {
                        return Err(format!("cuModuleGetFunction({}) failed: {r}", $sym));
                    }
                    c.$name = f;
                }};
            }
            kernel!(k_im2col, "k_im2col");
            kernel!(k_conv_transpose, "k_conv_transpose");
            kernel!(k_bn_relu, "k_bn_relu");
            kernel!(k_relu, "k_relu");
            kernel!(k_sigmoid, "k_sigmoid");
            kernel!(k_add_inplace, "k_add_inplace");
            kernel!(k_scale, "k_scale");
            kernel!(k_reflect_pad, "k_reflect_pad");
            kernel!(k_avgpool2x2, "k_avgpool2x2");
            kernel!(k_copy_plane, "k_copy_plane");
            kernel!(k_spec_pack, "k_spec_pack");
            kernel!(k_spec_unpack, "k_spec_unpack");
            kernel!(k_cplx_real_scale, "k_cplx_real_scale");
            kernel!(k_convt_col, "k_convt_col");
            kernel!(k_convt_put, "k_convt_put");
            kernel!(k_bias_plane, "k_bias_plane");
            kernel!(k_conv7x7, "k_conv7x7");
            kernel!(k_bn_relu_add, "k_bn_relu_add");
            Ok(c)
        }
    }

    fn err(&self, code: i32, what: &str) -> String {
        // SAFETY: cudaGetErrorString accepts any error code.
        let msg = unsafe {
            let p = (self.get_error)(code);
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        format!("{what}: cuda error {code} ({msg})")
    }

    fn alloc(&self, len: usize) -> Result<Dev, String> {
        let bytes = len.max(1) * 4;
        if let Some(p) = pool_acquire(bytes) {
            return Ok(Dev { ptr: p, len, owned: true });
        }
        let t = std::time::Instant::now();
        let mut p = ptr::null_mut();
        // SAFETY: `p` is a valid out-parameter.
        let r = unsafe { (self.malloc)(&mut p, bytes) };
        if r != 0 {
            return Err(self.err(r, "cudaMalloc"));
        }
        ALLOC_STATS.with(|s| {
            let mut s = s.borrow_mut();
            s.0 += 1;
            s.1 += t.elapsed().as_secs_f32();
        });
        POOL.with(|s| s.borrow_mut().misses += 1);
        Ok(Dev { ptr: p, len, owned: true })
    }

    /// Upload the entire weight blob in ONE `cudaMemcpy`.
    ///
    /// The blob's tensors are contiguous from offset 0 with no padding, so the
    /// device copy is a faithful image of the host mapping and every tensor can
    /// be addressed as `arena + offset` - see `Slice`.  Doing this once costs
    /// about as much as a fifth of the individual uploads it replaces.
    fn blob_arena(&self, blob: &[u8]) -> Result<std::rc::Rc<Dev>, String> {
        // Built on first use and reused for the process: the blob never changes.
        if let Some(a) = self.blob_arena.get() {
            return Ok(a.clone());
        }
        let t_arena = std::time::Instant::now();
        let n = (blob.len() + 3) / 4;
        // Straight from the driver, never through the pool: the pool's whole
        // purpose is to recycle short-lived activations, and this block lives
        // for the process.  If it came from the pool it would be released back
        // when the `Rc` drops and handed to the next activation, which would
        // overwrite the weights.
        let mut p = ptr::null_mut();
        // SAFETY: `p` is a valid out-parameter.
        let r = unsafe { (self.malloc)(&mut p, n.max(1) * 4) };
        if r != 0 {
            return Err(self.err(r, "cudaMalloc (weight arena)"));
        }
        let d = Dev { ptr: p, len: n, owned: true };
        let r = unsafe {
            (self.memcpy)(d.ptr, blob.as_ptr() as *const c_void, blob.len(), MEMCPY_H2D)
        };
        if r != 0 {
            return Err(self.err(r, "cudaMemcpy H2D (weight arena)"));
        }
        let rc = std::rc::Rc::new(d);
        let _ = self.blob_arena.set(rc.clone());
        if flag("LAMA_PROFILE") {
            eprintln!(
                "gpu weight arena: {:.1} MB in {:.3}s ({:.1} GB/s)",
                blob.len() as f32 / 1e6,
                t_arena.elapsed().as_secs_f32(),
                blob.len() as f32 / 1e9 / t_arena.elapsed().as_secs_f32().max(1e-6)
            );
        }
        Ok(rc)
    }

    /// A non-owning `Dev` window into the blob arena for a verbatim tensor.
    ///
    /// The arena is cached in `blob_arena` and, being a `OnceLock` owned by the
    /// process-wide `Cuda`, is never dropped, so the pointer stays valid for the
    /// life of the program and the `Dev` can safely be `owned: false`.
    fn blob_view(&self, store: &WeightStore, name: &str) -> Result<Dev, String> {
        let info = store.info(name)?.clone();
        let arena = self.blob_arena(store.blob.as_slice())?;
        // SAFETY: `info.offset + info.nbytes` was validated against the blob
        // length by `WeightStore::validate`, and the arena is a copy of the
        // whole blob, so the window is inside the allocation.
        Ok(Dev {
            ptr: unsafe { (arena.ptr as *mut u8).add(info.offset) as *mut c_void },
            len: info.nbytes / 4,
            owned: false,
        })
    }

    fn upload(&self, data: &[f32]) -> Result<Dev, String> {
        let d = self.alloc(data.len())?;
        if !data.is_empty() {
            // SAFETY: source is a valid slice, destination has len*4 bytes.
            let r = unsafe {
                (self.memcpy)(d.ptr, data.as_ptr() as *const c_void, data.len() * 4, MEMCPY_H2D)
            };
            if r != 0 {
                return Err(self.err(r, "cudaMemcpy H2D"));
            }
        }
        Ok(d)
    }

    fn download(&self, d: &Dev) -> Result<Vec<f32>, String> {
        let mut out = vec![0f32; d.len];
        if d.len > 0 {
            // SAFETY: destination is a valid slice, source has len*4 bytes.
            let r = unsafe {
                (self.memcpy)(out.as_mut_ptr() as *mut c_void, d.ptr, d.len * 4, MEMCPY_D2H)
            };
            if r != 0 {
                return Err(self.err(r, "cudaMemcpy D2H"));
            }
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_kernel(
        &self,
        f: *mut c_void,
        grid: u32,
        block: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        // SAFETY: `f` is a function obtained from this module and `args` holds
        // pointers to the argument values in the order the kernel declares them.
        // cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedMem, stream, args, extra)
        let r = unsafe {
            (self.launch)(
                f,
                grid,
                1,
                1,
                block,
                1,
                1,
                0,
                self.stream,
                args.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if r != 0 {
            return Err(self.err(r, "cudaLaunchKernel"));
        }
        LAUNCHES.with(|c| c.set(c.get() + 1));
        Ok(())
    }

    fn sync(&self) -> Result<(), String> {
        // SAFETY: `self.stream` is a stream created in this context.
        let r = unsafe { (self.stream_sync)(self.stream) };
        if r != 0 {
            return Err(self.err(r, "cudaStreamSynchronize"));
        }
        Ok(())
    }
}

// Count of kernels launched, for attributing per-op launch overhead.
thread_local! {
    static LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn grid_for(n: usize, block: u32) -> u32 {
    ((n as u64 + block as u64 - 1) / block as u64) as u32
}

const BLOCK: u32 = 256;

// ---------------------------------------------------------------------------
// Device activations and weights
// ---------------------------------------------------------------------------

/// A device-resident activation block `[c][h][w]`, NCHW row-major.
struct DActs {
    buf: Dev,
    c: usize,
    h: usize,
    w: usize,
}

impl DActs {
    fn new(c: usize, h: usize, w: usize) -> Result<Self, String> {
        let ctx = ctx()?;
        Ok(DActs { buf: ctx.alloc(c * h * w)?, c, h, w })
    }

    fn plane(&self) -> usize {
        self.h * self.w
    }

    /// A non-owning view of `channels` planes starting at `first`.
    ///
    /// The returned value borrows the parent buffer; it must not outlive it.
    /// Its `Dev` is marked `owned: false` so `Drop` leaves the pointer alone.
    fn view(&self, first: usize, channels: usize) -> DActs {
        DActs {
            buf: Dev {
                ptr: unsafe { (self.buf.ptr as *mut u8).add(first * self.plane() * 4) as *mut c_void },
                len: channels * self.plane(),
                owned: false,
            },
            c: channels,
            h: self.h,
            w: self.w,
        }
    }
}

/// A convolution weight `[cout][cin][kh][kw]` plus optional bias, on device.
struct DConv {
    w: Dev,
    bias: Option<Dev>,
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    /// The same weights reordered to `[cin][cout][kh][kw]`, used by the direct
    /// 7x7 kernel which walks one input channel at a time.  Only built for the
    /// shapes that kernel accepts.
    w_t: Option<Dev>,
}

/// Folded BatchNorm on device.
struct DBn {
    scale: Dev,
    shift: Dev,
    channels: usize,
}

#[allow(dead_code)]  // Mirrors `FfcWeights`: geometry carried for clarity.
struct DFfc {
    kernel: usize,
    stride: usize,
    pad: usize,
    reflect: bool,
    in_local: usize,
    in_global: usize,
    out_local: usize,
    out_global: usize,
    l2l: Option<DConv>,
    l2g: Option<DConv>,
    g2l: Option<DConv>,
    bn_l: Option<DBn>,
    bn_g: Option<DBn>,
    spectral: Option<DSpectral>,
}

/// The spectral (global-to-global) branch, mirroring `cpu::Spectral`:
///
///   pooled = avg_pool2x2(x) when stride == 2 else x
///   feat   = ReLU(bn1(conv1(pooled)))                     1x1, half -> half
///   fu     = irfft(bn_fu(fu_conv(stack(rfft(feat)))))     1x1 over 2*half
///   out    = conv2(feat + fu)                             1x1, half -> half
///
/// The FourierUnit adds the pre-transform `feat` to its inverse transform, and
/// the real/imaginary stacks are the channels its 1x1 convolution sees.
struct DSpectral {
    conv1: DConv,
    bn1: DBn,
    fu_conv: DConv,
    fu_bn: DBn,
    conv2: DConv,
    half: usize,
    stride: usize,
}

struct DConvT {
    w: Dev,
    bias: Dev,
    cin: usize,
    cout: usize,
    /// Per-phase weight matrices `[cout][cin*taps]` row-major for the phase
    /// GEMM path, in phase order `py*2 + px` with each phase's taps ordered as
    /// `kernels.cu`'s `CONVT_KY`/`CONVT_KX` tables list them.  `None` falls back
    /// to the gather kernel.
    phases: Option<Vec<Dev>>,
}

/// Tap sets per output phase for stride 2, pad 1, k 3; must match
/// `CONVT_KY`/`CONVT_KX` in kernels.cu.
const CONVT_TAPS: [[(usize, usize); 4]; 4] = [
    [(1, 1), (0, 0), (0, 0), (0, 0)],
    [(1, 0), (1, 2), (0, 0), (0, 0)],
    [(0, 1), (2, 1), (0, 0), (0, 0)],
    [(0, 0), (0, 2), (2, 0), (2, 2)],
];

/// Number of contributing taps per phase, matching `CONVT_TAPS` above and the
/// `CONVT_KY`/`CONVT_KX` tables in kernels.cu.
const CONVT_TAP_COUNT: [usize; 4] = [1, 2, 2, 4];

enum DStep {
    ReflectPad(usize),
    Ffc(DFfc),
    Res(DFfc, DFfc),
    Concat,
    Upsample(DConvT, DBn),
    OutConv(DConv),
}

fn ctx() -> Result<&'static Cuda, String> {
    Cuda::get().ok_or_else(|| match CUDA.get() {
        Some(Err(e)) => e.clone(),
        _ => "CUDA is unavailable".to_string(),
    })
}

/// Load-time staging for *derived* device tensors.
///
/// Most of the model is verbatim in the blob and now costs nothing to upload
/// (see `Cuda::blob_view`).  What is left are tensors the loader *computes* -
/// the folded BatchNorm scale/shift, the 7x7 weight reorder and the transposed
/// convolution's phase matrices - and there are 152 BatchNorms, so those tiny
/// uploads alone were 304 `cudaMemcpy` calls of a few hundred bytes each, every
/// one of them paying the full per-call latency.
///
/// `Staging` packs all of them into one host buffer; `finish` uploads that
/// buffer with a single `cudaMemcpy` and returns the arena.  Records are
/// `(byte_offset_in_arena, len_in_floats)`, and `window` turns one back into a
/// non-owning `Dev` pointing into the arena.
#[derive(Default)]
struct Staging {
    host: Vec<f32>,
    records: Vec<(usize, usize)>,
}

thread_local! {
    /// The load-time staging buffer, shared by every `load_*` helper.
    ///
    /// Threading it through the loader signatures would touch every function for
    /// no benefit: the whole load runs on one thread, once per process.
    static STAGING: std::cell::RefCell<Staging> = std::cell::RefCell::new(Staging::default());
    /// The uploaded staging arena's base pointer.
    ///
    /// A bare pointer rather than an `Rc<Dev>`, deliberately: these arenas live
    /// for the whole process, and a `Dev` stored in a thread-local would be
    /// dropped by the TLS destructor at exit - at which point `pool_release`
    /// would reach for the allocator's own thread-locals, which may already be
    /// gone, and the panic inside the destructor aborts the process *after* the
    /// result has been written.  The memory is intentionally never freed.
    static STAGING_ARENA: std::cell::Cell<*mut c_void> =
        const { std::cell::Cell::new(ptr::null_mut()) };
}

/// Record a derived tensor's values for the single staging upload, returning the
/// record index that `staged` later turns into a device window.
fn stage(data: &[f32]) -> usize {
    STAGING.with(|s| s.borrow_mut().push(data))
}

impl Staging {
    /// Copy `data.len()` floats into the buffer and return the record index.
    fn push(&mut self, data: &[f32]) -> usize {
        let off = self.host.len() * 4;
        self.host.extend_from_slice(data);
        // Round the next tensor up to a 16-byte boundary so every window is
        // aligned for vectorised kernel access.
        while self.host.len() * 4 % 16 != 0 {
            self.host.push(0.0);
        }
        self.records.push((off, data.len()));
        self.records.len() - 1
    }
}

/// Upload every staged tensor in ONE `cudaMemcpy` and remember the arena.
///
/// Called once, as soon as every folded BatchNorm has been recorded.
fn stage_commit() -> Result<(), String> {
    let host = STAGING.with(|s| std::mem::take(&mut s.borrow_mut().host));
    let c = ctx()?;
    // Straight from the driver, never the pool: this block lives for the process
    // and must never be recycled as an activation.  It is also never freed, so
    // no thread-local destructor has to touch the allocator at exit.
    let n = host.len();
    let mut p = ptr::null_mut();
    // SAFETY: `p` is a valid out-parameter.
    let r = unsafe { (c.malloc)(&mut p, n.max(1) * 4) };
    if r != 0 {
        return Err(c.err(r, "cudaMalloc (staged weights)"));
    }
    if n > 0 {
        // SAFETY: source has `n` floats, destination `n*4` bytes.
        let r = unsafe { (c.memcpy)(p, host.as_ptr() as *const c_void, n * 4, MEMCPY_H2D) };
        if r != 0 {
            return Err(c.err(r, "cudaMemcpy H2D (staged weights)"));
        }
    }
    STAGING_ARENA.with(|a| a.set(p));
    Ok(())
}

/// A device window for a staged tensor, valid after `stage_commit`.
fn staged(i: usize) -> Dev {
    let (off, len) = STAGING.with(|s| s.borrow().records[i]);
    STAGING_ARENA.with(|a| {
        let p = a.get();
        if p.is_null() {
            panic!("staged tensor used before stage_commit");
        }
        Dev {
            // SAFETY: `off + len*4` lies inside the arena, which is a copy of
            // exactly the host buffer the record was taken from.
            ptr: unsafe { (p as *mut u8).add(off) as *mut c_void },
            len,
            owned: false,
        }
    })
}

fn load_conv(store: &WeightStore, prefix: &str) -> Result<DConv, String> {
    let info = store.info(&format!("{prefix}.weight"))?.clone();
    // shapes are [cout][cin][kh][kw]
    let (cout, cin, kh, kw) = (info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
    let c = ctx()?;
    // The weight tensor is byte-identical in the blob and on the device, so it
    // is addressed inside the single arena copy rather than uploaded again.
    let w_dev = c.blob_view(store, &format!("{prefix}.weight"))?;
    let bias_dev = match store.info(&format!("{prefix}.bias")) {
        Ok(bi) if bi.numel() == cout => Some(c.blob_view(store, &format!("{prefix}.bias"))?),
        _ => None,
    };
    // The direct 7x7 kernel loops input channels outermost, so it wants
    // [cin][cout][kh][kw]; build that once here rather than per call.
    let w_t = if kh == 7 && kw == 7 && cout <= 8 {
        let w = store.f32(&format!("{prefix}.weight"))?;
        let mut t = vec![0f32; w.len()];
        for co in 0..cout {
            for ci in 0..cin {
                for ky in 0..kh {
                    for kx in 0..kw {
                        t[((ci * cout + co) * kh + ky) * kw + kx] = w[((co * cin + ci) * kh + ky) * kw + kx];
                    }
                }
            }
        }
        Some(c.upload(&t)?)
    } else {
        None
    };
    Ok(DConv {
        w: w_dev,
        bias: bias_dev,
        cin,
        cout,
        kh,
        kw,
        w_t,
    })
}

/// Fold one BatchNorm's four tensors into `(scale, shift)`, as `cpu::Bn::load`.
fn fold_bn(store: &WeightStore, prefix: &str) -> Result<(Vec<f32>, Vec<f32>), String> {
    let w = store.f32(&format!("{prefix}.weight"))?;
    let b = store.f32(&format!("{prefix}.bias"))?;
    let mean = store.f32(&format!("{prefix}.running_mean"))?;
    let var = store.f32(&format!("{prefix}.running_var"))?;
    let mut scale = vec![0f32; w.len()];
    let mut shift = vec![0f32; w.len()];
    for i in 0..w.len() {
        let s = w[i] / (var[i] + 1e-5).sqrt();
        scale[i] = s;
        shift[i] = b[i] - mean[i] * s;
    }
    Ok((scale, shift))
}

/// Fold and upload every BatchNorm in the model, once.
///
/// Every BatchNorm is identified by its `running_var` tensor, and there are 152
/// of them, i.e. 304 folded tensors - a few hundred bytes each, but 304 separate
/// `cudaMemcpy` calls at roughly 100 us of latency apiece.  They are packed into
/// one buffer (~317 KB in total, since the whole model has 40,576 BN channels)
/// and uploaded with a single copy instead.
fn stage_bns(store: &WeightStore) -> Result<(), String> {
    let mut names: Vec<String> = store
        .index
        .order
        .iter()
        .filter(|n| n.ends_with(".running_var"))
        .map(|n| n.trim_end_matches(".running_var").to_string())
        .collect();
    // Stable order so the record indices are reproducible, though the lookup
    // table below is keyed by name and does not depend on it.
    names.sort();
    let mut index = std::collections::HashMap::with_capacity(names.len() * 2);
    for name in &names {
        let (scale, shift) = fold_bn(store, name)?;
        // `stage` records in the first pass and only advances the cursor in the
        // second, so the two passes stay locked together - the index table is
        // kept from the recording pass.
        let si = stage(&scale);
        let ti = stage(&shift);
        index.insert(format!("{name}.scale"), si);
        index.insert(format!("{name}.shift"), ti);
    }
    stage_commit()?;
    BN_INDEX.with(|i| *i.borrow_mut() = Some(index));
    Ok(())
}

// Record indices of every staged BatchNorm tensor, keyed
// `"<module>.scale"` / `"<module>.shift"`, filled by `stage_bns`.
thread_local! {
    static BN_INDEX: std::cell::RefCell<Option<std::collections::HashMap<String, usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// BatchNorm is *derived*, not verbatim: the two device tensors are
/// `w/sqrt(var+eps)` and `b - mean*scale`, which do not appear in the blob.  They
/// are staged together with every other derived tensor, so each `DBn` holds
/// windows into one shared arena rather than owning two tiny allocations.
fn load_bn(store: &WeightStore, prefix: &str, channels: usize) -> Result<DBn, String> {
    // The folded values themselves are produced by `stage_bns`, which walks
    // every BatchNorm in one go; this call only needs the module's recorded
    // indices.  `fold_bn` here would be a duplicate fold whose result was
    // discarded, so the channel count is checked from the record instead.
    if BN_INDEX.with(|i| i.borrow().is_none()) {
        // The first BatchNorm folds the whole model's BNs at once; the upload
        // itself is committed after the entire traversal (see `load_steps`).
        stage_bns(store)?;
    }
    // The two record indices are looked up by module name, so no ordering
    // assumption is needed between this call and `stage_bns`'s traversal.
    let si = BN_INDEX.with(|i| {
        i.borrow()
            .as_ref()
            .and_then(|m| m.get(&format!("{prefix}.scale")).copied())
    });
    let ti = BN_INDEX.with(|i| {
        i.borrow()
            .as_ref()
            .and_then(|m| m.get(&format!("{prefix}.shift")).copied())
    });
    let (si, ti) = match (si, ti) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(format!("BatchNorm {prefix} was not staged")),
    };
    let dbn = DBn { scale: staged(si), shift: staged(ti), channels };
    if dbn.scale.len != channels || dbn.shift.len != channels {
        return Err(format!(
            "BatchNorm {prefix}: staged {} scale / {} shift values, expected {channels}",
            dbn.scale.len, dbn.shift.len
        ));
    }
    Ok(dbn)
}

fn load_dffc(store: &WeightStore, b: &FfcBlock) -> Result<DFfc, String> {
    let prefix = format!("model.{}", b.index);
    let ffc_prefix = match b.sub {
        0 => format!("{prefix}.ffc"),
        1 => format!("{prefix}.conv1.ffc"),
        _ => format!("{prefix}.conv2.ffc"),
    };
    let partial = match b.sub {
        0 => prefix.clone(),
        1 => format!("{prefix}.conv1"),
        _ => format!("{prefix}.conv2"),
    };
    let conv = |r: &ConvRef| -> Result<Option<DConv>, String> {
        match r {
            ConvRef::Absent => Ok(None),
            ConvRef::Present { name, .. } => Ok(Some(load_conv(store, &format!("{ffc_prefix}.{name}"))?)),
        }
    };
    let spectral = if b.g2g {
        // `Spectral::load` in cpu.rs uses the `convg2g` prefix: conv1/bn1 are
        // `conv1.0` and `conv1.1`, the FourierUnit is `fu.conv_layer`/`fu.bn`,
        // and conv2 closes the branch.
        let p = format!("{ffc_prefix}.convg2g");
        let half = b.out_global / 2;
        Some(DSpectral {
            conv1: load_conv(store, &format!("{p}.conv1.0"))?,
            bn1: load_bn(store, &format!("{p}.conv1.1"), half)?,
            fu_conv: load_conv(store, &format!("{p}.fu.conv_layer"))?,
            fu_bn: load_bn(store, &format!("{p}.fu.bn"), 2 * half)?,
            conv2: load_conv(store, &format!("{p}.conv2"))?,
            half,
            stride: b.stride,
        })
    } else {
        None
    };
    Ok(DFfc {
        kernel: b.kernel,
        stride: b.stride,
        pad: b.pad,
        reflect: b.reflection,
        in_local: b.in_local,
        in_global: b.in_global,
        out_local: b.out_local,
        out_global: b.out_global,
        l2l: conv(&b.l2l)?,
        l2g: conv(&b.l2g)?,
        g2l: conv(&b.g2l)?,
        bn_l: if b.bn_local {
            Some(load_bn(store, &format!("{partial}.bn_l"), b.out_local)?)
        } else {
            None
        },
        bn_g: if b.bn_global {
            Some(load_bn(store, &format!("{partial}.bn_g"), b.out_global)?)
        } else {
            None
        },
        spectral,
    })
}

/// Build the device step tree.
///
/// Only the folded BatchNorm tensors are staged (see `stage_bns`): they are the
/// overwhelming majority of the derived uploads - 152 modules, 304 tensors - and
/// staging them turns 304 `cudaMemcpy` calls into one.  The transposed
/// convolution's twelve phase matrices stay individual uploads, because staging
/// them would require a two-pass traversal whose second pass re-walks the whole
/// model and rebuilds the step tree, and measured against the twelve uploads it
/// replaces that trade is a net loss (load_steps 0.072 s single-pass versus
/// 0.079-0.082 s two-pass).
fn load_steps(store: &WeightStore, model: &Model) -> Result<Vec<DStep>, String> {
    let mut steps = Vec::new();
    for step in &model.steps {
        match step {
            Step::ReflectPad(p) => steps.push(DStep::ReflectPad(*p)),
            Step::Ffc(b) => steps.push(DStep::Ffc(load_dffc(store, b)?)),
            Step::ResBlock(rb) => steps.push(DStep::Res(
                load_dffc(store, &rb.conv1)?,
                load_dffc(store, &rb.conv2)?,
            )),
            Step::Concat => steps.push(DStep::Concat),
            Step::Upsample(u) => {
                let prefix = format!("model.{}", u.index);
                let info = store.info(&format!("{prefix}.weight"))?.clone();
                let w = store.f32(&format!("{prefix}.weight"))?;
                // The bias is not read here: it is passed to the phase kernel as
                // a separate argument, straight from `blob_view`.
                let c = ctx()?;
                // Reorder the [cin][cout][3][3] weight into one matrix per output
                // phase, shaped `[cout][cin*taps]`, so the phase GEMM can consume
                // it directly.  `CONVT_TAPS` fixes the tap order and must match
                // `CONVT_KY`/`CONVT_KX` in kernels.cu.
                let (cin, cout) = (info.shape[0], info.shape[1]);
                // One matrix per phase, `[cout][cin*taps]` row-major, where the
                // column index is `ci*taps + ti` to match the gather kernel's
                // `col[(ci*taps + ti)*n + j]`.  Source layout is [cin][cout][3][3].
                let mut phases = Vec::with_capacity(4);
                for (ph, taps) in CONVT_TAPS.iter().enumerate() {
                    let taps_n = CONVT_TAP_COUNT[ph];
                    let k = cin * taps_n;
                    let mut m = vec![0f32; cout * k];
                    for ci in 0..cin {
                        for ti in 0..taps_n {
                            let (ky, kx) = taps[ti];
                            for co in 0..cout {
                                let src = ((ci * cout + co) * 3 + ky) * 3 + kx;
                                m[co * k + ci * taps_n + ti] = w[src];
                            }
                        }
                    }
                    phases.push(c.upload(&m)?);
                }
                let convt = DConvT {
                    w: c.blob_view(store, &format!("{prefix}.weight"))?,
                    bias: c.blob_view(store, &format!("{prefix}.bias"))?,
                    cin,
                    cout,
                    phases: Some(phases),
                };
                // The upsample BN is a bare `model.N` module (not `model.N.bn`).
                let bn = load_bn(store, &format!("model.{}", u.bn_index), u.cout)?;
                steps.push(DStep::Upsample(convt, bn));
            }
            Step::OutConv(o) => {
                steps.push(DStep::OutConv(load_conv(store, &format!("model.{}", o.index))?))
            }
        }
    }
    Ok(steps)
}

// ---------------------------------------------------------------------------
// Device operations
// ---------------------------------------------------------------------------

/// im2col + SGEMM for a kxk convolution.
///
/// The patch matrix is `[cin*kh*kw][oh*ow]` row-major, which cuBLAS reads as an
/// `(oh*ow) x (cin*kh*kw)` column-major matrix without a transpose; the weights
/// `[cout][cin*kh*kw]` row-major read as `[cin*kh*kw] x [cout]` column-major.
/// So `C = op(A) * op(B)` with both operands "non-transposed" produces the
/// `[cout][oh*ow]` row-major output plane directly.
fn conv_forward(cv: &DConv, src: &DActs, pad: usize, stride: usize, reflect: bool) -> Result<DActs, String> {
    let c = ctx()?;
    if cv.kh == 1 && cv.kw == 1 && stride == 1 {
        return conv_1x1(cv, src);
    }
    // The 7x7 OutConv is the one shape where im2col dominates the arithmetic;
    // the direct kernel walks the receptive field instead.
    if cv.kh == 7 && cv.kw == 7 && stride == 1 && pad == 0 && cv.w_t.is_some() {
        return conv_7x7_direct(cv, src);
    }
    let oh = (src.h + 2 * pad - cv.kh) / stride + 1;
    let ow = (src.w + 2 * pad - cv.kw) / stride + 1;
    let k = cv.cin * cv.kh * cv.kw;
    let n = oh * ow;
    let col = c.alloc(k * n)?;
    let out = c.alloc(cv.cout * n)?;

    let mut src_p = src.buf.ptr;
    let mut col_p = col.ptr;
    let (mut cin, mut h, mut w) = (cv.cin as i32, src.h as i32, src.w as i32);
    let (mut kh, mut kw) = (cv.kh as i32, cv.kw as i32);
    let (mut ph, mut pw) = (pad as i32, pad as i32);
    let (mut st, mut oh_i, mut ow_i) = (stride as i32, oh as i32, ow as i32);
    let mut refl = if reflect { 1i32 } else { 0i32 };
    let mut args: Vec<*mut c_void> = vec![
        &mut src_p as *mut _ as *mut c_void,
        &mut col_p as *mut _ as *mut c_void,
        &mut cin as *mut _ as *mut c_void,
        &mut h as *mut _ as *mut c_void,
        &mut w as *mut _ as *mut c_void,
        &mut kh as *mut _ as *mut c_void,
        &mut kw as *mut _ as *mut c_void,
        &mut ph as *mut _ as *mut c_void,
        &mut pw as *mut _ as *mut c_void,
        &mut st as *mut _ as *mut c_void,
        &mut oh_i as *mut _ as *mut c_void,
        &mut ow_i as *mut _ as *mut c_void,
        &mut refl as *mut _ as *mut c_void,
    ];
    let t_im2col = std::time::Instant::now();
    c.launch_kernel(c.k_im2col, grid_for(k * n, BLOCK), BLOCK, &mut args)?;
    if flag("LAMA_PROFILE_SUB") {
        c.sync()?;
        SUB_STATS.with(|s| {
            let mut m = s.borrow_mut();
            let e = m.entry("im2col").or_insert((0usize, 0.0f32, 0usize, 0usize));
            e.0 += 1;
            e.1 += t_im2col.elapsed().as_secs_f32();
            // `cout` is the output channel count: the GEMM's M dimension is n,
            // its N dimension is cout and its K dimension is k.
            e.2 += n * cv.cout;
            e.3 += n * cv.cout * k;
        });
    }
    let t_gemm = std::time::Instant::now();
    let alpha = 1f32;
    let beta = 0f32;
    // SAFETY: every pointer below is a device buffer of the stated extent.
    let r = unsafe {
        (c.blas_sgemm)(
            c.blas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32,
            cv.cout as i32,
            k as i32,
            &alpha,
            col.ptr,
            n as i32,
            cv.w.ptr,
            k as i32,
            &beta,
            out.ptr,
            n as i32,
        )
    };
    if r != 0 {
        return Err(format!("cublasSgemm failed: {r}"));
    }
    if flag("LAMA_PROFILE_SUB") {
        c.sync()?;
        SUB_STATS.with(|s| {
            let mut m = s.borrow_mut();
            let e = m.entry("sgemm").or_insert((0usize, 0.0f32, 0usize, 0usize));
            let t = t_gemm.elapsed().as_secs_f32();
            e.0 += 1;
            e.1 += t;
            e.2 += n * cv.cout;
            e.3 += 2 * n * cv.cout * k;
        });
    }

    if let Some(bias) = &cv.bias {
        bias_plane(c, &out, bias, cv.cout, n)?;
    }

    Ok(DActs { buf: out, c: cv.cout, h: oh, w: ow })
}

/// Direct 7x7 convolution without im2col, one thread per output pixel.
fn conv_7x7_direct(cv: &DConv, src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let w_t = cv.w_t.as_ref().ok_or("conv_7x7_direct without a transposed weight")?;
    let (oh, ow) = (src.h - 6, src.w - 6);
    let out = c.alloc(cv.cout * oh * ow)?;
    let mut sp = src.buf.ptr;
    let mut wp = w_t.ptr;
    let mut bp = match &cv.bias {
        Some(b) => b.ptr,
        None => ptr::null_mut(),
    };
    let mut op = out.ptr;
    let (mut cin, mut cout) = (cv.cin as i32, cv.cout as i32);
    // The padded input keeps its own extent: 518x518 in, 512x512 out.
    let (mut h_in, mut w_in) = (src.h as i32, src.w as i32);
    let (mut h_out, mut w_out) = (oh as i32, ow as i32);
    let mut args: Vec<*mut c_void> = vec![
        &mut sp as *mut _ as *mut c_void,
        &mut wp as *mut _ as *mut c_void,
        &mut bp as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut cin as *mut _ as *mut c_void,
        &mut cout as *mut _ as *mut c_void,
        &mut h_in as *mut _ as *mut c_void,
        &mut w_in as *mut _ as *mut c_void,
        &mut h_out as *mut _ as *mut c_void,
        &mut w_out as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_conv7x7, grid_for(oh * ow, BLOCK), BLOCK, &mut args)?;
    Ok(DActs { buf: out, c: cv.cout, h: oh, w: ow })
}

/// Add a per-channel bias to a `[cout][n]` plane in place, on the device.
fn bias_plane(c: &Cuda, out: &Dev, bias: &Dev, cout: usize, n: usize) -> Result<(), String> {
    let mut op = out.ptr;
    let mut bp = bias.ptr;
    let mut cout_i = cout as i32;
    let mut n_i = n as i64;
    let mut args: Vec<*mut c_void> = vec![
        &mut op as *mut _ as *mut c_void,
        &mut bp as *mut _ as *mut c_void,
        &mut cout_i as *mut _ as *mut c_void,
        &mut n_i as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_bias_plane, grid_for(cout * n, BLOCK), BLOCK, &mut args)
}

/// The 1x1 case is a pure channel matmul over `[cin][n]` planes, so it runs as a
/// single SGEMM with no im2col at all.
fn conv_1x1(cv: &DConv, src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let n = src.h * src.w;
    let out = c.alloc(cv.cout * n)?;
    let t_gemm = std::time::Instant::now();
    let alpha = 1f32;
    let beta = 0f32;
    // SAFETY: device buffers of the stated extents.
    let r = unsafe {
        (c.blas_sgemm)(
            c.blas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n as i32,
            cv.cout as i32,
            cv.cin as i32,
            &alpha,
            src.buf.ptr,
            n as i32,
            cv.w.ptr,
            cv.cin as i32,
            &beta,
            out.ptr,
            n as i32,
        )
    };
    if r != 0 {
        return Err(format!("cublasSgemm (1x1) failed: {r}"));
    }
    if flag("LAMA_PROFILE_SUB") {
        c.sync()?;
        SUB_STATS.with(|s| {
            let mut m = s.borrow_mut();
            let e = m.entry("sgemm1x1").or_insert((0usize, 0.0f32, 0usize, 0usize));
            e.0 += 1;
            e.1 += t_gemm.elapsed().as_secs_f32();
            e.2 += n * cv.cout;
            e.3 += 2 * n * cv.cout * cv.cin;
        });
    }
    if let Some(bias) = &cv.bias {
        bias_plane(c, &out, bias, cv.cout, n)?;
    }
    Ok(DActs { buf: out, c: cv.cout, h: src.h, w: src.w })
}

/// BatchNorm + ReLU in place, using the folded scale/shift.
fn bn_relu_inplace(x: &mut DActs, bn: &DBn) -> Result<(), String> {
    let c = ctx()?;
    let plane = x.plane() as i64;
    let mut xp = x.buf.ptr;
    let mut sp = bn.scale.ptr;
    let mut tp = bn.shift.ptr;
    let mut pl = plane;
    let mut ch = bn.channels as i32;
    let mut args: Vec<*mut c_void> = vec![
        &mut xp as *mut _ as *mut c_void,
        &mut sp as *mut _ as *mut c_void,
        &mut tp as *mut _ as *mut c_void,
        &mut pl as *mut _ as *mut c_void,
        &mut ch as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_bn_relu, grid_for((x.plane() * bn.channels) as usize, BLOCK), BLOCK, &mut args)
}

/// BatchNorm + ReLU + residual add in one pass: `x = relu(scale*x + shift) + skip`.
///
/// Folding the residual add into the BN+ReLU removes one full pass over the
/// activation and one kernel launch per resblock, and keeps the block hot in L2
/// between the two.
fn bn_relu_add_inplace(x: &mut DActs, bn: &DBn, skip: Option<&DActs>) -> Result<(), String> {
    let c = ctx()?;
    let plane = x.plane() as i64;
    let mut xp = x.buf.ptr;
    let mut kp = match skip {
        Some(s) => s.buf.ptr,
        None => ptr::null_mut(),
    };
    let mut sp = bn.scale.ptr;
    let mut tp = bn.shift.ptr;
    let mut pl = plane;
    let mut ch = bn.channels as i32;
    let mut args: Vec<*mut c_void> = vec![
        &mut xp as *mut _ as *mut c_void,
        &mut kp as *mut _ as *mut c_void,
        &mut sp as *mut _ as *mut c_void,
        &mut tp as *mut _ as *mut c_void,
        &mut pl as *mut _ as *mut c_void,
        &mut ch as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_bn_relu_add, grid_for((x.plane() * bn.channels) as usize, BLOCK), BLOCK, &mut args)
}

fn sigmoid_inplace(x: &mut DActs) -> Result<(), String> {
    let c = ctx()?;
    let mut xp = x.buf.ptr;
    let mut n = x.buf.len as i64;
    let mut args: Vec<*mut c_void> = vec![
        &mut xp as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_sigmoid, grid_for(x.buf.len, BLOCK), BLOCK, &mut args)
}

fn add_inplace(a: &mut DActs, b: &DActs) -> Result<(), String> {
    let c = ctx()?;
    let mut ap = a.buf.ptr;
    let mut bp = b.buf.ptr;
    let mut n = a.buf.len as i64;
    let mut args: Vec<*mut c_void> = vec![
        &mut ap as *mut _ as *mut c_void,
        &mut bp as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_add_inplace, grid_for(a.buf.len, BLOCK), BLOCK, &mut args)
}

/// ReflectionPad2d.
fn reflect_pad(src: &DActs, pad: usize) -> Result<DActs, String> {
    let c = ctx()?;
    let (oh, ow) = (src.h + 2 * pad, src.w + 2 * pad);
    let out = c.alloc(src.c * oh * ow)?;
    let mut sp = src.buf.ptr;
    let mut op = out.ptr;
    let mut ch = src.c as i32;
    let mut h = src.h as i32;
    let mut w = src.w as i32;
    let mut p = pad as i32;
    let mut args: Vec<*mut c_void> = vec![
        &mut sp as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut ch as *mut _ as *mut c_void,
        &mut h as *mut _ as *mut c_void,
        &mut w as *mut _ as *mut c_void,
        &mut p as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_reflect_pad, grid_for(src.c * oh * ow, BLOCK), BLOCK, &mut args)?;
    Ok(DActs { buf: out, c: src.c, h: oh, w: ow })
}

fn avgpool2x2(src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let (oh, ow) = (src.h / 2, src.w / 2);
    let out = c.alloc(src.c * oh * ow)?;
    let mut sp = src.buf.ptr;
    let mut op = out.ptr;
    let mut ch = src.c as i32;
    let mut h = src.h as i32;
    let mut w = src.w as i32;
    let mut args: Vec<*mut c_void> = vec![
        &mut sp as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut ch as *mut _ as *mut c_void,
        &mut h as *mut _ as *mut c_void,
        &mut w as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_avgpool2x2, grid_for(src.c * oh * ow, BLOCK), BLOCK, &mut args)?;
    Ok(DActs { buf: out, c: src.c, h: oh, w: ow })
}

/// Concatenate two activation blocks along channels with a device kernel.
///
/// `cudaMemcpy` is a blocking API call and therefore serialises the stream; a
/// copy kernel keeps everything ordered on the stream and costs one launch.
fn concat_acts(a: &DActs, g: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let out = c.alloc(a.buf.len + g.buf.len)?;
    let mut off = 0usize;
    for src in [a, g] {
        let mut sp = src.buf.ptr;
        let mut op = unsafe { (out.ptr as *mut u8).add(off) as *mut c_void };
        let mut n = src.buf.len as i64;
        let mut args: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        c.launch_kernel(c.k_copy_plane, grid_for(src.buf.len, BLOCK), BLOCK, &mut args)?;
        off += src.buf.len * 4;
    }
    Ok(DActs { buf: out, c: a.c + g.c, h: a.h, w: a.w })
}

/// Transposed convolution through the four phase GEMMs.
///
/// For stride 2, pad 1 and a 3x3 kernel the output phase `(oy % 2, ox % 2)`
/// selects a fixed tap set (1, 2, 2 and 4 taps), so each phase is a dense
/// `[cout][cin*taps] x [cin*taps][n]` SGEMM over a gathered patch matrix - the
/// same im2col + cuBLAS shape as the forward convolutions, and much friendlier
/// to the memory system than the per-output gather kernel.
fn conv_transpose_phases(cv: &DConvT, src: &DActs, phases: &[Dev]) -> Result<DActs, String> {
    let c = ctx()?;
    let (oh, ow) = (src.h * 2, src.w * 2);
    let out = c.alloc(cv.cout * oh * ow)?;
    // No cudaMemset: each phase writes a disjoint strided subset of the output
    // pixels and `k_convt_put` writes (rather than accumulates) for the first
    // phase, so the buffer needs no pre-clearing - and cudaMemset is a blocking
    // API call that would serialise the stream anyway.
    // Per-phase output geometry: `ph_n` pixels per channel.
    for ph in 0..4 {
        let py = ph / 2;
        let px = ph % 2;
        let taps = CONVT_TAP_COUNT[ph];
        let k = cv.cin * taps;
        let n = ((oh + 1) / 2) * ((ow + 1) / 2);
        let col = c.alloc(k * n)?;
        let phase_out = c.alloc(cv.cout * n)?;

        let mut sp = src.buf.ptr;
        let mut cp = col.ptr;
        let (mut cin, mut h, mut w) = (cv.cin as i32, src.h as i32, src.w as i32);
        let (mut pyi, mut pxi, mut taps_i) = (py as i32, px as i32, taps as i32);
        let (mut n_i, mut oh_i, mut ow_i) = (n as i32, oh as i32, ow as i32);
        let mut args: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut cin as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut w as *mut _ as *mut c_void,
            &mut pyi as *mut _ as *mut c_void,
            &mut pxi as *mut _ as *mut c_void,
            &mut taps_i as *mut _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut oh_i as *mut _ as *mut c_void,
            &mut ow_i as *mut _ as *mut c_void,
        ];
        c.launch_kernel(c.k_convt_col, grid_for(k * n, BLOCK), BLOCK, &mut args)?;

        let alpha = 1f32;
        let beta = 0f32;
        // SAFETY: device buffers of the stated extents.
        let r = unsafe {
            (c.blas_sgemm)(
                c.blas,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32,
                cv.cout as i32,
                k as i32,
                &alpha,
                col.ptr,
                n as i32,
                phases[ph].ptr,
                k as i32,
                &beta,
                phase_out.ptr,
                n as i32,
            )
        };
        if r != 0 {
            return Err(format!("cublasSgemm (convT phase {ph}) failed: {r}"));
        }

        let mut op = phase_out.ptr;
        let mut dp = out.ptr;
        let mut bp = cv.bias.ptr;
        let (mut cout_i, mut n2, mut pyi2, mut pxi2) =
            (cv.cout as i32, n as i32, py as i32, px as i32);
        let (mut oh2, mut ow2) = (oh as i32, ow as i32);
        // `k_convt_put` always assigns (never accumulates) and every phase owns a
        // disjoint set of output pixels - phase (py, px) writes only the pixels
        // whose coordinates have those parities - so no output buffer clearing is
        // needed.  The flag only controls whether the bias is folded in.
        let mut put_mode = 1i32;
        let mut args: Vec<*mut c_void> = vec![
            &mut op as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut cout_i as *mut _ as *mut c_void,
            &mut n2 as *mut _ as *mut c_void,
            &mut pyi2 as *mut _ as *mut c_void,
            &mut pxi2 as *mut _ as *mut c_void,
            &mut oh2 as *mut _ as *mut c_void,
            &mut ow2 as *mut _ as *mut c_void,
            &mut put_mode as *mut _ as *mut c_void,
        ];
        c.launch_kernel(c.k_convt_put, grid_for(cv.cout * n, BLOCK), BLOCK, &mut args)?;
    }
    Ok(DActs { buf: out, c: cv.cout, h: oh, w: ow })
}

fn conv_transpose(cv: &DConvT, src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let (oh, ow) = (src.h * 2, src.w * 2);
    let out = c.alloc(cv.cout * oh * ow)?;
    let mut sp = src.buf.ptr;
    let mut wp = cv.w.ptr;
    let mut bp = cv.bias.ptr;
    let mut op = out.ptr;
    let (mut cin, mut cout) = (cv.cin as i32, cv.cout as i32);
    let (mut h, mut w) = (src.h as i32, src.w as i32);
    let (mut oh_i, mut ow_i) = (oh as i32, ow as i32);
    let (mut kh, mut kw) = (3i32, 3i32);
    let (mut st, mut pd) = (2i32, 1i32);
    let mut args: Vec<*mut c_void> = vec![
        &mut sp as *mut _ as *mut c_void,
        &mut wp as *mut _ as *mut c_void,
        &mut bp as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut cin as *mut _ as *mut c_void,
        &mut cout as *mut _ as *mut c_void,
        &mut h as *mut _ as *mut c_void,
        &mut w as *mut _ as *mut c_void,
        &mut oh_i as *mut _ as *mut c_void,
        &mut ow_i as *mut _ as *mut c_void,
        &mut kh as *mut _ as *mut c_void,
        &mut kw as *mut _ as *mut c_void,
        &mut st as *mut _ as *mut c_void,
        &mut pd as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_conv_transpose, grid_for(cv.cout * oh * ow, BLOCK), BLOCK, &mut args)?;
    Ok(DActs { buf: out, c: cv.cout, h: oh, w: ow })
}

// ---------------------------------------------------------------------------
// Step engine
// ---------------------------------------------------------------------------

/// One FFC_BN_ACT evaluation, mirroring `cpu::run_ffc` exactly.
///
/// `inp` arrives as `[in_local + in_global][h][w]` with the local half first;
/// the local half feeds `convl2l`/`convl2g` and the global half feeds
/// `convg2l`/`spectral`.  The two outputs are `(local, global)`.
fn run_dffc(
    w: &DFfc,
    inp: &DActs,
    glob: Option<&DActs>,
    skip_local: Option<&DActs>,
    skip_global: Option<&DActs>,
) -> Result<(DActs, Option<DActs>), String> {
    let local_view;
    let local = if w.in_global == 0 {
        inp
    } else {
        local_view = inp.view(0, w.in_local);
        &local_view
    };

    let out_local = if w.out_local > 0 {
        let mut acc = match &w.l2l {
            Some(cv) => conv_forward(cv, local, w.pad, w.stride, w.reflect)?,
            None => DActs::new(0, inp.h, inp.w)?,
        };
        if let Some(cv) = &w.g2l {
            let g = glob.ok_or("convg2l present without a global branch")?;
            let o = conv_forward(cv, g, w.pad, w.stride, w.reflect)?;
            if acc.c == 0 {
                acc = o;
            } else {
                add_inplace(&mut acc, &o)?;
            }
        }
        if let Some(bn) = &w.bn_l {
            // For the closing convolution of a resblock the residual add folds
            // into this BN+ReLU, saving a pass over the activation and a
            // launch; other FFC evaluations pass no skip.
            bn_relu_add_inplace(&mut acc, bn, skip_local)?;
        }
        Some(acc)
    } else {
        None
    };

    let out_global = if w.out_global > 0 {
        let mut acc = match &w.l2g {
            Some(cv) => conv_forward(cv, local, w.pad, w.stride, w.reflect)?,
            None => DActs::new(0, inp.h, inp.w)?,
        };
        if let Some(sp) = &w.spectral {
            let g = glob.ok_or("convg2g present without a global branch")?;
            // The spectral branch works on the global half only.
            let g_view = g.view(g.c - w.in_global, w.in_global);
            let o = spectral_forward(sp, &g_view)?;
            if acc.c == 0 {
                acc = o;
            } else {
                add_inplace(&mut acc, &o)?;
            }
        }
        if let Some(bn) = &w.bn_g {
            bn_relu_add_inplace(&mut acc, bn, skip_global)?;
        }
        Some(acc)
    } else {
        None
    };

    // An absent local branch still has to carry the plane geometry.
    let local_out = match out_local {
        Some(l) => l,
        None => DActs::new(0, inp.h, inp.w)?,
    };
    Ok((local_out, out_global))
}

fn copy_acts(src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let out = c.alloc(src.buf.len)?;
    let mut sp = src.buf.ptr;
    let mut op = out.ptr;
    let mut n = src.buf.len as i64;
    let mut args: Vec<*mut c_void> = vec![
        &mut sp as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_copy_plane, grid_for(src.buf.len, BLOCK), BLOCK, &mut args)?;
    Ok(DActs { buf: out, c: src.c, h: src.h, w: src.w })
}

/// The spectral branch, mirroring `cpu::Spectral::forward`.
///
/// The Fourier unit runs on the device through cuFFT.  `cufftExecR2C` produces
/// the half-spectrum `[half][h][w/2+1]` that PyTorch's `rfftn(norm='ortho')`
/// defines, and `cufftExecC2R` inverts it with the same Hermitian conventions
/// `irfft2_ortho` implements by hand (the imaginary parts of the first and last
/// columns are ignored, which is what makes the result real).  Both transforms
/// are scaled by `1/sqrt(h*w)` afterwards to get the ortho normalisation.
///
/// Set `LAMA_FFT_HOST=1` to run the transforms on the CPU instead; the two paths
/// agree to float rounding and the host one is kept for cross-checking.
fn spectral_forward(sp: &DSpectral, g: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let pooled = if sp.stride == 2 { avgpool2x2(g)? } else { copy_acts(g)? };
    let mut feat = conv_forward(&sp.conv1, &pooled, 0, 1, false)?;
    bn_relu_inplace(&mut feat, &sp.bn1)?;

    let (h, w) = (feat.h, feat.w);
    let half = sp.half;
    let hw = w / 2 + 1;
    let scale = 1.0f32 / ((h * w) as f32).sqrt();

    let t_fft = std::time::Instant::now();
    let spec_acts = if flag("LAMA_FFT_HOST") {
        let host = c.download(&feat.buf)?;
        let mut spec = vec![0f32; 2 * half * h * hw];
        for ch in 0..half {
            crate::cpu::rfft2_ortho(&host[ch * h * w..(ch + 1) * h * w], h, w, &mut spec[2 * ch * h * hw..]);
        }
        DActs { buf: c.upload(&spec)?, c: 2 * half, h, w: hw }
    } else {
        // Forward R2C into `[half][h][hw]` complex, then pack to the stacked
        // real/imag layout the 1x1 convolution consumes.
        let complex = c.alloc(2 * half * h * hw)?;
        let plan = fft_plan(c, h, w, half, false)?;
        // SAFETY: buffers match the plan's (h, w, batch) geometry.
        let r = unsafe { (c.cufft_exec_r2c)(plan, feat.buf.ptr, complex.ptr) };
        if r != 0 {
            return Err(format!("cufftExecR2C failed: {r}"));
        }
        let packed = c.alloc(2 * half * h * hw)?;
        let mut cp = complex.ptr;
        let mut pp = packed.ptr;
        let mut hf = half as i32;
        let mut pl = (h * hw) as i32;
        // cuFFT transforms are unnormalised; the forward 1/sqrt(h*w) factor of
        // the ortho pair (the inverse transform applies the other) is folded into
        // this packing pass, which saves a launch and a whole pass over the data
        // per spectral call.  No synchronise is needed: the transform, the pack
        // and everything downstream share one stream.
        let mut sc = scale;
        let mut args: Vec<*mut c_void> = vec![
            &mut cp as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut hf as *mut _ as *mut c_void,
            &mut pl as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        c.launch_kernel(c.k_spec_pack, grid_for(half * h * hw, BLOCK), BLOCK, &mut args)?;
        DActs { buf: packed, c: 2 * half, h, w: hw }
    };

    let mut fu = conv_forward(&sp.fu_conv, &spec_acts, 0, 1, false)?;
    bn_relu_inplace(&mut fu, &sp.fu_bn)?;

    let mut inv_acts = if flag("LAMA_FFT_HOST") {
        let fu_host = c.download(&fu.buf)?;
        let mut inv = vec![0f32; half * h * w];
        for ch in 0..half {
            crate::cpu::irfft2_ortho(&fu_host[2 * ch * h * hw..], h, w, &mut inv[ch * h * w..(ch + 1) * h * w]);
        }
        DActs { buf: c.upload(&inv)?, c: half, h, w }
    } else {
        // Unpack the stacked layout back to interleaved complex, run the inverse
        // C2R into its own real plane, and apply the ortho scale.
        let complex = c.alloc(2 * half * h * hw)?;
        let mut ip = fu.buf.ptr;
        let mut cp = complex.ptr;
        let mut hf = half as i32;
        let mut pl = (h * hw) as i32;
        let mut args: Vec<*mut c_void> = vec![
            &mut ip as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut hf as *mut _ as *mut c_void,
            &mut pl as *mut _ as *mut c_void,
        ];
        c.launch_kernel(c.k_spec_unpack, grid_for(half * h * hw, BLOCK), BLOCK, &mut args)?;
        let plan = fft_plan(c, h, w, half, true)?;
        let real = c.alloc(half * h * w)?;
        // SAFETY: C2R reads the packed half-spectrum and writes `half*h*w` reals.
        let r = unsafe { (c.cufft_exec_c2r)(plan, complex.ptr, real.ptr) };
        if r != 0 {
            return Err(format!("cufftExecC2R failed: {r}"));
        }
        scale_inplace(c, real.ptr, half * h * w, scale)?;
        DActs { buf: real, c: half, h, w }
    };
    fft_stat_add(t_fft.elapsed().as_secs_f32());
    // Add the pre-transform feature (the FourierUnit residual) and close with conv2.
    add_inplace(&mut inv_acts, &feat)?;
    conv_forward(&sp.conv2, &inv_acts, 0, 1, false)
}

/// (h, w, batch, inverse) -> plan, cached for the life of the process.
fn fft_plan(c: &Cuda, h: usize, w: usize, batch: usize, inverse: bool) -> Result<*mut c_void, String> {
    let key = (h, w, batch, inverse);
    FFT_PLANS.with(|m| {
        let mut m = m.borrow_mut();
        if let Some(p) = m.get(&key) {
            return Ok(*p);
        }
        // A batched 2-D transform over `batch` planes, tightly packed.
        let mut n = [h as i32, w as i32];
        let hw = (w / 2 + 1) as i32;
        // R2C reads `h*w` reals per plane and writes `h*(w/2+1)` complex; C2R is
        // the reverse.  cuFFT wants each embed to describe that side's own
        // layout, so the real side is {h, w} and the complex side {h, w/2+1}.
        let mut inembed;
        let mut onembed;
        // cuFFT 11 types are CUFFT_R2C = 0x2a = 42 and CUFFT_C2R = 0x2c = 44;
        // the small 1/2 values are the cuFFT 10 enum.
        const CUFFT_C2R: i32 = 0x2c;
        const CUFFT_R2C: i32 = 0x2a;
        let (idist, odist, kind) = if inverse {
            inembed = [h as i32, hw];
            onembed = [h as i32, w as i32];
            ((h as i32 * hw), (h * w) as i32, CUFFT_C2R)
        } else {
            inembed = [h as i32, w as i32];
            onembed = [h as i32, hw];
            ((h * w) as i32, (h as i32 * hw), CUFFT_R2C)
        };
        let mut plan = ptr::null_mut();
        if flag("LAMA_DEBUG_CUFFT") {
            eprintln!(
                "cufftPlanMany(rank=2, n=[{} {}], inembed=[{} {}], istride=1, idist={idist}, \
                 onembed=[{} {}], ostride=1, odist={odist}, kind={kind}, batch={batch})",
                n[0], n[1], inembed[0], inembed[1], onembed[0], onembed[1]
            );
        }
        // SAFETY: the geometry arrays are valid for the duration of the call.
        let r = unsafe {
            (c.cufft_plan_many)(
                &mut plan,
                2,
                n.as_mut_ptr(),
                inembed.as_mut_ptr(),
                1,
                idist,
                onembed.as_mut_ptr(),
                1,
                odist,
                kind,
                batch as i32,
            )
        };
        if r != 0 {
            return Err(format!(
                "cufftPlanMany failed: {r} (h={h} w={w} batch={batch} inverse={inverse} \
                 n=[{} {}] inembed=[{} {}] idist={idist} onembed=[{} {}] odist={odist} kind={kind})",
                n[0], n[1], inembed[0], inembed[1], onembed[0], onembed[1]
            ));
        }
        // SAFETY: the stream belongs to this context.
        unsafe { (c.cufft_set_stream)(plan, c.stream) };
        m.insert(key, plan);
        Ok(plan)
    })
}

/// In-place scaling of `n` device floats, used for the unnormalised cuFFT
/// transforms (which need the `1/sqrt(h*w)` ortho factor).
fn scale_inplace(c: &Cuda, p: *mut c_void, n: usize, s: f32) -> Result<(), String> {
    let mut xp = p;
    let mut nn = n as i64;
    let mut ss = s;
    let mut args: Vec<*mut c_void> = vec![
        &mut xp as *mut _ as *mut c_void,
        &mut nn as *mut _ as *mut c_void,
        &mut ss as *mut _ as *mut c_void,
    ];
    c.launch_kernel(c.k_scale, grid_for(n, BLOCK), BLOCK, &mut args)
}

thread_local! {
    static FFT_PLANS: std::cell::RefCell<std::collections::HashMap<(usize, usize, usize, bool), *mut c_void>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The full forward pass, mirroring `cpu::Net::forward`.
pub fn run(
    store: &WeightStore,
    model: &Model,
    input: &[f32],
    h: usize,
    w: usize,
) -> Result<Vec<f32>, String> {
    let t_run = std::time::Instant::now();
    let t_ctx = std::time::Instant::now();
    let c = ctx()?;
    if flag("LAMA_PROFILE") {
        eprintln!("gpu ctx: {:.3}s", t_ctx.elapsed().as_secs_f32());
    }
    // Loading the weights onto the device is a one-off per process but it is a
    // cudaMemcpy H2D per tensor, and it happens inside this function, so the
    // caller's "inference" timing includes it while the per-step profile does
    // not.  Time it separately so the two can be told apart.
    let t_load = std::time::Instant::now();
    let steps = load_steps(store, model)?;
    if flag("LAMA_PROFILE") {
        eprintln!("gpu load_steps: {:.3}s", t_load.elapsed().as_secs_f32());
    }
    let t_up = std::time::Instant::now();
    let mut acts = DActs { buf: c.upload(input)?, c: 4, h, w };
    if flag("LAMA_PROFILE") {
        c.sync()?;
        eprintln!("gpu input upload: {:.3}s", t_up.elapsed().as_secs_f32());
    }
    let t_loop = std::time::Instant::now();
    let mut global: Option<DActs> = None;
    // `LAMA_DUMP_STEP=<n>` writes the same activation layout as the CPU path so
    // the two engines can be diffed step by step.
    let dump_after: Option<usize> = std::env::var("LAMA_DUMP_STEP").ok().and_then(|v| v.parse().ok());
    let mut step_no = 0usize;
    let profile = flag("LAMA_PROFILE");

    for step in &steps {
        let t_step = std::time::Instant::now();
        match step {
            DStep::ReflectPad(p) => {
                // Both branches carry spatial padding, exactly as the CPU path.
                acts = op_scope("pad", || reflect_pad(&acts, *p))?;
                global = match global {
                    Some(g) => Some(op_scope("pad", || reflect_pad(&g, *p))?),
                    None => None,
                };
            }
            DStep::Ffc(b) => {
                let (l, g) = op_scope("ffc", || run_dffc(b, &acts, global.as_ref(), None, None))?;
                acts = l;
                global = g;
            }
            DStep::Res(rb1, rb2) => {
                // conv1 then conv2, then the residual add on both branches,
                // exactly as `cpu.rs` does it.  Neither `run_dffc` writes through
                // its input, so the pre-resblock activation is still intact and
                // can serve as the skip directly - no copy and no separate add
                // pass: the add folds into conv2's closing BN+ReLU.
                let (l1, g1) = op_scope("res_a", || run_dffc(rb1, &acts, global.as_ref(), None, None))?;
                let (l2, g2) = op_scope("res_b", || {
                    run_dffc(rb2, &l1, g1.as_ref(), Some(&acts), global.as_ref())
                })?;
                acts = l2;
                global = g2;
            }
            DStep::Concat => {
                // local | global stacked along channels, copied on the device so
                // the stream is never blocked by a host-side cudaMemcpy.
                let g = global.take().ok_or("Concat without a global branch")?;
                acts = concat_acts(&acts, &g)?;
            }
            DStep::Upsample(convt, bn) => {
                // The four phase GEMMs are the fast path; `LAMA_CONVT_GATHER=1`
                // selects the reference gather kernel for cross-checking.
                let mut up = op_scope("upsample", || match &convt.phases {
                    Some(ph) if !flag("LAMA_CONVT_GATHER") => conv_transpose_phases(convt, &acts, ph),
                    _ => conv_transpose(convt, &acts),
                })?;
                bn_relu_inplace(&mut up, bn)?;
                acts = up;
                // The upsample collapses the two branches into one.
                global = None;
            }
            DStep::OutConv(cv) => {
                // The plan emits ReflectPad(3) immediately before this step, so
                // the 7x7 convolution itself adds no padding (pad 0).
                let mut out = op_scope("outconv", || conv_forward(cv, &acts, 0, 1, false))?;
                sigmoid_inplace(&mut out)?;
                c.sync()?;
                if profile {
                    eprintln!("gpu step loop: {:.3}s", t_loop.elapsed().as_secs_f32());
                }
                let t_dl = std::time::Instant::now();
                let r = c.download(&out.buf);
                if profile {
                    eprintln!(
                        "gpu output download: {:.3}s ({} floats)",
                        t_dl.elapsed().as_secs_f32(),
                        out.buf.len
                    );
                    eprintln!("gpu run total: {:.3}s", t_run.elapsed().as_secs_f32());
                }
                return r;
            }
        }
        if dump_after == Some(step_no) {
            c.sync()?;
            let mut all = c.download(&acts.buf)?;
            if let Some(g) = &global {
                all.extend_from_slice(&c.download(&g.buf)?);
            }
            let mut bytes = Vec::with_capacity(all.len() * 4);
            for v in &all {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write(format!("/tmp/gpu_step_{step_no}.f32"), &bytes);
            eprintln!(
                "gpu dumped step {step_no}: local {}x{}x{} global {:?}",
                acts.c, acts.h, acts.w, global.as_ref().map(|g| (g.c, g.h, g.w))
            );
        }
        if flag("LAMA_PROFILE_SYNC") {
            // The per-step figure above only measures *submission*: every kernel
            // launch is asynchronous, so the timings say how long the host took
            // to queue the work, not how long the device took to run it.  With a
            // sync at the end of the step the figure becomes submit + device,
            // and the difference between the two is the device time.
            c.sync()?;
        }
        if profile {
            let (nfft, t_fft) = FFT_STATS.with(|s| {
                let s = s.borrow();
                (s.0, s.1)
            });
            eprintln!(
                "gpu step {step_no:2}: {:>6.3}s local {}x{}x{} global {:?}  [fft calls {} {:.3}s]",
                t_step.elapsed().as_secs_f32(),
                acts.c,
                acts.h,
                acts.w,
                global.as_ref().map(|g| (g.c, g.h, g.w)),
                nfft,
                t_fft,
            );
        }
        step_no += 1;
    }
    c.sync()?;
    Ok(c.download(&acts.buf)?)
}

// (spectral host-FFT calls, seconds spent in the device<->host round trip).
thread_local! {
    static FFT_STATS: std::cell::RefCell<(usize, f32)> = const { std::cell::RefCell::new((0, 0.0)) };
}

// (cudaMalloc calls, seconds).
thread_local! {
    static ALLOC_STATS: std::cell::RefCell<(usize, f32)> = const { std::cell::RefCell::new((0, 0.0)) };
}

/// Time one step kind when `LAMA_PROFILE_OPS` is set.  Each call synchronises,
/// so the totals are attributable kernel time at the price of removing all
/// overlap, which the per-step numbers alone are not.
fn op_scope<T>(kind: &'static str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    if !flag("LAMA_PROFILE_OPS") {
        return f();
    }
    let c = match Cuda::get() {
        Some(c) => c,
        None => return f(),
    };
    c.sync()?;
    let n0 = LAUNCHES.with(|c| c.get());
    let t = std::time::Instant::now();
    let out = f()?;
    c.sync()?;
    let n1 = LAUNCHES.with(|c| c.get());
    OP_STATS.with(|s| {
        let mut m = s.borrow_mut();
        let e = m.entry(kind).or_insert((0usize, 0.0f32, 0usize));
        e.0 += 1;
        e.1 += t.elapsed().as_secs_f32();
        e.2 += n1 - n0;
    });
    Ok(out)
}

// (sub-op) -> (calls, seconds, output elements, FLOPs) for the convolution
// internals, so a profiled pass says whether im2col or the GEMM dominates.
thread_local! {
    static SUB_STATS: std::cell::RefCell<std::collections::HashMap<&'static str, (usize, f32, usize, usize)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Print the convolution sub-step breakdown when `LAMA_PROFILE_SUB` is set.
pub fn print_sub_stats() {
    if !flag("LAMA_PROFILE_SUB") {
        return;
    }
    let mut v: Vec<(&str, (usize, f32, usize, usize))> =
        SUB_STATS.with(|s| s.borrow().iter().map(|(k, x)| (*k, *x)).collect());
    v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    for (k, (n, t, elems, flops)) in &v {
        eprintln!(
            "sub {k:9} {n:4} calls {:>7.3}s  ({} us avg)  {:.1} MFLOP  {:.0} GFLOP/s  {:.0} MB moved",
            t,
            (*t * 1e6 / *n as f32) as i64,
            *flops as f64 / 1e6,
            *flops as f64 / 1e9 / (*t as f64).max(1e-9),
            elems * 4 / 1_000_000
        );
    }
}

// (op kind) -> (calls, seconds, launches).
thread_local! {
    static OP_STATS: std::cell::RefCell<std::collections::HashMap<&'static str, (usize, f32, usize)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Print the per-op-kind counters when `LAMA_PROFILE_OPS` is set.
pub fn print_op_stats() {
    if !flag("LAMA_PROFILE_OPS") {
        return;
    }
    let mut v: Vec<(&str, (usize, f32, usize))> =
        OP_STATS.with(|s| s.borrow().iter().map(|(k, x)| (*k, *x)).collect());
    v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    let total: f32 = v.iter().map(|(_, x)| x.1).sum();
    for (k, (n, t, l)) in &v {
        eprintln!(
            "op {k:14} {n:5} calls {:>7.3}s  ({} us avg, {} launches/call)",
            t,
            (*t * 1e6 / *n as f32) as i64,
            *l / *n
        );
    }
    eprintln!("op total {total:.3}s");
}

/// Print the allocation and spectral counters when `LAMA_PROFILE` is set.
pub fn print_stats() {
    if !flag("LAMA_PROFILE") {
        return;
    }
    let (na, ta) = ALLOC_STATS.with(|s| {
        let s = s.borrow();
        (s.0, s.1)
    });
    let (nf, tf) = FFT_STATS.with(|s| {
        let s = s.borrow();
        (s.0, s.1)
    });
    let (hits, misses, cached) = POOL.with(|s| {
        let s = s.borrow();
        (s.hits, s.misses, s.bytes)
    });
    eprintln!(
        "gpu totals: {na} cudaMalloc in {ta:.3}s; {nf} spectral FFT round trips in {tf:.3}s; \
         pool {hits} hits / {misses} misses, {} MB cached",
        cached >> 20
    );
}

fn fft_stat_add(secs: f32) {
    FFT_STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.0 += 1;
        s.1 += secs;
    });
}
