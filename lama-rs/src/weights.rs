//! Weight index + flat FP32 blob loader.
//!
//! The GPU path mmaps the blob and uploads slices of the mapping directly to
//! device buffers.  The CPU path reads it once and repacks tensors into the
//! layouts the kernels want (e.g. `[k][k][cin][cout]` for the direct
//! convolutions).

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// One tensor in the blob: contiguous row-major FP32, 64-byte aligned.
#[allow(dead_code)]  // `name` is kept for diagnostics/error messages.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct WeightIndex {
    pub tensors: HashMap<String, TensorInfo>,
    pub order: Vec<String>,
}

impl WeightIndex {
    /// Parse the JSON written by `export_weights.py` with a hand-rolled reader,
    /// so the engine needs no serde: the document is a flat array of
    /// fixed-key objects.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut tensors = HashMap::new();
        let mut order = Vec::new();
        for obj in text.split('{').skip(1) {
            let Some(end) = obj.find('}') else { continue };
            let obj = &obj[..end];
            let Some(name) = json_str(obj, "name") else { continue };
            let Some(offset) = json_usize(obj, "offset") else { continue };
            let Some(nbytes) = json_usize(obj, "nbytes") else { continue };
            let shape = json_usize_list(obj, "shape")?;
            order.push(name.clone());
            tensors.insert(name.clone(), TensorInfo { name, shape, offset, nbytes });
        }
        if tensors.is_empty() {
            return Err(format!("no tensors found in {}", path.display()));
        }
        Ok(Self { tensors, order })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor in weight blob: {name}"))
    }
}

fn json_str(obj: &str, key: &str) -> Option<String> {
    let at = obj.find(&format!("\"{key}\""))?;
    let rest = &obj[at + key.len() + 2..];
    let start = rest.find('"')? + 1;
    let end = rest[start..].find('"')? + start;
    Some(rest[start..end].to_string())
}

fn json_usize(obj: &str, key: &str) -> Option<usize> {
    let at = obj.find(&format!("\"{key}\""))?;
    let rest = &obj[at + key.len() + 2..];
    let start = rest.find(':')? + 1;
    let digits: String = rest[start..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

fn json_usize_list(obj: &str, key: &str) -> Result<Vec<usize>, String> {
    let at = obj
        .find(&format!("\"{key}\""))
        .ok_or_else(|| format!("missing {key}"))?;
    let rest = &obj[at + key.len() + 2..];
    let open = rest.find('[').ok_or("missing [")?;
    let close = rest.find(']').ok_or("missing ]")?;
    let body = &rest[open + 1..close];
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(|s| s.trim().parse::<usize>().map_err(|e| e.to_string()))
        .collect()
}

/// The blob bytes, either mmap'd (GPU path, avoids a 205 MB read) or owned.
pub struct Blob {
    ptr: *const u8,
    len: usize,
    owned: Option<Vec<u8>>,
    _file: Option<File>,
}

impl Blob {
    pub fn map(path: &Path) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        if len == 0 {
            return Err("weight blob is empty".into());
        }
        // SAFETY: read-only MAP_PRIVATE of a regular file; the mapping is
        // owned for the lifetime of `self` and never written through.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                std::os::unix::io::AsRawFd::as_raw_fd(&f),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap failed: {}", std::io::Error::last_os_error()));
        }
        Ok(Blob { ptr: ptr as *const u8, len, owned: None, _file: Some(f) })
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        let mut b = Blob { ptr: std::ptr::null(), len: buf.len(), owned: Some(buf), _file: None };
        b.ptr = b.owned.as_ref().unwrap().as_ptr();
        Ok(b)
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr`/`len` describe a valid readable region for `self`'s life.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for Blob {
    fn drop(&mut self) {
        if self.owned.is_none() && !self.ptr.is_null() {
            // SAFETY: `ptr`/`len` came from `mmap` in `Blob::map`.
            unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
        }
    }
}


/// Weight blob + index, with typed access.  The CPU path copies tensors out as
/// `Vec<f32>`; the GPU path hands raw blob pointers straight to `cudaMemcpy`.
pub struct WeightStore {
    pub index: WeightIndex,
    pub blob: Blob,
}

impl WeightStore {
    pub fn open(bin: &std::path::Path, json: &std::path::Path, map: bool) -> Result<Self, String> {
        let index = WeightIndex::load(json)?;
        let blob = if map { Blob::map(bin)? } else { Blob::read(bin)? };
        let store = WeightStore { index, blob };
        store.validate()?;
        Ok(store)
    }

    /// Every tensor must sit inside the mapping and match its declared size.
    fn validate(&self) -> Result<(), String> {
        for name in &self.index.order {
            let t = self.index.get(name)?;
            if t.offset + t.nbytes > self.blob.len() {
                return Err(format!(
                    "tensor {name} at {}..{} exceeds blob size {}",
                    t.offset,
                    t.offset + t.nbytes,
                    self.blob.len()
                ));
            }
            if t.nbytes != t.numel() * 4 {
                return Err(format!("tensor {name} is not FP32: {t:?}"));
            }
        }
        Ok(())
    }

    /// Look up a tensor's shape/offset without copying its data.
    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.index.get(name)
    }

    pub fn total_bytes(&self) -> usize {
        self.index
            .order
            .iter()
            .filter_map(|n| self.index.get(n).ok())
            .map(|t| t.nbytes)
            .sum()
    }

    /// Copy a tensor out of the blob as host `f32`.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.index.get(name)?;
        let src = &self.blob.as_slice()[t.offset..t.offset + t.nbytes];
        let mut out = vec![0f32; t.numel()];
        // The exporter writes host-order little-endian FP32 and this engine only
        // targets little-endian Linux, so the byte copy is bit-exact.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr() as *const f32,
                out.as_mut_ptr(),
                t.numel(),
            );
        }
        Ok(out)
    }
}
