//! One-shot big-lama inpainting worker.
//!
//! Same contract as `inpaint.py`: read an image and a mask, run the ported
//! big-lama generator, composite the result over the untouched pixels and write
//! a PNG of the same size.  The process is deliberately short lived - it loads
//! the weights, runs one forward pass and exits - so nothing stays resident on
//! the GPU (the caller holds the shared GPU lock).
//!
//! The GPU path is chosen when a CUDA device is usable; otherwise the engine
//! falls back to the CPU implementation with identical arithmetic.

mod cpu;
mod cuda;
mod image;
mod model;
mod weights;

use std::path::PathBuf;
use std::time::Instant;

use image::{Gray8, Rgb8};

fn log(msg: &str) {
    eprintln!("{msg}");
}

struct Args {
    image: PathBuf,
    mask: PathBuf,
    output: PathBuf,
    weights: PathBuf,
    index: PathBuf,
    force_cpu: bool,
    force_gpu: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut image = None;
    let mut mask = None;
    let mut output = None;
    let mut weights = None;
    let mut index = None;
    let mut force_cpu = false;
    let mut force_gpu = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| -> Result<String, String> {
            it.next().ok_or_else(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "--image" => image = Some(PathBuf::from(val("--image")?)),
            "--mask" => mask = Some(PathBuf::from(val("--mask")?)),
            "--output" => output = Some(PathBuf::from(val("--output")?)),
            "--weights" => weights = Some(PathBuf::from(val("--weights")?)),
            "--index" => index = Some(PathBuf::from(val("--index")?)),
            "--cpu" => force_cpu = true,
            "--gpu" => force_gpu = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let image = image.ok_or("--image is required")?;
    let mask = mask.ok_or("--mask is required")?;
    let output = output.ok_or("--output is required")?;
    if image.as_os_str() == "-" && mask.as_os_str() == "-" {
        return Err("--image and --mask cannot both be '-' (stdin has only one stream)".to_string());
    }

    let weights = weights.unwrap_or_else(|| default_weights("big-lama.bin"));
    let index = index.unwrap_or_else(|| {
        let mut p = weights.clone();
        p.set_extension("json");
        p
    });
    Ok(Args {
        image,
        mask,
        output,
        weights,
        index,
        force_cpu,
        force_gpu,
    })
}

const USAGE: &str = "\
usage: lama-inpaint --image IN.png --mask MASK.png --output OUT.png
                    [--weights big-lama.bin] [--index big-lama.json]
                    [--cpu | --gpu]

Runs the big-lama FFC ResNet generator over IN.png, inpainting the pixels the
mask marks (any non-black pixel is a hole), and writes OUT.png at the input
size.  Use - for either input (but not both) to read a PNG from stdin, or for
the output to write PNG data to stdout.  Diagnostics always go to stderr.
GPU is used when available unless --cpu is given.";

/// Where the weight blob lives when `--weights` is not given.
///
/// The weights are two files that travel together (`big-lama.bin` and its
/// `.json` index), so the search starts next to the executable: unpacking the
/// release archive gives a runnable directory.  Then the historical
/// `models/` layout, then up the tree from the executable (so a checkout works
/// from `target/release`), and finally the current directory.
fn default_weights(rel: &str) -> PathBuf {
    const LEGACY: &str = "models";
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join(rel);
            if beside.exists() {
                return beside;
            }
            let legacy = dir.join(LEGACY).join(rel);
            if legacy.exists() {
                return legacy;
            }
            let mut base = dir.to_path_buf();
            for _ in 0..6 {
                let cand = base.join(rel);
                if cand.exists() {
                    return cand;
                }
                let cand = base.join(LEGACY).join(rel);
                if cand.exists() {
                    return cand;
                }
                if !base.pop() {
                    break;
                }
            }
        }
    }
    PathBuf::from(rel)
}

/// Build the 4-channel network input: RGB premultiplied by `1 - mask`, then the
/// mask, in `[c][h][w]` planes.
fn build_input(img: &Rgb8, mask: &Gray8, pad_h: usize, pad_w: usize) -> Vec<f32> {
    let (h, w) = (img.height, img.width);
    let oh = h + pad_h;
    let ow = w + pad_w;
    let plane = oh * ow;
    let mut data = vec![0f32; 4 * plane];
    for y in 0..oh {
        let sy = image::reflect_index(y as isize, h);
        for x in 0..ow {
            let sx = image::reflect_index(x as isize, w);
            let m = mask.data[sy * w + sx] as f32 / 255.0;
            let m = if m > 0.0 { 1.0 } else { 0.0 };
            let o = y * ow + x;
            for c in 0..3 {
                let v = img.data[(sy * w + sx) * 3 + c] as f32 / 255.0;
                data[c * plane + o] = v * (1.0 - m);
            }
            data[3 * plane + o] = m;
        }
    }
    data
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let t_start = Instant::now();

    let src = image::read_rgb(&args.image)?;
    let mask = image::read_gray(&args.mask)?;
    if mask.width != src.width || mask.height != src.height {
        return Err(format!(
            "mask is {}x{} but image is {}x{}",
            mask.width, mask.height, src.width, src.height
        ));
    }
    let (width, height) = (src.width, src.height);
    log(&format!(
        "loaded {width}x{height} image + mask in {:.2}s",
        t_start.elapsed().as_secs_f32()
    ));

    let store = weights::WeightStore::open(&args.weights, &args.index, !args.force_cpu)?;
    log(&format!(
        "weights: {} tensors, {:.1} MB",
        store.index.order.len(),
        store.total_bytes() as f32 / (1024.0 * 1024.0)
    ));

    let plan = model::build();
    log(&format!("plan: {}", model::describe(&plan)));

    // The generator has three stride-2 stages, so it can only process multiples
    // of 8.  Pad, run, crop - the same contract as inpaint.py.
    const PAD_MOD: usize = 8;
    let pad_h = (PAD_MOD - height % PAD_MOD) % PAD_MOD;
    let pad_w = (PAD_MOD - width % PAD_MOD) % PAD_MOD;

    let input = build_input(&src, &mask, pad_h, pad_w);

    let t0 = Instant::now();
    let out = if args.force_cpu {
        cpu::run(&store, &plan, input, height + pad_h, width + pad_w)
            .map_err(|e| format!("cpu inference failed: {e}"))?
    } else {
        match cuda::run(&store, &plan, &input, height + pad_h, width + pad_w) {
            Ok(o) => {
                cuda::print_op_stats();
                cuda::print_sub_stats();
                cuda::print_stats();
                o
            }
            Err(e) if args.force_gpu => {
                return Err(format!("gpu inference failed: {e}\n(--cpu runs the CPU engine instead)"))
            }
            Err(e) => {
                log(&format!("gpu unavailable ({e}); falling back to CPU"));
                cpu::run(&store, &plan, input, height + pad_h, width + pad_w)
                    .map_err(|err| format!("cpu inference failed: {err}"))?
            }
        }
    };
    log(&format!("inference {width}x{height} in {:.2}s", t0.elapsed().as_secs_f32()));

    // Composite: keep the original pixels wherever the mask is zero.  The
    // network output is already constrained to `[0, 1]` by the final sigmoid,
    // but clamp anyway so the cast is well defined.
    let mask_data = if pad_h == 0 && pad_w == 0 {
        mask.data.clone()
    } else {
        let mut m = vec![0u8; (height + pad_h) * (width + pad_w)];
        for y in 0..height + pad_h {
            let sy = image::reflect_index(y as isize, height);
            for x in 0..width + pad_w {
                let sx = image::reflect_index(x as isize, width);
                m[y * (width + pad_w) + x] = mask.data[sy * width + sx];
            }
        }
        m
    };

    let mut out_img = Rgb8::new(width, height);
    let plane = (height + pad_h) * (width + pad_w);
    for y in 0..height {
        for x in 0..width {
            let pi = y * (width + pad_w) + x;
            let hole = mask_data[pi] > 0;
            let o = (y * width + x) * 3;
            if !hole {
                out_img.data[o] = src.data[o];
                out_img.data[o + 1] = src.data[o + 1];
                out_img.data[o + 2] = src.data[o + 2];
                continue;
            }
            for c in 0..3 {
                let v = out[c * plane + pi].clamp(0.0, 1.0) * 255.0;
                out_img.data[o + c] = (v + 0.5) as u8;
            }
        }
    }

    image::write_rgb(&args.output, &out_img)?;
    log(&format!("wrote {}", args.output.display()));
    log(&format!("total {:.2}s", t_start.elapsed().as_secs_f32()));
    Ok(())
}
