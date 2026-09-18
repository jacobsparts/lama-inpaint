# lama-inpaint

Fast, dependency-light inpainting of objects, damage or watermarks in photos.
A single self-contained binary runs the LaMa FFC ResNet generator on a GPU
(or on the CPU), with no Python, no PyTorch and no CUDA toolkit installed.
The GPU path talks to the driver directly: kernels are precompiled into the
executable and the driver, cuBLAS and cuFFT are opened at run time with
`dlopen`. The binary links only `libc`, `libm` and `libgcc_s`.

This is a Linux x86-64 release; the GPU path needs an NVIDIA GPU of compute
capability 6.1 or newer, while the CPU engine runs anywhere Rust does.

![Demo: original, masked with a black hole, and inpainted result](docs/demo-strip.png)

Two independent backends live in `lama-rs/`:

| engine | file | notes |
| --- | --- | --- |
| CUDA | `src/cuda.rs` | cuBLAS SGEMM over im2col patches, cuFFT for the Fourier units, custom kernels for the rest; 0.4 s for 512x512 on a GTX 1080 |
| CPU | `src/cpu.rs` | direct convolution, rayon-parallel; authoritative for semantics - every GPU change is diffed against it (max 1/255, commonly 0) |

## Quick start (release binary)

Download the binary and the two weight files from the
[latest release](https://github.com/jacobsparts/lama-inpaint/releases/latest) into one
directory and run:

```
./lama-inpaint --image photo.png --mask mask.png --output out.png
```

The weights are found next to the executable by default, so no paths are
needed. `--mask` is a PNG in which any non-black pixel marks a hole; the output
is written at the input size.

Use `-` for either input to read that PNG from standard input. The image and
mask cannot both be `-`, because standard input is a single stream. Use
`--output -` to write the result PNG to standard output; progress and
diagnostics remain on standard error.

```sh
cat input.png | ./lama-inpaint --image - --mask mask.png --output - > output.png
```

```
usage: lama-inpaint --image IN.png --mask MASK.png --output OUT.png
                    [--weights big-lama.bin] [--index big-lama.json]
                    [--cpu | --gpu]
                    [--tile | --sections]

Runs the big-lama FFC ResNet generator over IN.png, inpainting the pixels the
mask marks (any non-black pixel is a hole), and writes OUT.png at the input
size.  Use - for either input (but not both) to read a PNG from stdin, or for
the output to write PNG data to stdout.  Diagnostics always go to stderr.
GPU is used when available unless --cpu is given.
Images at or below 512px on the short side are run whole even with --tile.

--tile runs the network on a square window around the mask instead of the whole
image: the window is twice the mask bounding box, at least 512x512, centred on
the mask and clamped inside the image.

Only the masked pixels are taken from the network, exactly as without --tile, so
the output is identical to cropping the window out by hand, running this binary
on it and pasting the hole back.  Best for large images with small holes, and
best with masks of 256x256 or less.

--sections fills a mask larger than 256x256 in discrete 256x256 sections, one
512x512 window per section, from the edge of the hole inward.  Each pass masks
only its own section and writes back only its own section, so the rest of the
hole is context on purpose: passes read the original pixels (the leak this mode
accepts) and the fills of earlier passes.  Sections do not overlap, so every
masked pixel is decided by exactly one pass, and the rim of the hole is filled
before the interior.  The sections are the 256x256 crop the weights were
trained on.
Sections and --tile are mutually exclusive.
```

* `--weights` / `--index` default to `big-lama.bin` and its `.json` index,
  searched next to the executable, then in a `models/` subdirectory, then up
  the tree, then the current directory.
* `--gpu` makes a GPU failure fatal instead of falling back to the CPU engine
  (~1 minute for a 512x512 image), and the error names the GPU it found, the
  architectures the binary supports, and how to reach the CPU path.
* `--tile` runs the network on a square window around the mask instead of the
  whole image. The window is twice the mask's bounding box, at least 512x512,
  centred on the mask and clamped inside the image; images at or below 512px on
  their short side are run whole regardless. Only masked pixels are taken from
  the network, exactly as without `--tile`, so the result is identical to
  cropping the window out by hand and pasting the hole back - see
  [Large images](#large-images---tile).
* `--sections` fills a mask larger than 256x256 in discrete 256x256 pieces, one
  512x512 window per piece, from the rim of the hole inward, so every pass runs
  the network at the scale it was trained for. Every masked pixel is decided by
  exactly one pass; it is a different tradeoff from `--tile`, not a better one -
  see [Large masks](#large-masks---sections).

## Weights

The released weight files are:

| file | size | what it is |
| --- | --- | --- |
| `big-lama.bin` | 204 MB | all 989 tensors concatenated as little-endian FP32, 64-byte aligned |
| `big-lama.json` | 131 KB | per-tensor `name`, `shape`, `offset`, `nbytes` plus the layer order |

They are derived from the official big-lama checkpoint by
`lama-rs/export_weights.py`, which is run once and needs PyTorch:

```
python3 lama-rs/export_weights.py big-lama.pt big-lama.bin
```

At run time the blob is `mmap`ed read-only and uploaded to the GPU in a single
`cudaMemcpy`; the original `.pt` file is not needed afterwards.

## Building from source

Requires a Rust toolchain. **No CUDA toolkit and no nvcc are needed** - the
kernel image is checked in.

```
cd lama-rs
cargo build --release
```

To regenerate the embedded kernel image after editing `src/kernels.cu`
(needs `nvcc`):

```
sh src/build_kernels.sh
```

The script emits SASS for compute 6.1, 7.5, 8.0, 8.6, 8.9 and 9.0, so one
binary covers Pascal through Hopper without JIT. PTX is deliberately not
embedded: nvcc 12.4 emits `.version 8.4` PTX for every target and drivers
refuse PTX newer than the ISA they implement, so it would add megabytes without
adding a single supported GPU. Adding a new architecture means one `-gencode`
line here and one entry in `KERNEL_ARCHS` (`src/cuda.rs`).

## GPU requirements

Any NVIDIA GPU of compute capability 6.1 or newer with a matching driver. The
userspace driver must be the same version as the loaded kernel module: a
mismatch makes `cuInit` fail with error 804. If the driver is installed
somewhere unusual, point the loader at it:

```
LD_LIBRARY_PATH=/path/to/driver ./lama-inpaint ... --gpu
```

If the running GPU has no matching code in the binary, the error says which GPU
it is and which architectures are supported.
If the run would not fit in the device's memory, it is refused before anything
is allocated, with the estimated need, the free and total device memory, and the
flag to use instead: a `--tile` window is sized from the mask, so a large mask
can ask for a window far bigger than the card holds (peak use is about 1.1 GiB
per megapixel of window, so a 3472x3472 window needs ~13 GB). `--sections` caps
every pass at 512x512, and the plain path needs the whole image, so the hint
names whichever of those fits better.

## Performance

512x512, GTX 1080 (8.9 TFLOPS FP32):

| | time |
| --- | --- |
| this binary, GPU | **0.4 s** (0.45 s whole process) |
| this binary, CPU (`--cpu`) | ~58 s |
| PyTorch, warm forward | 0.115 s |
| PyTorch, whole process | 2.4-3.1 s (1.0 s of it model construction) |

For one image per process - the way the service calls it - the standalone
binary is several times faster end to end; PyTorch only wins per forward pass
once a warm model already exists in memory.

## Large images (`--tile`)

The generator was trained on 256x256 and 512x512 crops and its receptive field
is a few hundred pixels, so on a 2048x2048 photo a small hole gets no more
context than it would at 512 - while the whole-frame forward pass costs 12x
more and, because the mask occupies a smaller fraction of the frame, often
produces *worse* fill: the network has to invent texture at a scale it never
saw in training.

`--tile` crops a square window around the mask, runs the network on that, and
pastes the result back:

```
./lama-inpaint --image big.png --mask mask.png --output out.png --tile
```

* The window is centred on the mask's bounding box and sized to **twice the
  longer side** of that box, never less than 512x512, rounded to a multiple of
  8 (the network has three stride-2 stages) and clamped inside the image.
* Only masked pixels are taken from the network; every other pixel is copied
  from the input. Cropping therefore changes nothing about how masked and
  unmasked areas are treated, and the output is byte-for-byte what you would
  get by cropping the window out by hand, running the binary on it, and pasting
  the hole back.
* Images at or below 512px on the short side are run whole even with `--tile`,
  so it is always safe to pass the flag.

A 300x300 mask in a 2048x2048 image gives a 600x600 window:

| | window | time (GTX 1080) |
| --- | --- | --- |
| without `--tile` | 2048x2048 | 4.58 s |
| with `--tile` | 600x600 | **0.67 s** |

Best results come from masks of 256x256 or smaller, which is also the regime the
weights were trained on.

## Large masks (`--sections`)

`--tile` does not help when the hole itself is large - the window is then the
whole image again. A mask well beyond 256x256 is also outside the regime the
weights were trained on, and a single pass over a big hole invents texture at a
scale the network never saw. `--sections` fills such a mask in discrete 256x256
pieces, one 512x512 window per piece, so every forward pass sees a hole the size
the weights were trained for:

```
./lama-inpaint --image big.png --mask bigmask.png --output out.png --sections
```

* Pieces are filled **from the rim of the hole inward**, so the early passes
  have the real image around the whole boundary of their piece and the later
  ones are surrounded by fill.
* **Each pass masks only its own piece.** The rest of the hole is left unmasked
  on purpose: the network sees the original content there, which gives it
  something continuous to work from instead of a hard 512-wide hole.
* Pieces **do not overlap**: they tile the mask's bounding box a whole 256x256
  block at a time, so every masked pixel is written by exactly one pass and
  nothing is re-solved.
* `--sections` and `--tile` are mutually exclusive. Either flag on an image at or
  below 512px on its short side runs the whole image instead, with a note.

Measured on a 1000x667 photo (350x270 mask across a boardwalk), with the four
piece boundaries set by the mask's bounding box:

| | discontinuity across a piece boundary | interior control |
| --- | --- | --- |
| `--sections` | 20.7 / 12.3 | 18.2 / 15.1 |

The two figures are the mean absolute step across the vertical and horizontal
boundaries, over masked pixels either side; the controls are the same measure
across an interior line 64px away. Neither boundary stands out against the fill
around it, so the piece edges are not visible as seams on this image. On a
1024x1024 image with a 399x299 solid ellipse the filling behaves as the
mechanism predicts instead: 79% of the hole's pixels end up more than a quarter
of the way toward the object's colour, because past ~100px inside the hole every
pass is handed pieces that are still original object. (Overlapping pieces, which
this mode used to have, measured 75%: re-solving a boundary with the fill already
around it recovers a little of the original content, at the cost of more passes.
It did not make the boundary cleaner - on the same photo the overlapping build
measured a *larger* step across the piece boundary, 26.4 and 14.8 against the
20.7 and 12.3 above.)

**What this means in practice:** `--sections` is for large masks that are thin or
textured enough that the original content reads as background around each piece.
For a large *solid* object - a sign, a logo, a whole person - it reproduces the
object inside its own hole; `--tile` (or the plain whole-image path) is not
merely better there, it is close to exact. Try `--tile` first: if the window is
much larger than 512x512, `--sections` is the alternative, with the caveat above.

## Verification

Every change is checked against two fixed references:

| comparison | max | mean |
| --- | --- | --- |
| CUDA engine vs PyTorch reference | 1/255 | 0.072606 |
| CUDA engine vs Rust CPU engine | 1/255 | 0.000052 |
| Rust CPU engine, rerun | 0 | 0.000000 (bit-identical) |

```
./lama-inpaint --image testdata/in.png --mask testdata/mask.png --output /tmp/out.png
```

## Profiling

`LAMA_PROFILE=1` prints a per-phase breakdown (context setup, weight upload,
step loop, download, totals). Finer instruments: `LAMA_PROFILE_OPS`,
`LAMA_PROFILE_SUB`, `LAMA_PROFILE_SYNC`, `LAMA_DUMP_STEP=n`, `LAMA_FFT_HOST`,
`LAMA_DEBUG_CUFFT`, `LAMA_CONVT_GATHER`, `LAMA_NO_POOL`. Without
`LAMA_PROFILE_SYNC`, per-step timings measure submission only.

## Credits and licence

The model architecture and the `big-lama` weights come from the
[saicinpainting](https://github.com/advimman/lama) project (Apache-2.0);
`big-lama.bin` is a repacking of that checkpoint, not a new model. The code in
this repository is Apache-2.0 - see `LICENSE`.
