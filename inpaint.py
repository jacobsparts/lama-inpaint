#!/root/lama-service/.venv/bin/python3
"""One-shot LaMa inpainting worker.

Reads an image and a mask, runs the ported big-lama FFC ResNet generator
(``lamacore``) over them, composites the result back over the original pixels
and writes a PNG of the same size as the input.

The mask is an opaque black/white image in the same spirit as pixeldeck's
masker output: any non-black pixel marks a hole (matching the original
saicinpainting pipeline, which thresholds with ``mask > 0``).

The process is deliberately short lived: it loads the weights, runs one
forward pass and exits, so nothing stays resident on the GPU.  The caller
holds the shared GPU lock; this script takes no locks of its own.
"""

import argparse
import os
import sys
import time

# The CPU-side work here is a handful of NumPy casts and one 205 MB weight
# copy; every convolution runs on the GPU.  Left to itself torch builds an
# OpenMP team the size of the machine (16 here) and the weight copy becomes
# nondeterministically slow (measured 0.03 s .. 5.2 s for the same workload),
# which shows up as a ~6 s "model ready" on some runs and ~1 s on others.
# Pin to a single thread before torch is imported so the team is never built.
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lamacore import FFCResNetGenerator  # noqa: E402

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models', 'big-lama.pt')

# The generator has three stride-2 downsample/upsample stages.
PAD_MOD = 8


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _driver_version():
    """Return the version string the *loaded kernel module* reports, or None."""
    try:
        with open('/proc/driver/nvidia/version') as fh:
            first = fh.readline()
    except OSError:
        return None
    parts = first.split()
    return parts[7] if "Kernel" in parts and len(parts) > 7 else None


def _cuda_unavailable_reason():
    """Explain a CUDA init failure in terms of the userspace/kernel driver split.

    ``cuInit`` returns CUDA_ERROR_FORWARD_COMPATIBILITY (804) when libcuda.so.1
    does not match the loaded nvidia kernel module.  On this box the default
    search path resolves to a 550.x libcuda while the module is 535.x, and a
    non-matching 535.274 userspace fails the same way, so the only build that
    works is the one matching the module exactly.
    """
    kernel = _driver_version()
    libdir = next((p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':')
                   if p and os.path.exists(os.path.join(p, 'libcuda.so.1'))), None)
    lines = ['CUDA is not available in this process.']
    if kernel:
        lines.append(f'  nvidia kernel module: {kernel}')
    lines.append(f'  libcuda.so.1 resolved from: {libdir or "(default search path)"}')
    lines.append('  If those versions disagree, cuInit fails with error 804.')
    lines.append('  Point LD_LIBRARY_PATH at the userspace driver matching the '
                 'module, e.g. /home/jacob/nvidia-535.216.01.')
    return "\n".join(lines)


def load_generator(device):
    """Build the generator and load big-lama's TorchScript generator weights.

    The blob is a scripted DefaultInpaintingTrainingModule; the generator lives
    under ``.generator`` and its weights carry the original module names.
    """
    if not os.path.exists(CHECKPOINT):
        raise SystemExit(f'checkpoint not found: {CHECKPOINT}')

    blob = torch.jit.load(CHECKPOINT, map_location='cpu')
    model = FFCResNetGenerator()
    missing, unexpected = model.load_state_dict(blob.generator.state_dict(), strict=False)
    if missing or unexpected:
        raise SystemExit(
            f'checkpoint/architecture mismatch: '
            f'missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}')
    del blob
    model.eval()
    model.to(device)
    return model


def load_mask(path, size):
    """Return a float (H, W) mask, 1.0 on holes, matching ``mask > 0``."""
    mask_img = Image.open(path).convert('L')
    if mask_img.size != size:
        mask_img = mask_img.resize(size, Image.NEAREST)
    mask = np.asarray(mask_img, dtype=np.uint8)
    return (mask > 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', required=True, help='input image path')
    ap.add_argument('--mask', required=True, help='hole mask path (white = hole)')
    ap.add_argument('--output', required=True, help='output PNG path')
    args = ap.parse_args()

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise SystemExit(_cuda_unavailable_reason())

    t0 = time.perf_counter()
    src = Image.open(args.image).convert('RGB')
    width, height = src.size
    img = np.asarray(src, dtype=np.float32) / 255.0          # HWC
    mask = load_mask(args.mask, (width, height))             # HW
    log(f'loaded {width}x{height} image + mask in {time.perf_counter() - t0:.2f}s')

    t0 = time.perf_counter()
    model = load_generator(device)
    log(f'model ready in {time.perf_counter() - t0:.2f}s')

    img_t = torch.from_numpy(img).permute(2, 0, 1)[None]                 # 1,3,H,W
    mask_t = torch.from_numpy(mask)[None, None]                          # 1,1,H,W

    # The generator downsamples and upsamples by 2 three times, so it can only
    # process multiples of PAD_MOD. Pad (reflecting, so the border does not
    # inject a hard edge into the Fourier path) and crop the result back, as
    # the original saicinpainting inference pipeline does with pad_mod=8.
    pad_h = (-height) % PAD_MOD
    pad_w = (-width) % PAD_MOD
    if pad_h or pad_w:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode='reflect')
        mask_t = F.pad(mask_t, (0, pad_w, 0, pad_h), mode='replicate')

    inp = torch.cat([img_t * (1 - mask_t), mask_t], dim=1).to(device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(inp)
        # Same compositing as the original scripted module / predict.py.
        result = mask_t.to(device) * out + (1 - mask_t.to(device)) * img_t.to(device)
    torch.cuda.synchronize()
    log(f'inference {width}x{height} in {time.perf_counter() - t0:.2f}s')

    if pad_h or pad_w:
        result = result[:, :, :height, :width]

    result = result[0].permute(1, 2, 0).cpu().numpy()
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(result, 'RGB').save(args.output, format='PNG')
    log(f'wrote {args.output}')


if __name__ == '__main__':
    main()
