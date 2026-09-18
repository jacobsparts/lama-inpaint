
// Device kernels for the big-lama CUDA path.  Compiled to PTX ahead of time and
// loaded through the driver API, so the Rust build needs neither nvcc nor the
// CUDA headers.
//
// Conventions match the CPU engine in cpu.rs and PyTorch: planes are row-major
// [h][w] inside a contiguous [c][h][w] buffer.

extern "C" {

// im2col for a kxk convolution.  `col` is [cin*kh*kw][oh*ow] with the patch index
// outermost, which is the layout cuBLAS reads as an (oh*ow x cin*kh*kw)
// column-major matrix.
__global__ void k_im2col(
    const float* __restrict__ src, float* __restrict__ col,
    int cin, int h, int w, int kh, int kw,
    int pad_h, int pad_w, int stride, int oh, int ow, int reflect)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)cin * kh * kw * oh * ow;
    if (idx >= total) return;
    int ox = (int)(idx % ow);
    long long t = idx / ow;
    int oy = (int)(t % oh); t /= oh;
    int kx = (int)(t % kw); t /= kw;
    int ky = (int)(t % kh); t /= kh;
    int ci = (int)t;
    int iy = oy * stride + ky - pad_h;
    int ix = ox * stride + kx - pad_w;
    float v = 0.f;
    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
        v = src[((long long)ci * h + iy) * w + ix];
    } else if (reflect) {
        // PyTorch reflect padding: mirror about the edge, edge sample not repeated.
        if (h > 1) { int p = 2 * (h - 1); iy = ((iy % p) + p) % p; if (iy >= h) iy = p - iy; }
        else iy = 0;
        if (w > 1) { int p = 2 * (w - 1); ix = ((ix % p) + p) % p; if (ix >= w) ix = p - ix; }
        else ix = 0;
        v = src[((long long)ci * h + iy) * w + ix];
    }
    col[idx] = v;
}

// Transposed 3x3 stride-2 convolution (the learned upsample).  Gathered rather
// than scattered so no atomics and no zero-fill are needed.
// weight layout is [cin][cout][kh][kw]; output is [cout][oh][ow].
__global__ void k_conv_transpose(
    const float* __restrict__ in, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ out,
    int cin, int cout, int h, int w, int oh, int ow, int kh, int kw, int stride, int pad)
{
    // stride 2, pad 1, k 3: only taps whose (oy + pad - ky) is even and lands in
    // range contribute, so the valid ky set depends only on `oy % stride`.  On
    // the grid-stride loop each thread walks several outputs, so the tap sets are
    // resolved once per step instead of per tap.
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)cout * oh * ow;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += step) {
        int ox = (int)(idx % ow);
        long long t = idx / ow;
        int oy = (int)(t % oh);
        int co = (int)(t / oh);
        float acc = bias ? bias[co] : 0.f;
        // Valid ky for this row: ky == (oy + pad) mod stride, then + stride.
        int y0 = ((oy + pad) % stride + stride) % stride;
        for (int ky = y0; ky < kh; ky += stride) {
            int iy = (oy + pad - ky) / stride;
            if (iy < 0 || iy >= h) continue;
            int x0 = ((ox + pad) % stride + stride) % stride;
            for (int kx = x0; kx < kw; kx += stride) {
                int ix = (ox + pad - kx) / stride;
                if (ix < 0 || ix >= w) continue;
                const float* kp = weight + ((long long)co * kh + ky) * kw + kx;
                // weight is [cin][cout][kh][kw]; stride over ci.
                for (int ci = 0; ci < cin; ++ci) {
                    acc += in[(long long)ci * h * w + (long long)iy * w + ix] * kp[(long long)ci * cout * kh * kw];
                }
            }
        }
        out[idx] = acc;
    }
}

// BatchNorm (folded to scale/shift) followed by ReLU, per channel.
__global__ void k_bn_relu(float* __restrict__ x, const float* __restrict__ scale,
                          const float* __restrict__ shift, long long plane, int channels)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = plane * channels;
    if (idx >= total) return;
    int c = (int)(idx / plane);
    float v = x[idx] * scale[c] + shift[c];
    x[idx] = v > 0.f ? v : 0.f;
}

__global__ void k_relu(float* __restrict__ x, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v > 0.f ? v : 0.f; }
}

__global__ void k_sigmoid(float* __restrict__ x, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] = 1.f / (1.f + __expf(-x[i])); }
}

__global__ void k_add_inplace(float* __restrict__ a, const float* __restrict__ b, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void k_sum(const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ out, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void k_scale(float* __restrict__ x, long long n, float s)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// ReflectionPad2d over [c][h][w] -> [c][h+2p][w+2p].
__global__ void k_reflect_pad(const float* __restrict__ src, float* __restrict__ dst,
                              int c, int h, int w, int pad)
{
    int oh = h + 2 * pad, ow = w + 2 * pad;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)c * oh * ow;
    if (idx >= total) return;
    int x = (int)(idx % ow);
    long long t = idx / ow;
    int y = (int)(t % oh);
    int ch = (int)(t / oh);
    int sy = y - pad, sx = x - pad;
    if (h > 1) { int p = 2 * (h - 1); sy = ((sy % p) + p) % p; if (sy >= h) sy = p - sy; } else sy = 0;
    if (w > 1) { int p = 2 * (w - 1); sx = ((sx % p) + p) % p; if (sx >= w) sx = p - sx; } else sx = 0;
    dst[idx] = src[((long long)ch * h + sy) * w + sx];
}

// 2x2 stride-2 average pool (the SpectralTransform downsample for stride 2).
__global__ void k_avgpool2x2(const float* __restrict__ src, float* __restrict__ dst,
                             int c, int h, int w)
{
    int oh = h / 2, ow = w / 2;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)c * oh * ow;
    if (idx >= total) return;
    int x = (int)(idx % ow);
    long long t = idx / ow;
    int y = (int)(t % oh);
    int ch = (int)(t / oh);
    const float* p = src + (long long)ch * h * w;
    float a = p[(long long)(2 * y) * w + 2 * x];
    float b = p[(long long)(2 * y) * w + 2 * x + 1];
    float cc = p[(long long)(2 * y + 1) * w + 2 * x];
    float d = p[(long long)(2 * y + 1) * w + 2 * x + 1];
    dst[idx] = (a + b + cc + d) * 0.25f;
}

// cuFFT stores complex values interleaved; the reference stacks the real and
// imaginary parts into two channel planes before its 1x1 convolution.
__global__ void k_deinterleave(const float* __restrict__ cplx, float* __restrict__ r,
                               float* __restrict__ im, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { r[i] = cplx[2 * i]; im[i] = cplx[2 * i + 1]; }
}

__global__ void k_interleave(const float* __restrict__ r, const float* __restrict__ im,
                             float* __restrict__ cplx, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { cplx[2 * i] = r[i]; cplx[2 * i + 1] = im[i]; }
}

// Copy one plane; used to build the concatenated local|global tensor and to
// gather the final 3-channel output.
__global__ void k_copy_plane(const float* __restrict__ src, float* __restrict__ dst, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// cuFFT R2C writes an interleaved complex plane per channel; the spectral 1x1
// convolution expects the reference's channel-major stacking, i.e. for each
// channel a contiguous real plane followed by a contiguous imaginary plane
// (c0.re, c0.im, c1.re, c1.im, ...).  `k_spec_pack` produces that layout and
// `k_spec_unpack` reverses it before the inverse transform.
// Pack the interleaved complex spectrum into the stacked real/imag channel
// layout the 1x1 convolution consumes, applying the 1/sqrt(h*w) ortho factor of
// the forward transform in the same pass (cuFFT is unnormalised).  Folding the
// scale in here saves a whole extra kernel launch and a pass over the data per
// spectral call.
__global__ void k_spec_pack(const float* __restrict__ cplx, float* __restrict__ out,
                            int half, int plane, float scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)half * plane) return;
    long long c = idx / plane;
    long long i = idx - c * plane;
    out[c * 2 * plane + i] = cplx[2 * idx] * scale;
    out[c * 2 * plane + plane + i] = cplx[2 * idx + 1] * scale;
}

__global__ void k_spec_unpack(const float* __restrict__ in, float* __restrict__ cplx, int half, int plane)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)half * plane) return;
    long long c = idx / plane;
    long long i = idx - c * plane;
    cplx[2 * idx] = in[c * 2 * plane + i];
    cplx[2 * idx + 1] = in[c * 2 * plane + plane + i];
}

// Real part of an interleaved complex plane with the ortho scale folded in.
__global__ void k_cplx_real_scale(const float* __restrict__ cplx, float* __restrict__ out,
                                  long long n, float scale)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cplx[2 * i] * scale;
}




// Transposed convolution as four phase GEMMs.  With stride 2, pad 1 and a 3x3
// kernel an output pixel (oy, ox) only sees the taps whose (oy + 1 - ky) and
// (ox + 1 - kx) are even, so the tap set depends only on the phase
// (oy % 2, ox % 2): 1 tap for (0,0), 2 for (0,1) and (1,0), 4 for (1,1).
//
// `k_convt_col` gathers, for one phase, the patch matrix [K][n] (K = cin*taps,
// n = number of outputs in the phase) laid out exactly like im2col so cuBLAS
// can consume it.  The phase tap order is row-major over (ky, kx) restricted to
// the phase's tap set, matching the weight reorder done at load time in cuda.rs.
// The tap (ky, kx) pairs for each output phase (oy%2, ox%2), in the order the
// host reorders the weights into.  With stride 2, pad 1 and k = 3 only taps
// with even (oy + 1 - ky) and (ox + 1 - kx) contribute.
__device__ const int CONVT_KY[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 2}};
__device__ const int CONVT_KX[4][4] = {{1, 0, 0, 0}, {0, 2, 0, 0}, {1, 1, 0, 0}, {0, 2, 0, 2}};

__global__ void k_convt_col(
    const float* __restrict__ in, float* __restrict__ col,
    int cin, int h, int w, int py, int px, int taps, int n, int oh, int ow)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)cin * taps * n) return;
    int j = (int)(idx % n);
    long long t = idx / n;
    int ti = (int)(t % taps);
    int ci = (int)(t / taps);
    // The phase grid holds every other output pixel, so its width is half the
    // output width (rounded up for the odd phase).
    int gw = (ow + 1 - px) / 2;
    int i = j / gw;   // phase row index
    int jj = j % gw;  // phase column index
    int oy = 2 * i + py;
    int ox = 2 * jj + px;
    int ph = py * 2 + px;
    int ky = CONVT_KY[ph][ti];
    int kx = CONVT_KX[ph][ti];
    int iy = (oy + 1 - ky) / 2;
    int ix = (ox + 1 - kx) / 2;
    float v = 0.f;
    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
        v = in[((long long)ci * h + iy) * w + ix];
    }
    col[((long long)ci * taps + ti) * n + j] = v;
}

__global__ void k_convt_put(
    const float* __restrict__ out_phase, float* __restrict__ out,
    const float* __restrict__ bias, int cout, int n, int py, int px, int oh, int ow,
    int add_bias)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)cout * n) return;
    int j = (int)(idx % n);
    long long co = idx / n;
    int gw = (ow + 1 - px) / 2;
    int i = j / gw;
    int jj = j % gw;
    int oy = 2 * i + py;
    int ox = 2 * jj + px;
    float v = out_phase[idx];
    if (add_bias && bias) v += bias[co];
    out[(co * oh + oy) * ow + ox] = v;
}


// Add a per-channel bias to a `[cout][n]` plane in place.  Used by the
// convolution paths, which own their output buffers; doing it on the device
// avoids materialising a `cout*n` broadcast plane on the host.
__global__ void k_bias_plane(float* __restrict__ out, const float* __restrict__ bias,
                             int cout, long long n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)cout * n) return;
    out[idx] += bias[idx / n];
}


// Direct 7x7 convolution, one thread per output spatial position.
//
// The OutConv is 7x7 with only three output channels over a 518x518 plane, so
// the im2col path materialises a `(cin*49) x (h*w)` patch matrix - 3.37 GB of
// traffic at 512x512 - to feed a 5 GFLOP GEMM, only 1.5 FLOP/byte.  Here each
// thread walks its 7x7 receptive field directly and keeps the output
// accumulators in registers: no patch matrix, and the overlapping reads of
// neighbouring threads hit cache.  The caller has already applied the
// reflection padding, so plain bounds checks are enough.
//
// COUT_MAX caps the register accumulators; the host picks this kernel only when
// `cout <= COUT_MAX` and otherwise falls back to im2col + SGEMM.
#define COUT_MAX 8

__global__ void k_conv7x7(const float* __restrict__ in, const float* __restrict__ w,
                          const float* __restrict__ bias, float* __restrict__ out,
                          int cin, int cout, int h_in, int w_in, int h_out, int w_out)
{
    // Output geometry drives the indexing; the input has its own (larger) extent
    // and row stride - conflating the two silently reads the wrong rows whenever
    // the input and output widths differ, which they do here (518 -> 512).
    long long n = (long long)h_out * w_out;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int y = (int)(idx / w_out);
    int x = (int)(idx % w_out);
    float acc[COUT_MAX];
    for (int co = 0; co < COUT_MAX; ++co) acc[co] = 0.f;
    // Slide the kernel over the receptive field.  The inner loop walks x with a
    // fixed dy, so the input reads are contiguous and the weight reads are the
    // small 7x7xcin block reused by every thread.
    long long in_plane = (long long)h_in * w_in;
    for (int ci = 0; ci < cin; ++ci) {
        const float* plane = in + (long long)ci * in_plane;
        const float* kp = w + ((long long)ci * cout) * 49;
        for (int dy = 0; dy < 7; ++dy) {
            int iy = y + dy;
            if (iy >= h_in) break;
            const float* row = plane + (long long)iy * w_in;
            for (int dx = 0; dx < 7; ++dx) {
                int ix = x + dx;
                if (ix >= w_in) break;
                float v = row[ix];
                // `kp` is [cout][7][7] for this input channel; each output
                // channel reads its own 49-element slice.
                const float* wc = kp + (long long)dy * 7 + dx;
                for (int co = 0; co < cout && co < COUT_MAX; ++co) {
                    acc[co] += v * wc[(long long)co * 49];
                }
            }
        }
    }
    for (int co = 0; co < cout && co < COUT_MAX; ++co) {
        float b = bias ? bias[co] : 0.f;
        out[(long long)co * n + idx] = acc[co] + b;
    }
}


// Fused BatchNorm + ReLU + residual add: `out = relu(scale*x + shift) + skip`.
//
// The resblock tail otherwise needs two passes over the activation (a BN+ReLU
// and an add) plus a full device-to-device copy of the input for the skip.  One
// pass keeps the activation hot in L2 and removes a launch and a copy.
__global__ void k_bn_relu_add(float* __restrict__ x, const float* __restrict__ skip,
                              const float* __restrict__ scale, const float* __restrict__ shift,
                              long long plane, int channels)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plane * channels) return;
    int ch = (int)(idx / plane);
    float v = x[idx] * scale[ch] + shift[ch];
    v = v > 0.f ? v : 0.f;
    x[idx] = skip ? v + skip[idx] : v;
}


}  // extern "C"
