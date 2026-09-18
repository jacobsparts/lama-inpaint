//! PNG I/O and the padding the network's letterboxing needs.

use png::{BitDepth, ColorType, Decoder, Encoder};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// An 8-bit RGB image, row-major HWC.
pub struct Rgb8 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// An 8-bit single-channel image, row-major HW.
pub struct Gray8 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Rgb8 {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0; width * height * 3] }
    }
}

struct Raw {
    width: usize,
    height: usize,
    color: ColorType,
    depth: BitDepth,
    channels: usize,
    buf: Vec<u8>,
    palette: Vec<u8>,
}

fn decode_raw(path: &Path) -> Result<Raw, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let dec = Decoder::new(file);
    let mut rdr = dec.read_info().map_err(|e| e.to_string())?;
    let (width, height, color, depth) = {
        let i = rdr.info();
        (i.width as usize, i.height as usize, i.color_type, i.bit_depth)
    };
    let channels = match color {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => 1,
    };
    let sample_bytes = match depth {
        BitDepth::Sixteen => 2,
        BitDepth::Eight => 1,
        other => return Err(format!("unsupported PNG bit depth {other:?}")),
    };
    let mut buf = vec![0u8; width * height * channels * sample_bytes];
    rdr.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let palette = rdr.info().palette.as_ref().map(|p| p.to_vec()).unwrap_or_default();
    Ok(Raw { width, height, color, depth, channels, buf, palette })
}

impl Raw {
    /// Sample `index` (0-based within the row) as 8-bit, taking the high byte
    /// of 16-bit samples.
    #[inline]
    fn sample(&self, y: usize, x: usize, c: usize) -> u8 {
        let idx = (y * self.width + x) * self.channels + c;
        if self.depth == BitDepth::Sixteen {
            self.buf[idx * 2]
        } else {
            self.buf[idx]
        }
    }
}

/// Decode a PNG to 8-bit RGB, expanding greyscale / palette / 16-bit as needed.
pub fn read_rgb(path: &Path) -> Result<Rgb8, String> {
    let raw = decode_raw(path)?;
    let mut img = Rgb8::new(raw.width, raw.height);
    let palette_entries = raw.palette.len() / 3;
    for y in 0..raw.height {
        for x in 0..raw.width {
            let (r, g, b) = match raw.color {
                ColorType::Rgb => (raw.sample(y, x, 0), raw.sample(y, x, 1), raw.sample(y, x, 2)),
                ColorType::Rgba => (raw.sample(y, x, 0), raw.sample(y, x, 1), raw.sample(y, x, 2)),
                ColorType::Grayscale => {
                    let v = raw.sample(y, x, 0);
                    (v, v, v)
                }
                ColorType::GrayscaleAlpha => {
                    let v = raw.sample(y, x, 0);
                    (v, v, v)
                }
                ColorType::Indexed => {
                    let i = (raw.sample(y, x, 0) as usize).min(palette_entries.saturating_sub(1));
                    let o = i * 3;
                    if o + 2 < raw.palette.len() {
                        (raw.palette[o], raw.palette[o + 1], raw.palette[o + 2])
                    } else {
                        (0, 0, 0)
                    }
                }
            };
            let o = (y * raw.width + x) * 3;
            img.data[o] = r;
            img.data[o + 1] = g;
            img.data[o + 2] = b;
        }
    }
    Ok(img)
}

/// Decode any PNG to 8-bit greyscale using the red channel.
pub fn read_gray(path: &Path) -> Result<Gray8, String> {
    let rgb = read_rgb(path)?;
    let mut g = Gray8 { width: rgb.width, height: rgb.height, data: vec![0u8; rgb.width * rgb.height] };
    for (px, out) in rgb.data.chunks_exact(3).zip(g.data.iter_mut()) {
        *out = px[0];
    }
    Ok(g)
}

pub fn write_rgb(path: &Path, img: &Rgb8) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = Encoder::new(BufWriter::new(file), img.width as u32, img.height as u32);
    enc.set_color(ColorType::Rgb);
    enc.set_depth(BitDepth::Eight);
    let mut wtr = enc.write_header().map_err(|e| e.to_string())?;
    wtr.write_image_data(&img.data).map_err(|e| e.to_string())?;
    wtr.finish().map_err(|e| e.to_string())?;
    Ok(())
}

/// Map a possibly out-of-range index back inside `[0, n)` by reflection,
/// mirroring PyTorch's `reflect` padding (the edge sample is not repeated).
pub fn reflect_index(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n as isize - 1);
    let mut m = i.rem_euclid(period);
    if m >= n as isize {
        m = period - m;
    }
    m as usize
}
