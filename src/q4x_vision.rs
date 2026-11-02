//! Qwen3.8's image encoder (`mmproj-*.gguf` beside the model, llama.cpp's
//! `qwen3vl_merger`): a 27-layer vision transformer over 16-pixel patches with
//! 2-D rotary positions and learned position embeddings, then a merger that
//! folds each 2x2 block of patches into one embedding row of the text model.
//! Runs on the CPU (f16 weights read through the file map); the rows go into
//! the prompt in place of `<|image_pad|>`, with 3-D (time, row, column)
//! rotary positions (`Engine::rope_pos`). Checked against llama.cpp's
//! `clip_graph_qwen3vl` (tools/mtmd/models/qwen3vl.cpp).

use super::{f16_to_f32, h2f};
use crate::gguf::{Gguf, Ty};
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// The longest side the app sends is limited by pixels, as Qwen's processor
/// does: at most this many (768 x 768 = 576 rows for the model).
#[cfg(test)]
pub const MAX_PIXELS: usize = 768 * 768;
#[cfg(test)]
pub const MIN_PIXELS: usize = 8 * 32 * 32;

/// An encoded picture: `nx` x `ny` rows of the text model's width, row-major.
pub struct Image {
    pub nx: usize,
    pub ny: usize,
    pub rows: Vec<f32>,
}

/// The encoder file beside a model, if there is one.
pub fn find_mmproj(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?;
    let mut c: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.to_ascii_lowercase().starts_with("mmproj") && n.ends_with(".gguf"))).collect();
    // the f16 one first (the others are smaller copies of the same)
    c.sort_by_key(|p| !p.to_string_lossy().to_ascii_uppercase().contains("F16"));
    c.into_iter().next()
}

/// The size Qwen's processor resizes a w x h picture to: sides multiples of
/// 32 (two 16-pixel patches), the area kept within MIN..MAX_PIXELS, the
/// shape kept ("smart_resize").
#[cfg(test)]
pub fn target_size(w: usize, h: usize) -> (usize, usize) {
    let f = 32.0;
    let (wf, hf) = (w.max(1) as f64, h.max(1) as f64);
    let mut hb = ((hf / f).round() * f).max(f);
    let mut wb = ((wf / f).round() * f).max(f);
    if hb * wb > MAX_PIXELS as f64 {
        let beta = (hf * wf / MAX_PIXELS as f64).sqrt();
        hb = ((hf / beta / f).floor() * f).max(f);
        wb = ((wf / beta / f).floor() * f).max(f);
    } else if hb * wb < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    (wb as usize, hb as usize)
}

struct Hp {
    d: usize,
    heads: usize,
    ff: usize,
    layers: usize,
    patch: usize,
    eps: f32,
    out: usize,
}

pub struct Encoder {
    g: Gguf,
    hp: Hp,
}

impl Encoder {
    pub fn open(path: &Path) -> Result<Encoder> {
        let g = Gguf::open(path).with_context(|| format!("opening {}", path.display()))?;
        if g.s("clip.projector_type") != Some("qwen3vl_merger") {
            bail!("{} isn't a Qwen3-VL image encoder (projector {:?})", path.display(), g.s("clip.projector_type"));
        }
        let hp = Hp {
            d: g.u("clip.vision.embedding_length")?,
            heads: g.u("clip.vision.attention.head_count")?,
            ff: g.u("clip.vision.feed_forward_length")?,
            layers: g.u("clip.vision.block_count")?,
            patch: g.u("clip.vision.patch_size")?,
            eps: g.f("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6),
            out: g.u("clip.vision.projection_dim")?,
        };
        if hp.patch != 16 || g.u("clip.vision.spatial_merge_size").unwrap_or(2) != 2 || hp.d % hp.heads != 0 || (hp.d / hp.heads) % 4 != 0 {
            bail!("unexpected image encoder shape");
        }
        Ok(Encoder { g, hp })
    }

    /// The text model's width the rows come out at.
    pub fn out_dim(&self) -> usize {
        self.hp.out
    }

    fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.g.tensor(name)?;
        let b = self.g.bytes(t);
        Ok(match t.ty {
            Ty::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Ty::F16 => b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
            _ => bail!("{name}: unexpected type"),
        })
    }

    /// An f16 matrix as (rows of f16 bits, rows, cols).
    fn mat(&self, name: &str) -> Result<(&[u16], usize, usize)> {
        let t = self.g.tensor(name)?;
        if t.ty != Ty::F16 || t.dims.len() != 2 {
            bail!("{name}: expected a 2-D f16 matrix");
        }
        let b = self.g.bytes(t);
        // SAFETY: the map is 2-byte aligned at a tensor start (gguf aligns to 32)
        let w = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) };
        Ok((w, t.dims[1], t.dims[0]))
    }

    /// x (n x k) times the matrix (+ bias): n x rows.
    fn linear(&self, x: &[f32], n: usize, name: &str) -> Result<Vec<f32>> {
        let (w, m, k) = self.mat(&format!("{name}.weight"))?;
        let b = self.f32s(&format!("{name}.bias"))?;
        debug_assert_eq!(x.len(), n * k);
        Ok(matmul(x, n, k, w, m, &b))
    }

    /// Encodes an RGB picture (w x h, multiples of 32, 3 bytes a pixel).
    pub fn encode(&self, rgb: &[u8], w: usize, h: usize) -> Result<Image> {
        let hp = &self.hp;
        let (d, p) = (hp.d, hp.patch);
        if w % 32 != 0 || h % 32 != 0 || rgb.len() != w * h * 3 {
            bail!("the picture must be sized in multiples of 32 ({w} x {h}, {} bytes)", rgb.len());
        }
        let (pw, ph) = (w / p, h / p);
        let n = pw * ph;
        // patches in merge order: each 2x2 block's four in a row, blocks row-major
        let order: Vec<(usize, usize)> = (0..ph / 2).flat_map(|by| (0..pw / 2).flat_map(move |bx| [(2 * by, 2 * bx), (2 * by, 2 * bx + 1), (2 * by + 1, 2 * bx), (2 * by + 1, 2 * bx + 1)])).collect();

        // patch embedding: both frames' kernels on the one picture, summed;
        // pixels scaled to -1..1 (mean 0.5, std 0.5)
        let w0 = self.f32s("v.patch_embd.weight")?;
        let w1 = self.f32s("v.patch_embd.weight.1")?;
        let wk: Vec<f32> = w0.iter().zip(&w1).map(|(a, b)| a + b).collect();
        let pb = self.f32s("v.patch_embd.bias")?;
        let kin = 3 * p * p;
        let mut x = vec![0f32; n * d];
        x.par_chunks_mut(d).zip(order.par_iter()).for_each(|(xr, &(py, px))| {
            // the patch as [channel][row][col], the kernel's layout
            let mut v = vec![0f32; kin];
            for c in 0..3 {
                for ky in 0..p {
                    for kx in 0..p {
                        let px_ = rgb[((py * p + ky) * w + px * p + kx) * 3 + c] as f32 / 255.0;
                        v[(c * p + ky) * p + kx] = (px_ - 0.5) / 0.5;
                    }
                }
            }
            for (o, (wr, b)) in xr.iter_mut().zip(wk.chunks_exact(kin).zip(&pb)) {
                *o = crate::quant::dot_f32(wr, &v) + b;
            }
        });

        // learned position embeddings: the 48x48 table resized to the patch
        // grid (bilinear, corners aligned), in merge order
        let pe = self.f32s("v.position_embd.weight")?;
        let side = ((pe.len() / d) as f64).sqrt() as usize;
        let (sfx, sfy) = (if pw > 1 && side > 1 { (pw - 1) as f32 / (side - 1) as f32 } else { 1.0 }, if ph > 1 && side > 1 { (ph - 1) as f32 / (side - 1) as f32 } else { 1.0 });
        x.par_chunks_mut(d).zip(order.par_iter()).for_each(|(xr, &(py, px))| {
            let at = |v: f32, sf: f32| {
                let s = v / sf;
                let a = (s.floor() as isize).clamp(0, side as isize - 1) as usize;
                let b = (a + 1).min(side - 1);
                (a, b, (s - a as f32).clamp(0.0, 1.0))
            };
            let (y0, y1, dy) = if ph == side { (py, py, 0.0) } else { at(py as f32, sfy) };
            let (x0, x1, dx) = if pw == side { (px, px, 0.0) } else { at(px as f32, sfx) };
            let row = |y: usize, x: usize| &pe[(y * side + x) * d..(y * side + x + 1) * d];
            let (a, b, c, e) = (row(y0, x0), row(y0, x1), row(y1, x0), row(y1, x1));
            for i in 0..d {
                xr[i] += a[i] * (1.0 - dx) * (1.0 - dy) + b[i] * dx * (1.0 - dy) + c[i] * (1.0 - dx) * dy + e[i] * dx * dy;
            }
        });

        // 2-D rotary angles per patch: the first quarter of each head's dims
        // pairs with the row, the second with the column
        let hd = d / hp.heads;
        let q4 = hd / 4;
        let scale = 10000f32.powf(-2.0 / (hd / 2) as f32);
        let cs: Vec<Vec<(f32, f32)>> = order
            .iter()
            .map(|&(py, px)| {
                let mut v = Vec::with_capacity(hd / 2);
                for (pos, _) in [(py, 0), (px, 1)] {
                    let mut th = pos as f32;
                    for _ in 0..q4 {
                        v.push((th.cos(), th.sin()));
                        th *= scale;
                    }
                }
                v
            })
            .collect();

        for l in 0..hp.layers {
            let pre = format!("v.blk.{l}");
            let y = self.layer_norm(&x, &format!("{pre}.ln1"))?;
            let qkv = self.linear(&y, n, &format!("{pre}.attn_qkv"))?;
            let a = attention(&qkv, n, d, hp.heads, &cs);
            let o = self.linear(&a, n, &format!("{pre}.attn_out"))?;
            x.par_iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            let y = self.layer_norm(&x, &format!("{pre}.ln2"))?;
            let mut u = self.linear(&y, n, &format!("{pre}.ffn_up"))?;
            u.par_iter_mut().for_each(|v| *v = gelu(*v));
            let dn = self.linear(&u, n, &format!("{pre}.ffn_down"))?;
            x.par_iter_mut().zip(&dn).for_each(|(a, b)| *a += b);
            debug_assert_eq!(u.len(), n * hp.ff);
        }
        let x = self.layer_norm(&x, "v.post_ln")?;
        // each 2x2 block's four patches side by side, then the merger
        let nm = n / 4;
        let mut m = self.linear(&x, nm, "mm.0")?;
        m.par_iter_mut().for_each(|v| *v = gelu(*v));
        let rows = self.linear(&m, nm, "mm.2")?;
        Ok(Image { nx: pw / 2, ny: ph / 2, rows })
    }

    fn layer_norm(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let (w, b) = (self.f32s(&format!("{name}.weight"))?, self.f32s(&format!("{name}.bias"))?);
        let d = w.len();
        let eps = self.hp.eps;
        let mut y = vec![0f32; x.len()];
        y.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(yr, xr)| {
            let mean = xr.iter().sum::<f32>() / d as f32;
            let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let r = 1.0 / (var + eps).sqrt();
            for i in 0..d {
                yr[i] = (xr[i] - mean) * r * w[i] + b[i];
            }
        });
        Ok(y)
    }
}

/// GELU, tanh form (ggml_gelu)
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

/// Full (not causal) attention over the patches: qkv rows [q | k | v], heads
/// of hd dims with the rotary angles `cs` (pairs i, i + hd/2).
fn attention(qkv: &[f32], n: usize, d: usize, heads: usize, cs: &[Vec<(f32, f32)>]) -> Vec<f32> {
    let hd = d / heads;
    let half = hd / 2;
    // each head's queries, keys and values together, rotated
    let split = |off: usize, rot: bool| -> Vec<f32> {
        let mut v = vec![0f32; heads * n * hd];
        v.par_chunks_mut(n * hd).enumerate().for_each(|(h, vh)| {
            for (i, r) in vh.chunks_mut(hd).enumerate() {
                r.copy_from_slice(&qkv[i * 3 * d + off + h * hd..i * 3 * d + off + (h + 1) * hd]);
                if rot {
                    for (j, &(c, s)) in cs[i].iter().enumerate() {
                        let (a, b) = (r[j], r[j + half]);
                        r[j] = a * c - b * s;
                        r[j + half] = a * s + b * c;
                    }
                }
            }
        });
        v
    };
    let (q, k, v) = (split(0, true), split(d, true), split(2 * d, false));
    let scale = 1.0 / (hd as f32).sqrt();
    let mut o = vec![0f32; n * d];
    // (head, query) results, then into rows
    let mut oh = vec![0f32; heads * n * hd];
    oh.par_chunks_mut(hd).enumerate().for_each(|(i, out)| {
        let (h, t) = (i / n, i % n);
        let qr = &q[(h * n + t) * hd..(h * n + t + 1) * hd];
        let kh = &k[h * n * hd..(h + 1) * n * hd];
        let vh = &v[h * n * hd..(h + 1) * n * hd];
        let mut sc: Vec<f32> = kh.chunks_exact(hd).map(|kr| crate::quant::dot_f32(qr, kr) * scale).collect();
        let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut den = 0f32;
        for s in sc.iter_mut() {
            *s = (*s - mx).exp();
            den += *s;
        }
        out.fill(0.0);
        for (s, vr) in sc.iter().zip(vh.chunks_exact(hd)) {
            for (a, b) in out.iter_mut().zip(vr) {
                *a += s * b;
            }
        }
        out.iter_mut().for_each(|a| *a /= den);
    });
    o.par_chunks_mut(d).enumerate().for_each(|(t, r)| {
        for h in 0..heads {
            r[h * hd..(h + 1) * hd].copy_from_slice(&oh[(h * n + t) * hd..(h * n + t + 1) * hd]);
        }
    });
    o
}

/// x (n x k) times w (m rows of k, f16) plus b: n x m. Blocks of 16 weight
/// rows are widened once and run over every row of x.
fn matmul(x: &[f32], n: usize, k: usize, w: &[u16], m: usize, b: &[f32]) -> Vec<f32> {
    const RB: usize = 16;
    let mut t = vec![0f32; m * n];
    t.par_chunks_mut(RB * n).enumerate().for_each(|(blk, tb)| {
        let r0 = blk * RB;
        let nr = (m - r0).min(RB);
        let mut wf = vec![0f32; nr * k];
        for r in 0..nr {
            h2f(&w[(r0 + r) * k..(r0 + r + 1) * k], &mut wf[r * k..(r + 1) * k]);
        }
        for i in 0..n {
            let xr = &x[i * k..(i + 1) * k];
            let mut r = 0;
            while r + 4 <= nr {
                let s = dot4(&wf[r * k..(r + 4) * k], k, xr);
                for u in 0..4 {
                    tb[(r + u) * n + i] = s[u] + b[r0 + r + u];
                }
                r += 4;
            }
            while r < nr {
                tb[r * n + i] = crate::quant::dot_f32(&wf[r * k..(r + 1) * k], xr) + b[r0 + r];
                r += 1;
            }
        }
    });
    // to rows of x
    let mut out = vec![0f32; n * m];
    out.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
        for (j, v) in row.iter_mut().enumerate() {
            *v = t[j * n + i];
        }
    });
    out
}

/// Four rows of `w` (each k long) dotted with x, x read once.
fn dot4(w: &[f32], k: usize, x: &[f32]) -> [f32; 4] {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        use std::arch::x86_64::*;
        let (pw, px) = (w.as_ptr(), x.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 8 <= k {
            let xv = _mm256_loadu_ps(px.add(i));
            for (u, a) in acc.iter_mut().enumerate() {
                *a = _mm256_fmadd_ps(_mm256_loadu_ps(pw.add(u * k + i)), xv, *a);
            }
            i += 8;
        }
        let mut out = [0f32; 4];
        for (u, a) in acc.iter().enumerate() {
            let mut l = [0f32; 8];
            _mm256_storeu_ps(l.as_mut_ptr(), *a);
            out[u] = l.iter().sum::<f32>() + (i..k).map(|j| w[u * k + j] * x[j]).sum::<f32>();
        }
        out
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        let mut out = [0f32; 4];
        for (u, o) in out.iter_mut().enumerate() {
            *o = w[u * k..(u + 1) * k].iter().zip(x).map(|(a, b)| a * b).sum();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn target_size() {
        assert_eq!(super::target_size(768, 768), (768, 768));
        assert_eq!(super::target_size(1920, 1080), (1024, 576));
        let (w, h) = super::target_size(4000, 3000);
        assert!(w * h <= super::MAX_PIXELS && w % 32 == 0 && h % 32 == 0);
        assert_eq!(super::target_size(10, 10), (96, 96));
    }
}
