//! CPU kernels of the DeepSeek V4 engine (`ds4`): Q4_K expert rows against
//! activations quantized to 8 bits per 256-value block (as llama.cpp's CPU
//! path does), and f32 vector helpers.

/// Activations quantized for Q4_K rows: per 256 values a scale and i8
/// values, and the sum of each 32 values.
pub struct Q8x {
    q: Vec<i8>,
    d: Vec<f32>,
    bsum: Vec<i32>,
}

impl Q8x {
    pub fn new(x: &[f32]) -> Q8x {
        assert!(x.len() % 256 == 0);
        let nb = x.len() / 256;
        let mut q = vec![0i8; x.len()];
        let mut d = vec![0f32; nb];
        let mut bsum = vec![0i32; nb * 8];
        for b in 0..nb {
            let xs = &x[b * 256..b * 256 + 256];
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            let id = 127.0 / amax;
            d[b] = amax / 127.0;
            for (i, v) in xs.iter().enumerate() {
                let r = (v * id).round().clamp(-127.0, 127.0) as i8;
                q[b * 256 + i] = r;
                bsum[b * 8 + i / 32] += r as i32;
            }
        }
        Q8x { q, d, bsum }
    }
}

#[inline]
fn f16(b: &[u8]) -> f32 {
    crate::ds4::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

/// Q4_K sub-block scales and mins (6 bits each, in 12 bytes).
#[inline]
fn scales(s: &[u8]) -> ([i32; 8], [i32; 8]) {
    let (mut sc, mut m) = ([0i32; 8], [0i32; 8]);
    for j in 0..8 {
        if j < 4 {
            sc[j] = (s[j] & 63) as i32;
            m[j] = (s[j + 4] & 63) as i32;
        } else {
            sc[j] = ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4)) as i32;
            m[j] = ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as i32;
        }
    }
    (sc, m)
}

fn row_scalar(w: &[u8], x: &Q8x) -> f32 {
    let mut total = 0f32;
    for (b, blk) in w.chunks_exact(144).enumerate() {
        let (d, dmin) = (f16(&blk[0..2]), f16(&blk[2..4]));
        let (sc, m) = scales(&blk[4..16]);
        let qs = &blk[16..144];
        let xq = &x.q[b * 256..b * 256 + 256];
        let mut isum = 0i32;
        let mut msum = 0i32;
        for c in 0..4 {
            let (mut lo, mut hi) = (0i32, 0i32);
            for i in 0..32 {
                let v = qs[32 * c + i];
                lo += (v & 15) as i32 * xq[64 * c + i] as i32;
                hi += (v >> 4) as i32 * xq[64 * c + 32 + i] as i32;
            }
            isum += sc[2 * c] * lo + sc[2 * c + 1] * hi;
        }
        for j in 0..8 {
            msum += m[j] * x.bsum[b * 8 + j];
        }
        total += x.d[b] * (d * isum as f32 - dmin * msum as f32);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
unsafe fn row_vnni(w: &[u8], x: &Q8x) -> f32 {
    use std::arch::x86_64::*;
    let mask = _mm256_set1_epi8(15);
    let mut total = 0f32;
    for (b, blk) in w.chunks_exact(144).enumerate() {
        let (d, dmin) = (f16(&blk[0..2]), f16(&blk[2..4]));
        let (sc, m) = scales(&blk[4..16]);
        let qs = blk.as_ptr().add(16);
        let xq = x.q.as_ptr().add(b * 256);
        let mut acc = _mm512_setzero_si512();
        for c in 0..4 {
            let q = _mm256_loadu_si256(qs.add(32 * c) as *const __m256i);
            let lo = _mm256_and_si256(q, mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), mask);
            let qv = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            let xv = _mm512_loadu_si512(xq.add(64 * c) as *const __m512i);
            let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qv, xv);
            let s = _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32(sc[2 * c]), _mm512_set1_epi32(sc[2 * c + 1]));
            acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dp, s));
        }
        let isum = _mm512_reduce_add_epi32(acc);
        let mut msum = 0i32;
        for j in 0..8 {
            msum += m[j] * x.bsum[b * 8 + j];
        }
        total += x.d[b] * (d * isum as f32 - dmin * msum as f32);
    }
    total
}

fn vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(|| is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// out[r] = row r of Q4_K matrix `w` (rows of x.len() values) . x
pub fn q4k_rows(w: &[u8], x: &Q8x, out: &mut [f32]) {
    let bpr = x.d.len() * 144;
    let fast = vnni();
    for (r, o) in out.iter_mut().enumerate() {
        let row = &w[r * bpr..(r + 1) * bpr];
        #[cfg(target_arch = "x86_64")]
        {
            if fast {
                // SAFETY: the features were detected above
                *o = unsafe { row_vnni(row, x) };
                continue;
            }
        }
        *o = row_scalar(row, x);
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::quant::dot_f32(a, b)
}

/// y += a * x
#[inline]
pub fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (yv, xv) in y.iter_mut().zip(x) {
        *yv += a * xv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vnni_matches_scalar() {
        let mut s = 1u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        };
        let blocks = 4;
        let mut w = vec![0u8; blocks * 144];
        for b in 0..blocks {
            let blk = &mut w[b * 144..(b + 1) * 144];
            blk[0..2].copy_from_slice(&0x2c00u16.to_le_bytes());
            blk[2..4].copy_from_slice(&0x2800u16.to_le_bytes());
            for v in blk[4..].iter_mut() {
                *v = rnd() as u8;
            }
        }
        let x: Vec<f32> = (0..blocks * 256).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
        let q = Q8x::new(&x);
        let a = row_scalar(&w, &q);
        if vnni() {
            let b = unsafe { row_vnni(&w, &q) };
            assert!((a - b).abs() <= 1e-3 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
}
