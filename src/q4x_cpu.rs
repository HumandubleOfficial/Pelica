//! CPU kernels of the Qwen3.8 engine (`q4x`): expert rows of the types its
//! GGUF uses (IQ3_S, IQ4_XS, IQ4_NL, Q8_0) against activations quantized to
//! 8 bits per 32 values, and an expert (gate, up, down) for a few tokens.
//!
//! Every kernel works out the same integer sums per 32 values; the AVX-512
//! VNNI paths add them up in a different order than the scalar ones (float
//! rounding only).

use crate::gguf::Ty;
use crate::iq_tables::{IQ3S_GRID, KVALUES_IQ4NL};

/// Activations quantized per 32 values: a scale and i8 values.
pub struct Q8 {
    pub q: Vec<i8>,
    pub d: Vec<f32>,
    /// the values again in "plane" order per 256 (when a multiple of 256):
    /// plane j (64 bytes) holds values 4i + j, as the IQ3_S kernel that
    /// looks up grid bytes a plane at a time wants them
    pub p: Vec<i8>,
}

impl Q8 {
    pub fn new(x: &[f32]) -> Q8 {
        assert!(x.len() % 32 == 0);
        let mut q = vec![0i8; x.len()];
        let mut d = vec![0f32; x.len() / 32];
        for (b, (xs, qs)) in x.chunks_exact(32).zip(q.chunks_exact_mut(32)).enumerate() {
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            d[b] = amax / 127.0;
            let id = 127.0 / amax;
            for (o, v) in qs.iter_mut().zip(xs) {
                *o = (v * id).round().clamp(-127.0, 127.0) as i8;
            }
        }
        let mut p = Vec::new();
        if x.len() % 256 == 0 {
            p = vec![0i8; x.len()];
            for (src, dst) in q.chunks_exact(256).zip(p.chunks_exact_mut(256)) {
                for i in 0..64 {
                    for j in 0..4 {
                        dst[64 * j + i] = src[4 * i + j];
                    }
                }
            }
        }
        Q8 { q, d, p }
    }
}

#[inline]
fn f16(b: &[u8]) -> f32 {
    crate::ds4::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

#[inline]
fn idot(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// IQ3_S sub-block `ib` (32 values) of a 110-byte block, as i8.
#[inline]
fn iq3s_sub(blk: &[u8], ib: usize, out: &mut [i8; 32]) -> i32 {
    let qs = &blk[2..66];
    let qh = blk[66 + ib];
    let signs = &blk[74..106];
    let sc = (blk[106 + ib / 2] >> (4 * (ib % 2))) & 15;
    for k in 0..8 {
        let idx = qs[8 * ib + k] as usize | (((qh >> k) & 1) as usize) << 8;
        let g = IQ3S_GRID[idx].to_le_bytes();
        for j in 0..4 {
            let v = 4 * k + j;
            let neg = (signs[4 * ib + v / 8] >> (v % 8)) & 1 == 1;
            out[v] = if neg { -(g[j] as i8) } else { g[j] as i8 };
        }
    }
    1 + 2 * sc as i32
}

fn row_scalar(ty: Ty, w: &[u8], x: &Q8) -> f32 {
    let mut total = 0f32;
    match ty {
        Ty::Q8_0 => {
            for (b, blk) in w.chunks_exact(34).enumerate() {
                let q: &[i8] = unsafe { std::slice::from_raw_parts(blk[2..].as_ptr() as *const i8, 32) };
                total += f16(blk) * x.d[b] * idot(q, &x.q[32 * b..32 * b + 32]) as f32;
            }
        }
        Ty::Iq4Nl => {
            for (b, blk) in w.chunks_exact(18).enumerate() {
                let mut q = [0i8; 32];
                for i in 0..16 {
                    q[i] = KVALUES_IQ4NL[(blk[2 + i] & 15) as usize];
                    q[16 + i] = KVALUES_IQ4NL[(blk[2 + i] >> 4) as usize];
                }
                total += f16(blk) * x.d[b] * idot(&q, &x.q[32 * b..32 * b + 32]) as f32;
            }
        }
        Ty::Iq4Xs => {
            for (b, blk) in w.chunks_exact(136).enumerate() {
                let d = f16(blk);
                let sh = u16::from_le_bytes([blk[2], blk[3]]);
                for ib in 0..8 {
                    let ls = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 15) as i32 | (((sh >> (2 * ib)) & 3) as i32) << 4;
                    let qs = &blk[8 + 16 * ib..8 + 16 * ib + 16];
                    let mut q = [0i8; 32];
                    for i in 0..16 {
                        q[i] = KVALUES_IQ4NL[(qs[i] & 15) as usize];
                        q[16 + i] = KVALUES_IQ4NL[(qs[i] >> 4) as usize];
                    }
                    let s = 8 * b + ib;
                    total += d * (ls - 32) as f32 * x.d[s] * idot(&q, &x.q[32 * s..32 * s + 32]) as f32;
                }
            }
        }
        Ty::Iq3S => {
            for (b, blk) in w.chunks_exact(110).enumerate() {
                let d = f16(blk);
                let mut q = [0i8; 32];
                for ib in 0..8 {
                    let sc = iq3s_sub(blk, ib, &mut q);
                    let s = 8 * b + ib;
                    total += d * sc as f32 * x.d[s] * idot(&q, &x.q[32 * s..32 * s + 32]) as f32;
                }
            }
        }
        t => panic!("no CPU kernel for {}", t.name()),
    }
    total
}

#[cfg(target_arch = "x86_64")]
mod vnni {
    use super::*;
    use std::arch::x86_64::*;

    /// 8 lanes of sum(w * x) for 32 i8 weights against 32 i8 activations
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl,avx2,f16c")]
    unsafe fn dot32(w: __m256i, x: __m256i) -> __m256i {
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(w), _mm256_sign_epi8(x, w))
    }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl,avx2,f16c")]
    unsafe fn hsum(v: __m256) -> f32 {
        let h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
        _mm_cvtss_f32(h)
    }

    /// 16 packed nibbles (low ones are values 0..16, high ones 16..32) through
    /// the IQ4 table
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl,avx2,f16c")]
    unsafe fn iq4_32(p: *const u8, tab: __m256i) -> __m256i {
        let q = _mm_loadu_si128(p as *const __m128i);
        let lo = _mm_and_si128(q, _mm_set1_epi8(15));
        let hi = _mm_and_si128(_mm_srli_epi16(q, 4), _mm_set1_epi8(15));
        _mm256_shuffle_epi8(tab, _mm256_set_m128i(hi, lo))
    }

    /// an f16 scale (F16C)
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl,avx2,f16c")]
    unsafe fn h(p: *const u8) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((p as *const u16).read_unaligned() as i32)))
    }

    /// The IQ3_S grid as two 512-byte tables of nibble pairs: entry i's bytes
    /// 0 and 1 (A), 2 and 3 (B), each 1..15.
    fn iq3s_tables() -> &'static [[u8; 512]; 2] {
        static T: std::sync::OnceLock<[[u8; 512]; 2]> = std::sync::OnceLock::new();
        T.get_or_init(|| {
            let mut t = [[0u8; 512]; 2];
            for (i, g) in IQ3S_GRID.iter().enumerate() {
                let b = g.to_le_bytes();
                t[0][i] = b[0] | b[1] << 4;
                t[1][i] = b[2] | b[3] << 4;
            }
            t
        })
    }

    /// For plane j, byte i (qword q = i / 8, k = i % 8) of the bit-shuffle
    /// control: where value 4i + j's sign bit sits in qword q of the signs
    /// spread as [s0 s0 s1 s1 s2 s2 s3 s3].
    const fn sign_sel() -> [[u8; 64]; 4] {
        let mut t = [[0u8; 64]; 4];
        let mut j = 0;
        while j < 4 {
            let mut i = 0;
            while i < 64 {
                t[j][i] = (32 * ((i / 8) % 2) + 4 * (i % 8) + j) as u8;
                i += 1;
            }
            j += 1;
        }
        t
    }
    static SIGN_SEL: [[u8; 64]; 4] = sign_sel();

    /// 512 grid entries' nibble pairs for the 64 indices (low 8 bits in
    /// `lo`, bit 7 also as `m7`, bit 8 as `m8`)
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn look(t: &[__m512i; 8], lo: __m512i, m7: u64, m8: u64) -> __m512i {
        let r0 = _mm512_permutex2var_epi8(t[0], lo, t[1]);
        let r1 = _mm512_permutex2var_epi8(t[2], lo, t[3]);
        let r2 = _mm512_permutex2var_epi8(t[4], lo, t[5]);
        let r3 = _mm512_permutex2var_epi8(t[6], lo, t[7]);
        _mm512_mask_blend_epi8(m8, _mm512_mask_blend_epi8(m7, r0, r1), _mm512_mask_blend_epi8(m7, r2, r3))
    }

    /// An IQ3_S row a whole 256-value block at a time, without gathers: the
    /// 64 grid indices go through byte permutes (two bytes of each entry per
    /// lookup), against the activations in plane order (`Q8::p`); each
    /// sign bit is picked out for its plane with a bit shuffle.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl,avx512vbmi,avx512bitalg,avx2,f16c,fma")]
    pub unsafe fn row_iq3s(w: &[u8], x: &Q8) -> f32 {
        let tabs = iq3s_tables();
        let ld = |t: &[u8; 512]| -> [__m512i; 8] { std::array::from_fn(|k| _mm512_loadu_si512(t.as_ptr().add(64 * k) as *const _)) };
        let (ta, tb) = (ld(&tabs[0]), ld(&tabs[1]));
        let sel: [__m512i; 4] = std::array::from_fn(|j| _mm512_loadu_si512(SIGN_SEL[j].as_ptr() as *const _));
        let spread = _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3);
        let pairs = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        let nib_sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
        let low = _mm512_set1_epi8(15);
        let mut acc = _mm512_setzero_ps();
        let xp = x.p.as_ptr();
        for (b, blk) in w.chunks_exact(110).enumerate() {
            let p = blk.as_ptr();
            // the 8 sub-blocks' scales: d * (1 + 2 sc) * the activations' scale
            let scb = (p.add(106) as *const u32).read_unaligned() as i32;
            let nib = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(scb), nib_sh), _mm256_set1_epi32(15));
            let scf = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32::<1>(nib), _mm256_set1_epi32(1)));
            let s8 = _mm256_mul_ps(_mm256_mul_ps(scf, _mm256_loadu_ps(x.d.as_ptr().add(8 * b))), _mm256_set1_ps(h(p)));
            let s16 = _mm512_permutexvar_ps(pairs, _mm512_castps256_ps512(s8));
            // entry i (sub-block i / 8): index qs[i] | qh bit i << 8
            let lo = _mm512_loadu_si512(p.add(2) as *const _);
            let m7 = _mm512_test_epi8_mask(lo, _mm512_set1_epi8(-128));
            let m8 = (p.add(66) as *const u64).read_unaligned();
            let a = look(&ta, lo, m7, m8);
            let bb = look(&tb, lo, m7, m8);
            let wv = [_mm512_and_si512(a, low), _mm512_and_si512(_mm512_srli_epi16::<4>(a), low), _mm512_and_si512(bb, low), _mm512_and_si512(_mm512_srli_epi16::<4>(bb), low)];
            let signs = _mm512_permutexvar_epi64(spread, _mm512_castsi256_si512(_mm256_loadu_si256(p.add(74) as *const __m256i)));
            let mut dp = _mm512_setzero_si512();
            for j in 0..4 {
                let neg = _mm512_bitshuffle_epi64_mask(signs, sel[j]);
                let xv = _mm512_loadu_si512(xp.add(256 * b + 64 * j) as *const _);
                dp = _mm512_dpbusd_epi32(dp, wv[j], _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv));
            }
            // dword m: entries 4m..4m+4, sub-block m / 2
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), s16, acc);
        }
        _mm512_reduce_add_ps(acc)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl,avx2,f16c,fma")]
    pub unsafe fn row(ty: Ty, w: &[u8], x: &Q8) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let xq = x.q.as_ptr();
        let tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i));
        match ty {
            Ty::Q8_0 => {
                for (b, blk) in w.chunks_exact(34).enumerate() {
                    let wv = _mm256_loadu_si256(blk.as_ptr().add(2) as *const __m256i);
                    let s = dot32(wv, _mm256_loadu_si256(xq.add(32 * b) as *const __m256i));
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(h(blk.as_ptr()) * x.d[b]), acc);
                }
            }
            Ty::Iq4Nl => {
                for (b, blk) in w.chunks_exact(18).enumerate() {
                    let wv = iq4_32(blk.as_ptr().add(2), tab);
                    let s = dot32(wv, _mm256_loadu_si256(xq.add(32 * b) as *const __m256i));
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(h(blk.as_ptr()) * x.d[b]), acc);
                }
            }
            Ty::Iq4Xs => {
                for (b, blk) in w.chunks_exact(136).enumerate() {
                    let d = h(blk.as_ptr());
                    let sh = u16::from_le_bytes([blk[2], blk[3]]);
                    for ib in 0..8 {
                        let ls = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 15) as i32 | (((sh >> (2 * ib)) & 3) as i32) << 4;
                        let wv = iq4_32(blk.as_ptr().add(8 + 16 * ib), tab);
                        let s = 8 * b + ib;
                        let p = dot32(wv, _mm256_loadu_si256(xq.add(32 * s) as *const __m256i));
                        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * (ls - 32) as f32 * x.d[s]), acc);
                    }
                }
            }
            Ty::Iq3S => {
                // two sub-blocks (64 values) per 512-bit step: lanes 0..8 the
                // first's 8 grid entries, 8..16 the second's
                let grid = IQ3S_GRID.as_ptr() as *const i32;
                let lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                let one = _mm512_set1_epi32(1);
                let nib_sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
                let pick = [0, 1, 2, 3].map(|pr: i32| _mm512_setr_epi32(2 * pr, 2 * pr, 2 * pr, 2 * pr, 2 * pr, 2 * pr, 2 * pr, 2 * pr, 2 * pr + 1, 2 * pr + 1, 2 * pr + 1, 2 * pr + 1, 2 * pr + 1, 2 * pr + 1, 2 * pr + 1, 2 * pr + 1));
                let mut acc5 = _mm512_setzero_ps();
                for (b, blk) in w.chunks_exact(110).enumerate() {
                    let p = blk.as_ptr();
                    // the 8 sub-blocks' scales: d * (1 + 2 sc) * the activations' scale
                    let scb = (p.add(106) as *const u32).read_unaligned() as i32;
                    let nib = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(scb), nib_sh), _mm256_set1_epi32(15));
                    let scf = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32::<1>(nib), _mm256_set1_epi32(1)));
                    let s8 = _mm256_mul_ps(_mm256_mul_ps(scf, _mm256_loadu_ps(x.d.as_ptr().add(8 * b))), _mm256_set1_ps(h(p)));
                    let s16 = _mm512_castps256_ps512(s8);
                    let qh = (p.add(66) as *const u64).read_unaligned();
                    for pr in 0..4 {
                        let ib = 2 * pr;
                        // grid index: 8 low bits from qs, the 9th from qh (bit k of
                        // the sub-block's byte: bit `lane` of the pair)
                        let lo = _mm512_cvtepu8_epi32(_mm_loadu_si128(p.add(2 + 8 * ib) as *const __m128i));
                        let pair = ((qh >> (8 * ib)) & 0xffff) as i32;
                        let hi = _mm512_slli_epi32::<8>(_mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32(pair), lane), one));
                        let mag = _mm512_i32gather_epi32::<4>(_mm512_or_si512(lo, hi), grid);
                        let sm = (p.add(74 + 4 * ib) as *const u64).read_unaligned();
                        let xv = _mm512_loadu_si512(xq.add(32 * (8 * b + ib)) as *const _);
                        let xs = _mm512_mask_sub_epi8(xv, sm, _mm512_setzero_si512(), xv);
                        let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), mag, xs);
                        acc5 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dp), _mm512_permutexvar_ps(pick[pr], s16), acc5);
                    }
                }
                return _mm512_reduce_add_ps(acc5);
            }
            t => panic!("no CPU kernel for {}", t.name()),
        }
        hsum(acc)
    }
}

/// AVX-512 VBMI and BITALG too (the gather-free IQ3_S kernel)
fn vbmi() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(|| vnni() && std::env::var("PELICA_NO_VBMI").is_err() && is_x86_feature_detected!("avx512vbmi") && is_x86_feature_detected!("avx512bitalg"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

fn vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(|| {
            std::env::var("PELICA_SCALAR").is_err() && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Whether the CPU kernels handle `ty`.
pub fn supports(ty: Ty) -> bool {
    matches!(ty, Ty::Q8_0 | Ty::Iq4Nl | Ty::Iq4Xs | Ty::Iq3S)
}

/// One row of type `ty` (as many values as x has) . x
#[inline]
pub fn row(ty: Ty, w: &[u8], x: &Q8) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if ty == Ty::Iq3S && !x.p.is_empty() && vbmi() {
            // SAFETY: the features were detected
            return unsafe { vnni::row_iq3s(w, x) };
        }
        if vnni() {
            // SAFETY: the features were detected
            return unsafe { vnni::row(ty, w, x) };
        }
    }
    row_scalar(ty, w, x)
}

/// out[r * xs.len() + t] = row r of `w` . xs[t], for out.len() / xs.len() rows
pub fn rows(ty: Ty, w: &[u8], xs: &[&Q8], out: &mut [f32]) {
    let nt = xs.len();
    if nt == 0 {
        return;
    }
    let bpr = ty.bytes(xs[0].q.len());
    for (r, o) in out.chunks_mut(nt).enumerate() {
        let rw = &w[r * bpr..(r + 1) * bpr];
        for (x, v) in xs.iter().zip(o.iter_mut()) {
            *v = row(ty, rw, x);
        }
    }
}

/// One expert for some tokens: silu(gate x) * (up x) * the token's routing
/// weight, through down. `gw`/`uw` are ff rows of d values (type tys[0]),
/// `dw` d rows of ff (tys[2]); returns [token][d].
pub fn expert(tys: [Ty; 3], gw: &[u8], uw: &[u8], dw: &[u8], d: usize, ff: usize, xs: &[&Q8], wts: &[f32]) -> Vec<f32> {
    use rayon::prelude::*;
    let nt = xs.len();
    const CH: usize = 32;
    let (bg, bd) = (tys[0].bytes(d), tys[2].bytes(ff));
    let gch = ff.div_ceil(CH);
    let mut gu = vec![0f32; 2 * ff * nt];
    let (g_out, u_out) = gu.split_at_mut(ff * nt);
    let work = |(ci, o): (usize, &mut [f32])| {
        if ci < gch {
            rows(tys[0], &gw[ci * CH * bg..], xs, o)
        } else {
            rows(tys[1], &uw[(ci - gch) * CH * bg..], xs, o)
        }
    };
    let jobs: Vec<(usize, &mut [f32])> = g_out.chunks_mut(CH * nt).chain(u_out.chunks_mut(CH * nt)).enumerate().collect();
    // (rows split over all threads, also when called from one: decode has
    // only a few experts a layer for the CPU)
    jobs.into_par_iter().for_each(work);
    let hq: Vec<Q8> = (0..nt)
        .map(|t| {
            let h: Vec<f32> = (0..ff)
                .map(|j| {
                    let (g, u) = (gu[j * nt + t], gu[(ff + j) * nt + t]);
                    g / (1.0 + (-g).exp()) * u * wts[t]
                })
                .collect();
            Q8::new(&h)
        })
        .collect();
    let hs: Vec<&Q8> = hq.iter().collect();
    let mut o = vec![0f32; d * nt];
    let down = |(ci, oc): (usize, &mut [f32])| rows(tys[2], &dw[ci * CH * bd..], &hs, oc);
    o.par_chunks_mut(CH * nt).enumerate().for_each(down);
    let mut out = vec![0f32; nt * d];
    for t in 0..nt {
        for r in 0..d {
            out[t * d + r] = o[r * nt + t];
        }
    }
    out
}

/// Dequantizes one row of `n` values of a type the kernels handle.
pub fn deq_row(ty: Ty, w: &[u8], out: &mut [f32]) {
    match ty {
        Ty::Q8_0 => {
            for (blk, o) in w.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
                let d = f16(blk);
                for i in 0..32 {
                    o[i] = d * blk[2 + i] as i8 as f32;
                }
            }
        }
        Ty::Iq4Nl => {
            for (blk, o) in w.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
                let d = f16(blk);
                for i in 0..16 {
                    o[i] = d * KVALUES_IQ4NL[(blk[2 + i] & 15) as usize] as f32;
                    o[16 + i] = d * KVALUES_IQ4NL[(blk[2 + i] >> 4) as usize] as f32;
                }
            }
        }
        Ty::Iq4Xs => {
            for (blk, o) in w.chunks_exact(136).zip(out.chunks_exact_mut(256)) {
                let d = f16(blk);
                let sh = u16::from_le_bytes([blk[2], blk[3]]);
                for ib in 0..8 {
                    let ls = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 15) as i32 | (((sh >> (2 * ib)) & 3) as i32) << 4;
                    let dl = d * (ls - 32) as f32;
                    for i in 0..16 {
                        let q = blk[8 + 16 * ib + i];
                        o[32 * ib + i] = dl * KVALUES_IQ4NL[(q & 15) as usize] as f32;
                        o[32 * ib + 16 + i] = dl * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
                    }
                }
            }
        }
        Ty::Iq3S => {
            for (blk, o) in w.chunks_exact(110).zip(out.chunks_exact_mut(256)) {
                let d = f16(blk);
                let mut q = [0i8; 32];
                for ib in 0..8 {
                    let sc = iq3s_sub(blk, ib, &mut q);
                    for i in 0..32 {
                        o[32 * ib + i] = d * sc as f32 * q[i] as f32;
                    }
                }
            }
        }
        Ty::F32 => {
            for (c, o) in w.chunks_exact(4).zip(out.iter_mut()) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        t => panic!("no CPU decoder for {}", t.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(s: &mut u32) -> u32 {
        *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *s >> 8
    }

    /// The fast kernels give the scalar sums, and both match dequantized
    /// weights against the quantized activations, for every type.
    #[test]
    fn rows_match_dequant() {
        let n = 2560;
        let mut s = 5u32;
        let x: Vec<f32> = (0..n).map(|_| (rnd(&mut s) % 2001) as f32 / 1000.0 - 1.0).collect();
        let q = Q8::new(&x);
        let xd: Vec<f32> = (0..n).map(|i| q.q[i] as f32 * q.d[i / 32]).collect();
        for ty in [Ty::Q8_0, Ty::Iq4Nl, Ty::Iq4Xs, Ty::Iq3S] {
            let mut w = vec![0u8; ty.bytes(n)];
            let bb = ty.block().1;
            for blk in w.chunks_mut(bb) {
                for v in blk.iter_mut() {
                    *v = rnd(&mut s) as u8;
                }
                // a small positive f16 scale
                blk[0..2].copy_from_slice(&0x2400u16.to_le_bytes());
            }
            let mut wd = vec![0f32; n];
            deq_row(ty, &w, &mut wd);
            let want: f32 = wd.iter().zip(&xd).map(|(a, b)| a * b).sum();
            let sc = row_scalar(ty, &w, &q);
            let fast = row(ty, &w, &q);
            let tol = 1e-3 * want.abs().max(1.0);
            assert!((sc - want).abs() < tol, "{} scalar {sc} vs {want}", ty.name());
            assert!((fast - want).abs() < tol, "{} fast {fast} vs {want}", ty.name());
        }
    }

    /// Decode's CPU experts: 288 different Qwen3.8 experts (IQ3_S gate/up,
    /// IQ4_NL down) for one token each, side by side. `cargo test --release
    /// bench_experts -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_experts() {
        use rayon::prelude::*;
        let (d, ff, n) = (2560usize, 640usize, 288usize);
        let tys = [Ty::Iq3S, Ty::Iq3S, Ty::Iq4Nl];
        let (gb, db) = (ff * tys[0].bytes(d), d * tys[2].bytes(ff));
        let mut s = 7u32;
        let mk = |len: usize, bb: usize, s: &mut u32| -> Vec<u8> {
            let mut w: Vec<u8> = (0..len).map(|_| rnd(s) as u8).collect();
            for blk in w.chunks_mut(bb) {
                blk[0..2].copy_from_slice(&0x2400u16.to_le_bytes());
            }
            w
        };
        let ws: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..n).map(|_| (mk(gb, 110, &mut s), mk(gb, 110, &mut s), mk(db, 18, &mut s))).collect();
        let x: Vec<f32> = (0..d).map(|_| (rnd(&mut s) % 2001) as f32 / 1000.0 - 1.0).collect();
        let q = Q8::new(&x);
        let bytes = n * (2 * gb + db);
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let sum: f32 = ws.par_iter().map(|(g, u, dw)| expert(tys, g, u, dw, d, ff, &[&q], &[0.1])[0]).sum();
            let el = t.elapsed().as_secs_f64();
            println!("{n} experts in {:.1} ms ({:.1} GB/s) [{sum}]", el * 1e3, bytes as f64 / el / 1e9);
        }
        // the two kinds of rows alone
        for (name, ty, len, nr, cols) in [("IQ3_S gate", tys[0], gb, ff, d), ("IQ4_NL down", tys[2], db, d, ff)] {
            let xq = Q8::new(&x[..cols]);
            let t = std::time::Instant::now();
            let sum: f32 = ws
                .par_iter()
                .map(|(g, _, dw)| {
                    let mut o = vec![0f32; nr];
                    rows(ty, if name.starts_with("IQ3") { g } else { dw }, &[&xq], &mut o);
                    o[0]
                })
                .sum();
            let el = t.elapsed().as_secs_f64();
            println!("{name}: {:.1} ms ({:.1} GB/s) [{sum}]", el * 1e3, (n * len) as f64 / el / 1e9);
        }
        // as decode runs them: about 6 per layer
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let mut sum = 0f32;
            for c in ws.chunks(6) {
                sum += c.par_iter().map(|(g, u, dw)| expert(tys, g, u, dw, d, ff, &[&q], &[0.1])[0]).sum::<f32>();
            }
            let el = t.elapsed().as_secs_f64();
            println!("6 at a time: {:.1} ms ({:.1} GB/s) [{sum}]", el * 1e3, bytes as f64 / el / 1e9);
        }
    }
}
