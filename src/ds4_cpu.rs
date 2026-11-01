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
    // block sums stay in lanes (scaled as floats) and are added up once per row
    let mut lanes = _mm512_setzero_ps();
    let mut mins = 0f32;
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
        lanes = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(x.d[b] * d), lanes);
        let mut msum = 0i32;
        for j in 0..8 {
            msum += m[j] * x.bsum[b * 8 + j];
        }
        mins += x.d[b] * dmin * msum as f32;
    }
    _mm512_reduce_add_ps(lanes) - mins
}

/// row_vnni for CPUs with AVX2 but no AVX-512 (older laptops running
/// `pelica bridge`): the same integer sums, 8 lanes instead of 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_avx2(w: &[u8], x: &Q8x) -> f32 {
    use std::arch::x86_64::*;
    let mask = _mm256_set1_epi8(15);
    let ones = _mm256_set1_epi16(1);
    let mut lanes = _mm256_setzero_ps();
    let mut mins = 0f32;
    for (b, blk) in w.chunks_exact(144).enumerate() {
        let (d, dmin) = (f16(&blk[0..2]), f16(&blk[2..4]));
        let (sc, m) = scales(&blk[4..16]);
        let qs = blk.as_ptr().add(16);
        let xq = x.q.as_ptr().add(b * 256);
        let mut acc = _mm256_setzero_si256();
        for c in 0..4 {
            let q = _mm256_loadu_si256(qs.add(32 * c) as *const __m256i);
            let lo = _mm256_and_si256(q, mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), mask);
            let xlo = _mm256_loadu_si256(xq.add(64 * c) as *const __m256i);
            let xhi = _mm256_loadu_si256(xq.add(64 * c + 32) as *const __m256i);
            let slo = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, xlo), ones);
            let shi = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, xhi), ones);
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(slo, _mm256_set1_epi32(sc[2 * c])));
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(shi, _mm256_set1_epi32(sc[2 * c + 1])));
        }
        lanes = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), _mm256_set1_ps(x.d[b] * d), lanes);
        let mut msum = 0i32;
        for j in 0..8 {
            msum += m[j] * x.bsum[b * 8 + j];
        }
        mins += x.d[b] * dmin * msum as f32;
    }
    let h = _mm_add_ps(_mm256_castps256_ps128(lanes), _mm256_extractf128_ps(lanes, 1));
    let h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    let h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    _mm_cvtss_f32(h) - mins
}

fn avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *OK.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// The fastest kernel this CPU has, for display ("AVX-512 VNNI", "AVX2", "scalar").
pub fn kernel_name() -> &'static str {
    if vnni() {
        "AVX-512 VNNI"
    } else if avx2() {
        "AVX2"
    } else {
        "scalar"
    }
}

/// One row with the fastest kernel this CPU has.
#[inline]
fn row(w: &[u8], x: &Q8x) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if vnni() {
            // SAFETY: the features were detected
            return unsafe { row_vnni(w, x) };
        }
        if avx2() {
            // SAFETY: the features were detected
            return unsafe { row_avx2(w, x) };
        }
    }
    row_scalar(w, x)
}

/// tokens one pass over a row serves (row_vnni_multi)
const MULTI: usize = 16;

/// row_vnni for up to MULTI activations at once: each block's weights are
/// unpacked once and used for every token; each token's sum is worked out
/// exactly as row_vnni does it, so the results are the same.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vl")]
unsafe fn row_vnni_multi(w: &[u8], xs: &[&Q8x], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let mask = _mm256_set1_epi8(15);
    let nt = xs.len().min(MULTI);
    let mut lanes = [_mm512_setzero_ps(); MULTI];
    let mut mins = [0f32; MULTI];
    for (b, blk) in w.chunks_exact(144).enumerate() {
        let (d, dmin) = (f16(&blk[0..2]), f16(&blk[2..4]));
        let (sc, m) = scales(&blk[4..16]);
        let qs = blk.as_ptr().add(16);
        let mut qv = [_mm512_setzero_si512(); 4];
        let mut sv = [_mm512_setzero_si512(); 4];
        for c in 0..4 {
            let q = _mm256_loadu_si256(qs.add(32 * c) as *const __m256i);
            let lo = _mm256_and_si256(q, mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), mask);
            qv[c] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            sv[c] = _mm512_mask_blend_epi32(0xFF00, _mm512_set1_epi32(sc[2 * c]), _mm512_set1_epi32(sc[2 * c + 1]));
        }
        for t in 0..nt {
            let x = xs[t];
            let xq = x.q.as_ptr().add(b * 256);
            let mut acc = _mm512_setzero_si512();
            for c in 0..4 {
                let xv = _mm512_loadu_si512(xq.add(64 * c) as *const __m512i);
                let dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), qv[c], xv);
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dp, sv[c]));
            }
            lanes[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_set1_ps(x.d[b] * d), lanes[t]);
            let mut msum = 0i32;
            for j in 0..8 {
                msum += m[j] * x.bsum[b * 8 + j];
            }
            mins[t] += x.d[b] * dmin * msum as f32;
        }
    }
    for t in 0..nt {
        out[t] = _mm512_reduce_add_ps(lanes[t]) - mins[t];
    }
}

/// out[r * xs.len() + t] = row r of Q4_K matrix `w` . xs[t], for out.len() /
/// xs.len() rows: the weights are read once for all the activations.
pub fn q4k_rows_multi(w: &[u8], xs: &[&Q8x], out: &mut [f32]) {
    let nt = xs.len();
    if nt == 0 {
        return;
    }
    let bpr = xs[0].d.len() * 144;
    let fast = vnni();
    for (r, o) in out.chunks_mut(nt).enumerate() {
        let row = &w[r * bpr..(r + 1) * bpr];
        for (g, og) in xs.chunks(MULTI).zip(o.chunks_mut(MULTI)) {
            #[cfg(target_arch = "x86_64")]
            {
                if fast {
                    // SAFETY: the features were detected above
                    unsafe { row_vnni_multi(row, g, og) };
                    continue;
                }
            }
            for (x, v) in g.iter().zip(og.iter_mut()) {
                *v = self::row(row, x);
            }
        }
    }
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
    for (r, o) in out.iter_mut().enumerate() {
        *o = row(&w[r * bpr..(r + 1) * bpr], x);
    }
}

/// One Q4_K expert for some tokens: silu(min(gate x, lim)) * clamp(up x)
/// * the token's routing weight, through down. `gw`/`uw` are ff rows of
/// d values, `dw` d rows of ff; returns [token][d]. The engine's CPU experts
/// and `pelica bridge` both use this, so the two give the same sums.
pub fn expert(gw: &[u8], uw: &[u8], dw: &[u8], d: usize, ff: usize, xs: &[&Q8x], wts: &[f32], lim: f32) -> Vec<f32> {
    use rayon::prelude::*;
    let nt = xs.len();
    let (bpr_g, bpr_d) = (d / 256 * 144, ff / 256 * 144);
    // gate rows then up rows, in one pass across threads; [row][token]
    let mut gu = vec![0f32; 2 * ff * nt];
    let gch = ff / 64;
    debug_assert_eq!(ff % 64, 0);
    gu.par_chunks_mut(64 * nt).enumerate().for_each(|(ci, o)| if ci < gch { q4k_rows_multi(&gw[ci * 64 * bpr_g..], xs, o) } else { q4k_rows_multi(&uw[(ci - gch) * 64 * bpr_g..], xs, o) });
    let hq: Vec<Q8x> = (0..nt)
        .map(|t| {
            let h: Vec<f32> = (0..ff)
                .map(|j| {
                    let (gv, uv) = (gu[j * nt + t], gu[(ff + j) * nt + t]);
                    let (gv, uv) = if lim > 0.0 { (gv.min(lim), uv.clamp(-lim, lim)) } else { (gv, uv) };
                    gv / (1.0 + (-gv).exp()) * uv * wts[t]
                })
                .collect();
            Q8x::new(&h)
        })
        .collect();
    let hs: Vec<&Q8x> = hq.iter().collect();
    let mut o = vec![0f32; d * nt];
    o.par_chunks_mut(64 * nt).enumerate().for_each(|(ci, oc)| q4k_rows_multi(&dw[ci * 64 * bpr_d..], &hs, oc));
    let mut out = vec![0f32; nt * d];
    for t in 0..nt {
        for r in 0..d {
            out[t * d + r] = o[r * nt + t];
        }
    }
    out
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

    /// One-token expert speed at DeepSeek V4 Flash's shape (d 4096, ff 2048,
    /// Q4_K): GB/s of weights, over 8 experts so they don't sit in cache.
    /// `cargo test --release expert_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn expert_speed() {
        let (d, ff) = (4096usize, 2048usize);
        let (bpr_g, bpr_d) = (d / 256 * 144, ff / 256 * 144);
        let mut s = 5u32;
        let mut mk = |n: usize| -> Vec<u8> {
            let mut w = vec![0u8; n];
            for blk in w.chunks_mut(144) {
                blk[0..2].copy_from_slice(&0x1c00u16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x1800u16.to_le_bytes());
                for v in blk[4..].iter_mut() {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    *v = (s >> 24) as u8;
                }
            }
            w
        };
        let ex: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = (0..8).map(|_| (mk(ff * bpr_g), mk(ff * bpr_g), mk(d * bpr_d))).collect();
        let bytes = (2 * ff * bpr_g + d * bpr_d) as f64;
        for nt in [1usize, 2, 4] {
            let xs: Vec<Q8x> = (0..nt).map(|t| Q8x::new(&(0..d).map(|i| ((i * 7 + t) % 13) as f32 * 0.05 - 0.3).collect::<Vec<_>>())).collect();
            let xr: Vec<&Q8x> = xs.iter().collect();
            let wts = vec![0.5f32; nt];
            let t0 = std::time::Instant::now();
            let n = 160;
            for i in 0..n {
                let (g, u, dn) = &ex[i % 8];
                std::hint::black_box(expert(g, u, dn, d, ff, &xr, &wts, 10.0));
            }
            let per = t0.elapsed().as_secs_f64() / n as f64;
            eprintln!("expert, {nt} token(s): {:.0} us, {:.1} GB/s of weights ({})", per * 1e6, bytes / per / 1e9, kernel_name());
        }
    }

    /// What this PC's RAM gives one pass over 1 GB across all threads, and
    /// the expert kernel's rate when several experts run side by side (one
    /// thread each) instead of one after another across all threads.
    /// `cargo test --release mem_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn mem_speed() {
        use rayon::prelude::*;
        let v = vec![1u64; 1 << 27];
        for _ in 0..2 {
            let t0 = std::time::Instant::now();
            let s: u64 = v.par_chunks(1 << 16).map(|c| c.iter().fold(0u64, |a, &b| a.wrapping_add(b))).sum();
            std::hint::black_box(s);
            eprintln!("RAM read: {:.1} GB/s", (v.len() * 8) as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
        let (d, ff) = (4096usize, 2048usize);
        let (bpr_g, bpr_d) = (d / 256 * 144, ff / 256 * 144);
        let ex: Vec<Vec<u8>> = (0..24).map(|i| vec![(i * 37) as u8; 2 * ff * bpr_g + d * bpr_d]).collect();
        let x = Q8x::new(&(0..d).map(|i| (i % 13) as f32 * 0.05 - 0.3).collect::<Vec<_>>());
        let xs = [&x];
        for round in 0..2 {
            let t0 = std::time::Instant::now();
            for e in &ex {
                let (g, rest) = e.split_at(ff * bpr_g);
                let (u, dn) = rest.split_at(ff * bpr_g);
                std::hint::black_box(expert(g, u, dn, d, ff, &xs, &[0.5], 10.0));
            }
            let a = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            ex.par_iter().for_each(|e| {
                let (g, rest) = e.split_at(ff * bpr_g);
                let (u, dn) = rest.split_at(ff * bpr_g);
                let mut gu = vec![0f32; 2 * ff];
                q4k_rows_multi(g, &xs, &mut gu[..ff]);
                q4k_rows_multi(u, &xs, &mut gu[ff..]);
                let h: Vec<f32> = (0..ff).map(|j| gu[j] / (1.0 + (-gu[j]).exp()) * gu[ff + j]).collect();
                let hq = Q8x::new(&h);
                let mut o = vec![0f32; d];
                q4k_rows_multi(dn, &[&hq], &mut o);
                std::hint::black_box(o);
            });
            let b = t0.elapsed().as_secs_f64();
            let gb = (ex.len() * ex[0].len()) as f64 / 1e9;
            let mut nested = Vec::new();
            for k in [2usize, 3, 4, 6] {
                let t0 = std::time::Instant::now();
                ex.chunks(k).for_each(|grp| {
                    grp.par_iter().for_each(|e| {
                        let (g, rest) = e.split_at(ff * bpr_g);
                        let (u, dn) = rest.split_at(ff * bpr_g);
                        std::hint::black_box(expert(g, u, dn, d, ff, &xs, &[0.5], 10.0));
                    })
                });
                nested.push(format!("{k} at a time {:.1}", gb / t0.elapsed().as_secs_f64()));
            }
            if round == 1 {
                eprintln!("24 experts: one after another {:.1} GB/s, side by side {:.1} GB/s; nested: {}", gb / a, gb / b, nested.join(", "));
            }
        }
    }

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
        if avx2() {
            let b = unsafe { row_avx2(&w, &q) };
            assert!((a - b).abs() <= 1e-3 * a.abs().max(1.0), "avx2: {a} vs {b}");
        }
    }

    #[test]
    fn multi_matches_single() {
        let mut s = 7u32;
        let mut rnd = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            s
        };
        let (blocks, rows, toks) = (4, 3, 19);
        let mut w = vec![0u8; rows * blocks * 144];
        for blk in w.chunks_mut(144) {
            blk[0..2].copy_from_slice(&0x2c00u16.to_le_bytes());
            blk[2..4].copy_from_slice(&0x2800u16.to_le_bytes());
            for v in blk[4..].iter_mut() {
                *v = rnd() as u8;
            }
        }
        let qs: Vec<Q8x> = (0..toks).map(|_| Q8x::new(&(0..blocks * 256).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect::<Vec<_>>())).collect();
        let xs: Vec<&Q8x> = qs.iter().collect();
        let mut multi = vec![0f32; rows * toks];
        q4k_rows_multi(&w, &xs, &mut multi);
        for (t, q) in qs.iter().enumerate() {
            let mut one = vec![0f32; rows];
            q4k_rows(&w, q, &mut one);
            for r in 0..rows {
                assert_eq!(one[r].to_bits(), multi[r * toks + t].to_bits(), "row {r} token {t}");
            }
        }
    }
}
