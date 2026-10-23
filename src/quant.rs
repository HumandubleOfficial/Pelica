//! Weight formats and the matrix-vector kernels that read them.
//!
//! Layout follows transformers: a matrix is [rows = outputs, cols = inputs],
//! and `matmul` computes out[t, o] = sum_i W[o, i] * x[t, i].
//!
//! Formats:
//! * F32 — exact.
//! * Q8  — int8 per row, one f32 scale per row (dense weights by default).
//! * Q4  — int4 in groups of 64 columns: two values per byte, low nibble first,
//!         stored as level + 8, one f32 scale per group. This is the layout of
//!         the experts in Colibri's int4 container, so they are used in place.
//! * Q4T — the same int4 weights with each group's scale given as a one-byte
//!         index into a 256-entry table (see `scalepack`): identical values,
//!         a quarter of the scale bytes.

use rayon::prelude::*;

pub const Q4_GROUP: usize = 64;

pub enum Mat {
    F32 { w: Vec<f32>, rows: usize, cols: usize },
    Q8 { q: Vec<i8>, s: Vec<f32>, rows: usize, cols: usize },
    Q4 { q: Vec<u8>, s: Vec<f32>, rows: usize, cols: usize },
}

#[derive(Clone, Copy)]
pub enum Kind<'a> {
    F32(&'a [f32]),
    Q8(&'a [i8], &'a [f32]),
    Q4(&'a [u8], &'a [f32]),
    /// weights, per-group table index, table
    Q4T(&'a [u8], &'a [u8], &'a [f32]),
}

/// A borrowed view of a matrix (owned dense weights or a cached expert).
#[derive(Clone, Copy)]
pub struct MRef<'a> {
    pub kind: Kind<'a>,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    /// Quantize an f32 matrix to `bits` (32, 8 or 4). Int4 needs the column
    /// count to be a multiple of 64; otherwise it falls back to int8.
    pub fn from_f32(w: Vec<f32>, rows: usize, cols: usize, bits: u32) -> Mat {
        assert_eq!(w.len(), rows * cols);
        match bits {
            32 => Mat::F32 { w, rows, cols },
            4 if cols % Q4_GROUP == 0 => {
                let (q, s) = quantize_q4(&w, rows, cols);
                Mat::Q4 { q, s, rows, cols }
            }
            _ => {
                let (q, s) = quantize_q8(&w, rows, cols);
                Mat::Q8 { q, s, rows, cols }
            }
        }
    }

    pub fn r(&self) -> MRef<'_> {
        match self {
            Mat::F32 { w, rows, cols } => MRef { kind: Kind::F32(w), rows: *rows, cols: *cols },
            Mat::Q8 { q, s, rows, cols } => MRef { kind: Kind::Q8(q, s), rows: *rows, cols: *cols },
            Mat::Q4 { q, s, rows, cols } => MRef { kind: Kind::Q4(q, s), rows: *rows, cols: *cols },
        }
    }

    pub fn rows(&self) -> usize {
        self.r().rows
    }

    pub fn bytes(&self) -> usize {
        match self {
            Mat::F32 { w, .. } => w.len() * 4,
            Mat::Q8 { q, s, .. } => q.len() + s.len() * 4,
            Mat::Q4 { q, s, .. } => q.len() + s.len() * 4,
        }
    }
}

impl<'a> MRef<'a> {
    /// Rows [r0, r0 + n) as their own matrix.
    pub fn rows_range(&self, r0: usize, n: usize) -> MRef<'a> {
        assert!(r0 + n <= self.rows);
        let c = self.cols;
        let kind = match self.kind {
            Kind::F32(w) => Kind::F32(&w[r0 * c..(r0 + n) * c]),
            Kind::Q8(q, s) => Kind::Q8(&q[r0 * c..(r0 + n) * c], &s[r0..r0 + n]),
            Kind::Q4(q, s) => {
                let (rb, g) = (c / 2, c / Q4_GROUP);
                Kind::Q4(&q[r0 * rb..(r0 + n) * rb], &s[r0 * g..(r0 + n) * g])
            }
            Kind::Q4T(q, i, t) => {
                let (rb, g) = (c / 2, c / Q4_GROUP);
                Kind::Q4T(&q[r0 * rb..(r0 + n) * rb], &i[r0 * g..(r0 + n) * g], t)
            }
        };
        MRef { kind, rows: n, cols: c }
    }

    #[inline]
    pub fn dot_row(&self, o: usize, x: &[f32]) -> f32 {
        let c = self.cols;
        match self.kind {
            Kind::F32(w) => dot_f32(&w[o * c..(o + 1) * c], &x[..c]),
            Kind::Q8(q, s) => dot_q8(&q[o * c..(o + 1) * c], &x[..c]) * s[o],
            Kind::Q4(q, s) => {
                let (rb, g) = (c / 2, c / Q4_GROUP);
                dot_q4(&q[o * rb..(o + 1) * rb], &s[o * g..(o + 1) * g], &x[..c])
            }
            Kind::Q4T(q, i, t) => {
                let (rb, g) = (c / 2, c / Q4_GROUP);
                dot_q4t(&q[o * rb..(o + 1) * rb], &i[o * g..(o + 1) * g], t, &x[..c])
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}
impl SyncPtr {
    #[inline]
    unsafe fn write(&self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

/// out[t * rows + o] = W[o] . x[t * cols ..] for t in 0..s.
pub fn matmul(out: &mut [f32], w: MRef, x: &[f32], s: usize) {
    let (rows, cols) = (w.rows, w.cols);
    assert!(out.len() >= s * rows, "matmul: output too small");
    assert!(x.len() >= s * cols, "matmul: input too small");
    let p = SyncPtr(out.as_mut_ptr());
    // Each task owns whole output rows, so every element has one writer.
    let min_len = (rows / (rayon::current_num_threads() * 4)).clamp(1, 256);
    (0..rows).into_par_iter().with_min_len(min_len).for_each(|o| {
        for t in 0..s {
            let v = w.dot_row(o, &x[t * cols..(t + 1) * cols]);
            unsafe { p.write(t * rows + o, v) };
        }
    });
}

/// Single-threaded matmul, for use inside an outer parallel loop.
pub fn matmul_serial(out: &mut [f32], w: MRef, x: &[f32], s: usize) {
    let (rows, cols) = (w.rows, w.cols);
    for t in 0..s {
        for o in 0..rows {
            out[t * rows + o] = w.dot_row(o, &x[t * cols..(t + 1) * cols]);
        }
    }
}

// ---------------------------------------------------------------- quantizers

/// Int8 per row: scale = amax / 127, round half to even (matches C lrintf).
pub fn quantize_q8(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut s = vec![0f32; rows];
    q.par_chunks_mut(cols).zip(s.par_iter_mut()).enumerate().for_each(|(r, (qr, sr))| {
        let row = &w[r * cols..(r + 1) * cols];
        let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let step = (amax / 127.0).max(1e-8);
        *sr = step;
        for (d, v) in qr.iter_mut().zip(row) {
            *d = (v / step).round_ties_even().clamp(-128.0, 127.0) as i8;
        }
    });
    (q, s)
}

/// Int4 in groups of 64: scale = amax / 7, levels in [-8, 7] stored + 8.
pub fn quantize_q4(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let (rb, g) = (cols / 2, cols / Q4_GROUP);
    let mut q = vec![0u8; rows * rb];
    let mut s = vec![0f32; rows * g];
    q.par_chunks_mut(rb).zip(s.par_chunks_mut(g)).enumerate().for_each(|(r, (qr, sr))| {
        let row = &w[r * cols..(r + 1) * cols];
        for gi in 0..g {
            let grp = &row[gi * Q4_GROUP..(gi + 1) * Q4_GROUP];
            let amax = grp.iter().fold(0f32, |m, v| m.max(v.abs()));
            let step = (amax / 7.0).max(1e-8);
            sr[gi] = step;
            for (j, v) in grp.iter().enumerate() {
                let level = (v / step).round_ties_even().clamp(-8.0, 7.0) as i32;
                let nib = (level + 8) as u8;
                let c = gi * Q4_GROUP + j;
                qr[c >> 1] |= if c & 1 == 1 { nib << 4 } else { nib };
            }
        }
    });
    (q, s)
}

// ------------------------------------------------------------------- kernels

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
mod simd {
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[inline]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for k in 0..4 {
                acc[k] = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8 * k)), _mm256_loadu_ps(pb.add(i + 8 * k)), acc[k]);
            }
            i += 32;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc[0]);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    /// `dot_f32` with `a` given as bf16 bits: the same lanes, accumulators
    /// and order, and bf16 -> f32 is exact, so the result is bit-identical.
    pub unsafe fn dot_bf16(a: &[u16], b: &[f32]) -> f32 {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let load = |p: *const u16| _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(p as *const __m128i)), 16));
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i + 32 <= n {
            for k in 0..4 {
                acc[k] = _mm256_fmadd_ps(load(pa.add(i + 8 * k)), _mm256_loadu_ps(pb.add(i + 8 * k)), acc[k]);
            }
            i += 32;
        }
        while i + 8 <= n {
            acc[0] = _mm256_fmadd_ps(load(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc[0]);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
        while i < n {
            s += f32::from_bits((a[i] as u32) << 16) * b[i];
            i += 1;
        }
        s
    }

    #[inline]
    pub unsafe fn dot_q8(w: &[i8], x: &[f32]) -> f32 {
        let n = w.len();
        let (pw, px) = (w.as_ptr(), x.as_ptr());
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let v = _mm_loadu_si128(pw.add(i) as *const __m128i);
            let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
            let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
            a0 = _mm256_fmadd_ps(lo, _mm256_loadu_ps(px.add(i)), a0);
            a1 = _mm256_fmadd_ps(hi, _mm256_loadu_ps(px.add(i + 8)), a1);
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(a0, a1));
        while i < n {
            s += w[i] as f32 * x[i];
            i += 1;
        }
        s
    }

    /// 16 unsigned nibble values (already unpacked to bytes) times x, minus 8.
    #[inline]
    unsafe fn fma16(nib: __m128i, x: *const f32, acc: __m256) -> __m256 {
        let eight = _mm256_set1_epi32(8);
        let w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nib), eight));
        let w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8)), eight));
        let acc = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x), acc);
        _mm256_fmadd_ps(w1, _mm256_loadu_ps(x.add(8)), acc)
    }

    #[inline]
    pub unsafe fn dot_q4(q: &[u8], s: &[f32], x: &[f32]) -> f32 {
        dot_q4_by(q, s.len(), |g| *s.get_unchecked(g), x)
    }

    #[inline]
    pub unsafe fn dot_q4t(q: &[u8], idx: &[u8], table: &[f32], x: &[f32]) -> f32 {
        dot_q4_by(q, idx.len(), |g| *table.get_unchecked(*idx.get_unchecked(g) as usize), x)
    }

    /// int4 row times x, with group g's scale from `scale(g)`.
    #[inline]
    unsafe fn dot_q4_by(q: &[u8], groups: usize, scale: impl Fn(usize) -> f32, x: &[f32]) -> f32 {
        let mask = _mm256_set1_epi8(0x0f);
        let mut total = _mm256_setzero_ps();
        for g in 0..groups {
            let scale = scale(g);
            let b = _mm256_loadu_si256(q.as_ptr().add(g * 32) as *const __m256i);
            let lo = _mm256_and_si256(b, mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), mask);
            // unpack stays inside 128-bit lanes: il0 = cols 0..15 | 32..47, il1 = 16..31 | 48..63
            let il0 = _mm256_unpacklo_epi8(lo, hi);
            let il1 = _mm256_unpackhi_epi8(lo, hi);
            let xg = x.as_ptr().add(g * 64);
            let mut acc = _mm256_setzero_ps();
            acc = fma16(_mm256_castsi256_si128(il0), xg, acc);
            acc = fma16(_mm256_castsi256_si128(il1), xg.add(16), acc);
            acc = fma16(_mm256_extracti128_si256(il0, 1), xg.add(32), acc);
            acc = fma16(_mm256_extracti128_si256(il1, 1), xg.add(48), acc);
            total = _mm256_fmadd_ps(acc, _mm256_set1_ps(scale), total);
        }
        hsum(total)
    }
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        simd::dot_f32(a, b)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

/// Dot product of bf16 weights (as bits) with f32 activations; equal to
/// `dot_f32` on the weights widened to f32.
pub fn dot_bf16(a: &[u16], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        simd::dot_bf16(a, b)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        a.iter().zip(b).map(|(x, y)| f32::from_bits((*x as u32) << 16) * y).sum()
    }
}

#[inline]
fn dot_q8(w: &[i8], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        simd::dot_q8(w, x)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        w.iter().zip(x).map(|(a, b)| *a as f32 * b).sum()
    }
}

#[inline]
fn dot_q4(q: &[u8], s: &[f32], x: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        simd::dot_q4(q, s, x)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        dot_q4_scalar(q, s, x)
    }
}

#[inline]
fn dot_q4t(q: &[u8], idx: &[u8], table: &[f32], x: &[f32]) -> f32 {
    assert!(table.len() >= 256);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        simd::dot_q4t(q, idx, table, x)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        let s: Vec<f32> = idx.iter().map(|&i| table[i as usize]).collect();
        dot_q4_scalar(q, &s, x)
    }
}

#[allow(dead_code)]
pub fn dot_q4_scalar(q: &[u8], s: &[f32], x: &[f32]) -> f32 {
    let mut total = 0f32;
    for (g, scale) in s.iter().enumerate() {
        let mut acc = 0f32;
        for j in 0..32 {
            let b = q[g * 32 + j];
            let c = g * 64 + 2 * j;
            acc += ((b & 0xf) as i32 - 8) as f32 * x[c] + ((b >> 4) as i32 - 8) as f32 * x[c + 1];
        }
        total += acc * scale;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn q4_kernel_matches_scalar() {
        let (rows, cols) = (5, 256);
        let w = rand_vec(rows * cols, 1);
        let x = rand_vec(cols, 2);
        let m = Mat::from_f32(w.clone(), rows, cols, 4);
        let r = m.r();
        let Kind::Q4(q, s) = r.kind else { panic!() };
        for o in 0..rows {
            let fast = r.dot_row(o, &x);
            let slow = dot_q4_scalar(&q[o * cols / 2..(o + 1) * cols / 2], &s[o * 4..(o + 1) * 4], &x);
            assert!((fast - slow).abs() < 1e-4, "{fast} vs {slow}");
            let exact: f32 = w[o * cols..(o + 1) * cols].iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((fast - exact).abs() < 0.2, "quantization error too large: {fast} vs {exact}");
        }
    }

    #[test]
    fn q8_and_f32_kernels() {
        let (rows, cols) = (7, 203);
        let w = rand_vec(rows * cols, 3);
        let x = rand_vec(cols * 2, 4);
        let exact: Vec<f32> = (0..2)
            .flat_map(|t| {
                let (w, x) = (&w, &x);
                (0..rows).map(move |o| w[o * cols..(o + 1) * cols].iter().zip(&x[t * cols..]).map(|(a, b)| a * b).sum())
            })
            .collect();
        for bits in [32, 8] {
            let m = Mat::from_f32(w.clone(), rows, cols, bits);
            let mut out = vec![0f32; rows * 2];
            matmul(&mut out, m.r(), &x, 2);
            for (a, b) in out.iter().zip(&exact) {
                assert!((a - b).abs() < if bits == 32 { 1e-4 } else { 0.05 }, "bits {bits}: {a} vs {b}");
            }
        }
    }
}

#[cfg(test)]
mod bf16_tests {
    use super::*;

    #[test]
    fn dot_bf16_matches_dot_f32_bitwise() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        };
        for n in [4096usize, 4095, 37, 8, 3] {
            for _ in 0..50 {
                let w: Vec<u16> = (0..n).map(|_| (rnd().to_bits() >> 16) as u16).collect();
                let wf: Vec<f32> = w.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
                let x: Vec<f32> = (0..n).map(|_| rnd() * 3.0).collect();
                assert_eq!(dot_bf16(&w, &x).to_bits(), dot_f32(&wf, &x).to_bits(), "n = {n}");
            }
        }
    }
}
