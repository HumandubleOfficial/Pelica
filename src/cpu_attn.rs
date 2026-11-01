//! Decode attention on the CPU, for the cache positions held in RAM or on
//! the drive: DDR feeds the CPU several times faster than PCIe feeds the
//! GPU. Computes the same per-chunk partials as `moe_kernels::attn_dec`,
//! bit for bit, by doing exactly its arithmetic: 4 interleaved key streams
//! (its subgroups), lane partial sums (lane i: dims i, i+32, ...) in dim
//! order with fmas, the lanes summed in its xor-shuffle order, its `pexp`,
//! and fmas where it has them. IEEE adds, multiplies and fmas round the same
//! everywhere, so the partials match whatever tier a chunk lives in.

use std::ops::Range;
use crate::moe_kernels::{CHUNK, EXP_C1, EXP_C2, EXP_LOG2E, EXP_MIN, EXP_P};

/// Attention shape: dims per head, kv heads, query heads per kv head.
#[derive(Clone, Copy)]
pub struct Shape {
    pub hd: usize,
    pub n_kv: usize,
    pub hpk: usize,
}

impl Shape {
    /// f16 values per cache row (a position's keys or values)
    pub fn row(&self) -> usize {
        self.n_kv * self.hd
    }
}

/// `moe_kernels`' `pexp`.
#[inline(always)]
pub fn pexp(x: f32) -> f32 {
    if x < EXP_MIN {
        return 0.0;
    }
    let n = x.mul_add(EXP_LOG2E, 0.5).floor();
    let mut r = n.mul_add(-EXP_C1, x);
    r = n.mul_add(-EXP_C2, r);
    let mut p = EXP_P[0];
    for &c in &EXP_P[1..] {
        p = p.mul_add(r, c);
    }
    let mut y = p.mul_add(r * r, r);
    y += 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = match e {
        0 if m == 0 => s,
        0 => {
            // subnormal: normalise
            let mut e = 113u32;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (e << 23) | ((m & 0x3ff) << 13)
        }
        31 => s | 0x7f80_0000 | (m << 13),
        _ => s | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Partials of one chunk for the query heads of kv heads `kv`: keys and
/// values of rows j0..j0+n (`k`, `v`: n rows of `sh.row()` f16), those
/// heads' queries `q` (hd each, already scaled). Writes per head (max, sum,
/// hd sums of v) to `out`.
pub fn chunk(sh: &Shape, kv: Range<usize>, k: &[u16], v: &[u16], q: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c") {
        // SAFETY: the features were just checked
        return unsafe { chunk_avx2(sh, kv, k, v, q, out) };
    }
    chunk_scalar(sh, kv, k, v, q, out)
}

/// Running state of the 4 key streams: per (stream, head) max, sum, sums of v.
struct State {
    m: Vec<f32>,
    l: Vec<f32>,
    o: Vec<f32>,
}

impl State {
    fn new(nh: usize, hd: usize) -> Self {
        State { m: vec![-3.0e38; 4 * nh], l: vec![0.0; 4 * nh], o: vec![0.0; 4 * nh * hd] }
    }

    /// `attn_dec`'s merge of the 4 streams into `out`.
    fn finish(&self, nh: usize, hd: usize, out: &mut [f32]) {
        for h in 0..nh {
            let s = [h, nh + h, 2 * nh + h, 3 * nh + h];
            let m = s.map(|i| self.m[i]);
            let mx = m[0].max(m[1]).max(m[2].max(m[3]));
            let w = m.map(|x| pexp(x - mx));
            let ob = &mut out[h * (hd + 2)..(h + 1) * (hd + 2)];
            ob[0] = mx;
            let mut l = self.l[s[0]] * w[0];
            for t in 1..4 {
                l = self.l[s[t]].mul_add(w[t], l);
            }
            ob[1] = l;
            for i in 0..hd {
                let mut o = self.o[s[0] * hd + i] * w[0];
                for t in 1..4 {
                    o = self.o[s[t] * hd + i].mul_add(w[t], o);
                }
                ob[2 + i] = o;
            }
        }
    }
}

pub fn chunk_scalar(sh: &Shape, kv: Range<usize>, k: &[u16], v: &[u16], q: &[f32], out: &mut [f32]) {
    let (hd, rl, nh) = (sh.hd, sh.row(), kv.len() * sh.hpk);
    let d = hd / 32;
    let n = k.len() / rl;
    debug_assert!(n <= CHUNK);
    let mut st = State::new(nh, hd);
    let (mut kf, mut vf) = (vec![0f32; hd], vec![0f32; hd]);
    for j in 0..n {
        let sg = j % 4;
        for g in kv.clone() {
            let at = j * rl + g * hd;
            for i in 0..hd {
                kf[i] = f16_to_f32(k[at + i]);
                vf[i] = f16_to_f32(v[at + i]);
            }
            for hh in (g - kv.start) * sh.hpk..(g + 1 - kv.start) * sh.hpk {
                let qh = &q[hh * hd..(hh + 1) * hd];
                let mut x = [0f32; 32];
                for (lane, x) in x.iter_mut().enumerate() {
                    let mut a = qh[lane] * kf[lane];
                    for i in 1..d {
                        a = qh[lane + 32 * i].mul_add(kf[lane + 32 * i], a);
                    }
                    *x = a;
                }
                for off in [16, 8, 4, 2, 1] {
                    for i in 0..off {
                        x[i] += x[i + off];
                    }
                }
                let s = sg * nh + hh;
                let mn = st.m[s].max(x[0]);
                let c = pexp(st.m[s] - mn);
                let p = pexp(x[0] - mn);
                st.l[s] = st.l[s].mul_add(c, p);
                let o = &mut st.o[s * hd..(s + 1) * hd];
                for i in 0..hd {
                    o[i] = o[i].mul_add(c, p * vf[i]);
                }
                st.m[s] = mn;
            }
        }
    }
    st.finish(nh, hd, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn chunk_avx2(sh: &Shape, kv: Range<usize>, k: &[u16], v: &[u16], q: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let (hd, rl, nh) = (sh.hd, sh.row(), kv.len() * sh.hpk);
    let (d, nb) = (hd / 32, hd / 8);
    let n = k.len() / rl;
    debug_assert!(n <= CHUNK && v.len() == k.len() && q.len() >= nh * hd && out.len() >= nh * (hd + 2));
    let mut st = State::new(nh, hd);
    let mut kf = vec![0f32; hd];
    let mut vf = vec![0f32; hd];
    let (kp, vp, qp) = (k.as_ptr(), v.as_ptr(), q.as_ptr());
    for j in 0..n {
        let sg = j % 4;
        for g in kv.clone() {
            let at = j * rl + g * hd;
            for b in 0..nb {
                _mm256_storeu_ps(kf.as_mut_ptr().add(8 * b), _mm256_cvtph_ps(_mm_loadu_si128(kp.add(at + 8 * b) as *const __m128i)));
                _mm256_storeu_ps(vf.as_mut_ptr().add(8 * b), _mm256_cvtph_ps(_mm_loadu_si128(vp.add(at + 8 * b) as *const __m128i)));
            }
            for hh in (g - kv.start) * sh.hpk..(g + 1 - kv.start) * sh.hpk {
                let qh = qp.add(hh * hd);
                // lanes 8b..8b+7 of the 32: dims lane + 32i, in i order
                let mut x = [_mm256_setzero_ps(); 4];
                for (b, x) in x.iter_mut().enumerate() {
                    let mut a = _mm256_mul_ps(_mm256_loadu_ps(qh.add(8 * b)), _mm256_loadu_ps(kf.as_ptr().add(8 * b)));
                    for i in 1..d {
                        a = _mm256_fmadd_ps(_mm256_loadu_ps(qh.add(32 * i + 8 * b)), _mm256_loadu_ps(kf.as_ptr().add(32 * i + 8 * b)), a);
                    }
                    *x = a;
                }
                // xor 16, 8, 4, 2, 1
                let b8 = _mm256_add_ps(_mm256_add_ps(x[0], x[2]), _mm256_add_ps(x[1], x[3]));
                let c4 = _mm_add_ps(_mm256_castps256_ps128(b8), _mm256_extractf128_ps(b8, 1));
                let d2 = _mm_add_ps(c4, _mm_movehl_ps(c4, c4));
                let sc = _mm_cvtss_f32(_mm_add_ss(d2, _mm_shuffle_ps(d2, d2, 1)));
                let s = sg * nh + hh;
                let mn = st.m[s].max(sc);
                let c = pexp(st.m[s] - mn);
                let p = pexp(sc - mn);
                st.l[s] = st.l[s].mul_add(c, p);
                let (cv, pv) = (_mm256_set1_ps(c), _mm256_set1_ps(p));
                let o = st.o.as_mut_ptr().add(s * hd);
                for b in 0..nb {
                    let ov = _mm256_loadu_ps(o.add(8 * b));
                    _mm256_storeu_ps(o.add(8 * b), _mm256_fmadd_ps(ov, cv, _mm256_mul_ps(pv, _mm256_loadu_ps(vf.as_ptr().add(8 * b)))));
                }
                st.m[s] = mn;
            }
        }
    }
    st.finish(nh, hd, out);
}
