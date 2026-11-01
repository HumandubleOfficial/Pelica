//! The GLM5-Next forward pass across the three tiers.
//!
//! Per layer and per site (attention, then FFN) the four mHC residual streams
//! are collapsed into one input, the block runs, and its output is mixed back
//! into the streams. Attention is either KDA (gated delta rule, constant-size
//! state) or MLA over a cache of 512-wide latents. The FFN is dense for the
//! first layers and a 288-expert MoE with a shared expert after that.
//!
//! Each matrix runs where it lives: VRAM-resident weights on the GPU, batched
//! so a block costs one or two GPU round trips; the rest on the CPU. In an MoE
//! layer the GPU (shared expert + experts in VRAM) and the CPU (experts from
//! RAM or SSD) work at the same time.

use crate::config::Config;
use crate::experts::{ExpertBuf, RANKS};
use crate::gpu::{Batch, GMat, Reg, W};
use crate::model::{Attn, Ffn, Hc, Kda, Mla, Mlp, Model};
use crate::prof;
use crate::quant::{dot_bf16, dot_f32, matmul, matmul_serial};
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

pub struct LayerState {
    kda_state: Vec<f32>,  // [heads][k][v]
    kda_window: Vec<f32>, // [3 * channels][kernel]
    latent: Vec<f32>,     // [cap][kv_lora]
    /// the same state in VRAM when the layer runs on the GPU:
    /// KDA [state | window], MLA [cap][kv_lora]
    gpu: Option<wgpu::Buffer>,
}

pub struct Session {
    layers: Vec<LayerState>,
    pub pos: usize,
    pub cap: usize,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct Timing {
    pub attn: f64,
    pub ffn: f64,
    pub hc: f64,
    pub head: f64,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn rms(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * g;
    }
}

fn rms_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for (v, g) in x.iter_mut().zip(w) {
        *v *= inv * g;
    }
}

/// SwiGLU with GLM-5.3's clamp: the gate is capped above, up on both sides.
fn swiglu_clamped(gate: &mut [f32], up: &[f32], limit: f32) {
    for (g, u) in gate.iter_mut().zip(up) {
        *g = silu(g.min(limit)) * u.clamp(-limit, limit);
    }
}

/// gate/up/swiglu/down chained on the GPU; returns the output region.
fn mlp_into(b: &mut Batch, gate: &GMat, up: &GMat, down: &GMat, x: Reg, n: usize, limit: f32) -> Reg {
    let g = b.matvec(gate, x, n);
    let u = b.matvec(up, x, n);
    b.swiglu(g, u, limit);
    b.matvec(down, g, n)
}

/// An expert's FFN on the CPU over its `r` gathered rows.
fn expert_cpu(buf: &ExpertBuf, xg: &[f32], r: usize, inter: usize, d: usize, limit: f32, split_rows: bool) -> Vec<f32> {
    let mut g = vec![0f32; r * inter];
    let mut u = vec![0f32; r * inter];
    let mut y = vec![0f32; r * d];
    if split_rows {
        matmul(&mut g, buf.gate(), xg, r);
        matmul(&mut u, buf.up(), xg, r);
        swiglu_clamped(&mut g, &u, limit);
        matmul(&mut y, buf.down(), &g, r);
    } else {
        matmul_serial(&mut g, buf.gate(), xg, r);
        matmul_serial(&mut u, buf.up(), xg, r);
        swiglu_clamped(&mut g, &u, limit);
        matmul_serial(&mut y, buf.down(), &g, r);
    }
    y
}

/// Collapse the hc streams of one token into the block input.
/// Returns post [hc] and comb [hc*hc] (comb[src * hc + dst]) for hc_post.
/// sigmoid(router . x) for every expert; split over threads when `par`
/// (a single token, where nothing else is running in parallel).
fn router_scores(router: &[u16], xt: &[f32], ne: usize, par: bool) -> Vec<f32> {
    let d = xt.len();
    let one = |e: usize| sigmoid(dot_bf16(&router[e * d..(e + 1) * d], xt));
    if par {
        (0..ne).into_par_iter().with_min_len(16).map(one).collect()
    } else {
        (0..ne).map(one).collect()
    }
}

/// `par`: compute the mixing dot products on several threads (single token).
fn hc_pre(out: &mut [f32], post: &mut [f32], comb: &mut [f32], input: &[f32], hc: &Hc, c: &Config, par: bool) {
    let (m, d) = (c.hc_mult, c.hidden);
    let flat = m * d;
    let mix_n = (2 + m) * m;
    let ms = input.iter().map(|v| v * v).sum::<f32>() / flat as f32;
    let inv = 1.0 / (ms + c.eps).sqrt();
    let mut mix = [0f32; 24];
    if par {
        mix[..mix_n].par_iter_mut().enumerate().for_each(|(r, m)| *m = dot_f32(&hc.fn_[r * flat..(r + 1) * flat], input) * inv);
    } else {
        for r in 0..mix_n {
            mix[r] = dot_f32(&hc.fn_[r * flat..(r + 1) * flat], input) * inv;
        }
    }
    let mut pre = [0f32; 4];
    for i in 0..m {
        pre[i] = sigmoid(mix[i] * hc.scale[0] + hc.base[i]) + c.hc_eps;
        post[i] = 2.0 * sigmoid(mix[m + i] * hc.scale[1] + hc.base[m + i]);
    }
    // comb: row softmax over destinations, then Sinkhorn towards doubly stochastic
    for r in 0..m {
        let mut top = f32::NEG_INFINITY;
        for col in 0..m {
            let k = 2 * m + r * m + col;
            let v = mix[k] * hc.scale[2] + hc.base[k];
            comb[r * m + col] = v;
            top = top.max(v);
        }
        let mut sum = 0.0;
        for col in 0..m {
            let v = (comb[r * m + col] - top).exp();
            comb[r * m + col] = v;
            sum += v;
        }
        for col in 0..m {
            comb[r * m + col] = comb[r * m + col] / sum + c.hc_eps;
        }
    }
    let norm_cols = |comb: &mut [f32]| {
        for col in 0..m {
            let s: f32 = (0..m).map(|r| comb[r * m + col]).sum();
            for r in 0..m {
                comb[r * m + col] /= s + c.hc_eps;
            }
        }
    };
    norm_cols(comb);
    for _ in 1..c.hc_iters {
        for r in 0..m {
            let s: f32 = comb[r * m..(r + 1) * m].iter().sum();
            for col in 0..m {
                comb[r * m + col] /= s + c.hc_eps;
            }
        }
        norm_cols(comb);
    }
    for (j, o) in out.iter_mut().enumerate().take(d) {
        let mut s = 0.0;
        for i in 0..m {
            s += pre[i] * input[i * d + j];
        }
        *o = s;
    }
}

fn hc_post(out: &mut [f32], branch: &[f32], residual: &[f32], post: &[f32], comb: &[f32], m: usize, d: usize) {
    for dst in 0..m {
        for j in 0..d {
            let mut v = 0.0;
            for src in 0..m {
                v += comb[src * m + dst] * residual[src * d + j];
            }
            out[dst * d + j] = v + post[dst] * branch[j];
        }
    }
}

/// Routing trace for studying cache policies: with PELICA_TRACE=<file>, each
/// decode step's picks are appended as `layer e1 e2 ...` lines.
fn trace(tag: &str, layer: usize, picks: &[(usize, f32)]) {
    use std::io::Write;
    static FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let f = FILE.get_or_init(|| {
        let path = std::env::var_os("PELICA_TRACE")?;
        std::fs::OpenOptions::new().create(true).append(true).open(path).ok().map(std::sync::Mutex::new)
    });
    if let Some(f) = f {
        let ids: Vec<String> = picks.iter().map(|p| p.0.to_string()).collect();
        let _ = writeln!(f.lock().unwrap(), "{tag}{layer} {}", ids.join(" "));
    }
}

/// Router-score trace for cache-policy research: with PELICA_SCORES=<file>,
/// each decode step appends layer (u32) then the sigmoid score of all experts
/// and the same plus the selection bias (f32 each).
fn trace_scores(layer: usize, score: &[f32], bias: &[f32]) {
    use std::io::Write;
    static FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let f = FILE.get_or_init(|| {
        let path = std::env::var_os("PELICA_SCORES")?;
        std::fs::OpenOptions::new().create(true).append(true).open(path).ok().map(std::sync::Mutex::new)
    });
    if let Some(f) = f {
        let mut rec = Vec::with_capacity(4 + 8 * score.len());
        rec.extend((layer as u32).to_le_bytes());
        rec.extend(score.iter().flat_map(|v| v.to_le_bytes()));
        rec.extend(score.iter().zip(bias).flat_map(|(v, b)| (v + b).to_le_bytes()));
        let _ = f.lock().unwrap().write_all(&rec);
    }
}

/// A refined guess keeps earlier guesses it ranks this high (picks fall in
/// its top 8 ~75% of the time and in its top 12 ~85%).
const REFINE_KEEP: usize = 12;


/// Promotions to VRAM per MoE layer call: a few while slots are free, then one.
const PROMOTE_FREE: usize = 4;
/// In prefill, an expert with at least this many tokens is streamed to the GPU
/// and run there instead of on the CPU.
const STAGE_MIN_ROWS: usize = 4;

impl Model {
    pub fn new_session(&self, cap: usize) -> Session {
        let c = &self.cfg;
        let pw = c.kda_heads * c.kda_hd;
        let gpu_state = |floats: usize| {
            let g = self.gpu.as_ref().unwrap();
            g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("attention state"),
                size: (floats * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let layers = self
            .layers
            .iter()
            .map(|l| match &l.attn {
                Attn::Mla(a) if a.gpu.is_some() => {
                    LayerState { kda_state: vec![], kda_window: vec![], latent: vec![], gpu: Some(gpu_state(cap * c.kv_lora)) }
                }
                Attn::Mla(_) => LayerState { kda_state: vec![], kda_window: vec![], latent: vec![0.0; cap * c.kv_lora], gpu: None },
                Attn::Kda(k) if k.gpu.is_some() => LayerState {
                    kda_state: vec![],
                    kda_window: vec![],
                    latent: vec![],
                    gpu: Some(gpu_state(c.kda_heads * c.kda_hd * c.kda_hd + 3 * pw * c.conv_k)),
                },
                Attn::Kda(_) => LayerState {
                    kda_state: vec![0.0; c.kda_heads * c.kda_hd * c.kda_hd],
                    kda_window: vec![0.0; 3 * pw * c.conv_k],
                    latent: vec![],
                    gpu: None,
                },
            })
            .collect();
        Session { layers, pos: 0, cap }
    }

    /// W x for `n` tokens on whichever device holds W.
    fn mm(&self, w: &W, x: &[f32], n: usize) -> Vec<f32> {
        match w {
            W::Cpu(m) => {
                let mut o = vec![0f32; n * m.rows()];
                matmul(&mut o, m.r(), x, n);
                o
            }
            W::Gpu(g) => {
                let mut b = self.gpu.as_ref().expect("GPU weight without a GPU").batch();
                let xr = b.input(x);
                let o = b.matvec(g, xr, n);
                b.read(o);
                b.run().pop().unwrap()
            }
        }
    }

    /// A gate/up/down MLP with the clamped SwiGLU.
    fn mlp(&self, m: &Mlp, x: &[f32], n: usize) -> Vec<f32> {
        let limit = self.cfg.swiglu_limit;
        if let (Some(gpu), Some(g), Some(u), Some(dn)) = (&self.gpu, m.gate.gpu(), m.up.gpu(), m.down.gpu()) {
            let mut b = gpu.batch();
            let xr = b.input(x);
            let y = mlp_into(&mut b, g, u, dn, xr, n, limit);
            b.read(y);
            return b.run().pop().unwrap();
        }
        let mut g = self.mm(&m.gate, x, n);
        let u = self.mm(&m.up, x, n);
        swiglu_clamped(&mut g, &u, limit);
        self.mm(&m.down, &g, n)
    }

    /// Run `tokens` at the session's next positions; returns the logits of the last one.
    pub fn forward(&self, s: &mut Session, tokens: &[u32], timing: &mut Timing) -> Result<Vec<f32>> {
        self.forward_rows(s, tokens, timing, 1)
    }

    /// Like `forward`, but returns the logits of the last `want` positions, row after row.
    pub fn forward_rows(&self, s: &mut Session, tokens: &[u32], timing: &mut Timing, want: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (n, m, d) = (tokens.len(), c.hc_mult, c.hidden);
        if s.pos + n > s.cap {
            bail!("context full: {} positions of {}", s.pos + n, s.cap);
        }
        if s.pos + n > c.dense_attention_limit() {
            bail!(
                "context above {} tokens needs the DSA indexer, which is not implemented yet",
                c.dense_attention_limit()
            );
        }

        let mut streams = vec![0f32; n * m * d];
        let mut row = vec![0f32; d];
        for (t, &tok) in tokens.iter().enumerate() {
            if tok as usize >= c.vocab {
                bail!("token {tok} outside the vocabulary");
            }
            self.shards.row_f32(&self.embed, tok as usize, &mut row);
            for h in 0..m {
                streams[(t * m + h) * d..(t * m + h + 1) * d].copy_from_slice(&row);
            }
        }
        // The first MoE layer has no MoE layer before it to look ahead from
        // (see `moe`), and the disk would sit idle across the token boundary:
        // guess its experts from the token's embedding (~40% right) while the
        // dense layers run, then again from the last dense layer's output
        // (~75%), which also drops first guesses that have not started.
        let early = n == 1 && self.lookahead_layers > 0;
        if early {
            self.prefetch_guess(c.first_dense, &streams, false);
        }
        let mut next = vec![0f32; n * m * d];
        let mut collapsed = vec![0f32; n * d];
        let mut normed = vec![0f32; n * d];
        let mut post = vec![0f32; n * m];
        let mut comb = vec![0f32; n * m * m];

        for (i, layer) in self.layers.iter().enumerate() {
            for site in 0..2 {
                let t0 = Instant::now();
                let (hc, ln) = if site == 0 { (&layer.hc_attn, &layer.in_ln) } else { (&layer.hc_ffn, &layer.post_ln) };
                collapsed
                    .par_chunks_mut(d)
                    .zip(post.par_chunks_mut(m))
                    .zip(comb.par_chunks_mut(m * m))
                    .zip(streams.par_chunks(m * d))
                    .for_each(|(((o, p), cb), inp)| hc_pre(o, p, cb, inp, hc, c, n == 1));
                for t in 0..n {
                    rms(&mut normed[t * d..(t + 1) * d], &collapsed[t * d..(t + 1) * d], ln, c.eps);
                }
                timing.hc += t0.elapsed().as_secs_f64();

                let t1 = Instant::now();
                let branch = if site == 0 {
                    let st = &mut s.layers[i];
                    let out = match &layer.attn {
                        Attn::Kda(k) => self.kda_layer(k, &normed, n, st),
                        Attn::Mla(a) => self.mla_layer(a, &normed, n, st, s.pos),
                    };
                    timing.attn += t1.elapsed().as_secs_f64();
                    out
                } else {
                    let out = match &layer.ffn {
                        Ffn::Dense(mm) => self.mlp(mm, &normed, n),
                        Ffn::Moe { router, bias, shared } => self.moe(i, router, bias, shared, &normed, n)?,
                    };
                    timing.ffn += t1.elapsed().as_secs_f64();
                    out
                };

                let t3 = Instant::now();
                next.par_chunks_mut(m * d)
                    .zip(branch.par_chunks(d))
                    .zip(streams.par_chunks(m * d))
                    .zip(post.par_chunks(m))
                    .zip(comb.par_chunks(m * m))
                    .for_each(|((((o, b), r), p), cb)| hc_post(o, b, r, p, cb, m, d));
                std::mem::swap(&mut streams, &mut next);
                timing.hc += t3.elapsed().as_secs_f64();
                if early && site == 1 && i + 1 == c.first_dense {
                    self.prefetch_guess(i + 1, &streams, true);
                }
                // experiment: guess the next MoE layer from the streams as they leave this FFN
                if site == 1 && n == 1 && std::env::var_os("PELICA_TRACE").is_some() {
                    if let Some(nl) = self.layers.get(i + 1) {
                        if let Ffn::Moe { router: r2, bias: b2, .. } = &nl.ffn {
                            let (mut o, mut p2, mut cb) = (vec![0f32; d], vec![0f32; m], vec![0f32; m * m]);
                            hc_pre(&mut o, &mut p2, &mut cb, &streams, &nl.hc_ffn, c, true);
                            let mut xn = vec![0f32; d];
                            rms(&mut xn, &o, &nl.post_ln, c.eps);
                            let ne = c.n_experts;
                            let mut sc: Vec<(usize, f32)> =
                                router_scores(r2, &xn, ne, false).into_iter().enumerate().map(|(e, s)| (e, s + b2[e])).collect();
                            sc.sort_by(|a, b| b.1.total_cmp(&a.1));
                            sc.truncate(24);
                            trace("h", i + 1, &sc);

                        }
                    }
                }
            }
        }

        // the streams close with an unweighted mean, then the final norm and the head
        let t4 = Instant::now();
        let want = want.clamp(1, n);
        let mut mean = vec![0f32; want * d];
        for (r, t) in (n - want..n).enumerate() {
            let st = &streams[t * m * d..(t + 1) * m * d];
            let row = &mut mean[r * d..(r + 1) * d];
            for (j, v) in row.iter_mut().enumerate() {
                *v = (0..m).map(|h| st[h * d + j]).sum::<f32>() / m as f32;
            }
            rms_inplace(row, &self.final_norm, c.eps);
        }
        let logits = self.mm(&self.head, &mean, want);
        timing.head += t4.elapsed().as_secs_f64();

        if let Some(v) = &self.vram {
            v.tick(n);
        }
        s.pos += n;
        Ok(logits)
    }

    /// Prefetch MoE layer `l`'s likely experts for one decode token, from the
    /// hidden streams as they stand: its router applied to what its FFN input
    /// would be now. A `refine` pass also drops earlier guesses for the layer
    /// that it ranks below REFINE_KEEP (stopping them if not yet started).
    fn prefetch_guess(&self, l: usize, streams: &[f32], refine: bool) {
        let c = &self.cfg;
        let (d, m, ne) = (c.hidden, c.hc_mult, c.n_experts);
        let Some(nl) = self.layers.get(l) else { return };
        let Ffn::Moe { router, bias, .. } = &nl.ffn else { return };
        let (mut o, mut p, mut cb) = (vec![0f32; d], vec![0f32; m], vec![0f32; m * m]);
        hc_pre(&mut o, &mut p, &mut cb, streams, &nl.hc_ffn, c, true);
        let mut xn = vec![0f32; d];
        rms(&mut xn, &o, &nl.post_ln, c.eps);
        let mut sc: Vec<(usize, f32)> = router_scores(router, &xn, ne, true).into_iter().enumerate().map(|(e, s)| (e, s + bias[e])).collect();
        sc.sort_by(|a, b| b.1.total_cmp(&a.1));
        if refine {
            let keep: Vec<usize> = sc.iter().take(REFINE_KEEP.max(self.lookahead_width)).map(|g| g.0).collect();
            self.experts.regress(l, &keep);
        }
        let mut guess: Vec<usize> = sc.iter().take(self.lookahead_width).map(|g| g.0).collect();
        guess.retain(|&e| self.vram.as_ref().is_none_or(|v| !v.contains((l * ne + e) as u32)));
        self.experts.prefetch(l, &guess);
    }

    /// The MoE FFN of one layer: shared expert plus the routed top-k.
    #[allow(clippy::too_many_arguments)]
    fn moe(&self, layer: usize, router: &[u16], bias: &[f32], shared: &Mlp, x: &[f32], n: usize) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (d, ne, k) = (c.hidden, c.n_experts, c.topk);
        let key = |e: usize| (layer * ne + e) as u32;

        let t_route = Instant::now();
        // Selection ranks score + bias; the weight uses the plain score.
        let (picks, ranked): (Vec<Vec<(usize, f32)>>, Vec<Vec<usize>>) = (0..n)
            .into_par_iter()
            .map(|t| {
                let xt = &x[t * d..(t + 1) * d];
                let score = router_scores(router, xt, ne, n == 1);
                if n == 1 {
                    trace_scores(layer, &score, bias);
                }
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|&a, &b| (score[b] + bias[b]).total_cmp(&(score[a] + bias[a])));
                let chosen = &order[..k];
                let total: f32 = chosen.iter().map(|&e| score[e]).sum();
                let picks = chosen.iter().map(|&e| (e, score[e] / (total + 1e-20) * c.routed_scale)).collect();
                order.truncate(RANKS);
                (picks, order)
            })
            .unzip();
        self.experts.observe(layer, &ranked, k);
        if n == 1 {
            trace("", layer, &picks[0]);
            if let Some(Ffn::Moe { router: r2, bias: b2, .. }) = self.layers.get(layer + 1).map(|l| &l.ffn) {
                if std::env::var_os("PELICA_TRACE").is_some() {
                    let mut sc: Vec<(usize, f32)> = router_scores(r2, &x[..d], ne, false).into_iter().enumerate().map(|(e, s)| (e, s + b2[e])).collect();
                    sc.sort_by(|a, b| b.1.total_cmp(&a.1));
                    sc.truncate(24);
                    trace("g", layer + 1, &sc);
                }
            }
        }
        let mut routed: Vec<usize> = picks.iter().flatten().map(|p| p.0).collect();
        routed.sort_unstable();
        routed.dedup();
        self.experts.retire(layer, &routed);
        let vram = self.vram.as_ref();
        if let Some(v) = vram {
            let keys: Vec<u32> = picks.iter().flatten().map(|&(e, _)| key(e)).collect();
            v.touch(&keys, 0);
        }

        // Lookahead: the residual stream drifts slowly, so this layer's FFN input
        // through the next layers' routers is a good guess at what they will pick.
        // Queue reads for the guesses that aren't already in VRAM.
        for ahead in 1..=self.lookahead_layers {
            if let Some(Ffn::Moe { router: r2, bias: b2, .. }) = self.layers.get(layer + ahead).map(|l| &l.ffn) {
                let width = self.lookahead_width;
                // each token's top guesses, then all of them most confident first
                let mut best = vec![f32::NEG_INFINITY; ne];
                let per_token: Vec<Vec<(usize, f32)>> = (0..n)
                    .into_par_iter()
                    .map(|t| {
                        let xt = &x[t * d..(t + 1) * d];
                        let mut sc: Vec<(usize, f32)> =
                            router_scores(r2, xt, ne, n == 1).into_iter().enumerate().map(|(e, s)| (e, s + b2[e])).collect();
                        sc.select_nth_unstable_by(width - 1, |a, b| b.1.total_cmp(&a.1));
                        sc.truncate(width);
                        sc
                    })
                    .collect();
                for &(e, s) in per_token.iter().flatten() {
                    best[e] = best[e].max(s);
                }
                let mut guess: Vec<usize> = (0..ne).filter(|&e| best[e] > f32::NEG_INFINITY).collect();
                guess.sort_by(|&a, &b| best[b].total_cmp(&best[a]));
                let next = layer + ahead;
                guess.retain(|&e| vram.is_none_or(|v| !v.contains((next * ne + e) as u32)));
                self.experts.prefetch(next, &guess);
            }
        }

        prof::add(prof::ROUTE, t_route);

        // group tokens by expert, and split by tier
        let mut by_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne];
        for (t, p) in picks.iter().enumerate() {
            for &(e, w) in p {
                by_expert[e].push((t, w));
            }
        }
        let gather = |e: usize| -> Vec<f32> {
            let rows = &by_expert[e];
            let mut xg = vec![0f32; rows.len() * d];
            for (j, &(t, _)) in rows.iter().enumerate() {
                xg[j * d..(j + 1) * d].copy_from_slice(&x[t * d..(t + 1) * d]);
            }
            xg
        };
        let mut on_gpu: Vec<(usize, usize)> = Vec::new(); // (expert, slot)
        let mut on_cpu: Vec<usize> = Vec::new();
        for e in (0..ne).filter(|&e| !by_expert[e].is_empty()) {
            match vram.and_then(|v| v.slot_of(key(e))) {
                Some(slot) => on_gpu.push((e, slot)),
                None => on_cpu.push(e),
            }
        }
        let shared_gpu = matches!((shared.gate.gpu(), shared.up.gpu(), shared.down.gpu()), (Some(_), Some(_), Some(_)));
        let limit = c.swiglu_limit;


        // GPU: shared expert and the VRAM-resident experts, in one batch.
        let gpu_part_inner = || -> (Option<Vec<f32>>, Vec<Vec<f32>>) {
            let Some(gpu) = self.gpu.as_ref() else { return (None, Vec::new()) };            if !shared_gpu && on_gpu.is_empty() {
                return (None, Vec::new());
            }
            let mut b = gpu.batch();
            let xr = b.input(x);
            if shared_gpu {
                let y = mlp_into(&mut b, shared.gate.gpu().unwrap(), shared.up.gpu().unwrap(), shared.down.gpu().unwrap(), xr, n, limit);
                b.read(y);
            }
            let v = vram.unwrap_or_else(|| unreachable!());
            for &(e, slot) in &on_gpu {
                let [g, u, dn] = v.mats(slot);
                let r = by_expert[e].len();
                let xg = if r == n && n == 1 { xr } else { b.input(&gather(e)) };
                let y = mlp_into(&mut b, &g, &u, &dn, xg, r, limit);
                b.read(y);
            }
            let mut out = b.run();
            let shared_y = shared_gpu.then(|| out.remove(0));
            (shared_y, out)
        };

        // CPU: shared expert if it isn't on the GPU, then experts from RAM or SSD.
        // Keeps the hottest few buffers it fetched as candidates for promotion.
        let cpu_part_inner = || -> Result<(Option<Vec<f32>>, Vec<Vec<f32>>, Vec<(usize, Arc<ExpertBuf>)>)> {
            let shared_y = (!shared_gpu).then(|| self.mlp(shared, x, n));
            let mut ys = Vec::with_capacity(on_cpu.len());
            let mut candidates: Vec<(usize, Arc<ExpertBuf>)> = Vec::new();
            let staging = match (self.gpu.as_ref(), vram) {
                (Some(g), Some(v)) if n > 1 && v.n_stage() > 0 => Some((g, v)),
                _ => None,
            };
            let threads = rayon::current_num_threads();
            let run = |e: usize, buf: &ExpertBuf, split: bool| {
                let r = by_expert[e].len();
                expert_cpu(buf, &gather(e), r, c.moe_inter, d, limit, split && r == 1)
            };
            for chunk in on_cpu.chunks(32) {
                let mut part: Vec<Vec<f32>> = vec![Vec::new(); chunk.len()];
                if staging.is_none() {
                    // compute what is in RAM while the misses are read
                    let f = self.experts.start(layer, chunk);
                    let early = f.ready();
                    let ys_early: Vec<(usize, Vec<f32>)> =
                        early.par_iter().map(|(j, buf)| (*j, run(chunk[*j], buf, early.len() < threads))).collect();
                    let bufs = self.experts.finish(f)?;
                    let late: Vec<usize> = (0..chunk.len()).filter(|j| !early.iter().any(|(k, _)| k == j)).collect();
                    let ys_late: Vec<(usize, Vec<f32>)> =
                        late.par_iter().map(|&j| (j, run(chunk[j], &bufs[j], late.len() < threads))).collect();
                    for (j, y) in ys_early.into_iter().chain(ys_late) {
                        part[j] = y;
                    }
                    ys.extend(part);
                    if vram.is_some() {
                        for (&e, buf) in chunk.iter().zip(bufs) {
                            candidates.push((e, buf));
                            if candidates.len() > PROMOTE_FREE {
                                candidates.sort_by_key(|(e, _)| std::cmp::Reverse(by_expert[*e].len()));
                                candidates.truncate(PROMOTE_FREE);
                            }
                        }
                    }
                    continue;
                }
                let bufs = self.experts.get(layer, chunk)?;
                let split = chunk.len() < threads;
                // prefill: experts with several tokens go through the GPU
                let (big, small): (Vec<usize>, Vec<usize>) =
                    (0..chunk.len()).partition(|&j| staging.is_some() && by_expert[chunk[j]].len() >= STAGE_MIN_ROWS);
                let on_gpu_side = || -> Vec<(usize, Vec<f32>)> {
                    let Some((g, v)) = staging else { return Vec::new() };
                    let mut out = Vec::new();
                    for group in big.chunks(v.n_stage()) {
                        let refs: Vec<&ExpertBuf> = group.iter().map(|&j| &*bufs[j]).collect();
                        let mats = v.stage(g, &refs);
                        let mut b = g.batch();
                        for (&j, [gm, um, dm]) in group.iter().zip(&mats) {
                            let e = chunk[j];
                            let xg = b.input(&gather(e));
                            let y = mlp_into(&mut b, gm, um, dm, xg, by_expert[e].len(), limit);
                            b.read(y);
                        }
                        out.extend(group.iter().copied().zip(b.run()));
                    }
                    out
                };
                let on_cpu_side = || -> Vec<(usize, Vec<f32>)> {
                    small
                        .par_iter()
                        .map(|&j| {
                            let e = chunk[j];
                            let r = by_expert[e].len();
                            (j, expert_cpu(&bufs[j], &gather(e), r, c.moe_inter, d, limit, split && r == 1))
                        })
                        .collect()
                };
                let (a, b) = rayon::join(on_gpu_side, on_cpu_side);
                for (j, y) in a.into_iter().chain(b) {
                    part[j] = y;
                }
                ys.extend(part);
                if vram.is_some() {
                    for (&e, buf) in chunk.iter().zip(bufs) {
                        candidates.push((e, buf));
                        if candidates.len() > PROMOTE_FREE {
                            // keep the hottest by routing weight in this call
                            candidates.sort_by_key(|(e, _)| std::cmp::Reverse(by_expert[*e].len()));
                            candidates.truncate(PROMOTE_FREE);
                        }
                    }
                }
            }
            Ok((shared_y, ys, candidates))
        };

        let gpu_part = || -> (Option<Vec<f32>>, Vec<Vec<f32>>) {
            let t0 = Instant::now();
            let r = gpu_part_inner();
            prof::add(prof::MOE_GPU, t0);
            r
        };
        let cpu_part = || {
            let t0 = Instant::now();
            let r = cpu_part_inner();
            prof::add(prof::MOE_CPU, t0);
            r
        };

        let ((g_shared, g_ys), cpu) = rayon::join(gpu_part, cpu_part);
        let t_promote = Instant::now();
        let (c_shared, c_ys, candidates) = cpu?;

        let mut out = g_shared.or(c_shared).expect("shared expert computed on one side");
        let mut add = |e: usize, y: &[f32]| {
            for (j, &(t, w)) in by_expert[e].iter().enumerate() {
                for (o, v) in out[t * d..(t + 1) * d].iter_mut().zip(&y[j * d..(j + 1) * d]) {
                    *o += w * v;
                }
            }
        };
        for ((e, _), y) in on_gpu.iter().zip(&g_ys) {
            add(*e, y);
        }
        for (&e, y) in on_cpu.iter().zip(&c_ys) {
            add(e, y);
        }


        // Heating: experts that keep coming back move up to VRAM.
        if let Some(v) = vram {
            v.touch(&[], on_gpu.iter().map(|(e, _)| by_expert[*e].len() as u64).sum());
            let budget = if v.stats().filled < v.n_slots() { PROMOTE_FREE } else { 1 };
            let mut moved = 0;
            for (e, buf) in &candidates {
                if moved == budget {
                    break;
                }
                if v.consider(key(*e), buf) {
                    self.experts.forget(layer, *e);
                    moved += 1;
                }
            }
        }
        prof::add(prof::PROMOTE, t_promote);
        Ok(out)
    }

    fn kda_layer(&self, l: &Kda, x: &[f32], n: usize, st: &mut LayerState) -> Vec<f32> {
        if let (Some(gpu), Some(cst), Some(state)) = (&self.gpu, &l.gpu, &st.gpu) {
            let t0 = Instant::now();
            let out = self.kda_gpu(gpu, l, cst, state, x, n);
            prof::add(prof::KDA_GPU, t0);
            return out;
        }
        let t0 = Instant::now();
        let c = &self.cfg;
        let (h, hd, kk) = (c.kda_heads, c.kda_hd, c.conv_k);
        let p = h * hd;

        // projections: one GPU round trip when they all live in VRAM
        let ws = [&l.q, &l.k, &l.v, &l.fa, &l.fb, &l.b, &l.ga, &l.gb];
        let (q, k, v, mut decay, beta, gate) = if let (Some(gpu), true) = (&self.gpu, ws.iter().all(|w| w.gpu().is_some())) {
            let g = |w: &W| w.gpu().unwrap().clone();
            let mut b = gpu.batch();
            let xr = b.input(x);
            let q = b.matvec(&g(&l.q), xr, n);
            let k = b.matvec(&g(&l.k), xr, n);
            let v = b.matvec(&g(&l.v), xr, n);
            let fa = b.matvec(&g(&l.fa), xr, n);
            let fb = b.matvec(&g(&l.fb), fa, n);
            let be = b.matvec(&g(&l.b), xr, n);
            let ga = b.matvec(&g(&l.ga), xr, n);
            let gb = b.matvec(&g(&l.gb), ga, n);
            for r in [q, k, v, fb, be, gb] {
                b.read(r);
            }
            let mut r = b.run().into_iter();
            let mut nx = || r.next().unwrap();
            (nx(), nx(), nx(), nx(), nx(), nx())
        } else {
            let low = self.mm(&l.fa, x, n);
            let decay = self.mm(&l.fb, &low, n);
            let low = self.mm(&l.ga, x, n);
            let gate = self.mm(&l.gb, &low, n);
            (self.mm(&l.q, x, n), self.mm(&l.k, x, n), self.mm(&l.v, x, n), decay, self.mm(&l.b, x, n), gate)
        };

        prof::add(prof::KDA_GPU, t0);
        let t1 = Instant::now();
        // decay: gate_lower_bound * sigmoid(exp(A_log[h]) * (W_fb W_fa x + dt_bias))
        let a: Vec<f32> = l.alog.iter().map(|v| v.exp()).collect();
        let mut mixed = vec![0f32; 3 * p];
        let mut core = vec![0f32; p];
        let mut normed = vec![0f32; n * p];
        let qscale = 1.0 / (hd as f32).sqrt();
        for t in 0..n {
            let dk = &mut decay[t * p..(t + 1) * p];
            for (i, v) in dk.iter_mut().enumerate() {
                *v = c.gate_lb * sigmoid(a[i / hd] * (*v + l.dt[i]));
            }
            let bt: Vec<f32> = beta[t * h..(t + 1) * h].iter().map(|&b| sigmoid(b)).collect();

            // short causal conv over each channel's history, then SiLU
            let qkv = [&q[t * p..(t + 1) * p], &k[t * p..(t + 1) * p], &v[t * p..(t + 1) * p]];
            st.kda_window
                .par_chunks_mut(kk)
                .zip(mixed.par_iter_mut())
                .enumerate()
                .with_min_len(512)
                .for_each(|(ch, (hist, mx))| {
                    hist.copy_within(1.., 0);
                    hist[kk - 1] = qkv[ch / p][ch % p];
                    let taps = &l.conv[ch * kk..(ch + 1) * kk];
                    let s: f32 = taps.iter().zip(hist.iter()).map(|(a, b)| a * b).sum();
                    *mx = silu(s);
                });

            // gated delta rule, one head per task
            let dk = &decay[t * p..(t + 1) * p];
            st.kda_state
                .par_chunks_mut(hd * hd)
                .zip(core.par_chunks_mut(hd))
                .enumerate()
                .for_each(|(hh, (s, res))| {
                    let qv = &mixed[hh * hd..(hh + 1) * hd];
                    let kv = &mixed[p + hh * hd..p + (hh + 1) * hd];
                    let vv = &mixed[2 * p + hh * hd..2 * p + (hh + 1) * hd];
                    let dec = &dk[hh * hd..(hh + 1) * hd];
                    let qn = qscale / (1e-6 + qv.iter().map(|x| x * x).sum::<f32>()).sqrt();
                    let kn = 1.0 / (1e-6 + kv.iter().map(|x| x * x).sum::<f32>()).sqrt();
                    let mut mem = vec![0f32; hd];
                    for kk_ in 0..hd {
                        let row = &mut s[kk_ * hd..(kk_ + 1) * hd];
                        let alpha = dec[kk_].exp();
                        let sk = kv[kk_] * kn;
                        for (r, m) in row.iter_mut().zip(mem.iter_mut()) {
                            *r *= alpha;
                            *m += sk * *r;
                        }
                    }
                    res.fill(0.0);
                    let b = bt[hh];
                    for kk_ in 0..hd {
                        let row = &mut s[kk_ * hd..(kk_ + 1) * hd];
                        let sk = kv[kk_] * kn * b;
                        let sq = qv[kk_] * qn;
                        for ((r, o), (val, m)) in row.iter_mut().zip(res.iter_mut()).zip(vv.iter().zip(&mem)) {
                            *r += sk * (val - m);
                            *o += sq * *r;
                        }
                    }
                });

            // per-head RMSNorm weighted by o_norm, times the sigmoid output gate
            let gt = &gate[t * p..(t + 1) * p];
            let nt = &mut normed[t * p..(t + 1) * p];
            for hh in 0..h {
                let src = &core[hh * hd..(hh + 1) * hd];
                let inv = 1.0 / (src.iter().map(|x| x * x).sum::<f32>() / hd as f32 + c.eps).sqrt();
                for dd in 0..hd {
                    nt[hh * hd + dd] = src[dd] * inv * l.onorm[dd] * sigmoid(gt[hh * hd + dd]);
                }
            }
        }
        prof::add(prof::KDA_CPU, t1);
        let t2 = Instant::now();
        let out = self.mm(&l.o, &normed, n);
        prof::add(prof::KDA_GPU, t2);
        out
    }

    /// A whole KDA layer on the GPU in one round trip.
    #[allow(clippy::too_many_arguments)]
    fn kda_gpu(&self, gpu: &crate::gpu::Gpu, l: &Kda, cst: &wgpu::Buffer, state: &wgpu::Buffer, x: &[f32], n: usize) -> Vec<f32> {
        use crate::gpu::AK;
        let c = &self.cfg;
        let (h, hd, kk) = (c.kda_heads, c.kda_hd, c.conv_k);
        let p = h * hd;
        let g = |w: &W| w.gpu().unwrap().clone();
        let mut b = gpu.batch();
        let xr = b.input(x);
        let q = b.matvec(&g(&l.q), xr, n);
        let k = b.matvec(&g(&l.k), xr, n);
        let v = b.matvec(&g(&l.v), xr, n);
        let fa = b.matvec(&g(&l.fa), xr, n);
        let fb = b.matvec(&g(&l.fb), fa, n);
        let be = b.matvec(&g(&l.b), xr, n);
        let ga = b.matvec(&g(&l.ga), xr, n);
        let gb = b.matvec(&g(&l.gb), ga, n);
        let mixed = b.space(n * 3 * p);
        let core = b.space(n * p);
        // consts layout: conv taps [3p][kk] | dt [p] | exp(A_log) [h] | o_norm [hd]
        let (dt_at, a_at) = (3 * p * kk, 3 * p * kk + p);
        let onorm_at = a_at + h;
        let u = |v: usize| v as u32;
        b.attn(
            AK::Conv,
            cst,
            state,
            &[u(3 * p), u(p), u(n), q.off(), k.off(), v.off(), mixed.off(), 0, u(h * hd * hd), u(kk)],
            (3 * p).div_ceil(256),
        );
        b.attn(
            AK::Delta,
            cst,
            state,
            &[
                u(h),
                u(hd),
                u(n),
                mixed.off(),
                fb.off(),
                be.off(),
                gb.off(),
                core.off(),
                u(dt_at),
                u(a_at),
                u(onorm_at),
                c.gate_lb.to_bits(),
                c.eps.to_bits(),
            ],
            h,
        );
        let o = b.matvec(&g(&l.o), core, n);
        b.read(o);
        b.run().pop().unwrap()
    }

    /// A whole MLA layer on the GPU in one round trip; the latent cache lives in `state`.
    #[allow(clippy::too_many_arguments)]
    fn mla_gpu(&self, gpu: &crate::gpu::Gpu, l: &Mla, cst: &wgpu::Buffer, state: &wgpu::Buffer, x: &[f32], n: usize, base: usize) -> Vec<f32> {
        use crate::gpu::AK;
        let c = &self.cfg;
        let (h, qk, vd, lr) = (c.n_heads, c.qk_nope, c.v_head, c.kv_lora);
        let g = |w: &W| w.gpu().unwrap().clone();
        let u = |v: usize| v as u32;
        let mut b = gpu.batch();
        let xr = b.input(x);
        let qa = b.matvec(&g(&l.qa), xr, n);
        let lat = b.matvec(&g(&l.kva), xr, n);
        b.attn(AK::Rms, cst, state, &[u(c.q_lora), u(n), qa.off(), 0, c.eps.to_bits()], n);
        b.attn(AK::Rms, cst, state, &[u(lr), u(n), lat.off(), u(c.q_lora), c.eps.to_bits()], n);
        b.attn(AK::Put, cst, state, &[u(n * lr), lat.off(), u(base * lr)], (n * lr).div_ceil(256));
        let q = b.matvec(&g(&l.qb), qa, n);
        // absorb W_k into each head's query: [t][h][lr]
        let abs = b.matvec_heads(&g(&l.kvb_kt), q, n, lr, h * qk, qk, h * lr);
        let pooled = b.space(n * h * lr);
        let scale = 1.0 / (qk as f32).sqrt();
        b.attn(AK::Mla, cst, state, &[u(h), u(lr), abs.off(), pooled.off(), u(base), scale.to_bits()], n * h);
        let ctx = b.matvec_heads(&g(&l.kvb_v), pooled, n, vd, h * lr, lr, h * vd);
        let o = b.matvec(&g(&l.o), ctx, n);
        b.read(o);
        b.run().pop().unwrap()
    }

    fn mla_layer(&self, l: &Mla, x: &[f32], n: usize, st: &mut LayerState, base: usize) -> Vec<f32> {
        if let (Some(gpu), Some(cst), Some(state)) = (&self.gpu, &l.gpu, &st.gpu) {
            let t0 = Instant::now();
            let out = self.mla_gpu(gpu, l, cst, state, x, n, base);
            prof::add(prof::MLA_GPU, t0);
            return out;
        }
        let c = &self.cfg;
        let (h, qk, vd, lr) = (c.n_heads, c.qk_nope, c.v_head, c.kv_lora);
        let t0 = Instant::now();

        let (mut qa, mut lat) = match (&self.gpu, l.qa.gpu(), l.kva.gpu()) {
            (Some(gpu), Some(a), Some(kv)) => {
                let mut b = gpu.batch();
                let xr = b.input(x);
                let qa = b.matvec(a, xr, n);
                let lat = b.matvec(kv, xr, n);
                b.read(qa);
                b.read(lat);
                let mut r = b.run().into_iter();
                (r.next().unwrap(), r.next().unwrap())
            }
            _ => (self.mm(&l.qa, x, n), self.mm(&l.kva, x, n)),
        };
        for t in 0..n {
            rms_inplace(&mut qa[t * c.q_lora..(t + 1) * c.q_lora], &l.qa_ln, c.eps);
            rms_inplace(&mut lat[t * lr..(t + 1) * lr], &l.kva_ln, c.eps);
        }
        let q = self.mm(&l.qb, &qa, n);
        prof::add(prof::MLA_GPU, t0);
        let t1 = Instant::now();
        st.latent[base * lr..(base + n) * lr].copy_from_slice(&lat);

        // absorb W_k into the query: one [kv_lora x qk_nope] product per head
        let W::Cpu(kt) = &l.kvb_kt else { unreachable!("per-head weights on the GPU go with the GPU path") };
        let kt = kt.r();
        let mut absorbed = vec![0f32; n * h * lr];
        absorbed.par_chunks_mut(lr).enumerate().for_each(|(th, dst)| {
            let (t, hh) = (th / h, th % h);
            matmul_serial(dst, kt.rows_range(hh * lr, lr), &q[(t * h + hh) * qk..(t * h + hh + 1) * qk], 1);
        });

        // causal attention in latent space (the indexer keeps every position at this length)
        let latent = &st.latent;
        let scale = 1.0 / (qk as f32).sqrt();
        let mut pooled = vec![0f32; n * h * lr];
        pooled.par_chunks_mut(lr).enumerate().for_each(|(th, pool)| {
            let t = th / h;
            let qv = &absorbed[th * lr..(th + 1) * lr];
            let seen = base + t + 1;
            let mut score: Vec<f32> = (0..seen).map(|j| dot_f32(qv, &latent[j * lr..(j + 1) * lr]) * scale).collect();
            let top = score.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0f64;
            for s in score.iter_mut() {
                *s = (*s - top).exp();
                total += *s as f64;
            }
            pool.fill(0.0);
            for (j, s) in score.iter().enumerate() {
                let w = (*s as f64 / total) as f32;
                for (p, v) in pool.iter_mut().zip(&latent[j * lr..(j + 1) * lr]) {
                    *p += w * v;
                }
            }
        });

        let W::Cpu(vm) = &l.kvb_v else { unreachable!("per-head weights on the GPU go with the GPU path") };
        let vm = vm.r();
        let mut context = vec![0f32; n * h * vd];
        context.par_chunks_mut(vd).enumerate().for_each(|(th, dst)| {
            let hh = th % h;
            matmul_serial(dst, vm.rows_range(hh * vd, vd), &pooled[th * lr..(th + 1) * lr], 1);
        });
        prof::add(prof::MLA_CPU, t1);
        let t2 = Instant::now();
        let out = self.mm(&l.o, &context, n);
        prof::add(prof::MLA_GPU, t2);
        out
    }
}
