//! Resident weights of GLM-5.3-Flash (glm5_next) and their loader.
//!
//! Everything except the routed experts is resident: attention, the dense
//! and shared FFNs, the routers, the mHC mixers and the output head. The
//! checkpoint stores these in BF16; they are quantized at load (int8 per row
//! by default). With a GPU the big matrices go to VRAM (tier 1) and the RAM
//! they would have used is left to the expert cache (tier 2).

use crate::config::Config;
use crate::experts::ExpertStore;
use crate::gpu::{Gpu, W};
use crate::quant::Mat;
use crate::vram::VramTier;
use crate::st::{Shards, TensorInfo};
use anyhow::{bail, Result};
use std::path::Path;
use std::time::Instant;

pub struct Hc {
    pub fn_: Vec<f32>, // [(2+hc)*hc, hc*hidden]
    pub base: Vec<f32>,
    pub scale: [f32; 3],
}

pub struct Kda {
    pub q: W,
    pub k: W,
    pub v: W,
    pub o: W,
    pub fa: W,
    pub fb: W,
    pub ga: W,
    pub gb: W,
    pub b: W,
    /// depthwise conv taps [3 (q,k,v)][channels][kernel], oldest tap first
    pub conv: Vec<f32>,
    pub dt: Vec<f32>,
    pub alog: Vec<f32>,
    pub onorm: Vec<f32>,
    /// with every weight in VRAM: [conv taps | dt | exp(A_log) | o_norm] for
    /// the GPU kernels, which then run the whole layer
    pub gpu: Option<wgpu::Buffer>,
}

pub struct Mla {
    pub qa: W,
    pub qa_ln: Vec<f32>,
    pub qb: W,
    pub kva: W,
    pub kva_ln: Vec<f32>,
    /// W_k^T per head: [heads * kv_lora, qk_nope] (absorbed into the query)
    pub kvb_kt: W,
    /// W_v per head: [heads * v_head, kv_lora] (applied after the latent average)
    pub kvb_v: W,
    pub o: W,
    /// with every weight in VRAM: [qa_ln | kva_ln] for the GPU kernels
    pub gpu: Option<wgpu::Buffer>,
}

pub enum Attn {
    Kda(Kda),
    Mla(Mla),
}

pub struct Mlp {
    pub gate: W,
    pub up: W,
    pub down: W,
}

pub enum Ffn {
    Dense(Mlp),
    /// `router` holds bf16 bits, as stored in the checkpoint (half the RAM of f32)
    Moe { router: Vec<u16>, bias: Vec<f32>, shared: Mlp },
}

pub struct Layer {
    pub in_ln: Vec<f32>,
    pub post_ln: Vec<f32>,
    pub hc_attn: Hc,
    pub hc_ffn: Hc,
    pub attn: Attn,
    pub ffn: Ffn,
}

pub struct Model {
    pub cfg: Config,
    pub shards: Shards,
    pub embed: TensorInfo,
    pub final_norm: Vec<f32>,
    pub head: W,
    pub layers: Vec<Layer>,
    pub experts: ExpertStore,
    /// resident bytes in RAM and in VRAM
    pub ram_bytes: usize,
    pub vram_bytes: usize,
    pub gpu: Option<Gpu>,
    pub vram: Option<VramTier>,
    /// MoE layers ahead to prefetch experts for (0 = off)
    pub lookahead_layers: usize,
    /// experts guessed per token per lookahead layer
    pub lookahead_width: usize,
}

const PREFIX: &str = "model.language_model.";

struct Loader<'a> {
    s: &'a Shards,
    bits: u32,
    bytes: usize,
    gpu: Option<&'a Gpu>,
    vram_left: usize,
    vram_used: usize,
}

impl Loader<'_> {
    /// A tensor stored as bf16, kept as its bits. Fails if any value is not
    /// exactly a bf16 (so nothing is ever rounded).
    fn bf16(&mut self, name: &str, n: usize) -> Result<Vec<u16>> {
        let v = self.s.f32(&format!("{PREFIX}{name}"), n)?;
        if v.iter().any(|x| x.to_bits() & 0xffff != 0) {
            anyhow::bail!("{name} is not bf16");
        }
        self.bytes += v.len() * 2;
        Ok(v.iter().map(|x| (x.to_bits() >> 16) as u16).collect())
    }

    fn vec(&mut self, name: &str, n: usize) -> Result<Vec<f32>> {
        let v = self.s.f32(&format!("{PREFIX}{name}"), n)?;
        self.bytes += v.len() * 4;
        Ok(v)
    }

    /// A dense matrix, in VRAM while the budget allows, otherwise in RAM.
    fn mat(&mut self, name: &str, rows: usize, cols: usize) -> Result<W> {
        let m = self.mat_cpu(name, rows, cols)?;
        Ok(self.place(m))
    }

    /// Move a matrix to VRAM if it fits the budget.
    fn place(&mut self, m: Mat) -> W {
        if let Some(g) = self.gpu {
            if m.bytes() <= self.vram_left {
                if let Some(gm) = g.upload(&m) {
                    self.vram_left -= m.bytes();
                    self.vram_used += m.bytes();
                    self.bytes -= m.bytes();
                    return W::Gpu(gm);
                }
            }
        }
        W::Cpu(m)
    }

    /// Constants for a GPU layer, if that layer's weights all made it to VRAM.
    fn consts(&self, all_gpu: bool, data: &[f32]) -> Option<wgpu::Buffer> {
        let g = self.gpu?;
        all_gpu.then(|| g.buffer("consts", unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }))
    }

    fn mat_cpu(&mut self, name: &str, rows: usize, cols: usize) -> Result<Mat> {
        let full = if name.starts_with("lm_head") { name.to_string() } else { format!("{PREFIX}{name}") };
        let info = self.s.info(&full)?;
        if info.shape.len() != 2 || info.shape[0] != rows || info.shape[1] != cols {
            bail!("{full}: shape {:?}, expected [{rows}, {cols}]", info.shape);
        }
        let w = self.s.f32(&full, rows * cols)?;
        let m = Mat::from_f32(w, rows, cols, self.bits);
        self.bytes += m.bytes();
        Ok(m)
    }

    fn mlp(&mut self, p: &str, inter: usize, hidden: usize) -> Result<Mlp> {
        Ok(Mlp {
            gate: self.mat(&format!("{p}.gate_proj.weight"), inter, hidden)?,
            up: self.mat(&format!("{p}.up_proj.weight"), inter, hidden)?,
            down: self.mat(&format!("{p}.down_proj.weight"), hidden, inter)?,
        })
    }

    fn hc(&mut self, p: &str, c: &Config) -> Result<Hc> {
        let mix = (2 + c.hc_mult) * c.hc_mult;
        let s = self.vec(&format!("{p}_scale"), 3)?;
        Ok(Hc {
            fn_: self.vec(&format!("{p}_fn"), mix * c.hc_mult * c.hidden)?,
            base: self.vec(&format!("{p}_base"), mix)?,
            scale: [s[0], s[1], s[2]],
        })
    }
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dir: &Path,
        dense_bits: u32,
        cache_bytes: usize,
        io_threads: usize,
        direct: bool,
        gpu: Option<Gpu>,
        vram_budget: usize,
        pack_scales: bool,
    ) -> Result<Model> {
        let cfg = Config::load(dir)?;
        let mut shards = Shards::open(dir)?;
        eprintln!("[pelica] {} tensors in {} shards", shards.len(), shards.files.len());

        let t0 = Instant::now();
        // keep room in VRAM for the scratch buffers and the attention state
        let scratch = 768 << 20;
        let mut ld = Loader {
            s: &shards,
            bits: dense_bits,
            bytes: 0,
            gpu: gpu.as_ref(),
            vram_left: vram_budget.saturating_sub(scratch),
            vram_used: 0,
        };
        let (d, c) = (cfg.hidden, &cfg);
        let mut layers = Vec::with_capacity(c.n_layers);
        for i in 0..c.n_layers {
            let p = format!("layers.{i}");
            let attn = if c.layer_full[i] {
                let (h, l) = (c.n_heads, c.kv_lora);
                // kv_b_proj is [heads * (qk_nope + v_head), kv_lora]; split each
                // head's block into W_k (transposed, for absorption) and W_v.
                let kvb = ld.s.f32(&format!("{PREFIX}{p}.self_attn.kv_b_proj.weight"), h * (c.qk_nope + c.v_head) * l)?;
                let mut kt = vec![0f32; h * l * c.qk_nope];
                let mut vv = vec![0f32; h * c.v_head * l];
                for hd in 0..h {
                    let block = &kvb[hd * (c.qk_nope + c.v_head) * l..];
                    for dd in 0..l {
                        for ii in 0..c.qk_nope {
                            kt[(hd * l + dd) * c.qk_nope + ii] = block[ii * l + dd];
                        }
                    }
                    vv[hd * c.v_head * l..(hd + 1) * c.v_head * l]
                        .copy_from_slice(&block[c.qk_nope * l..(c.qk_nope + c.v_head) * l]);
                }
                drop(kvb);
                let qa = ld.mat(&format!("{p}.self_attn.q_a_proj.weight"), c.q_lora, d)?;
                let qa_ln = ld.vec(&format!("{p}.self_attn.q_a_layernorm.weight"), c.q_lora)?;
                let qb = ld.mat(&format!("{p}.self_attn.q_b_proj.weight"), h * c.qk_nope, c.q_lora)?;
                let kva = ld.mat(&format!("{p}.self_attn.kv_a_proj_with_mqa.weight"), l, d)?;
                let kva_ln = ld.vec(&format!("{p}.self_attn.kv_a_layernorm.weight"), l)?;
                let o = ld.mat(&format!("{p}.self_attn.o_proj.weight"), d, h * c.v_head)?;
                let (kvb_kt, kvb_v) = (Mat::from_f32(kt, h * l, c.qk_nope, dense_bits), Mat::from_f32(vv, h * c.v_head, l, dense_bits));
                ld.bytes += kvb_kt.bytes() + kvb_v.bytes();
                // the per-head matrices go to VRAM only alongside the rest of the layer
                let rest_gpu = [&qa, &qb, &kva, &o].iter().all(|w| w.gpu().is_some());
                let (kvb_kt, kvb_v) = if rest_gpu { (ld.place(kvb_kt), ld.place(kvb_v)) } else { (W::Cpu(kvb_kt), W::Cpu(kvb_v)) };
                // the attention kernel keeps up to 2056 scores per head in workgroup memory
                let all_gpu = rest_gpu && kvb_kt.gpu().is_some() && kvb_v.gpu().is_some() && c.dense_attention_limit() <= 2056;
                let gpu = ld.consts(all_gpu, &[qa_ln.as_slice(), kva_ln.as_slice()].concat());
                Attn::Mla(Mla { qa, qa_ln, qb, kva, kva_ln, kvb_kt, kvb_v, o, gpu })
            } else {
                let (h, hd) = (c.kda_heads, c.kda_hd);
                let pw = h * hd;
                let mut conv = Vec::with_capacity(3 * pw * c.conv_k);
                for n in ["q", "k", "v"] {
                    conv.extend(ld.vec(&format!("{p}.self_attn.{n}_conv1d.weight"), pw * c.conv_k)?);
                }
                let mut k = Kda {
                    q: ld.mat(&format!("{p}.self_attn.q_proj.weight"), pw, d)?,
                    k: ld.mat(&format!("{p}.self_attn.k_proj.weight"), pw, d)?,
                    v: ld.mat(&format!("{p}.self_attn.v_proj.weight"), pw, d)?,
                    o: ld.mat(&format!("{p}.self_attn.o_proj.weight"), d, pw)?,
                    fa: ld.mat(&format!("{p}.self_attn.f_a_proj.weight"), hd, d)?,
                    fb: ld.mat(&format!("{p}.self_attn.f_b_proj.weight"), pw, hd)?,
                    ga: ld.mat(&format!("{p}.self_attn.g_a_proj.weight"), hd, d)?,
                    gb: ld.mat(&format!("{p}.self_attn.g_b_proj.weight"), pw, hd)?,
                    b: ld.mat(&format!("{p}.self_attn.b_proj.weight"), h, d)?,
                    conv,
                    dt: ld.vec(&format!("{p}.self_attn.dt_bias"), pw)?,
                    alog: ld.vec(&format!("{p}.self_attn.A_log"), h)?,
                    onorm: ld.vec(&format!("{p}.self_attn.o_norm.weight"), hd)?,
                    gpu: None,
                };
                // the delta-rule kernel runs one thread per head dimension
                let all_gpu = hd == 128
                    && [&k.q, &k.k, &k.v, &k.o, &k.fa, &k.fb, &k.ga, &k.gb, &k.b].iter().all(|w| w.gpu().is_some());
                let a: Vec<f32> = k.alog.iter().map(|v| v.exp()).collect();
                k.gpu = ld.consts(all_gpu, &[k.conv.as_slice(), k.dt.as_slice(), &a, k.onorm.as_slice()].concat());
                Attn::Kda(k)
            };
            let ffn = if i < c.first_dense {
                Ffn::Dense(ld.mlp(&format!("{p}.mlp"), c.dense_inter, d)?)
            } else {
                Ffn::Moe {
                    router: ld.bf16(&format!("{p}.mlp.gate.weight"), c.n_experts * d)?,
                    bias: ld.vec(&format!("{p}.mlp.gate.e_score_correction_bias"), c.n_experts)?,
                    shared: ld.mlp(&format!("{p}.mlp.shared_experts"), c.moe_inter, d)?,
                }
            };
            layers.push(Layer {
                in_ln: ld.vec(&format!("{p}.input_layernorm.weight"), d)?,
                post_ln: ld.vec(&format!("{p}.post_attention_layernorm.weight"), d)?,
                hc_attn: ld.hc(&format!("{p}.hc_attn"), c)?,
                hc_ffn: ld.hc(&format!("{p}.hc_ffn"), c)?,
                attn,
                ffn,
            });
            eprint!("\r[pelica] loading dense weights: layer {}/{} ({:.1} GB)   ", i + 1, c.n_layers, ld.bytes as f64 / 1e9);
        }
        let final_norm = ld.vec("norm.weight", d)?;
        let head = ld.mat("lm_head.weight", c.vocab, d)?;
        let embed = shards.info(&format!("{PREFIX}embed_tokens.weight"))?.clone();
        let (ram_bytes, vram_bytes, vram_left) = (ld.bytes, ld.vram_used, ld.vram_left);
        eprintln!(
            "\r[pelica] dense weights (int{}): {:.2} GB in VRAM, {:.2} GB in RAM, loaded in {:.1}s          ",
            dense_bits,
            vram_bytes as f64 / 1e9,
            ram_bytes as f64 / 1e9,
            t0.elapsed().as_secs_f64()
        );

        let pack = if pack_scales { crate::scalepack::Pack::ensure(dir, c, &shards, PREFIX) } else { None };
        let experts = ExpertStore::new(
            &shards,
            PREFIX,
            c.n_layers,
            c.first_dense..c.n_layers,
            c.n_experts,
            c.moe_inter,
            d,
            cache_bytes,
            io_threads,
            direct,
            pack.as_ref(),
        )?;

        // whatever VRAM the dense weights left becomes expert slots (tier 1)
        let vram = gpu.as_ref().and_then(|g| {
            let packed = experts.packed();
            let n = (vram_left as u64 / VramTier::slot_bytes(c.moe_inter, d, packed)) as usize;
            (n > 0).then(|| VramTier::new(g, n, c.moe_inter, d, c.n_layers * c.n_experts, packed))
        });
        eprintln!(
            "[pelica] {} routed experts x {} layers, {:.1} MB each: {} VRAM slots, RAM tier and SSD below",
            c.n_experts,
            c.n_layers - c.first_dense,
            experts.expert_bytes() as f64 / 1e6,
            vram.as_ref().map_or(0, |v| v.n_slots()),
        );

        shards.release_pages()?;
        let lookahead_width = cfg.topk;
        Ok(Model {
            cfg,
            shards,
            embed,
            final_norm,
            head,
            layers,
            experts,
            ram_bytes,
            vram_bytes,
            gpu,
            vram,
            lookahead_layers: 1,
            lookahead_width,
        })
    }
}
