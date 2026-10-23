//! Pelica: a tiered GPU + RAM + SSD inference engine for giant MoE models.

#[cfg(windows)]
mod aio;
mod app;
mod chat_template;
mod config;
mod cpu_attn;
mod ds4;
mod ds4_cpu;
mod ds4_gpu;
mod ds4_io;
mod experts;
mod forward;
mod gguf;
mod gpu;
mod model;
mod moe;
mod moe_kernels;
mod oss;
mod oss_batch;
mod oss_prefill;
mod oss_kernels;
mod prof;
mod quant;
mod scalepack;
mod st;
mod vram;
mod web;

use anyhow::{bail, Context, Result};
use forward::Timing;
use model::Model;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    prompt: String,
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    seed: u64,
    raw: bool,
    think: bool,
    effort: String,
    cache_gb: Option<f64>,
    dense_bits: u32,
    ctx: usize,
    chunk: usize,
    threads: Option<usize>,
    ids: Option<Vec<u32>>,
    teacher: bool,
    lookahead: usize,
    lookahead_width: Option<usize>,
    io_threads: usize,
    warm: bool,
    direct: bool,
    pack: bool,
    bench_disk: bool,
    gpu: bool,
    vram_gb: Option<f64>,
    ignore_eos: bool,
}

const USAGE: &str = "\
pelica - run giant MoE models from GPU + RAM + SSD

USAGE: pelica --model <dir> -p <prompt> [options]
       pelica --model <file.gguf> -p <prompt>   GPU-resident engine for models that
                                                fit in VRAM (gpt-oss)
       pelica bench-disk --model <dir>      expert read throughput by reader count
       pelica app [--model <file.gguf>]     local chat app (opens a window)

  -m, --model <dir>       checkpoint directory (or env PELICA_MODEL)
  -p, --prompt <text>     prompt text
      --prompt-file <f>   prompt text from a file
  -n, --max-tokens <n>    tokens to generate (default 128)
      --temp <t>          sampling temperature, 0 = greedy (default 0)
      --top-p <p>         nucleus sampling (default 0.95)
      --seed <n>          sampling seed
      --raw               send the prompt as-is, no chat template
      --no-think          skip the reasoning block
      --effort <e>        reasoning effort: low | high | max (default low)
      --cache-gb <gb>     RAM for cached experts (default: free - 2.5 GB)
      --dense-bits <b>    resident weight precision: 8, 4 or 32 (default 8)
      --ctx <n>           context capacity (default 2048)
      --chunk <n>         prompt tokens per forward (default 512)
      --threads <n>       worker threads
      --lookahead <n>     MoE layers ahead to prefetch experts for (default 1, 0 = off)
      --lookahead-width <k>  experts guessed per token per layer (default top-k)
      --io-threads <n>    prefetch readers (default 2; 4 more serve blocking reads)
      --warm              preload historically hot experts at start
      --no-gpu            CPU only (no VRAM tier)
      --vram-gb <gb>      VRAM to use for weights and hot experts (default: free - 0.75 GB)
      --direct            read experts unbuffered, bypassing the OS file cache (default)
      --buffered          read experts through the OS file cache
      --no-pack           read expert scales as stored instead of from the packed
                          side file (pelica_scales.bin, built on first run)
      --ids <a,b,c>       token ids instead of a prompt
      --teacher           print the argmax at every prompt position and the
                          generated ids (for checking against a reference)
      --ignore-eos        keep generating past end-of-turn tokens (.gguf models)
";

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: std::env::var("PELICA_MODEL").map(PathBuf::from).unwrap_or_default(),
        prompt: String::new(),
        max_tokens: 128,
        temp: 0.0,
        top_p: 0.95,
        seed: 0x9e3779b97f4a7c15,
        raw: false,
        think: true,
        effort: "low".into(),
        cache_gb: None,
        dense_bits: 8,
        ctx: 2048,
        chunk: 512,
        threads: None,
        ids: None,
        teacher: false,
        lookahead: 1,
        lookahead_width: None,
        io_threads: 2,
        warm: false,
        direct: true,
        pack: true,
        bench_disk: false,
        gpu: true,
        vram_gb: None,
        ignore_eos: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-m" | "--model" => a.model = PathBuf::from(val()?),
            "-p" | "--prompt" => a.prompt = val()?,
            "--prompt-file" => a.prompt = std::fs::read_to_string(val()?).context("reading --prompt-file")?,
            "-n" | "--max-tokens" => a.max_tokens = val()?.parse()?,
            "--temp" => a.temp = val()?.parse()?,
            "--top-p" => a.top_p = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--raw" => a.raw = true,
            "--no-think" => a.think = false,
            "--effort" => a.effort = val()?,
            "--cache-gb" => a.cache_gb = Some(val()?.parse()?),
            "--dense-bits" => a.dense_bits = val()?.parse()?,
            "--ctx" => a.ctx = val()?.parse()?,
            "--chunk" => a.chunk = val()?.parse::<usize>()?.max(1),
            "--threads" => a.threads = Some(val()?.parse()?),
            "--ids" => {
                a.ids = Some(val()?.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>()?)
            }
            "--teacher" => a.teacher = true,
            "--lookahead" => a.lookahead = val()?.parse()?,
            "--lookahead-width" => a.lookahead_width = Some(val()?.parse()?),
            "--io-threads" => a.io_threads = val()?.parse()?,
            "--warm" => a.warm = true,
            "--direct" => a.direct = true,
            "--buffered" => a.direct = false,
            "--no-pack" => a.pack = false,
            "bench-disk" => a.bench_disk = true,
            "--no-gpu" => a.gpu = false,
            "--vram-gb" => a.vram_gb = Some(val()?.parse()?),
            "--ignore-eos" => a.ignore_eos = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            _ => bail!("unknown option {flag}\n\n{USAGE}"),
        }
    }
    if a.model.as_os_str().is_empty() {
        bail!("--model is required\n\n{USAGE}");
    }
    if a.prompt.is_empty() && a.ids.is_none() && !a.bench_disk {
        bail!("--prompt is required\n\n{USAGE}");
    }
    if ![4, 8, 32].contains(&a.dense_bits) {
        bail!("--dense-bits must be 4, 8 or 32");
    }
    Ok(a)
}

/// VRAM left free for the driver, the desktop and other apps
/// (`PELICA_VRAM_MARGIN_GB` overrides it).
const VRAM_MARGIN: usize = 3 << 28;

pub(crate) fn vram_margin() -> usize {
    std::env::var("PELICA_VRAM_MARGIN_GB").ok().and_then(|v| v.parse::<f64>().ok()).map_or(VRAM_MARGIN, |gb| (gb * (1u64 << 30) as f64) as usize)
}

/// RAM left free for the OS and other apps when sizing the RAM tier
/// (`PELICA_RAM_MARGIN_GB` overrides it). The tier's own overhead is small now
/// that at most 12 expert reads are buffered at once.
const RAM_MARGIN_GB: f64 = 2.5;

/// Free VRAM in bytes, from nvidia-smi when it is there (wgpu does not report it).
pub(crate) fn free_vram() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let mib: usize = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse().ok()?;
    Some(mib << 20)
}

/// Fill the VRAM expert slots with the historically hottest experts, and seed
/// their heat so they are not evicted before the conversation shows its own.
fn warm_vram(model: &Model) -> Result<usize> {
    let (Some(v), Some(g)) = (model.vram.as_ref(), model.gpu.as_ref()) else { return Ok(0) };
    let ne = model.experts.n_experts();
    let ranked: Vec<_> = model.experts.ranked_by_usage().into_iter().take(v.n_slots()).collect();
    let top = ranked.first().map_or(1, |r| r.2).max(1) as f32;
    let mut placed = 0;
    for chunk in ranked.chunks(16) {
        use rayon::prelude::*;
        let bufs: Vec<_> = chunk.par_iter().map(|&(l, e, _)| model.experts.read_now(l, e)).collect();
        for (&(l, e, count), buf) in chunk.iter().zip(bufs) {
            let key = (l * ne + e) as u32;
            if v.place(g, key, &buf?) {
                v.set_heat(key, 8.0 * count as f32 / top);
                placed += 1;
            }
        }
        g.flush();
    }
    Ok(placed)
}

/// Random-expert read throughput, buffered and unbuffered, by reader count.
fn bench_disk(dir: &std::path::Path) -> Result<()> {
    let cfg = config::Config::load(dir)?;
    let shards = st::Shards::open(dir)?;
    #[cfg(windows)]
    if std::env::var_os("PELICA_AIO_BENCH").is_some() {
        // overlapped reads from one thread: request size x requests in flight
        let sizes: Vec<usize> = if std::env::var_os("PELICA_AIO_QUICK").is_some() { vec![16 << 20] } else { vec![397312, 1 << 20, 4 << 20, 16 << 20] };
        let depths: Vec<usize> = if std::env::var_os("PELICA_AIO_QUICK").is_some() { vec![2] } else { vec![1, 2, 4, 8] };
        for &size in &sizes {
            for &depth in &depths {
                let gbs = aio::bench(&shards.paths, size, depth, 3.0)?;
                println!("overlapped {:>2} MB x {depth:>2} in flight: {gbs:5.2} GB/s", size >> 20);
            }
        }
        return Ok(());
    }
    for direct in [false, true] {
        let store = experts::ExpertStore::new(
            &shards,
            "model.language_model.",
            cfg.n_layers,
            cfg.first_dense..cfg.n_layers,
            cfg.n_experts,
            cfg.moe_inter,
            cfg.hidden,
            0,
            1,
            direct,
            scalepack::Pack::open(dir, &cfg).as_ref(),
        )?;
        for threads in [1, 2, 4, 8, 16, 32] {
            let gbs = store.bench_reads(threads, 4.0);
            println!("{:<10} {threads:>2} readers: {gbs:5.2} GB/s", if direct { "unbuffered" } else { "buffered" });
        }
        if direct {
            for chunk in [1usize << 20, 2 << 20, 4 << 20] {
                for threads in [8, 16, 32, 64] {
                    let gbs = store.bench_chunks(threads, chunk, 3.0);
                    println!("unbuffered {threads:>2} readers x {} MB requests: {gbs:5.2} GB/s", chunk >> 20);
                }
            }
        }
    }
    Ok(())
}

/// GLM-5.3's chat template for a single user turn.
pub(crate) fn chat_prompt(user: &str, effort: &str, think: bool) -> String {
    let effort = match effort {
        "low" => "Low",
        "high" => "High",
        _ => "Max",
    };
    let mut s = format!("[gMASK]<sop><|system|>Reasoning Effort: {effort}<|user|>{user}<|assistant|><think>");
    if !think {
        s.push_str("</think>");
    }
    s
}

/// Free physical memory in bytes.
pub(crate) fn available_ram() -> u64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }
        let mut m: MemoryStatusEx = unsafe { std::mem::zeroed() };
        m.length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 {
            return m.avail_phys;
        }
    }
    8 << 30
}

/// Committed private memory of this process (Windows; 0 elsewhere).
fn private_bytes() -> u64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct Counters {
            cb: u32,
            page_faults: u32,
            peak_ws: usize,
            ws: usize,
            quota_peak_paged: usize,
            quota_paged: usize,
            quota_peak_nonpaged: usize,
            quota_nonpaged: usize,
            pagefile: usize,
            peak_pagefile: usize,
            private: usize,
        }
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn K32GetProcessMemoryInfo(process: isize, c: *mut Counters, cb: u32) -> i32;
        }
        let mut c: Counters = unsafe { std::mem::zeroed() };
        c.cb = std::mem::size_of::<Counters>() as u32;
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } != 0 {
            return c.private as u64;
        }
    }
    0
}

/// With PELICA_LOGHASH set, print a hash of each position's logits, to check
/// that different batchings compute bit-identical results.
fn loghash(pos: usize, v: &[f32]) {
    if std::env::var_os("PELICA_LOGHASH").is_some() {
        let h = v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100_0000_01b3));
        eprintln!("loghash {pos} {h:016x}");
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap()
}

fn join(v: &[u32]) -> String {
    v.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

struct Sampler {
    temp: f32,
    top_p: f32,
    state: u64,
}

impl Sampler {
    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / (1u64 << 24) as f32
    }

    fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temp <= 0.0 {
            return argmax(logits);
        }
        let top = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(u32, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i as u32, ((l - top) / self.temp).exp())).collect();
        let total: f32 = probs.iter().map(|p| p.1).sum();
        probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut kept = 0.0;
        let mut cut = probs.len();
        for (i, p) in probs.iter().enumerate() {
            kept += p.1 / total;
            if kept >= self.top_p {
                cut = i + 1;
                break;
            }
        }
        let probs = &probs[..cut];
        let mass: f32 = probs.iter().map(|p| p.1).sum();
        let mut r = self.next_f32() * mass;
        for p in probs {
            r -= p.1;
            if r <= 0.0 {
                return p.0;
            }
        }
        probs[0].0
    }
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("app") {
        return app::run(&std::env::args().skip(2).collect::<Vec<_>>());
    }
    if std::env::args().nth(1).as_deref() == Some("gguf-info") {
        // metadata (short values) and tensors of a GGUF file
        let g = gguf::Gguf::open(std::path::Path::new(&std::env::args().nth(2).context("gguf-info FILE")?))?;
        let mut keys: Vec<_> = g.meta.iter().collect();
        keys.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in keys {
            let s = format!("{v:?}");
            println!("{k} = {}", if s.len() > 200 { format!("{}... ({} chars)", &s[..s.char_indices().nth(200).map_or(s.len(), |c| c.0)], s.len()) } else { s });
        }
        let mut ts: Vec<_> = g.tensors.iter().collect();
        ts.sort_by_key(|t| t.1.offset);
        for (n, t) in ts.iter().filter(|t| !t.0.starts_with("blk.") || t.0.starts_with("blk.0.") || t.0.starts_with("blk.1.")) {
            println!("{n} {:?} {}", t.dims, t.ty.name());
        }
        let mut by: std::collections::BTreeMap<String, usize> = Default::default();
        for t in g.tensors.values() {
            *by.entry(t.ty.name()).or_default() += t.bytes();
        }
        println!("{by:?}");
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("oss-host-test") {
        return oss::host_test();
    }
    if let Some(w) = std::env::args().nth(1).as_deref().and_then(|a| a.strip_prefix("oss-alloc-test")) {
        return oss::alloc_test(w == "-write");
    }
    if std::env::args().nth(1).as_deref() == Some("gpu-test") {
        let g = gpu::Gpu::new()?;
        println!("{}", g.name);
        print!("{}", gpu::self_test(&g)?);
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("cpu-test") {
        // one routed expert's worth of int4 (gate, up: 2048x4096; down: 4096x2048)
        let mut seed = 1u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect()
        };
        let experts: Vec<[quant::Mat; 3]> = (0..16)
            .map(|_| {
                [
                    quant::Mat::from_f32(rnd(2048 * 4096), 2048, 4096, 4),
                    quant::Mat::from_f32(rnd(2048 * 4096), 2048, 4096, 4),
                    quant::Mat::from_f32(rnd(4096 * 2048), 4096, 2048, 4),
                ]
            })
            .collect();
        let bytes: usize = experts[0].iter().map(|m| m.bytes()).sum();
        let x = rnd(4096);
        let run = |e: &[quant::Mat; 3], split: bool| {
            let (mut g, mut u, mut y) = (vec![0f32; 2048], vec![0f32; 2048], vec![0f32; 4096]);
            let mm = if split { quant::matmul } else { quant::matmul_serial };
            mm(&mut g, e[0].r(), &x, 1);
            mm(&mut u, e[1].r(), &x, 1);
            mm(&mut y, e[2].r(), &g, 1);
            y[0]
        };
        for (name, split) in [("one expert, rows split over threads", true)] {
            let t0 = Instant::now();
            for e in experts.iter().cycle().take(64) {
                run(e, split);
            }
            let s = t0.elapsed().as_secs_f64() / 64.0;
            println!("{name}: {:.2} ms ({:.1} GB/s)", s * 1e3, bytes as f64 / s / 1e9);
        }
        use rayon::prelude::*;
        let t0 = Instant::now();
        for _ in 0..8 {
            experts.par_iter().for_each(|e| {
                run(e, false);
            });
        }
        let s = t0.elapsed().as_secs_f64() / (8.0 * 16.0);
        println!("16 experts in parallel, one per thread: {:.2} ms each ({:.1} GB/s)", s * 1e3, bytes as f64 / s / 1e9);
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("gpu-mem") {
        // fill VRAM 512 MB at a time and watch where the memory goes
        let g = gpu::Gpu::new()?;
        let mut held = Vec::new();
        let block = vec![1u8; 512 << 20];
        loop {
            let Some(b) = g.try_empty("probe", 512 << 20) else { break };
            g.queue.write_buffer(&b, 0, &block);
            g.queue.submit([]);
            let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            held.push(b);
            println!(
                "{:5.1} GB allocated | RAM free {:5.1} GB | private {:5.1} GB | VRAM free {:5.1} GB",
                held.len() as f64 * 0.5,
                available_ram() as f64 / 1e9,
                private_bytes() as f64 / 1e9,
                free_vram().unwrap_or(0) as f64 / 1e9
            );
            if held.len() >= 40 {
                break;
            }
        }
        return Ok(());
    }
    let args = parse_args()?;
    if args.bench_disk {
        return bench_disk(&args.model);
    }
    if args.model.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
        if gguf::Gguf::open(&args.model).is_ok_and(|g| ds4::handles(&g)) {
            return ds4::run(&args.model, &args.prompt, args.ids.as_deref(), args.raw, args.max_tokens, args.ctx, args.teacher, args.ignore_eos);
        }
        if gguf::Gguf::open(&args.model).is_ok_and(|g| moe::handles(&g)) {
            return moe::run(&args.model, &args.prompt, args.ids.as_deref(), args.raw, args.max_tokens, args.ctx, args.teacher, args.ignore_eos);
        }
        return oss::run(&args.model, &args.prompt, args.ids.as_deref(), args.raw, args.max_tokens, args.ctx, args.teacher, args.ignore_eos);
    }
    if let Some(n) = args.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }

    let tok = tokenizers::Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let t_load = Instant::now();
    let opts = TierOpts {
        dense_bits: args.dense_bits,
        io_threads: args.io_threads,
        direct: args.direct,
        gpu: args.gpu,
        vram_gb: args.vram_gb,
        cache_gb: args.cache_gb,
        pack: args.pack,
        lookahead: args.lookahead,
        lookahead_width: args.lookahead_width,
        warm: args.warm,
    };
    let model = load_tiered(&args.model, &opts)?;
    let usage_path = args.model.join(".pelica_usage");
    eprintln!("[pelica] ready in {:.1}s, {} threads", t_load.elapsed().as_secs_f64(), rayon::current_num_threads());

    let text = if args.raw { args.prompt.clone() } else { chat_prompt(&args.prompt, &args.effort, args.think) };
    let ids: Vec<u32> = match &args.ids {
        Some(ids) => ids.clone(),
        None => tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec(),
    };
    eprintln!("[pelica] prompt: {} tokens", ids.len());

    let mut session = model.new_session(args.ctx);
    let mut timing = Timing::default();
    let mut sampler = Sampler { temp: args.temp, top_p: args.top_p, state: args.seed | 1 };

    let t_prefill = Instant::now();
    let mut logits = Vec::new();
    let mut teacher = Vec::new();
    let vocab = model.cfg.vocab;
    for chunk in ids.chunks(args.chunk) {
        if args.teacher {
            let rows = model.forward_rows(&mut session, chunk, &mut timing, chunk.len())?;
            for (i, r) in rows.chunks(vocab).enumerate() {
                loghash(session.pos - chunk.len() + i, r);
                teacher.push(argmax(r));
            }
            logits = rows[rows.len() - vocab..].to_vec();
        } else {
            logits = model.forward(&mut session, chunk, &mut timing)?;
        }
        eprint!("\r[pelica] prefill {}/{}", session.pos, ids.len());
    }
    if args.teacher {
        println!("prompt_ids {}", join(&ids));
        println!("teacher_forcing {}", join(&teacher));
    }
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    let prefill_stats = model.experts.stats();
    let prefill_vram = model.vram.as_ref().map(|v| v.stats());
    eprintln!("\r[pelica] prefill {} tokens in {:.1}s ({:.2} tok/s)", ids.len(), prefill_secs, ids.len() as f64 / prefill_secs);
    print_time("prefill", &timing, prefill_stats.wait_secs);
    timing = Timing::default();
    prof::reset();

    // debugging batch consistency: PELICA_FORCE=<ids> feeds these tokens after
    // the prompt, PELICA_FORCE_CHUNK at a time, printing their logits' hashes
    if let Ok(force) = std::env::var("PELICA_FORCE") {
        let ids: Vec<u32> = force.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        let k: usize = std::env::var("PELICA_FORCE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        for chunk in ids.chunks(k) {
            let rows = model.forward_rows(&mut session, chunk, &mut timing, chunk.len())?;
            for (i, r) in rows.chunks(vocab).enumerate() {
                loghash(session.pos - chunk.len() + i, r);
            }
        }
        return Ok(());
    }

    // Prefill's scratch buffers are done with: unless VRAM use was capped,
    // turn what they held (and anything freed since load) into expert slots.
    if args.vram_gb.is_none() {
        grow_vram(&model);
    }

    let t_decode = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut printed = 0usize;
    let out = std::io::stdout();
    for _ in 0..args.max_tokens {
        let next = sampler.sample(&logits);
        if model.cfg.eos.contains(&next) {
            break;
        }
        generated.push(next);
        let text = tok.decode(&generated, false).map_err(|e| anyhow::anyhow!("decoding: {e}"))?;
        if !text.ends_with('\u{FFFD}') && text.len() > printed {
            let mut o = out.lock();
            write!(o, "{}", &text[printed..])?;
            o.flush()?;
            printed = text.len();
        }
        if session.pos >= args.ctx {
            break;
        }
        logits = model.forward(&mut session, &[next], &mut timing)?;
        loghash(session.pos - 1, &logits);
    }
    println!();
    if args.teacher {
        println!("greedy {}", join(&generated));
    }
    let decode_secs = t_decode.elapsed().as_secs_f64();
    if let Err(e) = model.experts.save_usage(&usage_path) {
        eprintln!("[pelica] could not save routing history: {e}");
    }
    let n = generated.len().max(1);
    let st = model.experts.stats();
    let used = |a: &experts::CacheStats| a.hits + a.prefetch_hits + a.misses;
    let dec_used = (used(&st) - used(&prefill_stats)).max(1);
    let dec_hits = st.hits - prefill_stats.hits;
    let dec_pf = st.prefetch_hits - prefill_stats.prefetch_hits;
    let dec_miss = st.misses - prefill_stats.misses;
    eprintln!(
        "[pelica] decode {} tokens in {:.1}s: {:.3} tok/s ({:.2} s/token)",
        generated.len(),
        decode_secs,
        generated.len() as f64 / decode_secs,
        decode_secs / n as f64
    );
    let vs = model.vram.as_ref().map(|v| v.stats());
    let dec_gpu = vs.map_or(0, |v| v.gpu_uses) - prefill_vram.map_or(0, |v| v.gpu_uses);
    let all = (dec_used + dec_gpu).max(1) as f64;
    eprintln!(
        "[pelica] decode expert uses: {:.1}% VRAM (GPU), {:.1}% RAM, {:.1}% prefetched, {:.1}% read from SSD on demand",
        100.0 * dec_gpu as f64 / all,
        100.0 * dec_hits as f64 / all,
        100.0 * dec_pf as f64 / all,
        100.0 * dec_miss as f64 / all,
    );
    if let Some(v) = vs {
        eprintln!("[pelica] VRAM tier: {}/{} slots filled, {} promotions from RAM", v.filled, v.slots, v.promotions);
    }
    eprintln!(
        "[pelica] expert tier: {} resident ({:.1} GB); decode read {:.2} GB/token, {} prefetches issued, {} cancelled, {} wasted",
        st.resident,
        st.resident_bytes as f64 / 1e9,
        (st.bytes_read - prefill_stats.bytes_read) as f64 / 1e9 / n as f64,
        st.prefetch_issued - prefill_stats.prefetch_issued,
        st.prefetch_cancelled - prefill_stats.prefetch_cancelled,
        st.prefetch_wasted - prefill_stats.prefetch_wasted
    );
    print_time("decode", &timing, st.wait_secs - prefill_stats.wait_secs);
    eprintln!("[pelica] decode profile: {}", prof::report());
    eprintln!(
        "[pelica] decode waits: {:.1}s on demand reads ({} of them), {:.1}s on prefetches in flight; disk {:.2} GB/s, busy {:.0}% at {:.2} GB/s",
        st.wait_demand_secs - prefill_stats.wait_demand_secs,
        dec_miss,
        (st.wait_secs - st.wait_demand_secs) - (prefill_stats.wait_secs - prefill_stats.wait_demand_secs),
        (st.disk_bytes - prefill_stats.disk_bytes) as f64 / 1e9 / decode_secs,
        100.0 * (st.disk_busy_secs - prefill_stats.disk_busy_secs) / decode_secs,
        (st.disk_bytes - prefill_stats.disk_bytes) as f64 / 1e9 / (st.disk_busy_secs - prefill_stats.disk_busy_secs).max(1e-9)
    );
    eprintln!(
        "[pelica] disk idle: {:.2}s at token boundaries, {:.2}s between layers",
        st.idle_boundary_secs - prefill_stats.idle_boundary_secs,
        st.idle_inner_secs - prefill_stats.idle_inner_secs
    );
    Ok(())
}

/// How to load a tiered model (the CLI's flags; `pelica app` uses defaults).
pub(crate) struct TierOpts {
    pub dense_bits: u32,
    pub io_threads: usize,
    pub direct: bool,
    pub gpu: bool,
    pub vram_gb: Option<f64>,
    pub cache_gb: Option<f64>,
    pub pack: bool,
    pub lookahead: usize,
    pub lookahead_width: Option<usize>,
    pub warm: bool,
}

impl Default for TierOpts {
    fn default() -> Self {
        TierOpts { dense_bits: 8, io_threads: 2, direct: true, gpu: true, vram_gb: None, cache_gb: None, pack: true, lookahead: 1, lookahead_width: None, warm: false }
    }
}

/// A checkpoint directory loaded into the tiers: resident weights in VRAM
/// (or RAM), the RAM tier sized to what is free, warmed from the routing history.
pub(crate) fn load_tiered(dir: &std::path::Path, o: &TierOpts) -> Result<Model> {
    // Tier 1: the GPU, if there is one.
    let gpu = if o.gpu {
        match gpu::Gpu::new() {
            Ok(g) => {
                eprintln!("[pelica] GPU: {}", g.name);
                Some(g)
            }
            Err(e) => {
                eprintln!("[pelica] no usable GPU ({e}); running on CPU + RAM + SSD");
                None
            }
        }
    } else {
        None
    };
    let vram_budget = match (o.vram_gb, &gpu) {
        (_, None) => 0,
        (Some(gb), _) => (gb * 1e9) as usize,
        (None, Some(_)) => free_vram().unwrap_or(4 << 30).saturating_sub(vram_margin()),
    };

    let mut model = Model::load(dir, o.dense_bits, 0, o.io_threads, o.direct, gpu, vram_budget, o.pack)?;

    // Tier 2: the RAM tier gets what is free once the resident weights are in, minus headroom.
    let cache_bytes = match o.cache_gb {
        Some(gb) => (gb * 1e9) as usize,
        None => {
            let margin = std::env::var("PELICA_RAM_MARGIN_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(RAM_MARGIN_GB);
            (available_ram() as f64 - margin * 1e9).max(1e9) as usize
        }
    };
    model.experts.set_budget(cache_bytes);
    eprintln!(
        "[pelica] tiers: VRAM {:.1} GB dense + {} expert slots | RAM {:.1} GB dense + {:.1} GB experts (~{}) | SSD the rest",
        model.vram_bytes as f64 / 1e9,
        model.vram.as_ref().map_or(0, |v| v.n_slots()),
        model.ram_bytes as f64 / 1e9,
        cache_bytes as f64 / 1e9,
        cache_bytes / model.experts.expert_bytes().max(1)
    );
    model.lookahead_layers = o.lookahead;
    if let Some(w) = o.lookahead_width {
        model.lookahead_width = w.clamp(1, model.cfg.n_experts);
    }
    // Routing history: Pelica's own, or Colibri's if that is all there is.
    let usage_path = dir.join(".pelica_usage");
    let mut history = model.experts.load_usage(&usage_path);
    if history == 0 {
        history = model.experts.load_usage(&dir.join(".coli_usage"));
    }
    if history > 0 && model.vram.is_some() {
        let n = warm_vram(&model)?;
        eprintln!("[pelica] VRAM tier warmed with the {n} most-used experts ({history} past selections)");
    }
    if o.warm && history > 0 {
        let n = model.experts.warm(0.9, |key| model.vram.as_ref().is_some_and(|v| v.contains(key as u32)));
        eprintln!("[pelica] warming RAM tier with the {n} most-used experts");
    }
    Ok(model)
}

/// Once a prompt is in: unless VRAM use was capped, turn the prefill scratch
/// (and anything freed since load) into more VRAM expert slots.
pub(crate) fn grow_vram(model: &Model) {
    if let (Some(g), Some(v)) = (model.gpu.as_ref(), model.vram.as_ref()) {
        if std::env::var("PELICA_GROW").as_deref() != Ok("0") {
            g.trim_scratch();
            let per = vram::VramTier::slot_bytes(model.cfg.moe_inter, model.cfg.hidden, model.experts.packed()) as usize;
            let free = free_vram().unwrap_or(0).saturating_sub(vram_margin());
            let added = v.grow(g, free / per);
            if added > 0 {
                eprintln!("[pelica] {added} more VRAM expert slots after prefill ({} in all)", v.n_slots());
            }
        }
    }
}

fn print_time(what: &str, t: &Timing, wait: f64) {
    eprintln!(
        "[pelica] {what} time: attention {:.1}s, ffn {:.1}s (blocked on SSD {:.1}s), mhc {:.1}s, head {:.1}s",
        t.attn, t.ffn, wait, t.hc, t.head
    );
}
