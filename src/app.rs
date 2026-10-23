//! `pelica app`: a small local chat app for both of Pelica's engines.
//!
//! A std-only HTTP server on 127.0.0.1 serves one page (`app.html`) and a
//! JSON API. One engine thread owns whichever model is loaded: gpt-oss in a
//! `.gguf` (`oss`), any other mixture-of-experts `.gguf` (`moe`: the experts
//! that don't fit in VRAM stay in RAM), or a giant MoE checkpoint directory
//! run from VRAM + RAM + SSD (GLM-5.3, the tiered engine). A chat reply streams back
//! as server-sent events, split into the model's reasoning and its answer.
//! Follow-up messages reuse what the model has already read: the gguf
//! engine's cache by common prefix, the tiered engine's session by
//! continuing it when the chat is the one it last answered.

#[path = "app_v2.rs"]
mod v2;

use crate::forward::{Session, Timing};
use crate::model::Model;
use crate::oss::{self, Engine};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PAGE: &str = include_str!("app.html");
const DRAFT: usize = 7;
/// prompt tokens per tiered forward
const CHUNK: usize = 512;

/// Asked of every model unless the user turns it off: the commonest
/// complaint about small local models is confident invention.
const GROUNDING: &str = "You run offline on the user's computer, with no internet access and no tools. If you are not sure of a fact, or it may have changed since your training data ends, say so plainly instead of guessing. Never invent names, numbers, dates, quotes, citations or links.";
/// The same with web search on.
const GROUNDING_WEB: &str = "You run on the user's computer and can use the web: web_search finds pages and open_page reads one. Use them for current events, recent facts, prices, versions, or anything you are not sure of; search first, open the most relevant results, then answer from what you read and cite the links you used. If the pages don't settle it, say so plainly. Never invent names, numbers, dates, quotes, citations or links.";
/// With web search on, for chat templates that don't print tools: the
/// calling format described in words (the one Qwen-style templates use).
const TOOLS_IN_WORDS: &str = "You have two tools.\n- web_search: searches the web. Arguments: {\"query\": \"...\"}\n- open_page: reads a web page. Arguments: {\"url\": \"https://...\"}\nTo use one, reply with only this and stop:\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"...\"}}\n</tool_call>\nThe result comes back inside <tool_response></tool_response>; then continue, calling another tool or answering.";
/// Tool calls per reply, at most.
const TOOL_ROUNDS: usize = 6;

enum Job {
    Load { path: PathBuf, ctx: usize },
    Unload,
    Chat { req: Value, tx: Sender<Value> },
}

struct App {
    jobs: Mutex<Sender<Job>>,
    status: Mutex<Value>,
    /// the loaded model's tokenizer, for counting while the engine is busy
    tok: Mutex<Option<Arc<tokenizers::Tokenizer>>>,
    stop: AtomicBool,
    busy: AtomicBool,
    /// seconds since the epoch of the last request (the page pings)
    seen: AtomicU64,
    models_dirs: Vec<PathBuf>,
    port: u16,
    /// a built UI to serve instead of app.html (Pelica Desktop)
    ui: Option<PathBuf>,
}

fn now_s() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

const HELP: &str = "\
pelica app - local chat app

USAGE: pelica app [options]

      --model <path>       model to load at start: a .gguf file or a checkpoint
                           directory (default: the first .gguf under --models-dir)
      --no-autoload        without --model, start with no model loaded (the
                           app loads one when asked)
      --models-dir <dir>   where to look for models; repeat for more (default
                           C:\\models and D:\\Models, or env PELICA_MODELS,
                           separated by ';')
      --ctx <n>            context window for .gguf models (default 32768)
      --port <n>           port on 127.0.0.1 (default 8642)
      --no-open            don't open a window; keep serving until stopped
      --ui <dir>           serve this built UI (index.html and its files)
                           instead of the built-in page (Pelica Desktop)
";

pub fn run(args: &[String]) -> Result<()> {
    let mut model: Option<PathBuf> = std::env::var("PELICA_MODEL").ok().map(PathBuf::from);
    let mut dirs: Vec<PathBuf> = match std::env::var("PELICA_MODELS") {
        Ok(v) => v.split(';').filter(|x| !x.is_empty()).map(PathBuf::from).collect(),
        Err(_) => [r"C:\models", r"D:\Models"].iter().map(PathBuf::from).filter(|p| p.is_dir()).collect(),
    };
    let mut dirs_given = false;
    let mut ctx = 32768usize;
    let mut port = 8642u16;
    let mut open = true;
    let mut child = false;
    let mut ui: Option<PathBuf> = None;
    let mut autoload = true;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "-m" | "--model" => model = Some(PathBuf::from(val()?)),
            "--models-dir" => {
                if !dirs_given {
                    dirs.clear();
                    dirs_given = true;
                }
                dirs.push(PathBuf::from(val()?));
            }
            "--ctx" => ctx = val()?.parse()?,
            "--port" => port = val()?.parse()?,
            "--no-open" => open = false,
            "--no-autoload" => autoload = false,
            "--child" => child = true,
            "--ui" => ui = Some(PathBuf::from(val()?)),
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            _ => bail!("unknown option {a}\n\n{HELP}"),
        }
    }
    let url = format!("http://127.0.0.1:{port}/");
    #[cfg(windows)]
    if open && !child {
        // run on in the background without a console, logging to a file
        use std::os::windows::process::CommandExt;
        let log_path = std::env::temp_dir().join("pelica-app.log");
        let log = std::fs::File::create(&log_path)?;
        std::process::Command::new(std::env::current_exe()?)
            .arg("app")
            .args(args)
            .arg("--child")
            .stdin(std::process::Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .spawn()
            .context("starting the app in the background")?;
        eprintln!("[pelica] app starting at {url} (log: {})", log_path.display());
        return Ok(());
    }
    let _ = child;
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(_) if open => {
            // already running: just show it
            eprintln!("[pelica] port {port} is in use; opening {url}");
            open_window(&url);
            return Ok(());
        }
        Err(e) => return Err(e).with_context(|| format!("binding 127.0.0.1:{port}")),
    };
    let model = model.filter(|p| kind_of(p).is_some()).or_else(|| if autoload { find_models(&dirs).into_iter().find(|(_, k)| *k == Kind::Gguf).map(|(p, _)| p) } else { None });
    let (tx, rx) = channel();
    let app = Arc::new(App {
        jobs: Mutex::new(tx),
        status: Mutex::new(json!({ "state": "empty" })),
        tok: Mutex::new(None),
        stop: AtomicBool::new(false),
        busy: AtomicBool::new(false),
        seen: AtomicU64::new(now_s()),
        models_dirs: dirs,
        port,
        ui,
    });
    {
        let a = app.clone();
        std::thread::Builder::new().name("pelica-engine".into()).spawn(move || engine_thread(&a, rx))?;
    }
    if let Some(p) = model {
        set_loading(&app, &p, ctx);
        app.jobs.lock().unwrap().send(Job::Load { path: p, ctx })?;
    }
    eprintln!("[pelica] app at {url}");
    if open {
        open_window(&url);
        // with a window, quit a while after it stops pinging
        let a = app.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(15));
            let idle = now_s().saturating_sub(a.seen.load(Ordering::Relaxed));
            let loading = a.status.lock().unwrap()["state"] == "loading";
            if idle > 180 && !a.busy.load(Ordering::Relaxed) && !loading {
                eprintln!("[pelica] window closed; exiting");
                std::process::exit(0);
            }
        });
    }
    for s in listener.incoming() {
        let Ok(s) = s else { continue };
        let a = app.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(&a, s) {
                eprintln!("[pelica] request failed: {e:#}");
            }
        });
    }
    Ok(())
}

/// The app in a window of its own: Edge or Chrome in app mode, else the
/// default browser.
fn open_window(url: &str) {
    let pf = |v: &str| std::env::var(v).unwrap_or_default();
    let candidates = [
        format!(r"{}\Microsoft\Edge\Application\msedge.exe", pf("ProgramFiles(x86)")),
        format!(r"{}\Microsoft\Edge\Application\msedge.exe", pf("ProgramFiles")),
        format!(r"{}\Google\Chrome\Application\chrome.exe", pf("ProgramFiles")),
        format!(r"{}\Google\Chrome\Application\chrome.exe", pf("LOCALAPPDATA")),
    ];
    for c in candidates.iter().filter(|c| Path::new(c).is_file()) {
        if std::process::Command::new(c).arg(format!("--app={url}")).arg("--window-size=1240,860").spawn().is_ok() {
            return;
        }
    }
    let _ = std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn();
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// a .gguf the GPU-resident engine runs
    Gguf,
    /// a checkpoint directory for the tiered engine
    Tiered,
}

fn kind_of(p: &Path) -> Option<Kind> {
    if p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")) {
        Some(Kind::Gguf)
    } else if p.is_dir() && p.join("config.json").is_file() && p.join("tokenizer.json").is_file() && crate::config::Config::load(p).is_ok() {
        Some(Kind::Tiered)
    } else {
        None
    }
}

/// Models under the `dirs` (three levels deep): .gguf files and checkpoint
/// directories, smallest first.
fn find_models(dirs: &[PathBuf]) -> Vec<(PathBuf, Kind)> {
    fn walk(d: &Path, depth: usize, out: &mut Vec<(PathBuf, Kind)>) {
        if let Some(k @ Kind::Tiered) = kind_of(d) {
            out.push((d.to_path_buf(), k));
            return;
        }
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() && depth > 0 {
                walk(&p, depth - 1, out);
            } else if let Some(k) = kind_of(&p) {
                out.push((p, k));
            }
        }
    }
    let mut v = Vec::new();
    for d in dirs {
        walk(d, 3, &mut v);
    }
    v.sort_by_key(|(p, _)| size_of(p));
    v
}

/// Bytes on disk: the file, or a directory's files (one level).
fn size_of(p: &Path) -> u64 {
    if p.is_file() {
        return std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    }
    std::fs::read_dir(p).map(|rd| rd.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()).unwrap_or(0)
}

fn display_name(p: &Path) -> String {
    let n = if p.is_dir() { p.file_name() } else { p.file_stem() };
    let n = n.map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    match n.as_str() {
        "glm53_flash_i4" => "GLM-5.3-Flash (int4)".into(),
        // "gpt-oss-20b-MXFP4" -> "gpt-oss-20b (MXFP4)"
        // (or "...-Q4_K_M", whose quant has underscores of its own)
        _ => match n.rsplit_once(['-', '.']).filter(|(b, q)| !b.is_empty() && is_quant(q)).or_else(|| n.rsplit_once('_').filter(|(b, q)| !b.is_empty() && is_quant(q))) {
            Some((base, q)) => format!("{base} ({q})"),
            None => n,
        },
    }
}

fn is_quant(s: &str) -> bool {
    let u = s.to_ascii_uppercase();
    ["MXFP4", "F16", "BF16", "F32"].contains(&u.as_str())
        || ((u.starts_with('Q') || u.starts_with("IQ")) && u.chars().any(|c| c.is_ascii_digit()))
}

/// What the UI calls a model: "gpu" (gpt-oss), "moe" (another MoE .gguf)
/// or "tiered", and its context length if the file says.
fn ui_kind(p: &Path) -> (&'static str, Option<usize>) {
    // (remembered by path and modification time: reading a header takes a while)
    type Seen = HashMap<(PathBuf, Option<SystemTime>), (&'static str, Option<usize>)>;
    static SEEN: Mutex<Option<Seen>> = Mutex::new(None);
    let key = (p.to_path_buf(), std::fs::metadata(p).and_then(|m| m.modified()).ok());
    if let Some(&v) = SEEN.lock().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return v;
    }
    let v = ui_kind_read(p);
    SEEN.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, v);
    v
}

fn ui_kind_read(p: &Path) -> (&'static str, Option<usize>) {
    match kind_of(p) {
        Some(Kind::Tiered) => ("tiered", None),
        _ => match crate::gguf::Gguf::open(p) {
            Ok(g) => {
                let max = g.u(&format!("{}.context_length", g.arch())).ok();
                (if crate::moe::handles(&g) || crate::ds4::handles(&g) { "moe" } else { "gpu" }, max)
            }
            Err(_) => ("gpu", None),
        },
    }
}

fn set_loading(app: &App, path: &Path, ctx: usize) {
    oss::LOAD_PROGRESS.store(0, Ordering::Relaxed);
    crate::moe::LOAD_PROGRESS.store(0, Ordering::Relaxed);
    let (kind, efforts, effort) = match ui_kind(path).0 {
        "tiered" => ("tiered", json!(["off", "low", "high", "max"]), "low"),
        "moe" => ("moe", json!([]), ""),
        _ => ("gpu", json!(["low", "medium", "high"]), "medium"),
    };
    *app.status.lock().unwrap() = json!({ "state": "loading", "kind": kind, "model": display_name(path), "path": path.to_string_lossy(),
        "ctx": ctx, "efforts": efforts, "effort": effort, "since": now_s() });
}

// ---------- the engine thread ----------

struct Oss {
    eng: Engine,
    tok: Arc<tokenizers::Tokenizer>,
    /// tokens whose cache entries are valid, and how far passes wrote
    cached: Vec<u32>,
    written: usize,
}

struct Glm {
    model: Model,
    tok: Arc<tokenizers::Tokenizer>,
    dir: PathBuf,
    cap: usize,
    /// the session and exactly what it has read, the header it started
    /// with and the messages it has answered (to continue it next time)
    sess: Option<Session>,
    read: Vec<u32>,
    header: String,
    conv: Vec<(String, String)>,
}

/// A mixture-of-experts GGUF: the general engine, or DeepSeek V4's (whose
/// experts can also stream from the drive). Both take the same calls.
enum MoeEng {
    Gen(crate::moe::Engine),
    Ds4(crate::ds4::Engine),
}

impl MoeEng {
    fn cfg(&self) -> &crate::moe::Cfg {
        match self {
            MoeEng::Gen(e) => &e.cfg,
            MoeEng::Ds4(e) => &e.cfg,
        }
    }

    fn eos(&self) -> &[u32] {
        match self {
            MoeEng::Gen(e) => &e.eos,
            MoeEng::Ds4(e) => &e.eos,
        }
    }

    fn reusable(&self, cached: &[u32], prompt: &[u32]) -> usize {
        match self {
            MoeEng::Gen(e) => e.reusable(cached, prompt),
            MoeEng::Ds4(e) => e.reusable(cached, prompt),
        }
    }

    fn generate_with(&self, prompt: &[u32], n: usize, reuse: usize, on_token: impl FnMut(u32) -> bool, on_prefill: impl FnMut(usize, usize) -> bool) -> Result<oss::Stats> {
        match self {
            MoeEng::Gen(e) => e.generate_with(prompt, n, reuse, on_token, on_prefill),
            MoeEng::Ds4(e) => e.generate_with(prompt, n, reuse, on_token, on_prefill),
        }
    }

    fn trace(&self, toks: &[u32]) -> Result<Vec<Vec<Vec<(u32, f32)>>>> {
        match self {
            MoeEng::Gen(e) => e.trace(toks),
            MoeEng::Ds4(e) => e.trace(toks),
        }
    }

    fn save_usage(&self) -> Result<()> {
        match self {
            MoeEng::Gen(e) => e.save_usage(),
            MoeEng::Ds4(e) => e.save_usage(),
        }
    }

    fn activity(&self, prev: &mut Vec<u32>) -> Result<Vec<u32>> {
        match self {
            MoeEng::Gen(e) => e.activity(prev),
            MoeEng::Ds4(e) => e.activity(prev),
        }
    }

    fn expert_grid(&self) -> (usize, usize) {
        match self {
            MoeEng::Gen(e) => e.expert_grid(),
            MoeEng::Ds4(e) => e.expert_grid(),
        }
    }
}

struct Moe {
    eng: MoeEng,
    tok: Arc<tokenizers::Tokenizer>,
    template: Option<crate::chat_template::ChatTemplate>,
    /// the template has a thinking switch (`enable_thinking`)
    think: bool,
    /// tokens whose cache entries are valid
    cached: Vec<u32>,
}

enum Backend {
    Oss(Oss),
    Glm(Box<Glm>),
    Moe(Box<Moe>),
}

fn engine_thread(app: &App, rx: Receiver<Job>) {
    let mut be: Option<Backend> = None;
    for job in rx {
        match job {
            Job::Load { path, ctx } => {
                // free the old model first: VRAM and RAM are what both need
                be = None;
                *app.tok.lock().unwrap() = None;
                let t0 = Instant::now();
                match load(&path, ctx) {
                    Ok((b, mut st)) => {
                        let tok = match &b {
                            Backend::Oss(o) => o.tok.clone(),
                            Backend::Glm(g) => g.tok.clone(),
                            Backend::Moe(m) => m.tok.clone(),
                        };
                        *app.tok.lock().unwrap() = Some(tok);
                        st["state"] = json!("ready");
                        st["model"] = json!(display_name(&path));
                        st["path"] = json!(path.to_string_lossy());
                        st["load_s"] = json!(t0.elapsed().as_secs_f64());
                        // trained for tool calls (its template takes tools): models that
                        // aren't tend to explain how to do a task instead of doing it
                        st["agent"] = json!(match &b {
                            Backend::Moe(m) => m.template.as_ref().is_some_and(|t| t.supports_tools()),
                            Backend::Oss(_) | Backend::Glm(_) => true,
                        });
                        *app.status.lock().unwrap() = st;
                        be = Some(b);
                    }
                    Err(e) => {
                        let mut st = app.status.lock().unwrap();
                        st["state"] = json!("error");
                        st["error"] = json!(format!("{e:#}"));
                    }
                }
            }
            Job::Unload => {
                be = None;
                *app.tok.lock().unwrap() = None;
                *app.status.lock().unwrap() = json!({ "state": "empty" });
            }
            Job::Chat { req, tx } => {
                app.stop.store(false, Ordering::Relaxed);
                let v2 = req["v"].as_u64() == Some(2);
                let trace = req.get("trace").is_some();
                let r = match be.as_mut() {
                    None => Err(anyhow::anyhow!("no model loaded")),
                    Some(Backend::Moe(m)) if trace => v2::trace_moe(m, &req, &tx),
                    Some(_) if trace => Err(anyhow::anyhow!("tracing works with mixture-of-experts GGUF models")),
                    Some(Backend::Oss(o)) if v2 => v2::chat_oss(o, &req, &tx, &app.stop),
                    Some(Backend::Glm(g)) if v2 => v2::chat_glm(g, &req, &tx, &app.stop),
                    Some(Backend::Moe(m)) if v2 => v2::chat_moe(m, &req, &tx, &app.stop),
                    Some(Backend::Oss(o)) => chat_oss(o, &req, &tx, &app.stop),
                    Some(Backend::Glm(g)) => chat_glm(g, &req, &tx, &app.stop),
                    Some(Backend::Moe(m)) => chat_moe(m, &req, &tx, &app.stop),
                };
                if let Err(e) = r {
                    let _ = tx.send(json!({ "t": "error", "error": format!("{e:#}") }));
                }
                app.busy.store(false, Ordering::Relaxed);
            }
        }
    }
}

fn load(path: &Path, ctx: usize) -> Result<(Backend, Value)> {
    match kind_of(path) {
        Some(Kind::Gguf) => {
            let g = crate::gguf::Gguf::open(path)?;
            let tok = Arc::new(crate::gguf::tokenizer(&g)?);
            if crate::ds4::handles(&g) {
                let template = crate::chat_template::ChatTemplate::from_gguf(&g);
                let think = g.s("tokenizer.chat_template").is_some_and(|t| t.contains("enable_thinking"));
                drop(g);
                let eng = crate::ds4::Engine::load(path, ctx)?;
                let total = eng.vram_experts + eng.host_experts;
                // (the cache sits in RAM, read by the CPU: no VRAM / RAM / drive split to show)
                let st = json!({
                    "kind": "moe", "ctx": eng.cfg.ctx, "max_ctx": eng.max_ctx, "gpu": eng.name, "arch": eng.cfg.arch,
                    "experts": total, "experts_ram": eng.host_experts, "ram_gb": eng.host_bytes as f64 / 1e9,
                    "efforts": if think { json!(["off", "on"]) } else { json!([]) }, "effort": if think { "on" } else { "" },
                    "note": "Experts that aren't in VRAM or RAM stream from the drive: expect a few tokens per second.",
                });
                return Ok((Backend::Moe(Box::new(Moe { eng: MoeEng::Ds4(eng), tok, template, think, cached: Vec::new() })), st));
            }
            if crate::moe::handles(&g) {
                let template = crate::chat_template::ChatTemplate::from_gguf(&g);
                let think = g.s("tokenizer.chat_template").is_some_and(|t| t.contains("enable_thinking"));
                drop(g);
                let eng = crate::moe::Engine::load(path, ctx)?;
                let total = eng.vram_experts + eng.host_experts;
                let st = json!({
                    "kind": "moe", "ctx": eng.cfg.ctx, "max_ctx": eng.max_ctx, "gpu": eng.name, "arch": eng.cfg.arch,
                    "experts": total, "experts_ram": eng.host_experts, "ram_gb": eng.host_bytes as f64 / 1e9,
                    "kv_vram": eng.kv_vram.min(eng.cfg.ctx), "kv_ram": eng.kv_ram, "kv_drive": eng.cfg.ctx.saturating_sub(eng.kv_vram + eng.kv_ram),
                    "efforts": if think { json!(["off", "on"]) } else { json!([]) }, "effort": if think { "on" } else { "" },
                });
                return Ok((Backend::Moe(Box::new(Moe { eng: MoeEng::Gen(eng), tok, template, think, cached: Vec::new() })), st));
            }
            drop(g);
            let eng = Engine::load(path, ctx)?;
            let st = json!({
                "kind": "gpu", "ctx": eng.cfg.ctx, "max_ctx": eng.max_ctx, "gpu": eng.name,
                // the cache past `resident` positions lives in RAM (slower)
                "resident": eng.resident, "ram_gb": eng.host_bytes as f64 / 1e9,
                "efforts": ["low", "medium", "high"], "effort": "medium",
            });
            Ok((Backend::Oss(Oss { eng, tok, cached: Vec::new(), written: 0 }), st))
        }
        Some(Kind::Tiered) => {
            let tok = Arc::new(tokenizers::Tokenizer::from_file(path.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?);
            let model = crate::load_tiered(path, &crate::TierOpts::default())?;
            let cap = model.cfg.dense_attention_limit();
            let gpu = model.gpu.as_ref().map(|g| g.name.clone()).unwrap_or_else(|| "CPU".into());
            let st = json!({
                "kind": "tiered", "ctx": cap, "max_ctx": cap, "gpu": gpu,
                "efforts": ["off", "low", "high", "max"], "effort": "low",
                "note": "Experts stream from the SSD: expect a few tokens per second.",
            });
            Ok((Backend::Glm(Box::new(Glm { model, tok, dir: path.to_path_buf(), cap, sess: None, read: Vec::new(), header: String::new(), conv: Vec::new() })), st))
        }
        None => bail!("{} is not a model Pelica can run", path.display()),
    }
}

/// Today as YYYY-MM-DD (UTC), for the system message.
fn today() -> String {
    let z = (now_s() / 86400) as i64 + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02}")
}

/// The user's instructions, after the grounding ones unless turned off
/// (`web`: the version for a model with web tools).
fn instructions(req: &Value, web: bool) -> String {
    let user = req["system"].as_str().unwrap_or("").trim();
    let ground = req["grounded"].as_bool().unwrap_or(true);
    let g = if web { GROUNDING_WEB } else { GROUNDING };
    match (ground, user.is_empty()) {
        (true, true) => g.to_string(),
        (true, false) => format!("{g}\n\n{user}"),
        (false, _) => user.to_string(),
    }
}

/// Holds back a tool call from the streamed answer: text before
/// `<tool_call>` passes (minus what may be half the tag), the call's body is
/// kept, and `done` is set at `</tool_call>`.
struct ToolGate {
    buf: String,
    inside: bool,
    done: bool,
    /// the answer text before the call
    before: String,
    call: Option<String>,
}

impl ToolGate {
    fn new() -> Self {
        ToolGate { buf: String::new(), inside: false, done: false, before: String::new(), call: None }
    }
    fn feed(&mut self, d: &str) -> String {
        if self.done {
            return String::new();
        }
        self.buf += d;
        let mut vis = String::new();
        if !self.inside {
            const TAG: &str = "<tool_call>";
            if let Some(i) = self.buf.find(TAG) {
                vis = self.buf[..i].to_string();
                self.buf = self.buf[i + TAG.len()..].to_string();
                self.inside = true;
            } else {
                let n = self.buf.len();
                let keep = (1..TAG.len()).rev().find(|&k| n >= k && self.buf.is_char_boundary(n - k) && TAG.starts_with(&self.buf[n - k..])).unwrap_or(0);
                vis = self.buf[..n - keep].to_string();
                self.buf = self.buf[n - keep..].to_string();
            }
            self.before += &vis;
        }
        if self.inside {
            if let Some(j) = self.buf.find("</tool_call>") {
                self.call = Some(self.buf[..j].trim().to_string());
                self.buf.clear();
                self.done = true;
            }
        }
        vis
    }
    /// At the end of the reply: an unclosed call still counts; held-back
    /// text is released.
    fn finish(&mut self) -> String {
        if self.done {
            return String::new();
        }
        if self.inside {
            self.call = Some(std::mem::take(&mut self.buf).trim().to_string());
            self.done = true;
            return String::new();
        }
        let rest = std::mem::take(&mut self.buf);
        self.before += &rest;
        rest
    }
}

/// Runs a tool call (`{"name": ..., "arguments": {...}}`): what the model
/// reads back, and what the page shows. `chars`: how much page text fits.
fn run_tool(call: &str, chars: usize) -> (String, String, Value, Value) {
    let v: Value = serde_json::from_str(call).unwrap_or(Value::Null);
    let name = v["name"].as_str().unwrap_or("").to_string();
    let mut args = v["arguments"].clone();
    if let Some(s) = args.as_str() {
        args = serde_json::from_str(s).unwrap_or(Value::Null);
    }
    let (model, ui) = match name.as_str() {
        "web_search" => {
            let q = args["query"].as_str().unwrap_or("").trim().to_string();
            if q.is_empty() {
                ("Error: web_search needs a \"query\".".to_string(), json!({ "kind": "search", "query": q, "error": "no query" }))
            } else {
                match crate::web::search(&q) {
                    Ok(hits) => {
                        let mut t = String::new();
                        for (i, h) in hits.iter().enumerate() {
                            t += &format!("{}. {}\n   {}\n   {}\n", i + 1, h.title, h.url, h.snippet);
                        }
                        (t, json!({ "kind": "search", "query": q, "results": hits.iter().map(|h| json!({ "title": h.title, "url": h.url })).collect::<Vec<_>>() }))
                    }
                    Err(e) => (format!("Error: the search failed: {e}"), json!({ "kind": "search", "query": q, "error": e.to_string() })),
                }
            }
        }
        "open_page" => {
            let url = args["url"].as_str().unwrap_or("").trim().to_string();
            match crate::web::open(&url, chars) {
                Ok(p) => (format!("Title: {}\nURL: {}\n\n{}", p.title, p.url, p.text), json!({ "kind": "page", "url": p.url, "title": p.title })),
                Err(e) => (format!("Error: couldn't open {url}: {e}"), json!({ "kind": "page", "url": url, "error": e.to_string() })),
            }
        }
        _ => (format!("Error: there is no tool called \"{name}\"; the tools are web_search and open_page."), json!({ "kind": "unknown", "name": name, "error": "unknown tool" })),
    };
    (name, model, args, ui)
}

/// The conversation in gpt-oss's harmony format, ending where the
/// assistant's reply starts. Earlier replies keep only their final answer.
fn harmony(req: &Value, effort: &str, msgs: &[Value]) -> String {
    let mut t = format!(
        "<|start|>system<|message|>You are a helpful assistant (gpt-oss-20b) running locally through Pelica.\nKnowledge cutoff: 2024-06\nCurrent date: {}\n\nReasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>",
        today()
    );
    let ins = instructions(req, false);
    if !ins.is_empty() {
        t += &format!("<|start|>developer<|message|># Instructions\n\n{ins}<|end|>");
    }
    for m in msgs {
        let c = m["content"].as_str().unwrap_or("");
        match m["role"].as_str() {
            Some("assistant") => t += &format!("<|start|>assistant<|channel|>final<|message|>{c}<|end|>"),
            _ => t += &format!("<|start|>user<|message|>{c}<|end|>"),
        }
    }
    t + "<|start|>assistant"
}

/// Sends the newly completed text of a message being decoded (decoding the
/// whole message keeps multi-byte characters intact).
struct Delta {
    ids: Vec<u32>,
    sent: usize,
    text: String,
}

impl Delta {
    fn new() -> Self {
        Delta { ids: Vec::new(), sent: 0, text: String::new() }
    }
    fn push(&mut self, tok: &tokenizers::Tokenizer, t: u32) -> Option<String> {
        self.ids.push(t);
        let s = tok.decode(&self.ids, false).ok()?;
        if s.len() > self.sent && s.is_char_boundary(self.sent) && !s.ends_with('\u{FFFD}') {
            let d = s[self.sent..].to_string();
            self.sent = s.len();
            self.text = s;
            return Some(d);
        }
        None
    }
}

fn chat_oss(m: &mut Oss, req: &Value, tx: &Sender<Value>, stop: &AtomicBool) -> Result<()> {
    let effort = match req["effort"].as_str() {
        Some(e @ ("low" | "medium" | "high")) => e,
        _ => "medium",
    };
    let msgs = req["messages"].as_array().cloned().unwrap_or_default();
    let text = harmony(req, effort, &msgs);
    let ids = m.tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec();
    let ctx = m.eng.cfg.ctx;
    if ids.len() + 2 > ctx {
        bail!("This conversation is {} tokens, more than the {ctx}-token context window. Start a new chat or load the model with a larger context.", ids.len());
    }
    let n = (req["max_tokens"].as_u64().unwrap_or(4096) as usize).clamp(1, ctx - ids.len());
    let reuse = m.eng.reusable(&m.cached, m.written, &ids);
    m.cached.clear();
    let t0 = Instant::now();
    let _ = tx.send(json!({ "t": "start", "prompt": ids.len(), "reused": reuse, "ctx": ctx }));

    let tok = m.tok.clone();
    let id = |name: &str| tok.token_to_id(name).unwrap_or(u32::MAX);
    let (t_start, t_channel, t_message, t_end, t_constrain) = (id("<|start|>"), id("<|channel|>"), id("<|message|>"), id("<|end|>"), id("<|constrain|>"));
    let stops = [id("<|return|>"), id("<|call|>"), id("<|endoftext|>")];
    #[derive(PartialEq)]
    enum Mode {
        Header,
        Channel,
        Body,
    }
    let mut mode = Mode::Header;
    let mut hdr = Vec::new();
    let mut part = Delta::new();
    let mut chan = String::from("final");
    let mut gen = Vec::new();
    let (mut think_t0, mut think_t1, mut first) = (None::<Instant>, None::<Instant>, None::<Instant>);
    let (mut think_n, mut answer_n) = (0usize, 0usize);
    let mut gone = false;
    let stats = m.eng.generate_with(&ids, n, DRAFT, false, reuse, |t| {
        gen.push(t);
        first.get_or_insert_with(Instant::now);
        if stops.contains(&t) {
            return false;
        }
        if t == t_channel {
            mode = Mode::Channel;
            hdr.clear();
        } else if t == t_message {
            chan = tok.decode(&hdr, false).unwrap_or_default().split_whitespace().next().unwrap_or("final").to_string();
            if chan == "final" && think_t0.is_some() && think_t1.is_none() {
                think_t1 = Some(Instant::now());
            }
            mode = Mode::Body;
            part = Delta::new();
        } else if t == t_end || t == t_start {
            mode = Mode::Header;
        } else if mode == Mode::Channel || t == t_constrain {
            hdr.push(t);
        } else if mode == Mode::Body {
            let think = chan != "final";
            if think {
                think_n += 1;
                think_t0.get_or_insert_with(Instant::now);
            } else {
                answer_n += 1;
            }
            if let Some(d) = part.push(&tok, t) {
                if tx.send(json!({ "t": if think { "think" } else { "text" }, "d": d })).is_err() {
                    gone = true;
                }
            }
        }
        !gone && !stop.load(Ordering::Relaxed)
    }, |done, total| {
        // a long prompt: how far reading it has got
        tx.send(json!({ "t": "prefill", "done": done, "total": total })).is_ok() && !stop.load(Ordering::Relaxed)
    })?;
    if gen.is_empty() && stats.prompt + 1 < ids.len() {
        // stopped while reading the prompt: the cache holds its start
        m.cached = ids[..stats.prompt].to_vec();
        m.written = stats.prompt;
    } else {
        // every generated token but the last went through the model; passes
        // still in flight at the end may have written a few positions more
        m.cached = ids.clone();
        m.cached.extend_from_slice(&gen[..gen.len().saturating_sub(1)]);
        m.written = ids.len() + gen.len() + 4 * 8;
    }
    let dec = stats.generated.saturating_sub(1);
    let stopped = stop.load(Ordering::Relaxed);
    let ended = gen.last().is_some_and(|t| stops.contains(t));
    let _ = tx.send(json!({
        "t": "done",
        "prompt": ids.len(), "reused": reuse, "generated": gen.len(),
        "think_tokens": think_n, "answer_tokens": answer_n,
        "think_s": think_t0.map(|a| (think_t1.unwrap_or_else(Instant::now) - a).as_secs_f64()),
        "first_s": first.map(|f| (f - t0).as_secs_f64()),
        "prefill_s": stats.prefill_s, "decode_s": stats.decode_s,
        "tps": dec as f64 / stats.decode_s.max(1e-9),
        "ctx_used": ids.len() + gen.len(), "ctx": ctx,
        "stopped": stopped, "truncated": !stopped && !ended,
    }));
    Ok(())
}

/// Splits streamed text into thinking (inside <think>...</think>) and
/// answer, holding back the end of the text while it may be half a tag.
struct ThinkSplit {
    think: bool,
    buf: String,
    answered: bool,
}

impl ThinkSplit {
    fn feed(&mut self, d: &str) -> Vec<(bool, String)> {
        self.buf += d;
        let mut out = Vec::new();
        loop {
            let tag = if self.think { "</think>" } else { "<think>" };
            if let Some(i) = self.buf.find(tag) {
                let before = self.buf[..i].to_string();
                self.emit(&mut out, before);
                self.buf = self.buf[i + tag.len()..].to_string();
                self.think = !self.think;
                continue;
            }
            let n = self.buf.len();
            let keep = (1..tag.len()).rev().find(|&k| n >= k && self.buf.is_char_boundary(n - k) && tag.starts_with(&self.buf[n - k..])).unwrap_or(0);
            let now = self.buf[..n - keep].to_string();
            self.emit(&mut out, now);
            self.buf = self.buf[n - keep..].to_string();
            return out;
        }
    }
    fn finish(&mut self) -> Vec<(bool, String)> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        self.emit(&mut out, rest);
        out
    }
    fn emit(&mut self, out: &mut Vec<(bool, String)>, mut s: String) {
        if !self.think && !self.answered {
            // the answer starts after the blank lines that follow thinking
            s = s.trim_start().to_string();
        }
        if !s.is_empty() {
            self.answered |= !self.think;
            out.push((self.think, s));
        }
    }
}

/// The prompt for a MoE model's chat: its own template (with the system
/// instructions, or failing that with them in the first message), else a
/// plain transcript.
fn moe_prompt(m: &Moe, conv: &[Value], ins: &str, extra: &serde_json::Map<String, Value>) -> Result<String> {
    Ok(match &m.template {
        Some(t) => {
            let mut with_sys = conv.to_vec();
            if !ins.is_empty() {
                with_sys.insert(0, json!({ "role": "system", "content": ins }));
            }
            match t.render(&with_sys, extra) {
                Ok(x) => x,
                // (templates without a system role)
                Err(_) if !ins.is_empty() => {
                    let mut conv = conv.to_vec();
                    if let Some(c) = conv.iter_mut().find(|x| x["role"] == "user") {
                        c["content"] = json!(format!("{ins}\n\n{}", c["content"].as_str().unwrap_or("")));
                    }
                    t.render(&conv, extra)?
                }
                Err(e) => return Err(e),
            }
        }
        None => {
            let mut t = if ins.is_empty() { String::new() } else { format!("{ins}\n\n") };
            for x in conv {
                t += &format!("{}: {}\n", if x["role"] == "assistant" { "Assistant" } else { "User" }, x["content"].as_str().unwrap_or(""));
            }
            t + "Assistant:"
        }
    })
}

fn chat_moe(m: &mut Moe, req: &Value, tx: &Sender<Value>, stop: &AtomicBool) -> Result<()> {
    let msgs = req["messages"].as_array().cloned().unwrap_or_default();
    let web = req["web"].as_bool().unwrap_or(false);
    let native = web && m.template.as_ref().is_some_and(|t| t.supports_tools());
    let mut ins = instructions(req, web);
    if web && !native {
        ins = if ins.is_empty() { TOOLS_IN_WORDS.to_string() } else { format!("{ins}\n\n{TOOLS_IN_WORDS}") };
    }
    let mut conv: Vec<Value> = msgs
        .iter()
        .map(|x| json!({ "role": if x["role"] == "assistant" { "assistant" } else { "user" }, "content": x["content"].as_str().unwrap_or("") }))
        .collect();
    let mut extra = serde_json::Map::new();
    if m.think {
        extra.insert("enable_thinking".into(), json!(req["effort"].as_str() != Some("off")));
    }
    if native {
        extra.insert("tools".into(), crate::web::tool_specs());
    }
    let ctx = m.eng.cfg().ctx;
    let tok = m.tok.clone();
    let eos = m.eng.eos().to_vec();
    let t0 = Instant::now();
    let (mut think_t0, mut think_t1, mut first) = (None::<Instant>, None::<Instant>, None::<Instant>);
    let (mut think_n, mut answer_n, mut gen_n, mut dec_n) = (0usize, 0usize, 0usize, 0usize);
    let (mut prefill_s, mut decode_s) = (0f64, 0f64);
    let mut gone = false;
    let (mut ids, mut gen): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut reuse0 = 0;
    for round in 0..=TOOL_ROUNDS {
        let text = moe_prompt(m, &conv, &ins, &extra)?;
        ids = m.tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec();
        if ids.len() + 8 > ctx {
            if round > 0 {
                bail!("The pages read so far fill the {ctx}-token context window. Load the model with a larger context, or ask for less.");
            }
            bail!("This conversation is {} tokens, more than the {ctx}-token context window. Start a new chat or load the model with a larger context.", ids.len());
        }
        let n = (req["max_tokens"].as_u64().unwrap_or(4096) as usize).clamp(1, ctx - ids.len() - 4);
        let reuse = m.eng.reusable(&m.cached, &ids);
        m.cached.clear();
        if round == 0 {
            reuse0 = reuse;
            let _ = tx.send(json!({ "t": "start", "prompt": ids.len(), "reused": reuse, "ctx": ctx }));
        }
        let mut split = ThinkSplit { think: text.trim_end().ends_with("<think>"), buf: String::new(), answered: false };
        let mut gate = ToolGate::new();
        let mut part = Delta::new();
        gen = Vec::new();
        let send = |evs: Vec<(bool, String)>, gate: &mut ToolGate, gone: &mut bool| {
            for (think, d) in evs {
                let d = if think || !web { d } else { gate.feed(&d) };
                if !d.is_empty() && tx.send(json!({ "t": if think { "think" } else { "text" }, "d": d })).is_err() {
                    *gone = true;
                }
            }
        };
        let stats = m.eng.generate_with(&ids, n, reuse, |t| {
            gen.push(t);
            first.get_or_insert_with(Instant::now);
            if eos.contains(&t) {
                return false;
            }
            if split.think {
                think_n += 1;
                think_t0.get_or_insert_with(Instant::now);
            } else {
                answer_n += 1;
                if think_t0.is_some() && think_t1.is_none() {
                    think_t1 = Some(Instant::now());
                }
            }
            if let Some(d) = part.push(&tok, t) {
                send(split.feed(&d), &mut gate, &mut gone);
            }
            !gone && !gate.done && !stop.load(Ordering::Relaxed)
        }, |done, total| tx.send(json!({ "t": "prefill", "done": done, "total": total })).is_ok() && !stop.load(Ordering::Relaxed))?;
        send(split.finish(), &mut gate, &mut gone);
        if web {
            let rest = gate.finish();
            if !rest.is_empty() && tx.send(json!({ "t": "text", "d": rest })).is_err() {
                gone = true;
            }
        }
        if gen.is_empty() && stats.prompt + 1 < ids.len() {
            m.cached = ids[..stats.prompt].to_vec();
        } else {
            // every generated token but the last went through the model
            m.cached = ids.clone();
            m.cached.extend_from_slice(&gen[..gen.len().saturating_sub(1)]);
        }
        gen_n += gen.len();
        dec_n += stats.generated.saturating_sub(1);
        prefill_s += stats.prefill_s;
        decode_s += stats.decode_s;
        let Some(call) = gate.call.take() else { break };
        if gone || stop.load(Ordering::Relaxed) {
            break;
        }
        // the call: run it, show it, and give the model its result
        let _ = tx.send(json!({ "t": "tool", "call": call }));
        let room = ctx.saturating_sub(ids.len() + gen.len() + 1024);
        let (name, result, args, ui) = if round == TOOL_ROUNDS {
            ("".to_string(), "Error: no more tool calls in this reply; answer with what you have.".to_string(), Value::Null, json!({ "kind": "limit", "error": "tool call limit reached" }))
        } else {
            run_tool(&call, (room * 3).clamp(1000, 16000))
        };
        if tx.send(json!({ "t": "tool_done", "tool": ui })).is_err() {
            break;
        }
        let before = gate.before.trim().to_string();
        if native {
            conv.push(json!({ "role": "assistant", "content": before, "tool_calls": [{ "type": "function", "function": { "name": name, "arguments": args } }] }));
            conv.push(json!({ "role": "tool", "content": result }));
        } else {
            conv.push(json!({ "role": "assistant", "content": format!("{before}\n<tool_call>\n{call}\n</tool_call>").trim_start().to_string() }));
            conv.push(json!({ "role": "user", "content": format!("<tool_response>\n{result}\n</tool_response>") }));
        }
    }
    // which experts this chat used: the next load keeps those in VRAM
    let _ = m.eng.save_usage();
    let stopped = stop.load(Ordering::Relaxed);
    let ended = gen.last().is_some_and(|t| eos.contains(t));
    let _ = tx.send(json!({
        "t": "done",
        "prompt": ids.len(), "reused": reuse0, "generated": gen_n,
        "think_tokens": think_n, "answer_tokens": answer_n,
        "think_s": think_t0.map(|a| (think_t1.unwrap_or_else(Instant::now) - a).as_secs_f64()),
        "first_s": first.map(|f| (f - t0).as_secs_f64()),
        "prefill_s": prefill_s, "decode_s": decode_s,
        "tps": dec_n as f64 / decode_s.max(1e-9),
        "ctx_used": ids.len() + gen.len(), "ctx": ctx,
        "stopped": stopped, "truncated": !stopped && !ended,
    }));
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

fn chat_glm(g: &mut Glm, req: &Value, tx: &Sender<Value>, stop: &AtomicBool) -> Result<()> {
    let effort = match req["effort"].as_str() {
        Some(e @ ("off" | "low" | "high" | "max")) => e,
        _ => "low",
    };
    let think = effort != "off";
    let level = match effort {
        "high" => "High",
        "max" => "Max",
        _ => "Low",
    };
    let mut header = format!("[gMASK]<sop><|system|>Reasoning Effort: {level}");
    let ins = instructions(req, false);
    if !ins.is_empty() {
        header += &format!("<|system|>{ins}");
    }
    let open = if think { "<|assistant|><think>" } else { "<|assistant|><think></think>" };
    let msgs: Vec<(String, String)> = req["messages"]
        .as_array()
        .map(|a| a.iter().map(|m| (m["role"].as_str().unwrap_or("user").to_string(), m["content"].as_str().unwrap_or("").to_string())).collect())
        .unwrap_or_default();
    let Some((_, last)) = msgs.last().filter(|(r, _)| r == "user") else { bail!("the last message must be the user's") };

    // continue the session when this chat is the one it last answered
    let cont = g.sess.is_some() && header == g.header && msgs.len() == g.conv.len() + 1 && msgs[..g.conv.len()] == g.conv[..];
    let text = if cont {
        format!("<|user|>{last}{open}")
    } else {
        let mut t = header.clone();
        for (r, c) in &msgs {
            if r == "assistant" {
                t += &format!("<|assistant|><think></think>{}", c.trim());
            } else {
                t += &format!("<|user|>{c}");
            }
        }
        t + open
    };
    let new = g.tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec();
    let base = if cont { g.read.len() } else { 0 };
    if base + new.len() + 8 > g.cap {
        bail!("This conversation is {} tokens; GLM-5.3 runs up to {} tokens here (its sparse-attention indexer is not implemented yet). Start a new chat, or use Off or Low effort for shorter replies.", base + new.len(), g.cap);
    }
    if !cont {
        g.sess = Some(g.model.new_session(g.cap));
        g.read.clear();
        g.header = header;
    }
    let t0 = Instant::now();
    let _ = tx.send(json!({ "t": "start", "prompt": base + new.len(), "reused": base, "ctx": g.cap }));
    let sess = g.sess.as_mut().unwrap();
    let mut timing = Timing::default();
    let mut logits = Vec::new();
    // (until this reply ends on its own, the session can't be continued)
    g.conv.clear();
    for (i, chunk) in new.chunks(CHUNK).enumerate() {
        logits = g.model.forward(sess, chunk, &mut timing)?;
        g.read.extend_from_slice(chunk);
        let _ = tx.send(json!({ "t": "prefill", "done": (i * CHUNK + chunk.len()), "total": new.len() }));
        if stop.load(Ordering::Relaxed) {
            g.sess = None;
            let _ = tx.send(json!({ "t": "done", "prompt": base + new.len(), "reused": base, "generated": 0, "think_tokens": 0, "answer_tokens": 0, "prefill_s": t0.elapsed().as_secs_f64(), "decode_s": 0.0, "tps": 0.0, "ctx_used": g.read.len(), "ctx": g.cap, "stopped": true, "truncated": false }));
            return Ok(());
        }
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let think_open = g.tok.token_to_id("<think>").unwrap_or(u32::MAX);
    let think_close = g.tok.token_to_id("</think>").unwrap_or(u32::MAX);
    let eos = g.model.cfg.eos.clone();
    let n = (req["max_tokens"].as_u64().unwrap_or(4096) as usize).min(g.cap - g.read.len() - 1);
    let mut thinking = think;
    let (mut th, mut ans) = (Delta::new(), Delta::new());
    let (mut think_t0, mut think_t1) = (None::<Instant>, None::<Instant>);
    let (mut gen, mut think_n, mut answer_n, mut ended) = (0usize, 0usize, 0usize, false);
    while gen < n {
        let next = argmax(&logits);
        if eos.contains(&next) {
            ended = true;
            break;
        }
        gen += 1;
        if next == think_close {
            thinking = false;
            if think_t0.is_some() {
                think_t1 = Some(Instant::now());
            }
        } else if next != think_open {
            let (d, kind) = if thinking {
                think_n += 1;
                think_t0.get_or_insert_with(Instant::now);
                (th.push(&g.tok, next), "think")
            } else {
                answer_n += 1;
                (ans.push(&g.tok, next), "text")
            };
            if let Some(d) = d {
                if tx.send(json!({ "t": kind, "d": d })).is_err() {
                    stop.store(true, Ordering::Relaxed);
                }
            }
        }
        // every generated token goes through the model, so the session
        // always holds the whole reply
        logits = g.model.forward(sess, &[next], &mut timing)?;
        g.read.push(next);
        if stop.load(Ordering::Relaxed) {
            break;
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    // a reply that did not end on its own leaves the session mid-message
    if ended && !stop.load(Ordering::Relaxed) {
        g.conv = msgs;
        g.conv.push(("assistant".into(), ans.text.clone()));
    } else {
        g.sess = None;
    }
    if let Err(e) = g.model.experts.save_usage(&g.dir.join(".pelica_usage")) {
        eprintln!("[pelica] could not save routing history: {e}");
    }
    let stopped = stop.load(Ordering::Relaxed);
    let _ = tx.send(json!({
        "t": "done",
        "prompt": base + new.len(), "reused": base, "generated": gen,
        "think_tokens": think_n, "answer_tokens": answer_n,
        "think_s": think_t0.map(|a| (think_t1.unwrap_or_else(Instant::now) - a).as_secs_f64()),
        "first_s": prefill_s,
        "prefill_s": prefill_s, "decode_s": decode_s,
        "tps": gen as f64 / decode_s.max(1e-9),
        "ctx_used": g.read.len(), "ctx": g.cap,
        "stopped": stopped, "truncated": !stopped && !ended,
    }));
    Ok(())
}

// ---------- HTTP ----------

fn respond(s: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> Result<()> {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Service Unavailable",
    };
    write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", body.len())?;
    s.write_all(body)?;
    Ok(())
}

fn respond_json(s: &mut TcpStream, code: u16, v: &Value) -> Result<()> {
    respond(s, code, "application/json", v.to_string().as_bytes())
}

/// A file of the built UI under `dir` (index.html for `/` and for paths
/// that aren't files: the UI routes them itself).
fn serve_ui(s: &mut TcpStream, dir: &Path, path: &str) -> Result<()> {
    let rel = path.split(['?', '#']).next().unwrap_or("/").trim_start_matches('/');
    let rel = crate::web::percent_decode(rel);
    let ok = !rel.split(['/', '\\']).any(|c| c == ".." || c.contains(':'));
    let mut f = dir.join(if rel.is_empty() || !ok { "index.html" } else { rel.as_str() });
    if !f.is_file() {
        f = dir.join("index.html");
    }
    let ctype = match f.extension().and_then(|x| x.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    };
    match std::fs::read(&f) {
        Ok(b) => respond(s, 200, ctype, &b),
        Err(_) => respond(s, 404, "text/plain", b"not found"),
    }
}

fn handle(app: &Arc<App>, mut s: TcpStream) -> Result<()> {
    let mut r = BufReader::new(s.try_clone()?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    let (mut len, mut host, mut origin) = (0usize, String::new(), String::new());
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            let v = v.trim().to_string();
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => len = v.parse().unwrap_or(0),
                "host" => host = v,
                "origin" => origin = v,
                _ => {}
            }
        }
    }
    // only this page may drive the API (no other sites, no DNS rebinding)
    let ours = [format!("127.0.0.1:{}", app.port), format!("localhost:{}", app.port)];
    if !ours.contains(&host) || (!origin.is_empty() && !ours.iter().any(|o| origin == format!("http://{o}"))) {
        return respond(&mut s, 403, "text/plain", b"forbidden");
    }
    let mut body = vec![0u8; len.min(16 << 20)];
    r.read_exact(&mut body)?;
    let body: Value = if body.is_empty() { Value::Null } else { serde_json::from_slice(&body).unwrap_or(Value::Null) };
    app.seen.store(now_s(), Ordering::Relaxed);

    match (method.as_str(), path.as_str()) {
        ("GET", p) if app.ui.is_some() && !p.starts_with("/api/") => serve_ui(&mut s, app.ui.as_ref().unwrap(), p),
        ("GET", "/") => respond(&mut s, 200, "text/html; charset=utf-8", PAGE.as_bytes()),
        ("GET", "/api/status") => {
            let mut st = app.status.lock().unwrap().clone();
            let moe = st["kind"] == "moe";
            st["progress"] = json!(if moe { crate::moe::LOAD_PROGRESS.load(Ordering::Relaxed) } else { oss::LOAD_PROGRESS.load(Ordering::Relaxed) });
            st["busy"] = json!(app.busy.load(Ordering::Relaxed));
            st["now"] = json!(now_s());
            // chat API v2 (messages + tools, stops at tool calls)
            st["tools"] = json!(true);
            st["engine"] = st["kind"].clone();
            respond_json(&mut s, 200, &st)
        }
        ("GET", "/api/models") => {
            let v: Vec<Value> = find_models(&app.models_dirs)
                .iter()
                .map(|(p, _)| {
                    let (kind, max) = ui_kind(p);
                    json!({ "path": p.to_string_lossy(), "name": display_name(p), "gb": size_of(p) as f64 / 1e9, "kind": kind, "max_ctx": max })
                })
                .collect();
            let dirs: Vec<String> = app.models_dirs.iter().map(|d| d.to_string_lossy().into_owned()).collect();
            respond_json(&mut s, 200, &json!({ "dir": dirs.join("; "), "models": v }))
        }
        ("POST", "/api/load") => {
            let Some(p) = body["path"].as_str().map(PathBuf::from) else { return respond_json(&mut s, 400, &json!({ "error": "path needed" })) };
            if kind_of(&p).is_none() {
                return respond_json(&mut s, 400, &json!({ "error": "not a model Pelica can run" }));
            }
            if app.busy.load(Ordering::Relaxed) || app.status.lock().unwrap()["state"] == "loading" {
                return respond_json(&mut s, 409, &json!({ "error": "busy" }));
            }
            let ctx = body["ctx"].as_u64().unwrap_or(32768).clamp(512, 1 << 20) as usize;
            set_loading(app, &p, ctx);
            app.jobs.lock().unwrap().send(Job::Load { path: p, ctx })?;
            respond_json(&mut s, 200, &json!({ "ok": true }))
        }
        ("POST", "/api/unload") => {
            if app.busy.load(Ordering::Relaxed) || app.status.lock().unwrap()["state"] == "loading" {
                return respond_json(&mut s, 409, &json!({ "error": "busy" }));
            }
            app.jobs.lock().unwrap().send(Job::Unload)?;
            respond_json(&mut s, 200, &json!({ "ok": true }))
        }
        ("POST", "/api/count") => {
            let tok = app.tok.lock().unwrap().clone();
            if body["parts"].is_object() {
                let v = match &tok {
                    Some(t) => v2::count_parts(t, &body["parts"]),
                    None => json!({ "error": "no model loaded" }),
                };
                return respond_json(&mut s, 200, &v);
            }
            let n = match (tok, body["text"].as_str()) {
                (Some(t), Some(x)) => t.encode(x, false).map(|e| e.len()).unwrap_or(0),
                _ => 0,
            };
            respond_json(&mut s, 200, &json!({ "tokens": n }))
        }
        ("POST", "/api/gguf") => {
            let p = PathBuf::from(body["path"].as_str().unwrap_or(""));
            match v2::gguf_info(&p) {
                Ok(v) => respond_json(&mut s, 200, &v),
                Err(e) => respond_json(&mut s, 400, &json!({ "error": format!("{e:#}") })),
            }
        }
        ("POST", "/api/tensor") => {
            let p = PathBuf::from(body["path"].as_str().unwrap_or(""));
            match v2::tensor_values(&p, body["name"].as_str().unwrap_or("")) {
                Ok(v) => respond_json(&mut s, 200, &v),
                Err(e) => respond_json(&mut s, 400, &json!({ "error": format!("{e:#}") })),
            }
        }
        ("POST", "/api/stop") => {
            app.stop.store(true, Ordering::Relaxed);
            respond_json(&mut s, 200, &json!({ "ok": true }))
        }
        ("POST", "/api/chat") => {
            if app.status.lock().unwrap()["state"] != "ready" {
                return respond_json(&mut s, 503, &json!({ "error": "no model loaded" }));
            }
            if app.busy.swap(true, Ordering::Relaxed) {
                return respond_json(&mut s, 409, &json!({ "error": "already answering" }));
            }
            let (tx, rx) = channel();
            if app.jobs.lock().unwrap().send(Job::Chat { req: body, tx }).is_err() {
                app.busy.store(false, Ordering::Relaxed);
                return respond_json(&mut s, 503, &json!({ "error": "engine stopped" }));
            }
            write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n")?;
            for ev in rx {
                if write!(s, "data: {ev}\n\n").and_then(|_| s.flush()).is_err() {
                    // the page went away: stop, and let the engine finish
                    app.stop.store(true, Ordering::Relaxed);
                }
            }
            Ok(())
        }
        ("POST", "/api/trace") => {
            // the experts each token of a text goes to (one answer, not a stream)
            if app.busy.swap(true, Ordering::Relaxed) {
                return respond_json(&mut s, 409, &json!({ "error": "the model is busy answering" }));
            }
            let (tx, rx) = channel();
            let req = json!({ "trace": body["text"].as_str().unwrap_or("") });
            if app.jobs.lock().unwrap().send(Job::Chat { req, tx }).is_err() {
                app.busy.store(false, Ordering::Relaxed);
                return respond_json(&mut s, 503, &json!({ "error": "engine stopped" }));
            }
            match rx.recv() {
                Ok(v) if v["t"] == "error" => respond_json(&mut s, 400, &json!({ "error": v["error"] })),
                Ok(v) => respond_json(&mut s, 200, &v),
                Err(_) => respond_json(&mut s, 500, &json!({ "error": "the engine stopped" })),
            }
        }
        _ => respond(&mut s, 404, "text/plain", b"not found"),
    }
}

#[cfg(test)]
mod tests {
    use super::ThinkSplit;

    fn run(start: bool, pieces: &[&str]) -> (String, String) {
        let mut s = ThinkSplit { think: start, buf: String::new(), answered: false };
        let (mut think, mut text) = (String::new(), String::new());
        let mut evs: Vec<(bool, String)> = pieces.iter().flat_map(|p| s.feed(p)).collect();
        evs.extend(s.finish());
        for (t, d) in evs {
            if t { think += &d } else { text += &d }
        }
        (think, text)
    }

    #[test]
    fn think_split() {
        assert_eq!(run(false, &["<th", "ink>a b", "</thi", "nk>\n\nAns", "wer <x>"]), ("a b".into(), "Answer <x>".into()));
        assert_eq!(run(true, &["reason", "ing</think>", "\nok"]), ("reasoning".into(), "ok".into()));
        assert_eq!(run(false, &["plain ", "text <", "b>"]), (String::new(), "plain text <b>".into()));
    }

    /// (visible text, the call) after feeding `pieces`
    fn gate(pieces: &[&str]) -> (String, Option<String>) {
        let mut g = super::ToolGate::new();
        let mut vis: String = pieces.iter().map(|p| g.feed(p)).collect();
        vis += &g.finish();
        (vis, g.call)
    }

    #[test]
    fn tool_gate() {
        let call = r#"{"name": "web_search", "arguments": {"query": "x"}}"#;
        assert_eq!(gate(&["Let me check.\n<tool", "_call>\n{\"name\": \"web_search\", ", "\"arguments\": {\"query\": \"x\"}}\n</tool_", "call>", "tail"]), ("Let me check.\n".into(), Some(call.into())));
        assert_eq!(gate(&["<tool_call>", call]), (String::new(), Some(call.into())));
        assert_eq!(gate(&["no call <to", "ol> here"]), ("no call <tool> here".into(), None));
    }
}
