//! Pelica Bridge: another computer's RAM (an old laptop, a second PC) as a
//! fourth expert tier for DeepSeek V4 and Qwen3.8, between this PC's RAM and
//! its drive.
//!
//! The other computer runs `pelica bridge`. It holds experts in its RAM and
//! works them itself with the same CPU code as the engine (`ds4_cpu::expert`,
//! or `q4x_cpu::expert` for Qwen3.8, whose experts differ in type and size
//! by layer), so per layer only the token's activations go out (16 KB a
//! token, 10 KB for Qwen3.8) and one summed output comes back, never the
//! experts (14 MB, 2.3 MB). Over a
//! wired network that costs about a millisecond a layer, and it runs while
//! this PC works its own experts and reads from its drive.
//!
//! Wire format (TCP, little-endian): frames of [u32 length][u8 kind][body].
//! A host opens two connections per device: one for RUN (the engine's
//! thread, one request at a time) and one for PUT/DROP (filling the device
//! in the background). Both start with HELLO and the device's pairing code.
//! The device accepts nothing else: no files, no commands.

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::ds4_cpu::{self as cpu, Q8x};

pub const PORT: u16 = 8650;
const DISCOVER_PORT: u16 = 8651;
/// 2: RUN_OK ends with the experts the device didn't hold (dropped meanwhile)
/// 3: SETUP says the model's kind, and for Qwen3.8 each layer's expert types and sizes
const VERSION: u32 = 3;
/// model kinds: DeepSeek V4 (one expert size), Qwen3.8 (types and sizes by layer)
pub const KIND_DS4: u32 = 0;
pub const KIND_Q4X: u32 = 1;
const MAX_FRAME: usize = 64 << 20;

const HELLO: u8 = 1;
const HELLO_OK: u8 = 2;
const SETUP: u8 = 3;
const PUT: u8 = 4;
const DROP: u8 = 5;
const RUN: u8 = 6;
const RUN_OK: u8 = 7;
const PUT_OK: u8 = 8;
const ERR: u8 = 9;
const SETUP_OK: u8 = 10;
const FULL: u8 = 11;

// ---------- frames ----------

#[derive(Default)]
struct Wr(Vec<u8>);

impl Wr {
    fn u16(&mut self, v: u16) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn f32(&mut self, v: f32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    fn str(&mut self, s: &str) -> &mut Self {
        let b = &s.as_bytes()[..s.len().min(u16::MAX as usize)];
        self.u16(b.len() as u16);
        self.0.extend_from_slice(b);
        self
    }
    fn f32s(&mut self, v: &[f32]) -> &mut Self {
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
        self
    }
}

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Rd<'a> {
    fn new(b: &'a [u8]) -> Self {
        Rd { b, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.at + n > self.b.len() {
            bail!("message too short");
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn rest(&mut self) -> &'a [u8] {
        let s = &self.b[self.at..];
        self.at = self.b.len();
        s
    }
}

fn send(w: &mut impl Write, kind: u8, body: &[u8]) -> Result<()> {
    w.write_all(&((body.len() + 1) as u32).to_le_bytes())?;
    w.write_all(&[kind])?;
    w.write_all(body)?;
    w.flush()?;
    Ok(())
}

fn recv(r: &mut impl Read, max: usize) -> Result<(u8, Vec<u8>)> {
    let mut h = [0u8; 5];
    r.read_exact(&mut h)?;
    let n = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
    if n == 0 || n > max {
        bail!("bad message size {n}");
    }
    let mut b = vec![0u8; n - 1];
    r.read_exact(&mut b)?;
    Ok((h[4], b))
}

fn this_name() -> String {
    std::env::var("COMPUTERNAME").unwrap_or_else(|_| "this computer".into())
}

/// The address other computers on the network reach this one at (the
/// interface with the default route; nothing is sent).
fn local_ip() -> Option<String> {
    let s = UdpSocket::bind("0.0.0.0:0").ok()?;
    s.connect("8.8.8.8:80").ok()?;
    Some(s.local_addr().ok()?.ip().to_string())
}

// ---------- the device: `pelica bridge` ----------

#[derive(Default)]
struct Dev {
    model: u64,
    d: usize,
    ff: usize,
    lens: [usize; 3],
    kind: u32,
    /// Qwen3.8: experts per layer, and each layer's piece types (ggml ids) and sizes
    n_exp: usize,
    layers: Vec<([u32; 3], [usize; 3])>,
    experts: HashMap<u32, Arc<Vec<u8>>>,
    used: u64,
    cap: u64,
}

impl Dev {
    /// an expert's piece sizes
    fn lens_of(&self, key: u32) -> Option<[usize; 3]> {
        if self.kind == KIND_Q4X {
            self.layers.get(key as usize / self.n_exp.max(1)).map(|l| l.1)
        } else {
            Some(self.lens)
        }
    }
}

/// `pelica bridge [--port N] [--ram-gb G] [--code C] [--name S] [--bind A]`:
/// lends this computer's RAM and CPU to Pelica on another PC (`--bind
/// 127.0.0.1` for one on this PC only, as tests do: not reachable from the
/// network, nothing for the firewall to ask about).
pub fn serve(args: &[String]) -> Result<()> {
    let mut port = PORT;
    let mut gb: Option<f64> = None;
    let mut code: Option<String> = None;
    let mut name = this_name();
    let mut bind = "0.0.0.0".to_string();
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--port" => port = v.parse().context("--port")?,
            "--ram-gb" => gb = Some(v.parse().context("--ram-gb")?),
            "--code" => code = Some(v),
            "--name" => name = v,
            "--bind" => bind = v,
            "-h" | "--help" => {
                println!("pelica bridge [--port {PORT}] [--ram-gb G] [--code C] [--name S]\n  Lends this computer's RAM and CPU to Pelica running DeepSeek V4 or Qwen3.8 on another PC.");
                return Ok(());
            }
            other => bail!("unknown option {other}"),
        }
        i += 2;
    }
    let cap = gb.map(|g| (g * 1e9) as u64).unwrap_or_else(|| crate::available_ram().saturating_sub(3 << 30));
    if cap < 1 << 30 {
        bail!("less than 1 GB of RAM free to lend");
    }
    let code = code.unwrap_or_else(|| {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(7, |d| d.as_nanos() as u64);
        let mut x = t ^ 0x9e37_79b9_7f4a_7c15 ^ std::process::id() as u64;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        format!("{:06}", x % 1_000_000)
    });
    let listener = TcpListener::bind((bind.as_str(), port)).with_context(|| format!("listening on {bind}:{port}"))?;
    let local = bind.starts_with("127.");
    let dev = Arc::new(Mutex::new(Dev { cap, ..Default::default() }));
    let code = Arc::new(code);
    if !local {
        let (name, cap) = (name.clone(), cap);
        std::thread::spawn(move || discover_answer(&name, port, cap));
    }
    println!("Pelica Bridge on {name}");
    println!("  lending up to {:.1} GB of RAM, CPU kernel {}, {} threads", cap as f64 / 1e9, cpu::kernel_name(), rayon::current_num_threads());
    println!("  address: {}:{port}", if local { bind.clone() } else { local_ip().unwrap_or_else(|| "this computer's IP".into()) });
    println!("  pairing code: {}", code);
    println!("  In Pelica on the other PC: Settings > Devices > Add, with this address and code.");
    println!("  A wired connection (Ethernet, or a cable straight between the two) works far better than Wi-Fi.");
    println!("  Keep this window open. Ctrl+C stops lending (Pelica carries on without it).");
    for s in listener.incoming() {
        match s {
            Ok(s) => {
                let (dev, code, name) = (dev.clone(), code.clone(), name.clone());
                std::thread::spawn(move || {
                    let peer = s.peer_addr().map(|a| a.to_string()).unwrap_or_default();
                    if let Err(e) = handle(s, &dev, &code, &name) {
                        eprintln!("[bridge] {peer}: {e:#}");
                    }
                });
            }
            Err(e) => eprintln!("[bridge] {e}"),
        }
    }
    Ok(())
}

/// Answers "PELICA?" broadcasts so the PC can find this computer.
fn discover_answer(name: &str, port: u16, cap: u64) {
    let Ok(s) = UdpSocket::bind(("0.0.0.0", DISCOVER_PORT)) else { return };
    let mut b = [0u8; 64];
    loop {
        if let Ok((n, from)) = s.recv_from(&mut b) {
            if &b[..n] == b"PELICA?" {
                let _ = s.send_to(format!("PELICA!{port}|{:.1}|{name}", cap as f64 / 1e9).as_bytes(), from);
            }
        }
    }
}

fn handle(sock: TcpStream, dev: &Mutex<Dev>, code: &str, name: &str) -> Result<()> {
    let peer = sock.peer_addr()?;
    sock.set_nodelay(true)?;
    let mut r = BufReader::with_capacity(1 << 20, sock.try_clone()?);
    let mut w = BufWriter::with_capacity(1 << 20, sock);
    let (k, b) = recv(&mut r, 4096)?;
    if k != HELLO {
        bail!("not a Pelica host");
    }
    let mut c = Rd::new(&b);
    let (ver, got) = (c.u32()?, c.str()?);
    if ver != VERSION {
        send(&mut w, ERR, Wr::default().str(&format!("this bridge speaks version {VERSION}, the PC {ver}: use the same Pelica version on both")).0.as_slice())?;
        return Ok(());
    }
    if locked_out() {
        send(&mut w, ERR, Wr::default().str("too many wrong pairing codes: try again in a minute").0.as_slice())?;
        bail!("refused: locked out after wrong pairing codes");
    }
    if got.trim() != code {
        wrong_code();
        send(&mut w, ERR, Wr::default().str("wrong pairing code").0.as_slice())?;
        bail!("refused: wrong pairing code");
    }
    {
        let d = dev.lock().unwrap();
        let mut o = Wr::default();
        o.u32(VERSION).u64(d.cap).u64(d.used).u32(rayon::current_num_threads() as u32).str(cpu::kernel_name()).str(name);
        send(&mut w, HELLO_OK, &o.0)?;
    }
    eprintln!("[bridge] {peer} connected");
    loop {
        let (k, b) = match recv(&mut r, MAX_FRAME) {
            Ok(x) => x,
            Err(_) => break,
        };
        let mut c = Rd::new(&b);
        match k {
            SETUP => {
                let model = c.u64()?;
                let (d, ff) = (c.u32()? as usize, c.u32()? as usize);
                let lens = [c.u32()? as usize, c.u32()? as usize, c.u32()? as usize];
                let kind = c.u32()?;
                let (mut n_exp, mut layers) = (0usize, Vec::new());
                if kind == KIND_Q4X {
                    n_exp = c.u32()? as usize;
                    let nl = c.u32()? as usize;
                    for _ in 0..nl {
                        let tys = [c.u32()?, c.u32()?, c.u32()?];
                        let ls = [c.u32()? as usize, c.u32()? as usize, c.u32()? as usize];
                        if tys.iter().any(|&t| crate::gguf::Ty::from_id(t).map_or(true, |t| !crate::q4x_cpu::supports(t))) {
                            send(&mut w, ERR, Wr::default().str("this bridge can't work these experts: update Pelica here").0.as_slice())?;
                            bail!("unsupported expert types");
                        }
                        layers.push((tys, ls));
                    }
                } else if kind != KIND_DS4 {
                    send(&mut w, ERR, Wr::default().str("an unknown kind of model: update Pelica here").0.as_slice())?;
                    bail!("unknown model kind {kind}");
                }
                let mut dv = dev.lock().unwrap();
                if dv.model != model || dv.d != d || dv.ff != ff || dv.lens != lens || dv.kind != kind || dv.layers != layers {
                    // another model: what's here is no use to it
                    dv.experts.clear();
                    dv.used = 0;
                    (dv.model, dv.d, dv.ff, dv.lens, dv.kind, dv.n_exp, dv.layers) = (model, d, ff, lens, kind, n_exp, layers);
                }
                let mut o = Wr::default();
                o.u32(dv.experts.len() as u32);
                for &key in dv.experts.keys() {
                    o.u32(key);
                }
                drop(dv);
                send(&mut w, SETUP_OK, &o.0)?;
            }
            PUT => {
                let key = c.u32()?;
                let bytes = c.rest();
                let mut dv = dev.lock().unwrap();
                let want: usize = dv.lens_of(key).map_or(0, |l| l.iter().sum());
                if dv.model == 0 || want == 0 || bytes.len() != want {
                    drop(dv);
                    send(&mut w, ERR, Wr::default().str("expert of the wrong size").0.as_slice())?;
                    continue;
                }
                if !dv.experts.contains_key(&key) {
                    if dv.used + bytes.len() as u64 > dv.cap {
                        drop(dv);
                        send(&mut w, FULL, Wr::default().u32(key).0.as_slice())?;
                        continue;
                    }
                    dv.used += bytes.len() as u64;
                    dv.experts.insert(key, Arc::new(bytes.to_vec()));
                }
                drop(dv);
                send(&mut w, PUT_OK, Wr::default().u32(key).0.as_slice())?;
            }
            DROP => {
                let key = c.u32()?;
                let mut dv = dev.lock().unwrap();
                if let Some(e) = dv.experts.remove(&key) {
                    dv.used -= e.len() as u64;
                }
            }
            RUN => match run(dev, &b) {
                Ok(o) => send(&mut w, RUN_OK, &o)?,
                Err(e) => send(&mut w, ERR, Wr::default().str(&format!("{e:#}")).0.as_slice())?,
            },
            _ => bail!("unknown message {k}"),
        }
    }
    eprintln!("[bridge] {peer} left");
    Ok(())
}

/// Wrong pairing codes in the last minute (a 6-digit code would otherwise
/// fall to guessing): each one waits a second, and after 10 the bridge
/// refuses everyone for a minute.
static WRONG: Mutex<Vec<Instant>> = Mutex::new(Vec::new());

fn wrong_code() {
    {
        let mut w = WRONG.lock().unwrap();
        w.retain(|t| t.elapsed() < Duration::from_secs(60));
        w.push(Instant::now());
    }
    std::thread::sleep(Duration::from_secs(1));
}

fn locked_out() -> bool {
    let mut w = WRONG.lock().unwrap();
    w.retain(|t| t.elapsed() < Duration::from_secs(60));
    w.len() >= 10
}

/// RUN: req, tokens, experts; the tokens' activations; per expert its key,
/// clamp and (token, weight) rows. Answers with the summed outputs of the
/// experts held here, then the keys of any that aren't (the PC dropped them
/// while this RUN was on its way: it works those itself).
fn run(dev: &Mutex<Dev>, b: &[u8]) -> Result<Vec<u8>> {
    let mut c = Rd::new(b);
    let (req, nt, ne) = (c.u32()?, c.u16()? as usize, c.u16()? as usize);
    let (d, ff, kind) = {
        let dv = dev.lock().unwrap();
        (dv.d, dv.ff, dv.kind)
    };
    if d == 0 {
        bail!("not set up");
    }
    let x = c.f32s(nt * d)?;
    if kind == KIND_Q4X {
        return run_q4x(dev, &mut c, req, nt, ne, &x);
    }
    let lens = dev.lock().unwrap().lens;
    let qs: Vec<Q8x> = x.chunks(d).map(Q8x::new).collect();
    let mut jobs = Vec::with_capacity(ne);
    let mut missing = Vec::new();
    for _ in 0..ne {
        let (key, lim, nr) = (c.u32()?, c.f32()?, c.u16()? as usize);
        let mut rows = Vec::with_capacity(nr);
        for _ in 0..nr {
            let (t, wt) = (c.u16()? as usize, c.f32()?);
            if t >= nt {
                bail!("token out of range");
            }
            rows.push((t, wt));
        }
        match dev.lock().unwrap().experts.get(&key).cloned() {
            Some(e) => jobs.push((e, lim, rows)),
            None => missing.push(key),
        }
    }
    let mut y = vec![0f32; nt * d];
    for (e, lim, rows) in &jobs {
        let (gw, rest) = e.split_at(lens[0]);
        let (uw, dw) = rest.split_at(lens[1]);
        let xs: Vec<&Q8x> = rows.iter().map(|&(t, _)| &qs[t]).collect();
        let wts: Vec<f32> = rows.iter().map(|&(_, w)| w).collect();
        let o = cpu::expert(gw, uw, dw, d, ff, &xs, &wts, *lim);
        for (j, &(t, _)) in rows.iter().enumerate() {
            y[t * d..(t + 1) * d].iter_mut().zip(&o[j * d..(j + 1) * d]).for_each(|(a, b)| *a += b);
        }
    }
    let mut o = Wr::default();
    o.u32(req).u16(nt as u16).f32s(&y).u16(missing.len() as u16);
    for k in missing {
        o.u32(k);
    }
    Ok(o.0)
}

/// RUN for Qwen3.8: as `run`, each expert worked with its layer's types.
fn run_q4x(dev: &Mutex<Dev>, c: &mut Rd, req: u32, nt: usize, ne: usize, x: &[f32]) -> Result<Vec<u8>> {
    use rayon::prelude::*;
    let (d, ff, n_exp, layers) = {
        let dv = dev.lock().unwrap();
        (dv.d, dv.ff, dv.n_exp, dv.layers.clone())
    };
    let qs: Vec<crate::q4x_cpu::Q8> = x.chunks(d).map(crate::q4x_cpu::Q8::new).collect();
    let mut jobs = Vec::with_capacity(ne);
    let mut missing = Vec::new();
    for _ in 0..ne {
        let (key, _lim, nr) = (c.u32()?, c.f32()?, c.u16()? as usize);
        let mut rows = Vec::with_capacity(nr);
        for _ in 0..nr {
            let (t, wt) = (c.u16()? as usize, c.f32()?);
            if t >= nt {
                bail!("token out of range");
            }
            rows.push((t, wt));
        }
        let Some((tys, lens)) = layers.get(key as usize / n_exp.max(1)).copied() else { bail!("expert out of range") };
        match dev.lock().unwrap().experts.get(&key).cloned() {
            Some(e) => jobs.push((e, tys, lens, rows)),
            None => missing.push(key),
        }
    }
    let ty = |id: u32| crate::gguf::Ty::from_id(id).unwrap_or(crate::gguf::Ty::F32);
    let outs: Vec<Vec<f32>> = jobs
        .par_iter()
        .map(|(e, tys, lens, rows)| {
            let (gw, rest) = e.split_at(lens[0]);
            let (uw, dw) = rest.split_at(lens[1]);
            let xs: Vec<&crate::q4x_cpu::Q8> = rows.iter().map(|&(t, _)| &qs[t]).collect();
            let wts: Vec<f32> = rows.iter().map(|&(_, w)| w).collect();
            crate::q4x_cpu::expert([ty(tys[0]), ty(tys[1]), ty(tys[2])], gw, uw, dw, d, ff, &xs, &wts)
        })
        .collect();
    let mut y = vec![0f32; nt * d];
    for ((_, _, _, rows), o) in jobs.iter().zip(&outs) {
        for (j, &(t, _)) in rows.iter().enumerate() {
            y[t * d..(t + 1) * d].iter_mut().zip(&o[j * d..(j + 1) * d]).for_each(|(a, b)| *a += b);
        }
    }
    let mut o = Wr::default();
    o.u32(req).u16(nt as u16).f32s(&y).u16(missing.len() as u16);
    for k in missing {
        o.u32(k);
    }
    Ok(o.0)
}

// ---------- the PC: devices it borrows from ----------

/// A device to borrow from: its address (host or host:port) and pairing code.
#[derive(Clone, PartialEq)]
pub struct DeviceCfg {
    pub addr: String,
    pub code: String,
}

static CFG: Mutex<Option<Vec<DeviceCfg>>> = Mutex::new(None);

/// The devices to use (set by the app; PELICA_DEVICES="host/code,host:port/code" otherwise).
pub fn config() -> Vec<DeviceCfg> {
    let mut c = CFG.lock().unwrap();
    c.get_or_insert_with(|| {
        std::env::var("PELICA_DEVICES")
            .unwrap_or_default()
            .split(',')
            .filter_map(|s| {
                let (a, code) = s.trim().split_once('/')?;
                Some(DeviceCfg { addr: a.trim().into(), code: code.trim().into() })
            })
            .collect()
    })
    .clone()
}

pub fn set_config(v: Vec<DeviceCfg>) {
    *CFG.lock().unwrap() = Some(v);
}

/// The loaded engine's devices, if a DeepSeek V4 or Qwen3.8 model is loaded.
static CURRENT: Mutex<Option<Weak<Remote>>> = Mutex::new(None);

/// For the app: every configured device, with its state when a model uses it.
pub fn status_json() -> serde_json::Value {
    let live = CURRENT.lock().unwrap().as_ref().and_then(|w| w.upgrade());
    let st: Vec<serde_json::Value> = live.as_ref().map(|r| r.status().as_array().cloned().unwrap_or_default()).unwrap_or_default();
    let list: Vec<serde_json::Value> = config()
        .iter()
        .map(|c| {
            let mut v = st.iter().find(|s| s["addr"] == c.addr.as_str()).cloned().unwrap_or_else(|| serde_json::json!({ "addr": c.addr, "state": "waiting" }));
            v["code"] = serde_json::json!(c.code);
            v
        })
        .collect();
    serde_json::json!({ "active": live.is_some(), "devices": list })
}

/// Computers on this network running `pelica bridge`: (address, GB lent, name).
pub fn scan() -> Vec<(String, f64, String)> {
    let mut out = Vec::new();
    let Ok(s) = UdpSocket::bind("0.0.0.0:0") else { return out };
    let _ = s.set_broadcast(true);
    let _ = s.set_read_timeout(Some(Duration::from_millis(300)));
    for _ in 0..2 {
        let _ = s.send_to(b"PELICA?", ("255.255.255.255", DISCOVER_PORT));
    }
    let t0 = Instant::now();
    let mut b = [0u8; 256];
    while t0.elapsed() < Duration::from_millis(1200) {
        if let Ok((n, from)) = s.recv_from(&mut b) {
            let msg = String::from_utf8_lossy(&b[..n]);
            if let Some(rest) = msg.strip_prefix("PELICA!") {
                let mut p = rest.splitn(3, '|');
                let port = p.next().unwrap_or("8650");
                let gb = p.next().and_then(|g| g.parse().ok()).unwrap_or(0.0);
                let name = p.next().unwrap_or("").to_string();
                let addr = format!("{}:{port}", from.ip());
                if !out.iter().any(|(a, _, _)| *a == addr) {
                    out.push((addr, gb, name));
                }
            }
        }
    }
    out
}

struct Chan {
    r: BufReader<TcpStream>,
    w: BufWriter<TcpStream>,
}

#[derive(Default, Clone)]
struct Info {
    name: String,
    kernel: String,
    threads: u32,
    cap: u64,
    used: u64,
    state: &'static str,
    error: String,
    full: bool,
}

/// One device in use.
pub struct Link {
    cfg: DeviceCfg,
    info: Mutex<Info>,
    ctl: Mutex<Option<Chan>>,
    alive: AtomicBool,
    runs: AtomicU64,
    run_ns: AtomicU64,
    experts_run: AtomicU64,
    retry_at: Mutex<Option<Instant>>,
}

/// The engine's devices: which experts each holds, filling them in the
/// background from the model file, in heat order.
pub struct Remote {
    model: u64,
    d: usize,
    ff: usize,
    lens: [usize; 3],
    kind: u32,
    /// Qwen3.8: each layer's piece types (ggml ids)
    tys: Vec<[u32; 3]>,
    /// the model's files (the layout's `file` indexes them)
    paths: Vec<PathBuf>,
    lay: crate::ds4_io::Layout,
    links: Mutex<Vec<Arc<Link>>>,
    owner: Mutex<HashMap<u32, Arc<Link>>>,
    /// experts the devices should hold, wanted most first
    want: Mutex<Vec<u32>>,
    stop: AtomicBool,
}

fn connect(cfg: &DeviceCfg, timeout: Duration) -> Result<(Chan, Vec<u8>)> {
    let addr = if cfg.addr.contains(':') { cfg.addr.clone() } else { format!("{}:{PORT}", cfg.addr) };
    let sa = addr.to_socket_addrs().with_context(|| format!("can't find {addr}"))?.next().with_context(|| format!("can't find {addr}"))?;
    let s = TcpStream::connect_timeout(&sa, timeout).with_context(|| format!("can't reach {addr} (is `pelica bridge` running there, and allowed through its firewall?)"))?;
    s.set_nodelay(true)?;
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut ch = Chan { r: BufReader::with_capacity(1 << 20, s.try_clone()?), w: BufWriter::with_capacity(1 << 20, s) };
    let mut h = Wr::default();
    h.u32(VERSION).str(&cfg.code);
    send(&mut ch.w, HELLO, &h.0)?;
    let (k, b) = recv(&mut ch.r, 1 << 16)?;
    if k == ERR {
        bail!("{}", Rd::new(&b).str().unwrap_or_default());
    }
    if k != HELLO_OK {
        bail!("not a Pelica bridge");
    }
    Ok((ch, b))
}

impl Remote {
    /// `lay`: where each expert's pieces sit in the file.
    pub fn new(path: PathBuf, model: u64, d: usize, ff: usize, lay: crate::ds4_io::Layout) -> Option<Arc<Remote>> {
        let lens = [lay.pieces[0][0].1, lay.pieces[0][1].1, lay.pieces[0][2].1];
        if lay.pieces.iter().any(|p| [p[0].1, p[1].1, p[2].1] != lens) {
            return None;
        }
        Self::start_new(vec![path], model, d, ff, lay, KIND_DS4, Vec::new())
    }

    /// For Qwen3.8: `tys` holds each layer's piece types (ggml ids), and
    /// the layout may span several files.
    pub fn new_q4x(paths: Vec<PathBuf>, model: u64, d: usize, ff: usize, lay: crate::ds4_io::Layout, tys: Vec<[u32; 3]>) -> Option<Arc<Remote>> {
        if tys.len() != lay.pieces.len() {
            return None;
        }
        Self::start_new(paths, model, d, ff, lay, KIND_Q4X, tys)
    }

    fn start_new(paths: Vec<PathBuf>, model: u64, d: usize, ff: usize, lay: crate::ds4_io::Layout, kind: u32, tys: Vec<[u32; 3]>) -> Option<Arc<Remote>> {
        let lens = [lay.pieces[0][0].1, lay.pieces[0][1].1, lay.pieces[0][2].1];
        let r = Arc::new(Remote { model, d, ff, lens, kind, tys, paths, lay, links: Mutex::new(Vec::new()), owner: Mutex::new(HashMap::new()), want: Mutex::new(Vec::new()), stop: AtomicBool::new(false) });
        let weak = Arc::downgrade(&r);
        *CURRENT.lock().unwrap() = Some(weak.clone());
        std::thread::Builder::new().name("pelica-bridge".into()).spawn(move || manage(weak)).ok()?;
        Some(r)
    }

    /// The device holding `key`, if one does and it's up.
    pub fn holder(&self, key: u32) -> Option<Arc<Link>> {
        self.owner.lock().unwrap().get(&key).filter(|l| l.alive.load(Ordering::Relaxed)).cloned()
    }

    pub fn has(&self, key: u32) -> bool {
        self.holder(key).is_some()
    }

    pub fn held(&self) -> Vec<u32> {
        self.owner.lock().unwrap().iter().filter(|(_, l)| l.alive.load(Ordering::Relaxed)).map(|(&k, _)| k).collect()
    }

    /// An expert's bytes.
    fn per_of(&self, key: u32) -> u64 {
        let l = key as usize / self.lay.n_exp.max(1);
        self.lay.pieces.get(l).map_or(0, |p| p.iter().map(|x| x.1).sum::<usize>()) as u64
    }

    /// An expert's bytes on average, for sizing.
    fn per(&self) -> u64 {
        (self.lay.pieces.iter().map(|p| p.iter().map(|x| x.1).sum::<usize>()).sum::<usize>() / self.lay.pieces.len().max(1)) as u64
    }

    /// Experts held across the devices that are up (for the engine's sizing).
    pub fn capacity(&self) -> usize {
        let per = self.per();
        self.links.lock().unwrap().iter().filter(|l| l.alive.load(Ordering::Relaxed)).map(|l| l.info.lock().unwrap().cap / per.max(1)).sum::<u64>() as usize
    }

    pub fn set_want(&self, keys: Vec<u32>) {
        *self.want.lock().unwrap() = keys;
    }

    /// Sends a RUN: `x` holds the tokens' activations, `experts` per expert
    /// its key, clamp and (token, weight) rows.
    pub fn start(&self, link: &Link, nt: usize, x: &[f32], experts: &[(u32, f32, Vec<(u16, f32)>)]) -> Result<()> {
        let mut o = Wr::default();
        o.u32(0).u16(nt as u16).u16(experts.len() as u16).f32s(&x[..nt * self.d]);
        for (key, lim, rows) in experts {
            o.u32(*key).f32(*lim).u16(rows.len() as u16);
            for &(t, w) in rows {
                o.u16(t).f32(w);
            }
        }
        let mut g = link.ctl.lock().unwrap();
        let ch = g.as_mut().context("not connected")?;
        send(&mut ch.w, RUN, &o.0)
    }

    /// The summed outputs [token][d] of the RUN sent last to `link`, and the
    /// experts it no longer held (to work here).
    pub fn finish(&self, link: &Link, t0: Instant, n_exp: usize) -> Result<(Vec<f32>, Vec<u32>)> {
        let mut g = link.ctl.lock().unwrap();
        let ch = g.as_mut().context("not connected")?;
        let (k, b) = recv(&mut ch.r, MAX_FRAME)?;
        let mut c = Rd::new(&b);
        if k == ERR {
            bail!("{}", c.str()?);
        }
        if k != RUN_OK {
            bail!("unexpected answer");
        }
        let (_req, nt) = (c.u32()?, c.u16()? as usize);
        let y = c.f32s(nt * self.d)?;
        let nm = c.u16()? as usize;
        let mut missing = Vec::with_capacity(nm);
        for _ in 0..nm {
            missing.push(c.u32()?);
        }
        link.runs.fetch_add(1, Ordering::Relaxed);
        link.experts_run.fetch_add((n_exp - nm.min(n_exp)) as u64, Ordering::Relaxed);
        link.run_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok((y, missing))
    }

    /// A device stopped answering: its experts go back to this PC (the
    /// manager tries it again later).
    pub fn fail(&self, link: &Arc<Link>, err: &str) {
        if link.alive.swap(false, Ordering::Relaxed) {
            eprintln!("[pelica] device {} dropped out ({err}); its experts run here", link.cfg.addr);
        }
        *link.ctl.lock().unwrap() = None;
        let mut i = link.info.lock().unwrap();
        i.state = "error";
        i.error = err.to_string();
        i.used = 0;
        drop(i);
        self.owner.lock().unwrap().retain(|_, l| !Arc::ptr_eq(l, link));
        *link.retry_at.lock().unwrap() = Some(Instant::now() + Duration::from_secs(10));
    }

    /// For the app: each device's state.
    pub fn status(&self) -> serde_json::Value {
        let owner = self.owner.lock().unwrap();
        let links: Vec<serde_json::Value> = self
            .links
            .lock()
            .unwrap()
            .iter()
            .map(|l| {
                let i = l.info.lock().unwrap().clone();
                let held = owner.values().filter(|o| Arc::ptr_eq(o, l)).count();
                let runs = l.runs.load(Ordering::Relaxed);
                serde_json::json!({
                    "addr": l.cfg.addr, "name": i.name, "state": i.state, "error": i.error,
                    "kernel": i.kernel, "threads": i.threads,
                    "lend_gb": i.cap as f64 / 1e9, "used_gb": i.used as f64 / 1e9,
                    "experts": held, "full": i.full,
                    "runs": runs, "experts_run": l.experts_run.load(Ordering::Relaxed),
                    "avg_ms": if runs > 0 { l.run_ns.load(Ordering::Relaxed) as f64 / runs as f64 / 1e6 } else { 0.0 },
                })
            })
            .collect();
        serde_json::json!(links)
    }

    fn expert_bytes(&self, files: &[std::fs::File], key: u32) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.per_of(key) as usize);
        for p in 0..3 {
            let f = files.get(self.lay.file_of(key, p)).context("a model part is missing")?;
            let (at, len) = self.lay.span(key, p);
            let mut buf = vec![0u8; len];
            let mut got = 0;
            while got < len {
                #[cfg(windows)]
                let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[got..], at + got as u64)?;
                #[cfg(not(windows))]
                let n = std::os::unix::fs::FileExt::read_at(f, &mut buf[got..], at + got as u64)?;
                if n == 0 {
                    bail!("short read");
                }
                got += n;
            }
            out.extend_from_slice(&buf);
        }
        Ok(out)
    }
}

impl Drop for Remote {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Background: keeps the links in step with the configured devices
/// (connecting, retrying ones that dropped out), and fills each with the
/// wanted experts it doesn't hold, swapping out ones no longer wanted.
fn manage(weak: Weak<Remote>) {
    let mut data: HashMap<String, Chan> = HashMap::new();
    let swapped: Mutex<HashMap<String, Instant>> = Mutex::new(HashMap::new());
    let mut file: Option<Vec<std::fs::File>> = None;
    loop {
        let Some(r) = weak.upgrade() else { return };
        if r.stop.load(Ordering::Relaxed) {
            return;
        }
        if file.is_none() {
            file = r.paths.iter().map(std::fs::File::open).collect::<std::io::Result<Vec<_>>>().ok();
        }
        // links: add new devices, drop removed ones, (re)connect
        let cfg = config();
        {
            let mut links = r.links.lock().unwrap();
            let gone: Vec<Arc<Link>> = links.iter().filter(|l| !cfg.contains(&l.cfg)).cloned().collect();
            for l in &gone {
                l.alive.store(false, Ordering::Relaxed);
                r.owner.lock().unwrap().retain(|_, o| !Arc::ptr_eq(o, l));
                data.remove(&l.cfg.addr);
            }
            links.retain(|l| cfg.contains(&l.cfg));
            for c in &cfg {
                if !links.iter().any(|l| l.cfg == *c) {
                    links.push(Arc::new(Link {
                        cfg: c.clone(),
                        info: Mutex::new(Info { state: "connecting", ..Default::default() }),
                        ctl: Mutex::new(None),
                        alive: AtomicBool::new(false),
                        runs: AtomicU64::new(0),
                        run_ns: AtomicU64::new(0),
                        experts_run: AtomicU64::new(0),
                        retry_at: Mutex::new(None),
                    }));
                }
            }
        }
        let links: Vec<Arc<Link>> = r.links.lock().unwrap().clone();
        for l in &links {
            if l.alive.load(Ordering::Relaxed) {
                continue;
            }
            if l.retry_at.lock().unwrap().is_some_and(|t| Instant::now() < t) {
                continue;
            }
            match link_up(&r, l) {
                Ok(dch) => {
                    data.insert(l.cfg.addr.clone(), dch);
                }
                Err(e) => {
                    let mut i = l.info.lock().unwrap();
                    i.state = "error";
                    i.error = format!("{e:#}");
                    *l.retry_at.lock().unwrap() = Some(Instant::now() + Duration::from_secs(10));
                }
            }
        }
        // fill: the devices hold the top of the wanted list, as many as fit.
        // An expert goes in while there is room; once full, one leaves only
        // when it has dropped out of that top window and one inside it is
        // missing, at most a couple a second (each is a drive read and a
        // network send), so a shifting list doesn't churn.
        let mut busy = false;
        if let Some(f) = file.as_ref() {
            let want = r.want.lock().unwrap().clone();
            let per = r.per();
            let room: usize = links.iter().filter(|l| l.alive.load(Ordering::Relaxed)).map(|l| (l.info.lock().unwrap().cap / per) as usize).sum();
            let window: HashSet<u32> = want.iter().take(room).copied().collect();
            for l in &links {
                if !l.alive.load(Ordering::Relaxed) {
                    continue;
                }
                let Some(ch) = data.get_mut(&l.cfg.addr) else { continue };
                let taken: HashSet<u32> = r.owner.lock().unwrap().keys().copied().collect();
                let next: Vec<u32> = want.iter().take(room).copied().filter(|k| !taken.contains(k)).take(8).collect();
                for key in next {
                    if r.stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let per = r.per_of(key);
                    let (used, cap) = {
                        let i = l.info.lock().unwrap();
                        (i.used, i.cap)
                    };
                    if used + per > cap {
                        let mut swap = swapped.lock().unwrap();
                        if swap.get(&l.cfg.addr).is_some_and(|t: &Instant| t.elapsed() < Duration::from_millis(500)) {
                            break;
                        }
                        let out = r.owner.lock().unwrap().iter().find(|(k, o)| Arc::ptr_eq(o, l) && !window.contains(k)).map(|(&k, _)| k);
                        let Some(out) = out else { break };
                        r.owner.lock().unwrap().remove(&out);
                        if send(&mut ch.w, DROP, Wr::default().u32(out).0.as_slice()).is_err() {
                            r.fail(l, "lost the connection");
                            break;
                        }
                        let mut i = l.info.lock().unwrap();
                        i.used = i.used.saturating_sub(r.per_of(out));
                        drop(i);
                        swap.insert(l.cfg.addr.clone(), Instant::now());
                    }
                    let bytes = match r.expert_bytes(f, key) {
                        Ok(b) => b,
                        Err(_) => break,
                    };
                    let mut body = Wr::default();
                    body.u32(key);
                    body.0.extend_from_slice(&bytes);
                    let res = send(&mut ch.w, PUT, &body.0).and_then(|_| recv(&mut ch.r, 4096));
                    match res {
                        Ok((PUT_OK, _)) => {
                            r.owner.lock().unwrap().insert(key, l.clone());
                            let mut i = l.info.lock().unwrap();
                            i.used += per;
                            i.full = i.used + per > i.cap;
                            busy = true;
                        }
                        Ok((FULL, _)) => {
                            let mut i = l.info.lock().unwrap();
                            i.used = i.cap;
                            i.full = true;
                            break;
                        }
                        Ok(_) => break,
                        Err(e) => {
                            r.fail(l, &format!("{e:#}"));
                            break;
                        }
                    }
                }
            }
        }
        drop(r);
        std::thread::sleep(Duration::from_millis(if busy { 5 } else { 1000 }));
    }
}

/// Connects a device's two channels and sets it up for this model; returns
/// the data channel (the RUN channel goes into the link).
fn link_up(r: &Remote, l: &Arc<Link>) -> Result<Chan> {
    let to = Duration::from_secs(3);
    let (ctl, hello) = connect(&l.cfg, to)?;
    let (mut dch, _) = connect(&l.cfg, to)?;
    let mut c = Rd::new(&hello);
    let (_ver, cap, used, threads) = (c.u32()?, c.u64()?, c.u64()?, c.u32()?);
    let (kernel, name) = (c.str()?, c.str()?);
    let mut s = Wr::default();
    s.u64(r.model).u32(r.d as u32).u32(r.ff as u32).u32(r.lens[0] as u32).u32(r.lens[1] as u32).u32(r.lens[2] as u32).u32(r.kind);
    if r.kind == KIND_Q4X {
        s.u32(r.lay.n_exp as u32).u32(r.tys.len() as u32);
        for (t, p) in r.tys.iter().zip(&r.lay.pieces) {
            s.u32(t[0]).u32(t[1]).u32(t[2]).u32(p[0].1 as u32).u32(p[1].1 as u32).u32(p[2].1 as u32);
        }
    }
    send(&mut dch.w, SETUP, &s.0)?;
    let (k, b) = recv(&mut dch.r, MAX_FRAME)?;
    if k != SETUP_OK {
        bail!("set-up refused");
    }
    let mut c = Rd::new(&b);
    let n = c.u32()? as usize;
    let mut keys = Vec::with_capacity(n);
    for _ in 0..n {
        keys.push(c.u32()?);
    }
    let per = r.per();
    {
        let mut own = r.owner.lock().unwrap();
        for &k in &keys {
            own.insert(k, l.clone());
        }
    }
    *l.ctl.lock().unwrap() = Some(ctl);
    *l.info.lock().unwrap() = Info { name, kernel, threads, cap, used: used.max(keys.len() as u64 * per), state: "connected", error: String::new(), full: used + per > cap };
    l.alive.store(true, Ordering::Relaxed);
    eprintln!("[pelica] device {} connected: {:.1} GB to lend, {} experts already there", l.cfg.addr, cap as f64 / 1e9, keys.len());
    Ok(dch)
}
