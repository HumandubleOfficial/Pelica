//! Expert reads for the DeepSeek V4 engine (`ds4`): a RAM cache of expert
//! slots kept by heat (use counts), filled from the GGUF by reader threads
//! with unbuffered reads, so the OS file cache doesn't take the RAM.
//!
//! An expert is three pieces (its rows of the gate, up and down tensors),
//! each contiguous in the file. A slot holds the three, each read as the
//! 4 KiB-aligned span around it. Slots in use are pinned; a miss takes the
//! coldest unpinned slot.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use anyhow::{bail, Result};

const ALIGN: usize = 4096;
/// reader threads (two to four large reads in flight saturate an NVMe drive)
const READERS: usize = 4;

/// Where expert pieces sit: per layer, per piece (gate, up, down), the
/// tensor's file offset and the bytes one expert takes.
#[derive(Clone)]
pub struct Layout {
    pub pieces: Vec<[(u64, usize); 3]>,
    pub n_exp: usize,
}

impl Layout {
    fn span(&self, key: u32, p: usize) -> (u64, usize) {
        let (l, e) = (key as usize / self.n_exp, key as usize % self.n_exp);
        let (off, len) = self.pieces[l][p];
        (off + (e * len) as u64, len)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Fill {
    Empty,
    Loading,
    Ready,
    Failed,
}

struct Slot {
    key: u32,
    pins: u32,
    fill: Fill,
    done: u8,
    /// where each piece's data starts in its aligned span
    skip: [u32; 3],
}

struct St {
    slots: Vec<Slot>,
    map: HashMap<u32, usize>,
    heat: Vec<f32>,
    jobs: VecDeque<(usize, u8)>,
    reads: u64,
    hits: u64,
}

struct Shared {
    file: File,
    lay: Layout,
    /// bytes per piece buffer
    cap: usize,
    mem: Vec<usize>,
    st: Mutex<St>,
    work: Condvar,
    ready: Condvar,
    stop: AtomicBool,
}

pub struct Store {
    sh: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

/// An expert's bytes, held in place until dropped.
pub struct Pin {
    sh: Arc<Shared>,
    slot: usize,
    skip: [u32; 3],
    lens: [usize; 3],
}

impl Pin {
    /// piece p (0 gate, 1 up, 2 down)
    pub fn piece(&self, p: usize) -> &[u8] {
        // SAFETY: the slot is pinned and fully read; nothing writes it until
        // the pin is dropped
        unsafe { std::slice::from_raw_parts((self.sh.mem[self.slot] + p * self.sh.cap + self.skip[p] as usize) as *const u8, self.lens[p]) }
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        let mut st = self.sh.st.lock().unwrap();
        st.slots[self.slot].pins -= 1;
    }
}

fn open_unbuffered(path: &std::path::Path) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        std::fs::OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING).open(path)
    }
    #[cfg(not(windows))]
    {
        File::open(path)
    }
}

fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        f.seek_read(buf, off)
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_at(buf, off)
    }
}

impl Store {
    /// A cache of `slots` experts; `heat` seeds each expert's use count.
    pub fn new(path: &std::path::Path, lay: Layout, slots: usize, heat: Vec<f32>) -> Result<Store> {
        let file = open_unbuffered(path)?;
        let piece = lay.pieces.iter().flat_map(|p| p.iter().map(|x| x.1)).max().unwrap_or(0);
        let cap = (piece + 2 * ALIGN).div_ceil(ALIGN) * ALIGN;
        let mut mem = Vec::with_capacity(slots);
        for _ in 0..slots {
            let layout = std::alloc::Layout::from_size_align(3 * cap, ALIGN).unwrap();
            // SAFETY: non-zero size; freed in Drop with the same layout
            let p = unsafe { std::alloc::alloc(layout) };
            if p.is_null() {
                break;
            }
            mem.push(p as usize);
        }
        if mem.len() < 16 {
            for &p in &mem {
                unsafe { std::alloc::dealloc(p as *mut u8, std::alloc::Layout::from_size_align(3 * cap, ALIGN).unwrap()) };
            }
            bail!("not enough RAM for the expert cache");
        }
        let n = mem.len();
        let sh = Arc::new(Shared {
            file,
            lay,
            cap,
            mem,
            st: Mutex::new(St {
                slots: (0..n).map(|_| Slot { key: u32::MAX, pins: 0, fill: Fill::Empty, done: 0, skip: [0; 3] }).collect(),
                map: HashMap::new(),
                heat,
                jobs: VecDeque::new(),
                reads: 0,
                hits: 0,
            }),
            work: Condvar::new(),
            ready: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let threads = (0..READERS)
            .map(|_| {
                let sh = sh.clone();
                std::thread::spawn(move || reader(sh))
            })
            .collect();
        Ok(Store { sh, threads })
    }

    pub fn slots(&self) -> usize {
        self.sh.mem.len()
    }

    pub fn slot_bytes(&self) -> usize {
        3 * self.sh.cap
    }

    /// Counts a use of each expert (key = layer * n_exp + expert).
    pub fn touch(&self, keys: &[u32]) {
        let mut st = self.sh.st.lock().unwrap();
        for &k in keys {
            st.heat[k as usize] += 1.0;
        }
    }

    pub fn heat(&self) -> Vec<f32> {
        self.sh.st.lock().unwrap().heat.clone()
    }

    /// Whether the expert is in RAM now (read or being read).
    pub fn has(&self, key: u32) -> bool {
        self.sh.st.lock().unwrap().map.contains_key(&key)
    }

    /// (reads from the drive, requests served from RAM)
    pub fn counts(&self) -> (u64, u64) {
        let st = self.sh.st.lock().unwrap();
        (st.reads, st.hits)
    }

    /// Pins the experts, starting reads for those not in RAM (in order).
    /// Fails only if every slot is pinned.
    pub fn request(&self, keys: &[u32]) -> Result<Vec<Pin>> {
        let sh = &self.sh;
        let mut st = sh.st.lock().unwrap();
        let mut out = Vec::with_capacity(keys.len());
        for &key in keys {
            let lens = [sh.lay.span(key, 0).1, sh.lay.span(key, 1).1, sh.lay.span(key, 2).1];
            let slot = match st.map.get(&key).copied() {
                Some(s) if st.slots[s].fill != Fill::Failed => {
                    st.hits += 1;
                    s
                }
                _ => {
                    // the coldest slot nobody holds
                    let mut best: Option<(usize, f32)> = None;
                    for (i, s) in st.slots.iter().enumerate() {
                        if s.pins > 0 || s.fill == Fill::Loading {
                            continue;
                        }
                        let h = if s.fill == Fill::Empty || s.fill == Fill::Failed { f32::NEG_INFINITY } else { st.heat[s.key as usize] };
                        if best.is_none_or(|b| h < b.1) {
                            best = Some((i, h));
                        }
                    }
                    let Some((s, _)) = best else { bail!("every expert slot is in use") };
                    let old = st.slots[s].key;
                    if old != u32::MAX && st.map.get(&old) == Some(&s) {
                        st.map.remove(&old);
                    }
                    let mut skip = [0u32; 3];
                    for (p, sk) in skip.iter_mut().enumerate() {
                        *sk = (sh.lay.span(key, p).0 % ALIGN as u64) as u32;
                    }
                    st.slots[s] = Slot { key, pins: 0, fill: Fill::Loading, done: 0, skip };
                    st.map.insert(key, s);
                    for p in 0..3u8 {
                        st.jobs.push_back((s, p));
                    }
                    st.reads += 1;
                    sh.work.notify_all();
                    s
                }
            };
            st.slots[slot].pins += 1;
            out.push(Pin { sh: sh.clone(), slot, skip: st.slots[slot].skip, lens });
        }
        Ok(out)
    }

    /// Waits until the pinned expert's bytes are all in.
    pub fn wait(&self, pin: &Pin) -> Result<()> {
        let mut st = self.sh.st.lock().unwrap();
        loop {
            match st.slots[pin.slot].fill {
                Fill::Ready => return Ok(()),
                Fill::Failed => bail!("reading an expert from the drive failed"),
                _ => st = self.sh.ready.wait(st).unwrap(),
            }
        }
    }

    /// Whether the pinned expert is ready (without waiting).
    pub fn is_ready(&self, pin: &Pin) -> bool {
        self.sh.st.lock().unwrap().slots[pin.slot].fill == Fill::Ready
    }

    /// Reads one expert into a new buffer, outside the cache (warm-up).
    pub fn read_now(&self, key: u32) -> Result<Vec<u8>> {
        let pins = self.request(&[key])?;
        self.wait(&pins[0])?;
        let mut v = Vec::new();
        for p in 0..3 {
            v.extend_from_slice(pins[0].piece(p));
        }
        Ok(v)
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Relaxed);
        {
            let _st = self.sh.st.lock().unwrap();
            self.sh.work.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        let layout = std::alloc::Layout::from_size_align(3 * self.sh.cap, ALIGN).unwrap();
        for &p in &self.sh.mem {
            // SAFETY: allocated in new with this layout; the readers are gone
            unsafe { std::alloc::dealloc(p as *mut u8, layout) };
        }
    }
}

fn reader(sh: Arc<Shared>) {
    loop {
        let (slot, p, key, skip) = {
            let mut st = sh.st.lock().unwrap();
            loop {
                if sh.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some((s, p)) = st.jobs.pop_front() {
                    break (s, p as usize, st.slots[s].key, st.slots[s].skip[p as usize] as usize);
                }
                st = sh.work.wait(st).unwrap();
            }
        };
        let (off, len) = sh.lay.span(key, p);
        let start = off - skip as u64;
        let want = (skip + len).div_ceil(ALIGN) * ALIGN;
        // SAFETY: this slot's piece is written only by this job while the
        // slot is Loading (pinned by its requester, never handed out)
        let buf = unsafe { std::slice::from_raw_parts_mut((sh.mem[slot] + p * sh.cap) as *mut u8, want) };
        let mut got = 0usize;
        let mut ok = true;
        while got < skip + len {
            match read_at(&sh.file, &mut buf[got..], start + got as u64) {
                Ok(0) => {
                    ok = false;
                    break;
                }
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => {
                    eprintln!("[pelica] expert read failed: {e}");
                    ok = false;
                    break;
                }
            }
        }
        let mut st = sh.st.lock().unwrap();
        let s = &mut st.slots[slot];
        if !ok {
            s.fill = Fill::Failed;
        } else {
            s.done += 1;
            if s.done == 3 && s.fill == Fill::Loading {
                s.fill = Fill::Ready;
            }
        }
        sh.ready.notify_all();
    }
}
