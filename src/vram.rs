//! Tier 1 for experts: a fixed set of VRAM slots holding the hottest experts.
//!
//! Every expert carries a heat score: +1 each time a token routes to it,
//! halved every `DECAY_EVERY` tokens, so heat tracks what the conversation is
//! using now. Experts heat up on disk (tier 3), get cached in RAM (tier 2) and,
//! once hotter than the coldest expert in VRAM, are promoted to tier 1, where
//! the GPU runs them. Cooled-off experts are the ones that get replaced.

use crate::experts::{piece_len, ExpertBuf};
use crate::gpu::{GKind, GMat, Gpu};
use std::collections::HashMap;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, RwLock};

const DECAY_EVERY: usize = 64;
/// a challenger must beat the coldest resident by this factor plus one use
const PROMOTE_MARGIN: f32 = 1.5;

struct State {
    owner: Vec<Option<u32>>, // slot -> expert key
    map: HashMap<u32, usize>, // expert key -> slot
    heat: Vec<f32>,           // per expert key
    since_decay: usize,
    gpu_uses: u64,
    promotions: u64,
}

/// Extra slots that prefill streams experts through, to run them on the GPU.
pub const STAGE_SLOTS: usize = 8;

pub struct VramTier {
    /// (weights, scales); grows once after prefill (see `grow`)
    slots: Arc<RwLock<Vec<(wgpu::Buffer, wgpu::Buffer)>>>,
    stage: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    st: Arc<Mutex<State>>,
    /// promotions waiting for the upload thread: (slot, expert key, data)
    promote: Mutex<Sender<(usize, u32, Arc<ExpertBuf>)>>,
    inter: usize,
    hidden: usize,
    packed: bool,
    q_bytes: u64,
    s_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct VramStats {
    pub slots: usize,
    pub filled: usize,
    pub gpu_uses: u64,
    pub promotions: u64,
}

impl VramTier {
    pub fn new(gpu: &Gpu, n_slots: usize, inter: usize, hidden: usize, n_keys: usize, packed: bool) -> VramTier {
        let q_bytes = piece_len(inter * hidden, 0, packed) as u64;
        let s_bytes = piece_len(inter * hidden, 1, packed) as u64;
        // allocate until the budget or the driver says stop, then hand a few
        // slots back so the scratch buffers can still grow
        let mut slots: Vec<(wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let n_slots = n_slots.saturating_sub(STAGE_SLOTS);
        let mut stage = Vec::new();
        while stage.len() < STAGE_SLOTS {
            let Some(w) = gpu.try_empty("stage w", 3 * q_bytes) else { break };
            let Some(s) = gpu.try_empty("stage s", 3 * s_bytes) else { break };
            stage.push((w, s));
        }
        while slots.len() < n_slots {
            let Some(w) = gpu.try_empty("expert w", 3 * q_bytes) else { break };
            let Some(s) = gpu.try_empty("expert s", 3 * s_bytes) else { break };
            slots.push((w, s));
        }
        if slots.len() < n_slots {
            slots.truncate(slots.len().saturating_sub(16));
        }
        let n_slots = slots.len();
        let slots = Arc::new(RwLock::new(slots));
        let st = Arc::new(Mutex::new(State {
            owner: vec![None; n_slots],
            map: HashMap::new(),
            heat: vec![0.0; n_keys],
            since_decay: 0,
            gpu_uses: 0,
            promotions: 0,
        }));
        // Promotions upload on their own thread so the forward pass never waits
        // for the copy; an expert joins the map (and gets used) once it is in.
        let (tx, rx) = std::sync::mpsc::channel::<(usize, u32, Arc<ExpertBuf>)>();
        let up = gpu.uploader(16 << 20);
        let (bufs, st2) = (slots.clone(), st.clone());
        std::thread::Builder::new()
            .name("pelica-vram".into())
            .spawn(move || {
                for (slot, key, buf) in rx {
                    let (w, s) = bufs.read().unwrap()[slot].clone();
                    up.write_all(&Self::pieces(&w, &s, q_bytes, s_bytes, &buf));
                    let mut g = st2.lock().unwrap();
                    if g.owner[slot] == Some(key) {
                        g.map.insert(key, slot);
                    }
                }
            })
            .expect("starting the VRAM upload thread");
        VramTier {
            slots,
            stage,
            st,
            promote: Mutex::new(tx),
            inter,
            hidden,
            q_bytes,
            s_bytes,
            packed,
        }
    }

    pub fn slot_bytes(inter: usize, hidden: usize, packed: bool) -> u64 {
        3 * (piece_len(inter * hidden, 0, packed) + piece_len(inter * hidden, 1, packed)) as u64
    }

    pub fn n_slots(&self) -> usize {
        self.slots.read().unwrap().len()
    }

    /// Add up to `n` slots (prefill is over and its scratch buffers have been
    /// trimmed, so their VRAM can hold experts instead). Returns how many.
    pub fn grow(&self, gpu: &Gpu, n: usize) -> usize {
        let mut fresh = Vec::new();
        while fresh.len() < n {
            let Some(w) = gpu.try_empty("expert w", 3 * self.q_bytes) else { break };
            let Some(s) = gpu.try_empty("expert s", 3 * self.s_bytes) else { break };
            fresh.push((w, s));
        }
        let added = fresh.len();
        let mut slots = self.slots.write().unwrap();
        let mut g = self.st.lock().unwrap();
        slots.extend(fresh);
        g.owner.resize(slots.len(), None);
        added
    }

    pub fn stats(&self) -> VramStats {
        let g = self.st.lock().unwrap();
        VramStats {
            slots: self.n_slots(),
            filled: g.map.len(),
            gpu_uses: g.gpu_uses,
            promotions: g.promotions,
        }
    }

    pub fn n_stage(&self) -> usize {
        self.stage.len()
    }

    /// Copy experts into the stage slots, one per slot, for a GPU run; the
    /// slots are free again once that run has been read back.
    pub fn stage(&self, gpu: &Gpu, bufs: &[&ExpertBuf]) -> Vec<[GMat; 3]> {
        let mut items = Vec::new();
        for (slot, buf) in bufs.iter().enumerate() {
            let (w, s) = &self.stage[slot];
            for (i, piece) in [0usize, 2, 4].iter().enumerate() {
                items.push((w, i as u64 * self.q_bytes, buf.piece(*piece)));
                items.push((s, i as u64 * self.s_bytes, buf.piece(piece + 1)));
            }
        }
        gpu.write_all(&items);
        (0..bufs.len()).map(|i| self.views(&self.stage[i])).collect()
    }

    /// gate, up, down of the expert in `slot`, as views into the slot buffers.
    pub fn mats(&self, slot: usize) -> [GMat; 3] {
        let pair = self.slots.read().unwrap()[slot].clone();
        self.views(&pair)
    }

    fn views(&self, (w, s): &(wgpu::Buffer, wgpu::Buffer)) -> [GMat; 3] {
        let (qw, sf) = ((self.q_bytes / 4) as u32, (self.s_bytes / 4) as u32);
        let m = |i: u32, rows, cols| GMat {
            kind: GKind::Q4,
            w: w.clone(),
            s: s.clone(),
            w_off: i * qw,
            s_off: i * sf,
            rows,
            cols,
            packed: self.packed,
        };
        [m(0, self.inter, self.hidden), m(1, self.inter, self.hidden), m(2, self.hidden, self.inter)]
    }

    pub fn slot_of(&self, key: u32) -> Option<usize> {
        self.st.lock().unwrap().map.get(&key).copied()
    }

    pub fn contains(&self, key: u32) -> bool {
        self.st.lock().unwrap().map.contains_key(&key)
    }

    /// Record routing: each key was chosen once by one token.
    pub fn touch(&self, keys: &[u32], on_gpu: u64) {
        let mut g = self.st.lock().unwrap();
        for &k in keys {
            g.heat[k as usize] += 1.0;
        }
        g.gpu_uses += on_gpu;
    }

    /// Advance the clock by `tokens`; heat halves every DECAY_EVERY tokens.
    pub fn tick(&self, tokens: usize) {
        let mut g = self.st.lock().unwrap();
        g.since_decay += tokens;
        while g.since_decay >= DECAY_EVERY {
            g.since_decay -= DECAY_EVERY;
            for h in g.heat.iter_mut() {
                *h *= 0.5;
            }
        }
    }

    pub fn set_heat(&self, key: u32, heat: f32) {
        self.st.lock().unwrap().heat[key as usize] = heat;
    }

    /// The six writes that put an expert's weights and scales into a slot.
    fn pieces<'a>(w: &'a wgpu::Buffer, s: &'a wgpu::Buffer, q_bytes: u64, s_bytes: u64, buf: &'a ExpertBuf) -> Vec<(&'a wgpu::Buffer, u64, &'a [u8])> {
        let mut items = Vec::new();
        for (i, piece) in [0usize, 2, 4].iter().enumerate() {
            items.push((w, i as u64 * q_bytes, buf.piece(*piece)));
            items.push((s, i as u64 * s_bytes, buf.piece(piece + 1)));
        }
        items
    }

    fn upload(&self, gpu: &Gpu, slot: usize, buf: &ExpertBuf) {
        let (w, s) = self.slots.read().unwrap()[slot].clone();
        gpu.write_all(&Self::pieces(&w, &s, self.q_bytes, self.s_bytes, buf));
    }

    /// Put an expert into a free slot (used to warm the tier). False when full.
    pub fn place(&self, gpu: &Gpu, key: u32, buf: &ExpertBuf) -> bool {
        let mut g = self.st.lock().unwrap();
        if g.map.contains_key(&key) {
            return true;
        }
        let Some(slot) = g.owner.iter().position(Option::is_none) else { return false };
        g.owner[slot] = Some(key);
        g.map.insert(key, slot);
        drop(g);
        self.upload(gpu, slot, buf);
        true
    }

    /// Promote `key` if a slot is free or it is clearly hotter than the
    /// coldest resident. Returns true if it moved into VRAM.
    pub fn consider(&self, key: u32, buf: &Arc<ExpertBuf>) -> bool {
        let mut g = self.st.lock().unwrap();
        if g.map.contains_key(&key) || g.owner.is_empty() {
            return false;
        }
        let heat = g.heat[key as usize];
        let slot = match g.owner.iter().position(Option::is_none) {
            Some(free) => free,
            None => {
                let (slot, cold) = g
                    .owner
                    .iter()
                    .enumerate()
                    .map(|(i, o)| (i, g.heat[o.unwrap() as usize]))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                if heat <= cold * PROMOTE_MARGIN + 1.0 {
                    return false;
                }
                let old = g.owner[slot].take().unwrap();
                g.map.remove(&old);
                slot
            }
        };
        // the slot is taken now; the expert is usable once the upload lands
        g.owner[slot] = Some(key);
        g.promotions += 1;
        drop(g);
        let _ = self.promote.lock().unwrap().send((slot, key, buf.clone()));
        true
    }
}
