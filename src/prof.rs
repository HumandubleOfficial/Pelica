//! Cheap wall-clock accounting for the parts of a forward pass, printed
//! after decode to show where the time goes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub const NAMES: [&str; 8] =
    ["kda gpu", "kda cpu", "mla gpu", "mla cpu", "route", "moe gpu", "moe cpu", "promote"];
pub const KDA_GPU: usize = 0;
pub const KDA_CPU: usize = 1;
pub const MLA_GPU: usize = 2;
pub const MLA_CPU: usize = 3;
pub const ROUTE: usize = 4;
pub const MOE_GPU: usize = 5;
pub const MOE_CPU: usize = 6;
pub const PROMOTE: usize = 7;

static NS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

pub fn add(slot: usize, since: Instant) {
    NS[slot].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

pub fn reset() {
    for n in &NS {
        n.store(0, Ordering::Relaxed);
    }
}

/// "name 1.2s, ..." for every slot that saw time.
pub fn report() -> String {
    NAMES
        .iter()
        .zip(&NS)
        .map(|(n, v)| (n, v.load(Ordering::Relaxed) as f64 / 1e9))
        .filter(|(_, s)| *s > 0.0)
        .map(|(n, s)| format!("{n} {s:.1}s"))
        .collect::<Vec<_>>()
        .join(", ")
}
