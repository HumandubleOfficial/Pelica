//! `pelica repack <model.gguf> [--undo] [--first N]`: puts each expert's three pieces
//! (gate, up, down rows) side by side in a DeepSeek V4 or Qwen3.8 GGUF, so the
//! engine reads an expert from the drive in one read instead of three (DeepSeek:
//! 14 MB instead of 4.7 MB; Qwen3.8: 2.3 MB instead of ~0.8 MB, which this NVMe
//! reads ~35% faster). A model split in parts (`-00001-of-00003.gguf`) is done
//! part by part, each with its own journal; a layer whose pieces are in
//! different parts stays as it is.
//!
//! A layer's region runs from its first expert tensor to the end of its last.
//! Other tensors may sit in gaps between them (Qwen3.8: the shared expert's
//! down matrix, the router): repacked, the experts fill the region's start,
//! expert by expert, and each gap's bytes follow, in order, with the offsets
//! of the tensors in them moved in the header. The expert tensors' offsets
//! never change, so the gaps (and undo) are known from them. The tensors are renamed
//! `*_exps` -> `*_expm` (same length, so the header doesn't move). Other GGUF
//! readers then refuse the file instead of loading scrambled weights, and
//! `--undo` puts everything back.
//!
//! Each layer is journaled: its original region goes to
//! `<model>.repack-journal` (flushed) before the region is rewritten, the
//! rewrite is read back and compared, then the names change and the journal
//! is cleared. After a crash, the next run finishes the renaming (when any
//! name already changed, the data was complete) or writes the journal back.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};

use crate::gguf::Gguf;

const MAGIC: &[u8; 8] = b"PELREPK1";
/// journal header bytes (the region's copy follows)
const HEAD: u64 = 64;
const PIECES: [&str; 3] = ["ffn_gate", "ffn_up", "ffn_down"];
/// the architectures whose engines read repacked layers
const ARCHS: [&str; 2] = ["deepseek4", "qwen4exp"];

/// Whether `pelica repack` can work on this model (its first part).
pub fn supports(g: &Gguf) -> bool {
    ARCHS.contains(&g.arch())
}

/// How far a model (all its parts) is repacked: "yes", "no" or "partial"
/// (counting the layers that can be).
pub fn state(model: &Path) -> Result<&'static str> {
    let (mut done, mut all) = (0, 0);
    for p in crate::q4x::split_paths(model) {
        let g = Gguf::open(&p)?;
        for y in layers(&g, true)? {
            all += 1;
            done += (y.merged == 3) as usize;
        }
    }
    Ok(if all > 0 && done == all { "yes" } else if done == 0 { "no" } else { "partial" })
}

/// Whether layer `l`'s experts are repacked (`*_expm`). A layer with only
/// some names changed was interrupted: `pelica repack` finishes it.
pub fn layer_state(g: &Gguf, l: usize) -> Result<bool> {
    let m = PIECES.iter().filter(|p| g.tensors.contains_key(&format!("blk.{l}.{p}_expm.weight"))).count();
    match m {
        0 => Ok(false),
        3 => Ok(true),
        _ => bail!("layer {l} of this model was left half repacked: run `pelica repack` on it again to finish"),
    }
}

/// One layer's expert region.
struct Layer {
    l: usize,
    /// per piece (gate, up, down): tensor offset, bytes per expert, name position
    /// of the "exps"/"expm" suffix in the header
    off: [u64; 3],
    len: [usize; 3],
    tag_at: [u64; 3],
    /// how many of the names say "expm"
    merged: usize,
    base: u64,
    /// the experts' bytes
    total: usize,
    n_exp: usize,
    /// bytes from `base` to the end of the last expert tensor
    span: usize,
    /// the gaps between expert tensors as in the GGUF: (start from base, bytes);
    /// repacked, gap h starts at `total` + the earlier gaps' bytes
    holes: Vec<(usize, usize)>,
    /// the other tensors in the region: where their offset sits in the header,
    /// and their bytes
    movers: Vec<(u64, usize)>,
    /// the file's data start and alignment
    data: u64,
    align: usize,
}

impl Layer {
    /// Where gap h starts repacked (from base).
    fn packed_hole(&self, h: usize) -> usize {
        self.total + self.holes[..h].iter().map(|x| x.1).sum::<usize>()
    }

    /// A tensor at `r` (from base, `n` bytes) in the region: its gap, and
    /// whether it sits as repacked.
    fn hole_of(&self, r: usize, n: usize) -> Option<(usize, bool)> {
        for (h, &(a, len)) in self.holes.iter().enumerate() {
            if r >= a && r + n <= a + len {
                return Some((h, false));
            }
            let pa = self.packed_hole(h);
            if r >= pa && r + n <= pa + len {
                return Some((h, true));
            }
        }
        None
    }
}

/// The layers of this file whose three expert tensors can be repacked. With
/// `part` (a file of a split model), a layer with pieces elsewhere or with
/// gaps between them is left out instead of refused.
fn layers(g: &Gguf, part: bool) -> Result<Vec<Layer>> {
    let mut ls: Vec<usize> = g
        .tensors
        .keys()
        .filter_map(|n| {
            let (l, r) = n.strip_prefix("blk.")?.split_once('.')?;
            let piece = PIECES.iter().any(|p| r.starts_with(p)) && (r.ends_with("_exps.weight") || r.ends_with("_expm.weight"));
            if piece { l.parse().ok() } else { None }
        })
        .collect();
    ls.sort_unstable();
    ls.dedup();
    let mut out = Vec::new();
    for l in ls {
        let mut off = [0u64; 3];
        let mut len = [0usize; 3];
        let mut tag_at = [0u64; 3];
        let mut merged = 0;
        let mut found = 0;
        let mut n_exp = 0;
        for (i, p) in PIECES.iter().enumerate() {
            for (suffix, m) in [("exps", 0), ("expm", 1)] {
                let name = format!("blk.{l}.{p}_{suffix}.weight");
                if let Some(t) = g.tensors.get(&name) {
                    if t.dims.len() != 3 {
                        bail!("{name}: not an expert tensor");
                    }
                    n_exp = t.dims[2];
                    off[i] = t.offset as u64;
                    len[i] = t.bytes() / n_exp;
                    tag_at[i] = (t.name_at + name.find(suffix).unwrap()) as u64;
                    merged += m;
                    found += 1;
                }
            }
        }
        if found != 3 {
            if part {
                continue;
            }
            bail!("layer {l}: expected gate, up and down expert tensors");
        }
        // the region from the first expert tensor to the end of the last, and
        // the gaps between them
        let mut order = [0usize, 1, 2];
        order.sort_by_key(|&i| off[i]);
        let base = off[order[0]];
        let total = len.iter().sum::<usize>() * n_exp;
        let mut holes = Vec::new();
        let mut at = base;
        for &i in &order {
            if off[i] > at {
                holes.push(((at - base) as usize, (off[i] - at) as usize));
            }
            at = at.max(off[i] + (len[i] * n_exp) as u64);
        }
        let span = (at - base) as usize;
        let mut y = Layer { l, off, len, tag_at, merged, base, total, n_exp, span, holes, movers: Vec::new(), data: g.data as u64, align: g.align };
        // repacked gaps must lie past every gap as it was (so a tensor's place
        // says which way it sits), and move by whole alignment steps
        let ok = y.span == y.total + y.holes.iter().map(|x| x.1).sum::<usize>()
            && y.holes.iter().enumerate().all(|(h, &(a, len))| a + len <= y.total && (y.packed_hole(h) - a) % y.align == 0);
        let mut fits = ok;
        if ok {
            for (name, t) in &g.tensors {
                let r = t.offset as i64 - base as i64;
                if r < 0 || r as usize >= span || t.dims.len() == 3 && PIECES.iter().any(|p| name.starts_with(&format!("blk.{l}.{p}_exp"))) {
                    continue;
                }
                if y.hole_of(r as usize, t.bytes()).is_none() {
                    fits = false;
                    break;
                }
                let field = t.name_at + name.len() + 4 + 8 * t.dims.len() + 4;
                y.movers.push((field as u64, t.bytes()));
            }
        }
        if !fits {
            if part {
                continue;
            }
            bail!("layer {l}: the expert tensors' region holds tensors that can't be moved, so it can't be repacked");
        }
        out.push(y);
    }
    if out.is_empty() && !part {
        bail!("no expert tensors found");
    }
    Ok(out)
}

/// Where expert `e`'s piece `p` sits in the region: as in the GGUF (apart),
/// and repacked (together).
fn apart(y: &Layer, e: usize, p: usize) -> usize {
    (y.off[p] - y.base) as usize + e * y.len[p]
}
fn together(y: &Layer, e: usize, p: usize) -> usize {
    e * y.len.iter().sum::<usize>() + y.len[..p].iter().sum::<usize>()
}

/// The region rearranged: apart -> together (`pack`), or back.
fn permute(y: &Layer, src: &[u8], pack: bool) -> Vec<u8> {
    let mut dst = vec![0u8; src.len()];
    for e in 0..y.n_exp {
        for p in 0..3 {
            let (from, to) = if pack { (apart(y, e, p), together(y, e, p)) } else { (together(y, e, p), apart(y, e, p)) };
            dst[to..to + y.len[p]].copy_from_slice(&src[from..from + y.len[p]]);
        }
    }
    for (h, &(a, len)) in y.holes.iter().enumerate() {
        let pa = y.packed_hole(h);
        let (from, to) = if pack { (a, pa) } else { (pa, a) };
        dst[to..to + len].copy_from_slice(&src[from..from + len]);
    }
    dst
}

/// Points the moved tensors' header offsets at their gap as it sits
/// repacked (`packed`) or as in the GGUF; ones already there stay.
fn fix_offsets(f: &mut File, y: &Layer, packed: bool) -> Result<()> {
    for &(field, n) in &y.movers {
        let mut b = [0u8; 8];
        read_exact_at(f, field, &mut b)?;
        let rel = u64::from_le_bytes(b);
        let r = (y.data + rel - y.base) as usize;
        let (h, is_packed) = y.hole_of(r, n).context("a tensor in the experts' region is in no gap")?;
        if is_packed != packed {
            let d = (y.packed_hole(h) - y.holes[h].0) as u64;
            let new = if packed { rel + d } else { rel - d };
            write_all_at(f, field, &new.to_le_bytes())?;
        }
    }
    f.sync_all()?;
    Ok(())
}

fn read_exact_at(f: &mut File, at: u64, buf: &mut [u8]) -> Result<()> {
    f.seek(SeekFrom::Start(at))?;
    f.read_exact(buf)?;
    Ok(())
}

fn write_all_at(f: &mut File, at: u64, buf: &[u8]) -> Result<()> {
    f.seek(SeekFrom::Start(at))?;
    for c in buf.chunks(64 << 20) {
        f.write_all(c)?;
    }
    Ok(())
}

fn set_names(f: &mut File, y: &Layer, merged: bool) -> Result<()> {
    for &at in &y.tag_at {
        write_all_at(f, at, if merged { b"expm" } else { b"exps" })?;
    }
    f.sync_all()?;
    Ok(())
}

fn journal_path(model: &Path) -> PathBuf {
    let mut p = model.as_os_str().to_owned();
    p.push(".repack-journal");
    PathBuf::from(p)
}

/// A repack is running on the model, or was stopped partway: a layer may be
/// half rewritten until `pelica repack` runs again and puts it right.
pub fn unfinished(model: &Path) -> bool {
    crate::q4x::split_paths(model).iter().any(|p| journal_path(p).exists())
}

/// Finishes a layer an earlier run was in the middle of, if any.
fn recover(model: &Path, f: &mut File) -> Result<()> {
    let jp = journal_path(model);
    let Ok(mut j) = OpenOptions::new().read(true).write(true).open(&jp) else { return Ok(()) };
    let mut h = [0u8; HEAD as usize];
    if j.read(&mut h)? < 29 || &h[..8] != MAGIC {
        drop(j);
        std::fs::remove_file(&jp)?;
        return Ok(());
    }
    let l = u32::from_le_bytes(h[8..12].try_into()?) as usize;
    let to_merged = h[12] == 1;
    let base = u64::from_le_bytes(h[13..21].try_into()?);
    let total = u64::from_le_bytes(h[21..29].try_into()?) as usize;
    let g = Gguf::open(model)?;
    let ys = layers(&g, true)?;
    drop(g);
    let y = ys.iter().find(|y| y.l == l).context("the journal names a layer the model doesn't have")?;
    if y.base != base || y.span != total {
        bail!("{} doesn't match this model: move it away and run again", jp.display());
    }
    let want = if to_merged { 3 } else { 0 };
    if y.merged == want {
        eprintln!("layer {l}: already done");
    } else if (to_merged && y.merged > 0) || (!to_merged && y.merged < 3) {
        // a name had changed, so the rewrite was complete and checked
        eprintln!("layer {l}: finishing the renaming an earlier run was doing");
        fix_offsets(f, y, to_merged)?;
        set_names(f, y, to_merged)?;
    } else {
        eprintln!("layer {l}: putting back the region an earlier run was rewriting");
        let mut buf = vec![0u8; total];
        read_exact_at(&mut j, HEAD, &mut buf)?;
        write_all_at(f, base, &buf)?;
        f.sync_all()?;
        fix_offsets(f, y, !to_merged)?;
    }
    drop(j);
    std::fs::remove_file(&jp)?;
    Ok(())
}

pub fn run(args: &[String]) -> Result<()> {
    let undo = args.iter().any(|a| a == "--undo");
    // (--first N: only the first N layers still to do, for a trial run)
    let first = args.iter().position(|a| a == "--first").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok());
    let Some(model) = args.iter().enumerate().find(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--first")).map(|(_, a)| PathBuf::from(a)) else {
        bail!("usage: pelica repack <model.gguf> [--undo]");
    };
    let g = Gguf::open(&model)?;
    if !supports(&g) {
        bail!("pelica repack is for DeepSeek V4 and Qwen3.8 GGUFs");
    }
    drop(g);
    let paths = crate::q4x::split_paths(&model);
    let split = paths.len() > 1;
    let mut fs = Vec::new();
    let mut todo: Vec<(usize, Layer)> = Vec::new();
    let mut n_all = 0;
    let target = if undo { 0 } else { 3 };
    for (i, path) in paths.iter().enumerate() {
        let mut o = OpenOptions::new();
        o.read(true).write(true);
        // nobody else may write it meanwhile: a second repack would take this
        // run's journal for a stopped one
        #[cfg(windows)]
        std::os::windows::fs::OpenOptionsExt::share_mode(&mut o, 1 /* FILE_SHARE_READ */);
        let mut f = o.open(path).with_context(|| format!("opening {} for writing (is Pelica still using it, or already repacking it?)", path.display()))?;
        recover(path, &mut f)?;
        let g = Gguf::open(path)?;
        let ys = layers(&g, split)?;
        drop(g);
        n_all += ys.len();
        todo.extend(ys.into_iter().filter(|y| y.merged != target).map(|y| (i, y)));
        fs.push(f);
    }
    todo.truncate(first.unwrap_or(usize::MAX));
    let gb = todo.iter().map(|(_, y)| y.span).sum::<usize>() as f64 / 1e9;
    eprintln!("{} {} of {} layers ({gb:.1} GB){}", if undo { "un-repacking" } else { "repacking" }, todo.len(), n_all, if todo.is_empty() { ": nothing to do" } else { "" });
    // (--dry-run: only say what it would do)
    if args.iter().any(|a| a == "--dry-run") {
        return Ok(());
    }
    let t0 = Instant::now();
    for (i, (fi, y)) in todo.iter().enumerate() {
        let f = &mut fs[*fi];
        let jp = journal_path(&paths[*fi]);
        let t = Instant::now();
        let mut src = vec![0u8; y.span];
        read_exact_at(f, y.base, &mut src)?;
        // 1. the original region, kept until this layer is done
        let mut j = OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&jp)?;
        write_all_at(&mut j, HEAD, &src)?;
        j.sync_all()?;
        let mut h = [0u8; HEAD as usize];
        h[..8].copy_from_slice(MAGIC);
        h[8..12].copy_from_slice(&(y.l as u32).to_le_bytes());
        h[12] = !undo as u8;
        h[13..21].copy_from_slice(&y.base.to_le_bytes());
        h[21..29].copy_from_slice(&(y.span as u64).to_le_bytes());
        write_all_at(&mut j, 0, &h)?;
        j.sync_all()?;
        // 2. the region rearranged, read back and compared
        let dst = permute(y, &src, !undo);
        drop(src);
        write_all_at(f, y.base, &dst)?;
        f.sync_all()?;
        let mut back = vec![0u8; y.span];
        read_exact_at(f, y.base, &mut back)?;
        if back != dst {
            bail!("layer {}: the rewrite didn't read back the same; run `pelica repack` again to put the region back from {}", y.l, jp.display());
        }
        drop(back);
        drop(dst);
        // 3. the moved tensors' offsets, then the names say how the layer
        // sits now; 4. the journal goes
        fix_offsets(f, y, !undo)?;
        set_names(f, y, !undo)?;
        j.set_len(0)?;
        j.sync_all()?;
        drop(j);
        let done = (i + 1) as f64 / todo.len() as f64;
        let el = t0.elapsed().as_secs_f64();
        eprintln!("layer {:>2} done in {:.1}s ({:.0}%, about {:.0}s left)", y.l, t.elapsed().as_secs_f64(), done * 100.0, el / done - el);
    }
    for p in &paths {
        let jp = journal_path(p);
        if jp.exists() {
            std::fs::remove_file(&jp)?;
        }
    }
    if !todo.is_empty() {
        eprintln!("{} in {:.0}s", if undo { "Un-repacked: the file is a plain GGUF again" } else { "Repacked: Pelica reads each expert in one go now. `pelica repack --undo` puts it back for other programs" }, t0.elapsed().as_secs_f64());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny GGUF: `arch` (if any) and, per (layer, name), a Q4_K tensor in
    /// that order: "ffn_gate" etc. an expert tensor of 4 experts x 2 rows x 256
    /// columns, a name ending in "shexp" or "inp" a 2 x 256 matrix.
    fn tiny(arch: Option<&str>, tensors: &[(usize, &str)], seed: u32) -> Vec<u8> {
        let (ne, rows, cols) = (4usize, 2usize, 256usize);
        let per = rows * cols / 256 * 144;
        let s = |b: &mut Vec<u8>, x: &str| {
            b.extend_from_slice(&(x.len() as u64).to_le_bytes());
            b.extend_from_slice(x.as_bytes());
        };
        let mut kv = Vec::new();
        if let Some(a) = arch {
            s(&mut kv, "general.architecture");
            kv.extend_from_slice(&8u32.to_le_bytes());
            s(&mut kv, a);
        }
        let mut dir_b = Vec::new();
        let mut off = 0u64;
        for &(l, p) in tensors {
            let small = p.ends_with("shexp") || p.ends_with("inp");
            s(&mut dir_b, &if small { format!("blk.{l}.{p}.weight") } else { format!("blk.{l}.{p}_exps.weight") });
            let dims: &[u64] = if small { &[cols as u64, rows as u64] } else { &[cols as u64, rows as u64, ne as u64] };
            dir_b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                dir_b.extend_from_slice(&d.to_le_bytes());
            }
            dir_b.extend_from_slice(&12u32.to_le_bytes());
            dir_b.extend_from_slice(&off.to_le_bytes());
            off += (per * if small { 1 } else { ne }) as u64;
        }
        let mut file = Vec::new();
        file.extend_from_slice(b"GGUF");
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        file.extend_from_slice(&(arch.is_some() as u64).to_le_bytes());
        file.extend_from_slice(&kv);
        file.extend_from_slice(&dir_b);
        while file.len() % 32 != 0 {
            file.push(0);
        }
        let mut x = seed;
        for _ in 0..off {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            file.push((x >> 24) as u8);
        }
        file
    }

    /// A Qwen3.8-style model in three parts (the first without tensors, one
    /// layer's pieces across parts 2 and 3, small tensors between a layer's
    /// expert tensors): the whole layers are repacked with the small tensors
    /// moved (and found through the header), the split one is left alone, and
    /// undo restores every byte.
    #[test]
    fn split_model() {
        let dir = std::env::temp_dir().join(format!("pelica-repack-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let parts = [dir.join("m-00001-of-00003.gguf"), dir.join("m-00002-of-00003.gguf"), dir.join("m-00003-of-00003.gguf")];
        let files = [
            tiny(Some("qwen4exp"), &[], 1),
            tiny(None, &[(0, "ffn_down"), (0, "ffn_down_shexp"), (0, "ffn_gate"), (0, "ffn_gate_inp"), (0, "ffn_gate_shexp"), (0, "ffn_up"), (0, "ffn_up_shexp"), (1, "ffn_down"), (1, "ffn_gate"), (1, "ffn_up"), (2, "ffn_down")], 2),
            tiny(None, &[(2, "ffn_gate"), (2, "ffn_up"), (3, "ffn_down"), (3, "ffn_gate"), (3, "ffn_up")], 3),
        ];
        for (p, f) in parts.iter().zip(&files) {
            std::fs::write(p, f).unwrap();
        }
        assert_eq!(state(&parts[0]).unwrap(), "no");
        let small = ["blk.0.ffn_down_shexp.weight", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight"];
        let g = Gguf::open(&parts[1]).unwrap();
        let small_before: Vec<Vec<u8>> = small.iter().map(|n| g.bytes(g.tensor(n).unwrap()).to_vec()).collect();
        drop(g);
        run(&[parts[0].to_string_lossy().into_owned()]).unwrap();
        assert_eq!(state(&parts[0]).unwrap(), "yes");
        assert!(!unfinished(&parts[0]));
        let g2 = Gguf::open(&parts[1]).unwrap();
        assert!(layer_state(&g2, 0).unwrap() && layer_state(&g2, 1).unwrap());
        assert!(g2.tensors.contains_key("blk.2.ffn_down_exps.weight"), "the split layer keeps its name");
        let small_after: Vec<Vec<u8>> = small.iter().map(|n| g2.bytes(g2.tensor(n).unwrap()).to_vec()).collect();
        assert_eq!(small_before, small_after, "the moved tensors are found through the header");
        // each repacked expert's pieces sit together where the engine looks
        let packed = std::fs::read(&parts[1]).unwrap();
        let ys = layers(&g2, true).unwrap();
        for y in &ys {
            for e in 0..y.n_exp {
                for p in 0..3 {
                    let was = &files[1][y.off[p] as usize + e * y.len[p]..][..y.len[p]];
                    let now = &packed[y.base as usize + together(y, e, p)..][..y.len[p]];
                    assert_eq!(was, now);
                }
            }
        }
        drop(g2);
        run(&[parts[0].to_string_lossy().into_owned(), "--undo".into()]).unwrap();
        for (p, f) in parts.iter().zip(&files) {
            assert_eq!(&std::fs::read(p).unwrap(), f, "undo restores every byte");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tiny GGUF with two layers of expert tensors (gate, down, up in that
    /// file order, as in DeepSeek V4 files), repacked and undone: the engine's
    /// offsets find each piece where it was, and undo restores every byte.
    #[test]
    fn repack_and_undo() {
        let dir = std::env::temp_dir().join(format!("pelica-repack-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.gguf");
        // Q4_K: 144 bytes per 256 values; 2 experts of 2 rows x 256 columns
        let (ne, rows, cols) = (4usize, 2usize, 256usize);
        let per = rows * cols / 256 * 144;
        let mut kv = Vec::new();
        let s = |b: &mut Vec<u8>, x: &str| {
            b.extend_from_slice(&(x.len() as u64).to_le_bytes());
            b.extend_from_slice(x.as_bytes());
        };
        s(&mut kv, "general.architecture");
        kv.extend_from_slice(&8u32.to_le_bytes());
        s(&mut kv, "deepseek4");
        let mut dir_b = Vec::new();
        let mut off = 0u64;
        let mut names = Vec::new();
        for l in 0..2 {
            for p in ["ffn_gate", "ffn_down", "ffn_up"] {
                let n = format!("blk.{l}.{p}_exps.weight");
                s(&mut dir_b, &n);
                dir_b.extend_from_slice(&3u32.to_le_bytes());
                for d in [cols as u64, rows as u64, ne as u64] {
                    dir_b.extend_from_slice(&d.to_le_bytes());
                }
                dir_b.extend_from_slice(&12u32.to_le_bytes());
                dir_b.extend_from_slice(&off.to_le_bytes());
                off += (per * ne) as u64;
                names.push(n);
            }
        }
        let mut file = Vec::new();
        file.extend_from_slice(b"GGUF");
        file.extend_from_slice(&3u32.to_le_bytes());
        file.extend_from_slice(&(names.len() as u64).to_le_bytes());
        file.extend_from_slice(&1u64.to_le_bytes());
        file.extend_from_slice(&kv);
        file.extend_from_slice(&dir_b);
        while file.len() % 32 != 0 {
            file.push(0);
        }
        let data = file.len();
        let mut x = 9u32;
        for _ in 0..off {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            file.push((x >> 24) as u8);
        }
        std::fs::write(&path, &file).unwrap();
        let orig = file.clone();
        // what the engine reads for (layer, expert, piece) before
        let piece = |bytes: &[u8], g: &Gguf, l: usize, e: usize, p: usize| -> Vec<u8> {
            let merged = layer_state(g, l).unwrap();
            let ys = layers(g, false).unwrap();
            let y = &ys[l];
            let at = if merged { y.base as usize + together(y, e, p) } else { y.off[p] as usize + e * y.len[p] };
            bytes[at..at + y.len[p]].to_vec()
        };
        let g0 = Gguf::open(&path).unwrap();
        let before: Vec<Vec<u8>> = (0..2).flat_map(|l| (0..ne).flat_map(move |e| (0..3).map(move |p| (l, e, p)))).map(|(l, e, p)| piece(&orig, &g0, l, e, p)).collect();
        drop(g0);
        let arg = vec![path.to_string_lossy().into_owned()];
        run(&arg).unwrap();
        let packed = std::fs::read(&path).unwrap();
        assert_ne!(packed, orig);
        let g1 = Gguf::open(&path).unwrap();
        assert!(layer_state(&g1, 0).unwrap() && layer_state(&g1, 1).unwrap());
        let after: Vec<Vec<u8>> = (0..2).flat_map(|l| (0..ne).flat_map(move |e| (0..3).map(move |p| (l, e, p)))).map(|(l, e, p)| piece(&packed, &g1, l, e, p)).collect();
        drop(g1);
        assert_eq!(before, after, "every piece is found where the engine looks");
        // each expert's pieces are back to back
        assert!(data > 0);
        // an interrupted run: journal for layer 1 with the names unchanged ->
        // the region comes back from the journal
        run(&[path.to_string_lossy().into_owned(), "--undo".into()]).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), orig, "undo restores every byte");
        // a crash after the rewrite but before the renaming
        {
            let g = Gguf::open(&path).unwrap();
            let ys = layers(&g, false).unwrap();
            drop(g);
            let y = &ys[1];
            let src = orig[y.base as usize..y.base as usize + y.span].to_vec();
            let mut j = File::create(journal_path(&path)).unwrap();
            let mut h = [0u8; HEAD as usize];
            h[..8].copy_from_slice(MAGIC);
            h[8..12].copy_from_slice(&1u32.to_le_bytes());
            h[12] = 1;
            h[13..21].copy_from_slice(&y.base.to_le_bytes());
            h[21..29].copy_from_slice(&(y.span as u64).to_le_bytes());
            j.write_all(&h).unwrap();
            j.write_all(&src).unwrap();
            drop(j);
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            write_all_at(&mut f, y.base, &permute(y, &src, true)).unwrap();
        }
        let mut f = OpenOptions::new().read(true).write(true).open(&path).unwrap();
        recover(&path, &mut f).unwrap();
        drop(f);
        assert_eq!(std::fs::read(&path).unwrap(), orig, "recovery puts the region back");
        assert!(!journal_path(&path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
