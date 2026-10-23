//! Reader for GGUF files (llama.cpp's single-file model format): metadata,
//! tensor directory, and the tensor bytes through a memory map.

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// A metadata value.
#[derive(Debug, Clone)]
pub enum Val {
    Int(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Val>),
}

impl Val {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Val::Int(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::F(v) => Some(*v),
            Val::Int(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Val::Bool(b) => Some(*b),
            Val::Int(v) => Some(*v != 0),
            _ => None,
        }
    }
}

/// ggml tensor types. The ones Pelica's kernels decode have names; the rest
/// are kept by id (so any file opens, and only using such a tensor fails).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ty {
    F32,
    F16,
    BF16,
    /// blocks of 32: f16 scale + 16 bytes of 4-bit values (offset 8)
    Q4_0,
    /// blocks of 32: f16 scale, f16 min + 16 bytes of 4-bit values
    Q4_1,
    /// blocks of 32: f16 scale + 32 high bits + 16 bytes of low nibbles (offset 16)
    Q5_0,
    /// blocks of 32: f16 scale, f16 min + 32 high bits + 16 bytes of low nibbles
    Q5_1,
    /// blocks of 32: f16 scale + 32 x i8
    Q8_0,
    /// super-blocks of 256: f16 scale and min, 6-bit sub-scales and mins, 4-bit values
    Q4K,
    /// as Q4K plus a fifth bit per value
    Q5K,
    /// super-blocks of 256: 4 + 2 bit values, int8 sub-scales, f16 scale
    Q6K,
    /// super-blocks of 256: f16 scale, 6-bit sub-scales, 4-bit indices into a table
    Iq4Xs,
    /// blocks of 32: e8m0 scale + 16 bytes of e2m1 pairs
    Mxfp4,
    /// a type the kernels do not decode
    Other(u32),
}

/// (values per block, bytes per block) of every ggml type id, by id.
fn ggml_block(id: u32) -> Option<(usize, usize)> {
    Some(match id {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        34 => (256, 54),
        35 => (256, 66),
        39 => (32, 17),
        _ => return None,
    })
}

impl Ty {
    fn from_id(id: u32) -> Result<Ty> {
        Ok(match id {
            0 => Ty::F32,
            1 => Ty::F16,
            2 => Ty::Q4_0,
            3 => Ty::Q4_1,
            6 => Ty::Q5_0,
            7 => Ty::Q5_1,
            8 => Ty::Q8_0,
            12 => Ty::Q4K,
            13 => Ty::Q5K,
            14 => Ty::Q6K,
            23 => Ty::Iq4Xs,
            30 => Ty::BF16,
            39 => Ty::Mxfp4,
            _ if ggml_block(id).is_some() => Ty::Other(id),
            _ => bail!("unknown ggml tensor type {id}"),
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Ty::F32 => 0,
            Ty::F16 => 1,
            Ty::Q4_0 => 2,
            Ty::Q4_1 => 3,
            Ty::Q5_0 => 6,
            Ty::Q5_1 => 7,
            Ty::Q8_0 => 8,
            Ty::Q4K => 12,
            Ty::Q5K => 13,
            Ty::Q6K => 14,
            Ty::Iq4Xs => 23,
            Ty::BF16 => 30,
            Ty::Mxfp4 => 39,
            Ty::Other(id) => id,
        }
    }

    /// The ggml name, for messages.
    pub fn name(self) -> String {
        match self {
            Ty::F32 => "F32".into(),
            Ty::F16 => "F16".into(),
            Ty::BF16 => "BF16".into(),
            Ty::Q4_0 => "Q4_0".into(),
            Ty::Q4_1 => "Q4_1".into(),
            Ty::Q5_0 => "Q5_0".into(),
            Ty::Q5_1 => "Q5_1".into(),
            Ty::Q8_0 => "Q8_0".into(),
            Ty::Q4K => "Q4_K".into(),
            Ty::Q5K => "Q5_K".into(),
            Ty::Q6K => "Q6_K".into(),
            Ty::Iq4Xs => "IQ4_XS".into(),
            Ty::Mxfp4 => "MXFP4".into(),
            Ty::Other(id) => match id {
                10 => "Q2_K".into(),
                11 => "Q3_K".into(),
                16..=22 | 29 => format!("IQ type {id}"),
                34 | 35 => format!("ternary type {id}"),
                _ => format!("type {id}"),
            },
        }
    }

    /// (values, bytes) per block as stored.
    pub fn block(self) -> (usize, usize) {
        ggml_block(self.id()).expect("known type")
    }

    /// Bytes for `n` elements.
    pub fn bytes(self, n: usize) -> usize {
        let (v, b) = self.block();
        n / v * b
    }

    /// u32 words per block in the GPU layout (`gpu_layout`), or None for a
    /// type the kernels do not decode. F32/F16/BF16 count blocks of 4 values.
    pub fn gpu_words(self) -> Option<usize> {
        Some(match self {
            Ty::F32 => 4,
            Ty::F16 | Ty::BF16 => 2,
            Ty::Q4_0 | Ty::Q4_1 | Ty::Mxfp4 => 5,
            Ty::Q5_0 | Ty::Q5_1 => 6,
            Ty::Q8_0 => 9,
            Ty::Q4K => 36,
            Ty::Q5K => 44,
            Ty::Q6K => 53,
            Ty::Iq4Xs => 34,
            Ty::Other(_) => return None,
        })
    }

    /// Values per block in the GPU layout.
    pub fn gpu_values(self) -> usize {
        match self {
            Ty::F32 | Ty::F16 | Ty::BF16 => 4,
            _ => self.block().0,
        }
    }
}

/// `raw` (whole blocks of type `ty`) in the GPU layout: every field starts
/// on a 4-byte boundary, so kernels read words. Types whose blocks are
/// already word-sized multiples are unchanged; the rest gain padding
/// (Q4_0, Q5_0, Q8_0 and MXFP4 after their scale, Q6_K at the end).
pub fn gpu_layout(ty: Ty, raw: &[u8]) -> Vec<u8> {
    let (_, bb) = ty.block();
    let out_b = ty.gpu_words().expect("decodable type") * 4;
    // (float types: groups of 4 values, as stored)
    if bb == out_b || matches!(ty, Ty::F32 | Ty::F16 | Ty::BF16) {
        return raw.to_vec();
    }
    let nb = raw.len() / bb;
    let mut out = vec![0u8; nb * out_b];
    out.par_chunks_mut(out_b * 1024).enumerate().for_each(|(c, o)| {
        for (i, ob) in o.chunks_exact_mut(out_b).enumerate() {
            let src = &raw[(c * 1024 + i) * bb..][..bb];
            match ty {
                // scale (2 or 1 bytes), padding, rest
                Ty::Q4_0 | Ty::Q5_0 | Ty::Q8_0 => {
                    ob[..2].copy_from_slice(&src[..2]);
                    ob[4..4 + bb - 2].copy_from_slice(&src[2..]);
                }
                Ty::Mxfp4 => {
                    ob[0] = src[0];
                    ob[4..4 + bb - 1].copy_from_slice(&src[1..]);
                }
                // Q6_K: padding at the end
                _ => ob[..bb].copy_from_slice(src),
            }
        }
    });
    out
}

#[derive(Debug, Clone)]
pub struct Tensor {
    /// ggml order: dims[0] is the contiguous (input) dimension
    pub dims: Vec<usize>,
    pub ty: Ty,
    /// absolute offset in the file
    pub offset: usize,
}

impl Tensor {
    pub fn n(&self) -> usize {
        self.dims.iter().product()
    }
    pub fn bytes(&self) -> usize {
        self.ty.bytes(self.n())
    }
}

pub struct Gguf {
    pub meta: HashMap<String, Val>,
    pub tensors: HashMap<String, Tensor>,
    map: memmap2::Mmap,
}

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let s = self.b.get(self.at..self.at + n).context("truncated GGUF header")?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn val(&mut self, ty: u32) -> Result<Val> {
        Ok(match ty {
            0 => Val::Int(self.take(1)?[0] as i64),
            1 => Val::Int(self.take(1)?[0] as i8 as i64),
            2 => Val::Int(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            3 => Val::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Val::Int(self.u32()? as i64),
            5 => Val::Int(self.u32()? as i32 as i64),
            6 => Val::F(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Val::Bool(self.take(1)?[0] != 0),
            8 => Val::Str(self.str()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.val(et)?);
                }
                Val::Arr(v)
            }
            10 => Val::Int(self.u64()? as i64),
            11 => Val::Int(self.u64()? as i64),
            12 => Val::F(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            _ => bail!("unknown GGUF value type {ty}"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf> {
        let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let map = unsafe { memmap2::Mmap::map(&f)? };
        let mut r = Rd { b: &map, at: 0 };
        if r.take(4)? != b"GGUF" {
            bail!("{} is not a GGUF file", path.display());
        }
        let ver = r.u32()?;
        if ver < 2 {
            bail!("GGUF version {ver} is too old");
        }
        let (nt, nkv) = (r.u64()? as usize, r.u64()? as usize);
        let mut meta = HashMap::new();
        for _ in 0..nkv {
            let k = r.str()?;
            let t = r.u32()?;
            meta.insert(k, r.val(t)?);
        }
        let mut dir = Vec::with_capacity(nt);
        for _ in 0..nt {
            let name = r.str()?;
            let nd = r.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(r.u64()? as usize);
            }
            let ty = Ty::from_id(r.u32()?).with_context(|| format!("tensor {name}"))?;
            let off = r.u64()? as usize;
            dir.push((name, dims, ty, off));
        }
        let align = meta.get("general.alignment").and_then(Val::as_u64).unwrap_or(32) as usize;
        let data = r.at.div_ceil(align) * align;
        let mut tensors = HashMap::new();
        for (name, dims, ty, off) in dir {
            let t = Tensor { dims, ty, offset: data + off };
            if t.offset + t.bytes() > map.len() {
                bail!("tensor {name} runs past the end of the file (incomplete download?)");
            }
            tensors.insert(name, t);
        }
        Ok(Gguf { meta, tensors, map })
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not in the GGUF file"))
    }

    pub fn bytes(&self, t: &Tensor) -> &[u8] {
        &self.map[t.offset..t.offset + t.bytes()]
    }

    pub fn u(&self, key: &str) -> Result<usize> {
        self.meta.get(key).and_then(Val::as_u64).map(|v| v as usize).with_context(|| format!("GGUF key {key} missing"))
    }

    pub fn f(&self, key: &str) -> Result<f32> {
        self.meta.get(key).and_then(Val::as_f64).map(|v| v as f32).with_context(|| format!("GGUF key {key} missing"))
    }

    pub fn s(&self, key: &str) -> Option<&str> {
        self.meta.get(key).and_then(Val::as_str)
    }

    pub fn b(&self, key: &str) -> Option<bool> {
        self.meta.get(key).and_then(Val::as_bool)
    }

    pub fn arch(&self) -> &str {
        self.s("general.architecture").unwrap_or("")
    }

    pub fn strs(&self, key: &str) -> Result<Vec<&str>> {
        match self.meta.get(key) {
            Some(Val::Arr(v)) => v.iter().map(|x| x.as_str().context("expected strings")).collect(),
            _ => bail!("GGUF key {key} missing"),
        }
    }

    pub fn ints(&self, key: &str) -> Result<Vec<i64>> {
        match self.meta.get(key) {
            Some(Val::Arr(v)) => v.iter().map(|x| x.as_u64().map(|u| u as i64).context("expected ints")).collect(),
            _ => bail!("GGUF key {key} missing"),
        }
    }
}

/// A byte-level BPE tokenizer for the GGUF's vocabulary ("gpt2" model; the
/// pre-tokenizer named in the file), built through the `tokenizers` crate.
pub fn tokenizer(g: &Gguf) -> Result<tokenizers::Tokenizer> {
    if g.s("tokenizer.ggml.model") != Some("gpt2") {
        bail!("this model's tokenizer ({:?}) is not supported yet: only byte-level BPE (\"gpt2\") ones are", g.s("tokenizer.ggml.model").unwrap_or("none"));
    }
    let pre = g.s("tokenizer.ggml.pre").unwrap_or("default");
    // the split patterns llama.cpp uses for each pre-tokenizer name
    let patterns: &[&str] = match pre {
        "gpt-4o" => &[r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"],
        "qwen2" | "qwen35" | "stablelm2" | "hunyuan" => &[r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"],
        "llama-bpe" | "llama3" | "smaug-bpe" | "falcon3" | "dbrx" | "pixtral" | "glm4" => &[r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"],
        // (llama.cpp gives DeepSeek V4's "joyai-llm" the same split as V3)
        "deepseek-v3" | "joyai-llm" => &[r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+", r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##],
        "default" | "gpt-2" | "gpt2" | "olmo" | "mpt" | "jais" | "trillion" => &[r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"],
        _ => bail!("GGUF pre-tokenizer {pre:?} not supported yet"),
    };
    let tokens = g.strs("tokenizer.ggml.tokens")?;
    let types = g.ints("tokenizer.ggml.token_type").unwrap_or_default();
    let merges = g.strs("tokenizer.ggml.merges")?;
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        // 3 = control, 4 = user-defined: matched whole, before BPE
        if matches!(types.get(i), Some(3) | Some(4)) {
            added.push(serde_json::json!({"id": i, "content": t, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}));
        }
        vocab.insert(t.to_string(), serde_json::json!(i));
    }
    let mut splits: Vec<serde_json::Value> = patterns.iter().map(|p| serde_json::json!({"type": "Split", "pattern": {"Regex": p}, "behavior": "Isolated", "invert": false})).collect();
    splits.push(serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}));
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": splits},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": merges}
    });
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json)?).map_err(|e| anyhow::anyhow!("building tokenizer: {e}"))
}
