//! Multi-shard safetensors index.
//!
//! Every shard is memory-mapped for the dense weights, which are read once at
//! load. Experts are read with positioned file reads instead (see experts.rs),
//! so the engine decides what stays in RAM rather than the page cache.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
    F16,
    U8,
    I8,
}

impl DType {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => DType::F32,
            "BF16" => DType::BF16,
            "F16" => DType::F16,
            "U8" => DType::U8,
            "I8" => DType::I8,
            _ => bail!("unsupported safetensors dtype {s}"),
        })
    }
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 | DType::F16 => 2,
            DType::U8 | DType::I8 => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shard: usize,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// absolute byte offset in the shard file
    pub offset: u64,
    pub len: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct Shards {
    pub files: Vec<File>,
    pub paths: Vec<std::path::PathBuf>,
    maps: Vec<Mmap>,
    index: HashMap<String, TensorInfo>,
}

impl Shards {
    /// Drop every mapped page from the working set by mapping the files anew.
    /// Loading touches gigabytes of the checkpoint; those pages stay in the
    /// process and hide that memory from the RAM tier until they are unmapped,
    /// after which the OS counts them as free standby cache.
    pub fn release_pages(&mut self) -> Result<()> {
        for (map, file) in self.maps.iter_mut().zip(&self.files) {
            // SAFETY: as in `open`.
            *map = unsafe { Mmap::map(file)? };
        }
        Ok(())
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }

        let mut files = Vec::with_capacity(paths.len());
        let mut maps = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (shard, path) in paths.iter().enumerate() {
            let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: the checkpoint is treated as read-only for the life of the process.
            let map = unsafe { Mmap::map(&file)? };
            if map.len() < 8 {
                bail!("{} is too short", path.display());
            }
            let header_len = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&map[8..8 + header_len])
                .with_context(|| format!("parsing header of {}", path.display()))?;
            let data_start = 8 + header_len as u64;
            for (name, meta) in header.as_object().context("header is not an object")? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = DType::parse(meta["dtype"].as_str().unwrap_or(""))?;
                let shape: Vec<usize> = meta["shape"]
                    .as_array()
                    .context("tensor without shape")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                let off = meta["data_offsets"].as_array().context("tensor without offsets")?;
                let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
                let info = TensorInfo {
                    shard,
                    dtype,
                    shape,
                    offset: data_start + a,
                    len: (b - a) as usize,
                };
                if info.numel() * dtype.size() != info.len {
                    bail!("{name}: shape and byte length disagree");
                }
                index.insert(name.clone(), info);
            }
            files.push(file);
            maps.push(map);
        }
        Ok(Shards { files, paths, maps, index })
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.index.get(name).with_context(|| format!("missing tensor {name}"))
    }

    pub fn bytes(&self, info: &TensorInfo) -> &[u8] {
        let start = info.offset as usize;
        &self.maps[info.shard][start..start + info.len]
    }

    /// Any float tensor as f32, checking the element count.
    pub fn f32(&self, name: &str, numel: usize) -> Result<Vec<f32>> {
        let info = self.info(name)?;
        if info.numel() != numel {
            bail!("{name}: {} values, expected {numel}", info.numel());
        }
        let raw = self.bytes(info);
        Ok(match info.dtype {
            DType::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            DType::BF16 => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            DType::F16 => raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            d => bail!("{name}: dtype {d:?} is not a float type"),
        })
    }

    /// One row of a 2-D bf16/f32 tensor, as f32 (used for the embedding table).
    pub fn row_f32(&self, info: &TensorInfo, row: usize, out: &mut [f32]) {
        let cols = info.shape[1];
        let es = info.dtype.size();
        let raw = &self.bytes(info)[row * cols * es..(row + 1) * cols * es];
        match info.dtype {
            DType::BF16 => {
                for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
                    *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                }
            }
            DType::F32 => {
                for (o, c) in out.iter_mut().zip(raw.chunks_exact(4)) {
                    *o = f32::from_le_bytes(c.try_into().unwrap());
                }
            }
            _ => unreachable!("row_f32 on {:?}", info.dtype),
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}
