//! Model hyperparameters, read from the checkpoint's config.json.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub n_layers: usize,
    pub vocab: usize,
    pub eps: f32,

    // mHC residual streams
    pub hc_mult: usize,
    pub hc_iters: usize,
    pub hc_eps: f32,

    // feed-forward
    pub first_dense: usize,
    pub dense_inter: usize,
    pub moe_inter: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub routed_scale: f32,
    pub swiglu_limit: f32,

    /// true = DSA (MLA) attention, false = KDA linear attention
    pub layer_full: Vec<bool>,

    // KDA
    pub kda_heads: usize,
    pub kda_hd: usize,
    pub conv_k: usize,
    pub gate_lb: f32,

    // MLA
    pub n_heads: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_nope: usize,
    pub v_head: usize,

    // DSA indexer
    pub index_topk: usize,
    pub index_kpool: usize,

    pub eos: Vec<u32>,
}

fn num(v: &Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("config.json: missing number `{key}`"))
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let root: Value = serde_json::from_str(&text)?;
        let t = root.get("text_config").unwrap_or(&root);

        let model_type = t.get("model_type").and_then(Value::as_str).unwrap_or("");
        if model_type != "glm5_next_text" {
            bail!("unsupported model_type `{model_type}` (Pelica currently runs GLM-5.3-Flash / glm5_next)");
        }

        let n_layers = num(t, "num_hidden_layers")? as usize;
        let layer_full: Vec<bool> = t
            .get("layer_types")
            .and_then(Value::as_array)
            .context("config.json: missing layer_types")?
            .iter()
            .map(|v| v.as_str() == Some("deepseek_sparse_attention"))
            .collect();
        if layer_full.len() != n_layers {
            bail!("layer_types has {} entries for {} layers", layer_full.len(), n_layers);
        }

        let lin = t.get("linear_attn_config").context("config.json: missing linear_attn_config")?;
        let eos = match t.get("eos_token_id") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
            Some(v) => v.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            None => vec![],
        };

        let cfg = Config {
            hidden: num(t, "hidden_size")? as usize,
            n_layers,
            vocab: num(t, "vocab_size")? as usize,
            eps: num(t, "rms_norm_eps")? as f32,
            hc_mult: num(t, "hc_mult")? as usize,
            hc_iters: num(t, "hc_sinkhorn_iters")? as usize,
            hc_eps: num(t, "hc_eps")? as f32,
            first_dense: num(t, "first_k_dense_replace")? as usize,
            dense_inter: num(t, "intermediate_size")? as usize,
            moe_inter: num(t, "moe_intermediate_size")? as usize,
            n_experts: num(t, "n_routed_experts")? as usize,
            topk: num(t, "num_experts_per_tok")? as usize,
            routed_scale: num(t, "routed_scaling_factor")? as f32,
            swiglu_limit: num(t, "swiglu_limit")? as f32,
            layer_full,
            kda_heads: num(lin, "num_heads")? as usize,
            kda_hd: num(lin, "head_dim")? as usize,
            conv_k: num(lin, "short_conv_kernel_size")? as usize,
            gate_lb: num(lin, "gate_lower_bound")? as f32,
            n_heads: num(t, "num_attention_heads")? as usize,
            q_lora: num(t, "q_lora_rank")? as usize,
            kv_lora: num(t, "kv_lora_rank")? as usize,
            qk_nope: num(t, "qk_nope_head_dim")? as usize,
            v_head: num(t, "v_head_dim")? as usize,
            index_topk: num(t, "index_topk")? as usize,
            index_kpool: num(t, "index_kpool")? as usize,
            eos,
        };
        if cfg.hc_mult != 4 {
            bail!("hc_mult {} unsupported (expected 4)", cfg.hc_mult);
        }
        if num(t, "qk_rope_head_dim").unwrap_or(0.0) != 0.0 {
            bail!("MLA with RoPE dims is not supported yet");
        }
        Ok(cfg)
    }

    /// Positions the DSA indexer can keep in full; at or below this context
    /// every position is selected, so sparse attention equals dense causal attention.
    pub fn dense_attention_limit(&self) -> usize {
        self.index_topk + self.index_kpool - 1
    }
}
