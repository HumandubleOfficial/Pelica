//! A GGUF model's own chat template (`tokenizer.chat_template`, Jinja),
//! rendered with minijinja the way Hugging Face's `apply_chat_template`
//! does: messages, `add_generation_prompt`, and the special token strings.

use crate::gguf::Gguf;
use anyhow::{Context, Result};
use serde_json::Value;

pub struct ChatTemplate {
    source: String,
    bos: String,
    eos: String,
}

/// A token's text by id, if the vocabulary has it.
fn token_text(g: &Gguf, key: &str) -> String {
    let (Some(id), Ok(toks)) = (g.meta.get(key).and_then(|v| v.as_u64()), g.strs("tokenizer.ggml.tokens")) else { return String::new() };
    toks.get(id as usize).map(|s| s.to_string()).unwrap_or_default()
}

impl ChatTemplate {
    pub fn from_gguf(g: &Gguf) -> Option<ChatTemplate> {
        let source = g.s("tokenizer.chat_template")?.to_string();
        Some(ChatTemplate { source, bos: token_text(g, "tokenizer.ggml.bos_token_id"), eos: token_text(g, "tokenizer.ggml.eos_token_id") })
    }

    /// Does the template print a `tools` list (function calling)?
    pub fn supports_tools(&self) -> bool {
        self.source.contains("tools") && self.source.contains("tool_call")
    }

    /// The prompt for `messages` ([{role, content}, ...]), ending where the
    /// assistant's reply starts. `extra` adds template variables (such as
    /// `enable_thinking` or `reasoning_effort`).
    pub fn render(&self, messages: &[Value], extra: &serde_json::Map<String, Value>) -> Result<String> {
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        // Python-style string and list methods (.startswith, .split, ...)
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> { Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg)) });
        env.add_function("strftime_now", |fmt: String| -> String { strftime_now(&fmt) });
        env.add_template("chat", &self.source).context("parsing the model's chat template")?;
        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".into(), Value::Array(messages.to_vec()));
        ctx.insert("add_generation_prompt".into(), Value::Bool(true));
        ctx.insert("bos_token".into(), Value::String(self.bos.clone()));
        ctx.insert("eos_token".into(), Value::String(self.eos.clone()));
        ctx.insert("tools".into(), Value::Null);
        for (k, v) in extra {
            ctx.insert(k.clone(), v.clone());
        }
        let t = env.get_template("chat")?;
        t.render(minijinja::Value::from(minijinja::value::Serde(Value::Object(ctx)))).context("rendering the model's chat template")
    }
}

/// The current local date for templates that print it (only the common
/// "%d %b %Y" / "%Y-%m-%d" pieces).
fn strftime_now(fmt: &str) -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let days = secs.div_euclid(86400);
    // civil from days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    const MON: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const MONTH: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    fmt.replace("%d", &format!("{d:02}"))
        .replace("%m", &format!("{m:02}"))
        .replace("%Y", &y.to_string())
        .replace("%B", MONTH[(m - 1) as usize])
        .replace("%b", MON[(m - 1) as usize])
}
