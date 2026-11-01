//! Chat API v2, for Pelica Desktop (see desktop/ARCHITECTURE.md): stateless
//! completions. The caller sends the whole conversation (system, user,
//! assistant with tool calls, tool results) and the tools it offers; the
//! reply streams back and stops at a tool call, which the caller runs.
//! Every engine takes tools: the chat template's own tool format when it has
//! one, else the calls described in words (`<tool_call>{json}</tool_call>`);
//! gpt-oss in harmony's functions namespace.

use super::*;

/// Live decoding speed: a `stats` event at most every 250 ms.
struct Live {
    first: Option<Instant>,
    last: Instant,
    n: usize,
}

impl Live {
    fn new() -> Self {
        Live { first: None, last: Instant::now(), n: 0 }
    }
    /// (false: the caller went away)
    fn token(&mut self, tx: &Sender<Value>) -> bool {
        self.n += 1;
        let now = Instant::now();
        let first = *self.first.get_or_insert(now);
        if self.n > 1 && now - self.last >= std::time::Duration::from_millis(250) {
            self.last = now;
            let el = (now - first).as_secs_f64();
            return tx.send(json!({ "t": "stats", "tps": (self.n - 1) as f64 / el.max(1e-9), "generated": self.n, "elapsed_s": el })).is_ok();
        }
        true
    }
}

/// The tools as the caller describes them: `{name, description, parameters}`
/// (OpenAI's `{type: function, function: {...}}` is accepted too).
fn tools_of(req: &Value) -> Vec<Value> {
    req["tools"].as_array().map(|a| a.iter().map(|t| if t["function"].is_object() { t["function"].clone() } else { t.clone() }).filter(|t| t["name"].is_string()).collect()).unwrap_or_default()
}

/// The tools for a template that prints them (OpenAI function schemas).
fn template_tools(tools: &[Value]) -> Value {
    Value::Array(tools.iter().map(|t| json!({ "type": "function", "function": { "name": t["name"], "description": t["description"], "parameters": t["parameters"] } })).collect())
}

/// The tools and the calling format described in words, for templates that
/// don't print tools.
fn tools_in_words(tools: &[Value]) -> String {
    let mut s = String::from("# Tools\n\nYou can call these tools:\n");
    for t in tools {
        s += &format!("- {}: {}\n  Arguments (JSON Schema): {}\n", t["name"].as_str().unwrap_or(""), t["description"].as_str().unwrap_or("").trim(), t["parameters"]);
    }
    s += "\nTo call a tool, reply with only this and stop:\n<tool_call>\n{\"name\": \"TOOL_NAME\", \"arguments\": {...}}\n</tool_call>\nThe result comes back inside <tool_response></tool_response>; then continue: call another tool, or answer.";
    // one worked example: small models follow a sample far better than a schema
    let has = |n: &str| tools.iter().any(|t| t["name"] == n);
    if has("write_file") {
        s += "\n\nFor example, to create a file:\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"hello.py\", \"content\": \"print('Hello')\\n\"}}\n</tool_call>\nUse the tools to do the work yourself instead of telling the user how to do it.";
    } else if has("web_search") {
        s += "\n\nFor example, to search:\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"python latest release\"}}\n</tool_call>";
    }
    s
}

/// A call written by the model: (name, arguments object). Takes
/// `{"name", "arguments"}` (arguments as an object or a JSON string), or
/// `"parameters"` for the arguments.
fn parse_call(call: &str) -> (String, Value) {
    let v: Value = serde_json::from_str(call.trim()).unwrap_or(Value::Null);
    let name = v["name"].as_str().unwrap_or("").to_string();
    let mut args = if v["arguments"].is_null() { v["parameters"].clone() } else { v["arguments"].clone() };
    if let Some(s) = args.as_str() {
        args = serde_json::from_str(s).unwrap_or(json!({}));
    }
    if !args.is_object() {
        args = json!({});
    }
    (name, args)
}

/// A call a model wrote as plain JSON instead of in the tool format (small
/// models do): the first complete `{"name": <one of `names`>, "arguments":
/// {...}}` object in `text`, with where it starts.
fn loose_call(text: &str, names: &[&str]) -> Option<(usize, String, Value)> {
    let b = text.as_bytes();
    let mut from = 0;
    while let Some(i) = text[from..].find('{').map(|i| i + from) {
        // the balanced object starting here (strings respected)
        let (mut depth, mut instr, mut esc, mut end) = (0i32, false, false, None);
        for (j, &c) in b[i..].iter().enumerate() {
            if instr {
                if esc {
                    esc = false;
                } else if c == b'\\' {
                    esc = true;
                } else if c == b'"' {
                    instr = false;
                }
                continue;
            }
            match c {
                b'"' => instr = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i + j + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let e = end?;
        if let Ok(v) = serde_json::from_str::<Value>(&text[i..e]) {
            let name = v["name"].as_str().unwrap_or("");
            if names.contains(&name) && (v["arguments"].is_object() || v["parameters"].is_object()) {
                let (n, a) = parse_call(&text[i..e]);
                return Some((i, n, a));
            }
        }
        from = i + 1;
    }
    None
}

fn call_id() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("call_{:x}", n & 0xffff_ffff_ffff)
}

/// The system text: every system message, joined.
fn system_of(msgs: &[Value]) -> String {
    msgs.iter().filter(|m| m["role"] == "system").filter_map(|m| m["content"].as_str()).map(str::trim).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n")
}

fn text_of(m: &Value) -> &str {
    m["content"].as_str().unwrap_or("")
}

/// A call as the words format writes it.
fn call_words(c: &Value) -> String {
    format!("<tool_call>\n{}\n</tool_call>", json!({ "name": c["name"], "arguments": c["arguments"] }))
}

/// The conversation for a chat template: native tool messages, or tool calls
/// and results in words.
fn template_conv(msgs: &[Value], native: bool) -> Vec<Value> {
    let mut conv: Vec<Value> = Vec::new();
    for m in msgs {
        match m["role"].as_str().unwrap_or("user") {
            "system" => {}
            "assistant" => {
                let calls = m["tool_calls"].as_array().cloned().unwrap_or_default();
                let mut x = if native && !calls.is_empty() {
                    let tc: Vec<Value> = calls.iter().map(|c| json!({ "type": "function", "id": c["id"], "function": { "name": c["name"], "arguments": c["arguments"] } })).collect();
                    json!({ "role": "assistant", "content": text_of(m), "tool_calls": tc })
                } else {
                    let mut t = text_of(m).trim().to_string();
                    for c in &calls {
                        if !t.is_empty() {
                            t += "\n";
                        }
                        t += &call_words(c);
                    }
                    json!({ "role": "assistant", "content": t })
                };
                if let Some(r) = m["reasoning"].as_str().filter(|r| !r.is_empty()) {
                    x["reasoning_content"] = json!(r);
                }
                conv.push(x);
            }
            "tool" => {
                let c = text_of(m);
                if native {
                    conv.push(json!({ "role": "tool", "content": c, "name": m["name"], "tool_call_id": m["tool_call_id"] }));
                } else {
                    let r = format!("<tool_response>\n{c}\n</tool_response>");
                    match conv.last_mut() {
                        // (several results in a row: one message)
                        Some(l) if l["role"] == "user" && l["content"].as_str().is_some_and(|s| s.starts_with("<tool_response>")) => {
                            l["content"] = json!(format!("{}\n{r}", l["content"].as_str().unwrap_or("")));
                        }
                        _ => conv.push(json!({ "role": "user", "content": r })),
                    }
                }
            }
            _ => conv.push(json!({ "role": "user", "content": text_of(m) })),
        }
    }
    conv
}

/// JSON as Python's json.dumps writes it (", " and ": ", non-ASCII kept),
/// which DeepSeek V4's encoder uses for tool schemas and arguments.
fn py_json(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}: {}", Value::String(k.clone()), py_json(v))).collect::<Vec<_>>().join(", ")),
        x => x.to_string(),
    }
}

const DS4_TOOLS: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{schemas}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

/// A tool call as DeepSeek V4 writes it (DSML).
fn ds4_call(c: &Value) -> String {
    let args = match &c["arguments"] {
        Value::String(s) => serde_json::from_str::<Value>(s).ok().filter(Value::is_object).unwrap_or_else(|| json!({ "arguments": s })),
        Value::Object(_) => c["arguments"].clone(),
        _ => json!({}),
    };
    let params: Vec<String> = args
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, v)| match v {
                    Value::String(s) => format!("<｜DSML｜parameter name=\"{k}\" string=\"true\">{s}</｜DSML｜parameter>"),
                    v => format!("<｜DSML｜parameter name=\"{k}\" string=\"false\">{}</｜DSML｜parameter>", py_json(v)),
                })
                .collect()
        })
        .unwrap_or_default();
    format!("<｜DSML｜invoke name=\"{}\">\n{}\n</｜DSML｜invoke>", c["name"].as_str().unwrap_or(""), params.join("\n"))
}

/// DeepSeek V4's prompt as its official encoder (encoding_dsv4.py) writes
/// it. The GGUF's own template drops the blank lines in the tool section and
/// before tool calls, so prompts drifted from what the model was trained on.
/// Tool results join the user turn as <tool_result> blocks; the reasoning
/// effort text leads in thinking mode; an earlier turn sent without its
/// reasoning reads as a non-thinking turn.
fn ds4_prompt(msgs: &[Value], sys: &str, tools: &[Value], thinking: bool, effort: &str) -> String {
    enum Turn<'a> {
        User(Vec<String>),
        Asst(&'a Value),
    }
    let mut turns: Vec<Turn> = Vec::new();
    for m in msgs {
        let part = match m["role"].as_str().unwrap_or("user") {
            "system" => continue,
            "assistant" => {
                turns.push(Turn::Asst(m));
                continue;
            }
            "tool" => format!("<tool_result>{}</tool_result>", text_of(m)),
            _ => text_of(m).to_string(),
        };
        // consecutive user messages and tool results are one user turn
        match turns.last_mut() {
            Some(Turn::User(parts)) => parts.push(part),
            _ => turns.push(Turn::User(vec![part])),
        }
    }
    let last_user = turns.iter().rposition(|t| matches!(t, Turn::User(_)));
    let thinks = |i: usize| match turns.get(i) {
        Some(Turn::Asst(a)) => thinking && (Some(i) > last_user || a["reasoning"].as_str().is_some_and(|r| !r.is_empty())),
        _ => false,
    };
    let mut t = String::from("<｜begin▁of▁sentence｜>");
    if thinking {
        t += crate::app::ds4_effort(effort);
    }
    t += sys;
    if !tools.is_empty() {
        let schemas: Vec<String> = tools.iter().map(|x| py_json(&json!({ "name": x["name"], "description": x["description"], "parameters": x["parameters"] }))).collect();
        t += "\n\n";
        t += &DS4_TOOLS.replace("{schemas}", &schemas.join("\n"));
    }
    for (i, turn) in turns.iter().enumerate() {
        match turn {
            Turn::User(parts) => {
                t += "<｜User｜>";
                t += &parts.join("\n\n");
                match turns.get(i + 1) {
                    Some(Turn::Asst(_)) => t += if thinks(i + 1) { "<｜Assistant｜><think>" } else { "<｜Assistant｜></think>" },
                    None => t += if thinking { "<｜Assistant｜><think>" } else { "<｜Assistant｜></think>" },
                    _ => {}
                }
            }
            Turn::Asst(a) => {
                if thinks(i) {
                    t += a["reasoning"].as_str().unwrap_or("");
                    t += "</think>";
                }
                t += text_of(a);
                let calls: Vec<String> = a["tool_calls"].as_array().map(|v| v.iter().map(ds4_call).collect()).unwrap_or_default();
                if !calls.is_empty() {
                    t += &format!("\n\n<｜DSML｜tool_calls>\n{}\n</｜DSML｜tool_calls>", calls.join("\n"));
                }
                t += "<｜end▁of▁sentence｜>";
            }
        }
    }
    t
}

/// The `done` event.
#[allow(clippy::too_many_arguments)]
fn done(tx: &Sender<Value>, prompt: usize, reused: usize, gen: usize, think_n: usize, answer_n: usize, think_s: Option<f64>, first_s: Option<f64>, prefill_s: f64, decode_s: f64, ctx: usize, finish: &str) {
    let _ = tx.send(json!({
        "t": "done",
        "prompt": prompt, "reused": reused, "generated": gen,
        "think_tokens": think_n, "answer_tokens": answer_n,
        "think_s": think_s, "first_s": first_s,
        "prefill_s": prefill_s, "decode_s": decode_s,
        "tps": gen.saturating_sub(1) as f64 / decode_s.max(1e-9),
        "ctx_used": prompt + gen, "ctx": ctx,
        "stopped": finish == "stopped", "truncated": finish == "length",
        "finish": finish,
    }));
}

fn too_long(n: usize, ctx: usize) -> anyhow::Error {
    anyhow::anyhow!("This conversation is {n} tokens, more than the {ctx}-token context window. Compact it (/compact), start a new chat, or load the model with a larger context.")
}

/// Standard base64 (no line breaks) to bytes.
fn unbase64(s: &str) -> Result<Vec<u8>> {
    let val = |c: u8| -> Result<u32> {
        Ok(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => bail!("bad base64"),
        } as u32)
    };
    let b: Vec<u8> = s.bytes().filter(|&c| c != b'=' && !c.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(b.len() * 3 / 4);
    for ch in b.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in ch.iter().enumerate() {
            n |= val(c)? << (18 - 6 * i);
        }
        out.extend_from_slice(&n.to_be_bytes()[1..ch.len()]);
    }
    Ok(out)
}

/// The pictures in the user's messages (`images`: {w, h, rgb} with the
/// pixels resized by the app as Qwen's processor does), encoded; each user
/// message gets a marker per picture before its text, which becomes the
/// picture's rows after tokenizing. Returns (id, rows) in order.
fn take_images(m: &Moe, msgs: &mut [Value]) -> Result<Vec<(u32, usize)>> {
    let mut out = Vec::new();
    if !m.eng.vision() {
        return Ok(out);
    }
    // (the user's photos, and pictures a tool answered with: a connector's screenshot or chart)
    for msg in msgs.iter_mut().filter(|x| x["role"] == "user" || x["role"] == "tool") {
        let Some(imgs) = msg["images"].as_array().cloned() else { continue };
        let mut marks = String::new();
        for im in &imgs {
            let (Some(w), Some(h), Some(b)) = (im["w"].as_u64(), im["h"].as_u64(), im["rgb"].as_str()) else { continue };
            let rgb = unbase64(b)?;
            let (id, nx, ny) = m.eng.add_image(&rgb, w as usize, h as usize)?;
            out.push((id, nx * ny));
            marks += "<|vision_start|><|image_pad|><|vision_end|>";
        }
        if !marks.is_empty() {
            msg["content"] = json!(format!("{marks}{}", text_of(msg)));
        }
    }
    Ok(out)
}

pub(super) fn chat_moe(m: &mut Moe, req: &Value, tx: &Sender<Value>, stop: &AtomicBool) -> Result<()> {
    let mut msgs = req["messages"].as_array().cloned().unwrap_or_default();
    let pictures = take_images(m, &mut msgs)?;
    let tools = tools_of(req);
    let native = !tools.is_empty() && m.template.as_ref().is_some_and(|t| t.supports_tools());
    let mut sys = system_of(&msgs);
    if !tools.is_empty() && !native {
        sys = if sys.is_empty() { tools_in_words(&tools) } else { format!("{sys}\n\n{}", tools_in_words(&tools)) };
    }
    let effort = req["effort"].as_str().unwrap_or("");
    let text = if m.ds4 {
        // DeepSeek's sampling (model card: temperature 1.0, top-p 0.95 for
        // agent work, 1.0 otherwise); greedy decoding loops and overthinks
        let temp = req["temperature"].as_f64().unwrap_or(1.0) as f32;
        let top_p = req["top_p"].as_f64().unwrap_or(if tools.is_empty() { 1.0 } else { 0.95 }) as f32;
        m.eng.set_sampling(temp, top_p, req["seed"].as_u64());
        ds4_prompt(&msgs, &system_of(&msgs), &tools, m.think && !matches!(effort, "off" | "none"), effort)
    } else {
        // greedy unless the caller asks for a temperature (the local API)
        m.eng.set_sampling(req["temperature"].as_f64().unwrap_or(0.0) as f32, req["top_p"].as_f64().unwrap_or(1.0) as f32, req["seed"].as_u64());
        let conv = template_conv(&msgs, native);
        let mut extra = serde_json::Map::new();
        if m.think {
            extra.insert("enable_thinking".into(), json!(!matches!(effort, "off" | "none")));
        }
        if let Some(e) = template_effort(m, effort) {
            extra.insert("reasoning_effort".into(), json!(e));
        }
        let sys = with_effort(m, sys, effort);
        if native {
            extra.insert("tools".into(), template_tools(&tools));
        }
        moe_prompt(m, &conv, &sys, &extra)?
    };
    let mut ids = m.tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec();
    if let (Some(pad), false) = (m.eng.image_pad(), pictures.is_empty()) {
        // each picture's marker becomes its rows, in order
        let mut pics = pictures.iter();
        let mut out = Vec::with_capacity(ids.len() + pictures.iter().map(|p| p.1).sum::<usize>());
        for &t in &ids {
            match (t == pad).then(|| pics.next()).flatten() {
                Some(&(id, n)) => out.extend((0..n).map(|r| crate::q4x::Engine::image_token(id, r))),
                None => out.push(t),
            }
        }
        ids = out;
    }
    let ctx = m.eng.cfg.ctx;
    if ids.len() + 8 > ctx {
        return Err(too_long(ids.len(), ctx));
    }
    // max_tokens limits the answer; thinking may use the rest of the context
    // (DeepSeek V4 at High/Max can think past 8K tokens; counting thinking
    // against the cap ended such replies before they answered)
    let room = ctx - ids.len() - 4;
    let answer_cap = (req["max_tokens"].as_u64().unwrap_or(8192) as usize).clamp(1, room);
    let n = if text.trim_end().ends_with("<think>") { room } else { answer_cap };
    let reuse = m.eng.reusable(&m.cached, &ids);
    m.cached.clear();
    let t0 = Instant::now();
    let _ = tx.send(json!({ "t": "start", "prompt": ids.len(), "reused": reuse, "ctx": ctx }));
    let want_experts = req["experts"].as_bool().unwrap_or(false);
    if want_experts {
        let (l, e) = m.eng.expert_grid();
        let _ = tx.send(json!({ "t": "experts", "layers": l, "experts": e, "hits": [] }));
    }

    let tok = m.tok.clone();
    let eos = m.eng.eos.clone();
    let mut split = ThinkSplit { think: text.trim_end().ends_with("<think>"), buf: String::new(), answered: false };
    let mut gate = ToolGate::new();
    let use_gate = !tools.is_empty();
    let mut part = Delta::new();
    let mut live = Live::new();
    let (mut think_t0, mut think_t1, mut first) = (None::<Instant>, None::<Instant>, None::<Instant>);
    let (mut think_n, mut answer_n) = (0usize, 0usize);
    let mut gone = false;
    let mut gen = Vec::new();
    let eng = &m.eng;
    // expert activity, read every 300 ms: (counters as last read, when)
    let seen = std::cell::RefCell::new((Vec::new(), Instant::now()));
    let poll = |force: bool| {
        let mut s = seen.borrow_mut();
        if want_experts && (force || s.1.elapsed().as_millis() >= 300) {
            s.1 = Instant::now();
            if let Ok(h) = eng.activity(&mut s.0) {
                if !h.is_empty() {
                    let _ = tx.send(json!({ "t": "experts", "hits": h }));
                }
            }
        }
    };
    poll(true);
    // the answer as shown, and a call written as plain JSON in it
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    let mut shown = String::new();
    let mut loose: Option<(usize, String, Value)> = None;
    let send = |evs: Vec<(bool, String)>, gate: &mut ToolGate, gone: &mut bool, shown: &mut String, loose: &mut Option<(usize, String, Value)>| {
        for (think, d) in evs {
            let d = if think || !use_gate { d } else { gate.feed(&d) };
            if !d.is_empty() && tx.send(json!({ "t": if think { "think" } else { "text" }, "d": d })).is_err() {
                *gone = true;
            }
            if !think && use_gate && loose.is_none() {
                *shown += &d;
                if d.contains('}') {
                    *loose = loose_call(shown, &names);
                }
            }
        }
    };
    let sp0 = m.eng.speed_counters();
    let stats = eng.generate_with(&ids, n, reuse, |t| {
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
            send(split.feed(&d), &mut gate, &mut gone, &mut shown, &mut loose);
        }
        if !live.token(tx) {
            gone = true;
        }
        poll(false);
        !gone && !gate.done && loose.is_none() && answer_n < answer_cap && !stop.load(Ordering::Relaxed)
    }, |d, total| {
        poll(false);
        tx.send(json!({ "t": "prefill", "done": d, "total": total })).is_ok() && !stop.load(Ordering::Relaxed)
    })?;
    send(split.finish(), &mut gate, &mut gone, &mut shown, &mut loose);
    if use_gate {
        let rest = gate.finish();
        if !rest.is_empty() {
            let _ = tx.send(json!({ "t": "text", "d": rest }));
        }
    }
    poll(true);
    if gen.is_empty() && stats.prompt + 1 < ids.len() {
        m.cached = ids[..stats.prompt].to_vec();
    } else {
        m.cached = ids.clone();
        m.cached.extend_from_slice(&gen[..gen.len().saturating_sub(1)]);
    }
    let _ = m.eng.save_usage();
    if let (Some(a), Some(b)) = (sp0, m.eng.speed_counters()) {
        let mut v = crate::q4x::speed_report(&a, &b);
        v["t"] = json!("speed");
        v["passes"] = json!(stats.passes);
        let _ = tx.send(v);
    }
    let stopped = gone || stop.load(Ordering::Relaxed);
    let finish = if let Some(call) = gate.call.take().filter(|_| !stopped) {
        let (name, args) = parse_call(&call);
        let _ = tx.send(json!({ "t": "tool_call", "id": call_id(), "name": name, "arguments": args, "raw": if name.is_empty() { json!(call) } else { Value::Null } }));
        "tool_call"
    } else if let Some((at, name, args)) = loose.filter(|_| !stopped) {
        // (with the text shown before it: the caller drops the JSON it already showed)
        let before = shown[..at].trim_end().trim_end_matches("```json").trim_end_matches("```").trim_end().to_string();
        let _ = tx.send(json!({ "t": "tool_call", "id": call_id(), "name": name, "arguments": args, "before": before }));
        "tool_call"
    } else if stopped {
        "stopped"
    } else if gen.last().is_some_and(|t| eos.contains(t)) {
        "stop"
    } else {
        "length"
    };
    let think_s = think_t0.map(|a| (think_t1.unwrap_or_else(Instant::now) - a).as_secs_f64());
    if finish == "length" {
        // where it ran out: the answer cap, or the context (still thinking)
        let _ = tx.send(json!({ "t": "cut", "while": if split.think { "thinking" } else { "answer" }, "context": gen.len() + 1 >= n && n == room }));
    }
    done(tx, ids.len(), reuse, gen.len(), think_n, answer_n, think_s, first.map(|f| (f - t0).as_secs_f64()), stats.prefill_s, stats.decode_s, ctx, finish);
    Ok(())
}

/// A JSON Schema type as TypeScript, for harmony's tool declarations.
fn ts_type(s: &Value) -> String {
    if let Some(e) = s["enum"].as_array() {
        return e.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" | ");
    }
    match s["type"].as_str() {
        Some("string") => "string".into(),
        Some("number" | "integer") => "number".into(),
        Some("boolean") => "boolean".into(),
        Some("array") => format!("{}[]", if s["items"].is_object() { ts_type(&s["items"]) } else { "any".into() }),
        Some("object") if s["properties"].is_object() => {
            let req: Vec<&str> = s["required"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
            let mut t = String::from("{\n");
            for (k, v) in s["properties"].as_object().unwrap() {
                if let Some(d) = v["description"].as_str() {
                    t += &format!("// {}\n", d.replace('\n', " "));
                }
                t += &format!("{k}{}: {},\n", if req.contains(&k.as_str()) { "" } else { "?" }, ts_type(v));
            }
            t + "}"
        }
        _ => "any".into(),
    }
}

/// The conversation in harmony with tools, ending where the assistant's
/// reply starts. Reasoning is kept for the turns after the last user message
/// (a chain of tool calls), dropped before.
fn harmony_v2(effort: &str, msgs: &[Value], tools: &[Value]) -> String {
    let mut t = format!(
        "<|start|>system<|message|>You are a helpful assistant (gpt-oss-20b) running locally through Pelica.\nKnowledge cutoff: 2024-06\nCurrent date: {}\n\nReasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.{}<|end|>",
        today(),
        if tools.is_empty() { "" } else { "\nCalls to these tools must go to the commentary channel: 'functions'." }
    );
    let sys = system_of(msgs);
    if !sys.is_empty() || !tools.is_empty() {
        t += "<|start|>developer<|message|>";
        if !sys.is_empty() {
            t += &format!("# Instructions\n\n{sys}\n\n");
        }
        if !tools.is_empty() {
            t += "# Tools\n\n## functions\n\nnamespace functions {\n\n";
            for x in tools {
                if let Some(d) = x["description"].as_str() {
                    t += &format!("// {}\n", d.trim().replace('\n', "\n// "));
                }
                let p = &x["parameters"];
                let has = p["properties"].as_object().is_some_and(|o| !o.is_empty());
                t += &format!("type {} = ({}) => any;\n\n", x["name"].as_str().unwrap_or(""), if has { format!("_: {}", ts_type(p)) } else { String::new() });
            }
            t += "} // namespace functions";
        }
        t = t.trim_end().to_string() + "<|end|>";
    }
    let last_user = msgs.iter().rposition(|m| m["role"] == "user").unwrap_or(0);
    for (i, m) in msgs.iter().enumerate() {
        let c = text_of(m);
        match m["role"].as_str().unwrap_or("user") {
            "system" => {}
            "assistant" => {
                let calls = m["tool_calls"].as_array().cloned().unwrap_or_default();
                if i > last_user {
                    if let Some(r) = m["reasoning"].as_str().filter(|r| !r.is_empty()) {
                        t += &format!("<|start|>assistant<|channel|>analysis<|message|>{r}<|end|>");
                    }
                }
                if !c.trim().is_empty() {
                    t += &format!("<|start|>assistant<|channel|>{}<|message|>{c}<|end|>", if calls.is_empty() { "final" } else { "commentary" });
                }
                for k in &calls {
                    t += &format!("<|start|>assistant<|channel|>commentary to=functions.{} <|constrain|>json<|message|>{}<|call|>", k["name"].as_str().unwrap_or(""), k["arguments"]);
                }
            }
            "tool" => t += &format!("<|start|>functions.{} to=assistant<|channel|>commentary<|message|>{c}<|end|>", m["name"].as_str().unwrap_or("")),
            _ => t += &format!("<|start|>user<|message|>{c}<|end|>"),
        }
    }
    t + "<|start|>assistant"
}

pub(super) fn chat_oss(m: &mut Oss, req: &Value, tx: &Sender<Value>, stop: &AtomicBool) -> Result<()> {
    let effort = match req["effort"].as_str() {
        Some(e @ ("low" | "medium" | "high")) => e,
        Some("max") => "high",
        Some("off" | "none") => "low",
        _ => "medium",
    };
    let msgs = req["messages"].as_array().cloned().unwrap_or_default();
    let tools = tools_of(req);
    let text = harmony_v2(effort, &msgs, &tools);
    let ids = m.tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec();
    let ctx = m.eng.cfg.ctx;
    if ids.len() + 2 > ctx {
        return Err(too_long(ids.len(), ctx));
    }
    let n = (req["max_tokens"].as_u64().unwrap_or(8192) as usize).clamp(1, ctx - ids.len());
    let reuse = m.eng.reusable(&m.cached, m.written, &ids);
    m.cached.clear();
    let t0 = Instant::now();
    let _ = tx.send(json!({ "t": "start", "prompt": ids.len(), "reused": reuse, "ctx": ctx }));

    let tok = m.tok.clone();
    let id = |name: &str| tok.token_to_id(name).unwrap_or(u32::MAX);
    let (t_start, t_channel, t_message, t_end, t_constrain) = (id("<|start|>"), id("<|channel|>"), id("<|message|>"), id("<|end|>"), id("<|constrain|>"));
    let t_call = id("<|call|>");
    let stops = [id("<|return|>"), t_call, id("<|endoftext|>")];
    #[derive(PartialEq)]
    enum Mode {
        Header,
        Channel,
        Body,
    }
    let mut mode = Mode::Header;
    // the header's tokens (role, recipient, channel, constraint)
    let mut hdr = Vec::new();
    let mut part = Delta::new();
    let mut chan = String::from("final");
    let mut recipient: Option<String> = None;
    let mut args = Delta::new();
    let mut gen = Vec::new();
    let mut live = Live::new();
    let (mut think_t0, mut think_t1, mut first) = (None::<Instant>, None::<Instant>, None::<Instant>);
    let (mut think_n, mut answer_n) = (0usize, 0usize);
    let mut gone = false;
    let stats = m.eng.generate_with(&ids, n, DRAFT, false, reuse, |t| {
        gen.push(t);
        first.get_or_insert_with(Instant::now);
        if stops.contains(&t) {
            return false;
        }
        if !live.token(tx) {
            gone = true;
        }
        if t == t_channel {
            mode = Mode::Channel;
            hdr.push(t);
        } else if t == t_message {
            let h = tok.decode(&hdr, false).unwrap_or_default();
            // "assistant<|channel|>commentary to=functions.x <|constrain|>json"
            let after = h.split("<|channel|>").nth(1).unwrap_or("final");
            chan = after.split_whitespace().next().unwrap_or("final").to_string();
            recipient = h.split_whitespace().find_map(|w| w.strip_prefix("to=")).map(|r| r.trim_end_matches("<|constrain|>").to_string());
            if chan == "final" && think_t0.is_some() && think_t1.is_none() {
                think_t1 = Some(Instant::now());
            }
            mode = Mode::Body;
            part = Delta::new();
            args = Delta::new();
            hdr.clear();
        } else if t == t_end || t == t_start {
            mode = Mode::Header;
            hdr.clear();
        } else if mode != Mode::Body || t == t_constrain {
            hdr.push(t);
        } else if recipient.is_some() {
            // a tool call's arguments: kept, not shown
            args.push(&tok, t);
        } else {
            let think = chan == "analysis";
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
    }, |done, total| tx.send(json!({ "t": "prefill", "done": done, "total": total })).is_ok() && !stop.load(Ordering::Relaxed))?;
    if gen.is_empty() && stats.prompt + 1 < ids.len() {
        m.cached = ids[..stats.prompt].to_vec();
        m.written = stats.prompt;
    } else {
        m.cached = ids.clone();
        m.cached.extend_from_slice(&gen[..gen.len().saturating_sub(1)]);
        m.written = ids.len() + gen.len() + 4 * 8;
    }
    let stopped = gone || stop.load(Ordering::Relaxed);
    let finish = match (&recipient, gen.last()) {
        (Some(r), Some(&t)) if t == t_call && !stopped => {
            let name = r.strip_prefix("functions.").unwrap_or(r).to_string();
            let mut a: Value = serde_json::from_str(args.text.trim()).unwrap_or(json!({}));
            if !a.is_object() {
                a = json!({});
            }
            let _ = tx.send(json!({ "t": "tool_call", "id": call_id(), "name": name, "arguments": a }));
            "tool_call"
        }
        _ if stopped => "stopped",
        (_, Some(t)) if stops.contains(t) => "stop",
        _ => "length",
    };
    let think_s = think_t0.map(|a| (think_t1.unwrap_or_else(Instant::now) - a).as_secs_f64());
    done(tx, ids.len(), reuse, gen.len(), think_n, answer_n, think_s, first.map(|f| (f - t0).as_secs_f64()), stats.prefill_s, stats.decode_s, ctx, finish);
    Ok(())
}

pub(super) fn chat_glm(g: &mut Glm, req: &Value, tx: &Sender<Value>, stop: &AtomicBool) -> Result<()> {
    let effort = match req["effort"].as_str() {
        Some(e @ ("off" | "low" | "high" | "max")) => e,
        Some("none") => "off",
        Some("medium") => "low",
        _ => "low",
    };
    let think = effort != "off";
    let level = match effort {
        "high" => "High",
        "max" => "Max",
        _ => "Low",
    };
    let msgs = req["messages"].as_array().cloned().unwrap_or_default();
    let tools = tools_of(req);
    let mut sys = system_of(&msgs);
    if !tools.is_empty() {
        sys = if sys.is_empty() { tools_in_words(&tools) } else { format!("{sys}\n\n{}", tools_in_words(&tools)) };
    }
    let obs = if g.tok.token_to_id("<|observation|>").is_some() { "<|observation|>" } else { "<|user|>" };
    let mut text = format!("[gMASK]<sop><|system|>Reasoning Effort: {level}");
    if !sys.is_empty() {
        text += &format!("<|system|>{sys}");
    }
    for m in template_conv(&msgs, false) {
        let c = text_of(&m);
        if m["role"] == "assistant" {
            text += &format!("<|assistant|><think></think>{}", c.trim());
        } else if c.starts_with("<tool_response>") {
            text += &format!("{obs}{c}");
        } else {
            text += &format!("<|user|>{c}");
        }
    }
    text += if think { "<|assistant|><think>" } else { "<|assistant|><think></think>" };
    let all = g.tok.encode(text.as_str(), false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?.get_ids().to_vec();
    if all.len() + 8 > g.cap {
        bail!("This conversation is {} tokens; GLM-5.3 runs up to {} tokens here. Compact it (/compact) or start a new chat.", all.len(), g.cap);
    }
    // continue the session when it has read a prefix of this prompt
    let cont = g.sess.is_some() && g.read.len() < all.len() && all.starts_with(&g.read);
    if !cont {
        g.sess = Some(g.model.new_session(g.cap));
        g.read.clear();
    }
    // (v1 can't continue a session v2 used)
    g.conv.clear();
    g.header.clear();
    let base = g.read.len();
    let new = all[base..].to_vec();
    let t0 = Instant::now();
    let _ = tx.send(json!({ "t": "start", "prompt": all.len(), "reused": base, "ctx": g.cap }));
    let sess = g.sess.as_mut().unwrap();
    let mut timing = Timing::default();
    let mut logits = Vec::new();
    for (i, chunk) in new.chunks(CHUNK).enumerate() {
        logits = g.model.forward(sess, chunk, &mut timing)?;
        g.read.extend_from_slice(chunk);
        let _ = tx.send(json!({ "t": "prefill", "done": (i * CHUNK + chunk.len()), "total": new.len() }));
        if stop.load(Ordering::Relaxed) {
            g.sess = None;
            done(tx, all.len(), base, 0, 0, 0, None, None, t0.elapsed().as_secs_f64(), 0.0, g.cap, "stopped");
            return Ok(());
        }
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let think_open = g.tok.token_to_id("<think>").unwrap_or(u32::MAX);
    let think_close = g.tok.token_to_id("</think>").unwrap_or(u32::MAX);
    let eos = g.model.cfg.eos.clone();
    let n = (req["max_tokens"].as_u64().unwrap_or(8192) as usize).min(g.cap - g.read.len() - 1);
    let mut thinking = think;
    let (mut th, mut ans) = (Delta::new(), Delta::new());
    let mut gate = ToolGate::new();
    let mut live = Live::new();
    let (mut think_t0, mut think_t1) = (None::<Instant>, None::<Instant>);
    let (mut gen, mut think_n, mut answer_n, mut ended) = (0usize, 0usize, 0usize, false);
    while gen < n {
        let next = argmax(&logits);
        if eos.contains(&next) {
            ended = true;
            break;
        }
        gen += 1;
        if !live.token(tx) {
            stop.store(true, Ordering::Relaxed);
        }
        if next == think_close {
            thinking = false;
            if think_t0.is_some() {
                think_t1 = Some(Instant::now());
            }
        } else if next != think_open {
            let d = if thinking {
                think_n += 1;
                think_t0.get_or_insert_with(Instant::now);
                th.push(&g.tok, next).map(|d| ("think", d))
            } else {
                answer_n += 1;
                ans.push(&g.tok, next).map(|d| ("text", if tools.is_empty() { d } else { gate.feed(&d) }))
            };
            if let Some((kind, d)) = d.filter(|(_, d)| !d.is_empty()) {
                if tx.send(json!({ "t": kind, "d": d })).is_err() {
                    stop.store(true, Ordering::Relaxed);
                }
            }
        }
        logits = g.model.forward(sess, &[next], &mut timing)?;
        g.read.push(next);
        if gate.done || stop.load(Ordering::Relaxed) {
            break;
        }
    }
    if !tools.is_empty() {
        let rest = gate.finish();
        if !rest.is_empty() {
            let _ = tx.send(json!({ "t": "text", "d": rest }));
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    if let Err(e) = g.model.experts.save_usage(&g.dir.join(".pelica_usage")) {
        eprintln!("[pelica] could not save routing history: {e}");
    }
    let stopped = stop.load(Ordering::Relaxed);
    let finish = if let Some(call) = gate.call.take().filter(|_| !stopped) {
        let (name, args) = parse_call(&call);
        let _ = tx.send(json!({ "t": "tool_call", "id": call_id(), "name": name, "arguments": args }));
        "tool_call"
    } else if stopped {
        "stopped"
    } else if ended {
        "stop"
    } else {
        "length"
    };
    let think_s = think_t0.map(|a| (think_t1.unwrap_or_else(Instant::now) - a).as_secs_f64());
    done(tx, all.len(), base, gen, think_n, answer_n, think_s, Some(prefill_s), prefill_s, decode_s, g.cap, finish);
    Ok(())
}

/// `/api/trace`: the experts each token of `req["trace"]` (plain text, no
/// template, up to 384 tokens) goes to in every layer, and with what weight.
pub(super) fn trace_moe(m: &mut Moe, req: &Value, tx: &Sender<Value>) -> Result<()> {
    let text = req["trace"].as_str().unwrap_or("");
    let enc = m.tok.encode(text, false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().iter().copied().take(384).collect();
    if ids.is_empty() {
        bail!("type something to trace");
    }
    // the passes overwrite the cache from position 0
    m.cached.clear();
    let routes = m.eng.trace(&ids)?;
    let tokens: Vec<String> = ids.iter().map(|&i| m.tok.decode(&[i], false).unwrap_or_default()).collect();
    let (layers, experts) = m.eng.expert_grid();
    let routes: Vec<Value> = routes.iter().map(|tk| json!(tk.iter().map(|l| l.iter().map(|&(e, w)| json!([e, (w * 1e4).round() / 1e4])).collect::<Vec<_>>()).collect::<Vec<_>>())).collect();
    let _ = tx.send(json!({ "t": "trace", "tokens": tokens, "layers": layers, "experts": experts, "routes": routes }));
    Ok(())
}

/// `/api/count` with `parts`: tokens of the system text, the messages and
/// the tool schemas, for the context breakdown.
pub(super) fn count_parts(tok: &tokenizers::Tokenizer, parts: &Value) -> Value {
    let n = |s: &str| if s.is_empty() { 0 } else { tok.encode(s, false).map(|e| e.len()).unwrap_or(0) };
    let system = n(parts["system"].as_str().unwrap_or(""));
    let mut messages = 0;
    let mut tool_results = 0;
    for m in parts["messages"].as_array().into_iter().flatten() {
        // (a few tokens of each message's framing)
        let mut k = n(text_of(m)) + 4;
        if let Some(r) = m["reasoning"].as_str() {
            k += n(r);
        }
        for c in m["tool_calls"].as_array().into_iter().flatten() {
            k += n(&call_words(c));
        }
        if m["role"] == "tool" {
            tool_results += k;
        } else {
            messages += k;
        }
    }
    let tools = parts["tools"].as_array().map(|a| a.iter().map(|t| n(&t.to_string())).sum::<usize>()).unwrap_or(0);
    let mcp = parts["mcp_tools"].as_array().map(|a| a.iter().map(|t| n(&t.to_string())).sum::<usize>()).unwrap_or(0);
    json!({ "system": system, "messages": messages, "tool_results": tool_results, "tools": tools, "mcp": mcp, "total": system + messages + tool_results + tools + mcp })
}

/// `/api/gguf`: a GGUF file's structure for the "brain" inspector: short
/// metadata, every tensor (name, shape, type, bytes) and, when Pelica has
/// run the model, how often each expert was picked.
pub(super) fn gguf_info(path: &Path) -> Result<Value> {
    let g = crate::gguf::Gguf::open(path)?;
    let mut meta = serde_json::Map::new();
    for (k, v) in &g.meta {
        use crate::gguf::Val;
        let j = match v {
            Val::Int(i) => json!(i),
            Val::F(f) => json!(f),
            Val::Bool(b) => json!(b),
            Val::Str(s) if s.len() <= 300 => json!(s),
            Val::Str(s) => json!(format!("({} characters)", s.len())),
            Val::Arr(a) => json!(format!("({} values)", a.len())),
        };
        meta.insert(k.clone(), j);
    }
    let mut ts: Vec<_> = g.tensors.iter().collect();
    ts.sort_by_key(|t| t.1.offset);
    let tensors: Vec<Value> = ts.iter().map(|(n, t)| json!({ "name": n, "dims": t.dims, "type": t.ty.name(), "bytes": t.bytes() })).collect();
    let usage: Option<Vec<u32>> = std::fs::read(format!("{}.pelica-usage", path.display())).ok().map(|b| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect());
    Ok(json!({ "arch": g.arch(), "meta": meta, "tensors": tensors, "usage": usage, "bytes": std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) }))
}

/// `/api/tensor`: an F32 or F16 tensor's values (norm gains and the like),
/// as (min, max) per bucket, at most 512 buckets.
pub(super) fn tensor_values(path: &Path, name: &str) -> Result<Value> {
    let g = crate::gguf::Gguf::open(path)?;
    let t = g.tensor(name)?;
    let b = g.bytes(t);
    let v: Vec<f32> = match t.ty.name().as_str() {
        "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        "F16" => b.chunks_exact(2).map(|c| crate::cpu_attn::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect(),
        other => bail!("{other} tensors are quantized: only their shape is shown"),
    };
    if v.len() > 1 << 22 {
        bail!("{} values is too many to show", v.len());
    }
    let nb = v.len().min(512);
    let per = v.len().div_ceil(nb.max(1));
    let buckets: Vec<[f32; 2]> = v.chunks(per.max(1)).map(|c| [c.iter().cloned().fold(f32::INFINITY, f32::min), c.iter().cloned().fold(f32::NEG_INFINITY, f32::max)]).collect();
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64;
    let sd = (v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt();
    Ok(json!({ "n": v.len(), "buckets": buckets, "mean": mean, "sd": sd }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ds4_prompt_format() {
        let tools = vec![json!({"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"b": {"type": "integer"}, "a": {"type": "string"}}}})];
        let msgs = vec![
            json!({"role": "system", "content": "S"}),
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": "a1"}),
            json!({"role": "user", "content": "q2"}),
            json!({"role": "assistant", "content": "", "reasoning": "r", "tool_calls": [{"id": "1", "name": "f", "arguments": {"b": 2, "a": "x"}}]}),
            json!({"role": "tool", "content": "out", "tool_call_id": "1"}),
            json!({"role": "user", "content": "more"}),
        ];
        let p = ds4_prompt(&msgs, "S", &tools, true, "low");
        assert!(p.starts_with("<｜begin▁of▁sentence｜>S

## Tools

You have access"));
        // schema keys keep their order, Python-style separators
        assert!(p.contains(r#"{"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"b": {"type": "integer"}, "a": {"type": "string"}}}}"#));
        // an earlier turn without reasoning reads as a non-thinking turn
        assert!(p.contains("<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>"));
        assert!(p.contains("<｜User｜>q2<｜Assistant｜><think>r</think>

<｜DSML｜tool_calls>
<｜DSML｜invoke name=\"f\">
<｜DSML｜parameter name=\"b\" string=\"false\">2</｜DSML｜parameter>
<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls><｜end▁of▁sentence｜>"));
        // tool results and a following user message are one turn
        assert!(p.ends_with("<｜User｜><tool_result>out</tool_result>

more<｜Assistant｜><think>"));
        let q = ds4_prompt(&msgs[..2], "", &[], false, "max");
        assert_eq!(q, "<｜begin▁of▁sentence｜><｜User｜>q1<｜Assistant｜></think>");
        assert!(ds4_prompt(&msgs[..2], "", &[], true, "max").starts_with("<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum"));
    }

    /// Prints a sample agent conversation as a GGUF's own template renders
    /// it (PELICA_TMPL_MODEL=<file.gguf>; reads only the metadata).
    #[test]
    #[ignore]
    fn render_sample() {
        let Some(path) = std::env::var_os("PELICA_TMPL_MODEL") else { return };
        let g = crate::gguf::Gguf::open(std::path::Path::new(&path)).unwrap();
        let t = crate::chat_template::ChatTemplate::from_gguf(&g).unwrap();
        let tools = vec![json!({"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "the path"}}, "required": ["path"]}})];
        let msgs = vec![
            json!({"role": "system", "content": "SYS"}),
            json!({"role": "user", "content": "first question"}),
            json!({"role": "assistant", "content": "first answer", "reasoning": "old thoughts"}),
            json!({"role": "user", "content": "read a.txt"}),
            json!({"role": "assistant", "content": "", "reasoning": "need to read", "tool_calls": [{"id": "c1", "name": "read_file", "arguments": {"path": "a.txt"}}]}),
            json!({"role": "tool", "content": "hello", "name": "read_file", "tool_call_id": "c1"}),
        ];
        for think in [true, false] {
            let conv = template_conv(&msgs, true);
            let mut extra = serde_json::Map::new();
            extra.insert("enable_thinking".into(), json!(think));
            extra.insert("tools".into(), template_tools(&tools));
            let mut with_sys = conv.clone();
            with_sys.insert(0, json!({"role": "system", "content": system_of(&msgs)}));
            println!("===== thinking {think} =====
{}
=====", t.render(&with_sys, &extra).unwrap());
        }
        let tok = crate::gguf::tokenizer(&g).unwrap();
        for think in [true, false] {
            let p = ds4_prompt(&msgs, &system_of(&msgs), &tools, think, "high");
            let ids = tok.encode(p.as_str(), false).unwrap().get_ids().to_vec();
            println!("===== native thinking {think} ({} tokens) =====
{p}
=====", ids.len());
            let shown: Vec<String> = ids.iter().take(12).map(|&i| format!("{i}:{:?}", tok.id_to_token(i).unwrap_or_default())).collect();
            println!("first ids: {}", shown.join(" "));
            for sp in ["<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "<｜end▁of▁sentence｜>", "<｜DSML｜tool_calls>", "</｜DSML｜invoke>"] {
                let v = tok.encode(sp, false).unwrap().get_ids().to_vec();
                println!("  {sp} -> {v:?}");
            }
        }
    }

    #[test]
    fn calls() {
        assert_eq!(parse_call(r#"{"name": "f", "arguments": {"a": 1}}"#), ("f".into(), json!({"a": 1})));
        assert_eq!(parse_call(r#"{"name": "f", "arguments": "{\"a\": 1}"}"#), ("f".into(), json!({"a": 1})));
        assert_eq!(parse_call(r#"{"name": "f", "parameters": {"a": 1}}"#), ("f".into(), json!({"a": 1})));
        assert_eq!(parse_call("nonsense"), (String::new(), json!({})));
    }

    #[test]
    fn loose_calls() {
        let t = "Let me check.\n```json\n{\n  \"name\": \"web_search\",\n  \"arguments\": {\"query\": \"a } b\"}\n}\n```";
        let (at, n, a) = loose_call(t, &["web_search"]).unwrap();
        assert_eq!((n.as_str(), a), ("web_search", json!({"query": "a } b"})));
        assert_eq!(&t[..at], "Let me check.\n```json\n");
        assert!(loose_call(r#"{"name": "other", "arguments": {}}"#, &["web_search"]).is_none());
        assert!(loose_call(r#"{"name": "web_search", "arguments": {"q"#, &["web_search"]).is_none());
    }

    #[test]
    fn conv_words() {
        let msgs = vec![
            json!({"role": "system", "content": "be brief"}),
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "1", "name": "f", "arguments": {"a": 1}}]}),
            json!({"role": "tool", "tool_call_id": "1", "name": "f", "content": "r1"}),
            json!({"role": "tool", "tool_call_id": "2", "name": "f", "content": "r2"}),
        ];
        let c = template_conv(&msgs, false);
        assert_eq!(c.len(), 3);
        assert_eq!(c[1]["content"], "Checking.\n<tool_call>\n{\"name\":\"f\",\"arguments\":{\"a\":1}}\n</tool_call>");
        assert_eq!(c[2]["content"], "<tool_response>\nr1\n</tool_response>\n<tool_response>\nr2\n</tool_response>");
        assert_eq!(system_of(&msgs), "be brief");
        let n = template_conv(&msgs, true);
        assert_eq!(n.len(), 4);
        assert_eq!(n[1]["tool_calls"][0]["function"]["name"], "f");
    }

    #[test]
    fn harmony_tools() {
        let tools = vec![json!({"name": "read_file", "description": "Reads a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "the file"}, "max": {"type": "integer"}}, "required": ["path"]}})];
        let msgs = vec![json!({"role": "user", "content": "read a.txt"}), json!({"role": "assistant", "content": "", "reasoning": "need it", "tool_calls": [{"name": "read_file", "arguments": {"path": "a.txt"}}]}), json!({"role": "tool", "name": "read_file", "content": "hello"})];
        let h = harmony_v2("low", &msgs, &tools);
        assert!(h.contains("type read_file = (_: {\n// the file\npath: string,\nmax?: number,\n}) => any;"));
        assert!(h.contains("<|start|>assistant<|channel|>analysis<|message|>need it<|end|><|start|>assistant<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>{\"path\":\"a.txt\"}<|call|><|start|>functions.read_file to=assistant<|channel|>commentary<|message|>hello<|end|><|start|>assistant"));
    }
}
