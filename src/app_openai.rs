//! OpenAI-compatible endpoints (`GET /v1/models`, `POST /v1/chat/completions`,
//! streamed or not) over chat API v2, so other programs on this computer can
//! use the loaded model. Pelica Desktop serves them on a fixed port too
//! (Settings → Local API).

use super::*;

pub(super) fn models(app: &App) -> Value {
    let st = app.status.lock().unwrap();
    let data: Vec<Value> = if st["state"] == "ready" { vec![json!({ "id": st["model"], "object": "model", "created": now_s(), "owned_by": "pelica" })] } else { Vec::new() };
    json!({ "object": "list", "data": data })
}

fn err(msg: &str, kind: &str) -> Value {
    json!({ "error": { "message": msg, "type": kind } })
}

/// A message's text: a string, or the text parts of a list.
fn text(c: &Value) -> String {
    match c {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().filter(|p| p["type"] == "text" || p["type"].is_null()).filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// An OpenAI message as a v2 one (tool calls with their arguments parsed).
fn msg(m: &Value) -> Value {
    let mut x = json!({ "role": m["role"], "content": text(&m["content"]) });
    if let Some(calls) = m["tool_calls"].as_array() {
        let calls: Vec<Value> = calls
            .iter()
            .map(|c| {
                let f = &c["function"];
                let args = match &f["arguments"] {
                    Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                    v => v.clone(),
                };
                json!({ "id": c["id"], "name": f["name"], "arguments": args })
            })
            .collect();
        x["tool_calls"] = json!(calls);
    }
    for k in ["tool_call_id", "name"] {
        if !m[k].is_null() {
            x[k] = m[k].clone();
        }
    }
    if let Some(r) = m["reasoning_content"].as_str() {
        x["reasoning"] = json!(r);
    }
    x
}

/// An OpenAI chat request as a v2 one.
pub(super) fn to_v2(body: &Value) -> Value {
    let msgs: Vec<Value> = body["messages"].as_array().map(|a| a.iter().map(msg).collect()).unwrap_or_default();
    let mut req = json!({ "v": 2, "messages": msgs });
    if body["tools"].is_array() && body["tool_choice"] != "none" {
        req["tools"] = body["tools"].clone();
    }
    for k in ["temperature", "top_p", "seed"] {
        if !body[k].is_null() {
            req[k] = body[k].clone();
        }
    }
    if let Some(n) = body["max_completion_tokens"].as_u64().or(body["max_tokens"].as_u64()) {
        req["max_tokens"] = json!(n);
    }
    if let Some(e) = body["reasoning_effort"].as_str() {
        req["effort"] = json!(e);
    }
    req
}

pub(super) fn chat(app: &Arc<App>, s: &mut TcpStream, body: &Value) -> Result<()> {
    if app.status.lock().unwrap()["state"] != "ready" {
        return respond_json(s, 503, &err("No model is loaded in Pelica.", "unavailable"));
    }
    if !body["messages"].is_array() {
        return respond_json(s, 400, &err("messages is required", "invalid_request_error"));
    }
    if app.busy.swap(true, Ordering::Relaxed) {
        return respond_json(s, 429, &err("Pelica is answering something else; try again when it's done.", "busy"));
    }
    let (tx, rx) = channel();
    if app.jobs.lock().unwrap().send(Job::Chat { req: to_v2(body), tx }).is_err() {
        app.busy.store(false, Ordering::Relaxed);
        return respond_json(s, 503, &err("the engine stopped", "unavailable"));
    }
    let model = app.status.lock().unwrap()["model"].clone();
    let created = now_s();
    let id = format!("chatcmpl-{:x}{:x}", created, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos()));
    let stream = body["stream"].as_bool() == Some(true);
    let chunk = |delta: Value, finish: Value| json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] });
    let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::<Value>::new());
    let (mut finish, mut usage, mut error) = ("stop", Value::Null, None::<String>);
    let mut gone = false;
    if stream {
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n")?;
        gone = write!(s, "data: {}\n\n", chunk(json!({ "role": "assistant", "content": "" }), Value::Null)).and_then(|_| s.flush()).is_err();
    }
    for ev in rx {
        let piece = match ev["t"].as_str() {
            Some("text") => {
                let d = ev["d"].as_str().unwrap_or("");
                content += d;
                Some(json!({ "content": d }))
            }
            Some("think") => {
                let d = ev["d"].as_str().unwrap_or("");
                reasoning += d;
                Some(json!({ "reasoning_content": d }))
            }
            Some("tool_call") => {
                let args = if ev["arguments"].is_null() { ev["raw"].as_str().unwrap_or("{}").to_string() } else { ev["arguments"].to_string() };
                let c = json!({ "id": ev["id"], "type": "function", "function": { "name": ev["name"], "arguments": args } });
                let mut d = c.clone();
                d["index"] = json!(calls.len());
                calls.push(c);
                Some(json!({ "tool_calls": [d] }))
            }
            Some("done") => {
                finish = match ev["finish"].as_str() {
                    Some("tool_call") => "tool_calls",
                    Some("length") => "length",
                    _ => "stop",
                };
                let (p, g) = (ev["prompt"].as_u64().unwrap_or(0), ev["generated"].as_u64().unwrap_or(0));
                usage = json!({ "prompt_tokens": p, "completion_tokens": g, "total_tokens": p + g });
                None
            }
            Some("error") => {
                error = Some(ev["error"].as_str().unwrap_or("error").to_string());
                None
            }
            _ => None,
        };
        if let (true, false, Some(d)) = (stream, gone, piece) {
            if write!(s, "data: {}\n\n", chunk(d, Value::Null)).and_then(|_| s.flush()).is_err() {
                // the caller went away: stop, and let the engine finish
                gone = true;
                app.stop.store(true, Ordering::Relaxed);
            }
        }
    }
    if stream {
        if gone {
            return Ok(());
        }
        if let Some(e) = error {
            write!(s, "data: {}\n\n", err(&e, "server_error"))?;
        } else {
            write!(s, "data: {}\n\n", chunk(json!({}), json!(finish)))?;
            if body["stream_options"]["include_usage"].as_bool() == Some(true) {
                write!(s, "data: {}\n\n", json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage }))?;
            }
        }
        write!(s, "data: [DONE]\n\n")?;
        return Ok(s.flush()?);
    }
    if let Some(e) = error {
        return respond_json(s, 500, &err(&e, "server_error"));
    }
    let mut message = json!({ "role": "assistant", "content": content });
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        if content.is_empty() {
            message["content"] = Value::Null;
        }
        message["tool_calls"] = json!(calls);
    }
    respond_json(s, 200, &json!({ "id": id, "object": "chat.completion", "created": created, "model": model, "choices": [{ "index": 0, "message": message, "finish_reason": finish }], "usage": usage }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn to_v2() {
        let body = json!({
            "model": "x", "max_tokens": 50, "temperature": 0.5,
            "messages": [
                { "role": "system", "content": "be brief" },
                { "role": "user", "content": [{ "type": "text", "text": "hi" }] },
                { "role": "assistant", "content": null, "tool_calls": [{ "id": "c1", "type": "function", "function": { "name": "f", "arguments": "{\"a\":1}" } }] },
                { "role": "tool", "tool_call_id": "c1", "content": "ok" },
            ],
        });
        let r = super::to_v2(&body);
        assert_eq!(r["v"], 2);
        assert_eq!(r["max_tokens"], 50);
        assert_eq!(r["messages"][1]["content"], "hi");
        assert_eq!(r["messages"][2]["tool_calls"][0]["arguments"]["a"], 1);
        assert_eq!(r["messages"][3]["tool_call_id"], "c1");
    }
}
