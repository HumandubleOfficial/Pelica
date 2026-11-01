//! The terminal chat's tools: the same files, commands, web and todo tools
//! the desktop app gives the models (desktop/src/agent/tools.ts), working on
//! the folder `pelica chat` was started in. Results come back as the text
//! the model reads plus the card the desktop app draws for the call (so a
//! chat begun in the terminal looks right when opened in the app).

use serde_json::{json, Value};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::term;

/// Folders list_files and search_files don't go into (as in the app).
const SKIP: &[&str] = &["node_modules", ".git", "target", "dist", "build", "__pycache__", ".venv", "venv", ".next", ".cache", ".idea", ".vs"];
const PROJ_FILE: &str = "proj.pel";

fn obj(props: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": props, "required": required })
}

fn spec(name: &str, description: &str, parameters: Value) -> Value {
    json!({ "name": name, "description": description, "parameters": parameters })
}

/// The tools for a turn: web (if on), then files and commands (read-only
/// ones only in Plan mode).
pub fn specs(web: bool, plan: bool) -> Vec<Value> {
    let s = |d: &str| json!({ "type": "string", "description": d });
    let i = |d: &str| json!({ "type": "integer", "description": d });
    let mut v = Vec::new();
    if web {
        v.push(spec("web_search", "Search the web (DuckDuckGo); returns titles, links and snippets. Use it for anything current or outside your knowledge: news, prices, recent releases and versions, docs for a library you aren't sure of. Search before answering rather than guessing; then open_page the best results. Never for greetings or small talk.", obj(json!({ "query": s("what to search for") }), &["query"])));
        v.push(spec("open_page", "Read a web page as plain text. Use it on web_search results (snippets are too short to answer from) or a URL the user gives. Cite the pages you used.", obj(json!({ "url": s("the page's http(s) URL") }), &["url"])));
    }
    v.push(spec("list_files", "List the files and folders in the workspace folder or a subfolder (folders end with /). Use it first to get your bearings in a project.", obj(json!({ "path": s("folder, relative to the workspace (default: the workspace)"), "depth": i("levels to show (default 2)") }), &[])));
    v.push(spec("read_file", "Read a text file; each line comes prefixed with its line number and two spaces (the prefix is not part of the file). Read a file before editing it. For long files, read in parts with offset and limit.", obj(json!({ "path": s("file path, relative to the workspace"), "offset": i("first line (default 1)"), "limit": i("lines to read (default 500)") }), &["path"])));
    if !plan {
        v.push(spec("write_file", "Create a new file, or replace a file's whole content; missing folders are created. Always the complete, working content: no placeholders, no \"...\" or \"rest unchanged\". For a small change to an existing file use edit_file instead.", obj(json!({ "path": s("file path, relative to the workspace"), "content": s("the complete file content") }), &["path", "content"])));
        v.push(spec("edit_file", "Change part of an existing file: old_string is replaced by new_string. Read the file first. old_string must be copied exactly from it (same spaces, tabs and indentation, without read_file's line-number prefix) and occur only once, so include enough surrounding lines to make it unique, unless replace_all is true. On a mismatch, read the file again and retry.", obj(json!({ "path": s("file path"), "old_string": s("exact text to replace"), "new_string": s("the new text"), "replace_all": { "type": "boolean", "description": "replace every occurrence" } }), &["path", "old_string", "new_string"])));
    }
    v.push(spec("search_files", "Search the workspace's files for a regular expression (case-insensitive); returns path:line: text for each match. Use it to find where something is defined or used before changing it.", obj(json!({ "pattern": s("regular expression"), "path": s("subfolder (optional)"), "glob": s("only files matching this, e.g. *.ts (optional)") }), &["pattern"])));
    if !plan {
        v.push(spec("run_command", "Run a command in the workspace folder with Windows cmd.exe and get its output and exit code: install packages, build, run tests, run scripts, use git. Use Windows syntax (dir, type, copy, del, where; chain with &&), not Unix tools. Set background to true for anything that keeps running (dev servers, watchers), otherwise it is stopped at the timeout. Depending on the permission mode the user may approve it first.", obj(json!({ "command": s("the command line"), "description": s("a few words on what it does, shown to the user (e.g. \"Build the app\")"), "timeout_seconds": i("stop it after this long (default 120)"), "background": { "type": "boolean", "description": "leave it running and return its first output" } }), &["command"])));
        v.push(spec("delete_file", "Move a file or folder in the workspace to the Recycle Bin.", obj(json!({ "path": s("path, relative to the workspace") }), &["path"])));
    }
    v.push(spec("todo_write", "Plan and track multi-step work: send the whole list every time, each item pending, in_progress (one at a time) or completed. Use it for any task with 3 or more steps, and keep working until every item is completed.", obj(json!({ "todos": { "type": "array", "items": obj(json!({ "content": s("the step"), "status": { "type": "string", "enum": ["pending", "in_progress", "completed"] } }), &["content", "status"]) } }), &["todos"])));
    v
}

/// What the permission mode leaves to the user.
pub enum Need {
    /// a file change (asked only in Manual mode)
    File,
    /// a command (asked in Manual and Accept edits)
    Command,
}

/// A tool's result: what the model reads, the desktop card, and a short
/// line for the terminal.
pub struct Out {
    pub content: String,
    pub ui: Value,
    pub todos: Option<Value>,
}

impl Out {
    fn new(content: impl Into<String>, ui: Value) -> Out {
        Out { content: content.into(), ui, todos: None }
    }
}

/// The folder, and how to ask the user.
pub struct Ctx<'a> {
    pub root: &'a Path,
    /// asks about an action: (what, detail, kind) -> allowed
    pub ask: &'a mut dyn FnMut(&str, &str, Need) -> bool,
}

/// Runs one call. Errors come back as "Error: …" for the model.
pub fn run(cx: &mut Ctx, name: &str, a: &Value) -> Out {
    let p = a["path"].as_str().unwrap_or("").to_string();
    match run_inner(cx, name, a, &p) {
        Ok(o) => o,
        Err(e) => Out::new(format!("Error: {e}"), json!({ "kind": "error", "title": name, "error": e, "path": if p.is_empty() { Value::Null } else { json!(p) } })),
    }
}

fn run_inner(cx: &mut Ctx, name: &str, a: &Value, p: &str) -> Result<Out, String> {
    let root = cx.root;
    let int = |k: &str| a[k].as_u64().or_else(|| a[k].as_str().and_then(|s| s.parse().ok()));
    match name {
        "web_search" => {
            let q = a["query"].as_str().unwrap_or("").trim().to_string();
            if q.is_empty() {
                return Err("web_search needs a \"query\"".into());
            }
            let hits = crate::web::search(&q).map_err(|e| format!("{e:#}"))?;
            let text = if hits.is_empty() { "No results.".to_string() } else { hits.iter().enumerate().map(|(i, h)| format!("{}. {}\n   {}\n   {}", i + 1, h.title, h.url, h.snippet)).collect::<Vec<_>>().join("\n") };
            let results: Vec<Value> = hits.iter().map(|h| json!({ "title": h.title, "url": h.url })).collect();
            Ok(Out::new(text, json!({ "kind": "search", "title": q, "results": results })))
        }
        "open_page" => {
            let url = a["url"].as_str().unwrap_or("").to_string();
            let page = crate::web::open(&url, 14000).map_err(|e| format!("{e:#}"))?;
            let title = if page.title.is_empty() { page.url.clone() } else { page.title.clone() };
            Ok(Out::new(format!("Title: {}\nURL: {}\n\n{}", page.title, page.url, page.text), json!({ "kind": "page", "title": title, "url": page.url })))
        }
        "list_files" => {
            let depth = int("depth").unwrap_or(2).clamp(1, 5) as usize;
            let t = list_files(root, if p.is_empty() { "." } else { p }, depth)?;
            Ok(Out::new(t.clone(), json!({ "kind": "list", "title": if p.is_empty() { "." } else { p }, "output": t })))
        }
        "read_file" => {
            let (text, lines, rel) = read_file(root, p, int("offset").unwrap_or(1) as usize, int("limit").unwrap_or(500).min(2000) as usize)?;
            Ok(Out::new(text, json!({ "kind": "read", "title": rel, "path": rel, "detail": format!("{lines} lines") })))
        }
        "write_file" => {
            let content = a["content"].as_str().unwrap_or("").to_string();
            if !(cx.ask)(&format!("Write {p}"), &format!("{} lines", content.lines().count()), Need::File) {
                return Ok(denied(&format!("Write {p}")));
            }
            let f = inside(root, p)?;
            let before = std::fs::read_to_string(&f).ok();
            if let Some(d) = f.parent() {
                std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
            }
            std::fs::write(&f, &content).map_err(|e| e.to_string())?;
            let rel = rel_of(root, &f);
            let n = content.split('\n').count();
            Ok(Out::new(format!("{} {rel} ({n} lines).", if before.is_none() { "Created" } else { "Wrote" }), json!({ "kind": "write", "title": rel, "path": rel, "before": before, "after": content })))
        }
        "edit_file" => {
            let (o, n) = (a["old_string"].as_str().unwrap_or(""), a["new_string"].as_str().unwrap_or(""));
            let detail = format!("Replace:\n{}\n\nwith:\n{}", clip_start(o, 600), clip_start(n, 600));
            if !(cx.ask)(&format!("Edit {p}"), &detail, Need::File) {
                return Ok(denied(&format!("Edit {p}")));
            }
            let f = inside(root, p)?;
            let before = std::fs::read_to_string(&f).map_err(|e| format!("{p}: {e}"))?;
            // the model may write \n where the file has \r\n
            let crlf = before.contains("\r\n") && !o.contains("\r\n");
            let (o, n) = if crlf { (o.replace('\n', "\r\n"), n.replace('\n', "\r\n")) } else { (o.to_string(), n.to_string()) };
            if o.is_empty() {
                return Err("old_string is empty; use write_file to create a file".into());
            }
            let count = before.matches(&o).count();
            let all = a["replace_all"].as_bool().unwrap_or(false);
            if count == 0 {
                return Err(format!("old_string was not found in {p}. Read the file again and copy the text exactly, including indentation."));
            }
            if count > 1 && !all {
                return Err(format!("old_string appears {count} times in {p}; add more surrounding lines to make it unique, or set replace_all"));
            }
            let after = if all { before.replace(&o, &n) } else { before.replacen(&o, &n, 1) };
            std::fs::write(&f, &after).map_err(|e| e.to_string())?;
            let rel = rel_of(root, &f);
            let k = if all { count } else { 1 };
            Ok(Out::new(format!("Edited {rel} ({k} replacement{}).", if k > 1 { "s" } else { "" }), json!({ "kind": "edit", "title": rel, "path": rel, "before": before, "after": after })))
        }
        "search_files" => {
            let pat = a["pattern"].as_str().unwrap_or("").to_string();
            let t = search_files(root, &pat, if p.is_empty() { "." } else { p }, a["glob"].as_str())?;
            Ok(Out::new(t.clone(), json!({ "kind": "grep", "title": pat, "output": t })))
        }
        "delete_file" => {
            if p.is_empty() {
                return Err("delete_file needs a \"path\"".into());
            }
            if !(cx.ask)(&format!("Move {p} to the Recycle Bin"), "It can be restored from the Recycle Bin.", Need::File) {
                return Ok(denied(&format!("Move {p} to the Recycle Bin")));
            }
            let f = inside(root, p)?;
            if !f.exists() {
                return Err(format!("{p} doesn't exist"));
            }
            if rel_of(root, &f) == "." {
                return Err("that is the workspace folder itself".into());
            }
            recycle(&f)?;
            Ok(Out::new(format!("Moved {p} to the Recycle Bin."), json!({ "kind": "trash", "title": p, "path": p })))
        }
        "todo_write" => {
            let todos: Vec<Value> = a["todos"]
                .as_array()
                .map(|v| {
                    v.iter()
                        .filter_map(|t| {
                            let c = t["content"].as_str().or(t["task"].as_str()).unwrap_or("").to_string();
                            let s = t["status"].as_str().filter(|s| ["pending", "in_progress", "completed"].contains(s)).unwrap_or("pending");
                            (!c.is_empty()).then(|| json!({ "content": c, "status": s }))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let left = todos.iter().filter(|t| t["status"] != "completed").count();
            let content = if left > 0 { format!("Todo list updated: {left} of {} left. Continue with the next item.", todos.len()) } else { "All items completed.".to_string() };
            let mut o = Out::new(content, json!({ "kind": "todo", "title": format!("{}/{} done", todos.len() - left, todos.len()), "todos": todos }));
            o.todos = Some(json!(todos));
            Ok(o)
        }
        "run_command" => {
            let command = a["command"].as_str().unwrap_or("").trim().to_string();
            if command.is_empty() {
                return Err("run_command needs a \"command\"".into());
            }
            let about: String = a["description"].as_str().unwrap_or("").trim().chars().take(120).collect();
            if !(cx.ask)(&command, &about, Need::Command) {
                return Ok(Out::new("The user declined to run this command. Ask them, or try another way.", json!({ "kind": "denied", "title": command, "detail": "Declined" })));
            }
            let timeout = int("timeout_seconds").unwrap_or(120).clamp(1, 1800);
            let r = run_command(root, &command, Duration::from_secs(timeout), a["background"].as_bool().unwrap_or(false));
            let status = if r.background { "still running in the background".to_string() } else if r.stopped { "stopped by the user".to_string() } else if r.timed_out { "timed out".to_string() } else { format!("exit code {}", r.code.map_or("?".into(), |c| c.to_string())) };
            let out = r.output.trim().to_string();
            let content = format!("{status}\n{}", if out.is_empty() { "(no output)".to_string() } else { clip_mid(&out, 16000) });
            let tail: String = { let n = out.chars().count(); out.chars().skip(n.saturating_sub(20000)).collect() };
            Ok(Out::new(content, json!({ "kind": "command", "title": command, "about": if about.is_empty() { Value::Null } else { json!(about) }, "output": tail, "code": r.code, "detail": status, "running": false })))
        }
        _ => Err(format!("there is no tool called \"{name}\"")),
    }
}

fn denied(what: &str) -> Out {
    Out::new(format!("The user declined: {what}. Ask them how to proceed."), json!({ "kind": "denied", "title": what, "detail": "Declined" }))
}

fn clip_start(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The start and end of a long text, with what was cut in between.
pub fn clip_mid(s: &str, n: usize) -> String {
    let c: Vec<char> = s.chars().collect();
    if c.len() <= n {
        return s.to_string();
    }
    let h = n / 2;
    format!("{}\n…[{} characters cut]…\n{}", c[..h].iter().collect::<String>(), c.len() - n, c[c.len() - h..].iter().collect::<String>())
}

// ---------- paths ----------

/// `rel` under `root` (absolute paths too), refusing anything outside it.
pub fn inside(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim();
    let r = Path::new(if rel.is_empty() { "." } else { rel });
    let joined = if r.is_absolute() { r.to_path_buf() } else { root.join(r) };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c.as_os_str()),
        }
    }
    let a = out.to_string_lossy().to_lowercase();
    let b = root.to_string_lossy().trim_end_matches(['\\', '/']).to_lowercase();
    if a == b || a.starts_with(&format!("{b}\\")) || a.starts_with(&format!("{b}/")) {
        Ok(out)
    } else {
        Err(format!("{rel} is outside the workspace folder ({})", root.display()))
    }
}

/// `p` relative to `root`, with forward slashes ("." for the root itself).
pub fn rel_of(root: &Path, p: &Path) -> String {
    let r = p.strip_prefix(root).map(|x| x.to_string_lossy().replace('\\', "/")).unwrap_or_else(|_| p.to_string_lossy().into_owned());
    if r.is_empty() {
        ".".into()
    } else {
        r
    }
}

fn is_binary(b: &[u8]) -> bool {
    b[..b.len().min(8000)].contains(&0)
}

fn list_files(root: &Path, rel: &str, depth: usize) -> Result<String, String> {
    let base = inside(root, rel)?;
    if !base.is_dir() {
        return Err(format!("{rel} is not a folder"));
    }
    let mut out = Vec::new();
    fn walk(d: &Path, lvl: usize, depth: usize, pre: &str, out: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        let mut ents: Vec<(bool, String, u64)> = rd
            .flatten()
            .map(|e| {
                let md = e.metadata().ok();
                (md.as_ref().is_some_and(|m| m.is_dir()), e.file_name().to_string_lossy().into_owned(), md.map_or(0, |m| m.len()))
            })
            .collect();
        ents.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.to_lowercase().cmp(&b.1.to_lowercase())));
        for (dir, name, size) in ents {
            if out.len() >= 500 {
                return;
            }
            if dir {
                if SKIP.contains(&name.as_str()) {
                    out.push(format!("{pre}{name}/ (skipped)"));
                    continue;
                }
                out.push(format!("{pre}{name}/"));
                if lvl < depth {
                    walk(&d.join(&name), lvl + 1, depth, &format!("{pre}  "), out);
                }
            } else if name != PROJ_FILE {
                out.push(format!("{pre}{name} ({})", if size < 1024 { format!("{size} B") } else { format!("{:.1} KB", size as f64 / 1024.0) }));
            }
        }
    }
    walk(&base, 1, depth, "", &mut out);
    if out.len() >= 500 {
        out.push("... (more entries not shown)".into());
    }
    Ok(format!("{}/\n{}", rel_of(root, &base), if out.is_empty() { "(empty)".to_string() } else { out.join("\n") }))
}

fn read_file(root: &Path, rel: &str, offset: usize, limit: usize) -> Result<(String, usize, String), String> {
    if rel.trim().is_empty() {
        return Err("read_file needs a \"path\"".into());
    }
    let p = inside(root, rel)?;
    let md = std::fs::metadata(&p).map_err(|e| format!("{rel}: {e}"))?;
    if md.is_dir() {
        return Err(format!("{rel} is a folder; use list_files"));
    }
    if md.len() > 5_000_000 {
        return Err(format!("{rel} is {:.1} MB, too large to read", md.len() as f64 / 1e6));
    }
    let b = std::fs::read(&p).map_err(|e| e.to_string())?;
    if is_binary(&b) {
        return Err(format!("{rel} is a binary file"));
    }
    let s = String::from_utf8_lossy(&b);
    let all: Vec<&str> = s.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l)).collect();
    let start = offset.max(1);
    let end = (start - 1 + limit.max(1)).min(all.len());
    let mut text: Vec<String> = Vec::new();
    for (i, l) in all.iter().enumerate().take(end).skip(start - 1) {
        let l: String = if l.chars().count() > 2000 { l.chars().take(2000).collect::<String>() + "…" } else { l.to_string() };
        text.push(format!("{:>5}  {l}", i + 1));
    }
    let more = if end < all.len() { format!("\n[lines {start}-{end} of {}; read more with offset]", all.len()) } else { String::new() };
    Ok((text.join("\n") + &more, all.len(), rel_of(root, &p)))
}

/// A glob (`*.ts`, `src/**/*.rs`) as a case-insensitive regex over paths.
fn glob_re(g: &str) -> Option<regex::Regex> {
    let mut s = String::new();
    let mut it = g.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '*' if it.peek() == Some(&'*') => {
                it.next();
                if it.peek() == Some(&'/') {
                    it.next();
                }
                s.push_str("(?:.*/)?");
            }
            '*' => s.push_str("[^/]*"),
            '?' => s.push_str("[^/]"),
            c => s.push_str(&regex::escape(&c.to_string())),
        }
    }
    regex::RegexBuilder::new(&format!("(^|/){s}$")).case_insensitive(true).build().ok()
}

fn search_files(root: &Path, pattern: &str, rel: &str, glob: Option<&str>) -> Result<String, String> {
    if pattern.is_empty() {
        return Err("search_files needs a \"pattern\"".into());
    }
    let base = inside(root, rel)?;
    let re = regex::RegexBuilder::new(pattern).case_insensitive(true).build().or_else(|_| regex::RegexBuilder::new(&regex::escape(pattern)).case_insensitive(true).build()).map_err(|e| e.to_string())?;
    let gre = glob.filter(|g| !g.is_empty()).and_then(glob_re);
    let mut out: Vec<String> = Vec::new();
    let mut files = 0usize;
    let mut stack = vec![base];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut ents: Vec<_> = rd.flatten().collect();
        ents.sort_by_key(|e| e.file_name());
        for e in ents.into_iter().rev() {
            if out.len() >= 200 || files > 5000 {
                break;
            }
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            let Ok(md) = e.metadata() else { continue };
            if md.is_dir() {
                if !SKIP.contains(&name.as_str()) {
                    stack.push(p);
                }
                continue;
            }
            if name == PROJ_FILE || md.len() > 2_000_000 {
                continue;
            }
            let r = rel_of(root, &p);
            if gre.as_ref().is_some_and(|g| !g.is_match(&r)) {
                continue;
            }
            files += 1;
            let Ok(b) = std::fs::read(&p) else { continue };
            if is_binary(&b) {
                continue;
            }
            for (i, l) in String::from_utf8_lossy(&b).lines().enumerate() {
                if out.len() >= 200 {
                    break;
                }
                if re.is_match(l) {
                    out.push(format!("{r}:{}: {}", i + 1, l.trim().chars().take(240).collect::<String>()));
                }
            }
        }
    }
    Ok(if out.is_empty() { "No matches.".into() } else { out.join("\n") + if out.len() >= 200 { "\n[first 200 matches]" } else { "" } })
}

// ---------- the Recycle Bin ----------

#[cfg(windows)]
fn recycle(p: &Path) -> Result<(), String> {
    #[repr(C)]
    struct Op {
        hwnd: isize,
        func: u32,
        from: *const u16,
        to: *const u16,
        flags: u16,
        aborted: i32,
        mappings: *mut std::ffi::c_void,
        title: *const u16,
    }
    #[link(name = "shell32")]
    extern "system" {
        fn SHFileOperationW(op: *mut Op) -> i32;
    }
    // (a list of paths, each NUL-terminated, ending with another NUL)
    let from: Vec<u16> = p.as_os_str().to_string_lossy().encode_utf16().chain([0, 0]).collect();
    // FO_DELETE with FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    let mut op = Op { hwnd: 0, func: 3, from: from.as_ptr(), to: std::ptr::null(), flags: 0x40 | 0x10 | 0x4 | 0x400, aborted: 0, mappings: std::ptr::null_mut(), title: std::ptr::null() };
    // SAFETY: `from` outlives the call; the struct matches SHFILEOPSTRUCTW on 64-bit Windows
    let r = unsafe { SHFileOperationW(&mut op) };
    if r != 0 || op.aborted != 0 {
        return Err(format!("couldn't move it to the Recycle Bin (error {r})"));
    }
    Ok(())
}

#[cfg(not(windows))]
fn recycle(_p: &Path) -> Result<(), String> {
    Err("the Recycle Bin is only available on Windows".into())
}

// ---------- commands ----------

/// Commands left running in the background, stopped when the chat ends.
static BACKGROUND: Mutex<Vec<Child>> = Mutex::new(Vec::new());

pub struct Ran {
    pub output: String,
    pub code: Option<i32>,
    pub timed_out: bool,
    pub stopped: bool,
    pub background: bool,
}

fn kill_tree(pid: u32) {
    let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null()).status();
}

/// Stops what run_command left running (at exit).
pub fn stop_background() {
    for mut c in BACKGROUND.lock().unwrap().drain(..) {
        if c.try_wait().ok().flatten().is_none() {
            kill_tree(c.id());
        }
    }
}

/// Runs `command` with cmd.exe in `root`, showing its newest line as it goes.
/// Ctrl+C stops it (and everything it started).
pub fn run_command(root: &Path, command: &str, timeout: Duration, background: bool) -> Ran {
    let mut cmd = Command::new("cmd.exe");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.raw_arg(format!("/D /S /C \"{command}\""));
        // a background command gets its own group, so Ctrl+C in the chat doesn't stop it
        if background {
            cmd.creation_flags(0x0000_0200);
        }
    }
    #[cfg(not(windows))]
    cmd.args(["/C", command]);
    let child = cmd.current_dir(root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => return Ran { output: format!("couldn't start cmd.exe: {e}"), code: None, timed_out: false, stopped: false, background: false },
    };
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    for mut pipe in [child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>), child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>)].into_iter().flatten() {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Ok(n) = pipe.read(&mut buf) {
                if n == 0 || tx.send(buf[..n].to_vec()).is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);
    let t0 = Instant::now();
    let mut out: Vec<u8> = Vec::new();
    let first = Duration::from_secs(4);
    let (mut timed_out, mut stopped) = (false, false);
    loop {
        match rx.recv_timeout(Duration::from_millis(120)) {
            Ok(b) => out.extend_from_slice(&b),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if child.try_wait().ok().flatten().is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(60));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        if out.len() > 4_000_000 {
            let cut = out.len() - 2_000_000;
            out.drain(..cut);
        }
        let last = String::from_utf8_lossy(&out).lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
        term::status(&format!("  Running… {}s  {}", t0.elapsed().as_secs(), last));
        if let Ok(Some(_)) = child.try_wait() {
            // (what's still in the pipes)
            while let Ok(b) = rx.recv_timeout(Duration::from_millis(150)) {
                out.extend_from_slice(&b);
            }
            break;
        }
        if background && t0.elapsed() >= first {
            term::clear_status();
            // the pipes keep draining in their threads, so it never blocks on them
            std::thread::spawn(move || while rx.recv().is_ok() {});
            BACKGROUND.lock().unwrap().push(child);
            return Ran { output: String::from_utf8_lossy(&out).into_owned(), code: None, timed_out: false, stopped: false, background: true };
        }
        // (left set, so the whole reply stops too)
        if term::INTERRUPT.load(Ordering::SeqCst) {
            stopped = true;
            kill_tree(child.id());
        } else if t0.elapsed() > timeout {
            timed_out = true;
            kill_tree(child.id());
        }
        if stopped || timed_out {
            let _ = child.wait();
            while let Ok(b) = rx.recv_timeout(Duration::from_millis(150)) {
                out.extend_from_slice(&b);
            }
            break;
        }
    }
    term::clear_status();
    let code = child.wait().ok().and_then(|s| s.code());
    Ran { output: String::from_utf8_lossy(&out).replace("\r\n", "\n"), code, timed_out, stopped, background: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_stay_inside() {
        let root = Path::new(r"C:\work\proj");
        assert_eq!(inside(root, "src/a.rs").unwrap(), PathBuf::from(r"C:\work\proj\src\a.rs"));
        assert_eq!(inside(root, "").unwrap(), PathBuf::from(r"C:\work\proj"));
        assert!(inside(root, "../other/x").is_err());
        assert!(inside(root, r"C:\work\projx\a").is_err());
        assert!(inside(root, r"C:\Windows\system32").is_err());
        assert_eq!(inside(root, r"C:\WORK\proj\a\..\b").unwrap(), PathBuf::from(r"C:\WORK\proj\b"));
    }

    #[test]
    fn globs() {
        let g = glob_re("*.ts").unwrap();
        assert!(g.is_match("src/app.ts") && g.is_match("a.TS") && !g.is_match("a.tsx"));
        let g = glob_re("src/**/*.rs").unwrap();
        assert!(g.is_match("src/a.rs") && g.is_match("src/x/y/a.rs") && !g.is_match("lib/a.rs"));
    }

    #[test]
    fn files_round_trip() {
        let dir = std::env::temp_dir().join(format!("pelica-tools-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut ask = |_: &str, _: &str, _: Need| true;
        let mut cx = Ctx { root: &dir, ask: &mut ask };
        let w = run(&mut cx, "write_file", &json!({ "path": "sub/a.txt", "content": "one\ntwo\nthree\n" }));
        assert!(w.content.starts_with("Created sub/a.txt"), "{}", w.content);
        let r = run(&mut cx, "read_file", &json!({ "path": "sub/a.txt" }));
        assert!(r.content.contains("    2  two"), "{}", r.content);
        let e = run(&mut cx, "edit_file", &json!({ "path": "sub/a.txt", "old_string": "two", "new_string": "TWO" }));
        assert!(e.content.starts_with("Edited"), "{}", e.content);
        let bad = run(&mut cx, "edit_file", &json!({ "path": "sub/a.txt", "old_string": "nope", "new_string": "x" }));
        assert!(bad.content.starts_with("Error: old_string was not found"), "{}", bad.content);
        let s = run(&mut cx, "search_files", &json!({ "pattern": "t[wh]", "glob": "*.txt" }));
        assert!(s.content.contains("sub/a.txt:2: TWO") && s.content.contains("sub/a.txt:3: three"), "{}", s.content);
        let l = run(&mut cx, "list_files", &json!({}));
        assert!(l.content.contains("sub/") && l.content.contains("a.txt"), "{}", l.content);
        let out = run(&mut cx, "read_file", &json!({ "path": "../x.txt" }));
        assert!(out.content.contains("outside the workspace"), "{}", out.content);
        let c = run(&mut cx, "run_command", &json!({ "command": "echo hello&& exit /b 3" }));
        assert!(c.content.starts_with("exit code 3") && c.content.contains("hello"), "{}", c.content);
        let _ = std::fs::remove_file(dir.join("sub").join("a.txt"));
        let _ = std::fs::remove_dir(dir.join("sub"));
        let _ = std::fs::remove_dir(&dir);
    }
}
