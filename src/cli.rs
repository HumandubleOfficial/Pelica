//! `pelica` (no arguments) or `pelica chat`: Pelica in the terminal. A chat
//! that works in the folder it was started in, with the same tools, system
//! prompt and permission modes as the desktop app (desktop/src/agent). It
//! talks to the engine over the chat API v2: the desktop app's engine when
//! the app is open (so the model is loaded once), else an engine started in
//! this process. Chats are saved with the app's, so they show up there too.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::cli_tools::{self, Ctx, Need};
use crate::term::{self, blue, bold, cyan, dim, green, orange, red, yellow};

const HELP: &str = "Pelica in the terminal: a chat that works in this folder.

usage: pelica [chat] [options]

  -m, --model <name|path>   the model to load (a name from /model, or a .gguf path)
      --ctx <tokens>        context size when loading (default: the app's last one)
      --mode <mode>         manual | edits | plan | auto (default: the app's setting)
      --effort <level>      thinking effort, e.g. off | low | high | max
      --web / --no-web      web search on or off
  -c, --continue            carry on the latest chat in this folder
  -p, --print <message>     do one task, print the answer and exit
      --show-thinking       print the model's thinking as it goes
  -h, --help                this help

In the chat, type / for commands (/help lists them). Ctrl+C stops a reply;
at the prompt, press it twice to quit. End a line with \\ to write more lines.
";

const MODES: [(&str, &str, &str); 4] = [
    ("ask", "Manual", "every file change and command asks first"),
    ("edits", "Accept edits", "file changes go ahead, commands ask"),
    ("plan", "Plan", "reading only: Pelica looks and plans, changes nothing"),
    ("auto", "Bypass permissions", "nothing asks"),
];

fn mode_name(m: &str) -> &'static str {
    MODES.iter().find(|x| x.0 == m).map_or("Accept edits", |x| x.1)
}

fn mode_of(s: &str) -> Option<&'static str> {
    let s = s.trim().to_lowercase();
    let s = match s.as_str() {
        "manual" | "ask" => "ask",
        "edits" | "accept" | "accept-edits" | "acceptedits" => "edits",
        "plan" => "plan",
        "auto" | "bypass" | "yolo" => "auto",
        _ => return None,
    };
    MODES.iter().find(|x| x.0 == s).map(|x| x.0)
}

// ---------- the desktop app's files ----------

/// The app's data folder (settings, chats).
fn user_dir() -> PathBuf {
    PathBuf::from(std::env::var("APPDATA").unwrap_or_else(|_| ".".into())).join("Pelica")
}

fn read_json(p: &Path) -> Value {
    std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

fn uid() -> String {
    // like the app's: the time, then six random characters
    use std::hash::{BuildHasher, Hasher};
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(N.fetch_add(1, Ordering::Relaxed));
    let r = radix36(h.finish());
    format!("{}{}", radix36(now_ms()), &r[r.len().saturating_sub(6)..])
}

fn radix36(mut n: u64) -> String {
    let d = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut s = Vec::new();
    loop {
        s.push(d[(n % 36) as usize]);
        n /= 36;
        if n == 0 {
            break;
        }
    }
    s.reverse();
    String::from_utf8(s).unwrap()
}

// ---------- HTTP to the engine ----------

struct Engine {
    port: u16,
    /// the desktop app's engine (not ours)
    shared: bool,
    /// where our engine's log lines go
    log: Option<PathBuf>,
}

fn connect(port: u16) -> Result<TcpStream> {
    let s = TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(2)).with_context(|| format!("connecting to the engine on port {port}"))?;
    Ok(s)
}

fn send_req(s: &mut TcpStream, port: u16, method: &str, path: &str, body: Option<&Value>) -> Result<()> {
    let b = body.map(|v| v.to_string()).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len())?;
    Ok(())
}

/// (status code, body) after the headers.
fn read_head(r: &mut BufReader<TcpStream>) -> Result<(u16, Option<usize>)> {
    let mut line = String::new();
    r.read_line(&mut line)?;
    let code = line.split_whitespace().nth(1).and_then(|c| c.parse().ok()).context("no answer from the engine")?;
    let mut len = None;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().ok();
            }
        }
    }
    Ok((code, len))
}

impl Engine {
    fn call(&self, method: &str, path: &str, body: Option<&Value>) -> Result<(u16, Value)> {
        let mut s = connect(self.port)?;
        s.set_read_timeout(Some(Duration::from_secs(30)))?;
        send_req(&mut s, self.port, method, path, body)?;
        let mut r = BufReader::new(s);
        let (code, len) = read_head(&mut r)?;
        let mut b = Vec::new();
        match len {
            Some(n) => {
                b.resize(n, 0);
                r.read_exact(&mut b)?;
            }
            None => {
                r.read_to_end(&mut b)?;
            }
        }
        Ok((code, serde_json::from_slice(&b).unwrap_or(Value::Null)))
    }
    fn get(&self, path: &str) -> Result<Value> {
        Ok(self.call("GET", path, None)?.1)
    }
    fn status(&self) -> Value {
        self.get("/api/status").unwrap_or(Value::Null)
    }
}

/// The desktop app's engine if it's running, else one in this process.
fn start_engine() -> Result<Engine> {
    let info = read_json(&user_dir().join("engine.json"));
    if let Some(port) = info["port"].as_u64().map(|p| p as u16) {
        let e = Engine { port, shared: true, log: None };
        if e.status().is_object() {
            return Ok(e);
        }
    }
    let log = term::quiet_stderr();
    let mut dirs = crate::app::default_model_dirs();
    let st = read_json(&user_dir().join("settings.json"));
    for d in st["modelDirs"].as_array().into_iter().flatten().chain(std::iter::once(&st["downloadDir"])) {
        if let Some(d) = d.as_str().filter(|d| !d.is_empty()).map(PathBuf::from) {
            if d.is_dir() && !dirs.iter().any(|x| x.to_string_lossy().eq_ignore_ascii_case(&d.to_string_lossy())) {
                dirs.push(d);
            }
        }
    }
    let port = crate::app::serve_background(dirs)?;
    Ok(Engine { port, shared: false, log })
}

// ---------- streaming Markdown ----------

/// Writes Markdown to the terminal as it streams: **bold**, `code`,
/// headings and fenced code blocks styled as they arrive. A line's first
/// characters wait until it's clear whether they open a heading or a fence.
#[derive(Default)]
struct Md {
    held: String,
    mid: bool,
    fence: bool,
    bold: bool,
    code: bool,
    star: bool,
    heading: bool,
    out: String,
    /// a line ended: the next one starts under the reply dot
    indent: bool,
}

impl Md {
    /// Prints what is ready, each line after the first indented to line up
    /// with the text after the reply dot.
    fn emit(&mut self) {
        let mut o = String::new();
        for c in std::mem::take(&mut self.out).chars() {
            if self.indent && c != '\n' {
                o.push_str("  ");
                self.indent = false;
            }
            if c == '\n' {
                self.indent = true;
            }
            o.push(c);
        }
        print!("{o}");
    }

    fn push(&mut self, s: &str) {
        for c in s.chars() {
            self.ch(c);
        }
        self.emit();
        let _ = std::io::stdout().flush();
    }

    fn put(&mut self, s: &str) {
        self.out.push_str(s);
    }

    fn ch(&mut self, c: char) {
        if !self.mid {
            if c == '\n' {
                self.end_held();
                self.put("\n");
                return;
            }
            self.held.push(c);
            let h = self.held.trim_start().to_string();
            if "```".starts_with(&h) || h.starts_with("```") {
                return;
            }
            if self.fence {
                let h = std::mem::take(&mut self.held);
                self.gutter();
                self.put(term::esc("\x1b[36m"));
                self.put(&h);
                self.put(term::esc("\x1b[39m"));
                self.mid = true;
                return;
            }
            if h.len() <= 6 && h.chars().all(|x| x == '#') {
                return;
            }
            if h.starts_with('#') && h.trim_start_matches('#').starts_with(' ') {
                self.held.clear();
                self.heading = true;
                self.mid = true;
                self.put(term::esc("\x1b[1m"));
                return;
            }
            // "- " and "* " list items get a bullet (a lone "*" may still be "**")
            if h == "-" || h == "*" {
                return;
            }
            if h == "- " || h == "* " {
                let indent = self.held[..self.held.len() - 2].to_string();
                self.held.clear();
                self.mid = true;
                self.put(&indent);
                self.put(term::esc("\x1b[38;5;173m"));
                self.put("•");
                self.put(term::esc("\x1b[39m"));
                self.put(" ");
                return;
            }
            self.mid = true;
            for x in std::mem::take(&mut self.held).chars() {
                self.inline(x);
            }
            return;
        }
        if c == '\n' {
            self.end_line();
            self.put("\n");
            return;
        }
        if self.fence {
            self.put(term::esc("\x1b[36m"));
            self.out.push(c);
            self.put(term::esc("\x1b[39m"));
            return;
        }
        self.inline(c);
    }

    fn end_held(&mut self) {
        let h = std::mem::take(&mut self.held);
        if h.trim_start().starts_with("```") {
            self.fence = !self.fence;
            let lang = h.trim_start().trim_start_matches('`').trim();
            if self.fence && !lang.is_empty() {
                self.put(&dim(lang));
            }
        } else if self.fence {
            self.gutter();
            self.put(term::esc("\x1b[36m"));
            self.put(&h);
            self.put(term::esc("\x1b[39m"));
        } else {
            for x in h.chars() {
                self.inline(x);
            }
            self.end_line();
        }
    }

    fn end_line(&mut self) {
        if self.star {
            self.star = false;
            self.out.push('*');
        }
        if self.heading || self.bold {
            self.put(term::esc("\x1b[22m"));
        }
        if self.code {
            self.put(term::esc("\x1b[39m"));
        }
        self.heading = false;
        self.bold = false;
        self.code = false;
        self.mid = false;
    }

    fn inline(&mut self, c: char) {
        if self.star {
            self.star = false;
            if c == '*' && !self.code {
                self.bold = !self.bold;
                self.put(term::esc(if self.bold { "\x1b[1m" } else { "\x1b[22m" }));
                return;
            }
            self.out.push('*');
        }
        if c == '*' && !self.code {
            self.star = true;
            return;
        }
        if c == '`' {
            self.code = !self.code;
            self.put(term::esc(if self.code { "\x1b[36m" } else { "\x1b[39m" }));
            return;
        }
        self.out.push(c);
    }

    /// The bar in front of each line of a code block.
    fn gutter(&mut self) {
        self.put(term::esc("\x1b[90m"));
        self.put("▎ ");
        self.put(term::esc("\x1b[39m"));
    }

    /// Ends the line being written, if one is (a status line then goes below it).
    fn break_line(&mut self) {
        if !self.held.is_empty() {
            self.end_held();
        } else if self.mid {
            self.end_line();
        } else {
            return;
        }
        self.put("\n");
        self.emit();
        let _ = std::io::stdout().flush();
    }

    fn finish(&mut self) {
        if !self.held.is_empty() {
            self.end_held();
            self.put("\n");
        } else if self.mid {
            self.end_line();
            self.put("\n");
        }
        self.emit();
        // (not through emit: it would indent a line that never comes)
        print!("{}", term::esc("\x1b[0m"));
        let _ = std::io::stdout().flush();
    }
}

// ---------- the system prompt (as desktop/src/agent/agent.ts) ----------

fn today() -> String {
    // days since 1970 → a date, without a time crate
    let days = (now_ms() / 86_400_000) as i64;
    let (y, m, d) = civil(days);
    const M: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    format!("{} {d}, {y}", M[(m - 1) as usize])
}

/// Days since 1970-01-01 to (year, month, day).
fn civil(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + if m <= 2 { 1 } else { 0 }, m, d)
}

fn system_prompt(s: &Session, tools: &[Value]) -> String {
    let has = |n: &str| tools.iter().any(|t| t["name"] == n);
    let files = has("write_file");
    let folder = s.root.display();
    let mut parts: Vec<String> = Vec::new();
    let name = s.settings["userName"].as_str().unwrap_or("").trim().to_string();
    parts.push(format!(
        "You are Pelica, an AI assistant running in the user's terminal (Pelica's command line). You run entirely on the user's own Windows PC{}, loaded by Pelica's local engine: nothing is sent to a cloud service. Today is {}.{}\nYou are not a chatbot in a browser tab. You run in a terminal that gives you real tools, and the user expects you to act with them the way a coding agent in the terminal does: you do the work, you don't describe it.",
        if s.model.is_empty() { String::new() } else { format!(" as the model {}", s.model) },
        today(),
        if name.is_empty() { String::new() } else { format!(" The user's name is {name}.") }
    ));
    let mode_note = match s.mode.as_str() {
        "ask" => "Permission mode: Manual. The user approves every file change and every command before it happens; a declined action comes back as a tool result saying so.",
        "auto" => "Permission mode: Bypass permissions. File changes and commands run at once without asking (deleted files still go to the Recycle Bin).",
        _ => "Permission mode: Accept edits. File changes inside the folder happen at once; each command waits for the user's approval.",
    };
    if files {
        parts.push(format!("# Your workspace\nThe working folder is {folder}. Every path you pass to a tool is relative to it (subfolders are fine; you can't reach outside it). The computer runs Windows: run_command uses cmd.exe, so use Windows commands (dir, type, copy, move, del, where, set, mkdir) and Windows paths, chain with &&, and don't assume Unix tools like ls, cat, grep, rm or touch exist.\n{mode_note}"));
        parts.push("# You build things; you don't hand out instructions\nWhen the user asks you to make, build, write, create, set up, fix, change, install or run something, you do it yourself with your tools, right now, start to finish:\n- Files: create them with write_file and change them with edit_file. Never paste code into the chat for the user to copy into a file, and never tell them to \"create a file called…\", \"open your editor\", \"save this as…\" or \"paste this\".\n- Commands: run them with run_command (installing packages, building, testing, starting a server). Never tell the user to type something you could run yourself.\n- Never answer a request to build something with a tutorial or numbered steps for the user. Steps are for you to carry out, not for them.\n- Only show code in the chat when the user asks to see or explain code, or for a short snippet that illustrates an answer.\n- Finish the whole job. Don't stop halfway to ask \"should I continue?\" or \"would you like me to…?\" when the next step is obvious; ask only when a real choice is the user's to make or something is missing that you can't find yourself.".into());
        parts.push("# How to work\n1. Understand: if the request is unclear in a way that changes what you'd build, ask one short question; otherwise make sensible choices and go.\n2. Look first: for an existing project, list_files and read the relevant files (search_files to find things) before changing anything. Follow the code style, structure and libraries already there.\n3. Plan: for work with three or more steps, write the steps with todo_write first, then keep that list current (one item in_progress at a time) until every item is completed.\n4. Build: write complete, working code with no placeholders, no \"TODO: implement\", no \"rest of the code here\". Prefer small, focused edit_file changes to rewriting a whole file.\n5. Check: run it. Build it, run its tests, or start it with run_command, read the output and fix every error until it works. Don't call something done that you haven't run, unless it can't be run here (then say so).\n6. Report: finish with a short summary of what you made or changed (file names), what you checked and the result, and how to run or use it.".into());
    } else if has("read_file") {
        parts.push(format!("# Your workspace\nThe working folder is {folder}. Permission mode: Plan. You can list, read and search its files but not change anything or run commands. Investigate thoroughly with your tools, then write a concrete plan: which files to create or change, what goes in each, and the commands that will build and check it. The user switches the mode (/mode) when they want it carried out; then you carry it out yourself."));
    }
    let mut guide: Vec<&str> = Vec::new();
    if has("list_files") {
        guide.push("- list_files: see the folder's files and folders (depth sets how many levels). Start here in a folder you haven't looked at.");
    }
    if has("read_file") {
        guide.push("- read_file: read a file with line numbers (offset/limit for long files). Read a file before you edit it, every time: never edit from memory.");
    }
    if has("search_files") {
        guide.push("- search_files: find text or a regular expression across the files (glob limits it, e.g. *.py). Use it to find where something is defined or used.");
    }
    if has("write_file") {
        guide.push("- write_file: create a new file or replace a file's whole content. Always the complete content. Folders are created as needed.");
    }
    if has("edit_file") {
        guide.push("- edit_file: replace old_string with new_string in a file. old_string must be copied exactly from the file as read_file showed it (same spaces and indentation, without the line numbers) and must appear only once, so include a few lines of surrounding context. If it fails, read the file again and retry with the exact text.");
    }
    if has("delete_file") {
        guide.push("- delete_file: move a file or folder to the Recycle Bin.");
    }
    if has("run_command") {
        guide.push("- run_command: run a cmd.exe command in the folder and get its output and exit code. Give a short description. Set background: true for anything that keeps running (dev servers, watchers), then check its first output. Raise timeout_seconds for long installs or builds. Check that a tool exists (where python, node --version) before relying on it, and install what's missing (pip install, npm install) when the task needs it.");
    }
    if has("todo_write") {
        guide.push("- todo_write: your task list for multi-step work; send the whole list every time.");
    }
    if has("web_search") {
        guide.push("- web_search, then open_page: for current or outside facts (news, prices, recent versions and releases, documentation you aren't sure of). Search first instead of guessing, open the best results, and cite the links you used. Don't search for small talk or things you already know well.");
    }
    if !guide.is_empty() {
        parts.push(format!("# Your tools\n{}\nUsing them well:\n- Call a tool by actually calling it, in the tool-call format you were given. Don't write \"I will now run…\" or describe a call without making it, and don't put tool-call markup or JSON in your answer text.\n- One step at a time: after each result, look at what it says and decide the next step from it.\n- Read every result. If a tool returns an error, the error is the truth: fix the cause and try again, or try another way. Never pretend it worked.\n- Don't repeat a call that already succeeded with the same arguments; use its result.", guide.join("\n")));
    }
    parts.push(format!(
        "# Accuracy\n- Only say you did something if a tool result shows it happened. Never claim a file was created, a command passed or a test succeeded without seeing it in a result.\n- Never invent file contents, paths, function or library names, command output, version numbers, URLs, quotes or sources.{}{}\n- If you're not sure, say so plainly and say how to check, rather than guessing with confidence.\n- Keep what you verified separate from what you assume.",
        if files { " When you need to know what's in a file or what a command prints, read it or run it." } else { "" },
        if has("web_search") { " When you need a current fact, search for it." } else { "" }
    ));
    parts.push("# Style\nAnswer in Markdown, kept readable in a terminal: short paragraphs, lists, and code blocks; no tables wider than a terminal. Be direct and concise unless the user asks for depth; no filler, no repeating the question back.".into());
    for f in ["PELICA.md", "AGENTS.md", "CLAUDE.md"] {
        if let Ok(t) = std::fs::read_to_string(s.root.join(f)) {
            parts.push(format!("# Project notes (from the folder)\n{}", t.chars().take(6000).collect::<String>()));
            break;
        }
    }
    if let Some(ci) = s.settings["customInstructions"].as_str().map(str::trim).filter(|c| !c.is_empty()) {
        parts.push(format!("# The user's instructions\n{ci}"));
    }
    parts.join("\n\n")
}

/// Small talk gets no tools (small models otherwise search for anything).
fn small_talk(t: &str) -> bool {
    let t = t.trim();
    t.len() < 60 && regex::Regex::new(r"(?i)^\s*(hi|hello|hey|yo|hiya|howdy|sup|hola|thanks|thank you|thx|ty|ok|okay|cool|nice|great|bye|good (morning|afternoon|evening|night)|what'?s up|how are you|who are you|what are you)\b[\s!.?,]*([a-z]+[\s!.?,]*){0,4}$").unwrap().is_match(t)
}

/// A reply that told the user to do the work by hand, or announced a step
/// without taking it: what to tell the model (as the app's handBack).
fn hand_back(text: &str) -> Option<&'static str> {
    let t = text.trim();
    let manual = regex::Regex::new(r"(?i)\b(open (a|your|the) (text |code )?editor|open (vs ?code|notepad)|create a (new )?(file|folder|directory)( (named|called))?|make a (new )?(file|folder) (named|called)|copy (and paste )?(the|this|that|the following) code|paste (it|the|this|that) (code )?(in|into)|save (the|this|it) (file )?as|save the file|open (a|your|the) (terminal|command prompt|powershell|cmd)|in (your|a|the) (terminal|command prompt|cmd)|run the following (command|commands)|type the following|follow these steps|here('s| is| are) (how|the steps)|you(?:'ll| will)? need to (create|install|run|open|save)|step \d+\s*[:.)-]\s*(create|open|install|save|copy|paste|run))").unwrap();
    let numbered = regex::Regex::new(r"(?m)^\s*\d+[.)]\s").unwrap();
    if manual.is_match(t) && (t.contains("```") || numbered.is_match(t)) {
        return Some("Your previous reply told the user how to do the work by hand (create files, paste code, run commands). That's not how you work here: you have the tools, so do it yourself now. Call write_file for each file with its full content, edit_file for changes, and run_command to install, build and run it. Then report what you did.");
    }
    let tail: String = t.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
    let promise = regex::Regex::new(r"(?i)\b(let me|let's|i(?:'ll| will| am going to|'m going to)|now i(?:'ll| will)|next,? i(?:'ll| will))\b[^\n]{0,80}\b(create|write|run|build|edit|update|make|install|start|read|look|check|open|fix|add|search|test)\b[^\n]{0,120}[.:…]?\s*$").unwrap();
    if promise.is_match(&tail) {
        return Some("Your previous reply said you would take a step but stopped without calling a tool. Don't announce it: make the tool call now and carry on until the task is finished.");
    }
    None
}

// ---------- the session ----------

struct Session {
    engine: Engine,
    root: PathBuf,
    settings: Value,
    chat: Value,
    mode: String,
    effort: Option<String>,
    web: bool,
    show_think: bool,
    model: String,
    ctx: usize,
    ctx_used: usize,
    max_tokens: u64,
    max_steps: usize,
    /// commands allowed "for this chat"
    allowed: HashSet<String>,
    /// one task from -p: nothing can be asked
    print: bool,
}

fn new_chat(root: &Path) -> Value {
    let t = now_ms();
    json!({ "id": uid(), "title": "New chat", "created": t, "updated": t, "folder": root.to_string_lossy(), "messages": [] })
}

fn title_from(text: &str) -> String {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.chars().count() > 48 {
        let s: String = t.chars().take(46).collect();
        format!("{}…", s.rsplit_once(' ').map_or(s.as_str(), |x| x.0))
    } else if t.is_empty() {
        "New chat".into()
    } else {
        t
    }
}

impl Session {
    fn msgs(&mut self) -> &mut Vec<Value> {
        if !self.chat["messages"].is_array() {
            self.chat["messages"] = json!([]);
        }
        self.chat["messages"].as_array_mut().unwrap()
    }

    fn push(&mut self, m: Value) {
        self.msgs().push(m);
        self.save();
    }

    /// Into the app's chats folder (whole to a temporary file, then renamed).
    fn save(&mut self) {
        if self.chat["messages"].as_array().is_none_or(|m| m.is_empty()) {
            return;
        }
        self.chat["updated"] = json!(now_ms());
        let Some(id) = self.chat["id"].as_str().filter(|i| i.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')).map(String::from) else { return };
        let dir = user_dir().join("chats");
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let f = dir.join(format!("{id}.json"));
        let tmp = dir.join(format!("{id}.json.tmp"));
        if std::fs::write(&tmp, self.chat.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, &f);
        }
    }

    /// The conversation as the engine takes it (old tool output trimmed,
    /// calls left without a result answered as interrupted).
    fn wire(&self) -> Vec<Value> {
        let msgs: Vec<&Value> = self.chat["messages"].as_array().map(|v| v.iter().filter(|m| m["role"] != "event").collect()).unwrap_or_default();
        let users: Vec<usize> = msgs.iter().enumerate().filter(|(_, m)| m["role"] == "user").map(|(i, _)| i).collect();
        let keep_from = if users.len() >= 2 { users[users.len() - 2] } else { 0 };
        let last_user = users.last().copied().unwrap_or(0);
        let mut out = Vec::new();
        for (i, m) in msgs.iter().enumerate() {
            match m["role"].as_str() {
                Some("user") => out.push(json!({ "role": "user", "content": m["content"] })),
                Some("assistant") => {
                    let mut a = json!({ "role": "assistant", "content": m["content"] });
                    if i > last_user && m["reasoning"].is_string() {
                        a["reasoning"] = m["reasoning"].clone();
                    }
                    let calls = m["tool_calls"].as_array().cloned().unwrap_or_default();
                    if !calls.is_empty() {
                        a["tool_calls"] = json!(calls);
                    }
                    out.push(a);
                    for c in &calls {
                        if !msgs[i + 1..].iter().any(|x| x["role"] == "tool" && x["tool_call_id"] == c["id"]) {
                            out.push(json!({ "role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": "Error: interrupted by the user." }));
                        }
                    }
                }
                Some("tool") => {
                    let c = m["content"].as_str().unwrap_or("");
                    let c = if i < keep_from && c.chars().count() > 600 { format!("{}\n[…{} more characters of older output trimmed]", c.chars().take(600).collect::<String>(), c.chars().count() - 600) } else { c.to_string() };
                    out.push(json!({ "role": "tool", "tool_call_id": m["tool_call_id"], "name": m["name"], "content": c }));
                }
                _ => {}
            }
        }
        out
    }

    fn tools(&self) -> Vec<Value> {
        let msgs = self.chat["messages"].as_array();
        let last_user = msgs.and_then(|m| m.iter().rev().find(|x| x["role"] == "user")).and_then(|m| m["content"].as_str()).unwrap_or("");
        if small_talk(last_user) && !msgs.is_some_and(|m| m.iter().any(|x| x["role"] == "tool")) {
            return Vec::new();
        }
        cli_tools::specs(self.web, self.mode == "plan")
    }
}

// ---------- one call to the model ----------

struct Reply {
    content: String,
    reasoning: String,
    call: Option<Value>,
    done: Value,
    error: Option<String>,
    stopped: bool,
}

/// Streams one completion, writing it to the terminal as it comes.
fn complete(s: &mut Session, system: &str, tools: &[Value], quiet: bool) -> Result<Reply> {
    let mut messages = vec![json!({ "role": "system", "content": system })];
    messages.extend(s.wire());
    let mut req = json!({ "v": 2, "messages": messages, "tools": tools, "max_tokens": s.max_tokens });
    if let Some(e) = &s.effort {
        req["effort"] = json!(e);
    }
    complete_raw(s, &req, quiet)
}

fn complete_raw(s: &mut Session, req: &Value, quiet: bool) -> Result<Reply> {
    let port = s.engine.port;
    let mut sock = connect(port)?;
    sock.set_read_timeout(Some(Duration::from_secs(120)))?;
    send_req(&mut sock, port, "POST", "/api/chat", Some(req))?;
    let mut r = BufReader::new(sock.try_clone()?);
    let (code, len) = read_head(&mut r)?;
    // (then short waits, to keep the status line moving and see Ctrl+C)
    sock.set_read_timeout(Some(Duration::from_millis(250)))?;
    if code != 200 {
        let mut b = vec![0u8; len.unwrap_or(0)];
        let _ = r.read_exact(&mut b);
        let v: Value = serde_json::from_slice(&b).unwrap_or(Value::Null);
        let e = v["error"].as_str().unwrap_or("the engine refused").to_string();
        return Ok(Reply { content: String::new(), reasoning: String::new(), call: None, done: Value::Null, error: Some(if e == "already answering" { "The model is busy answering in the Pelica app. Try again when it's done.".into() } else { e }), stopped: false });
    }
    let mut rep = Reply { content: String::new(), reasoning: String::new(), call: None, done: Value::Null, error: None, stopped: false };
    let mut md = Md::default();
    let t0 = Instant::now();
    let mut started_text = false;
    let mut started_think = false;
    let mut last_status = Instant::now() - Duration::from_secs(1);
    let mut phase = String::from("Waiting for the model");
    // (bytes, so a character split across two reads isn't lost)
    let mut acc: Vec<u8> = Vec::new();
    let mut stop_sent = false;
    // tokens written, and when text last showed: a tool call is written out of
    // sight (a whole file, say), so a status line says the model is at work
    let mut generated = 0u64;
    let mut last_text = Instant::now();
    let mut working = false;
    loop {
        if term::INTERRUPT.swap(false, Ordering::SeqCst) && !stop_sent {
            stop_sent = true;
            let _ = s.engine.call("POST", "/api/stop", Some(&json!({})));
        }
        let unseen = started_text && last_text.elapsed() > Duration::from_millis(1500);
        let thinking_shown = started_think && s.show_think && !started_text;
        if !quiet && (!started_text || unseen) && !thinking_shown && last_status.elapsed() > Duration::from_millis(200) {
            if started_text {
                if !working {
                    md.break_line();
                    working = true;
                }
                term::status_raw(&format!("{} {}", orange(&spinner(t0).to_string()), dim(&format!("Working… {}s · {generated} tokens · ctrl+c to stop", t0.elapsed().as_secs()))));
            } else {
                term::status_raw(&format!("{} {}", orange(&spinner(t0).to_string()), dim(&format!("{phase}… {}s · ctrl+c to stop", t0.elapsed().as_secs()))));
            }
            last_status = Instant::now();
        }
        match r.read_until(b'\n', &mut acc) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => continue,
            Err(e) => return Err(e.into()),
        }
        if acc.last() != Some(&b'\n') {
            continue;
        }
        let line = String::from_utf8_lossy(&std::mem::take(&mut acc)).into_owned();
        let Some(data) = line.trim_end().strip_prefix("data:") else { continue };
        let Ok(ev) = serde_json::from_str::<Value>(data.trim()) else { continue };
        match ev["t"].as_str().unwrap_or("") {
            "start" => phase = "Reading".into(),
            "prefill" => {
                let (d, t) = (ev["done"].as_f64().unwrap_or(0.0), ev["total"].as_f64().unwrap_or(1.0).max(1.0));
                phase = format!("Reading the conversation {:.0}%", d * 100.0 / t);
            }
            "think" => {
                let d = ev["d"].as_str().unwrap_or("");
                rep.reasoning.push_str(d);
                phase = "Thinking".into();
                if s.show_think && !quiet {
                    if !started_think {
                        term::clear_status();
                        started_think = true;
                    }
                    print!("{}", dim(d));
                    let _ = std::io::stdout().flush();
                }
            }
            "text" => {
                let d = ev["d"].as_str().unwrap_or("");
                last_text = Instant::now();
                if working {
                    term::clear_status();
                    working = false;
                }
                if !started_text {
                    let d0 = d.trim_start();
                    if d0.is_empty() {
                        continue;
                    }
                    term::clear_status();
                    if started_think {
                        println!("\n");
                    }
                    started_text = true;
                    if !quiet {
                        print!("{} ", orange("●"));
                    }
                    rep.content.push_str(d0);
                    if !quiet {
                        md.push(d0);
                    }
                    continue;
                }
                rep.content.push_str(d);
                if !quiet {
                    md.push(d);
                }
            }
            "stats" => {
                generated = ev["generated"].as_u64().unwrap_or(generated);
                if !started_text && !started_think {
                    phase = format!("Writing ({:.1} tok/s)", ev["tps"].as_f64().unwrap_or(0.0));
                }
            }
            "tool_call" => {
                let name = ev["name"].as_str().filter(|n| !n.is_empty()).unwrap_or("unknown");
                rep.call = Some(json!({ "id": ev["id"], "name": name, "arguments": if ev["arguments"].is_null() { json!({}) } else { ev["arguments"].clone() } }));
                // a call written as plain JSON: keep only the text before it
                // (less a last line that only introduces it)
                if let Some(b) = ev["before"].as_str() {
                    rep.content = regex::Regex::new(r"\n*[^\n]*:\s*$").unwrap().replace(b, "").into_owned();
                }
            }
            "done" => {
                rep.stopped = ev["finish"] == "stopped";
                rep.done = ev;
            }
            "error" => rep.error = Some(ev["error"].as_str().unwrap_or("error").to_string()),
            _ => {}
        }
    }
    term::clear_status();
    if started_text && !quiet {
        md.finish();
    } else if started_think && !quiet {
        println!();
    }
    Ok(rep)
}

fn spinner(t0: Instant) -> char {
    ['✶', '✸', '✹', '✺', '✹', '✷'][(t0.elapsed().as_millis() / 160 % 6) as usize]
}

// ---------- showing tool calls ----------

fn verb(name: &str) -> &'static str {
    match name {
        "list_files" => "List",
        "read_file" => "Read",
        "write_file" => "Write",
        "edit_file" => "Update",
        "search_files" => "Search",
        "run_command" => "Run",
        "delete_file" => "Delete",
        "todo_write" => "Update Todos",
        "web_search" => "Web Search",
        "open_page" => "Fetch",
        _ => "Tool",
    }
}

fn call_head(name: &str, a: &Value) -> String {
    let arg = match name {
        "run_command" => a["command"].as_str().unwrap_or("").to_string(),
        "web_search" => a["query"].as_str().unwrap_or("").to_string(),
        "open_page" => a["url"].as_str().unwrap_or("").to_string(),
        "search_files" => format!("\"{}\"{}", a["pattern"].as_str().unwrap_or(""), a["glob"].as_str().map(|g| format!(" in {g}")).unwrap_or_default()),
        "todo_write" => String::new(),
        _ => a["path"].as_str().filter(|p| !p.is_empty()).unwrap_or(".").to_string(),
    };
    let arg: String = if arg.chars().count() > 90 { arg.chars().take(88).collect::<String>() + "…" } else { arg };
    if arg.is_empty() {
        format!("{} {}", blue("●"), bold(verb(name)))
    } else {
        format!("{} {}({})", blue("●"), bold(verb(name)), arg)
    }
}

/// Lines that changed between two versions of a file, the common start and
/// end set aside: (line number, '-' or '+', text), removed ones first.
fn diff_lines<'a>(before: &'a str, after: &'a str) -> Vec<(usize, char, &'a str)> {
    let a: Vec<&str> = before.lines().collect();
    let b: Vec<&str> = after.lines().collect();
    let mut pre = 0;
    while pre < a.len() && pre < b.len() && a[pre] == b[pre] {
        pre += 1;
    }
    let mut suf = 0;
    while suf < a.len() - pre && suf < b.len() - pre && a[a.len() - 1 - suf] == b[b.len() - 1 - suf] {
        suf += 1;
    }
    let del = a[pre..a.len() - suf].iter().enumerate().map(|(i, l)| (pre + i + 1, '-', *l));
    let add = b[pre..b.len() - suf].iter().enumerate().map(|(i, l)| (pre + i + 1, '+', *l));
    del.chain(add).collect()
}

/// Prints diff rows (at most `max`), each cut to `w` characters.
fn show_diff(rows: &[(usize, char, &str)], max: usize, w: usize) {
    for (n, k, l) in rows.iter().take(max) {
        let line = format!("{n:>7} {k} {l}");
        let line: String = if line.chars().count() > w { line.chars().take(w - 1).collect::<String>() + "…" } else { line };
        println!("{}", if *k == '+' { green(&line) } else { red(&line) });
    }
    if rows.len() > max {
        println!("{}", dim(&format!("        … {} more lines", rows.len() - max)));
    }
}

fn show_result(name: &str, o: &cli_tools::Out) {
    let ui = &o.ui;
    let pad = "  ⎿  ";
    let w = term::width().saturating_sub(8).max(30);
    let cut = |s: &str| -> String { if s.chars().count() > w { s.chars().take(w - 1).collect::<String>() + "…" } else { s.to_string() } };
    match ui["kind"].as_str().unwrap_or("") {
        "error" => println!("{pad}{}", red(&cut(&format!("Error: {}", ui["error"].as_str().unwrap_or(""))))),
        "denied" => println!("{pad}{}", yellow("Declined")),
        "read" => println!("{pad}{}", dim(&format!("Read {}", ui["detail"].as_str().unwrap_or("")))),
        "list" => {
            let n = o.content.lines().count().saturating_sub(1);
            println!("{pad}{}", dim(&format!("{n} entr{}", if n == 1 { "y" } else { "ies" })));
        }
        "grep" => {
            let n = if o.content == "No matches." { 0 } else { o.content.lines().filter(|l| !l.starts_with('[')).count() };
            println!("{pad}{}", dim(&format!("{n} match{}", if n == 1 { "" } else { "es" })));
        }
        "write" => {
            let after = ui["after"].as_str().unwrap_or("");
            match ui["before"].as_str() {
                None => {
                    let n = after.lines().count();
                    println!("{pad}{}", dim(&format!("Created with {n} line{}", if n == 1 { "" } else { "s" })));
                    for (i, l) in after.lines().take(8).enumerate() {
                        println!("{}", green(&cut(&format!("{:>7} + {l}", i + 1))));
                    }
                    if after.lines().count() > 8 {
                        println!("{}", dim(&format!("        … {} more lines", after.lines().count() - 8)));
                    }
                }
                Some(b) => {
                    println!("{pad}{}", dim("Rewrote the file"));
                    show_diff(&diff_lines(b, after), 16, w);
                }
            }
        }
        "edit" => {
            let (b, a) = (ui["before"].as_str().unwrap_or(""), ui["after"].as_str().unwrap_or(""));
            let d = diff_lines(b, a);
            let added = d.iter().filter(|x| x.1 == '+').count();
            let removed = d.len() - added;
            println!("{pad}{}", dim(&format!("{added} line{} added, {removed} removed", if added == 1 { "" } else { "s" })));
            show_diff(&d, 16, w);
        }
        "command" => {
            let out = ui["output"].as_str().unwrap_or("").trim_end();
            let lines: Vec<&str> = out.lines().collect();
            let st = ui["detail"].as_str().unwrap_or("");
            let ok = ui["code"].as_i64() == Some(0) || st.contains("background");
            if lines.is_empty() {
                println!("{pad}{}", if ok { dim(&format!("(no output) · {st}")) } else { red(&format!("(no output) · {st}")) });
            } else {
                let show = lines.len().min(8);
                if lines.len() > show {
                    println!("{pad}{}", dim(&format!("… {} lines above", lines.len() - show)));
                }
                for (i, l) in lines[lines.len() - show..].iter().enumerate() {
                    println!("{}{}", if i == 0 && lines.len() <= show { pad } else { "     " }, dim(&cut(l)));
                }
                println!("     {}", if ok { dim(st) } else { red(st) });
            }
        }
        "todo" => {
            for t in ui["todos"].as_array().into_iter().flatten() {
                let c = t["content"].as_str().unwrap_or("");
                match t["status"].as_str() {
                    Some("completed") => println!("     {} {}", green("☒"), dim(c)),
                    Some("in_progress") => println!("     {} {}", blue("◼"), bold(c)),
                    _ => println!("     ☐ {c}"),
                }
            }
        }
        "search" => {
            let r = ui["results"].as_array().cloned().unwrap_or_default();
            println!("{pad}{}", dim(&format!("{} results", r.len())));
            for h in r.iter().take(4) {
                println!("     {}", dim(&cut(h["title"].as_str().unwrap_or(""))));
            }
        }
        "page" => println!("{pad}{}", dim(&cut(ui["title"].as_str().unwrap_or("")))),
        "trash" => println!("{pad}{}", dim("Moved to the Recycle Bin")),
        _ => println!("{pad}{}", dim(&cut(o.content.lines().next().unwrap_or("")))),
    }
    let _ = name;
}

// ---------- the agent loop ----------

/// Runs the model on the chat until it answers (tools in between).
fn turn(s: &mut Session) -> Result<()> {
    let t0 = Instant::now();
    let mut generated = 0u64;
    let mut last = Value::Null;
    let mut nudged = 0;
    let mut nudge: Option<&'static str> = None;
    let mut seen: HashSet<String> = HashSet::new();
    let mut skipped = 0;
    let mut steps = 0;
    term::BUSY.store(true, Ordering::SeqCst);
    let res = (|| -> Result<()> {
        loop {
            if steps >= s.max_steps {
                println!("{}", yellow(&format!("Stopped after {steps} steps (the limit in the app's Settings → Capabilities). Send \"continue\" to keep going.")));
                break;
            }
            steps += 1;
            if s.ctx > 0 && s.ctx_used * 100 > s.ctx * 85 && s.settings["autocompact"].as_bool().unwrap_or(true) {
                println!("{}", dim("Context nearly full: compacting the conversation…"));
                compact(s, "", true)?;
            }
            let tools = if skipped >= 2 { Vec::new() } else { s.tools() };
            let mut system = system_prompt(s, &tools);
            if skipped >= 2 {
                system.push_str("\n\nYou kept repeating the same tool call. Tools are now off: answer the user from what you already have.");
            }
            if let Some(n) = nudge.take() {
                system.push_str("\n\n");
                system.push_str(n);
            }
            let rep = complete(s, &system, &tools, false)?;
            if let Some(e) = &rep.error {
                println!("{}", red(&format!("Error: {e}")));
                break;
            }
            generated += rep.done["generated"].as_u64().unwrap_or(0);
            if let Some(c) = rep.done["ctx_used"].as_u64() {
                s.ctx_used = c as usize;
            }
            if let Some(c) = rep.done["ctx"].as_u64() {
                s.ctx = c as usize;
            }
            last = rep.done.clone();
            let mut stats = json!({
                "tps": rep.done["tps"], "generated": rep.done["generated"], "prompt": rep.done["prompt"],
                "prefill_s": rep.done["prefill_s"], "decode_s": rep.done["decode_s"], "think_s": rep.done["think_s"],
                "ctx_used": rep.done["ctx_used"], "ctx": rep.done["ctx"], "first_s": rep.done["first_s"],
            });
            if stats["tps"].is_null() {
                stats = Value::Null;
            }
            let mut msg = json!({ "id": uid(), "role": "assistant", "content": rep.content, "ts": now_ms(), "model": s.model });
            if !rep.reasoning.is_empty() {
                msg["reasoning"] = json!(rep.reasoning);
            }
            if !stats.is_null() {
                msg["stats"] = stats;
            }
            if let Some(e) = &s.effort {
                msg["effort"] = json!(e);
            }
            if let Some(c) = &rep.call {
                msg["tool_calls"] = json!([c]);
            }
            s.push(msg);
            if rep.stopped {
                println!("{}", dim("(stopped)"));
                break;
            }
            let Some(call) = rep.call else {
                if rep.done["finish"] == "length" {
                    println!("{}", yellow("The reply hit the length limit. Send \"continue\" to keep going."));
                    break;
                }
                // told the user to do it by hand, or announced a step without taking it
                if nudged < 2 && tools.iter().any(|t| t["name"] == "write_file") {
                    if let Some(n) = hand_back(&rep.content) {
                        nudged += 1;
                        nudge = Some(n);
                        continue;
                    }
                }
                break;
            };
            let name = call["name"].as_str().unwrap_or("").to_string();
            let args = call["arguments"].clone();
            println!("{}", call_head(&name, &args));
            if let Some(d) = args["description"].as_str().filter(|d| !d.is_empty() && name == "run_command") {
                println!("     {}", dim(d));
            }
            let key = format!("{name}\u{1}{args}");
            let read_only = ["list_files", "read_file", "search_files", "web_search", "open_page"].contains(&name.as_str());
            let out = if read_only && seen.contains(&key) {
                skipped += 1;
                cli_tools::Out { content: format!("You already called {name} with these arguments in this turn; use that result instead of calling it again."), ui: json!({ "kind": "error", "title": name, "error": "Skipped a repeated call" }), todos: None }
            } else {
                let (mode, print) = (s.mode.clone(), s.print);
                let allowed = &mut s.allowed;
                let mut ask = |what: &str, detail: &str, need: Need| -> bool { approve(&mode, print, allowed, what, detail, need) };
                let mut cx = Ctx { root: &s.root, ask: &mut ask };
                cli_tools::run(&mut cx, &name, &args)
            };
            if read_only && !out.content.starts_with("Error") {
                seen.insert(key);
            }
            show_result(&name, &out);
            if let Some(t) = &out.todos {
                s.chat["todos"] = t.clone();
            }
            s.push(json!({ "id": uid(), "role": "tool", "tool_call_id": call["id"], "name": name, "content": out.content, "ui": out.ui, "ts": now_ms() }));
            if term::INTERRUPT.swap(false, Ordering::SeqCst) {
                println!("{}", dim("(stopped)"));
                break;
            }
        }
        Ok(())
    })();
    term::BUSY.store(false, Ordering::SeqCst);
    term::INTERRUPT.store(false, Ordering::SeqCst);
    res?;
    if !last.is_null() && !s.print {
        let el = t0.elapsed().as_secs_f64();
        let pct = if s.ctx > 0 { format!(" · context {}% of {}K", s.ctx_used * 100 / s.ctx, s.ctx / 1024) } else { String::new() };
        let first = last["first_s"].as_f64().map(|f| format!(" · first token {f:.1}s")).unwrap_or_default();
        println!("{} {}", orange("✻"), dim(&format!("{:.1} tok/s · {generated} tokens · {}{first}{pct}", last["tps"].as_f64().unwrap_or(0.0), dur(el))));
    }
    Ok(())
}

fn dur(s: f64) -> String {
    if s < 60.0 {
        format!("{s:.1}s")
    } else {
        format!("{}m {:02}s", (s / 60.0) as u64, (s % 60.0) as u64)
    }
}

/// "? Run this command?  y yes · a always … · n no ›"
fn question(q: &str, always: &str) -> String {
    let key = |k: &str| term::c256(173, &bold(k));
    format!("  {} {}   {} {}  {}  {} {}  {}  {} {}  {}", orange("?"), bold(q), key("y"), dim("yes"), dim("·"), key("a"), dim(always), dim("·"), key("n"), dim("no"), dim("›"))
}

/// Whether an action may go ahead in this permission mode, asking if needed.
fn approve(mode: &str, print: bool, allowed: &mut HashSet<String>, what: &str, detail: &str, need: Need) -> bool {
    let asks = match need {
        Need::File => mode == "ask",
        Need::Command => mode == "ask" || mode == "edits",
    };
    if !asks || allowed.contains(what) {
        return true;
    }
    if print {
        println!("     {}", yellow("Needs your approval, which -p can't ask for (use --mode auto to allow it)."));
        return false;
    }
    let q = match need {
        Need::File => {
            if !detail.is_empty() && detail.len() < 400 {
                for l in detail.lines().take(10) {
                    println!("     {}", dim(l));
                }
            }
            question("Let Pelica make this change?", "always for this file")
        }
        Need::Command => question("Run this command?", "always for it in this chat"),
    };
    match term::ask(&q, &['y', 'a', 'n']) {
        'y' => true,
        'a' => {
            allowed.insert(what.to_string());
            true
        }
        _ => false,
    }
}

// ---------- compacting ----------

fn transcript(msgs: &[Value], max: usize) -> String {
    let mut lines = Vec::new();
    for m in msgs {
        match m["role"].as_str() {
            Some("user") => lines.push(format!("User: {}", m["content"].as_str().unwrap_or(""))),
            Some("assistant") => {
                if let Some(c) = m["content"].as_str().filter(|c| !c.is_empty()) {
                    lines.push(format!("Assistant: {c}"));
                }
                for c in m["tool_calls"].as_array().into_iter().flatten() {
                    lines.push(format!("Assistant called {} {}", c["name"].as_str().unwrap_or(""), c["arguments"].to_string().chars().take(400).collect::<String>()));
                }
            }
            Some("tool") => {
                let c = m["content"].as_str().unwrap_or("");
                lines.push(format!("Result of {}: {}{}", m["name"].as_str().unwrap_or(""), c.chars().take(700).collect::<String>(), if c.chars().count() > 700 { " …" } else { "" }));
            }
            _ => {}
        }
    }
    let t = lines.join("\n\n");
    let n = t.chars().count();
    if n > max {
        format!("[…earlier part cut]\n{}", t.chars().skip(n - max).collect::<String>())
    } else {
        t
    }
}

/// Replaces the conversation so far with a summary the model writes (auto:
/// the current turn stays as it is).
fn compact(s: &mut Session, focus: &str, auto: bool) -> Result<()> {
    let msgs = s.chat["messages"].as_array().cloned().unwrap_or_default();
    let cut = if auto { msgs.iter().rposition(|m| m["role"] == "user").unwrap_or(msgs.len()) } else { msgs.len() };
    let (head, tail) = msgs.split_at(cut);
    if head.iter().filter(|m| m["role"] != "event").count() < 2 {
        println!("{}", dim("Nothing to compact yet."));
        return Ok(());
    }
    let text = transcript(head, (s.ctx.max(8192) as f64 * 0.55 * 3.0) as usize);
    let st = s.engine.status();
    let efforts: Vec<&str> = st["efforts"].as_array().map(|v| v.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
    let mut req = json!({
        "v": 2, "tools": [], "max_tokens": 2000,
        "messages": [
            { "role": "system", "content": "You write complete, compact summaries of conversations, so the work can continue with only the summary." },
            { "role": "user", "content": format!("Summarize the conversation below so it can be continued from your summary alone. Keep: what the user wants and asked for; decisions and preferences; files created or changed (paths) and what they contain; commands run and their results; facts and data found; the current state; unfinished work and next steps. Use short headings and bullet points.{}\n\n<conversation>\n{text}\n</conversation>", if focus.is_empty() { String::new() } else { format!(" Pay special attention to: {focus}.") }) },
        ],
    });
    if efforts.contains(&"off") {
        req["effort"] = json!("off");
    } else if efforts.contains(&"low") {
        req["effort"] = json!("low");
    }
    term::BUSY.store(true, Ordering::SeqCst);
    let rep = complete_raw(s, &req, true);
    term::BUSY.store(false, Ordering::SeqCst);
    let rep = rep?;
    if let Some(e) = rep.error {
        bail!("compacting failed: {e}");
    }
    if rep.stopped || term::INTERRUPT.swap(false, Ordering::SeqCst) {
        bail!("compacting stopped; the conversation is as it was");
    }
    let summary = rep.content.trim().to_string();
    if summary.is_empty() {
        bail!("the model wrote an empty summary");
    }
    let t = now_ms();
    let mut archive = s.chat["archive"].as_array().cloned().unwrap_or_default();
    archive.extend(head.iter().cloned());
    let mut new = vec![
        json!({ "id": uid(), "role": "event", "kind": "compact", "text": if auto { "Context nearly full: conversation compacted" } else { "Conversation compacted" }, "detail": summary, "ts": t }),
        json!({ "id": uid(), "role": "user", "content": format!("Here is a summary of our conversation so far:\n\n{summary}"), "summary": true, "ts": t }),
        json!({ "id": uid(), "role": "assistant", "content": "Got it. I'll continue from this summary.", "ack": true, "ts": t }),
    ];
    new.extend(tail.iter().cloned());
    s.chat["archive"] = json!(archive);
    s.chat["messages"] = json!(new);
    s.ctx_used = 0;
    s.save();
    println!("{}", dim(&format!("Compacted: {} messages summed up.", head.len())));
    Ok(())
}

// ---------- models ----------

/// Loads `path` and waits, with a progress line.
fn load_model(e: &Engine, path: &str, ctx: usize) -> Result<Value> {
    let (code, v) = e.call("POST", "/api/load", Some(&json!({ "path": path, "ctx": ctx })))?;
    if code != 200 {
        bail!("{}", v["error"].as_str().unwrap_or("the engine wouldn't load it"));
    }
    let t0 = Instant::now();
    let mut said = -1i64;
    loop {
        std::thread::sleep(Duration::from_millis(400));
        let st = e.status();
        match st["state"].as_str() {
            Some("ready") => {
                term::clear_status();
                return Ok(st);
            }
            Some("error") => {
                term::clear_status();
                bail!("{}", st["error"].as_str().unwrap_or("loading failed"));
            }
            _ => {
                let p = st["progress"].as_f64().unwrap_or(0.0);
                let n = (p / 5.0) as usize;
                let name = Path::new(path).file_stem().map_or(path.into(), |f| f.to_string_lossy());
                term::status_raw(&format!(
                    "{} Loading {name}  {}{}  {}",
                    orange(&spinner(t0).to_string()),
                    orange(&"━".repeat(n.min(20))),
                    dim(&"━".repeat(20 - n.min(20))),
                    dim(&format!("{p:.0}% · {}s", t0.elapsed().as_secs()))
                ));
                // (without a status line: a line every 25%)
                if !term::color() && (p as i64) / 25 > said {
                    said = (p as i64) / 25;
                    println!("Loading… {p:.0}%");
                }
            }
        }
    }
}

fn models(e: &Engine) -> Vec<Value> {
    e.get("/api/models").ok().and_then(|v| v["models"].as_array().cloned()).unwrap_or_default()
}

/// A model by path, file name or part of its name.
fn find_model(list: &[Value], want: &str) -> Option<String> {
    if Path::new(want).is_file() {
        return Some(want.to_string());
    }
    let w = want.to_lowercase();
    list.iter()
        .find(|m| m["name"].as_str().is_some_and(|n| n.to_lowercase() == w))
        .or_else(|| list.iter().find(|m| m["name"].as_str().is_some_and(|n| n.to_lowercase().contains(&w)) || m["path"].as_str().is_some_and(|p| p.to_lowercase().contains(&w))))
        .and_then(|m| m["path"].as_str().map(String::from))
}

fn pick_model(list: &[Value], current: &str) -> Option<String> {
    if list.is_empty() {
        println!("{}", yellow("No models found. Download one in the Pelica app (Models), or pass a .gguf path with -m."));
        return None;
    }
    for (i, m) in list.iter().enumerate() {
        let name = m["name"].as_str().unwrap_or("?");
        let cur = if !current.is_empty() && m["path"].as_str() == Some(current) { green(" (loaded)") } else { String::new() };
        println!("  {} {}  {}{}", dim(&format!("{:>2}.", i + 1)), name, dim(&format!("{:.1} GB", m["gb"].as_f64().unwrap_or(0.0))), cur);
    }
    match term::read_line(&format!("{} ", dim("Number (Enter to cancel):"))) {
        term::Line::Text(t) => t.trim().parse::<usize>().ok().filter(|&n| n >= 1 && n <= list.len()).and_then(|n| list[n - 1]["path"].as_str().map(String::from)),
        _ => None,
    }
}

/// Makes sure a model is loaded: the one asked for, else what the engine
/// has, else the app's last one, else one picked from the list.
fn ensure_model(s: &mut Session, want: Option<&str>, ctx: Option<usize>) -> Result<()> {
    let mut st = s.engine.status();
    while st["state"] == "loading" {
        term::status(&format!("The Pelica app is loading {}… {:.0}%", st["model"].as_str().unwrap_or("a model"), st["progress"].as_f64().unwrap_or(0.0)));
        std::thread::sleep(Duration::from_millis(500));
        st = s.engine.status();
    }
    term::clear_status();
    let list = models(&s.engine);
    let loaded = if st["state"] == "ready" { st["path"].as_str().unwrap_or("").to_string() } else { String::new() };
    let path = match want {
        Some(w) => Some(find_model(&list, w).with_context(|| format!("no model matches \"{w}\" (see /model)"))?),
        None if !loaded.is_empty() => None,
        None => {
            let last = s.settings["lastModel"].as_str().filter(|p| Path::new(p).is_file()).map(String::from);
            match last {
                Some(p) => Some(p),
                None if list.len() == 1 => list[0]["path"].as_str().map(String::from),
                None => {
                    println!("Which model?");
                    pick_model(&list, "")
                }
            }
        }
    };
    if let Some(p) = path.filter(|p| !p.eq_ignore_ascii_case(&loaded)) {
        if s.engine.shared && !loaded.is_empty() {
            println!("{}", dim(&format!("The Pelica app has {} loaded; switching it.", st["model"].as_str().unwrap_or("a model"))));
        }
        let c = ctx.or_else(|| s.settings["lastCtx"].as_u64().map(|c| c as usize)).unwrap_or(32768);
        st = load_model(&s.engine, &p, c)?;
    }
    if st["state"] != "ready" {
        bail!("no model loaded");
    }
    s.model = st["model"].as_str().unwrap_or("").to_string();
    s.ctx = st["ctx"].as_u64().unwrap_or(0) as usize;
    let efforts: Vec<String> = st["efforts"].as_array().map(|v| v.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
    if s.effort.as_ref().is_some_and(|e| !efforts.contains(e)) || s.effort.is_none() {
        let want = s.settings["effort"].as_str().unwrap_or("").to_string();
        s.effort = if efforts.contains(&want) { Some(want) } else { st["effort"].as_str().map(String::from) };
    }
    if st["agent"] == false {
        println!("{}", yellow("This model wasn't trained to use tools, so it may describe tasks instead of doing them."));
    }
    Ok(())
}

// ---------- chats ----------

fn same_folder(a: &str, b: &Path) -> bool {
    let n = |s: &str| s.trim_end_matches(['\\', '/']).replace('/', "\\").to_lowercase();
    n(a) == n(&b.to_string_lossy())
}

/// This folder's chats, newest first.
fn folder_chats(root: &Path) -> Vec<Value> {
    let dir = user_dir().join("chats");
    let mut v: Vec<Value> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .map(|e| read_json(&e.path()))
        .filter(|c| c["folder"].as_str().is_some_and(|f| same_folder(f, root)) && c["messages"].is_array())
        .collect();
    v.sort_by_key(|c| std::cmp::Reverse(c["updated"].as_u64().unwrap_or(0)));
    v
}

fn ago(t: u64) -> String {
    let m = now_ms().saturating_sub(t) / 60000;
    if m < 1 {
        "just now".into()
    } else if m < 60 {
        format!("{m} min ago")
    } else if m < 1440 {
        format!("{} h ago", m / 60)
    } else {
        format!("{} d ago", m / 1440)
    }
}

/// The last thing said, to show where a resumed chat left off.
fn show_tail(chat: &Value) {
    let msgs = chat["messages"].as_array().cloned().unwrap_or_default();
    let cut = |s: &str, n: usize| -> String { let s = s.split_whitespace().collect::<Vec<_>>().join(" "); if s.chars().count() > n { s.chars().take(n).collect::<String>() + "…" } else { s } };
    if let Some(u) = msgs.iter().rev().find(|m| m["role"] == "user" && m["summary"] != true) {
        println!("{} {}", dim(">"), dim(&cut(u["content"].as_str().unwrap_or(""), 200)));
    }
    if let Some(a) = msgs.iter().rev().find(|m| m["role"] == "assistant" && m["content"].as_str().is_some_and(|c| !c.trim().is_empty())) {
        println!("{} {}", orange("●"), dim(&cut(a["content"].as_str().unwrap_or(""), 300)));
    }
}

// ---------- slash commands ----------

const COMMANDS: &[(&str, &str)] = &[
    ("/help", "this list"),
    ("/clear", "start a new chat (this one stays in the app)"),
    ("/resume", "carry on an earlier chat in this folder"),
    ("/model [name]", "show or switch the model"),
    ("/mode [manual|edits|plan|auto]", "permission mode"),
    ("/effort [level]", "thinking effort"),
    ("/web [on|off]", "web search"),
    ("/think", "show or hide the model's thinking"),
    ("/compact [focus]", "summarize the conversation to free up context"),
    ("/context", "how full the context is"),
    ("/todos", "the current todo list"),
    ("/init", "write PELICA.md: notes about this project for future chats"),
    ("/status", "model, folder, engine and settings"),
    ("/exit", "quit (also Ctrl+C twice at the prompt)"),
];

/// Runs a /command. Returns a message to send instead, if it makes one.
fn command(s: &mut Session, line: &str) -> Result<Option<String>> {
    let (cmd, arg) = line.split_once(char::is_whitespace).map_or((line, ""), |(a, b)| (a, b.trim()));
    match cmd.to_lowercase().as_str() {
        "/help" | "/?" => {
            for (c, h) in COMMANDS {
                println!("  {}  {}", cyan(&format!("{c:<34}")), dim(h));
            }
            println!("  {}", dim("End a line with \\ to write more lines. Ctrl+C stops a reply."));
        }
        "/exit" | "/quit" | "/q" => {
            cli_tools::stop_background();
            std::process::exit(0);
        }
        "/clear" | "/new" => {
            s.chat = new_chat(&s.root);
            s.ctx_used = 0;
            s.allowed.clear();
            println!("{}", dim("New chat."));
        }
        "/resume" => {
            let list: Vec<Value> = folder_chats(&s.root).into_iter().filter(|c| c["id"] != s.chat["id"]).take(12).collect();
            if list.is_empty() {
                println!("{}", dim("No other chats in this folder yet."));
                return Ok(None);
            }
            for (i, c) in list.iter().enumerate() {
                println!("  {} {}  {}", dim(&format!("{:>2}.", i + 1)), c["title"].as_str().unwrap_or("Untitled"), dim(&ago(c["updated"].as_u64().unwrap_or(0))));
            }
            if let term::Line::Text(t) = term::read_line(&format!("{} ", dim("Number (Enter to cancel):"))) {
                if let Some(c) = t.trim().parse::<usize>().ok().filter(|&n| n >= 1 && n <= list.len()).map(|n| list[n - 1].clone()) {
                    s.chat = c;
                    s.ctx_used = 0;
                    s.allowed.clear();
                    println!("{}", dim(&format!("Resumed “{}”.", s.chat["title"].as_str().unwrap_or(""))));
                    show_tail(&s.chat);
                }
            }
        }
        "/model" => {
            let list = models(&s.engine);
            let cur = s.engine.status()["path"].as_str().unwrap_or("").to_string();
            let path = if arg.is_empty() { pick_model(&list, &cur) } else { find_model(&list, arg) };
            match path {
                Some(p) if !p.eq_ignore_ascii_case(&cur) => {
                    let c = s.settings["lastCtx"].as_u64().map(|c| c as usize).unwrap_or(32768);
                    load_model(&s.engine, &p, c)?;
                    ensure_model(s, None, None)?;
                    println!("{}", dim(&format!("Now using {}.", s.model)));
                }
                Some(_) => println!("{}", dim(&format!("Already using {}.", s.model))),
                None if !arg.is_empty() => println!("{}", yellow(&format!("No model matches \"{arg}\".")),),
                None => {}
            }
        }
        "/mode" => {
            if let Some(m) = mode_of(arg) {
                s.mode = m.into();
            } else if !arg.is_empty() {
                println!("{}", yellow("Modes: manual, edits, plan, auto."));
                return Ok(None);
            } else {
                // no argument: the next mode, as Shift+Tab in the app
                let i = MODES.iter().position(|x| x.0 == s.mode).unwrap_or(1);
                s.mode = MODES[(i + 1) % MODES.len()].0.into();
            }
            let d = MODES.iter().find(|x| x.0 == s.mode).map_or("", |x| x.2);
            println!("{} {}", bold(mode_name(&s.mode)), dim(&format!("· {d}")));
        }
        "/effort" => {
            let st = s.engine.status();
            let efforts: Vec<String> = st["efforts"].as_array().map(|v| v.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
            if efforts.is_empty() {
                println!("{}", dim("This model has no thinking levels."));
            } else if arg.is_empty() {
                println!("Effort: {} {}", bold(s.effort.as_deref().unwrap_or("default")), dim(&format!("({})", efforts.join(", "))));
            } else if efforts.iter().any(|e| e == &arg.to_lowercase()) {
                s.effort = Some(arg.to_lowercase());
                println!("Effort: {}", bold(&arg.to_lowercase()));
            } else {
                println!("{}", yellow(&format!("This model's levels: {}.", efforts.join(", "))));
            }
        }
        "/web" => {
            s.web = match arg.to_lowercase().as_str() {
                "on" => true,
                "off" => false,
                _ => !s.web,
            };
            println!("Web search {}", bold(if s.web { "on" } else { "off" }));
        }
        "/think" => {
            s.show_think = !s.show_think;
            println!("Thinking {}", bold(if s.show_think { "shown" } else { "hidden" }));
        }
        "/compact" => compact(s, arg, false)?,
        "/context" => {
            if s.ctx == 0 {
                println!("{}", dim("No reply yet in this session."));
            } else {
                let pct = s.ctx_used * 100 / s.ctx;
                let n = pct.min(100) / 5;
                println!("[{}{}] {pct}% · {} of {} tokens", "█".repeat(n), "·".repeat(20 - n), s.ctx_used, s.ctx);
            }
        }
        "/todos" => match s.chat["todos"].as_array() {
            Some(t) if !t.is_empty() => show_result("todo_write", &cli_tools::Out { content: String::new(), ui: json!({ "kind": "todo", "todos": t }), todos: None }),
            _ => println!("{}", dim("No todo list in this chat.")),
        },
        "/init" => return Ok(Some("Look through this project (list_files, read the key files) and write PELICA.md in its root: what the project is, how it's organized, how to build, run and test it, and conventions to follow. Keep it short and useful for working on the project later.".into())),
        "/status" => {
            println!("  Model    {}", s.model);
            println!("  Folder   {}", s.root.display());
            println!("  Mode     {}", mode_name(&s.mode));
            println!("  Effort   {}", s.effort.as_deref().unwrap_or("default"));
            println!("  Web      {}", if s.web { "on" } else { "off" });
            println!("  Engine   {}", if s.engine.shared { format!("the Pelica app's (port {})", s.engine.port) } else { "this window's own".into() });
            if let Some(l) = &s.engine.log {
                println!("  Log      {}", l.display());
            }
            println!("  Chat     {} ({} messages)", s.chat["title"].as_str().unwrap_or(""), s.chat["messages"].as_array().map_or(0, |m| m.len()));
        }
        _ => println!("{}", yellow(&format!("Unknown command {cmd}. /help lists them."))),
    }
    Ok(None)
}

// ---------- the look ----------

/// A folder for showing: the home folder as ~, a long one cut in the middle.
fn short_path(p: &Path) -> String {
    let mut t = p.display().to_string();
    if let Ok(home) = std::env::var("USERPROFILE") {
        if t.to_lowercase().starts_with(&home.to_lowercase()) && !home.is_empty() {
            t = format!("~{}", &t[home.len()..]);
        }
    }
    let n = t.chars().count();
    if n > 56 {
        let head: String = t.chars().take(22).collect();
        let tail: String = t.chars().skip(n - 33).collect();
        t = format!("{head}…{tail}");
    }
    t
}

fn welcome(s: &Session) {
    let label = |k: &str| dim(&format!("{k:<8}"));
    let ctx = if s.ctx > 0 { dim(&format!("  ·  {}K context", s.ctx / 1024)) } else { String::new() };
    let lines = vec![
        format!("{} {}", orange("✻"), bold("Welcome to Pelica")),
        String::new(),
        dim("/help for commands  ·  /status for your setup  ·  ctrl+c to stop"),
        String::new(),
        format!("{}{}", label("folder"), short_path(&s.root)),
        format!("{}{}{ctx}", label("model"), s.model),
        format!("{}{}", label("engine"), if s.engine.shared { "shared with the Pelica app" } else { "running in this window" }),
    ];
    term::boxed(&lines, 173);
    println!();
}

/// Under the prompt: the permission mode on the left, how full the context
/// is on the right.
fn footer(s: &Session) -> String {
    let mode = match s.mode.as_str() {
        "ask" => dim("◇ manual: asks before every change"),
        "plan" => term::c256(73, "⏸ plan mode on"),
        "auto" => term::c256(203, "⏵⏵ bypass permissions on"),
        _ => term::c256(141, "⏵⏵ accept edits on"),
    };
    let left = format!("  {mode}{}", dim("  ·  /mode to switch"));
    let right = if s.ctx > 0 && s.ctx_used > 0 { dim(&format!("context {}%  ", s.ctx_used * 100 / s.ctx)) } else { String::new() };
    let gap = term::width().saturating_sub(1 + term::vis_len(&left) + term::vis_len(&right));
    format!("{left}{}{right}", " ".repeat(gap))
}

// ---------- start ----------

pub fn run(args: &[String]) -> Result<()> {
    let mut want: Option<String> = None;
    let mut ctx: Option<usize> = None;
    let mut mode: Option<String> = None;
    let mut effort: Option<String> = None;
    let mut web: Option<bool> = None;
    let mut cont = false;
    let mut print: Option<String> = None;
    let mut show_think = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "-m" | "--model" => want = Some(val()?),
            "--ctx" => ctx = Some(val()?.parse().context("--ctx takes a number")?),
            "--mode" => {
                let v = val()?;
                mode = Some(mode_of(&v).with_context(|| format!("unknown mode {v} (manual, edits, plan, auto)"))?.into());
            }
            "--effort" => effort = Some(val()?.to_lowercase()),
            "--web" => web = Some(true),
            "--no-web" => web = Some(false),
            "-c" | "--continue" => cont = true,
            "-p" | "--print" => print = Some(val()?),
            "--show-thinking" => show_think = true,
            "-h" | "--help" => {
                print!("{HELP}");
                return Ok(());
            }
            _ => bail!("unknown option {a}\n\n{HELP}"),
        }
    }
    term::init();
    if print.is_none() {
        term::intro(&format!("local AI in your terminal · v{}", env!("CARGO_PKG_VERSION")));
    }
    let root = std::env::current_dir().context("the current folder")?;
    let settings = read_json(&user_dir().join("settings.json"));
    let engine = start_engine()?;
    let mode = mode.unwrap_or_else(|| settings["permMode"].as_str().and_then(mode_of).unwrap_or("edits").to_string());
    let mut s = Session {
        engine,
        chat: new_chat(&root),
        root,
        mode,
        effort,
        web: web.unwrap_or_else(|| settings["web"].as_bool().unwrap_or(true)),
        show_think,
        model: String::new(),
        ctx: 0,
        ctx_used: 0,
        max_tokens: settings["maxTokens"].as_u64().unwrap_or(8192),
        max_steps: settings["maxSteps"].as_u64().unwrap_or(150) as usize,
        allowed: HashSet::new(),
        print: print.is_some(),
        settings,
    };
    if cont {
        match folder_chats(&s.root).into_iter().next() {
            Some(c) => s.chat = c,
            None => println!("{}", dim("No earlier chat in this folder; starting a new one.")),
        }
    }
    ensure_model(&mut s, want.as_deref(), ctx)?;
    term::title(&format!("Pelica · {}", s.root.file_name().map_or(String::new(), |f| f.to_string_lossy().into_owned())));

    // one task, then out
    if let Some(p) = print {
        let t = now_ms();
        s.chat["title"] = json!(title_from(&p));
        s.push(json!({ "id": uid(), "role": "user", "content": p, "ts": t }));
        turn(&mut s)?;
        cli_tools::stop_background();
        return Ok(());
    }

    welcome(&s);
    if cont && s.chat["messages"].as_array().is_some_and(|m| !m.is_empty()) {
        println!("{}", dim(&format!("Continuing “{}”.", s.chat["title"].as_str().unwrap_or(""))));
        show_tail(&s.chat);
    }
    println!();
    let mut last_ctrl_c: Option<Instant> = None;
    loop {
        let line = match term::prompt(&footer(&s)) {
            term::Line::Text(t) => t,
            term::Line::Interrupted => {
                if last_ctrl_c.is_some_and(|t| t.elapsed() < Duration::from_secs(2)) {
                    break;
                }
                last_ctrl_c = Some(Instant::now());
                println!("{}", dim("(Ctrl+C again to quit)"));
                continue;
            }
            term::Line::Eof => break,
        };
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let msg = if t.starts_with('/') && !t.starts_with("//") {
            match command(&mut s, t) {
                Ok(Some(m)) => m,
                Ok(None) => continue,
                Err(e) => {
                    println!("{}", red(&format!("{e:#}")));
                    continue;
                }
            }
        } else {
            line.trim_end().to_string()
        };
        if s.chat["messages"].as_array().is_none_or(|m| m.is_empty()) {
            s.chat["title"] = json!(title_from(&msg));
        }
        s.push(json!({ "id": uid(), "role": "user", "content": msg, "ts": now_ms() }));
        println!();
        if let Err(e) = turn(&mut s) {
            println!("{}", red(&format!("{e:#}")));
        }
        last_ctrl_c = None;
        println!();
    }
    cli_tools::stop_background();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(20_756), (2026, 10, 30));
    }

    #[test]
    fn titles_and_modes() {
        assert_eq!(title_from("  fix   the build "), "fix the build");
        assert!(title_from(&"word ".repeat(30)).ends_with('…'));
        assert_eq!(mode_of("Bypass"), Some("auto"));
        assert_eq!(mode_of("manual"), Some("ask"));
        assert!(mode_of("nope").is_none());
    }

    #[test]
    fn nudges() {
        assert!(small_talk("hi there!") && !small_talk("hi, build me a snake game in python"));
        assert!(hand_back("Here are the steps:\n1. Create a file called app.py\n2. Run it").is_some());
        assert!(hand_back("Let me read the main file to check.").is_some());
        assert!(hand_back("Done: I created app.py and ran it; it prints 42.").is_none());
    }

    #[test]
    fn diffs() {
        assert_eq!(diff_lines("a\nb\nc\n", "a\nB\nc\n"), vec![(2, '-', "b"), (2, '+', "B")]);
        // (a + inside a removed line is not an added line)
        assert_eq!(diff_lines("x(2 + 3)\n", "x(2 * 3)\n").iter().filter(|r| r.1 == '+').count(), 1);
    }

    #[test]
    fn markdown_streams() {
        let mut m = Md::default();
        for piece in ["# Ti", "tle\nsome **bo", "ld** and `co", "de`\n```rust\nfn x() {}\n```\ndone"] {
            for c in piece.chars() {
                m.ch(c);
            }
        }
        // (no colours in tests: the markup is gone, the text stays)
        assert_eq!(m.out, "Title\nsome bold and code\nrust\n▎ fn x() {}\n\ndone");
        let mut m = Md::default();
        for c in "- one\n  * two\n**b** and *x\n---\n".chars() {
            m.ch(c);
        }
        assert_eq!(m.out, "• one\n  • two\nb and *x\n---\n");
    }
}
