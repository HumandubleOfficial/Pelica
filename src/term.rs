//! Terminal helpers for `pelica chat`: ANSI colours (switched on in the
//! Windows console), Ctrl+C as a stop signal instead of a kill, line input,
//! one-key questions, and moving the engine's log lines out of the way.

use std::io::{self, BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};

/// Ctrl+C was pressed while busy (cleared by whoever acts on it).
pub static INTERRUPT: AtomicBool = AtomicBool::new(false);
/// A reply or a question is under way: Ctrl+C stops it instead of quitting.
pub static BUSY: AtomicBool = AtomicBool::new(false);
/// Waiting for a line from the keyboard.
static READING: AtomicBool = AtomicBool::new(false);
static COLOR: AtomicBool = AtomicBool::new(false);

#[cfg(windows)]
mod win {
    pub const STD_INPUT_HANDLE: u32 = -10i32 as u32;
    pub const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    pub const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    #[repr(C)]
    #[derive(Default)]
    pub struct Csbi {
        pub size: [i16; 2],
        pub cursor: [i16; 2],
        pub attr: u16,
        pub window: [i16; 4],
        pub max: [i16; 2],
    }
    /// INPUT_RECORD holding a KEY_EVENT_RECORD
    #[repr(C)]
    pub struct KeyInput {
        pub kind: u16,
        pub _pad: u16,
        pub down: i32,
        pub repeat: u16,
        pub vk: u16,
        pub scan: u16,
        pub ch: u16,
        pub ctrl: u32,
    }
    extern "system" {
        pub fn GetStdHandle(n: u32) -> isize;
        pub fn SetStdHandle(n: u32, h: isize) -> i32;
        pub fn GetConsoleMode(h: isize, m: *mut u32) -> i32;
        pub fn SetConsoleMode(h: isize, m: u32) -> i32;
        pub fn SetConsoleCtrlHandler(f: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
        pub fn GetConsoleScreenBufferInfo(h: isize, info: *mut Csbi) -> i32;
        pub fn SetConsoleTitleW(t: *const u16) -> i32;
        pub fn WriteConsoleInputW(h: isize, r: *const KeyInput, n: u32, written: *mut u32) -> i32;
        pub fn FlushConsoleInputBuffer(h: isize) -> i32;
    }
    pub unsafe extern "system" fn on_ctrl(kind: u32) -> i32 {
        use std::sync::atomic::Ordering::SeqCst;
        // Ctrl+C and Ctrl+Break stop a reply, or answer a question with no, or
        // clear the prompt; otherwise (loading, say) they quit as usual. The
        // console keeps waiting for the line after a Ctrl+C, so an Enter is
        // typed for it to wake the read up.
        if kind > 1 {
            return 0;
        }
        let reading = super::READING.load(SeqCst);
        if !reading && !super::BUSY.load(SeqCst) {
            return 0;
        }
        super::INTERRUPT.store(true, SeqCst);
        if reading {
            let key = |down| KeyInput { kind: 1, _pad: 0, down, repeat: 1, vk: 0x0D, scan: 0x1C, ch: 13, ctrl: 0 };
            let keys = [key(1), key(0)];
            let mut n = 0;
            WriteConsoleInputW(GetStdHandle(STD_INPUT_HANDLE), keys.as_ptr(), 2, &mut n);
        }
        1
    }
}

/// Colours on when stdout is a terminal (and NO_COLOR isn't set); Ctrl+C
/// handled. Call once at the start.
pub fn init() {
    let tty = io::stdout().is_terminal();
    let mut color = tty && std::env::var_os("NO_COLOR").is_none();
    #[cfg(windows)]
    // SAFETY: plain console API calls on this process's own handles
    unsafe {
        let h = win::GetStdHandle(win::STD_OUTPUT_HANDLE);
        let mut m = 0u32;
        if color && win::GetConsoleMode(h, &mut m) != 0 && win::SetConsoleMode(h, m | 0x0004) == 0 {
            // (a console without virtual terminal sequences: plain text)
            color = false;
        }
        win::SetConsoleCtrlHandler(Some(win::on_ctrl), 1);
    }
    COLOR.store(color, Ordering::Relaxed);
}

/// The window title (shown on the taskbar and the terminal's tab).
pub fn title(t: &str) {
    #[cfg(windows)]
    {
        let w: Vec<u16> = t.encode_utf16().chain([0]).collect();
        // SAFETY: a NUL-terminated UTF-16 string that outlives the call
        unsafe { win::SetConsoleTitleW(w.as_ptr()) };
    }
    let _ = t;
}

/// Sends what the engine writes to stderr (load progress, timings) to a log
/// file, so it doesn't run through the chat. Returns the file's path.
pub fn quiet_stderr() -> Option<std::path::PathBuf> {
    let p = std::env::temp_dir().join("pelica-cli.log");
    let f = std::fs::File::create(&p).ok()?;
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        // SAFETY: the handle stays open for the life of the process (forgotten below)
        unsafe { win::SetStdHandle(win::STD_ERROR_HANDLE, f.as_raw_handle() as isize) };
        std::mem::forget(f);
    }
    Some(p)
}

/// Columns in the terminal window (80 when unknown).
pub fn width() -> usize {
    #[cfg(windows)]
    // SAFETY: fills a plain struct
    unsafe {
        let mut i = win::Csbi::default();
        if win::GetConsoleScreenBufferInfo(win::GetStdHandle(win::STD_OUTPUT_HANDLE), &mut i) != 0 {
            return (i.window[2] - i.window[0] + 1).max(20) as usize;
        }
    }
    80
}

fn paint(code: &str, s: &str) -> String {
    if COLOR.load(Ordering::Relaxed) {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}
pub fn bold(s: &str) -> String {
    paint("1", s)
}
pub fn dim(s: &str) -> String {
    paint("90", s)
}
pub fn red(s: &str) -> String {
    paint("31", s)
}
pub fn green(s: &str) -> String {
    paint("32", s)
}
pub fn yellow(s: &str) -> String {
    paint("33", s)
}
pub fn blue(s: &str) -> String {
    paint("94", s)
}
pub fn cyan(s: &str) -> String {
    paint("36", s)
}
pub fn orange(s: &str) -> String {
    paint("38;5;173", s)
}
/// A raw escape sequence, or nothing without colours.
pub fn esc(code: &str) -> &str {
    if COLOR.load(Ordering::Relaxed) {
        code
    } else {
        ""
    }
}
pub fn color() -> bool {
    COLOR.load(Ordering::Relaxed)
}

/// Rewrites the current line (a status that changes in place).
pub fn status(s: &str) {
    if !color() {
        return;
    }
    let w = width().saturating_sub(1);
    let s: String = s.chars().take(w).collect();
    print!("\r\x1b[2K{}", dim(&s));
    let _ = io::stdout().flush();
}

/// A status line that brings its own colours.
pub fn status_raw(s: &str) {
    if !color() {
        return;
    }
    print!("\r\x1b[2K{}\x1b[0m", vis_cut(s, width().saturating_sub(1)));
    let _ = io::stdout().flush();
}

/// Columns a string takes, escape sequences left out (one per character).
pub fn vis_len(s: &str) -> usize {
    let mut n = 0;
    let mut esc = false;
    for c in s.chars() {
        if esc {
            esc = !c.is_ascii_alphabetic();
        } else if c == '\x1b' {
            esc = true;
        } else {
            n += 1;
        }
    }
    n
}

/// The first `w` columns of a string, its escape sequences kept (… when cut).
pub fn vis_cut(s: &str, w: usize) -> String {
    if vis_len(s) <= w {
        return s.to_string();
    }
    let mut out = String::new();
    let mut n = 0;
    let mut esc = false;
    for c in s.chars() {
        if esc {
            esc = !c.is_ascii_alphabetic();
            out.push(c);
        } else if c == '\x1b' {
            esc = true;
            out.push(c);
        } else if n + 1 < w {
            out.push(c);
            n += 1;
        }
    }
    out.push('…');
    out
}

/// A colour from the 256-colour palette.
pub fn c256(n: u8, s: &str) -> String {
    paint(&format!("38;5;{n}"), s)
}

// ---------- the startup animation ----------

/// PELICA in the "ANSI Shadow" letters, one letter per entry.
const LOGO: [[&str; 6]; 6] = [
    ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔═══╝ ", "██║     ", "╚═╝     "],
    ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
    ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
    [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
];
/// Pelica's orange, dark to light across the letters.
const RAMP: [u8; 6] = [166, 172, 173, 209, 215, 216];

/// One frame of the logo: columns up to `edge` shown, the few at the edge
/// lit up white; `edge` past the end draws it settled.
fn logo_frame(rows: &[Vec<char>], edge: isize) -> Vec<String> {
    let w = rows[0].len();
    rows.iter()
        .map(|r| {
            let mut line = String::from("  ");
            for (x, &ch) in r.iter().enumerate() {
                let d = edge - x as isize;
                if d < 0 || ch == ' ' {
                    line.push(' ');
                    continue;
                }
                let block = ch == '█';
                let col = if d < 3 && edge < w as isize + 3 {
                    if block { 231 } else { 223 }
                } else if block {
                    RAMP[x * (RAMP.len() - 1) / (w - 1)]
                } else {
                    // the shadow, darker
                    130
                };
                line.push_str(&format!("\x1b[38;5;{col}m{ch}"));
            }
            line.push_str("\x1b[0m");
            line
        })
        .collect()
}

/// The logo sweeping in from the left (about 0.6 s), then a line under it.
/// Skipped without colours, in a narrow window, or with PELICA_NO_INTRO set;
/// Ctrl+C jumps to the end.
pub fn intro(tagline: &str) {
    let rows: Vec<Vec<char>> = (0..6).map(|r| LOGO.iter().flat_map(|l| l[r].chars()).collect()).collect();
    let w = rows[0].len() as isize;
    if !color() || width() < 50 || std::env::var_os("PELICA_NO_INTRO").is_some() {
        if color() && width() >= 50 {
            println!();
            for l in logo_frame(&rows, w + 10) {
                println!("{l}");
            }
            println!("  {}\n", dim(tagline));
        }
        return;
    }
    BUSY.store(true, Ordering::SeqCst);
    print!("\x1b[?25l\n");
    let frames = 20;
    for f in 0..=frames {
        let edge = if f == frames || INTERRUPT.load(Ordering::SeqCst) { w + 10 } else { (f + 1) as isize * (w + 6) / frames as isize - 3 };
        if f > 0 {
            print!("\x1b[6A");
        }
        for l in logo_frame(&rows, edge) {
            print!("\r\x1b[2K{l}\n");
        }
        let _ = io::stdout().flush();
        if edge > w + 3 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(30));
    }
    // the line under it types itself out
    print!("  ");
    for (i, c) in tagline.chars().enumerate() {
        print!("{}", dim(&c.to_string()));
        let _ = io::stdout().flush();
        if !INTERRUPT.load(Ordering::SeqCst) && i % 2 == 0 {
            std::thread::sleep(std::time::Duration::from_millis(8));
        }
    }
    print!("\n\n\x1b[?25h");
    let _ = io::stdout().flush();
    BUSY.store(false, Ordering::SeqCst);
    INTERRUPT.store(false, Ordering::SeqCst);
}

/// Lines (already coloured) in a box with rounded corners, as wide as the
/// longest line allows in the window.
pub fn boxed(lines: &[String], edge: u8) {
    let inner = lines.iter().map(|l| vis_len(l)).max().unwrap_or(0).min(width().saturating_sub(6)).max(10);
    let bar = "─".repeat(inner + 2);
    println!("{}", c256(edge, &format!("╭{bar}╮")));
    for l in lines {
        let l = vis_cut(l, inner);
        println!("{} {l}{} {}", c256(edge, "│"), " ".repeat(inner - vis_len(&l)), c256(edge, "│"));
    }
    println!("{}", c256(edge, &format!("╰{bar}╯")));
}

/// Reads a message between two lines across the window, with `footer` (one
/// line: the mode, hints) under them. Once sent, the box makes way for the
/// message as it was written. Without colours: a plain "> " prompt.
pub fn prompt(footer: &str) -> Line {
    if !color() {
        return read_line(&format!("{} ", bold(">")));
    }
    let w = width();
    let rule = dim(&"─".repeat(w.saturating_sub(1)));
    // rule, prompt line, rule, footer; then back up to the prompt line
    print!("{rule}\n\n{rule}\n{}\x1b[2A\r", vis_cut(footer, w.saturating_sub(1)));
    let (line, rows) = read_lines(&format!("{} ", bold(">")), w);
    // back to the top rule, and everything below it goes
    print!("\x1b[{}A\r\x1b[J", rows + 1);
    if let Line::Text(t) = &line {
        for (i, l) in t.split('\n').enumerate() {
            println!("{}{}", dim(if i == 0 { "> " } else { "  " }), l);
        }
    }
    let _ = io::stdout().flush();
    line
}
pub fn clear_status() {
    if color() {
        print!("\r\x1b[2K");
        let _ = io::stdout().flush();
    }
}

pub enum Line {
    Text(String),
    /// Ctrl+C at the prompt
    Interrupted,
    /// Ctrl+Z / end of input
    Eof,
}

/// Reads a line after `prompt`. A line ending in a backslash goes on to the
/// next one (for multi-line messages).
pub fn read_line(prompt: &str) -> Line {
    let (line, _) = read_lines(prompt, width());
    if matches!(line, Line::Interrupted) {
        println!();
    }
    line
}

/// `read_line`, also counting the rows the input took on screen (`w`
/// columns wide), the cursor being on the row after them at the end.
fn read_lines(prompt: &str, w: usize) -> (Line, usize) {
    let mut all = String::new();
    let mut rows = 0;
    let mut first = true;
    loop {
        print!("{}", if first { prompt.to_string() } else { dim("… ") });
        let _ = io::stdout().flush();
        first = false;
        let mut s = String::new();
        INTERRUPT.store(false, Ordering::SeqCst);
        let r = read_stdin(&mut s);
        let typed = s.trim_end_matches(['\r', '\n']).chars().count() + 2;
        rows += typed.saturating_sub(1) / w.max(1) + 1;
        if INTERRUPT.swap(false, Ordering::SeqCst) {
            return (Line::Interrupted, rows);
        }
        match r {
            Ok(0) => return (if all.is_empty() { Line::Eof } else { Line::Text(all) }, rows),
            Err(_) => return (Line::Interrupted, rows),
            Ok(_) => {}
        }
        let s = s.trim_end_matches(['\r', '\n']);
        if let Some(head) = s.strip_suffix('\\') {
            all.push_str(head);
            all.push('\n');
            continue;
        }
        all.push_str(s);
        return (Line::Text(all), rows);
    }
}

fn read_stdin(s: &mut String) -> io::Result<usize> {
    READING.store(true, Ordering::SeqCst);
    let r = io::stdin().lock().read_line(s);
    READING.store(false, Ordering::SeqCst);
    r
}

/// Asks a question answered with one of `keys` (the first is the default on
/// Enter); Ctrl+C or end of input answers with the last. A Ctrl+C stays in
/// INTERRUPT, so the reply stops too.
pub fn ask(q: &str, keys: &[char]) -> char {
    // (keys typed while the model was writing mustn't answer it)
    #[cfg(windows)]
    // SAFETY: this process's own console input handle
    unsafe {
        win::FlushConsoleInputBuffer(win::GetStdHandle(win::STD_INPUT_HANDLE));
    }
    loop {
        print!("{q} ");
        let _ = io::stdout().flush();
        let mut s = String::new();
        if INTERRUPT.load(Ordering::SeqCst) {
            println!();
            return *keys.last().unwrap();
        }
        let r = read_stdin(&mut s);
        if INTERRUPT.load(Ordering::SeqCst) || matches!(r, Ok(0) | Err(_)) {
            println!();
            return *keys.last().unwrap();
        }
        let t = s.trim().to_lowercase();
        let Some(c) = t.chars().next() else { return keys[0] };
        if keys.contains(&c) {
            return c;
        }
    }
}
