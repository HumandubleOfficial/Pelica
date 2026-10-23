//! Web tools for the chat app: a DuckDuckGo search (its plain HTML results
//! page) and reading a page as text. Requests go through `curl`, which
//! ships with Windows 10 and later and with every Unix, so Pelica needs no
//! TLS stack of its own.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::process::Command;

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36";

/// The tools as the model sees them (OpenAI-style function schemas, which
/// chat templates print into the system prompt).
pub fn tool_specs() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web (DuckDuckGo). Returns the top results: title, link and a snippet. Use it for current events, recent facts, or anything you are not sure of.",
                "parameters": {
                    "type": "object",
                    "properties": { "query": { "type": "string", "description": "What to search for, as you would type it into a search engine" } },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "open_page",
                "description": "Read a web page: returns its title and main text (long pages are cut). Use it on links from web_search results to get the details before answering.",
                "parameters": {
                    "type": "object",
                    "properties": { "url": { "type": "string", "description": "The page's full http(s) link" } },
                    "required": ["url"]
                }
            }
        }
    ])
}

pub struct Hit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

pub struct Page {
    pub url: String,
    pub title: String,
    pub text: String,
}

/// Fetches `url` with curl: (body, final URL, content type).
fn fetch(url: &str, post: Option<&str>) -> Result<(Vec<u8>, String, String)> {
    let mut c = Command::new("curl");
    c.args(["-sS", "-L", "--compressed", "--max-time", "20", "--max-redirs", "5", "--max-filesize", "4000000", "--proto", "=http,https", "--proto-redir", "=http,https", "-A", UA]);
    c.args(["-H", "Accept-Language: en-US,en;q=0.8"]);
    if let Some(q) = post {
        c.args(["--data-urlencode", &format!("q={q}")]);
    }
    // the status, final URL and type after the body, on a line of their own
    c.args(["-w", "\n\u{1}%{http_code} %{url_effective} %{content_type}"]);
    c.arg("--").arg(url);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = c.output().context("running curl (needed for web search)")?;
    if !out.status.success() {
        bail!("{}", String::from_utf8_lossy(&out.stderr).trim().trim_start_matches("curl: "));
    }
    let b = out.stdout;
    let at = b.iter().rposition(|&x| x == 1).context("unexpected curl output")?;
    let tail = String::from_utf8_lossy(&b[at + 1..]).to_string();
    let mut it = tail.splitn(3, ' ');
    let (code, eff, ty) = (it.next().unwrap_or(""), it.next().unwrap_or(url).to_string(), it.next().unwrap_or("").to_string());
    if !code.starts_with('2') {
        bail!("the site answered HTTP {code}");
    }
    let body = b[..at.saturating_sub(1)].to_vec();
    Ok((body, eff, ty))
}

/// Refuses links to this computer or the local network: the model picks
/// the links, and pages can suggest any.
fn check_url(url: &str) -> Result<()> {
    let rest = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://")).context("only http(s) links can be opened")?;
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = host.rsplit('@').next().unwrap_or(host);
    let host = if host.starts_with('[') { host.split(']').next().unwrap_or("").trim_start_matches('[') } else { host.split(':').next().unwrap_or("") }.to_ascii_lowercase();
    let local = host.is_empty() || host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") || host.ends_with(".lan") || host.ends_with(".internal") || !host.contains('.') && !host.contains(':');
    // (numeric forms such as 127.1 or 0x7f.0.0.1 also reach this computer)
    let numeric = host.chars().all(|c| c.is_ascii_digit() || c == '.') || host.split('.').any(|p| p.starts_with("0x"));
    let private = match host.parse::<std::net::IpAddr>() {
        Ok(ip) => is_private(ip),
        // a name: where it points now
        Err(_) if !local && !numeric => {
            use std::net::ToSocketAddrs;
            (host.as_str(), 443).to_socket_addrs().map(|mut a| a.any(|x| is_private(x.ip()))).unwrap_or(false)
        }
        Err(_) => false,
    };
    if local || private || numeric {
        bail!("links to this computer or the local network can't be opened");
    }
    Ok(())
}

fn is_private(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(a) => a.is_private() || a.is_loopback() || a.is_link_local() || a.is_unspecified() || a.octets()[0] == 100 && (a.octets()[1] & 0xc0) == 64,
        std::net::IpAddr::V6(a) => {
            a.is_loopback() || a.is_unspecified() || (a.segments()[0] & 0xfe00) == 0xfc00 || (a.segments()[0] & 0xffc0) == 0xfe80 || a.to_ipv4_mapped().is_some_and(|v| is_private(std::net::IpAddr::V4(v)))
        }
    }
}

/// The top results of a DuckDuckGo search (ads left out).
pub fn search(query: &str) -> Result<Vec<Hit>> {
    let (body, _, _) = fetch("https://html.duckduckgo.com/html/", Some(query))?;
    let s = String::from_utf8_lossy(&body);
    let mut hits = Vec::new();
    for block in s.split("<div class=\"result ").skip(1) {
        if block.starts_with("results_links") && !block.contains("result--ad") {
            let Some(a) = block.find("class=\"result__a\"") else { continue };
            let href = attr(&block[a..], "href").unwrap_or_default();
            let title = text_of(&block[a..], "</a>");
            let snippet = block.find("class=\"result__snippet\"").map(|i| text_of(&block[i..], "</a>")).unwrap_or_default();
            let url = real_link(&href);
            if url.starts_with("http") && !title.is_empty() {
                hits.push(Hit { title, url, snippet });
            }
        }
        if hits.len() == 8 {
            break;
        }
    }
    if hits.is_empty() {
        if s.contains("anomaly") || s.contains("challenge") {
            bail!("DuckDuckGo turned the search away (it sometimes blocks automated searches); try again in a minute");
        }
        bail!("no results");
    }
    Ok(hits)
}

/// The value of attribute `name` in the tag that starts `s`.
fn attr(s: &str, name: &str) -> Option<String> {
    let end = s.find('>')?;
    let tag = &s[..end];
    let i = tag.find(&format!("{name}=\""))? + name.len() + 2;
    let j = tag[i..].find('"')? + i;
    Some(entities(&tag[i..j]))
}

/// The text between the tag that starts `s` and `close`.
fn text_of(s: &str, close: &str) -> String {
    let Some(start) = s.find('>') else { return String::new() };
    let end = s[start..].find(close).map(|j| start + j).unwrap_or(s.len());
    squash(&entities(&strip_tags(&s[start + 1..end])))
}

/// DuckDuckGo sometimes wraps results in a redirect (`/l/?uddg=<link>`).
fn real_link(href: &str) -> String {
    if let Some(i) = href.find("uddg=") {
        let v = &href[i + 5..];
        let v = v.split('&').next().unwrap_or(v);
        return percent_decode(v);
    }
    if href.starts_with("//") {
        return format!("https:{href}");
    }
    href.to_string()
}

pub fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A page's title and readable text, at most `max_chars` of it.
pub fn open(url: &str, max_chars: usize) -> Result<Page> {
    check_url(url)?;
    let (body, eff, ty) = fetch(url, None)?;
    check_url(&eff)?;
    let ty = ty.to_ascii_lowercase();
    let raw = String::from_utf8_lossy(&body);
    let (title, mut text) = if ty.contains("html") || ty.is_empty() && raw.trim_start().starts_with('<') {
        let title = raw.find("<title").map(|i| text_of(&raw[i..], "</title>")).unwrap_or_default();
        (title, html_text(&raw))
    } else if ty.starts_with("text/") || ty.contains("json") || ty.contains("xml") {
        (String::new(), raw.to_string())
    } else {
        bail!("this link is a {} file, not a page Pelica can read", ty.split(';').next().unwrap_or("binary"));
    };
    if text.chars().count() > max_chars {
        let cut: String = text.chars().take(max_chars).collect();
        text = format!("{cut}\n[... the rest of the page is cut]");
    }
    if text.trim().is_empty() {
        bail!("the page has no readable text (it may need JavaScript)");
    }
    Ok(Page { url: eff, title, text })
}

/// Readable text of an HTML page: scripts, styles and page furniture
/// dropped, blocks on lines of their own, links' text kept.
pub fn html_text(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    // the main part, when the page marks one
    let (mut from, mut to) = (0, html.len());
    for (open, close) in [("<main", "</main>"), ("<article", "</article>")] {
        if let (Some(a), Some(b)) = (lower.find(open), lower.rfind(close)) {
            if b > a && b - a > 500 {
                (from, to) = (a, b);
                break;
            }
        }
    }
    if from == 0 {
        if let Some(a) = lower.find("<body") {
            from = a;
        }
    }
    let (html, lower) = (&html[from..to], &lower[from..to]);
    let skip = ["script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "template", "aside", "button", "select"];
    let blocks = ["p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "ul", "ol", "pre", "blockquote", "dd", "dt", "hr"];
    let mut out = String::with_capacity(html.len() / 4);
    let mut i = 0;
    let b = html.as_bytes();
    while i < b.len() {
        if b[i] == b'<' {
            let Some(j) = html[i..].find('>').map(|j| i + j) else { break };
            let tag = &lower[i + 1..j];
            let name: String = tag.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
            if tag.starts_with("!--") {
                i = lower[i..].find("-->").map(|k| i + k + 3).unwrap_or(b.len());
                continue;
            }
            if !tag.starts_with('/') && skip.contains(&name.as_str()) && !tag.ends_with('/') {
                let close = format!("</{name}");
                i = lower[j..].find(&close).and_then(|k| lower[j + k..].find('>').map(|m| j + k + m + 1)).unwrap_or(b.len());
                out.push('\n');
                continue;
            }
            if blocks.contains(&name.as_str()) {
                out.push('\n');
                if name.starts_with('h') && name.len() == 2 && !tag.starts_with('/') {
                    out.push_str("## ");
                } else if name == "li" && !tag.starts_with('/') {
                    out.push_str("- ");
                }
            } else if name == "td" || name == "th" {
                out.push_str(" | ");
            }
            i = j + 1;
        } else {
            let j = html[i..].find('<').map(|j| i + j).unwrap_or(b.len());
            out.push_str(&entities(&html[i..j]));
            i = j;
        }
    }
    // collapse runs of spaces and blank lines
    let mut text = String::with_capacity(out.len());
    let mut blank = 0;
    for line in out.lines() {
        let l = squash(line);
        if l.is_empty() || l == "-" || l == "##" {
            blank += 1;
            continue;
        }
        if !text.is_empty() {
            // (a closing tag and the next opening one leave one blank line
            // between any two blocks: only more is a paragraph break)
            text.push_str(if blank > 1 { "\n\n" } else { "\n" });
        }
        blank = 0;
        text.push_str(&l);
    }
    text
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// HTML character references: the named ones that matter, and numeric ones.
fn entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let end = rest[1..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '#')).map(|j| j + 1).unwrap_or(rest.len());
        let name = &rest[1..end];
        let semi = rest[end..].starts_with(';');
        let ch = match name {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            "ndash" => Some('–'),
            "mdash" => Some('—'),
            "hellip" => Some('…'),
            "rsquo" => Some('’'),
            "lsquo" => Some('‘'),
            "rdquo" => Some('”'),
            "ldquo" => Some('“'),
            "copy" => Some('©'),
            _ if name.starts_with("#x") || name.starts_with("#X") => u32::from_str_radix(&name[2..], 16).ok().and_then(char::from_u32),
            _ if name.starts_with('#') => name[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match ch {
            Some(c) => {
                out.push(c);
                rest = &rest[end + semi as usize..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls() {
        assert!(check_url("https://example.com/a").is_ok());
        for bad in ["http://localhost:8137/api/status", "http://127.0.0.1/", "http://192.168.1.1/", "http://[::1]/", "http://10.0.0.5:80/x", "file:///c:/x", "http://router/", "http://2130706433/", "http://user@127.0.0.1/"] {
            assert!(check_url(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn text() {
        let h = "<html><head><title>T &amp; U</title><script>var x=1;</script></head><body><nav>menu</nav><h1>Head</h1><p>One&nbsp;two <a href=x>link</a>.</p><ul><li>a</li><li>b</li></ul></body></html>";
        assert_eq!(html_text(h), "## Head\nOne two link.\n\n- a\n- b");
        assert_eq!(real_link("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.org%2Fx%3Fy%3D1&rut=z"), "https://a.org/x?y=1");
        assert_eq!(entities("&#x41;&#66;&lt;"), "AB<");
    }

    /// Needs the network: `cargo test web::tests::live -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live() {
        let hits = search("rust programming language latest version").unwrap();
        for h in &hits {
            println!("{} | {} | {}", h.title, h.url, h.snippet);
        }
        let p = open(&hits[0].url, 3000).unwrap();
        println!("\n{} ({})\n{}", p.title, p.url, p.text);
    }
}
