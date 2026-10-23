"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.checkUrl = checkUrl;
exports.entities = entities;
exports.htmlText = htmlText;
exports.search = search;
exports.openPage = openPage;
// Web tools for the models: search (DuckDuckGo, then Bing) and page reading, with
// refusal of local and private addresses (a model must not reach the
// user's router, localhost services or the like).
const promises_1 = __importDefault(require("node:dns/promises"));
const node_net_1 = __importDefault(require("node:net"));
const electron_1 = require("electron");
const UA = `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/${process.versions.chrome} Safari/537.36`;
function isPrivateIp(ip) {
    if (node_net_1.default.isIPv4(ip)) {
        const [a, b] = ip.split(".").map(Number);
        return a === 10 || a === 127 || a === 0 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b < 32) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b < 128) || a >= 224;
    }
    const v = ip.toLowerCase();
    if (v.startsWith("::ffff:"))
        return isPrivateIp(v.slice(7));
    return v === "::1" || v === "::" || v.startsWith("fc") || v.startsWith("fd") || v.startsWith("fe8") || v.startsWith("fe9") || v.startsWith("fea") || v.startsWith("feb");
}
/** Throws unless `raw` is a public http(s) URL. */
async function checkUrl(raw) {
    let u;
    try {
        u = new URL(raw);
    }
    catch {
        throw new Error("not a valid URL");
    }
    if (u.protocol !== "http:" && u.protocol !== "https:")
        throw new Error("only http and https pages can be opened");
    const host = u.hostname.replace(/^\[|\]$/g, "").toLowerCase();
    if (!host || host === "localhost" || /\.(local|lan|internal|localhost|home|corp)$/.test(host) || !host.includes(".") && !node_net_1.default.isIP(host))
        throw new Error("local addresses can't be opened");
    if (node_net_1.default.isIP(host)) {
        if (isPrivateIp(host))
            throw new Error("private addresses can't be opened");
    }
    else if (/^[0-9.]+$/.test(host) || /^0x/i.test(host)) {
        throw new Error("numeric hosts can't be opened");
    }
    else {
        const addrs = await promises_1.default.lookup(host, { all: true }).catch(() => []);
        if (addrs.some((a) => isPrivateIp(a.address)))
            throw new Error("that name points at a private address");
    }
    return u;
}
// Requests go through Chromium's network stack in a session of their own
// (cookies kept between searches), so search engines see an ordinary
// browser rather than a script, which they often turn away. Every request
// of that session, redirects and page resources included, passes the same
// private-address check.
const okHosts = new Map();
let ses = null;
function webSession() {
    if (ses)
        return ses;
    ses = electron_1.session.fromPartition("persist:pelica-web");
    ses.setUserAgent(UA);
    ses.setPermissionRequestHandler((_wc, _p, cb) => cb(false));
    ses.on("will-download", (e) => e.preventDefault());
    ses.webRequest.onBeforeRequest((d, cb) => {
        if (/^(data|blob|about|devtools):/.test(d.url))
            return cb({});
        let host = "";
        try {
            host = new URL(d.url).host;
        }
        catch { }
        if (host && (okHosts.get(host) || 0) > Date.now())
            return cb({});
        checkUrl(d.url).then(() => {
            if (host)
                okHosts.set(host, Date.now() + 600_000);
            cb({});
        }, () => cb({ cancel: true }));
    });
    return ses;
}
async function get(url, init = {}, maxBytes = 4_000_000) {
    await checkUrl(url);
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 20000);
    try {
        const r = await webSession().fetch(url, { ...init, signal: ctl.signal, headers: { Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9", ...(init.headers || {}) } });
        if (!r.ok)
            throw new Error(`HTTP ${r.status}`);
        const type = r.headers.get("content-type") || "";
        const reader = r.body?.getReader();
        const chunks = [];
        let n = 0;
        while (reader) {
            const { done, value } = await reader.read();
            if (done)
                break;
            chunks.push(value);
            n += value.length;
            if (n > maxBytes) {
                ctl.abort();
                break;
            }
        }
        return { text: Buffer.concat(chunks.map((c) => Buffer.from(c))).toString("utf8"), status: r.status, type };
    }
    catch (e) {
        throw new Error(e?.name === "AbortError" ? "the page took too long to answer" : String(e?.message || e));
    }
    finally {
        clearTimeout(timer);
    }
}
/** Loads `url` in a hidden window of the web session and runs `script` there once the page has settled. */
async function rendered(url, script, settle = 1500) {
    await checkUrl(url);
    webSession();
    const w = new electron_1.BrowserWindow({ show: false, width: 1280, height: 900, webPreferences: { partition: "persist:pelica-web", sandbox: true, images: false, backgroundThrottling: false } });
    w.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
    w.webContents.setAudioMuted(true);
    try {
        await Promise.race([w.loadURL(url), new Promise((_, no) => setTimeout(() => no(new Error("the page took too long to load")), 25000))]);
        await new Promise((ok) => setTimeout(ok, settle));
        const value = (await w.webContents.executeJavaScript(script, true));
        return { value, url: w.webContents.getURL() };
    }
    finally {
        w.destroy();
    }
}
const ENT = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", mdash: "—", ndash: "–", hellip: "…", rsquo: "’", lsquo: "‘", rdquo: "”", ldquo: "“", copy: "©", reg: "®", trade: "™", middot: "·", bull: "•" };
function entities(s) {
    return s.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (m, e) => {
        if (e[0] === "#") {
            const c = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
            return Number.isFinite(c) && c > 0 && c < 0x110000 ? String.fromCodePoint(c) : m;
        }
        return ENT[e.toLowerCase()] ?? m;
    });
}
const strip = (s) => entities(s.replace(/<[^>]*>/g, "")).replace(/\s+/g, " ").trim();
/** Readable text of an HTML page: the main region, headings as ##, list items as -. */
function htmlText(html) {
    const title = strip((html.match(/<title[^>]*>([\s\S]*?)<\/title>/i) || [])[1] || "");
    let body = html;
    for (const tag of ["main", "article"]) {
        const m = body.match(new RegExp(`<${tag}[\\s>][\\s\\S]*?</${tag}>`, "i"));
        if (m && m[0].length > 500) {
            body = m[0];
            break;
        }
    }
    body = body
        .replace(/<!--[\s\S]*?-->/g, "")
        .replace(/<(script|style|noscript|svg|nav|footer|header|aside|form|iframe|template|button|select)[\s>][\s\S]*?<\/\1>/gi, "")
        .replace(/<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi, (_m, _l, t) => `\n\n## ${strip(t)}\n\n`)
        .replace(/<li[^>]*>/gi, "\n- ")
        .replace(/<(br|hr)\s*\/?>/gi, "\n")
        .replace(/<\/(p|div|section|tr|table|ul|ol|blockquote|pre|dd|dt)>/gi, "\n\n")
        .replace(/<\/t[dh]>/gi, " | ")
        .replace(/<[^>]+>/g, "");
    const text = entities(body)
        .split("\n")
        .map((l) => l.replace(/[ \t ]+/g, " ").trim())
        .join("\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
    return { title, text };
}
function realLink(href) {
    const h = entities(href);
    const m = h.match(/[?&]uddg=([^&]+)/);
    if (m)
        return decodeURIComponent(m[1]);
    // Bing's click-tracking links carry the target as base64 after "a1"
    const b = h.match(/bing\.com\/ck\/a\?.*?[?&]u=a1([^&]+)/);
    if (b) {
        try {
            return Buffer.from(b[1].replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
        }
        catch { }
    }
    return h.startsWith("//") ? "https:" + h : h;
}
function keep(hits, url, title, snippet) {
    if (!/^https?:/.test(url) || /duckduckgo\.com\/y\.js|bing\.com\/(aclick|ck\/a)/.test(url) || hits.some((h) => h.url === url))
        return;
    hits.push({ title, url, snippet });
}
function duckHtml(html) {
    const hits = [];
    for (const b of html.split(/<div class="result results_links/).slice(1)) {
        if (b.startsWith("_ads") || b.includes("result--ad"))
            continue;
        const a = b.match(/<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/);
        if (!a)
            continue;
        const sn = b.match(/class="result__snippet"[^>]*>([\s\S]*?)<\/a>/);
        keep(hits, realLink(a[1]), strip(a[2]), sn ? strip(sn[1]) : "");
    }
    return hits;
}
function bingHtml(html) {
    const hits = [];
    for (const b of html.split('<li class="b_algo"').slice(1)) {
        const a = b.match(/<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/);
        if (!a)
            continue;
        const sn = b.match(/<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>([\s\S]*?)<\/p>/) || b.match(/<div class="b_caption"[^>]*>[\s\S]*?<p[^>]*>([\s\S]*?)<\/p>/);
        keep(hits, realLink(a[1]), strip(a[2]), sn ? strip(sn[1]).replace(/^Web\s/, "") : "");
    }
    return hits;
}
const DUCK_DOM = `(async () => {
  for (let i = 0; i < 40 && !document.querySelector('[data-testid=result]'); i++) await new Promise((r) => setTimeout(r, 200));
  return [...document.querySelectorAll('[data-testid=result]')].map((el) => {
    const a = el.querySelector('[data-testid=result-title-a]');
    const s = el.querySelector('[data-result=snippet]');
    return { title: a ? a.textContent.trim() : '', url: a ? a.href : '', snippet: s ? s.textContent.trim() : '' };
  });
})()`;
/** Top results for `query`: DuckDuckGo, then Bing, then DuckDuckGo rendered in a hidden browser window. */
async function search(query) {
    const q = query.trim().slice(0, 400);
    const errors = [];
    const attempts = [
        [
            "DuckDuckGo",
            async () => {
                const r = await get("https://html.duckduckgo.com/html/", { method: "POST", body: new URLSearchParams({ q, kl: "us-en" }).toString(), headers: { "Content-Type": "application/x-www-form-urlencoded", Origin: "https://html.duckduckgo.com", Referer: "https://html.duckduckgo.com/" } });
                if (r.status === 202 || /anomaly-modal|challenge-form/.test(r.text))
                    throw new Error("blocked the request");
                return duckHtml(r.text);
            },
        ],
        ["Bing", async () => bingHtml((await get(`https://www.bing.com/search?q=${encodeURIComponent(q)}&setlang=en&form=QBLH`)).text)],
        [
            "DuckDuckGo in a browser window",
            async () => {
                const r = await rendered(`https://duckduckgo.com/?q=${encodeURIComponent(q)}&kl=us-en&ia=web`, DUCK_DOM, 300);
                const hits = [];
                for (const h of r.value || [])
                    keep(hits, h.url, h.title, h.snippet);
                return hits;
            },
        ],
    ];
    for (const [name, run] of attempts) {
        try {
            const hits = await run();
            if (hits.length)
                return hits.slice(0, 8);
            errors.push(`${name}: no results`);
        }
        catch (e) {
            errors.push(`${name}: ${e?.message || e}`);
        }
    }
    if (errors.every((e) => e.endsWith("no results")))
        return [];
    throw new Error(`every search engine failed (${errors.join("; ")})`);
}
const PAGE_DOM = `(() => {
  const root = document.querySelector('main, article, [role=main]') || document.body;
  return { title: document.title, text: root ? root.innerText : '' };
})()`;
/** A page as readable text. Pages built by scripts are rendered in a hidden browser window first. */
async function openPage(url, maxChars = 12000) {
    let title = "";
    let text = "";
    let final = url;
    let fetchErr = "";
    try {
        const r = await get(url);
        if (/html|xml/.test(r.type) || /^\s*</.test(r.text))
            ({ title, text } = htmlText(r.text));
        else
            text = r.text;
    }
    catch (e) {
        fetchErr = String(e?.message || e);
    }
    if (text.length < 400) {
        // little or no text: a page built by scripts, or a site that refused a plain request
        try {
            const r = await rendered(url, PAGE_DOM);
            if (r.value.text.trim().length > text.length) {
                title = r.value.title || title;
                text = r.value.text.replace(/\n{3,}/g, "\n\n").trim();
                final = r.url;
            }
        }
        catch (e) {
            if (!text)
                throw new Error(fetchErr || String(e?.message || e));
        }
    }
    if (!text && fetchErr)
        throw new Error(fetchErr);
    if (text.length > maxChars)
        text = text.slice(0, maxChars) + `\n\n[page cut at ${maxChars} characters]`;
    return { title, url: final, text };
}
