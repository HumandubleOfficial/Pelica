"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.init = init;
exports.words = words;
exports.index = index;
exports.info = info;
exports.search = search;
exports.forget = forget;
// A project's documents: the folders it searches. Pelica reads the files in
// them once (text and code, Word, PowerPoint and Excel through Windows' own
// tar.exe, PDFs page by page through Windows' PDF renderer and OCR), cuts the
// text into passages and keeps a word index on this PC. A question then finds
// the passages that match it best (BM25), and they go into the message with
// numbers the reply cites. Nothing leaves the PC and nothing is downloaded.
const node_child_process_1 = require("node:child_process");
const node_crypto_1 = __importDefault(require("node:crypto"));
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
const SYS = process.env.SystemRoot || "C:\\Windows";
const TAR = node_path_1.default.join(SYS, "System32", "tar.exe");
const PS = node_path_1.default.join(SYS, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
const TEXT = new Set("txt md markdown rst org csv tsv json jsonl yaml yml toml ini cfg log xml html htm tex srt vtt js jsx ts tsx py rs go java kt c h cpp hpp cs rb php swift lua sh ps1 bat sql r m scala dart vue svelte css scss".split(" "));
const OFFICE = new Set(["docx", "pptx", "xlsx", "odt"]);
const SKIP_DIRS = new Set(["node_modules", ".git", "target", "dist", "build", ".venv", "venv", "__pycache__", ".next", "$RECYCLE.BIN", "System Volume Information"]);
const MAX_FILES = 5000;
const MAX_TEXT_BYTES = 8 << 20;
const MAX_PDF_PAGES = 300;
/** a passage: about this many characters, cut at paragraph or sentence ends */
const PASSAGE = 900;
let dir = "";
const loaded = new Map();
const busy = new Map();
function init(userData) {
    dir = node_path_1.default.join(userData, "docs");
}
const idxPath = (folder) => node_path_1.default.join(dir, node_crypto_1.default.createHash("sha1").update(folder.toLowerCase()).digest("hex").slice(0, 16) + ".json");
function loadIndex(folder) {
    const had = loaded.get(folder);
    if (had)
        return had;
    let idx = { folder, at: 0, files: {} };
    try {
        const v = JSON.parse(node_fs_1.default.readFileSync(idxPath(folder), "utf8"));
        if (v && v.files)
            idx = v;
    }
    catch { }
    loaded.set(folder, idx);
    return idx;
}
// ---------- words ----------
const STOP = new Set("a an and are as at be but by for from has have he her his i in is it its of on or our she that the their them they this to was were what when where which who will with you your not do does did can could should would than then there these those so if into about over also".split(" "));
/** Lower-case words, a plural's "s" dropped, common words left out. */
function words(t) {
    const out = [];
    for (const m of t.toLowerCase().matchAll(/[\p{L}\p{N}][\p{L}\p{N}'_-]*/gu)) {
        let w = m[0].replace(/'s$/, "").replace(/['_-]+$/, "");
        if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss"))
            w = w.slice(0, -1);
        if (w.length > 1 && w.length < 40 && !STOP.has(w))
            out.push(w);
    }
    return out;
}
function chunk(file, text, page) {
    const out = [];
    const paras = text.replace(/\r/g, "").split(/\n\s*\n|(?<=[.!?])\s+(?=[A-Z])/);
    let cur = "";
    const push = () => {
        const t = cur.replace(/[ \t]+/g, " ").trim();
        cur = "";
        if (t.length < 20)
            return;
        const ws = words(t);
        const tf = {};
        for (const w of ws)
            tf[w] = (tf[w] || 0) + 1;
        out.push({ file, page, text: t, tf, len: ws.length });
    };
    for (const p of paras) {
        if (cur && cur.length + p.length > PASSAGE)
            push();
        cur += (cur ? "\n" : "") + p;
        while (cur.length > PASSAGE * 1.6) {
            const cut = cur.lastIndexOf(" ", PASSAGE);
            const head = cur.slice(0, cut > 0 ? cut : PASSAGE);
            const rest = cur.slice(head.length);
            cur = head;
            push();
            cur = rest;
        }
    }
    push();
    return out;
}
// ---------- reading files ----------
function run(exe, args, timeoutMs, onLine) {
    return new Promise((resolve, reject) => {
        const p = (0, node_child_process_1.spawn)(exe, args, { windowsHide: true });
        const out = [];
        let err = "";
        let buf = "";
        p.stdout.on("data", (d) => {
            out.push(d);
            if (onLine) {
                buf += d.toString("utf8");
                let i;
                while ((i = buf.indexOf("\n")) >= 0) {
                    onLine(buf.slice(0, i).replace(/\r$/, ""));
                    buf = buf.slice(i + 1);
                }
            }
        });
        p.stderr.on("data", (d) => (err += d));
        const t = setTimeout(() => p.kill(), timeoutMs);
        p.on("error", reject);
        p.on("close", (code) => {
            clearTimeout(t);
            if (code === 0)
                resolve(Buffer.concat(out).toString("utf8"));
            else
                reject(new Error(err.replace(/\r/g, "").split("\n").find((l) => l.trim()) || `exit ${code}`));
        });
    });
}
const unxml = (x) => x
    .replace(/<\/(w:p|a:p|text:p|si|row)>/g, "\n")
    .replace(/<(w:tab|w:br)\/>/g, " ")
    .replace(/<[^>]+>/g, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_m, n) => String.fromCodePoint(+n))
    .replace(/&amp;/g, "&");
const zipEntry = (file, name) => run(TAR, ["-xOf", file, name], 30000);
async function office(file, ext) {
    if (ext === "docx")
        return [{ text: unxml(await zipEntry(file, "word/document.xml")) }];
    if (ext === "odt")
        return [{ text: unxml(await zipEntry(file, "content.xml")) }];
    if (ext === "xlsx")
        return [{ text: unxml(await zipEntry(file, "xl/sharedStrings.xml").catch(() => "")) }];
    // pptx: a slide at a time, in order
    const names = (await run(TAR, ["-tf", file], 30000))
        .split(/\r?\n/)
        .map((n) => /^ppt\/slides\/slide(\d+)\.xml$/.exec(n))
        .filter((m) => !!m)
        .sort((a, b) => +a[1] - +b[1]);
    const out = [];
    for (const m of names)
        out.push({ text: unxml(await zipEntry(file, m[0])), page: +m[1] });
    return out;
}
const PDF = `
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$m = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 }
$op = ($m | Where-Object { $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation\`1' })[0]
$act = ($m | Where-Object { $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' })[0]
function Await($o, [Type]$t) { $k = $op.MakeGenericMethod($t).Invoke($null, @($o)); $k.Wait(-1) | Out-Null; $k.Result }
function AwaitA($o) { $act.Invoke($null, @($o)).Wait(-1) | Out-Null }
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime] | Out-Null
$eng = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if (-not $eng) { throw 'No OCR language is installed in Windows (Settings > Time & language > Language).' }
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
$doc = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
[Console]::Out.WriteLine('PAGES ' + $doc.PageCount); [Console]::Out.Flush()
$n = [Math]::Min([int]$doc.PageCount, $max)
$side = [Math]::Min(1800, [Windows.Media.Ocr.OcrEngine]::MaxImageDimension)
for ($i = 0; $i -lt $n; $i++) {
  $page = $doc.GetPage($i)
  $s = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
  $o = New-Object Windows.Data.Pdf.PdfPageRenderOptions
  $k = $side / [Math]::Max($page.Size.Width, $page.Size.Height)
  $o.DestinationWidth = [uint32]($page.Size.Width * $k)
  $o.DestinationHeight = [uint32]($page.Size.Height * $k)
  AwaitA ($page.RenderToStreamAsync($s, $o))
  $dec = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($s)) ([Windows.Graphics.Imaging.BitmapDecoder])
  $bmp = Await ($dec.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
  $res = Await ($eng.RecognizeAsync($bmp)) ([Windows.Media.Ocr.OcrResult])
  [Console]::Out.WriteLine('PAGE ' + ($i + 1))
  foreach ($l in $res.Lines) { [Console]::Out.WriteLine($l.Text) }
  [Console]::Out.Flush()
  $bmp.Dispose(); $s.Dispose(); $page.Dispose()
}
`;
async function pdf(file, onPage) {
    const pages = [];
    let of = 0;
    const script = `$path = '${file.replace(/'/g, "''")}'\n$max = ${MAX_PDF_PAGES}\n${PDF}`;
    await run(PS, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", Buffer.from(script, "utf16le").toString("base64")], 30 * 60 * 1000, (l) => {
        if (l.startsWith("PAGES "))
            of = Math.min(+l.slice(6), MAX_PDF_PAGES);
        else if (/^PAGE \d+$/.test(l)) {
            pages.push({ text: "", page: +l.slice(5) });
            onPage(pages.length, of);
        }
        else if (pages.length)
            pages[pages.length - 1].text += l + "\n";
    });
    return pages;
}
const strip = (html) => unxml(html.replace(/<(script|style)[\s\S]*?<\/\1>/gi, "").replace(/<\/(p|div|li|h\d|tr|br)>/gi, "\n"));
async function read(file, onPage) {
    const ext = node_path_1.default.extname(file).slice(1).toLowerCase();
    if (ext === "pdf")
        return pdf(file, onPage);
    if (OFFICE.has(ext))
        return office(file, ext);
    const raw = node_fs_1.default.readFileSync(file, "utf8");
    return [{ text: ext === "html" || ext === "htm" ? strip(raw) : raw }];
}
const readable = (f) => {
    const ext = node_path_1.default.extname(f).slice(1).toLowerCase();
    return ext === "pdf" || OFFICE.has(ext) || TEXT.has(ext);
};
function walk(folder) {
    const out = [];
    const todo = [folder];
    while (todo.length && out.length < MAX_FILES) {
        const d = todo.pop();
        let ents = [];
        try {
            ents = node_fs_1.default.readdirSync(d, { withFileTypes: true });
        }
        catch {
            continue;
        }
        for (const e of ents) {
            if (e.name.startsWith(".") && e.name !== ".")
                continue;
            const p = node_path_1.default.join(d, e.name);
            if (e.isDirectory()) {
                if (!SKIP_DIRS.has(e.name))
                    todo.push(p);
            }
            else if (e.isFile() && readable(p)) {
                try {
                    const st = node_fs_1.default.statSync(p);
                    const ext = node_path_1.default.extname(p).slice(1).toLowerCase();
                    if (TEXT.has(ext) && st.size > MAX_TEXT_BYTES)
                        continue;
                    out.push({ file: p, mtime: st.mtimeMs, size: st.size });
                }
                catch { }
            }
        }
    }
    return out;
}
/** Reads what's new or changed in the folder (files already read stay as they are). */
function index(folder, progress) {
    const had = busy.get(folder);
    if (had)
        return had;
    const job = (async () => {
        const idx = loadIndex(folder);
        const now = walk(folder);
        const keep = new Set(now.map((f) => f.file));
        for (const f of Object.keys(idx.files))
            if (!keep.has(f))
                delete idx.files[f];
        const todo = now.filter((f) => {
            const e = idx.files[f.file];
            return !e || e.mtime !== f.mtime || e.size !== f.size;
        });
        let done = 0;
        for (const f of todo) {
            const name = node_path_1.default.relative(folder, f.file);
            progress({ folder, done, total: todo.length, file: name });
            try {
                const parts = await read(f.file, (n, of) => progress({ folder, done, total: todo.length, file: `${name} (page ${n} of ${of})` }));
                idx.files[f.file] = { mtime: f.mtime, size: f.size, chunks: parts.flatMap((p) => chunk(f.file, p.text, p.page)) };
            }
            catch (e) {
                idx.files[f.file] = { mtime: f.mtime, size: f.size, chunks: [], error: String(e?.message || e).slice(0, 200) };
            }
            done++;
        }
        idx.at = Date.now();
        node_fs_1.default.mkdirSync(dir, { recursive: true });
        node_fs_1.default.writeFileSync(idxPath(folder), JSON.stringify(idx));
        progress({ folder, done, total: todo.length, file: "" });
        return idx;
    })().finally(() => busy.delete(folder));
    busy.set(folder, job);
    return job;
}
function info(folder) {
    const idx = loadIndex(folder);
    const files = Object.values(idx.files);
    return {
        folder,
        at: idx.at,
        files: files.length,
        passages: files.reduce((n, f) => n + f.chunks.length, 0),
        unreadable: Object.entries(idx.files)
            .filter(([, f]) => f.error)
            .map(([p, f]) => ({ file: node_path_1.default.relative(folder, p), error: f.error }))
            .slice(0, 20),
        busy: busy.has(folder),
    };
}
/** The `k` passages in these folders that match `query` best. */
function search(folders, query, k = 6) {
    const q = [...new Set(words(query))];
    if (!q.length)
        return [];
    const all = [];
    for (const f of folders)
        for (const e of Object.values(loadIndex(f).files))
            all.push(...e.chunks);
    if (!all.length)
        return [];
    const n = all.length;
    const avg = all.reduce((s, c) => s + c.len, 0) / n || 1;
    const idf = {};
    for (const w of q) {
        const df = all.reduce((s, c) => s + (c.tf[w] ? 1 : 0), 0);
        idf[w] = Math.log(1 + (n - df + 0.5) / (df + 0.5));
    }
    const K1 = 1.4;
    const B = 0.75;
    const scored = all.map((c) => {
        let s = 0;
        for (const w of q) {
            const f = c.tf[w];
            if (f)
                s += (idf[w] * f * (K1 + 1)) / (f + K1 * (1 - B + (B * c.len) / avg));
        }
        return { c, s };
    });
    scored.sort((a, b) => b.s - a.s);
    const best = scored[0]?.s || 0;
    // (passages far weaker than the best one are noise)
    return scored
        .filter((x) => x.s > 0 && x.s >= best * 0.25)
        .slice(0, k)
        .map(({ c, s }) => ({ file: c.file, page: c.page, text: c.text, score: Math.round(s * 100) / 100 }));
}
/** Forgets a folder's index (removing it from a project). */
function forget(folder) {
    loaded.delete(folder);
    node_fs_1.default.rm(idxPath(folder), { force: true }, () => { });
}
