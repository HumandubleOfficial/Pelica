"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RELEASES_PAGE = exports.DEFAULT_FEED = void 0;
exports.older = older;
exports.check = check;
exports.signature = signature;
exports.download = download;
exports.tidy = tidy;
// Updates: Pelica asks an update feed what the newest version is: by default
// the latest.json of Pelica's newest GitHub release (DEFAULT_FEED), or the web
// address or file path set in Settings → About. The feed is the latest.json
// that installer/make.mjs writes beside each installer:
//   { "version": "0.13.0", "url": "Pelica-Setup-0.13.0.exe", "sha256": "…", "notes": [...] }
// `url` may be relative to the feed. An update is downloaded into the app's
// data folder, must match its sha256 (and, when this copy of Pelica is
// signed, carry a valid signature from the same publisher), and then runs
// silently (/S /RELAUNCH) after the app quits; the installer starts Pelica
// again when it's done. Nothing is downloaded until the user chooses Install.
const node_child_process_1 = require("node:child_process");
const node_crypto_1 = __importDefault(require("node:crypto"));
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
const node_url_1 = require("node:url");
/** the newest GitHub release's feed (each release carries its installer and latest.json) */
exports.DEFAULT_FEED = "https://github.com/HumandubleOfficial/Pelica/releases/latest/download/latest.json";
exports.RELEASES_PAGE = "https://github.com/HumandubleOfficial/Pelica/releases";
/** a < b, for versions like 0.12.0 */
function older(a, b) {
    const pa = a.split(/[.-]/).map((x) => parseInt(x, 10) || 0);
    const pb = b.split(/[.-]/).map((x) => parseInt(x, 10) || 0);
    for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
        if ((pa[i] || 0) !== (pb[i] || 0))
            return (pa[i] || 0) < (pb[i] || 0);
    }
    return false;
}
const isWeb = (s) => /^https?:\/\//i.test(s);
/** the feed as a URL (a plain path becomes file:) */
function feedUrl(feed) {
    const f = feed.trim();
    if (isWeb(f) || /^file:/i.test(f))
        return new URL(f);
    return (0, node_url_1.pathToFileURL)(node_path_1.default.resolve(f));
}
async function read(u, signal) {
    if (u.protocol === "file:")
        return node_fs_1.default.promises.readFile((0, node_url_1.fileURLToPath)(u));
    if (u.protocol !== "https:" && u.protocol !== "http:")
        throw new Error(`can't read ${u.protocol} addresses`);
    const r = await fetch(u, { signal, headers: { "Cache-Control": "no-cache" } });
    if (!r.ok)
        throw new Error(`${u.href}: ${r.status} ${r.statusText}`);
    return Buffer.from(await r.arrayBuffer());
}
async function check(feed, current) {
    if (!feed?.trim())
        return { current, newer: false, error: "No update feed is set." };
    try {
        const u = feedUrl(feed);
        const j = JSON.parse((await read(u, AbortSignal.timeout(15000))).toString("utf8"));
        if (typeof j.version !== "string" || typeof j.url !== "string" || !/^[0-9a-f]{64}$/i.test(j.sha256 || ""))
            throw new Error("the feed needs version, url and sha256");
        j.url = new URL(j.url, u).href;
        return { current, latest: j, newer: older(current, j.version) };
    }
    catch (e) {
        return { current, newer: false, error: `Couldn't read the update feed: ${e?.message || e}` };
    }
}
/** Windows' verdict on a file's signature: status and signer (PowerShell's Get-AuthenticodeSignature). */
function signature(file) {
    const ps = `$s = Get-AuthenticodeSignature -LiteralPath $env:PELICA_SIGFILE; Write-Output ([string]$s.Status); Write-Output ([string]$s.SignerCertificate.Subject)`;
    return new Promise((res) => {
        (0, node_child_process_1.execFile)("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", ps], { windowsHide: true, timeout: 30000, env: { ...process.env, PELICA_SIGFILE: file } }, (err, out) => {
            if (err)
                return res({ status: "Unknown", subject: "" });
            const [status = "", subject = ""] = String(out).split(/\r?\n/);
            res({ status: status.trim(), subject: subject.trim() });
        });
    });
}
/**
 * Downloads the update into `dir`, checks it, and returns the installer's path.
 * `self` is this app's exe: when it is signed, the update must be signed by the same publisher.
 */
async function download(f, dir, self, onProgress, signal) {
    node_fs_1.default.mkdirSync(dir, { recursive: true });
    const u = new URL(f.url);
    const name = node_path_1.default.basename(u.pathname).replace(/[^\w.-]/g, "_") || `Pelica-Setup-${f.version}.exe`;
    if (!/\.exe$/i.test(name))
        throw new Error("the update isn't an .exe");
    const out = node_path_1.default.join(dir, name);
    const part = out + ".part";
    const hash = node_crypto_1.default.createHash("sha256");
    if (u.protocol === "file:") {
        const src = (0, node_url_1.fileURLToPath)(u);
        const total = node_fs_1.default.statSync(src).size;
        await node_fs_1.default.promises.copyFile(src, part);
        hash.update(await node_fs_1.default.promises.readFile(part));
        onProgress({ got: total, total });
    }
    else {
        if (u.protocol !== "https:" && u.protocol !== "http:")
            throw new Error(`can't download ${u.protocol} addresses`);
        const r = await fetch(u, { signal });
        if (!r.ok || !r.body)
            throw new Error(`${u.href}: ${r.status} ${r.statusText}`);
        const total = Number(r.headers.get("content-length")) || 0;
        const w = node_fs_1.default.createWriteStream(part);
        let got = 0;
        try {
            for await (const chunk of r.body) {
                hash.update(chunk);
                got += chunk.length;
                if (!w.write(chunk))
                    await new Promise((r2) => w.once("drain", () => r2()));
                onProgress({ got, total });
            }
        }
        finally {
            await new Promise((r2) => w.end(() => r2()));
        }
    }
    const sum = hash.digest("hex");
    if (sum.toLowerCase() !== f.sha256.toLowerCase()) {
        node_fs_1.default.rmSync(part, { force: true });
        throw new Error("the download doesn't match the feed's sha256, so it wasn't run");
    }
    const mine = await signature(self);
    if (mine.status === "Valid") {
        const theirs = await signature(part);
        if (theirs.status !== "Valid" || theirs.subject !== mine.subject) {
            node_fs_1.default.rmSync(part, { force: true });
            throw new Error(`the update isn't signed by ${mine.subject} (${theirs.status}${theirs.subject ? ", " + theirs.subject : ""}), so it wasn't run`);
        }
    }
    node_fs_1.default.rmSync(out, { force: true });
    node_fs_1.default.renameSync(part, out);
    return out;
}
/** Old downloaded installers (not `keep`) go. */
function tidy(dir, keep) {
    try {
        for (const n of node_fs_1.default.readdirSync(dir)) {
            const p = node_path_1.default.join(dir, n);
            if (p !== keep && /^Pelica-Setup-.*\.exe(\.part)?$/i.test(n))
                node_fs_1.default.rmSync(p, { force: true });
        }
    }
    catch { }
}
