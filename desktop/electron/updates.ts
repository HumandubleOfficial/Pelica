// Updates: Pelica asks an update feed what the newest version is: by default
// the latest.json of Pelica's newest GitHub release (DEFAULT_FEED), or the web
// address or file path set in Settings → Updates. The feed is the latest.json
// that installer/make.mjs writes beside each installer:
//   { "version": "0.13.0", "url": "Pelica-Setup-0.13.0.exe", "sha256": "…", "notes": [...] }
// `url` may be relative to the feed. An update is downloaded into the app's
// data folder, must match its sha256 (and, when this copy of Pelica is
// signed, carry a valid signature from the same publisher), and then runs
// silently (/S /RELAUNCH) after the app quits; the installer starts Pelica
// again when it's done. Nothing is downloaded until the user chooses Install.
import { execFile } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/** the newest GitHub release's feed (each release carries its installer and latest.json) */
export const DEFAULT_FEED = "https://github.com/HumandubleOfficial/Pelica/releases/latest/download/latest.json";
export const RELEASES_PAGE = "https://github.com/HumandubleOfficial/Pelica/releases";
/** every release, previews (pre-releases) included, newest first */
const RELEASES_API = "https://api.github.com/repos/HumandubleOfficial/Pelica/releases?per_page=20";

export interface Feed {
  version: string;
  url: string;
  sha256: string;
  notes?: string[];
  title?: string;
}

export interface Check {
  current: string;
  latest?: Feed;
  newer: boolean;
  /** the offered version is a preview (a GitHub pre-release) */
  preview?: boolean;
  error?: string;
}

/** a < b, for versions like 0.12.0 */
export function older(a: string, b: string): boolean {
  const pa = a.split(/[.-]/).map((x) => parseInt(x, 10) || 0);
  const pb = b.split(/[.-]/).map((x) => parseInt(x, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    if ((pa[i] || 0) !== (pb[i] || 0)) return (pa[i] || 0) < (pb[i] || 0);
  }
  return false;
}

const isWeb = (s: string) => /^https?:\/\//i.test(s);

/** the feed as a URL (a plain path becomes file:) */
function feedUrl(feed: string): URL {
  const f = feed.trim();
  if (isWeb(f) || /^file:/i.test(f)) return new URL(f);
  return pathToFileURL(path.resolve(f));
}

async function read(u: URL, signal?: AbortSignal): Promise<Buffer> {
  if (u.protocol === "file:") return fs.promises.readFile(fileURLToPath(u));
  if (u.protocol !== "https:" && u.protocol !== "http:") throw new Error(`can't read ${u.protocol} addresses`);
  const r = await fetch(u, { signal, headers: { "Cache-Control": "no-cache" } });
  if (!r.ok) throw new Error(`${u.href}: ${r.status} ${r.statusText}`);
  return Buffer.from(await r.arrayBuffer());
}

/** With previews: the newest release on GitHub that carries a latest.json, pre-releases included. */
async function previewFeed(signal: AbortSignal): Promise<{ url: URL; preview: boolean }> {
  const r = await fetch(RELEASES_API, { signal, headers: { Accept: "application/vnd.github+json" } });
  if (!r.ok) throw new Error(`GitHub: ${r.status} ${r.statusText}`);
  for (const rel of (await r.json()) as any[]) {
    if (rel.draft) continue;
    const a = (rel.assets || []).find((x: any) => x.name === "latest.json");
    if (a) return { url: new URL(a.browser_download_url), preview: !!rel.prerelease };
  }
  throw new Error("no GitHub release has a latest.json");
}

/** What the feed offers. `preview`: with the GitHub feed, also pre-releases. */
export async function check(feed: string, current: string, opts: { preview?: boolean } = {}): Promise<Check> {
  if (!feed?.trim()) return { current, newer: false, error: "No update feed is set." };
  try {
    let u = feedUrl(feed);
    let preview = false;
    if (opts.preview && feed === DEFAULT_FEED) ({ url: u, preview } = await previewFeed(AbortSignal.timeout(15000)));
    const j = JSON.parse((await read(u, AbortSignal.timeout(15000))).toString("utf8")) as Feed;
    if (typeof j.version !== "string" || typeof j.url !== "string" || !/^[0-9a-f]{64}$/i.test(j.sha256 || "")) throw new Error("the feed needs version, url and sha256");
    j.url = new URL(j.url, u).href;
    return { current, latest: j, newer: older(current, j.version), preview };
  } catch (e: any) {
    return { current, newer: false, error: `Couldn't read the update feed: ${e?.message || e}` };
  }
}

/** Windows' verdict on a file's signature: status and signer (PowerShell's Get-AuthenticodeSignature). */
export function signature(file: string): Promise<{ status: string; subject: string }> {
  const ps = `$s = Get-AuthenticodeSignature -LiteralPath $env:PELICA_SIGFILE; Write-Output ([string]$s.Status); Write-Output ([string]$s.SignerCertificate.Subject)`;
  return new Promise((res) => {
    execFile("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", ps], { windowsHide: true, timeout: 30000, env: { ...process.env, PELICA_SIGFILE: file } }, (err, out) => {
      if (err) return res({ status: "Unknown", subject: "" });
      const [status = "", subject = ""] = String(out).split(/\r?\n/);
      res({ status: status.trim(), subject: subject.trim() });
    });
  });
}

export interface Progress {
  got: number;
  total: number;
}

/**
 * Downloads the update into `dir`, checks it, and returns the installer's path.
 * `self` is this app's exe: when it is signed, the update must be signed by the same publisher.
 */
export async function download(f: Feed, dir: string, self: string, onProgress: (p: Progress) => void, signal?: AbortSignal): Promise<string> {
  fs.mkdirSync(dir, { recursive: true });
  const u = new URL(f.url);
  const name = path.basename(u.pathname).replace(/[^\w.-]/g, "_") || `Pelica-Setup-${f.version}.exe`;
  if (!/\.exe$/i.test(name)) throw new Error("the update isn't an .exe");
  const out = path.join(dir, name);
  const part = out + ".part";
  const hash = crypto.createHash("sha256");
  if (u.protocol === "file:") {
    const src = fileURLToPath(u);
    const total = fs.statSync(src).size;
    await fs.promises.copyFile(src, part);
    hash.update(await fs.promises.readFile(part));
    onProgress({ got: total, total });
  } else {
    if (u.protocol !== "https:" && u.protocol !== "http:") throw new Error(`can't download ${u.protocol} addresses`);
    const r = await fetch(u, { signal });
    if (!r.ok || !r.body) throw new Error(`${u.href}: ${r.status} ${r.statusText}`);
    const total = Number(r.headers.get("content-length")) || 0;
    const w = fs.createWriteStream(part);
    let got = 0;
    try {
      for await (const chunk of r.body as any as AsyncIterable<Uint8Array>) {
        hash.update(chunk);
        got += chunk.length;
        if (!w.write(chunk)) await new Promise<void>((r2) => w.once("drain", () => r2()));
        onProgress({ got, total });
      }
    } finally {
      await new Promise<void>((r2) => w.end(() => r2()));
    }
  }
  const sum = hash.digest("hex");
  if (sum.toLowerCase() !== f.sha256.toLowerCase()) {
    fs.rmSync(part, { force: true });
    throw new Error("the download doesn't match the feed's sha256, so it wasn't run");
  }
  const mine = await signature(self);
  if (mine.status === "Valid") {
    const theirs = await signature(part);
    if (theirs.status !== "Valid" || theirs.subject !== mine.subject) {
      fs.rmSync(part, { force: true });
      throw new Error(`the update isn't signed by ${mine.subject} (${theirs.status}${theirs.subject ? ", " + theirs.subject : ""}), so it wasn't run`);
    }
  }
  fs.rmSync(out, { force: true });
  fs.renameSync(part, out);
  return out;
}

/** Old downloaded installers (not `keep`) go. */
export function tidy(dir: string, keep?: string) {
  try {
    for (const n of fs.readdirSync(dir)) {
      const p = path.join(dir, n);
      if (p !== keep && /^Pelica-Setup-.*\.exe(\.part)?$/i.test(n)) fs.rmSync(p, { force: true });
    }
  } catch {}
}
