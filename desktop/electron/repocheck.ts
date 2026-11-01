// A safety check for GitHub repositories a model is about to clone, install
// or read: what GitHub says about the repo and its owner (taken down, age,
// stars), and what is in it that runs code on install or on opening
// (package.json install scripts, setup.py, VS Code folder-open tasks,
// executables, obfuscated scripts). It reads GitHub's public API and raw
// files only; nothing is downloaded to disk or run.
import { fetchText } from "./web";

export type Verdict = "ok" | "caution" | "risky";

export interface RepoReport {
  repo: string;
  verdict: Verdict;
  /** one line per finding, worst first */
  reasons: string[];
  stars?: number;
  created?: string;
  pushed?: string;
}

const DAY = 86_400_000;
const cache = new Map<string, { at: number; r: RepoReport }>();

async function json(url: string): Promise<{ status: number; body: any }> {
  try {
    const r = await fetchText(url, { headers: { Accept: "application/vnd.github+json" } }, 6_000_000);
    return { status: r.status, body: JSON.parse(r.text) };
  } catch (e: any) {
    const m = String(e?.message || e).match(/HTTP (\d+)/);
    return { status: m ? Number(m[1]) : 0, body: null };
  }
}

async function raw(owner: string, repo: string, branch: string, path: string): Promise<string | null> {
  try {
    return (await fetchText(`https://raw.githubusercontent.com/${owner}/${repo}/${encodeURIComponent(branch)}/${path.split("/").map(encodeURIComponent).join("/")}`, {}, 400_000)).text;
  } catch {
    return null;
  }
}

/** Code in an install hook or task that fetches or runs things it shouldn't. */
const DANGER = /(curl|wget|iwr|invoke-webrequest|invoke-expression|\biex\b|downloadstring|frombase64string|powershell\s+-e(nc)?\b|-encodedcommand|certutil|bitsadmin|mshta|rundll32|regsvr32|base64\s+-d|\|\s*(sh|bash|cmd)\b|eval\s*\(|new\s+function\s*\(|child_process|\bexecsync\b|\bspawnsync\b|discord(app)?\.com\/api\/webhooks|api\.telegram\.org|pastebin\.com|transfer\.sh|ngrok\.io)/i;
const EXEC_EXT = /\.(exe|dll|scr|msi|com|pif|cpl|sys|lnk|jar|vbs|vbe|wsf|hta|ps1|bat|cmd|apk|dmg|pkg|appimage|node|pyc)$/i;

function age(iso?: string): number {
  return iso ? (Date.now() - Date.parse(iso)) / DAY : Infinity;
}

export async function checkRepo(ownerIn: string, repoIn: string): Promise<RepoReport> {
  const owner = ownerIn.trim();
  const repo = repoIn.trim().replace(/\.git$/i, "");
  if (!/^[A-Za-z0-9-]+$/.test(owner) || !/^[\w.-]+$/.test(repo) || repo === "." || repo === "..") return { repo: `${owner}/${repo}`, verdict: "caution", reasons: ["Not a valid GitHub repository name, so it couldn't be checked."] };
  const key =`${owner}/${repo}`.toLowerCase();
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < 30 * 60_000) return hit.r;

  const risky: string[] = [];
  const caution: string[] = [];
  const meta = await json(`https://api.github.com/repos/${owner}/${repo}`);
  if (meta.status === 404 || meta.status === 451) {
    const r: RepoReport = { repo: `${owner}/${repo}`, verdict: "risky", reasons: ["GitHub has no such public repository: it is private, was renamed, or was taken down (GitHub removes malware repos)."] };
    cache.set(key, { at: Date.now(), r });
    return r;
  }
  if (!meta.body || meta.status >= 400 || meta.status === 0) {
    // rate limit (60 an hour without sign-in) or no network: say so, don't block
    return { repo: `${owner}/${repo}`, verdict: "caution", reasons: [`Couldn't check the repository on GitHub (${meta.status === 403 ? "GitHub's rate limit" : meta.status ? `HTTP ${meta.status}` : "no answer"}). Treat it as unchecked.`] };
  }
  const m = meta.body;
  const branch = m.default_branch || "HEAD";
  if (m.disabled) risky.push("GitHub has disabled this repository.");
  if (m.archived) caution.push("Archived: no longer maintained.");
  const stars = Number(m.stargazers_count) || 0;
  const repoAge = age(m.created_at);
  if (stars < 5 && repoAge < 30) caution.push(`Brand new and unknown: created ${Math.round(repoAge)} days ago, ${stars} stars.`);
  else if (stars < 3) caution.push(`Almost no one uses it (${stars} stars).`);
  if (m.fork && m.parent?.full_name) caution.push(`A fork of ${m.parent.full_name}: compare it with the original, forks are a common way to slip in malware.`);
  if (!m.license) caution.push("No license.");

  const user = await json(`https://api.github.com/users/${owner}`);
  if (user.body && age(user.body.created_at) < 30) caution.push(`The owner's account is ${Math.round(age(user.body.created_at))} days old.`);

  // files: what runs on install or on open
  const tree = await json(`https://api.github.com/repos/${owner}/${repo}/git/trees/${encodeURIComponent(branch)}?recursive=1`);
  const files: { path: string; size: number }[] = Array.isArray(tree.body?.tree) ? tree.body.tree.filter((t: any) => t.type === "blob").map((t: any) => ({ path: String(t.path), size: Number(t.size) || 0 })) : [];
  const execs = files.filter((f) => EXEC_EXT.test(f.path) && !/(^|\/)(test|tests|fixtures|examples?|docs?)\//i.test(f.path));
  const bins = execs.filter((f) => /\.(exe|dll|scr|msi|com|pif|cpl|sys|lnk|jar|apk|dmg|pkg|appimage|node|pyc)$/i.test(f.path));
  const scripts = execs.filter((f) => !bins.includes(f));
  if (bins.length) (stars < 50 ? risky : caution).push(`Contains ${bins.length} executable file${bins.length > 1 ? "s" : ""} (${bins.slice(0, 3).map((f) => f.path).join(", ")}${bins.length > 3 ? ", …" : ""}).`);
  if (scripts.length && stars < 50) caution.push(`Contains Windows scripts (${scripts.slice(0, 3).map((f) => f.path).join(", ")}${scripts.length > 3 ? ", …" : ""}).`);
  const big = files.filter((f) => /\.(js|cjs|mjs|py|ps1)$/i.test(f.path) && f.size > 300_000 && !/(\.min\.|dist\/|build\/|vendor\/|node_modules\/)/i.test(f.path));
  if (big.length && stars < 50) caution.push(`Very large script files (${big.slice(0, 2).map((f) => f.path).join(", ")}): possibly obfuscated.`);

  const has = (p: string) => files.some((f) => f.path === p);
  if (has("package.json")) {
    const pj = await raw(owner, repo, branch, "package.json");
    try {
      const s = JSON.parse(pj || "{}").scripts || {};
      for (const hook of ["preinstall", "install", "postinstall", "prepare"]) {
        const cmd = s[hook];
        if (typeof cmd !== "string") continue;
        if (DANGER.test(cmd)) risky.push(`package.json "${hook}" runs on npm install and fetches or executes code: ${cmd.slice(0, 160)}`);
        else if (hook !== "prepare") caution.push(`package.json runs "${cmd.slice(0, 100)}" on npm install.`);
      }
    } catch {}
  }
  for (const p of ["setup.py", "install.py"]) {
    if (!has(p)) continue;
    const src = (await raw(owner, repo, branch, p)) || "";
    if (/(urllib|requests\.get|socket|subprocess|os\.system|exec\(|eval\(|base64)/.test(src) && /(cmdclass|install\b)/.test(src)) risky.push(`${p} fetches or runs code during install.`);
  }
  if (has(".vscode/tasks.json")) {
    const t = (await raw(owner, repo, branch, ".vscode/tasks.json")) || "";
    if (/folderOpen/i.test(t)) risky.push(".vscode/tasks.json runs a task as soon as the folder is opened in VS Code.");
  }
  for (const f of files.filter((f) => /(^|\/)(install|setup|run|start)\.(bat|cmd|ps1|sh)$/i.test(f.path)).slice(0, 2)) {
    const t = (await raw(owner, repo, branch, f.path)) || "";
    if (DANGER.test(t) && /(frombase64string|-encodedcommand|downloadstring|iex|invoke-expression|\|\s*(sh|bash)|webhooks|telegram|pastebin)/i.test(t)) risky.push(`${f.path} downloads and runs code in a way malware often does.`);
  }
  if (tree.body?.truncated) caution.push("The repository is too big to scan every file.");

  const verdict: Verdict = risky.length ? "risky" : caution.length ? "caution" : "ok";
  const reasons = [...risky, ...caution];
  if (verdict === "ok") reasons.push(`${stars.toLocaleString()} stars, created ${new Date(m.created_at).toISOString().slice(0, 10)}, nothing that runs on install or open.`);
  const r: RepoReport = { repo: m.full_name || `${owner}/${repo}`, verdict, reasons, stars, created: m.created_at, pushed: m.pushed_at };
  cache.set(key, { at: Date.now(), r });
  return r;
}

/** GitHub repositories named in a command line or URL. */
export function reposIn(text: string): { owner: string; repo: string }[] {
  const out: { owner: string; repo: string }[] = [];
  const add = (o: string, r: string) => {
    r = r.replace(/\.git$/i, "");
    // GitHub names: owners are letters, digits and hyphens; "." and ".." would
    // turn the API address into another path
    if (!/^[A-Za-z0-9-]+$/.test(o) || !/^[\w.-]+$/.test(r) || r === "." || r === "..") return;
    if (!o || !r || /^(settings|features|topics|marketplace|search|login|about|pricing|sponsors|apps|collections|trending|explore|orgs|users|notifications|issues|pulls|new|organizations|enterprise|site|security|customer-stories|readme)$/i.test(o)) return;
    if (!out.some((x) => x.owner.toLowerCase() === o.toLowerCase() && x.repo.toLowerCase() === r.toLowerCase())) out.push({ owner: o, repo: r });
  };
  const pats = [
    /(?:^|[\s"'(=@+])(?:https?:\/\/|git@|git\+https:\/\/|ssh:\/\/git@)?(?:www\.)?github\.com[/:]([\w.-]+)\/([\w.-]+)/gi,
    /(?:raw\.githubusercontent\.com|codeload\.github\.com)\/([\w.-]+)\/([\w.-]+)/gi,
    /\bgithub:([\w.-]+)\/([\w.-]+)/gi,
    /\bgh\s+repo\s+(?:clone|fork)\s+([\w.-]+)\/([\w.-]+)/gi,
    /\bdegit\s+(?:github:)?([\w.-]+)\/([\w.-]+)/gi,
  ];
  for (const p of pats) for (const m of text.matchAll(p)) add(m[1], m[2]);
  return out.slice(0, 4);
}
