// CQS, the code quality scanner: after the agent changes files (or on
// /cqs), it checks them: syntax checks it can do itself (JSON, brackets,
// HTML tags, leftovers like placeholders and conflict markers) and the
// project's own compilers (python, node, tsc, cargo: commands, so the user
// approves them). Problems go to a fixer subagent with a fresh context,
// then everything is checked again. `polish` also asks the fixer to tidy
// the code and UI without changing behaviour.
import { bridge, need } from "../lib/bridge";
import { app, beginLive, endLive, liveChat, toast, uid, updateLive, type Chat, type Msg } from "../state";
import { askApproval } from "./tools";
import { subagent } from "./agent";

export interface Issue {
  file: string;
  line?: number;
  severity: "error" | "warning";
  message: string;
}

const CODE = /\.(ts|tsx|js|jsx|mjs|cjs|py|rs|go|java|kt|c|h|cc|cpp|hpp|cs|css|scss|html|htm|json|vue|svelte|php|rb|lua|swift|sh|ps1)$/i;
const BRACES = /\.(ts|tsx|js|jsx|mjs|cjs|rs|go|java|kt|c|h|cc|cpp|hpp|cs|css|scss|json|php|swift)$/i;

async function raw(root: string, file: string): Promise<string> {
  const r = await need().fs.read(root, file, 1, 100000);
  return r.text
    .split("\n")
    .filter((l) => !/^\[lines \d+-\d+ of \d+/.test(l))
    .map((l) => l.slice(7))
    .join("\n");
}

/** Brackets out of balance, skipping strings and comments (roughly). */
function brackets(src: string, file: string, out: Issue[]) {
  const stack: { c: string; line: number }[] = [];
  const pair: Record<string, string> = { ")": "(", "]": "[", "}": "{" };
  let line = 1;
  const py = /\.py$/i.test(file);
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (ch === "\n") {
      line++;
      continue;
    }
    // comments
    if (!py && ch === "/" && src[i + 1] === "/") {
      while (i < src.length && src[i] !== "\n") i++;
      i--;
      continue;
    }
    if (!py && ch === "/" && src[i + 1] === "*") {
      const e = src.indexOf("*/", i + 2);
      const end = e < 0 ? src.length : e + 2;
      line += (src.slice(i, end).match(/\n/g) || []).length;
      i = end - 1;
      continue;
    }
    // strings (no regex literals: they rarely hold unbalanced brackets)
    if (ch === '"' || ch === "'" || ch === "`") {
      if (ch === "'" && /\.rs$/i.test(file) && /[a-z_]/i.test(src[i + 1] || "") && src[i + 2] !== "'") continue; // a lifetime
      let j = i + 1;
      while (j < src.length && src[j] !== ch) {
        if (src[j] === "\\") j++;
        else if (src[j] === "\n") {
          if (ch !== "`") break;
          line++;
        }
        j++;
      }
      i = j;
      continue;
    }
    if (ch === "(" || ch === "[" || ch === "{") stack.push({ c: ch, line });
    else if (ch === ")" || ch === "]" || ch === "}") {
      const top = stack.pop();
      if (!top || top.c !== pair[ch]) {
        out.push({ file, line, severity: "error", message: top ? `"${ch}" closes "${top.c}" opened on line ${top.line}` : `unmatched "${ch}"` });
        return;
      }
    }
  }
  if (stack.length) out.push({ file, line: stack[stack.length - 1].line, severity: "error", message: `"${stack[stack.length - 1].c}" is never closed` });
}

function htmlTags(src: string, file: string, out: Issue[]) {
  const body = src.replace(/<!--[\s\S]*?-->/g, "").replace(/<(script|style)[\s\S]*?<\/\1>/gi, "");
  for (const tag of ["div", "span", "section", "main", "ul", "ol", "table", "form", "button", "a", "p", "nav", "header", "footer"]) {
    const open = (body.match(new RegExp(`<${tag}[\\s>]`, "gi")) || []).length;
    const close = (body.match(new RegExp(`</${tag}>`, "gi")) || []).length;
    if (open !== close && tag !== "p") out.push({ file, severity: tag === "div" ? "error" : "warning", message: `${open} <${tag}> but ${close} </${tag}>` });
  }
}

/** Checks done without running anything. */
export function staticChecks(file: string, src: string): Issue[] {
  const out: Issue[] = [];
  const lines = src.split("\n");
  lines.forEach((l, i) => {
    if (/^(<{7}|={7}|>{7})( |$)/.test(l)) out.push({ file, line: i + 1, severity: "error", message: "merge conflict marker" });
    if (/(\.\.\.\s*(rest|remaining|existing|other) (of )?(the )?(code|implementation|content)|your code here|implement (this|me)|TODO: implement|placeholder implementation)/i.test(l)) out.push({ file, line: i + 1, severity: "error", message: `placeholder left in: "${l.trim().slice(0, 80)}"` });
    else if (/\b(TODO|FIXME|XXX)\b/.test(l)) out.push({ file, line: i + 1, severity: "warning", message: l.trim().slice(0, 100) });
  });
  if (/\.json$/i.test(file)) {
    try {
      JSON.parse(src);
    } catch (e: any) {
      out.push({ file, severity: "error", message: `invalid JSON: ${e.message}` });
    }
  } else if (BRACES.test(file)) brackets(src, file, out);
  if (/\.html?$/i.test(file)) htmlTags(src, file, out);
  if (/\.py$/i.test(file) && lines.some((l) => /^\t/.test(l)) && lines.some((l) => /^ {2,}\S/.test(l))) out.push({ file, severity: "error", message: "indentation mixes tabs and spaces" });
  if (/\.(js|jsx|ts|tsx)$/i.test(file)) {
    const n = lines.filter((l) => /console\.log\(/.test(l) && !/^\s*\/\//.test(l)).length;
    if (n > 3) out.push({ file, severity: "warning", message: `${n} console.log calls left in` });
  }
  return out;
}

async function exists(root: string, f: string) {
  try {
    await need().fs.read(root, f, 1, 1);
    return true;
  } catch {
    return false;
  }
}

/** The project's own checkers for these files: (command, which files). */
async function toolChecks(root: string, files: string[]): Promise<{ cmd: string; what: string }[]> {
  const q = (f: string) => `"${f.replace(/\//g, "\\")}"`;
  const out: { cmd: string; what: string }[] = [];
  const py = files.filter((f) => /\.py$/i.test(f));
  if (py.length) out.push({ cmd: `python -m py_compile ${py.map(q).join(" ")}`, what: "Python syntax" });
  const js = files.filter((f) => /\.(js|cjs|mjs)$/i.test(f));
  if (js.length) out.push({ cmd: js.map((f) => `node --check ${q(f)}`).join(" && "), what: "JavaScript syntax" });
  if (files.some((f) => /\.tsx?$/i.test(f)) && (await exists(root, "tsconfig.json")) && (await exists(root, "node_modules/typescript/package.json"))) out.push({ cmd: "npx --no-install tsc --noEmit -p .", what: "TypeScript types" });
  if (files.some((f) => /\.rs$/i.test(f)) && (await exists(root, "Cargo.toml"))) out.push({ cmd: "cargo check --quiet --message-format short", what: "Rust (cargo check)" });
  return out;
}

/** Problems in a checker's output, as issues (file:line: message when it says). */
function parseOutput(out: string, what: string): Issue[] {
  const issues: Issue[] = [];
  for (const l of out.split(/\r?\n/)) {
    const m = l.match(/^\s*(?:File ")?([\w./\\:-]+\.\w+)"?(?:[(:, ]+line )?[(:]?(\d+)[,:)]?.*?(error|warning|SyntaxError|Error)?[: ]+(.*)$/i);
    if (m && /error|warning|syntax|expected|unexpected|invalid/i.test(l)) issues.push({ file: m[1].replace(/\\/g, "/"), line: Number(m[2]) || undefined, severity: /warning/i.test(l) ? "warning" : "error", message: l.trim().slice(0, 300) });
  }
  if (!issues.length && out.trim()) issues.push({ file: what, severity: "error", message: out.trim().split("\n").slice(-12).join("\n").slice(0, 1200) });
  return issues;
}

/** References to files of the project that aren't there: relative imports
 *  and requires (JS/TS), and local scripts, styles and images (HTML) —
 *  models often use a file they meant to write and never did. */
function missingRefs(file: string, src: string, have: Set<string>, out: Issue[]) {
  const dir = file.includes("/") ? file.slice(0, file.lastIndexOf("/") + 1) : "";
  const norm = (p: string) => {
    const parts: string[] = [];
    for (const x of (dir + p).split("/")) {
      if (x === "..") parts.pop();
      else if (x && x !== ".") parts.push(x);
    }
    return parts.join("/");
  };
  const lineOf = (i: number) => src.slice(0, i).split("\n").length;
  if (/\.(ts|tsx|js|jsx|mjs|cjs|vue|svelte)$/i.test(file)) {
    const re = /(?:import\s[^'"]*?from\s*|import\s*\(\s*|require\s*\(\s*|^\s*import\s+)["'](\.{1,2}\/[^"']+)["']/gm;
    for (const m of src.matchAll(re)) {
      const base = norm(m[2].replace(/\?.*$/, ""));
      const tries = [base, ...[".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".vue", ".svelte", ".css"].map((e) => base + e), ...["/index.ts", "/index.tsx", "/index.js", "/index.jsx"].map((e) => base + e)];
      // (a .js import may name the .ts file it compiles from)
      if (/\.js$/.test(base)) tries.push(base.replace(/\.js$/, ".ts"), base.replace(/\.js$/, ".tsx"));
      if (!tries.some((t) => have.has(t))) out.push({ file, line: lineOf(m.index!), severity: "error", message: `imports "${m[2]}", which isn't in the project` });
    }
  }
  if (/\.html?$/i.test(file)) {
    const re = /<(script|link|img|source)\b[^>]*?\s(src|href)\s*=\s*["']([^"'#?]+)/gi;
    for (const m of src.matchAll(re)) {
      const ref = m[3].trim();
      if (!ref || /^([a-z]+:|\/\/|\{|\$)/i.test(ref) || (m[1].toLowerCase() === "link" && !/\.(css|ico|png|svg|webmanifest|json)$/i.test(ref))) continue;
      const p = ref.startsWith("/") ? ref.slice(1) : norm(ref);
      if (!have.has(p)) out.push({ file, line: lineOf(m.index!), severity: m[1].toLowerCase() === "img" || m[1].toLowerCase() === "source" ? "warning" : "error", message: `uses "${ref}", which isn't in the project` });
    }
  }
}

async function scan(chat: Chat, files: string[], asked: Set<string>): Promise<{ issues: Issue[]; ran: string[]; skipped: string[] }> {
  const root = chat.folder!;
  const issues: Issue[] = [];
  const have = new Set((await need().fs.find(root, "**/*").catch(() => [] as string[])).map((f) => f.replace(/\\/g, "/")));
  for (const f of files) {
    try {
      const src = await raw(root, f);
      issues.push(...staticChecks(f, src));
      if (have.size) missingRefs(f, src, have, issues);
    } catch {}
  }
  const ran: string[] = [];
  const skipped: string[] = [];
  const s = app.get().settings;
  for (const c of await toolChecks(root, files)) {
    // the checkers only read the code; cargo may run build scripts, so it
    // asks unless nothing asks (Full auto)
    const quiet = s.permMode === "auto" || (s.cqsQuiet && s.permMode !== "ask" && !c.cmd.startsWith("cargo"));
    const ok = quiet || asked.has(c.cmd) ? "allow" : await askApproval(chat, c.cmd, "CQS wants to check your code", `${c.what}. CQS runs your project's own checker to catch errors before you do.`);
    if (ok === "deny") {
      skipped.push(c.what);
      continue;
    }
    asked.add(c.cmd);
    const r = await need().cmd.run(uid(), root, c.cmd, 180000, false);
    ran.push(c.what);
    if (r.code !== 0) issues.push(...parseOutput(r.output, c.what));
  }
  return { issues, ran, skipped };
}

function changedFiles(chat: Chat): string[] {
  const s = new Set<string>();
  for (const m of chat.messages) if (m.role === "tool" && (m.ui.kind === "write" || m.ui.kind === "edit") && m.ui.path) s.add(m.ui.path);
  return [...s];
}

const FIXER = `You are CQS, the code quality fixer working inside the user's project. You get a list of problems a scanner found. Fix each real problem with the smallest correct change: read the file around the reported line, then edit_file. A file that is imported or linked but missing: create it with real, working content that fits how it's used (read the files that use it first), or fix the reference if it points to the wrong path. Don't rewrite working code, don't change behaviour that isn't broken, don't add features. If a reported problem is not real, leave it. When done, reply with a short list of what you fixed.`;
const POLISH = `You are CQS in polish mode, working inside the user's project. Make the given files cleaner without changing what they do: consistent formatting and naming, remove dead code and leftover debugging, clearer structure. For user interfaces: consistent spacing, alignment, readable contrast, sensible responsive layout and hover/focus states. Make focused edits with edit_file. When done, reply with a short list of what you improved.`;

/** Fixer rounds at most (each followed by a fresh scan). */
const ROUNDS = 2;

const listOf = (issues: Issue[]) => issues.map((i) => `- ${i.file}${i.line ? `:${i.line}` : ""}: ${i.message}`).join("\n");

/** Scans `files` (default: what this chat changed), fixes what it finds. */
export async function runCQS(files: string[] | null, mode: "check" | "polish", manual = false) {
  // the chat a reply is being written in, else the open one
  const own = !liveChat();
  const chat = liveChat() || app.get().chat;
  if (!chat?.folder || !bridge) return toast("CQS needs a workspace folder.");
  let list = (files && files.length ? files : changedFiles(chat)).filter((f) => CODE.test(f));
  if (!list.length && manual) list = (await need().fs.find(chat.folder, "**/*.*")).filter((f) => CODE.test(f)).slice(0, 40);
  if (!list.length) return manual ? toast("CQS found no code files to check.") : undefined;
  const id = uid();
  const card = (text: string, detail: any) =>
    updateLive((c) => {
      const has = c.messages.some((m) => m.id === id);
      const m: Msg = { id, role: "event", kind: "cqs", text, detail: JSON.stringify(detail), ts: Date.now() };
      return { ...c, messages: has ? c.messages.map((x) => (x.id === id ? m : x)) : [...c.messages, m] };
    });
  const running = app.get().running;
  if (own) beginLive(chat);
  if (!running) app.set({ running: true });
  app.set({ draft: { chatId: chat.id, content: "", reasoning: "", phase: "cqs", tps: 0, generated: 0, started: Date.now() } });
  const asked = new Set<string>();
  const now = () => liveChat() || chat;
  try {
    card(`CQS: scanning ${list.length} file${list.length > 1 ? "s" : ""}…`, { files: list, state: "scanning" });
    let r = await scan(now(), list, asked);
    let errors = r.issues.filter((i) => i.severity === "error");
    if (!errors.length && mode === "check") {
      card(r.issues.length ? `CQS: no errors · ${r.issues.length} note${r.issues.length > 1 ? "s" : ""}` : "CQS: all clear", { files: list, issues: r.issues, ran: r.ran, skipped: r.skipped, state: "clean" });
      return;
    }
    if (!app.get().settings.cqsFix && mode === "check") {
      card(`CQS: ${errors.length} error${errors.length > 1 ? "s" : ""} found`, { files: list, issues: r.issues, ran: r.ran, skipped: r.skipped, state: "found" });
      return;
    }
    const steps: string[] = [];
    const found = errors.length;
    let summary = "";
    const changed = new Set<string>();
    for (let round = 1; round <= ROUNDS; round++) {
      const label = mode === "polish" && round === 1 ? "CQS: polishing…" : `CQS: fixing ${errors.length} error${errors.length > 1 ? "s" : ""}${round > 1 ? ` (round ${round})` : ""}…`;
      card(label, { files: list, issues: r.issues, ran: r.ran, state: "fixing", steps });
      const task =
        mode === "polish" && round === 1
          ? `Polish these files: ${list.join(", ")}.${r.issues.length ? `\nAlso fix these problems the scanner found:\n${listOf(r.issues)}` : ""}`
          : `Fix these problems:\n${listOf(errors)}${round > 1 ? "\n\nAn earlier round tried and these are still there: look closer at each one." : ""}`;
      const fix = await subagent({
        folder: chat.folder,
        system: mode === "polish" && round === 1 ? POLISH : FIXER,
        task,
        tools: ["read_file", "edit_file", "write_file", "list_files", "search_files"],
        maxSteps: 30,
        onStep: (m) => {
          if (m.role === "tool" && m.ui.title && !m.ui.running) {
            steps.push(`${m.ui.kind === "edit" ? "Edited" : m.ui.kind === "write" ? "Wrote" : m.ui.kind === "read" ? "Read" : m.ui.kind} ${m.ui.title}`);
            card(label, { files: list, issues: r.issues, state: "fixing", steps });
          }
        },
      });
      summary = fix.summary;
      fix.changed.forEach((f) => changed.add(f));
      // files the fixer created are checked too
      list = [...new Set([...list, ...fix.changed.filter((f) => CODE.test(f))])];
      r = await scan(now(), list, asked);
      errors = r.issues.filter((i) => i.severity === "error");
      if (!errors.length) break;
    }
    const left = errors.length;
    card(
      left ? `CQS: fixed ${Math.max(0, found - left)} of ${found}, ${left} error${left > 1 ? "s" : ""} left` : mode === "polish" ? "CQS: polished" : `CQS: fixed ${found} error${found > 1 ? "s" : ""}`,
      { files: list, issues: r.issues, ran: r.ran, state: left ? "found" : "fixed", steps, summary, changed: [...changed] },
    );
  } catch (e: any) {
    card(`CQS stopped: ${e?.message || e}`, { files: list, state: "error" });
  } finally {
    app.set({ draft: null });
    if (!running) app.set({ running: false });
    if (own) endLive();
  }
}
