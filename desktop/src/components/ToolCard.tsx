// A tool call in the transcript, as Claude Code's app shows them: one row
// each ("Created rail.css +406 ▾") in a shared box, opening to the file's
// path and diff, a command's output, a search's sources or the plan.
import { useMemo, useState } from "react";
import hljs from "highlight.js/lib/common";
import { bridge } from "../lib/bridge";
import { diffLines, diffStats, hunks } from "../lib/diff";
import { app, openLink, useApp, type Photo, type ToolUI } from "../state";
import { Icon, TodoMark } from "./Icons";
import { openArtifact } from "./ArtifactPanel";

/** Opens a written file beside the chat (as a preview for pages, SVGs, Markdown). */
function ViewFile({ path, text }: { path?: string; text?: string }) {
  if (!path || text === undefined) return null;
  const ext = (path.split(".").pop() || "").toLowerCase();
  const pv = ["html", "htm", "svg", "md", "markdown"].includes(ext);
  return (
    <button className="tool-view" title={pv ? "Preview beside the chat" : "Open beside the chat"} onClick={(e) => (e.stopPropagation(), openArtifact(path.split("/").pop() || path, text, ext))}>
      <Icon name={pv ? "eye" : "external"} size={12} /> {pv ? "Preview" : "View"}
    </button>
  );
}

const host = (u: string) => {
  try {
    return new URL(u).hostname.replace(/^www\./, "");
  } catch {
    return u;
  }
};

const EXT_LANG: Record<string, string> = { js: "javascript", mjs: "javascript", cjs: "javascript", jsx: "javascript", ts: "typescript", tsx: "typescript", py: "python", rs: "rust", htm: "xml", html: "xml", svg: "xml", vue: "xml", md: "markdown", yml: "yaml", cs: "csharp", h: "c", hpp: "cpp", cc: "cpp", ps1: "powershell", sh: "bash", bat: "dos", cmd: "dos", toml: "ini" };

/** One line, syntax-coloured (each on its own: a construct spanning lines loses its colour). */
function lineHtml(text: string, lang: string): string {
  if (!lang || !text.trim()) return esc(text || " ");
  try {
    return hljs.highlight(text, { language: lang, ignoreIllegals: true }).value;
  } catch {
    return esc(text);
  }
}
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** An old and a new line with the part between their common start and end
 *  marked (null when they share too little to be the same line edited). */
function wordMarks(a: string, b: string, lang: string): [string, string] | null {
  let p = 0;
  while (p < a.length && p < b.length && a[p] === b[p]) p++;
  let s = 0;
  while (s < a.length - p && s < b.length - p && a[a.length - 1 - s] === b[b.length - 1 - s]) s++;
  if (p + s < 3 || (p + s) * 3 < Math.min(a.length, b.length)) return null;
  const mark = (t: string) => {
    const mid = t.slice(p, t.length - s);
    return `${lineHtml(t.slice(0, p), lang)}${mid ? `<span class="diff-word">${lineHtml(mid, lang)}</span>` : ""}${lineHtml(t.slice(t.length - s), lang)}`;
  };
  return [mark(a), mark(b)];
}

/** A command as typed, coloured as shell. */
function cmdHtml(c: string): string {
  try {
    return hljs.highlight(c, { language: "bash", ignoreIllegals: true }).value;
  } catch {
    return esc(c);
  }
}

export function Diff({ before, after, path }: { before: string | null; after: string; path?: string }) {
  const ext = (path?.split(".").pop() || "").toLowerCase();
  const lang = hljs.getLanguage(EXT_LANG[ext] || ext) ? EXT_LANG[ext] || ext : "";
  const rows = useMemo(() => {
    const lines = diffLines(before ?? "", after);
    const h = before === null ? lines.slice(0, 400).map((l) => ({ ...l, kind: "+" as const })) : hunks(lines, 3).slice(0, 600);
    const html = h.map((l) => (l === null ? "" : lineHtml(l.text, lang)));
    // a run of removed lines followed by added ones: pair them up and mark the words that changed
    if (before !== null)
      for (let i = 0; i < h.length; ) {
        if (h[i]?.kind !== "-") {
          i++;
          continue;
        }
        let j = i;
        while (h[j]?.kind === "-") j++;
        let k = j;
        while (h[k]?.kind === "+") k++;
        for (let n = 0; n < Math.min(j - i, k - j); n++) {
          const w = wordMarks(h[i + n]!.text, h[j + n]!.text, lang);
          if (w) [html[i + n], html[j + n]] = w;
        }
        i = k;
      }
    return { h: h.map((l, i) => (l === null ? null : { ...l, html: html[i] })), more: before === null && lines.length > 400 ? lines.length - 400 : 0 };
  }, [before, after, lang]);
  const created = before === null;
  // one gutter: the sign and the line's number ("+ 12"), the old number for removed lines
  return (
    <div className={`diff ${created ? "created" : ""}`}>
      {rows.h.map((l, i) =>
        l === null ? (
          <div key={i} className="diff-gap">
            <Icon name="more" size={12} />
          </div>
        ) : (
          <div key={i} className={`diff-line ${l.kind === "+" ? "add" : l.kind === "-" ? "del" : ""}`}>
            <span className="diff-gut">
              {l.kind === " " ? " " : l.kind} {l.kind === "-" ? l.a : l.b}
            </span>
            <span className="diff-text hljs" dangerouslySetInnerHTML={{ __html: l.html }} />
          </div>
        ),
      )}
      {rows.more > 0 && <div className="diff-gap">… {rows.more} more lines</div>}
    </div>
  );
}

/** GitHub's five little squares: how much of the change is added vs removed. */
export function DiffBar({ added, removed }: { added: number; removed: number }) {
  const total = added + removed;
  if (!total) return null;
  const g = Math.round((added / total) * 5);
  return (
    <span className="diffbar" aria-hidden="true">
      {Array.from({ length: 5 }, (_, i) => (
        <i key={i} className={i < g ? "add" : "del"} />
      ))}
    </span>
  );
}

/** A file's name in a row (the full path is its tooltip and shows when it opens). */
function FileName({ path }: { path?: string }) {
  if (!path) return null;
  return (
    <span className="tool-obj" title={path}>
      {path.split(/[\\/]/).pop()}
    </span>
  );
}

/** The full path above a diff; a click shows the file in Explorer. */
function FullPath({ path }: { path?: string }) {
  const folder = useApp((s) => s.chat?.folder);
  if (!path) return null;
  const abs = /^[a-z]:[\\/]/i.test(path) || !folder ? path : `${folder.replace(/[\\/]+$/, "")}/${path}`;
  return (
    <button className="tool-path" title="Show in Explorer" onClick={() => folder && bridge?.fs.reveal(folder, path)}>
      {abs.replace(/\//g, "\\")}
    </button>
  );
}

const Verb = ({ children }: { children: React.ReactNode }) => <span className="tool-verb">{children}</span>;

const WAITING = "Waiting for your approval…";

/** "Ran 3 commands, edited ui.tsx": what a row of tool calls did, in a few words. */
function summarize(items: ToolUI[]): React.ReactNode {
  const groups = new Map<string, ToolUI[]>();
  const key = (u: ToolUI) => (u.kind === "write" ? (u.before === null ? "create" : "rewrite") : u.kind === "browser" ? "page" : u.kind);
  for (const u of items) groups.set(key(u), [...(groups.get(key(u)) || []), u]);
  const name = (u: ToolUI) => (u.path || u.title || "").split(/[\\/]/).pop() || "";
  const parts: React.ReactNode[] = [];
  const say = (verb: string, list: ToolUI[], one: React.ReactNode, many: string) => parts.push(list.length === 1 ? <>{verb} {one}</> : <>{verb} {list.length} {many}</>);
  for (const [k, list] of groups) {
    const obj = <span className="tool-obj">{name(list[0])}</span>;
    if (k === "command") say(list.some((u) => u.detail === WAITING) ? "wants to run" : list.some((u) => u.running) ? "running" : "ran", list, "a command", "commands");
    else if (k === "edit") say("edited", list, obj, "files");
    else if (k === "create") say("created", list, obj, "files");
    else if (k === "rewrite") say("rewrote", list, obj, "files");
    else if (k === "read") say("read", list, obj, "files");
    else if (k === "page") say("read", list, "a page", "pages");
    else if (k === "search") say("searched the web", list, "", "times");
    else if (k === "grep") say("searched files", list, "", "times");
    else if (k === "list") say("listed", list, "a folder", "folders");
    else if (k === "todo") parts.push(<>updated the plan</>);
    else if (k === "denied") say("declined", list, "a step", "steps");
    else if (k === "error") say("hit", list, "an error", "errors");
    else if (k === "trash") say("moved", list, obj, "files to the Recycle Bin");
    else if (k === "memory") parts.push(list.every((u) => u.detail === "Forgot") ? <>forgot {list.length === 1 ? "something" : `${list.length} things`}</> : list.every((u) => u.detail === "Remembered") ? <>remembered {list.length === 1 ? "something" : `${list.length} things`}</> : <>updated its memory</>);
    else say("used", list, <span className="tool-obj">{list[0].title}</span>, "tools");
  }
  return parts.map((p, i) => (
    <span key={i} className={i ? "" : "tool-sum-first"}>
      {i ? ", " : ""}
      {p}
    </span>
  ));
}

/** Tool calls in a row: one line saying what they did, opening to their rows.
 *  It's open while one runs or waits for you, and while the reply is live. */
export function ToolGroup({ items, live, pictures = [] }: { items: ToolUI[]; live: boolean; pictures?: Photo[] }) {
  const busy = items.some((u) => u.running || u.detail === WAITING);
  const [open, setOpen] = useState<boolean | null>(null);
  const shown = open ?? (busy || live);
  return (
    <div className="tool-sum-wrap">
      <button className="tool-sum" onClick={() => setOpen(!shown)} aria-expanded={shown}>
        {busy && <span className="spinner" />}
        <span className="tool-sum-text">{summarize(items)}</span>
        <span className={`tool-tri ${shown ? "open" : ""}`} />
      </button>
      {shown && (
        <div className="tool-group">
          {items.map((u, i) => (
            <ToolCard key={i} ui={u} />
          ))}
        </div>
      )}
      {pictures.length > 0 && (
        <div className="tool-pictures">
          {pictures.map((p, i) => (
            <button key={i} className="tool-picture" title={p.name} onClick={() => app.set({ photo: p.data })}>
              <img src={p.data} alt={p.name} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function ToolCard({ ui }: { ui: ToolUI }) {
  const isFile = ui.kind === "write" || ui.kind === "edit";
  const [open, setOpen] = useState(ui.kind === "edit" || ui.kind === "todo" || (ui.kind === "write" && ui.before !== null) || (ui.kind === "command" && (!!ui.running || ui.detail === WAITING)));
  const stats = useMemo(() => (isFile && ui.after !== undefined ? diffStats(diffLines(ui.before ?? "", ui.after)) : null), [isFile, ui.before, ui.after]);

  let label: React.ReactNode = ui.title;
  let body: React.ReactNode = null;
  let extra: React.ReactNode = null;
  switch (ui.kind) {
    case "search":
      label = (
        <>
          <Verb>Searched the web</Verb> <span className="tool-obj">“{ui.title}”</span>
        </>
      );
      body = ui.results?.length ? (
        <div className="sources">
          {ui.results.map((r, i) => (
            <button key={i} className="source" onClick={() => openLink(r.url)} title={r.url}>
              <span className="source-host">{host(r.url)}</span>
              <span className="source-title">{r.title}</span>
            </button>
          ))}
        </div>
      ) : (
        <div className="muted small">No results</div>
      );
      break;
    case "page":
    case "browser":
      label = (
        <>
          <Verb>{ui.kind === "browser" ? "Opened in browser" : "Read"}</Verb>{" "}
          <button className="link-btn" onClick={(e) => (e.stopPropagation(), ui.url && openLink(ui.url))}>
            {ui.title?.slice(0, 80)}
          </button>{" "}
          <span className="tool-verb">{ui.url && host(ui.url)}</span>
        </>
      );
      break;
    case "read":
      label = (
        <>
          <Verb>Read</Verb> <FileName path={ui.path} /> <span className="tool-verb">{ui.detail}</span>
        </>
      );
      break;
    case "write":
    case "edit":
      label = (
        <>
          <Verb>{ui.kind === "edit" ? "Edited" : ui.before === null ? "Created" : "Rewrote"}</Verb> <FileName path={ui.path} />
          {stats && (
            <span className="diffstat">
              <span className="add">+{stats.added}</span>
              {stats.removed > 0 && <span className="del">−{stats.removed}</span>}
            </span>
          )}
        </>
      );
      extra = <ViewFile path={ui.path} text={ui.after} />;
      body =
        ui.after !== undefined ? (
          <>
            <FullPath path={ui.path} />
            <Diff before={ui.before ?? null} after={ui.after} path={ui.path} />
          </>
        ) : null;
      break;
    case "list":
      label = (
        <>
          <Verb>Listed</Verb> <span className="tool-obj">{ui.title}</span>
        </>
      );
      body = <pre className="tool-pre">{ui.output}</pre>;
      break;
    case "grep":
      label = (
        <>
          <Verb>Searched files for</Verb> <span className="tool-obj">{ui.title}</span>
        </>
      );
      body = <pre className="tool-pre">{ui.output}</pre>;
      break;
    case "command":
      label = (
        <>
          {ui.about ? (
            <span className="tool-about">{ui.detail === "Waiting for your approval…" ? `Wants to: ${ui.about}` : ui.about}</span>
          ) : (
            <>
              <Verb>{ui.running ? "Running" : ui.detail === "Waiting for your approval…" ? "Wants to run" : "Ran"}</Verb> <span className="tool-obj mono cmd-inline">{ui.title}</span>
            </>
          )}
          {!ui.running && ui.code !== undefined && ui.code !== null && <span className={`exit ${ui.code === 0 ? "ok" : "bad"}`}>{ui.code === 0 ? <Icon name="check" size={13} stroke={2.4} /> : `exit ${ui.code}`}</span>}
        </>
      );
      body = (
        <pre className="tool-cmd">
          <span className="cmd-dollar">$ </span>
          <code dangerouslySetInnerHTML={{ __html: cmdHtml(ui.title) }} />
          {(ui.output || ui.running) && <span className="cmd-out">{"\n\n" + (ui.output?.trimEnd() || "…")}</span>}
        </pre>
      );
      break;
    case "todo":
      label = (
        <>
          <Verb>Updated the plan</Verb> <span className="tool-obj">{ui.title}</span>
        </>
      );
      body = (
        <ul className="todos">
          {ui.todos?.map((t, i) => (
            <li key={i} className={t.status}>
              <TodoMark status={t.status} />
              {t.content}
            </li>
          ))}
        </ul>
      );
      break;
    case "mcp":
      label = <span className="tool-obj">{ui.title}</span>;
      body = <pre className="tool-pre">{ui.output}</pre>;
      break;
    case "memory":
      label = (
        <>
          <Verb>{ui.detail === "Forgot" ? "Forgot" : ui.detail === "Not found" ? "Couldn't find in memory" : ui.detail === "Already remembered" ? "Already remembered" : "Remembered"}</Verb> <span className="tool-obj">{ui.title}</span>
        </>
      );
      break;
    case "system":
      label = <Verb>Looked at this computer</Verb>;
      body = <pre className="tool-pre">{ui.output}</pre>;
      break;
    case "clipboard":
      label = <span className="tool-obj">{ui.title}</span>;
      body = ui.output ? <pre className="tool-pre">{ui.output}</pre> : null;
      break;
    case "open":
      label = (
        <>
          <Verb>Opened</Verb> <FileName path={ui.path} /> <Verb>in its app</Verb>
        </>
      );
      break;
    case "trash":
      label = (
        <>
          <Verb>Moved</Verb> <span className="tool-obj">{ui.title}</span> <Verb>to the Recycle Bin</Verb>
        </>
      );
      break;
    case "denied":
      label = (
        <>
          <Verb>Declined</Verb> <span className="tool-obj mono cmd-inline">{ui.title}</span>
        </>
      );
      break;
    case "error":
      label = (
        <>
          <span className="tool-obj">{ui.title}</span> <Verb>failed:</Verb> <span className="err-text">{ui.error}</span>
        </>
      );
      break;
  }
  const hasBody = !!body;
  return (
    <div className={`tool tool-${ui.kind} ${ui.running ? "running" : ""}`}>
      <div className={`tool-head ${hasBody ? "clickable" : ""}`} onClick={() => hasBody && setOpen(!open)}>
        {ui.running && <span className="spinner" />}
        <span className="tool-label">{label}</span>
        {hasBody && <span className={`tool-tri ${open ? "open" : ""}`} />}
        <span className="grow" />
        {extra}
      </div>
      {ui.safety?.map((r) => <Safety key={r.repo} r={r} />)}
      {hasBody && open && <div className="tool-body">{body}</div>}
    </div>
  );
}

/** The GitHub safety check of a repository the call fetches or reads. */
function Safety({ r }: { r: NonNullable<ToolUI["safety"]>[number] }) {
  const [open, setOpen] = useState(r.verdict === "risky");
  const word = { ok: "Checked", caution: "Use with care", risky: "Looks unsafe" }[r.verdict];
  return (
    <div className={`repo-safety ${r.verdict}`}>
      <button className="repo-safety-head" onClick={() => setOpen(!open)}>
        <Icon name={r.verdict === "ok" ? "shield" : "info"} size={13} stroke={2.4} />
        <span className="repo-safety-word">{word}</span>
        <span className="tool-verb">{r.repo} on GitHub</span>
        <span className={`tool-tri ${open ? "open" : ""}`} />
      </button>
      {open && (
        <ul className="repo-safety-list">
          {r.reasons.map((x, i) => (
            <li key={i}>{x}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
