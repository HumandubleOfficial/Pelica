// A tool call in the transcript, as the Claude app shows them: created and
// edited files with their diffs, commands with output, searches with
// their sources, the todo list.
import { useMemo, useState } from "react";
import hljs from "highlight.js/lib/common";
import { bridge } from "../lib/bridge";
import { diffLines, diffStats, hunks } from "../lib/diff";
import { openLink, useApp, type ToolUI } from "../state";
import { Icon } from "./Icons";
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

function Diff({ before, after, path }: { before: string | null; after: string; path?: string }) {
  const ext = (path?.split(".").pop() || "").toLowerCase();
  const lang = hljs.getLanguage(EXT_LANG[ext] || ext) ? EXT_LANG[ext] || ext : "";
  const rows = useMemo(() => {
    const lines = diffLines(before ?? "", after);
    const h = before === null ? lines.slice(0, 400).map((l) => ({ ...l, kind: "+" as const })) : hunks(lines, 3).slice(0, 600);
    return { h: h.map((l) => (l === null ? null : { ...l, html: lineHtml(l.text, lang) })), more: before === null && lines.length > 400 ? lines.length - 400 : 0 };
  }, [before, after, lang]);
  const created = before === null;
  return (
    <div className={`diff ${created ? "created" : ""}`}>
      {rows.h.map((l, i) =>
        l === null ? (
          <div key={i} className="diff-gap">
            <Icon name="more" size={12} />
          </div>
        ) : (
          <div key={i} className={`diff-line ${l.kind === "+" && !created ? "add" : l.kind === "-" ? "del" : ""}`}>
            {!created && <span className="diff-no">{l.kind === "+" ? "" : l.a}</span>}
            <span className="diff-no">{l.kind === "-" ? "" : l.b}</span>
            {!created && <span className="diff-sign">{l.kind === " " ? "" : l.kind === "-" ? "−" : "+"}</span>}
            <span className="diff-text hljs" dangerouslySetInnerHTML={{ __html: l.html }} />
          </div>
        ),
      )}
      {rows.more > 0 && <div className="diff-gap">… {rows.more} more lines</div>}
    </div>
  );
}

/** GitHub's five little squares: how much of the change is added vs removed. */
function DiffBar({ added, removed }: { added: number; removed: number }) {
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

function FileName({ path }: { path?: string }) {
  const folder = useApp((s) => s.chat?.folder);
  if (!path) return null;
  const name = path.split("/").pop();
  const dir = path.slice(0, path.length - (name?.length || 0));
  return (
    <button className="file-name" title={`Show ${path} in Explorer`} onClick={(e) => (e.stopPropagation(), folder && bridge?.fs.reveal(folder, path))}>
      <span className="file-dir">{dir}</span>
      {name}
    </button>
  );
}

export function ToolCard({ ui }: { ui: ToolUI }) {
  const isFile = ui.kind === "write" || ui.kind === "edit";
  const [open, setOpen] = useState(ui.kind === "edit" || ui.kind === "command" || ui.kind === "todo" || (ui.kind === "write" && ui.before !== null));
  const stats = useMemo(() => (isFile && ui.after !== undefined ? diffStats(diffLines(ui.before ?? "", ui.after)) : null), [isFile, ui.before, ui.after]);

  let icon = "info";
  let label: React.ReactNode = ui.title;
  let body: React.ReactNode = null;
  switch (ui.kind) {
    case "search":
      icon = "globe";
      label = (
        <>
          Searched the web <span className="muted">“{ui.title}”</span>
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
      icon = ui.kind === "browser" ? "external" : "file";
      label = (
        <>
          {ui.kind === "browser" ? "Opened in browser" : "Read"}{" "}
          <button className="link-btn" onClick={() => ui.url && openLink(ui.url)}>
            {ui.title?.slice(0, 80)}
          </button>{" "}
          <span className="muted">{ui.url && host(ui.url)}</span>
        </>
      );
      break;
    case "read":
      icon = "file";
      label = (
        <>
          Read <FileName path={ui.path} /> <span className="muted">{ui.detail}</span>
        </>
      );
      break;
    case "write":
    case "edit":
      icon = ui.kind === "edit" ? "edit" : "file";
      label = (
        <>
          {ui.kind === "edit" ? "Edited" : ui.before === null ? "Created" : "Rewrote"} <FileName path={ui.path} />
          {stats && (
            <span className="diffstat">
              <span className="add">+{stats.added}</span> {stats.removed > 0 && <span className="del">−{stats.removed}</span>}
              {ui.before !== null && <DiffBar added={stats.added} removed={stats.removed} />}
            </span>
          )}
          <ViewFile path={ui.path} text={ui.after} />
        </>
      );
      body = ui.after !== undefined ? <Diff before={ui.before ?? null} after={ui.after} path={ui.path} /> : null;
      break;
    case "list":
      icon = "folder";
      label = (
        <>
          Listed <span className="mono">{ui.title}</span>
        </>
      );
      body = <pre className="tool-pre">{ui.output}</pre>;
      break;
    case "grep":
      icon = "search";
      label = (
        <>
          Searched files for <span className="mono">{ui.title}</span>
        </>
      );
      body = <pre className="tool-pre">{ui.output}</pre>;
      break;
    case "command":
      icon = "terminal";
      label = (
        <>
          {ui.running ? "Running" : ui.detail === "Waiting for your approval…" ? "Wants to run" : "Ran"} <span className="mono cmd-inline">{ui.title}</span>
          {!ui.running && ui.code !== undefined && ui.code !== null && <span className={`exit ${ui.code === 0 ? "ok" : "bad"}`}>{ui.code === 0 ? "✓" : `exit ${ui.code}`}</span>}
        </>
      );
      body = (ui.output || ui.running) && <pre className="tool-pre term">{ui.output || "…"}</pre>;
      break;
    case "todo":
      icon = "list";
      label = <>Plan · {ui.title}</>;
      body = (
        <ul className="todos">
          {ui.todos?.map((t, i) => (
            <li key={i} className={t.status}>
              <span className="todo-box">{t.status === "completed" ? <Icon name="check" size={11} stroke={2.6} /> : t.status === "in_progress" ? <span className="todo-dot" /> : null}</span>
              {t.content}
            </li>
          ))}
        </ul>
      );
      break;
    case "mcp":
      icon = "plug";
      label = <>{ui.title}</>;
      body = <pre className="tool-pre">{ui.output}</pre>;
      break;
    case "system":
      icon = "monitor";
      label = <>Looked at this computer</>;
      body = <pre className="tool-pre">{ui.output}</pre>;
      break;
    case "clipboard":
      icon = "clipboard";
      label = <>{ui.title}</>;
      body = ui.output ? <pre className="tool-pre">{ui.output}</pre> : null;
      break;
    case "open":
      icon = "external";
      label = (
        <>
          Opened <FileName path={ui.path} /> <span className="muted">in its app</span>
        </>
      );
      break;
    case "trash":
      icon = "trash";
      label = (
        <>
          Moved <span className="mono">{ui.title}</span> to the Recycle Bin
        </>
      );
      break;
    case "denied":
      icon = "shield";
      label = (
        <>
          Declined <span className="mono cmd-inline">{ui.title}</span>
        </>
      );
      break;
    case "error":
      icon = "x";
      label = (
        <>
          <span className="mono">{ui.title}</span> failed: <span className="err-text">{ui.error}</span>
        </>
      );
      break;
  }
  const hasBody = !!body;
  return (
    <div className={`tool tool-${ui.kind} ${ui.running ? "running" : ""}`}>
      <div className={`tool-head ${hasBody ? "clickable" : ""}`} onClick={() => hasBody && setOpen(!open)}>
        <span className="tool-icon">{ui.running ? <span className="spinner" /> : <Icon name={icon} size={14} />}</span>
        <span className="tool-label">{label}</span>
        {hasBody && <Icon name={open ? "down" : "chevron"} size={13} className="tool-caret" />}
      </div>
      {hasBody && open && <div className="tool-body">{body}</div>}
    </div>
  );
}
