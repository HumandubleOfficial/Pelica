// The tool pane on the right, opened from the title bar: a terminal in the
// chat's folder, what this chat changed, a browser, and the folder's files.
import { createElement, useEffect, useMemo, useRef, useState } from "react";
import { pickFolder } from "../agent/commands";
import { diffLines, diffStats } from "../lib/diff";
import { bridge } from "../lib/bridge";
import { app, createStore, uid, useApp, useStore, type Dock as DockKind, type Msg } from "../state";
import { openArtifact } from "./ArtifactPanel";
import { Icon } from "./Icons";
import { Diff, DiffBar } from "./ToolCard";

export const DOCKS: { id: DockKind; icon: string; label: string }[] = [
  { id: "terminal", icon: "prompt", label: "Terminal" },
  { id: "changes", icon: "diff", label: "Changes" },
  { id: "browser", icon: "globe", label: "Browser" },
  { id: "files", icon: "folder", label: "Files" },
];

const base = (p: string) => p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;
const win = (p: string) => p.replace(/\//g, "\\");

function NoFolder({ what }: { what: string }) {
  return (
    <div className="dock-empty">
      <Icon name="folder" size={22} />
      <div>{what} works in this chat's folder.</div>
      <button className="btn small" onClick={() => pickFolder()}>
        Choose a folder
      </button>
    </div>
  );
}

// ---------- terminal ----------

interface Block {
  id: string;
  cmd: string;
  cwd: string;
  out: string;
  /** undefined while it runs */
  code?: number | null;
}

// kept while Pelica runs, so closing the pane doesn't lose the session
const term = createStore({ blocks: [] as Block[], root: "", cwd: "", history: [] as string[] });
let listening = false;
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07/g;
const strip = (s: string) => s.replace(ANSI, "").replace(/\r\n/g, "\n");
const setBlock = (id: string, f: (b: Block) => Partial<Block>) => term.set((s) => ({ blocks: s.blocks.map((b) => (b.id === id ? { ...b, ...f(b) } : b)) }));

async function runLine(root: string, cwd: string, ps: boolean, line: string) {
  const t = line.trim();
  if (!t || !bridge) return;
  term.set((s) => ({ history: [...s.history.filter((h) => h !== t), t].slice(-100) }));
  if (/^(cls|clear)$/i.test(t)) return term.set({ blocks: [] });
  if (!listening) {
    listening = true;
    bridge.cmd.onData(({ id, data }) => id.startsWith("term-") && setBlock(id, (b) => ({ out: (b.out + strip(data)).slice(-200000) })));
  }
  const id = `term-${uid()}`;
  term.set((s) => ({ blocks: [...s.blocks, { id, cmd: t, cwd, out: "" }].slice(-200) }));
  // each line is its own process, so the folder is set first; cd prints where it ended up
  const cd = /^(?:cd|chdir|set-location|sl)(?:\s+\/d)?\s+(.+)$/i.exec(t);
  const q = (p: string) => (ps ? `'${p.replace(/'/g, "''")}'` : `"${p}"`);
  const at = cwd.toLowerCase() === root.toLowerCase() ? "" : ps ? `Set-Location -LiteralPath ${q(cwd)}; ` : `cd /d ${q(cwd)} && `;
  const full = cd ? (ps ? `${at}Set-Location ${cd[1]}; (Get-Location).Path` : `${at}cd /d ${cd[1]} && cd`) : at + t;
  try {
    const r = await bridge.cmd.run(id, root, full, 30 * 60 * 1000, false);
    const out = strip(r.output);
    if (cd && r.code === 0) {
      const to = out.trim().split("\n").pop()?.trim();
      if (to) term.set({ cwd: to });
      setBlock(id, () => ({ out: "", code: 0 }));
    } else setBlock(id, () => ({ out: out + (r.timedOut ? "\n(stopped after 30 minutes)" : ""), code: r.code }));
  } catch (e: any) {
    setBlock(id, (b) => ({ out: b.out + String(e.message || e), code: -1 }));
  }
}

function Terminal() {
  const folder = useApp((s) => s.chat?.folder ?? null);
  const ps = useApp((s) => s.settings.shell === "powershell");
  const { blocks, root, cwd: cwd0, history } = useStore(term, (s) => s);
  const [input, setInput] = useState("");
  const [hi, setHi] = useState(-1);
  const box = useRef<HTMLDivElement>(null);
  const inp = useRef<HTMLInputElement>(null);
  // a new folder starts a fresh session
  useEffect(() => {
    if (folder && root !== folder) term.set({ root: folder, cwd: folder, blocks: [] });
  }, [folder, root]);
  useEffect(() => {
    box.current?.scrollTo({ top: box.current.scrollHeight });
  }, [blocks]);
  useEffect(() => inp.current?.focus(), []);
  if (!folder) return <NoFolder what="The terminal" />;
  if (!bridge) return <div className="dock-empty">The terminal needs the Pelica desktop app.</div>;
  const cwd = cwd0 || folder;
  const running = blocks.find((b) => b.code === undefined);
  const prompt = (p: string) => `${ps ? "PS " : ""}${win(p)}>`;
  const key = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !running) {
      runLine(folder, cwd, ps, input);
      setInput("");
      setHi(-1);
    } else if (e.key === "c" && e.ctrlKey && running && !window.getSelection()?.toString()) {
      e.preventDefault();
      bridge!.cmd.kill(running.id);
    } else if (e.key === "l" && e.ctrlKey) {
      e.preventDefault();
      term.set({ blocks: [] });
    } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      e.preventDefault();
      const n = e.key === "ArrowUp" ? (hi < 0 ? history.length - 1 : Math.max(0, hi - 1)) : hi < 0 ? -1 : hi + 1;
      if (n >= history.length || n < 0) return (setHi(-1), setInput(""));
      setHi(n);
      setInput(history[n]);
    }
  };
  return (
    <div className="term" ref={box} onMouseUp={() => !window.getSelection()?.toString() && inp.current?.focus()}>
      {!blocks.length && <div className="term-note">{ps ? "PowerShell" : "Command Prompt"} in {win(folder)}. Programs that wait for typed input aren't supported.</div>}
      {blocks.map((b) => (
        <div key={b.id} className="term-block">
          <div className="term-cmd">
            <span className="term-ps">{prompt(b.cwd)}</span> {b.cmd}
          </div>
          {b.out && <pre className="term-out">{b.out}</pre>}
          {b.code !== undefined && b.code !== 0 && b.code !== null && <div className="term-code">exit code {b.code}</div>}
        </div>
      ))}
      <div className="term-line">
        {running ? (
          <span className="term-ps">
            Running… <kbd>Ctrl+C</kbd> stops it
          </span>
        ) : (
          <span className="term-ps">{prompt(cwd)}</span>
        )}
        <input ref={inp} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={key} spellCheck={false} aria-label="Command" />
      </div>
    </div>
  );
}

// ---------- changes ----------

interface Change {
  path: string;
  before: string | null;
  after: string;
}

/** Every file this chat wrote or edited: as it was first, and as it is now. */
export function useChanges(): Change[] {
  const msgs = useApp((s) => s.chat?.messages);
  const archive = useApp((s) => s.chat?.archive);
  return useMemo(() => {
    const m = new Map<string, Change>();
    for (const x of [...(archive || []), ...(msgs || [])] as Msg[]) {
      if (x.role !== "tool" || (x.ui.kind !== "write" && x.ui.kind !== "edit") || !x.ui.path || x.ui.after === undefined) continue;
      const had = m.get(x.ui.path);
      m.delete(x.ui.path);
      m.set(x.ui.path, { path: x.ui.path, before: had ? had.before : (x.ui.before ?? null), after: x.ui.after });
    }
    return [...m.values()].reverse();
  }, [msgs, archive]);
}

function ChangeFile({ c }: { c: Change }) {
  const folder = useApp((s) => s.chat?.folder ?? null);
  const [open, setOpen] = useState(true);
  const st = useMemo(() => diffStats(diffLines(c.before ?? "", c.after)), [c]);
  const name = c.path.split("/").pop() || c.path;
  return (
    <div className="chg-file">
      <div className="chg-head" onClick={() => setOpen(!open)}>
        <Icon name={open ? "down" : "chevron"} size={12} className="muted" />
        <span className="chg-name ellipsis" title={c.path}>
          <span className="muted">{c.path.slice(0, c.path.length - name.length)}</span>
          {name}
        </span>
        {c.before === null && <span className="sp-new">new</span>}
        <span className="diffstat">
          <span className="add">+{st.added}</span> <span className="del">−{st.removed}</span>
        </span>
        <DiffBar added={st.added} removed={st.removed} />
        {folder && bridge && (
          <button className="sp-link" title="Show in Explorer" onClick={(e) => (e.stopPropagation(), bridge!.fs.reveal(folder, c.path))}>
            <Icon name="external" size={12} />
          </button>
        )}
      </div>
      {open && <Diff before={c.before} after={c.after} path={c.path} />}
    </div>
  );
}

function Changes() {
  const files = useChanges();
  const total = useMemo(() => files.reduce((a, c) => {
    const s = diffStats(diffLines(c.before ?? "", c.after));
    return { added: a.added + s.added, removed: a.removed + s.removed };
  }, { added: 0, removed: 0 }), [files]);
  if (!files.length) return <div className="dock-empty">Files Pelica creates or edits in this chat show up here, with what changed.</div>;
  return (
    <div className="chg">
      <div className="chg-sum">
        {files.length} file{files.length === 1 ? "" : "s"} changed
        <span className="diffstat">
          <span className="add">+{total.added}</span> <span className="del">−{total.removed}</span>
        </span>
      </div>
      {files.map((c) => (
        <ChangeFile key={c.path} c={c} />
      ))}
    </div>
  );
}

// ---------- browser ----------

let lastUrl = "";

function toUrl(text: string): string {
  const t = text.trim();
  if (/^[a-z][\w+.-]*:\/\//i.test(t)) return t;
  if (/^(localhost|127\.\d+\.\d+\.\d+)(:\d+)?(\/|$)/i.test(t)) return `http://${t}`;
  if (/^[\w-]+(\.[\w-]+)+(:\d+)?(\/\S*)?$/.test(t)) return `https://${t}`;
  return `https://duckduckgo.com/?q=${encodeURIComponent(t)}`;
}

function Browser() {
  const [url, setUrl] = useState(lastUrl);
  const [input, setInput] = useState(lastUrl);
  const [loading, setLoading] = useState(false);
  const view = useRef<any>(null);
  const [nav, setNav] = useState({ back: false, fwd: false });
  useEffect(() => {
    const v = view.current;
    if (!v || !bridge) return;
    const moved = (e: any) => {
      if (!e.isMainFrame && e.isMainFrame !== undefined) return;
      lastUrl = e.url;
      setInput(e.url);
      setNav({ back: v.canGoBack(), fwd: v.canGoForward() });
    };
    const start = () => setLoading(true);
    const stop = () => setLoading(false);
    v.addEventListener("did-navigate", moved);
    v.addEventListener("did-navigate-in-page", moved);
    v.addEventListener("did-start-loading", start);
    v.addEventListener("did-stop-loading", stop);
    return () => {
      v.removeEventListener("did-navigate", moved);
      v.removeEventListener("did-navigate-in-page", moved);
      v.removeEventListener("did-start-loading", start);
      v.removeEventListener("did-stop-loading", stop);
    };
  }, [url]);
  const go = (text: string) => {
    if (!text.trim()) return;
    const u = toUrl(text);
    setInput(u);
    lastUrl = u;
    if (view.current && url && bridge) view.current.loadURL(u);
    else setUrl(u);
  };
  const v = () => view.current;
  return (
    <div className="brw">
      <div className="brw-bar">
        <button className="sp-link" disabled={!nav.back} onClick={() => v()?.goBack()} title="Back">
          <Icon name="back" size={14} />
        </button>
        <button className="sp-link" disabled={!nav.fwd} onClick={() => v()?.goForward()} title="Forward">
          <Icon name="forward" size={14} />
        </button>
        <button className="sp-link" disabled={!url} onClick={() => (loading ? v()?.stop() : v()?.reload())} title={loading ? "Stop" : "Reload"}>
          <Icon name={loading ? "x" : "reload"} size={13} />
        </button>
        <input className="brw-url" value={input} placeholder="Search or type an address" onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && go(input)} onFocus={(e) => e.target.select()} spellCheck={false} autoFocus={!url} />
        <button className="sp-link" disabled={!url} onClick={() => bridge?.sys.external(input)} title="Open in your browser">
          <Icon name="external" size={13} />
        </button>
      </div>
      <div className="brw-body">
        {!url ? (
          <div className="dock-empty">
            <Icon name="globe" size={22} />
            <div>Search, or open a page or your dev server (like localhost:5173).</div>
          </div>
        ) : bridge ? (
          // (React has no type for Electron's <webview>)
          createElement("webview", { key: url, ref: view, src: url, partition: "persist:pelica-browser", allowpopups: "true", className: "brw-view" })
        ) : (
          <iframe key={url} src={url} className="brw-view" title="Browser" />
        )}
      </div>
    </div>
  );
}

// ---------- files ----------

interface Ent {
  name: string;
  dir: boolean;
  size?: string;
  skipped?: boolean;
}

function parseList(t: string): Ent[] {
  return t
    .split("\n")
    .slice(1)
    .filter((l) => l && !l.startsWith("(empty)") && !l.startsWith("... "))
    .map((l) => {
      if (l.endsWith("/ (skipped)")) return { name: l.slice(0, -11), dir: true, skipped: true };
      if (l.endsWith("/")) return { name: l.slice(0, -1), dir: true };
      const m = /^(.*) \(([^)]*)\)$/.exec(l);
      return m ? { name: m[1], dir: false, size: m[2] } : { name: l, dir: false };
    });
}

function Dir({ root, rel, depth, nonce }: { root: string; rel: string; depth: number; nonce: number }) {
  const [ents, setEnts] = useState<Ent[] | null>(null);
  const [err, setErr] = useState("");
  const [open, setOpen] = useState<Set<string>>(new Set());
  const flip = (n: string) =>
    setOpen((s) => {
      const t = new Set(s);
      if (!t.delete(n)) t.add(n);
      return t;
    });
  useEffect(() => {
    bridge?.fs
      .list(root, rel, 1)
      .then((t) => (setEnts(parseList(t)), setErr("")))
      .catch((e) => setErr(String(e.message || e)));
  }, [root, rel, nonce]);
  if (err) return <div className="files-note" style={{ paddingLeft: 10 + depth * 14 }}>{err}</div>;
  if (!ents) return null;
  if (!ents.length && depth) return <div className="files-note" style={{ paddingLeft: 10 + depth * 14 }}>Empty</div>;
  const path = (n: string) => (rel === "." ? n : `${rel}/${n}`);
  const show = async (e: Ent) => {
    try {
      const text = await bridge!.fs.raw(root, path(e.name));
      openArtifact(e.name, text, (e.name.split(".").pop() || "").toLowerCase());
    } catch (x: any) {
      app.set({ toast: String(x.message || x) });
    }
  };
  return (
    <>
      {ents.map((e) => {
        const isOpen = open.has(e.name);
        return (
          <div key={e.name}>
            <div
              className={`files-row ${e.skipped ? "muted" : ""}`}
              style={{ paddingLeft: 8 + depth * 14 }}
              title={path(e.name)}
              onClick={() => (e.dir ? !e.skipped && flip(e.name) : show(e))}
            >
              <Icon name={e.dir ? (isOpen ? "down" : "chevron") : "file"} size={e.dir ? 11 : 13} className="files-ico" />
              {e.dir && <Icon name="folder" size={13} className="files-ico" />}
              <span className="ellipsis">{e.name}</span>
              {e.size && <span className="files-size">{e.size}</span>}
              <button className="sp-link" title="Show in Explorer" onClick={(x) => (x.stopPropagation(), bridge!.fs.reveal(root, path(e.name)))}>
                <Icon name="external" size={11} />
              </button>
            </div>
            {e.dir && isOpen && <Dir root={root} rel={path(e.name)} depth={depth + 1} nonce={nonce} />}
          </div>
        );
      })}
    </>
  );
}

function Files() {
  const folder = useApp((s) => s.chat?.folder ?? null);
  const [nonce, setNonce] = useState(0);
  if (!folder) return <NoFolder what="The file list" />;
  if (!bridge) return <div className="dock-empty">Files need the Pelica desktop app.</div>;
  return (
    <div className="files">
      <div className="files-head">
        <Icon name="folder" size={14} />
        <b className="ellipsis" title={folder}>
          {base(folder)}
        </b>
        <span className="grow" />
        <button className="sp-link" title="Refresh" onClick={() => setNonce(nonce + 1)}>
          <Icon name="reload" size={12} />
        </button>
        <button className="sp-link" title="Open in Explorer" onClick={() => bridge!.sys.reveal(folder)}>
          <Icon name="external" size={12} />
        </button>
      </div>
      <div className="files-tree">
        <Dir root={folder} rel="." depth={0} nonce={nonce} />
      </div>
    </div>
  );
}

// ---------- the pane ----------

export function Dock() {
  const dock = useApp((s) => s.dock);
  const [width, setWidth] = useState(460);
  if (!dock) return null;
  return (
    <div className="dock" style={{ width }}>
      <div
        className="browser-resize"
        onMouseDown={(e) => {
          const x0 = e.clientX;
          const w0 = width;
          // (the browser pane would swallow the mouse moves while dragging)
          document.body.classList.add("resizing");
          const mv = (m: MouseEvent) => setWidth(Math.max(320, Math.min(window.innerWidth - 460, w0 + (x0 - m.clientX))));
          const up = () => (document.body.classList.remove("resizing"), window.removeEventListener("mousemove", mv), window.removeEventListener("mouseup", up));
          window.addEventListener("mousemove", mv);
          window.addEventListener("mouseup", up);
        }}
      />
      <div className="dock-bar">
        {DOCKS.map((d) => (
          <button key={d.id} className={`dock-tab ${d.id === dock ? "sel" : ""}`} onClick={() => app.set({ dock: d.id })}>
            <Icon name={d.icon} size={14} /> {d.label}
          </button>
        ))}
        <span className="grow" />
        <button className="sp-link" title="Close" onClick={() => app.set({ dock: null })}>
          <Icon name="x" size={14} />
        </button>
      </div>
      <div className="dock-body">{dock === "terminal" ? <Terminal /> : dock === "changes" ? <Changes /> : dock === "browser" ? <Browser /> : <Files />}</div>
    </div>
  );
}
