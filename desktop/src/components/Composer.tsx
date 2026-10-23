// The message box: text, slash commands, and the chips under it (folder,
// web search, effort, context, model).
import { useEffect, useMemo, useRef, useState } from "react";
import { send, stop } from "../agent/agent";
import { COMMANDS, command, pickFolder } from "../agent/commands";
import { bridge } from "../lib/bridge";
import { engine, type LocalModel } from "../lib/engine";
import { app, openChat, PERM_MODES, setChatFolder, setSettings, toast, useApp } from "../state";
import { ContextMeter } from "./ContextMeter";
import { EffortPicker } from "./Effort";
import { Icon } from "./Icons";
import { AttachChips, joinAttachments, photoText, readFiles, type Attached } from "./Attachments";

function ModelChip() {
  const st = useApp((s) => s.status);
  const lastCtx = useApp((s) => s.settings.lastCtx);
  const [open, setOpen] = useState(false);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [ctx, setCtx] = useState(lastCtx);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const f = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", f);
    return () => document.removeEventListener("mousedown", f);
  }, []);
  useEffect(() => {
    if (!open) return;
    engine.models().then((m) => setModels(m.models)).catch(() => {});
    // start from what's loaded, not from what was picked last time
    setCtx(st?.state === "ready" && st.ctx ? st.ctx : lastCtx);
  }, [open]);
  const label = st?.state === "loading" ? `Loading ${st.model || ""}… ${st.progress ? Math.round(st.progress) + "%" : ""}` : st?.state === "ready" ? st.model : st?.state === "error" ? "Load failed" : "No model";
  const cur = models.find((m) => m.path === st?.path);
  const cap = (m?: LocalModel) => (m?.max_ctx ? Math.min(ctx, m.max_ctx) : ctx);
  // the loaded model at a different context: it has to load again
  const changed = st?.state === "ready" && !!st.path && cap(cur) !== st.ctx;
  async function load(m: { path: string; max_ctx?: number | null }) {
    if (app.get().running) return toast("Wait for the reply to finish (or stop it) before loading a model.");
    try {
      await engine.load(m.path, cap(m as LocalModel));
      setSettings({ lastModel: m.path, lastCtx: ctx });
      setOpen(false);
    } catch (e: any) {
      toast(e.message === "busy" ? "Busy: wait for the reply or the load to finish." : e.message);
    }
  }
  return (
    <div className="pop-wrap" ref={ref}>
      <button className={`foot-btn model-chip ${st?.state || ""}`} onClick={() => setOpen(!open)} title={st?.gpu ? `${st.model} on ${st.gpu}` : "Choose a model"}>
        {st?.state !== "ready" && <span className={`status-dot ${st?.state || "empty"}`} />}
        {label?.replace(/\s*\((?:I?Q\d|F16|BF16|F32)[^)]*\)\s*$/i, "")}
      </button>
      {open && (
        <div className="popover up right model-pop">
          <div className="pop-title">
            Models on this PC
            <label className="ctx-select">
              context
              <select value={ctx} onChange={(e) => setCtx(Number(e.target.value))}>
                {[4096, 8192, 16384, 32768, 65536, 131072, 262144].map((n) => (
                  <option key={n} value={n}>
                    {n >= 1024 ? `${n / 1024}K` : n}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {changed && (
            <div className="ctx-apply">
              <span className="muted small">
                {st!.model} runs at {Math.round((st!.ctx || 0) / 1024)}K. {cur?.max_ctx && ctx > cur.max_ctx ? `It goes up to ${Math.round(cur.max_ctx / 1024)}K.` : ""}
              </span>
              <button className="btn primary small" onClick={() => load({ path: st!.path!, max_ctx: cur?.max_ctx || st!.max_ctx })}>
                Apply {Math.round(cap(cur) / 1024)}K (reloads)
              </button>
            </div>
          )}
          {models.length === 0 && <div className="muted small pad">No models found. Get one from the library.</div>}
          {models.map((m) => (
            <button key={m.path} className={`model-row ${st?.path === m.path ? "sel" : ""}`} onClick={() => load(m)} disabled={st?.state === "loading"}>
              <span className="model-name">{m.name}</span>
              <span className="muted small">
                {m.kind === "moe" ? "MoE" : m.kind === "gpu" ? "gpt-oss" : "tiered"} · {m.gb.toFixed(1)} GB{m.max_ctx ? ` · up to ${Math.round(m.max_ctx / 1024)}K` : ""}
              </span>
              {st?.path === m.path && <Icon name="check" size={14} />}
            </button>
          ))}
          <button className="model-row lib" onClick={() => (setOpen(false), app.set({ view: "library" }))}>
            <Icon name="library" size={14} /> Model library…
          </button>
        </div>
      )}
    </div>
  );
}

/** Shown when files and commands are on but the model was never trained to use tools. */
function AgentHint({ folder }: { folder: string | null }) {
  const st = useApp((s) => s.status);
  const [hidden, setHidden] = useState(false);
  if (hidden || !folder || st?.state !== "ready" || st.agent !== false) return null;
  return (
    <div className="agent-hint">
      <Icon name="info" size={13} />
      <span>
        {st.model} wasn&apos;t trained to use tools, so it may explain how to do a task instead of doing it. For building and editing files, load a tool-trained model such as Qwen3-30B-A3B.
      </span>
      <button className="ghost-btn" onClick={() => app.set({ view: "library" })}>
        Models
      </button>
      <button className="ghost-btn" title="Hide" onClick={() => setHidden(true)}>
        <Icon name="x" size={12} />
      </button>
    </div>
  );
}

/** A popover's open state, closed by a click outside it. */
function usePop() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const f = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const k = (e: KeyboardEvent) => e.key === "Escape" && (e.stopPropagation(), setOpen(false));
    document.addEventListener("mousedown", f);
    document.addEventListener("keydown", k, true);
    return () => {
      document.removeEventListener("mousedown", f);
      document.removeEventListener("keydown", k, true);
    };
  }, [open]);
  return { open, setOpen, ref };
}

const base = (p: string) => p.split(/[\\/]/).filter(Boolean).pop() || p;

/** This chat's folder: one of the recent ones, a new one, or none. */
export function FolderList({ folder, done }: { folder: string | null; done: () => void }) {
  const recent = useApp((s) => s.settings.recentFolders);
  return (
    <>
      {recent.map((p) => (
        <button key={p} className={`menu-row ${p === folder ? "sel" : ""}`} onClick={() => (setChatFolder(p), done())} title={p}>
          <Icon name="folder" size={14} />
          <span className="menu-text">
            <b>{base(p)}</b>
            <span className="muted small mono ellipsis">{p}</span>
          </span>
          {p === folder && <Icon name="check" size={14} />}
        </button>
      ))}
      <button className="menu-row" onClick={() => (done(), pickFolder())}>
        <Icon name="plus" size={14} />
        <span className="menu-text">Choose a folder…</span>
      </button>
      {folder && (
        <button className="menu-row" onClick={() => (setChatFolder(null), done())}>
          <Icon name="x" size={14} />
          <span className="menu-text">No folder (just chat)</span>
        </button>
      )}
    </>
  );
}

/** The + menu: everything that isn't typing or sending. */
function PlusMenu({ folder, onAttach }: { folder: string | null; onAttach: () => void }) {
  const web = useApp((s) => s.settings.web);
  const { open, setOpen, ref } = usePop();
  const [sub, setSub] = useState<"folder" | null>(null);
  const close = () => (setOpen(false), setSub(null));
  const row = (icon: string, label: string, hint: string, fn: () => void, extra?: React.ReactNode) => (
    <button className="menu-row" onClick={fn}>
      <Icon name={icon} size={15} />
      <span className="menu-text">
        <b>{label}</b>
        {hint && <span className="muted small">{hint}</span>}
      </span>
      {extra}
    </button>
  );
  return (
    <div className="pop-wrap" ref={ref}>
      <button className={`plus-btn ${open ? "open" : ""}`} onClick={() => (open ? close() : setOpen(true))} title="Attach, folder, web search, prompt maker…">
        <Icon name="plus" size={18} />
      </button>
      {open && (
        <div className="popover up plus-pop">
          {sub === "folder" ? (
            <>
              <button className="menu-back" onClick={() => setSub(null)}>
                <Icon name="back" size={14} /> Folder for this chat
              </button>
              <FolderList folder={folder} done={close} />
              <div className="muted small pad">Each chat has its own folder: other chats keep theirs.</div>
            </>
          ) : (
            <>
              {row("clip", "Add files or photos", "or drop or paste them", () => (close(), onAttach()))}
              {row("folder", folder ? base(folder) : "Choose a folder", folder ? "This chat's folder" : "Let Pelica work on files", () => setSub("folder"), <Icon name="chevron" size={13} />)}
              {row("globe", "Web search", web ? "On: searches when a question needs it" : "Off", () => setSettings({ web: !web }), <span className={`toggle ${web ? "on" : ""}`} />)}
              <div className="menu-sep" />
              {row("wand", "Prompt maker", "Answer a few questions, get a tuned prompt", () => (close(), app.set({ promptMaker: true })))}
              {row("command", "Commands", "/compact, /cqs, /export…", () => (close(), window.dispatchEvent(new CustomEvent("pelica:compose", { detail: "/" }))))}
              {row("compare", "Compare models", "With Astra, Fable, Opus, Kimi K3, GLM 5.3", () => (close(), app.set({ compare: "" })))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** What the agent may do without asking (Shift+Tab in the box cycles). */
function PermChip() {
  const mode = useApp((s) => s.settings.permMode);
  const { open, setOpen, ref } = usePop();
  const cur = PERM_MODES.find((m) => m.id === mode) || PERM_MODES[1];
  return (
    <div className="pop-wrap" ref={ref}>
      <button className={`foot-btn perm-chip perm-${mode}`} onClick={() => setOpen(!open)} title={`${cur.label}: ${cur.desc} (Shift+Tab to switch)`}>
        {cur.label}
      </button>
      {open && (
        <div className="popover up perm-pop">
          <div className="pop-title">
            Permissions <kbd>Shift+Tab</kbd>
          </div>
          {PERM_MODES.map((m) => (
            <button key={m.id} className={`perm-row ${m.id === mode ? "sel" : ""}`} onClick={() => (setSettings({ permMode: m.id }), setOpen(false))}>
              <span className={`perm-ico perm-${m.id}`}>
                <Icon name={m.id === "plan" ? "list" : m.id === "auto" ? "bolt" : m.id === "ask" ? "lock" : "shield"} size={14} />
              </span>
              <span className="perm-text">
                <b>{m.label}</b>
                <span className="muted small">{m.desc}</span>
              </span>
              {m.id === mode && <Icon name="check" size={14} />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function cyclePermMode() {
  const i = PERM_MODES.findIndex((m) => m.id === app.get().settings.permMode);
  const next = PERM_MODES[(i + 1) % PERM_MODES.length];
  setSettings({ permMode: next.id });
  toast(`${next.label}: ${next.desc}`);
}

/** Dictation into the box, with Windows' own speech recognizer. */
function MicButton({ onText }: { onText: (t: string) => void }) {
  const [state, setState] = useState<"off" | "starting" | "on">("off");
  const cb = useRef(onText);
  cb.current = onText;
  useEffect(() => {
    if (!bridge) return;
    const off = bridge.media.onDictate((e) => {
      if (e.kind === "ready") setState("on");
      else if (e.kind === "text") cb.current(e.text);
      else if (e.kind === "error") toast(e.error);
      else if (e.kind === "end") setState("off");
    });
    return () => {
      off();
      bridge!.media.dictateStop();
    };
  }, []);
  if (!bridge) return null;
  return (
    <button
      className={`chip icon-only mic ${state}`}
      onClick={() => (state === "off" ? (setState("starting"), bridge!.media.dictateStart()) : bridge!.media.dictateStop())}
      title={state === "on" ? "Listening… click to stop" : state === "starting" ? "Starting the microphone…" : "Speak your message (Windows speech recognition)"}
    >
      <Icon name="mic" size={15} />
      {state === "on" && <span className="mic-pulse" />}
    </button>
  );
}

// the workspace's files, for @-mentions (listed once per folder, refreshed now and then)
const fileCache = new Map<string, { at: number; files: string[] }>();
async function workspaceFiles(folder: string): Promise<string[]> {
  const c = fileCache.get(folder);
  if (c && Date.now() - c.at < 15000) return c.files;
  const files = bridge ? await bridge.fs.find(folder, "*").catch(() => [] as string[]) : [];
  fileCache.set(folder, { at: Date.now(), files });
  return files;
}

/** Files matching what follows the @: the name first, then anywhere in the path. */
function matchFiles(files: string[], q: string): string[] {
  const l = q.toLowerCase();
  const scored: [number, string][] = [];
  for (const f of files) {
    const p = f.toLowerCase();
    const base = p.slice(p.lastIndexOf("/") + 1);
    const s = !l ? 1 : base.startsWith(l) ? 300 - base.length : base.includes(l) ? 200 - base.length : p.includes(l) ? 100 - p.length / 10 : 0;
    if (s > 0) scored.push([s, f]);
  }
  return scored.sort((a, b) => b[0] - a[0]).slice(0, 8).map((x) => x[1]);
}

export function Composer({ big = false }: { big?: boolean }) {
  const running = useApp((s) => s.running);
  // a reply being written in another chat: this one can be read and typed in, not sent
  const elsewhere = useApp((s) => (s.running && s.liveId && s.liveId !== s.chat?.id ? s.liveId : null));
  const elsewhereTitle = useApp((s) => (elsewhere ? s.chats.find((c) => c.id === elsewhere)?.title || "another chat" : ""));
  const [ocr, setOcr] = useState(false);
  const folder = useApp((s) => s.chat?.folder ?? null);
  const enter = useApp((s) => s.settings.sendOnEnter);
  const [text, setText] = useState("");
  const [files, setFiles] = useState<Attached[]>([]);
  const [sel, setSel] = useState(0);
  const ta = useRef<HTMLTextAreaElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  const [caret, setCaret] = useState(0);
  const [wsFiles, setWsFiles] = useState<string[]>([]);
  // "@name" just before the caret: the mention being typed
  const mention = useMemo(() => {
    if (!folder || caret < 0) return null;
    const m = text.slice(0, caret).match(/(^|\s)@([^\s@]*)$/);
    return m ? { q: m[2], start: caret - m[2].length - 1 } : null;
  }, [text, caret, folder]);
  useEffect(() => {
    if (mention && folder) workspaceFiles(folder).then(setWsFiles);
  }, [mention !== null, folder]);
  const mentions = useMemo(() => (mention ? matchFiles(wsFiles, mention.q) : []), [mention, wsFiles]);
  useEffect(() => setSel(0), [mention?.q]);

  async function pickMention(rel: string) {
    if (!mention || !folder || !bridge) return;
    setText(text.slice(0, mention.start) + text.slice(caret));
    setCaret(mention.start);
    try {
      const body = await bridge.fs.raw(folder, rel);
      if (/\u0000/.test(body.slice(0, 8000))) return toast(`${rel} isn't text.`);
      setFiles((f) => (f.some((x) => x.name === rel) ? f : [...f, { name: rel, text: body.length > 120000 ? body.slice(0, 120000) + "\n…[cut]" : body }]));
    } catch (e: any) {
      toast(String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, ""));
    }
    ta.current?.focus();
  }

  const attach = async (list: FileList | File[]) => {
    const got = await readFiles(list);
    if (got.length) setFiles((f) => [...f, ...got]);
    ta.current?.focus();
  };

  useEffect(() => {
    // text (or text and files) to put in the box: ideas, edit-and-resend, the browser
    const compose = (e: Event) => {
      const d = (e as CustomEvent).detail;
      if (typeof d === "string") setText(d);
      else {
        setText(d.text || "");
        setFiles(d.files || []);
      }
      ta.current?.focus();
    };
    const drop = (e: Event) => attach((e as CustomEvent).detail);
    const focus = () => ta.current?.focus();
    window.addEventListener("pelica:compose", compose);
    window.addEventListener("pelica:drop", drop);
    window.addEventListener("pelica:focus", focus);
    return () => {
      window.removeEventListener("pelica:compose", compose);
      window.removeEventListener("pelica:drop", drop);
      window.removeEventListener("pelica:focus", focus);
    };
  }, []);
  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(320, el.scrollHeight) + "px";
  }, [text]);

  const slash = useMemo(() => {
    const m = text.match(/^\/(\w*)$/);
    return m ? COMMANDS.filter((c) => c.name.startsWith(m[1].toLowerCase())) : [];
  }, [text]);
  // a new filter starts from its first match, not from a row picked in the old list
  useEffect(() => setSel(0), [slash]);

  async function submit() {
    const t = text.trim();
    if ((!t && !files.length) || ocr) return;
    if (running) return send("");
    // (no model yet: keep what's in the box; send says why)
    if (app.get().status?.state !== "ready" && (files.length || !t.startsWith("/"))) return send("");
    if (files.length) {
      const photos = files.filter((f) => f.image);
      let list = files;
      if (photos.some((f) => !f.text)) {
        setOcr(true);
        list = await Promise.all(files.map(async (f) => (f.image !== undefined ? { ...f, text: await photoText(f) } : f)));
        setOcr(false);
      }
      const what = photos.length === files.length ? (files.length > 1 ? "Here are some photos." : "Here is a photo.") : files.length > 1 ? "Here are some files." : "Here is a file.";
      const msg = joinAttachments(t || what, list);
      setText("");
      setFiles([]);
      return send(
        msg,
        photos.map((f) => ({ name: f.name, data: f.image! })),
      );
    }
    setText("");
    if (await command(t)) return;
    send(t);
  }

  function key(e: React.KeyboardEvent) {
    if (e.key === "Escape" && running && !elsewhere) return e.preventDefault(), stop();
    if (e.key === "Tab" && e.shiftKey && !mentions.length && !slash.length) return e.preventDefault(), cyclePermMode();
    if (mentions.length) {
      if (e.key === "ArrowDown") return e.preventDefault(), setSel((sel + 1) % mentions.length);
      if (e.key === "ArrowUp") return e.preventDefault(), setSel((sel - 1 + mentions.length) % mentions.length);
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) return e.preventDefault(), void pickMention(mentions[sel] || mentions[0]);
      if (e.key === "Escape") return e.preventDefault(), setCaret(-1);
    }
    if (slash.length) {
      if (e.key === "ArrowDown") return e.preventDefault(), setSel((sel + 1) % slash.length);
      if (e.key === "ArrowUp") return e.preventDefault(), setSel((sel - 1 + slash.length) % slash.length);
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
        e.preventDefault();
        const c = slash[sel] || slash[0];
        setText(`/${c.name}${c.args ? " " : ""}`);
        if (e.key === "Enter" && !c.args) {
          setText("");
          command(`/${c.name}`);
        }
        return;
      }
    }
    if (e.key === "Enter" && (enter ? !e.shiftKey : e.ctrlKey)) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className={`composer-wrap ${big ? "big" : ""}`}>
    <div className={`composer slim ${big ? "big" : ""}`}>
      <AgentHint folder={folder} />
      {elsewhere && (
        <div className="agent-hint live-hint">
          <span className="live-dot" />
          <span>
            Pelica is replying in <b>{elsewhereTitle}</b>. You can read and type here; send when it&apos;s done.
          </span>
          <button className="ghost-btn" onClick={() => openChat(elsewhere)}>
            Go there
          </button>
        </div>
      )}
      {slash.length > 0 && (
        <div className="slash-menu">
          {slash.map((c, i) => (
            <button key={c.name} className={i === sel ? "sel" : ""} onMouseEnter={() => setSel(i)} onClick={() => (setText(`/${c.name}${c.args ? " " : ""}`), ta.current?.focus())}>
              <span className="mono">/{c.name}</span>
              {c.args && <span className="muted mono"> {c.args}</span>}
              <span className="muted slash-help">{c.help}</span>
            </button>
          ))}
        </div>
      )}
      {mentions.length > 0 && (
        <div className="slash-menu mention-menu">
          <div className="mention-head">Attach a file from {folder?.split(/[\\/]/).filter(Boolean).pop()}</div>
          {mentions.map((f, i) => {
            const cut = f.lastIndexOf("/") + 1;
            return (
              <button key={f} className={i === sel ? "sel" : ""} onMouseEnter={() => setSel(i)} onMouseDown={(e) => (e.preventDefault(), pickMention(f))}>
                <Icon name="file" size={13} />
                <span className="mono">{f.slice(cut)}</span>
                <span className="muted mono slash-help"><bdi>{f.slice(0, Math.max(0, cut - 1))}</bdi></span>
              </button>
            );
          })}
        </div>
      )}
      <AttachChips files={files} onRemove={(i) => setFiles((f) => f.filter((_, j) => j !== i))} />
      <div className="composer-row">
      <PlusMenu folder={folder} onAttach={() => picker.current?.click()} />
      <textarea
        ref={ta}
        value={text}
        onChange={(e) => (setText(e.target.value), setCaret(e.target.selectionStart), setSel(0))}
        onSelect={(e) => setCaret((e.target as HTMLTextAreaElement).selectionStart)}
        onKeyDown={key}
        onPaste={(e) => {
          if (e.clipboardData.files.length) {
            e.preventDefault();
            attach(e.clipboardData.files);
          }
        }}
        placeholder={big ? "How can I help you today?" : folder ? "Reply to Pelica…  (/ for commands, @ for files)" : "Reply to Pelica…  (/ for commands)"}
        rows={1}
        autoFocus
      />
      {running && !elsewhere ? (
        <button className="send-btn stop" onClick={stop} title="Stop (Esc)">
          <Icon name="stop" size={13} stroke={2.4} />
        </button>
      ) : (
        <button className={`send-btn ${ocr ? "busy" : ""} ${text.trim() || files.length ? "ready" : ""}`} onClick={submit} disabled={(!text.trim() && !files.length) || !!elsewhere || ocr} title={ocr ? "Reading the text in your photos…" : elsewhere ? "Pelica is replying in another chat" : enter ? "Send (Enter)" : "Send (Ctrl+Enter)"}>
          <Icon name={ocr ? "scan" : "enter"} size={17} />
        </button>
      )}
      <MicButton onText={(t) => setText((x) => (x && !/\s$/.test(x) ? x + " " : x) + t)} />
      </div>
      <input ref={picker} type="file" multiple hidden onChange={(e) => (e.target.files && attach(e.target.files), (e.target.value = ""))} />
    </div>
    <div className="composer-foot">
      <PermChip />
      <span className="grow" />
      <ModelChip />
      <EffortPicker foot />
      <ContextMeter ringOnly />
    </div>
    </div>
  );
}
