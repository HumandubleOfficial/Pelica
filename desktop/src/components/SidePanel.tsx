// The panel on the right of a chat (the session rail), as in the Claude app:
// the plan, where the chat works (this computer, its folder, permissions,
// web), how full the context is and what fills it, the connectors, and the
// files the chat wrote. Right-click it to choose which sections show.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { createPortal, flushSync } from "react-dom";
import { compactNow } from "../agent/agent";
import { bridge } from "../lib/bridge";
import { app, PERM_MODES, setSettings, useApp, type Msg } from "../state";
import { openArtifact } from "./ArtifactPanel";
import { FolderList } from "./Composer";
import { presetOf } from "./Connectors";
import { toolPerm } from "../agent/tools";
import { Grip, Icon, TodoMark } from "./Icons";

const fmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 100000 ? 0 : 1)}k` : String(Math.round(n)));
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

/** How a drag of a section starts (the panel moves the cards itself while it lasts). */
const DragCtx = createContext<{ start: (id: string) => (e: React.PointerEvent) => void }>({ start: () => () => {} });

/** A section with a title that folds it (remembered in the settings); a
 *  folded one can still show a one-line summary. The dots at its top right
 *  move it. */
function Section({ id, title, children, action, summary }: { id: string; title: string; children: React.ReactNode; action?: React.ReactNode; summary?: React.ReactNode }) {
  const folded = useApp((s) => s.settings.panelFolded.includes(id));
  const { start } = useContext(DragCtx);
  const toggle = () => {
    const f = app.get().settings.panelFolded;
    setSettings({ panelFolded: folded ? f.filter((x) => x !== id) : [...f, id] });
  };
  return (
    <section data-sec={id} className={`sp-card ${folded ? "folded" : ""}`}>
      <div className="sp-head">
        <button className="sp-title" onClick={toggle} aria-expanded={!folded}>
          {title}
          <span className="sp-caret" />
        </button>
        <span className="grow" />
        {folded ? summary : action}
        <button className="sp-grip" title="Drag to move" aria-label={`Move ${title}`} onPointerDown={start(id)}>
          <Grip />
        </button>
      </div>
      {!folded && <div className="sp-body">{children}</div>}
    </section>
  );
}

function Environment() {
  const st = useApp((s) => s.status);
  const folder = useApp((s) => s.chat?.folder ?? null);
  const mode = useApp((s) => s.settings.permMode);
  const web = useApp((s) => s.settings.web);
  const [pick, setPick] = useState<"folder" | "perm" | null>(null);
  const parts = folder ? folder.replace(/\\/g, "/").split("/").filter(Boolean) : [];
  const name = parts.pop();
  const parent = parts.length ? (parts.length > 2 ? "…/" : "") + parts.slice(-2).join("/") : "";
  const cur = PERM_MODES.find((m) => m.id === mode) || PERM_MODES[1];
  return (
    <Section id="env" title="Environment">
      <div className="sp-row" title={st?.gpu ? `Runs on ${st.gpu}` : "Runs on this PC"}>
        <Icon name="monitor" size={15} />
        <span className="sp-label">This computer</span>
        <span className="sp-meta ellipsis">{st?.gpu?.replace(/^(NVIDIA|AMD|Intel\(R\))\s+(GeForce\s+)?/i, "") || ""}</span>
      </div>
      <button className="sp-row click" onClick={() => setPick(pick === "folder" ? null : "folder")} title={folder || "Choose a folder for this chat"}>
        <Icon name="folder" size={15} />
        <span className="sp-label ellipsis">
          {folder ? (
            <>
              <b>{name}</b>
              {parent && <span className="muted"> · {parent}</span>}
            </>
          ) : (
            <span className="muted">No folder</span>
          )}
        </span>
        <Icon name={pick === "folder" ? "down" : "chevron"} size={12} className="muted" />
      </button>
      {pick === "folder" && (
        <div className="sp-sub">
          <FolderList folder={folder} done={() => setPick(null)} />
        </div>
      )}
      <button className="sp-row click" onClick={() => setPick(pick === "perm" ? null : "perm")} title={cur.desc}>
        <Icon name={mode === "plan" ? "list" : mode === "auto" ? "bolt" : mode === "ask" ? "lock" : "shield"} size={15} />
        <span className="sp-label">Permissions</span>
        <span className={`sp-meta perm-${mode}`}>{cur.label}</span>
      </button>
      {pick === "perm" && (
        <div className="sp-sub">
          {PERM_MODES.map((m) => (
            <button key={m.id} className={`menu-row ${m.id === mode ? "sel" : ""}`} onClick={() => (setSettings({ permMode: m.id }), setPick(null))}>
              <span className="menu-text">
                <b>{m.label}</b>
                <span className="muted small">{m.desc}</span>
              </span>
              {m.id === mode && <Icon name="check" size={13} />}
            </button>
          ))}
        </div>
      )}
      <div className="sp-row">
        <Icon name="globe" size={15} />
        <span className="sp-label">Web search</span>
        <button className={`toggle ${web ? "on" : ""}`} role="switch" aria-checked={web} onClick={() => setSettings({ web: !web })} title={web ? "Turn web search off" : "Turn web search on"} />
      </div>
    </Section>
  );
}

const PARTS = [
  { key: "messages", label: "Messages", color: "var(--c-msgs)" },
  { key: "tool_results", label: "Tool results", color: "var(--c-results)" },
  { key: "tools", label: "System tools", color: "var(--c-tools)" },
  { key: "mcp", label: "MCP tools", color: "var(--c-mcp)" },
  { key: "system", label: "System prompt", color: "var(--c-system)" },
] as const;

function Context() {
  const c = useApp((s) => s.context);
  const ready = useApp((s) => s.status?.state === "ready");
  const running = useApp((s) => s.running);
  const auto = useApp((s) => s.settings.autocompact);
  const at = useApp((s) => s.settings.compactAt);
  if (!c || !ready) {
    return (
      <Section id="ctx" title="Context">
        <div className="sp-empty">Load a model to see what fills its context.</div>
      </Section>
    );
  }
  const buffer = auto ? Math.round(c.ctx * (1 - at)) : 0;
  const free = Math.max(0, c.ctx - c.total - buffer);
  const until = auto ? Math.max(0, Math.round(c.ctx * at) - c.total) : 0;
  const used = Math.round((c.total / c.ctx) * 100);
  const summary = (
    <span className="sp-sum" title={`${fmt(c.total)} / ${fmt(c.ctx)} tokens`}>
      {used}%
      <span className="sp-mini">
        <i style={{ width: `${Math.min(100, used)}%` }} />
      </span>
    </span>
  );
  return (
    <Section id="ctx" title="Context" summary={summary}>
      <div className="sp-ctx-top">
        <b>{Math.round((c.total / c.ctx) * 100)}% used</b>
        <span className="muted">
          {fmt(c.total)} / {fmt(c.ctx)}
        </span>
      </div>
      <div className="sp-ctx-bar">
        {PARTS.map((p) => (
          <span key={p.key} style={{ width: `${(c[p.key] / c.ctx) * 100}%`, background: p.color }} />
        ))}
        {buffer > 0 && <span className="buffer" style={{ width: `${(buffer / c.ctx) * 100}%` }} />}
        <span className="track" />
      </div>
      <div className="sp-ctx-rows">
        {PARTS.map((p) => (
          <div key={p.key} className="sp-ctx-row">
            <i style={{ background: p.color }} />
            <span>{p.label}</span>
            <span className="muted">{fmt(c[p.key])}</span>
            <b>{pct(c[p.key] / c.ctx)}</b>
          </div>
        ))}
        {buffer > 0 && (
          <div className="sp-ctx-row" title={`Compacting starts at ${Math.round(at * 100)}%, so this much is kept free for it`}>
            <i className="buffer" />
            <span>Autocompact buffer</span>
            <span className="muted">{fmt(buffer)}</span>
            <b>{pct(buffer / c.ctx)}</b>
          </div>
        )}
        <div className="sp-ctx-row">
          <i className="free" />
          <span>Free space</span>
          <span className="muted">{fmt(free)}</span>
          <b>{pct(free / c.ctx)}</b>
        </div>
      </div>
      <div className="sp-ctx-foot">
        <span className="muted small ellipsis">{auto ? `${fmt(until)} to compact` : "Auto-compact off"}</span>
        <button className="btn small" disabled={running} onClick={() => compactNow()}>
          Compact session
        </button>
      </div>
    </Section>
  );
}

function Connectors() {
  const cfgs = useApp((s) => s.settings.mcpServers);
  const live = useApp((s) => s.mcp);
  const [all, setAll] = useState(false);
  const manage = (
    <button className="sp-link" onClick={() => app.set({ settingsOpen: true, settingsTab: "Connectors" })} title="Manage connectors">
      <Icon name="settings" size={13} />
    </button>
  );
  const shown = all ? cfgs : cfgs.slice(0, 4);
  return (
    <Section id="mcp" title="Connectors" action={manage}>
      {!cfgs.length && (
        <button className="sp-row click" onClick={() => app.set({ settingsOpen: true, settingsTab: "Connectors" })}>
          <Icon name="plus" size={15} />
          <span className="sp-label muted">Add a connector</span>
        </button>
      )}
      {shown.map((c) => {
        const m = live.find((x) => x.name === c.name);
        const p = presetOf(c);
        const state = c.disabled ? "off" : m?.running ? "on" : m ? "error" : "off";
        const n = m?.tools.filter((t) => toolPerm(c, t) !== "off").length || 0;
        const signIn = state === "error" && m?.auth === "needed";
        return (
          <button key={c.name} className="sp-row click" onClick={() => app.set({ settingsOpen: true, settingsTab: "Connectors" })} title={state === "error" ? m?.error : c.url || `${c.command} ${(c.args || []).join(" ")}`}>
            <span className="sp-conn-ico">
              <Icon name={p?.icon || (c.url ? "globe" : "plug")} size={13} />
            </span>
            <span className="sp-label ellipsis">{p?.name || c.name}</span>
            <span className="sp-meta">{state === "on" ? `${n} tool${n === 1 ? "" : "s"}` : signIn ? "Sign in" : state === "off" ? "off" : ""}</span>
            <span className={`status-dot ${state === "on" ? "ready" : state === "error" && !signIn ? "error" : "empty"}`} />
          </button>
        );
      })}
      {cfgs.length > 4 && (
        <button className="sp-more" onClick={() => setAll(!all)}>
          {all ? "Show less" : "Show more"}
        </button>
      )}
    </Section>
  );
}

function Outputs() {
  const msgs = useApp((s) => s.chat?.messages);
  const archive = useApp((s) => s.chat?.archive);
  const folder = useApp((s) => s.chat?.folder ?? null);
  const files = useMemo(() => {
    const m = new Map<string, { path: string; after: string; created: boolean }>();
    for (const x of [...(archive || []), ...(msgs || [])] as Msg[]) {
      if (x.role !== "tool" || (x.ui.kind !== "write" && x.ui.kind !== "edit") || !x.ui.path || x.ui.after === undefined) continue;
      const had = m.get(x.ui.path);
      m.delete(x.ui.path);
      m.set(x.ui.path, { path: x.ui.path, after: x.ui.after, created: had ? had.created : x.ui.kind === "write" && x.ui.before === null });
    }
    return [...m.values()].reverse();
  }, [msgs, archive]);
  return (
    <Section id="out" title="Outputs">
      {!files.length && <div className="sp-empty">Files Pelica creates or edits in this chat show up here.</div>}
      {files.map((f) => {
        const name = f.path.split("/").pop() || f.path;
        return (
          <div key={f.path} className="sp-row click sp-file" title={f.path}>
            <button className="sp-file-open" onClick={() => openArtifact(name, f.after, (name.split(".").pop() || "").toLowerCase())}>
              <Icon name="code" size={14} />
              <span className="sp-label ellipsis">{name}</span>
              {f.created && <span className="sp-new">new</span>}
            </button>
            {folder && bridge && (
              <button className="sp-link" title="Show in Explorer" onClick={() => bridge!.fs.reveal(folder, f.path)}>
                <Icon name="external" size={12} />
              </button>
            )}
          </div>
        );
      })}
    </Section>
  );
}

/** The agent's task list, while it has one. */
function Plan() {
  const todos = useApp((s) => s.chat?.todos);
  if (!todos?.length) return null;
  const done = todos.filter((t) => t.status === "completed").length;
  return (
    <Section id="plan" title="Plan" summary={<span className="sp-sum">{done}/{todos.length}</span>}>
      {todos.map((t, i) => (
        <div key={i} className={`sp-row sp-todo ${t.status}`}>
          <TodoMark status={t.status} />
          <span className="sp-label">{t.content}</span>
        </div>
      ))}
    </Section>
  );
}

const SECTIONS = [
  { id: "plan", label: "Plan" },
  { id: "env", label: "Environment" },
  { id: "ctx", label: "Context" },
  { id: "mcp", label: "Connectors" },
  { id: "out", label: "Outputs" },
];

/** Right-click on the panel: which sections it shows. */
function ShowMenu({ at, close }: { at: { x: number; y: number }; close: () => void }) {
  const hidden = useApp((s) => s.settings.panelHidden);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const out = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && close();
    const key = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("mousedown", out);
    window.addEventListener("keydown", key);
    window.addEventListener("blur", close);
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("mousedown", out);
      window.removeEventListener("keydown", key);
      window.removeEventListener("blur", close);
      window.removeEventListener("resize", close);
    };
  }, [close]);
  const flip = (id: string) => setSettings({ panelHidden: hidden.includes(id) ? hidden.filter((h) => h !== id) : [...hidden, id] });
  return createPortal(
    <div ref={ref} className="ctx-menu" style={{ left: Math.min(at.x, window.innerWidth - 210), top: Math.min(at.y, window.innerHeight - 250) }} onContextMenu={(e) => e.preventDefault()}>
      <div className="ctx-title">Show</div>
      {SECTIONS.map((s) => {
        const on = !hidden.includes(s.id);
        return (
          <button key={s.id} className="ctx-row" role="menuitemcheckbox" aria-checked={on} onClick={() => flip(s.id)}>
            <span className={`ctx-check ${on ? "on" : ""}`}>{on && <Icon name="check" size={11} stroke={2.75} />}</span>
            {s.label}
          </button>
        );
      })}
      <div className="menu-sep" />
      <button className="ctx-row" onClick={() => (setSettings({ panelHidden: [], panelFolded: [], panelOrder: [] }), close())}>
        Reset to defaults
      </button>
    </div>,
    document.body,
  );
}

const PARTS_OF: Record<string, () => React.ReactNode> = {
  plan: () => <Plan />,
  env: () => <Environment />,
  ctx: () => <Context />,
  mcp: () => <Connectors />,
  out: () => <Outputs />,
};

/** The saved order, with any section it doesn't name yet in its default place at the end. */
const orderOf = (saved: string[]) => [...saved.filter((id) => PARTS_OF[id]), ...SECTIONS.map((s) => s.id).filter((id) => !saved.includes(id))];

export function SidePanel() {
  const hidden = useApp((s) => s.settings.panelHidden);
  const saved = useApp((s) => s.settings.panelOrder);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const close = useCallback(() => setMenu(null), []);
  const aside = useRef<HTMLElement>(null);
  // (true while a dropped card glides into place)
  const settling = useRef(false);
  const on = (id: string) => !hidden.includes(id);
  const order = orderOf(saved);
  // press on a section's dots and drag: the card follows the pointer on a
  // spring (it trails a little and settles, never moves in a straight line),
  // the others glide out of its way, and letting go eases it into its place
  const start = (id: string) => (e: React.PointerEvent) => {
    if (e.button !== 0 || !aside.current || settling.current) return;
    e.preventDefault();
    const cards = [...aside.current.querySelectorAll<HTMLElement>("[data-sec]")].map((el) => {
      const r = el.getBoundingClientRect();
      return { id: el.dataset.sec!, el, top: r.top, h: r.height };
    });
    const from = cards.findIndex((c) => c.id === id);
    if (from < 0) return;
    const me = cards[from];
    const rest = cards.filter((c) => c !== me);
    const gap = cards.length > 1 ? cards[1].top - cards[0].top - cards[0].h : 10;
    const lo = cards[0].top - me.top;
    const hi = cards[cards.length - 1].top + cards[cards.length - 1].h - (me.top + me.h);
    const y0 = e.clientY;
    let to = from;
    let goal = 0; // where the pointer says the card should be
    let y = 0; // where it is
    let v = 0; // how fast it's going
    let raf = 0;
    let last = performance.now();
    const place = () => {
      me.el.style.transform = `translateY(${y}px) scale(1.02)`;
      // the others make room around the card's middle
      const mid = me.top + y + me.h / 2;
      const next = rest.filter((c) => c.top + c.h / 2 < mid).length;
      if (next === to) return;
      to = next;
      rest.forEach((c, i) => {
        // i is its place among the others: it moves up if the dragged card passed it going down, and down if going up
        const shift = from <= i && i < to ? -(me.h + gap) : to <= i && i < from ? me.h + gap : 0;
        c.el.style.transform = shift ? `translateY(${shift}px)` : "";
      });
    };
    // a stiff, well-damped spring: quick to follow, a soft finish, no wobble
    const tick = (now: number) => {
      // (in small fixed steps, so a slow frame can't make it overshoot)
      let left = Math.min(0.05, (now - last) / 1000);
      last = now;
      while (left > 0) {
        const dt = Math.min(left, 1 / 240);
        v += (2000 * (goal - y) - 86 * v) * dt;
        y += v * dt;
        left -= dt;
      }
      if (Math.abs(goal - y) < 0.1 && Math.abs(v) < 2) (y = goal), (v = 0);
      place();
      raf = v || y !== goal ? requestAnimationFrame(tick) : 0;
    };
    const kick = () => {
      if (!raf) (last = performance.now()), (raf = requestAnimationFrame(tick));
    };
    me.el.classList.add("lifted");
    rest.forEach((c) => (c.el.style.transition = "transform 0.36s cubic-bezier(0.22, 1, 0.36, 1)"));
    document.body.classList.add("dragging-sec");
    place();
    const move = (m: PointerEvent) => {
      goal = Math.max(lo, Math.min(hi, m.clientY - y0));
      kick();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      cancelAnimationFrame(raf);
      raf = 0;
      // glide into the slot it was dropped on, then take that place for real
      const slot = cards[0].top + rest.slice(0, to).reduce((s, c) => s + c.h + gap, 0) - me.top;
      settling.current = true;
      me.el.style.transition = "transform 0.32s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.32s ease, background-color 0.32s ease";
      me.el.classList.remove("lifted");
      me.el.style.transform = `translateY(${slot}px)`;
      window.setTimeout(() => {
        settling.current = false;
        document.body.classList.remove("dragging-sec");
        const shown = cards.map((c) => c.id).filter((x) => x !== id);
        shown.splice(to, 0, id);
        // the sections that aren't on screen (hidden, or a plan with no steps) keep their places
        const full = orderOf(app.get().settings.panelOrder);
        let k = 0;
        const next = full.map((x) => (cards.some((c) => c.id === x) ? shown[k++] : x));
        flushSync(() => setSettings({ panelOrder: next }));
        cards.forEach((c) => {
          c.el.style.transition = "none";
          c.el.style.transform = "";
        });
        requestAnimationFrame(() => cards.forEach((c) => (c.el.style.transition = "")));
      }, 320);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  };
  return (
    <DragCtx.Provider value={{ start }}>
      <aside ref={aside} className="side-panel" onContextMenu={(e) => (e.preventDefault(), setMenu({ x: e.clientX, y: e.clientY }))}>
        {order.filter(on).map((id) => (
          <PartSlot key={id} id={id} />
        ))}
        {SECTIONS.every((s) => !on(s.id)) && <div className="sp-hint">Right-click here to choose what shows.</div>}
        {menu && <ShowMenu at={menu} close={close} />}
      </aside>
    </DragCtx.Provider>
  );
}

function PartSlot({ id }: { id: string }) {
  return <>{PARTS_OF[id]()}</>;
}
