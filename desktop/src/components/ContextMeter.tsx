// How full the context is: a ring by the message box, and on click what
// fills it (system prompt, tools, MCP tools, messages, tool results).
import { useEffect, useRef, useState } from "react";
import { compactNow, refreshContext } from "../agent/agent";
import { setSettings, useApp } from "../state";

const fmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n));

const PARTS: { key: "system" | "tools" | "mcp" | "messages" | "tool_results"; label: string; color: string }[] = [
  { key: "system", label: "System prompt", color: "var(--c-system)" },
  { key: "tools", label: "Tools", color: "var(--c-tools)" },
  { key: "mcp", label: "MCP tools", color: "var(--c-mcp)" },
  { key: "messages", label: "Messages", color: "var(--c-msgs)" },
  { key: "tool_results", label: "Tool results", color: "var(--c-results)" },
];

export function ContextMeter({ ringOnly = false }: { ringOnly?: boolean }) {
  const c = useApp((s) => s.context);
  const chatLen = useApp((s) => s.chat?.messages.length ?? 0);
  const ready = useApp((s) => s.status?.state === "ready");
  const running = useApp((s) => s.running);
  const auto = useApp((s) => s.settings.autocompact);
  const at = useApp((s) => s.settings.compactAt);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ready && !running) refreshContext();
  }, [chatLen, ready, running]);
  useEffect(() => {
    const f = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const o = () => setOpen(true);
    document.addEventListener("mousedown", f);
    window.addEventListener("pelica:context", o);
    return () => {
      document.removeEventListener("mousedown", f);
      window.removeEventListener("pelica:context", o);
    };
  }, []);
  if (!c || !ready) return null;
  const frac = Math.min(1, c.total / c.ctx);
  const r = 8;
  const C = 2 * Math.PI * r;
  let off = 0;
  return (
    <div className="pop-wrap" ref={ref}>
      <button className={`ctx-ring ${frac > 0.8 ? "hot" : frac > 0.6 ? "warm" : ""}`} onClick={() => setOpen(!open)} title={`Context: ${fmt(c.total)} of ${fmt(c.ctx)} tokens`}>
        <svg width="22" height="22" viewBox="0 0 22 22">
          <circle cx="11" cy="11" r={r} fill="none" stroke="var(--line)" strokeWidth="3" />
          {PARTS.map((p) => {
            const len = (c[p.key] / c.ctx) * C;
            const el = len > 0.01 && <circle key={p.key} cx="11" cy="11" r={r} fill="none" stroke={p.color} strokeWidth="3" strokeDasharray={`${len} ${C - len}`} strokeDashoffset={-off} transform="rotate(-90 11 11)" />;
            off += len;
            return el;
          })}
        </svg>
        {!ringOnly && <span>{Math.round(frac * 100)}%</span>}
      </button>
      {open && (
        <div className={`popover up ctx-pop ${ringOnly ? "right" : ""}`}>
          <div className="pop-title">
            Context <span className="muted">{fmt(c.total)} / {fmt(c.ctx)} tokens</span>
          </div>
          <div className="ctx-bar">
            {PARTS.map((p) => (
              <span key={p.key} style={{ width: `${(c[p.key] / c.ctx) * 100}%`, background: p.color }} />
            ))}
            <i style={{ left: `${at * 100}%` }} title="Auto-compact point" />
          </div>
          {PARTS.map((p) => (
            <div key={p.key} className="ctx-row">
              <span className="ctx-dot" style={{ background: p.color }} />
              <span>{p.label}</span>
              <span className="muted">{fmt(c[p.key])}</span>
            </div>
          ))}
          <div className="ctx-row">
            <span className="ctx-dot" style={{ background: "var(--line)" }} />
            <span>Free</span>
            <span className="muted">{fmt(Math.max(0, c.ctx - c.total))}</span>
          </div>
          <div className="ctx-actions">
            <label className="switch-row">
              <input type="checkbox" checked={auto} onChange={(e) => setSettings({ autocompact: e.target.checked })} />
              <span>Auto-compact at {Math.round(at * 100)}%</span>
            </label>
            <button className="btn small" disabled={running} onClick={() => (setOpen(false), compactNow())}>
              Compact now
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
