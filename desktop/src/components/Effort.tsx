// Effort levels, each with its own animation: a still dot (off), a slow
// breath (low), orbiting sparks (medium), a faster swarm (high), a spinning
// plasma ring with flares (max). The same orb shows while the model thinks.
import { useEffect, useRef, useState } from "react";
import { app, setSettings, useApp } from "../state";
import { Icon } from "./Icons";

const NAMES: Record<string, string> = { off: "Off", none: "Off", low: "Low", medium: "Medium", high: "High", max: "Max", on: "On" };
const DESC: Record<string, string> = { off: "Answers right away, no thinking", low: "A quick think first", medium: "Thinks it through", high: "Thinks hard, slower", max: "Thinks as long as it needs", on: "Thinks before answering" };

export function level(l: string | undefined): string {
  if (!l || l === "none") return "off";
  return l === "on" ? "medium" : l;
}

export function EffortOrb({ effort, size = 18, active = true }: { effort?: string; size?: number; active?: boolean }) {
  const lv = level(effort);
  const sparks = lv === "medium" ? 2 : lv === "high" ? 3 : lv === "max" ? 4 : 0;
  return (
    <span className={`orb lv-${lv} ${active ? "on" : ""}`} style={{ width: size, height: size }}>
      <span className="orb-core" />
      {lv === "max" && <span className="orb-ring" />}
      {lv === "low" && <span className="orb-breath" />}
      {Array.from({ length: sparks }, (_, i) => (
        <span key={i} className="orb-spark" style={{ animationDelay: `${(-i * (lv === "max" ? 0.9 : 1.6)) / sparks}s` }} />
      ))}
    </span>
  );
}

export function EffortPicker({ foot = false }: { foot?: boolean }) {
  const efforts = useApp((s) => s.status?.efforts);
  const def = useApp((s) => s.status?.effort);
  const want = useApp((s) => s.settings.effort);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const f = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", f);
    return () => document.removeEventListener("mousedown", f);
  }, []);
  if (!efforts?.length) return null;
  const cur = efforts.includes(want) ? want : def || efforts[0];
  return (
    <div className="pop-wrap" ref={ref}>
      <button className={`${foot ? "foot-btn muted" : "chip"} effort-chip lv-${level(cur)}`} onClick={() => setOpen(!open)} title="Thinking effort">
        {!foot && <EffortOrb effort={cur} size={15} />}
        {NAMES[cur] || cur}
      </button>
      {open && (
        <div className={`popover up effort-pop ${foot ? "right" : ""}`}>
          <div className="pop-title">Thinking effort</div>
          {efforts.map((e) => (
            <button
              key={e}
              className={`effort-row ${e === cur ? "sel" : ""}`}
              onClick={() => {
                setSettings({ effort: e });
                setOpen(false);
                app.set({});
              }}
            >
              <EffortOrb effort={e} size={22} />
              <span className="effort-text">
                <b>{NAMES[e] || e}</b>
                <span>{DESC[e] || ""}</span>
              </span>
              {e === cur && <Icon name="check" size={14} />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
