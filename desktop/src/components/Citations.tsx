// A reply's citations: [1] in the text is a small button. Hovering it shows
// the passage it came from; clicking it opens that passage under the reply,
// in Sources, where the file can be opened too.
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { bridge } from "../lib/bridge";
import { cited, docName, excerptsOf, projectDocs, rootOf, type Excerpt } from "../lib/docs";
import { toast, useApp } from "../state";
import { Icon } from "./Icons";

const where = (e: Excerpt) => `${docName(e.file)}${e.page ? `, page ${e.page}` : ""}`;

function useOpenFile() {
  const projects = useApp((s) => s.settings.projects);
  const chat = useApp((s) => s.chat);
  const folders = projectDocs(chat, projects);
  return (e: Excerpt) => {
    const root = rootOf(e.file, folders);
    if (!root || !bridge) return toast("That folder isn't one of the project's documents any more.");
    bridge.sys.open(root, e.file.slice(root.replace(/[\\/]+$/, "").length + 1)).catch((err) => toast(String(err?.message || err)));
  };
}

interface Pop {
  e: Excerpt;
  /** the button's box */
  r: DOMRect;
}

/** The hover preview, kept inside the window, above the button if there's room. */
function Preview({ pop, keep, drop }: { pop: Pop; keep: () => void; drop: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [at, setAt] = useState<{ left: number; top: number; below: boolean } | null>(null);
  useLayoutEffect(() => {
    const el = ref.current!;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    const below = pop.r.top - h - 10 < 44;
    setAt({
      left: Math.max(10, Math.min(innerWidth - w - 10, pop.r.left + pop.r.width / 2 - w / 2)),
      top: below ? pop.r.bottom + 8 : pop.r.top - h - 8,
      below,
    });
  }, [pop]);
  return createPortal(
    <div ref={ref} className={`cite-pop ${at?.below ? "below" : ""}`} style={at ? { left: at.left, top: at.top } : { visibility: "hidden", left: 0, top: 0 }} onMouseEnter={keep} onMouseLeave={drop}>
      <div className="cite-pop-head">
        <Icon name="file" size={13} />
        <span>{where(pop.e)}</span>
      </div>
      <div className="cite-pop-text">{pop.e.text.slice(0, 420).trim()}{pop.e.text.length > 420 ? "…" : ""}</div>
    </div>,
    document.body,
  );
}

/** Hover and click for everything with data-cite inside `.turn-body`. */
export function useCites(src: string) {
  const ex = useMemo(() => excerptsOf(src), [src]);
  const [open, setOpen] = useState<number | null>(null);
  const [pop, setPop] = useState<Pop | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const later = (f: () => void, ms: number) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(f, ms);
  };
  const keep = () => {
    if (timer.current) clearTimeout(timer.current);
  };
  const drop = () => later(() => setPop(null), 140);
  useEffect(() => keep, []);
  const find = (t: EventTarget) => {
    const b = (t as HTMLElement).closest?.("[data-cite]") as HTMLElement | null;
    const e = b && ex.find((x) => x.n === +b.dataset.cite!);
    return b && e ? { b, e } : null;
  };
  const handlers = ex.length
    ? {
        onMouseOver(ev: React.MouseEvent) {
          const f = find(ev.target);
          if (f) later(() => setPop({ e: f.e, r: f.b.getBoundingClientRect() }), pop ? 0 : 160);
        },
        onMouseOut(ev: React.MouseEvent) {
          if (find(ev.target)) drop();
        },
        onClick(ev: React.MouseEvent) {
          const f = find(ev.target);
          if (!f) return;
          keep();
          setPop(null);
          setOpen((n) => (n === f.e.n && f.b.classList.contains("source-chip") ? null : f.e.n));
        },
      }
    : {};
  const preview = pop ? <Preview pop={pop} keep={keep} drop={drop} /> : null;
  return { ex, open, setOpen, handlers, preview, numbers: ex.map((e) => e.n).join(",") };
}

/** Where a reply's facts came from, under it. */
export function Sources({ ex, reply, open, close }: { ex: Excerpt[]; reply: string; open: number | null; close: () => void }) {
  const openFile = useOpenFile();
  const box = useRef<HTMLDivElement>(null);
  const shown = open === null ? null : ex.find((e) => e.n === open);
  // (a citation far above: bring the passage into view)
  useEffect(() => {
    if (shown) box.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [shown?.n]);
  if (!ex.length) return null;
  const nums = cited(reply).filter((n) => ex.some((e) => e.n === n));
  const list = nums.length ? nums.map((n) => ex.find((e) => e.n === n)!) : ex;
  return (
    <div className="sources-wrap" ref={box}>
      <div className="sources">
        <span className="sources-head">{nums.length ? "Sources" : "Passages it read"}</span>
        {list.map((e) => (
          <button key={e.n} className={`source-chip ${e.n === open ? "on" : ""}`} data-cite={e.n} aria-expanded={e.n === open}>
            <b>{e.n}</b> <span className="source-name">{docName(e.file)}</span>
            {e.page ? <span className="muted"> · p. {e.page}</span> : null}
          </button>
        ))}
      </div>
      <div className={`source-open ${shown ? "on" : ""}`}>
        <div className="source-open-in">
          {shown && (
            <div className="source-passage" key={shown.n}>
              <div className="source-passage-head">
                <b>{where(shown)}</b>
                <button className="btn small" onClick={() => openFile(shown)}>
                  Open file
                </button>
                <button className="ghost-btn" title="Close" onClick={close}>
                  <Icon name="x" size={14} />
                </button>
              </div>
              <div className="source-passage-text">{shown.text}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
