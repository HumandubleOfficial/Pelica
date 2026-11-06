// Find in chat (Ctrl+F): marks every match with the CSS Custom Highlight API,
// so the chat's own HTML is never touched; Enter / Shift+Enter step through
// them, Escape closes. Collapsed parts (thinking, folded tool output) aren't
// searched until they're open.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useApp } from "../state";
import { Icon } from "./Icons";

const hl = (globalThis as any).CSS?.highlights as Map<string, unknown> | undefined;
const Highlight = (globalThis as any).Highlight as { new (...r: Range[]): { add(r: Range): void } } | undefined;
/** Matches marked at most. (A letter in a long chat can match hundreds of thousands of times.) */
const MAX_FOUND = 5000;

function matches(root: HTMLElement, q: string): Range[] {
  const out: Range[] = [];
  if (!q) return out;
  const needle = q.toLowerCase();
  const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement?.closest(".find-bar, script, style") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  for (let n = walk.nextNode() as Text | null; n; n = walk.nextNode() as Text | null) {
    const t = n.data.toLowerCase();
    for (let i = t.indexOf(needle); i >= 0; i = t.indexOf(needle, i + needle.length)) {
      const r = document.createRange();
      r.setStart(n, i);
      r.setEnd(n, i + needle.length);
      out.push(r);
      if (out.length >= MAX_FOUND) return out;
    }
  }
  return out;
}

/** `onOpen` asks for every row of a long chat to be built; `ready` says they are. */
export function FindBar({ root, ready, onOpen }: { root: React.RefObject<HTMLElement | null>; ready: boolean; onOpen: (open: boolean) => void }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [cur, setCur] = useState(0);
  const [found, setFound] = useState<Range[]>([]);
  const input = useRef<HTMLInputElement>(null);
  const chat = useApp((s) => s.chat);
  const view = useApp((s) => s.view);

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "f" && view === "chat") {
        e.preventDefault();
        setOpen(true);
        requestAnimationFrame(() => input.current?.select());
      }
    };
    // (from a sidebar search: open on the words searched for)
    const ext = (e: Event) => {
      const v = String((e as CustomEvent).detail || "");
      setQ(v);
      setCur(0);
      setOpen(true);
    };
    window.addEventListener("keydown", key);
    window.addEventListener("pelica:find", ext);
    return () => {
      window.removeEventListener("keydown", key);
      window.removeEventListener("pelica:find", ext);
    };
  }, [view]);

  useEffect(() => onOpen(open), [open]);

  // (again when the chat changes: a reply added text, or another chat opened)
  useLayoutEffect(() => {
    if (!open || !root.current) return setFound([]);
    const r = matches(root.current, q.trim());
    setFound(r);
    setCur((c) => (r.length ? Math.min(c, r.length - 1) : 0));
  }, [q, open, chat, ready]);

  useEffect(() => {
    if (!hl || !Highlight) return;
    if (!found.length) {
      hl.delete("find");
      hl.delete("find-cur");
      return;
    }
    const all = new Highlight();
    found.forEach((r) => all.add(r));
    hl.set("find", all);
    hl.set("find-cur", new Highlight(found[cur]));
    return () => {
      hl.delete("find");
      hl.delete("find-cur");
    };
  }, [found, cur]);

  // the current match into view
  useEffect(() => {
    const r = found[cur];
    const box = root.current?.closest(".chat-scroll") as HTMLElement | null;
    if (!r || !box) return;
    const a = r.getBoundingClientRect();
    const b = box.getBoundingClientRect();
    if (a.top < b.top + 60 || a.bottom > b.bottom - 60) box.scrollTop += a.top - b.top - b.height / 2;
  }, [found, cur]);

  if (!open) return null;
  const step = (d: number) => found.length && setCur((c) => (c + d + found.length) % found.length);
  const close = () => (setOpen(false), setQ(""));
  return (
    <div className="find-wrap">
      <div className="find-bar" role="search">
        <Icon name="search" size={14} />
        <input
          ref={input}
          value={q}
          autoFocus
          placeholder="Find in chat"
          onChange={(e) => (setQ(e.target.value), setCur(0))}
          onKeyDown={(e) => {
            if (e.key === "Enter") step(e.shiftKey ? -1 : 1);
            else if (e.key === "Escape") close();
            else return;
            e.preventDefault();
            e.stopPropagation();
          }}
        />
        <span className="find-count">{q.trim() ? (found.length ? `${cur + 1} of ${found.length}${found.length >= MAX_FOUND ? "+" : ""}` : "No matches") : ""}</span>
        <button className="ghost-btn" title="Previous (Shift+Enter)" disabled={!found.length} onClick={() => step(-1)}>
          <Icon name="down" size={14} className="flip-y" />
        </button>
        <button className="ghost-btn" title="Next (Enter)" disabled={!found.length} onClick={() => step(1)}>
          <Icon name="down" size={14} />
        </button>
        <button className="ghost-btn" title="Close (Esc)" onClick={close}>
          <Icon name="x" size={14} />
        </button>
      </div>
    </div>
  );
}
