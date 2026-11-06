// One menu for the whole app, as in Claude's: rows with an icon, a label and
// a letter (press it while the menu is open), thin separators, submenus that
// open to the side, and Delete in red. It opens at once (no animation).
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { createPortal } from "react-dom";
import { Icon } from "./Icons";

export type MenuItem =
  | "sep"
  | { head: string }
  | {
      label: string;
      icon?: string;
      /** a letter (or text like "Ctrl+;") shown on the right; a single letter also works as a shortcut */
      key?: string;
      desc?: string;
      danger?: boolean;
      disabled?: boolean;
      /** a check mark on the right */
      checked?: boolean;
      /** something on the right instead of the letter (a switch) */
      right?: ReactNode;
      /** a menu that opens to the side */
      sub?: MenuItem[];
      /** keep the menu open after choosing (switches) */
      stay?: boolean;
      on?: () => void;
    };

type Row = Exclude<MenuItem, "sep" | { head: string }>;
const isRow = (i: MenuItem): i is Row => typeof i === "object" && "label" in i;

function List({ items, close, className = "" }: { items: MenuItem[]; close: () => void; className?: string }) {
  const [open, setOpen] = useState(-1);
  const ref = useRef<HTMLDivElement>(null);
  const [flip, setFlip] = useState(false);
  // a submenu that would run off the right edge opens to the left
  useLayoutEffect(() => {
    const r = ref.current?.getBoundingClientRect();
    if (r && className.includes("menu-sub")) setFlip(r.right > window.innerWidth - 8);
  }, [className]);
  const run = (i: Row) => {
    if (i.disabled) return;
    if (i.sub) return setOpen(items.indexOf(i));
    i.on?.();
    if (!i.stay) close();
  };
  return (
    <div className={`popover menu ${className} ${flip ? "flip" : ""}`} ref={ref} role="menu" onMouseDown={(e) => e.stopPropagation()}>
      {items.map((it, n) => {
        if (it === "sep") return <div key={n} className="menu-sep" />;
        if (!isRow(it)) return <div key={n} className="menu-head">{it.head}</div>;
        return (
          <div key={n} className="mi-wrap" onMouseEnter={() => setOpen(it.sub ? n : -1)}>
            <button className={`mi ${it.danger ? "danger" : ""} ${open === n ? "open" : ""}`} role="menuitem" disabled={it.disabled} onClick={() => run(it)}>
              {it.icon && <Icon name={it.icon} size={15} />}
              <span className="mi-text">
                <span className="mi-label">{it.label}</span>
                {it.desc && <span className="mi-desc">{it.desc}</span>}
              </span>
              {it.checked && <Icon name="check" size={15} stroke={2.4} className="mi-check" />}
              {it.right}
              {it.key && !it.sub && <span className="mi-key">{it.key}</span>}
              {it.sub && <Icon name="chevron" size={13} className="mi-more" />}
            </button>
            {it.sub && open === n && <List items={it.sub} close={close} className="menu-sub" />}
          </div>
        );
      })}
    </div>
  );
}

/** A menu: under its trigger (`className` places it), or at a point (`at`,
 *  for right-click). Letters run their rows; Escape and clicks outside close it. */
export function Menu({ items, close, at, className, ignore, minWidth, focus }: { items: MenuItem[]; close: () => void; at?: { x: number; y: number }; className?: string; ignore?: RefObject<HTMLElement | null>; minWidth?: number; focus?: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(at);
  useEffect(() => {
    if (!focus) return;
    const rows = box.current?.querySelectorAll<HTMLElement>(".mi");
    const on = box.current?.querySelector<HTMLElement>(".mi-check")?.closest<HTMLElement>(".mi");
    (on || rows?.[0])?.focus();
  }, []);
  useLayoutEffect(() => {
    if (!at || !box.current) return;
    const r = box.current.getBoundingClientRect();
    setPos({ x: Math.min(at.x, window.innerWidth - r.width - 8), y: Math.min(at.y, window.innerHeight - r.height - 8) });
  }, [at?.x, at?.y]);
  useEffect(() => {
    const down = (e: MouseEvent) => {
      const t = e.target as Node;
      if (box.current?.contains(t) || ignore?.current?.contains(t)) return;
      close();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") return e.stopPropagation(), close();
      if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const now = document.activeElement as HTMLElement | null;
        const inside = !!now && !!box.current?.contains(now);
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          const list = (inside ? now!.closest(".menu") : box.current?.querySelector(".menu")) as HTMLElement | null;
          const rows = [...(list?.querySelectorAll<HTMLElement>(":scope > .mi-wrap > .mi:not(:disabled)") || [])];
          if (!rows.length) return;
          const i = inside ? rows.indexOf(now!) : -1;
          const d = e.key === "ArrowDown" ? 1 : -1;
          rows[i < 0 ? (d > 0 ? 0 : rows.length - 1) : (i + d + rows.length) % rows.length].focus();
        } else if (inside && e.key === "ArrowRight" && now!.querySelector(".mi-more")) {
          now!.click();
          setTimeout(() => now!.closest(".mi-wrap")?.querySelector<HTMLElement>(".menu-sub .mi:not(:disabled)")?.focus());
        } else if (inside && e.key === "ArrowLeft") {
          now!.closest(".menu-sub")?.closest(".mi-wrap")?.querySelector<HTMLElement>(":scope > .mi")?.focus();
        } else return;
        e.preventDefault();
        e.stopPropagation();
        return;
      }
      if (e.ctrlKey || e.altKey || e.metaKey || e.key.length !== 1) return;
      if ((e.target as HTMLElement)?.closest?.("input, textarea, [contenteditable=true]")) return;
      const all = (list: MenuItem[]): Row[] => list.filter(isRow);
      const hit = all(items).find((i) => i.key?.length === 1 && i.key.toLowerCase() === e.key.toLowerCase() && !i.disabled && !i.sub);
      if (!hit) return;
      e.preventDefault();
      e.stopPropagation();
      hit.on?.();
      if (!hit.stay) close();
    };
    document.addEventListener("mousedown", down);
    document.addEventListener("keydown", key, true);
    return () => {
      document.removeEventListener("mousedown", down);
      document.removeEventListener("keydown", key, true);
    };
  }, [items, close]);
  const list = (
    <div ref={box} className={at ? "menu-at" : "menu-anchor"} style={at ? ({ left: pos?.x ?? at.x, top: pos?.y ?? at.y, "--menu-min": minWidth ? `${minWidth}px` : undefined } as React.CSSProperties) : undefined}>
      <List items={items} close={close} className={className} />
    </div>
  );
  return at ? createPortal(list, document.body) : list;
}
