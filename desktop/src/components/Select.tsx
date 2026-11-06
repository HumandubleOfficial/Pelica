// A dropdown in Pelica's own menu, in place of the plain Windows one. Arrow
// keys, Enter and Escape work; the menu starts on the chosen row.
import { useRef, useState } from "react";
import { Icon } from "./Icons";
import { Menu } from "./Menu";

export function Select<T extends string | number>({ value, options, onChange, className = "", label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; className?: string; label?: string }) {
  const btn = useRef<HTMLButtonElement>(null);
  const [at, setAt] = useState<{ x: number; y: number; w: number } | null>(null);
  const open = () => {
    if (at) return setAt(null);
    const r = btn.current!.getBoundingClientRect();
    setAt({ x: r.left, y: r.bottom + 4, w: r.width });
  };
  return (
    <>
      <button
        ref={btn}
        type="button"
        className={`select ${className} ${at ? "open" : ""}`}
        aria-haspopup="menu"
        aria-expanded={!!at}
        aria-label={label}
        onClick={open}
        onKeyDown={(e) => {
          if ((e.key === "ArrowDown" || e.key === "ArrowUp") && !at) {
            e.preventDefault();
            open();
          }
        }}
      >
        <span className="select-text">{options.find((o) => o.value === value)?.label ?? ""}</span>
        <Icon name="chevronDown" size={13} />
      </button>
      {at && (
        <Menu
          at={at}
          minWidth={at.w}
          focus
          className="select-menu"
          ignore={btn}
          close={() => {
            setAt(null);
            btn.current?.focus();
          }}
          items={options.map((o) => ({ label: o.label, checked: o.value === value, on: () => onChange(o.value) }))}
        />
      )}
    </>
  );
}
