// A part of a page that stays shut until it's wanted.
import { useState, type ReactNode } from "react";

export function Fold({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`fold ${open ? "open" : ""}`}>
      <button className="fold-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className="fold-caret" />
        {title}
      </button>
      {open && <div className="fold-body">{children}</div>}
    </div>
  );
}
