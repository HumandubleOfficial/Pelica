// Pelica's own tooltips: a small dark label above whatever the pointer rests
// on, for every element with a title (the title moves to data-tip so the
// browser's own tooltip doesn't show as well).
import { useEffect, useLayoutEffect, useRef, useState } from "react";

interface Tip {
  text: string;
  x: number;
  y: number;
  below: boolean;
}

export function Tooltips() {
  const [tip, setTip] = useState<Tip | null>(null);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let timer = 0;
    let cur: HTMLElement | null = null;
    const hide = () => {
      clearTimeout(timer);
      cur = null;
      setTip(null);
    };
    const over = (e: MouseEvent) => {
      const el = (e.target as Element | null)?.closest?.<HTMLElement>("[title], [data-tip]") ?? null;
      if (el === cur) return;
      hide();
      if (!el) return;
      const t = el.getAttribute("title");
      if (t !== null) {
        el.removeAttribute("title");
        el.dataset.tip = t;
        // (an icon-only button keeps its name for screen readers)
        if (t && !el.hasAttribute("aria-label") && !el.textContent?.trim()) el.setAttribute("aria-label", t);
      }
      const text = el.dataset.tip;
      if (!text) return;
      cur = el;
      timer = window.setTimeout(() => {
        if (!cur?.isConnected) return;
        const r = cur.getBoundingClientRect();
        const below = r.top < 44;
        setTip({ text, x: r.left + r.width / 2, y: below ? r.bottom + 6 : r.top - 6, below });
      }, 400);
    };
    const leave = (e: MouseEvent) => !e.relatedTarget && hide();
    document.addEventListener("mouseover", over);
    document.addEventListener("mouseout", leave);
    document.addEventListener("mousedown", hide, true);
    document.addEventListener("keydown", hide, true);
    document.addEventListener("scroll", hide, true);
    window.addEventListener("blur", hide);
    return () => {
      clearTimeout(timer);
      document.removeEventListener("mouseover", over);
      document.removeEventListener("mouseout", leave);
      document.removeEventListener("mousedown", hide, true);
      document.removeEventListener("keydown", hide, true);
      document.removeEventListener("scroll", hide, true);
      window.removeEventListener("blur", hide);
    };
  }, []);
  // kept inside the window
  useLayoutEffect(() => {
    const el = box.current;
    if (!el || !tip) return;
    const w = el.offsetWidth;
    el.style.left = `${Math.max(6, Math.min(window.innerWidth - w - 6, tip.x - w / 2))}px`;
  }, [tip]);
  if (!tip) return null;
  return (
    <div ref={box} className={`tooltip ${tip.below ? "below" : ""}`} style={{ left: tip.x, top: tip.y }} role="tooltip">
      {tip.text}
    </div>
  );
}
