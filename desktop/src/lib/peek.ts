// The hidden sidebar shown for a moment: the pointer at the window's left edge
// (or on the title bar's sidebar button) slides it over the page, and it goes
// again once the pointer leaves. "ready" keeps it built while it's away, so it
// doesn't have to lay out every chat each time it comes back.
import { useEffect } from "react";
import { app } from "../state";

const OPEN_WAIT = 90;
const CLOSE_WAIT = 260;
let opening = 0;
let closing = 0;

/** The pointer reached the edge: builds the sidebar now, shows it shortly. */
export function peekIn(wait = OPEN_WAIT) {
  const s = app.get();
  if (s.sidebar || s.compact || !s.settings.sidePeek) return;
  clearTimeout(closing);
  closing = 0;
  if (s.peek === "open") return;
  if (s.peek === "off") app.set({ peek: "ready" });
  clearTimeout(opening);
  opening = window.setTimeout(() => app.set({ peek: "open" }), wait);
}

/** The pointer left before it opened. */
export const peekOff = () => clearTimeout(opening);

const shut = () => {
  clearTimeout(closing);
  closing = 0;
  if (app.get().peek === "open") app.set({ peek: "ready" });
};

/** Whether the sidebar is being used without the pointer on it: an open menu,
 *  a chat being renamed or searched for, or a chat being dragged. */
function inUse(dragging: boolean) {
  const now = document.activeElement;
  return dragging || !!document.querySelector(".menu") || !!now?.closest(".sidebar.peek input, .sidebar.peek textarea");
}

/** While it's open: follows the pointer, and shuts it when it has left. */
export function usePeekFollow(open: boolean) {
  useEffect(() => {
    if (!open) return;
    let dragging = false;
    const box = () => document.querySelector(".sidebar.peek");
    const later = (ms = CLOSE_WAIT) => {
      if (closing) return;
      closing = window.setTimeout(() => {
        closing = 0;
        if (inUse(dragging)) later(200);
        else shut();
      }, ms);
    };
    const stay = () => {
      clearTimeout(closing);
      closing = 0;
    };
    const move = (e: PointerEvent) => {
      const r = box()?.getBoundingClientRect();
      if (r && e.clientX <= r.right + 6) stay();
      else later();
    };
    // (the title bar is a drag area, and the page gets nothing from over it;
    // past the window's left edge the pointer is only overshooting)
    const out = (e: MouseEvent) => {
      if (e.clientX > 4) later();
    };
    const down = (e: PointerEvent) => {
      const t = e.target as Element;
      if (!t.closest?.(".sidebar.peek, .menu, .side-edge")) shut();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !document.querySelector(".menu")) shut();
    };
    const blur = () => {
      if (!inUse(dragging)) shut();
    };
    const drag = () => (dragging = true);
    const dropped = () => (dragging = false);
    const root = document.documentElement;
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerdown", down, true);
    window.addEventListener("keydown", key);
    window.addEventListener("blur", blur);
    window.addEventListener("dragstart", drag);
    window.addEventListener("dragend", dropped);
    root.addEventListener("mouseleave", out);
    return () => {
      stay();
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerdown", down, true);
      window.removeEventListener("keydown", key);
      window.removeEventListener("blur", blur);
      window.removeEventListener("dragstart", drag);
      window.removeEventListener("dragend", dropped);
      root.removeEventListener("mouseleave", out);
    };
  }, [open]);
}
