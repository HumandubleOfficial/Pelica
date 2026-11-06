// Segmented controls (.seg: Settings' choices, Models' tabs, the phone's
// Same Wi-Fi / Tailscale…) get one highlight block that slides to the chosen
// button. The block is the .seg's ::before (polish.css); this keeps its place
// and size in --seg-x/y/w/h as buttons are picked, added or resized.
const sized = new ResizeObserver((list) => list.forEach((e) => place(e.target as HTMLElement)));
const known = new WeakSet<HTMLElement>();

function place(seg: HTMLElement) {
  const sel = seg.querySelector<HTMLElement>(":scope > button.sel");
  if (!sel || !sel.offsetWidth) {
    seg.classList.remove("seg-live");
    return;
  }
  seg.style.setProperty("--seg-x", `${sel.offsetLeft}px`);
  seg.style.setProperty("--seg-y", `${sel.offsetTop}px`);
  seg.style.setProperty("--seg-w", `${sel.offsetWidth}px`);
  seg.style.setProperty("--seg-h", `${sel.offsetHeight}px`);
  if (!seg.classList.contains("seg-live")) {
    // (the first time it shows where it is; after that it slides)
    seg.classList.add("seg-live");
    requestAnimationFrame(() => seg.classList.add("seg-ready"));
  }
}

let queued = false;
function scan() {
  queued = false;
  document.querySelectorAll<HTMLElement>(".seg").forEach((seg) => {
    if (!known.has(seg)) {
      known.add(seg);
      sized.observe(seg);
    }
    place(seg);
  });
}

/** Whether a change on the page touches a .seg (a reply being written changes
 *  the page every frame: that mustn't cost a look through all of it). */
function touches(r: MutationRecord) {
  const t = r.target as HTMLElement;
  if (r.type === "attributes") return !!t.parentElement?.classList.contains("seg") || t.classList.contains("seg");
  if (t.classList?.contains("seg")) return true;
  for (const n of r.addedNodes) if (n instanceof HTMLElement && (n.classList.contains("seg") || n.querySelector(".seg"))) return true;
  return false;
}

export function startSegs() {
  const later = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(scan);
  };
  new MutationObserver((list) => list.some(touches) && later()).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["class"] });
  later();
}
