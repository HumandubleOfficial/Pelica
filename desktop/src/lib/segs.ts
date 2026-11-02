// Segmented controls (.seg: Settings' choices, Models' tabs, the phone's
// Same Wi-Fi / Tailscale…) get one highlight block that slides to the chosen
// button, instead of the button lighting up in place. The block is the
// .seg's ::before (polish.css, "sliding segments"); this keeps its place and
// size in --seg-x / --seg-w as buttons are picked, added or resized.
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

// Lists get the same treatment, up and down: Settings' sections, the
// sidebar's pages and its chats. The block is the list's ::before (motion.css,
// "gliding lists"), placed at the chosen row in --glide-x/y/w/h.
const GLIDES: [list: string, chosen: string][] = [
  [".settings .set-nav-list", "button.sel"],
  [".sidebar .side-nav", ".nav-item.active"],
  [".sidebar .side-lists", ".chat-row.active"],
];

function glide(list: HTMLElement, chosen: string) {
  const sel = list.querySelector<HTMLElement>(chosen);
  if (!sel || !sel.offsetWidth) {
    list.classList.remove("glide-live", "glide-ready");
    return;
  }
  const a = list.getBoundingClientRect();
  const b = sel.getBoundingClientRect();
  list.style.setProperty("--glide-x", `${b.left - a.left + list.scrollLeft - list.clientLeft}px`);
  list.style.setProperty("--glide-y", `${b.top - a.top + list.scrollTop - list.clientTop}px`);
  list.style.setProperty("--glide-w", `${b.width}px`);
  list.style.setProperty("--glide-h", `${b.height}px`);
  if (!list.classList.contains("glide-live")) {
    list.classList.add("glide", "glide-live");
    requestAnimationFrame(() => list.classList.add("glide-ready"));
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
  for (const [sel, chosen] of GLIDES) {
    document.querySelectorAll<HTMLElement>(sel).forEach((list) => {
      if (!known.has(list)) {
        known.add(list);
        glides.observe(list);
      }
      glide(list, chosen);
    });
  }
}
const glides = new ResizeObserver((list) =>
  list.forEach((e) => {
    const g = GLIDES.find(([sel]) => (e.target as HTMLElement).matches(sel));
    if (g) glide(e.target as HTMLElement, g[1]);
  }),
);

export function startSegs() {
  const later = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(scan);
  };
  new MutationObserver(later).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["class"] });
  later();
}
