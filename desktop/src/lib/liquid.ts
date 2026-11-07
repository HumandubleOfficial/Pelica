// The Liquid Glass theme (experimental). Two kinds of glass:
//
// - Base surfaces (the sidebar, the chat sheet, the message box, your
//   messages) are drawn by the WebGL glass from the ALG library (src/glass):
//   a wallpaper behind the window, and real refracting, frosted glass shaped
//   like each element. The page's HTML sits on top, with no background of its own.
// - Things that float over text (menus, pickers, Settings, toasts) can't be
//   WebGL glass, because WebGL can't see the HTML under them. They're CSS glass
//   (backdrop-filter, styles in glass.css), and they grow out of the button that
//   opened them as one liquid shape: clear while moving, frosted once at rest,
//   content last, like iOS. Closing folds them back into the button.
//
// The glass only redraws when a shape moves, so it doesn't take GPU time
// from the model while you read or while a reply streams.
import { createGlassScene, Spring, type GlassHandle, type GlassOptions, type GlassScene, type RectLike } from "../glass";
import { app } from "../state";

export type GlassWall = "night" | "bloom" | "dunes" | "tiles" | "picture";
const PRESET: Record<Exclude<GlassWall, "picture">, number> = { bloom: 0, dunes: 1, tiles: 2, night: 4 };

type Surface = { sel: string; opts: GlassOptions; clip?: string };
const SURFACES: Surface[] = [
  { sel: ".shell > .sidebar:not(.peek)", opts: { tint: [0.03, 0.04, 0.07, 0.18], blur: 10, bevel: 14 } },
  { sel: ".shell > .main", opts: { tint: [0.02, 0.03, 0.06, 0.36], blur: 14, bevel: 16 } },
  // (floating: they refract the chat sheet under them as well as the wallpaper)
  { sel: ".composer", opts: { layer: 1, tint: [1, 1, 1, 0.05], blur: 6, bevel: 16 } },
  { sel: ".user-bubble", opts: { layer: 1, tint: [1, 1, 1, 0.09], blur: 4, bevel: 10 }, clip: ".chat-scroll" },
];

let scene: GlassScene | null = null;
let stops: (() => void)[] = [];
const shapes = new Map<HTMLElement, GlassHandle>();

/** The part of `el` inside its scroller (a message scrolled half out of view is half a shape). */
const clipped = (el: HTMLElement, sel: string) => (): RectLike => {
  const r = el.getBoundingClientRect();
  const box = el.closest(sel)?.getBoundingClientRect();
  if (!box) return r;
  const x0 = Math.max(r.left, box.left);
  const y0 = Math.max(r.top, box.top);
  const x1 = Math.min(r.right, box.right);
  const y1 = Math.min(r.bottom, box.bottom);
  return { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0) };
};

/** Glass for every surface on the page now; none for ones that went away. */
function sync() {
  if (!scene) return;
  const seen = new Set<HTMLElement>();
  for (const s of SURFACES) {
    document.querySelectorAll<HTMLElement>(s.sel).forEach((el) => {
      seen.add(el);
      if (!shapes.has(el)) shapes.set(el, scene!.add(el, s.clip ? { ...s.opts, rect: clipped(el, s.clip) } : s.opts));
    });
  }
  for (const [el, h] of shapes) {
    if (!seen.has(el)) {
      h.remove();
      shapes.delete(el);
    }
  }
}

// ---------- the wallpaper ----------

const DB = "pelica-glass";
function idb<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest): Promise<T> {
  return new Promise((ok, no) => {
    const open = indexedDB.open(DB, 1);
    open.onupgradeneeded = () => open.result.createObjectStore("wall");
    open.onerror = () => no(open.error);
    open.onsuccess = () => {
      const tx = open.result.transaction("wall", mode);
      const req = fn(tx.objectStore("wall"));
      req.onsuccess = () => ok(req.result as T);
      req.onerror = () => no(req.error);
    };
  });
}
/** Your own picture as the wallpaper (kept in this app's storage, not in settings.json). */
export const savePicture = (f: Blob) => idb<void>("readwrite", (s) => s.put(f, "picture"));
const loadPicture = () => idb<Blob | undefined>("readonly", (s) => s.get("picture"));

let wallNow = "";
async function setWall(w: GlassWall) {
  if (!scene || w === wallNow) return;
  wallNow = w;
  if (w !== "picture") {
    scene.setMaterial({ exposure: 1 });
    return scene.setBackground({ type: "procedural", preset: PRESET[w] ?? 4, animate: false });
  }
  const blob = await loadPicture().catch(() => undefined);
  const bmp = blob ? await createImageBitmap(blob).catch(() => null) : null;
  if (!scene || wallNow !== w) return;
  // (no picture, or one that can't be read: Night)
  scene.setBackground(bmp ? { type: "image", source: bmp } : { type: "procedural", preset: 4, animate: false });
  // a bright picture is dimmed until white text reads on it
  scene.setMaterial({ exposure: bmp ? Math.min(1, 0.22 / Math.max(0.05, brightness(bmp))) : 1 });
}

/** A picture's average brightness, 0..1. */
function brightness(src: ImageBitmap) {
  const c = document.createElement("canvas");
  c.width = c.height = 24;
  const g = c.getContext("2d", { willReadFrequently: true })!;
  g.drawImage(src, 0, 0, 24, 24);
  const d = g.getImageData(0, 0, 24, 24).data;
  let sum = 0;
  for (let i = 0; i < d.length; i += 4) sum += (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255;
  return sum / (d.length / 4);
}
/** After a new picture was saved: show it. */
export function reloadPicture() {
  if (wallNow === "picture") wallNow = "";
  setWall(app.get().settings.glassWall);
}

// ---------- menus and windows that grow out of their button ----------

const FLOAT = ".popover, .modal, .toast, .palette";
type Morph = { el: HTMLElement; from: DOMRect | null; k: Spring; closing: boolean; last: DOMRect | null };
/** Shapes moving now. */
const morphs = new Map<HTMLElement, Morph>();
/** Every menu or window open now: its box and where it came from, for folding it back. */
const known = new Map<HTMLElement, { last: DOMRect | null; from: DOMRect | null }>();
let pressed: { rect: DOMRect; t: number } | null = null;
let raf = 0;

const motion = () => app.get().settings.animations && !matchMedia("(prefers-reduced-motion: reduce)").matches;

/** The control a menu came from: the row a submenu opens beside, else what was just pressed. */
function originOf(el: HTMLElement): DOMRect | null {
  if (el.classList.contains("menu-sub")) return el.closest(".mi-wrap")?.querySelector(":scope > .mi")?.getBoundingClientRect() ?? null;
  if (pressed && performance.now() - pressed.t < 900) return pressed.rect;
  return null;
}

/** The element's own box and its CSS transform's translation (ours left out). */
function measure(el: HTMLElement) {
  el.style.removeProperty("translate");
  el.style.removeProperty("scale");
  const b = el.getBoundingClientRect();
  const m = new DOMMatrixReadOnly(getComputedStyle(el).transform === "none" ? undefined : getComputedStyle(el).transform);
  return { b, tx: m.e, ty: m.f };
}

function frame(t: number) {
  raf = 0;
  const dt = Math.min(0.05, (t - last) / 1000 || 1 / 60);
  last = t;
  for (const m of morphs.values()) {
    const { el } = m;
    const k = m.k.step(dt);
    const { b, tx, ty } = m.closing && m.last ? { b: m.last, tx: 0, ty: 0 } : measure(el);
    if (!m.closing) {
      m.last = b;
      const kn = known.get(el);
      if (kn) kn.last = b;
    }
    const a = m.from ?? new DOMRect(b.x + b.width * 0.35, b.y + b.height * 0.35, b.width * 0.3, b.height * 0.3);
    // height trails width a touch, as in the ALG demo
    const kx = Math.max(0, k);
    const ky = Math.max(0, k * 1.12 - 0.12);
    const sx = (a.width + (b.width - a.width) * kx) / Math.max(b.width, 1);
    const sy = (a.height + (b.height - a.height) * ky) / Math.max(b.height, 1);
    const cx = a.x + a.width / 2 + (b.x + b.width / 2 - (a.x + a.width / 2)) * kx;
    const cy = a.y + a.height / 2 + (b.y + b.height / 2 - (a.y + a.height / 2)) * ky;
    // (scale works about the layout box's centre; the CSS transform's own shift is undone)
    const lx = b.x + b.width / 2 - tx;
    const ly = b.y + b.height / 2 - ty;
    const px = cx - lx - sx * (b.x + b.width / 2 - lx);
    const py = cy - ly - sy * (b.y + b.height / 2 - ly);
    el.style.translate = `${px.toFixed(2)}px ${py.toFixed(2)}px`;
    el.style.scale = `${Math.max(0.01, sx).toFixed(4)} ${Math.max(0.01, sy).toFixed(4)}`;
    // clear and refractive while it moves, frosted at rest; the content comes in last
    const clarity = Math.min(1, Math.max(1 - k, Math.abs(m.k.velocity) / 5));
    const shown = m.closing ? Math.min(1, Math.max(0, (k - 0.55) / 0.35)) : Math.min(1, Math.max(0, (k - 0.4) / 0.5));
    el.style.setProperty("--lg-frost", (1 - clarity).toFixed(3));
    el.style.setProperty("--lg-in", (shown * shown * (3 - 2 * shown)).toFixed(3));
    el.style.opacity = m.closing ? Math.min(1, k * 4).toFixed(3) : Math.min(1, k * 8).toFixed(3);
    const done = m.k.settled || (m.closing && k < 0.03);
    if (done) {
      morphs.delete(el);
      if (m.closing) el.remove();
      else {
        el.classList.remove("lg-morph");
        for (const p of ["translate", "scale", "opacity", "--lg-frost", "--lg-in"]) el.style.removeProperty(p);
      }
    }
  }
  if (morphs.size) raf = requestAnimationFrame(frame);
}
let last = 0;
const run = () => {
  if (!raf) {
    last = performance.now();
    raf = requestAnimationFrame(frame);
  }
};

function grow(el: HTMLElement) {
  if (known.has(el)) return;
  const from = originOf(el);
  known.set(el, { last: null, from });
  if (!motion()) return;
  el.classList.add("lg-morph");
  el.style.opacity = "0";
  morphs.set(el, { el, from, k: new Spring(0, { stiffness: 230, damping: 19 }).set(1), closing: false, last: null });
  run();
}

/** A menu that just closed comes back for a moment, as a ghost, and folds into its button. */
function fold(el: HTMLElement) {
  const kn = known.get(el);
  known.delete(el);
  const was = morphs.get(el);
  morphs.delete(el);
  const b = was?.last ?? kn?.last ?? null;
  if (!b || !motion() || b.width < 4) return;
  const from = was?.from ?? kn?.from ?? null;
  for (const p of ["translate", "scale", "opacity"]) el.style.removeProperty(p);
  Object.assign(el.style, { position: "fixed", left: `${b.x}px`, top: `${b.y}px`, width: `${b.width}px`, height: `${b.height}px`, right: "auto", bottom: "auto", margin: "0", transform: "none", pointerEvents: "none", zIndex: "400" });
  el.classList.add("lg-morph", "lg-ghost");
  el.setAttribute("aria-hidden", "true");
  document.body.appendChild(el);
  const k = new Spring(Math.min(1.05, was?.k.value ?? 1), { stiffness: 330, damping: 28 });
  k.set(0);
  morphs.set(el, { el, from, k, closing: true, last: b });
  run();
}

let mo: MutationObserver | null = null;
function watch() {
  let queued = 0;
  mo = new MutationObserver((list) => {
    for (const r of list) {
      r.addedNodes.forEach((n) => {
        if (!(n instanceof HTMLElement) || n.classList.contains("lg-ghost")) return;
        const hits = n.matches(FLOAT) ? [n] : [...n.querySelectorAll<HTMLElement>(FLOAT)];
        // (a menu inside one that's opening grows with it)
        for (const h of hits) if (!hits.some((o) => o !== h && o.contains(h))) grow(h);
      });
      r.removedNodes.forEach((n) => {
        if (!(n instanceof HTMLElement) || n.classList.contains("lg-ghost")) return;
        for (const el of [...known.keys()]) if ((el === n || n.contains(el)) && !el.isConnected) fold(el);
      });
    }
    if (!queued) queued = requestAnimationFrame(() => ((queued = 0), sync()));
  });
  mo.observe(document.body, { childList: true, subtree: true });
  // a menu opened by what was just pressed (or chosen with the keyboard)
  const down = (e: Event) => {
    const b = (e.target as Element | null)?.closest?.("button, [role=button], .mi, .chip, a");
    if (b) pressed = { rect: b.getBoundingClientRect(), t: performance.now() };
  };
  const key = (e: KeyboardEvent) => (e.key === "Enter" || e.key === " ") && down({ target: document.activeElement } as unknown as Event);
  document.addEventListener("pointerdown", down, true);
  document.addEventListener("keydown", key, true);
  stops.push(() => {
    document.removeEventListener("pointerdown", down, true);
    document.removeEventListener("keydown", key, true);
  });
  // every menu open now keeps its size for a fold later
  const keep = setInterval(() => {
    document.querySelectorAll<HTMLElement>(FLOAT).forEach((el) => {
      if (el.classList.contains("lg-ghost") || morphs.has(el)) return;
      const kn = known.get(el);
      if (kn) kn.last = el.getBoundingClientRect();
      else known.set(el, { last: el.getBoundingClientRect(), from: null });
    });
    for (const el of known.keys()) if (!el.isConnected) known.delete(el);
  }, 250);
  stops.push(() => clearInterval(keep));
}

// ---------- on and off ----------

export function startGlass() {
  if (scene) return;
  document.documentElement.classList.add("lg-on");
  scene = createGlassScene({ material: { lift: 0.12, legibility: 0.6, shadow: 0.16 }, background: { type: "procedural", preset: 4, animate: false } });
  wallNow = "";
  setWall(app.get().settings.glassWall);
  stops.push(app.sub(() => setWall(app.get().settings.glassWall)));
  watch();
  sync();
  // (classes change without nodes coming or going: a backstop)
  const t = setInterval(sync, 1000);
  stops.push(() => clearInterval(t));
  (window as any).__glass = scene;
}

export function stopGlass() {
  if (!scene) return;
  mo?.disconnect();
  mo = null;
  stops.forEach((f) => f());
  stops = [];
  for (const [el, m] of morphs) {
    if (m.closing) el.remove();
    else for (const p of ["translate", "scale", "opacity", "--lg-frost", "--lg-in"]) el.style.removeProperty(p);
    el.classList.remove("lg-morph");
  }
  morphs.clear();
  known.clear();
  shapes.clear();
  scene.destroy();
  scene.canvas.remove();
  scene = null;
  wallNow = "";
  document.documentElement.classList.remove("lg-on");
  delete (window as any).__glass;
}
