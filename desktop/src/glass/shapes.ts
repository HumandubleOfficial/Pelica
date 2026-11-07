import type { GlassOptions, Material, RGBA, RectLike } from './types';
export type { GlassOptions };

export const MAX_SHAPES = 64;
/** Float texture layout: 4 RGBA texels per row, MAX_SHAPES shape rows + MAX_SHAPES group rows. */
export const DATA_W = 4;
export const DATA_H = MAX_SHAPES * 2;

export interface Entry {
  el: HTMLElement;
  opts: GlassOptions;
  order: number;
  tint: RGBA | null;
  tone?: 'light' | 'dark';
}

/** A shape that made it into this frame, in viewport CSS px. */
export interface Placed {
  entry: Entry;
  rect: RectLike;
  tint: RGBA;
}

export interface Packed {
  data: Float32Array;
  groups: number;
  placed: Placed[];
}

const styleCache = new WeakMap<Element, CSSStyleDeclaration>();
function styleOf(el: Element) {
  let s = styleCache.get(el);
  if (!s) {
    s = getComputedStyle(el);
    styleCache.set(el, s);
  }
  return s;
}

/** CSS opacity of the element times all of its ancestors (glass has no DOM pixels to fade). */
export function effectiveOpacity(el: HTMLElement): number {
  let o = 1;
  for (let n: HTMLElement | null = el; n && n !== document.body; n = n.parentElement) {
    const s = styleOf(n);
    if (s.display === 'none') return 0;
    o *= parseFloat(s.opacity);
    if (o < 0.004) return 0;
  }
  if (styleOf(el).visibility === 'hidden') return 0;
  return o;
}

function cornerRadius(el: HTMLElement, rect: RectLike): number {
  const raw = styleOf(el).borderTopLeftRadius;
  const ow = el.offsetWidth || rect.width;
  const oh = el.offsetHeight || rect.height;
  let r = parseFloat(raw) || 0;
  if (raw.endsWith('%')) r = (r / 100) * Math.min(ow, oh);
  // Bounding boxes include CSS transforms; scale the radius to match.
  const sf = Math.min(rect.width / Math.max(ow, 1), rect.height / Math.max(oh, 1));
  return Math.min(r * sf, Math.min(rect.width, rect.height) / 2);
}

const NO_TINT: RGBA = [1, 1, 1, 0];
const NEUTRAL: [number, number, number] = [0.15, 0.15, 0.19];

/** In reduce-transparency mode, weak tints become a solid neutral (keeping a hint of their hue). */
export function effectiveTint(t: RGBA, m: Material): RGBA {
  if (!m.reduceTransparency || t[3] >= 0.6) return t;
  const a = t[3];
  return [
    NEUTRAL[0] + (t[0] - NEUTRAL[0]) * a,
    NEUTRAL[1] + (t[1] - NEUTRAL[1]) * a,
    NEUTRAL[2] + (t[2] - NEUTRAL[2]) * a,
    0.62,
  ];
}

/** Frost for a shape: its own value, else the material default for its variant. */
export function effectiveBlur(o: GlassOptions, m: Material): number {
  const b = o.blur ?? (o.variant === 'clear' ? m.blur * 0.2 : m.blur);
  return m.reduceTransparency ? Math.max(b, 16) : b;
}

/** How strongly a shape lifts toward grey over dark backdrops. */
export const liftOf = (o: GlassOptions) => (o.variant === 'clear' ? 0.3 : 1);

export function packShapes(
  entries: Iterable<Entry>,
  scale: number,
  material: Material,
  viewW: number,
  viewH: number,
  out: Float32Array,
  layer: 0 | 1,
): Packed {
  type Item = { e: Entry; rect: RectLike; opacity: number; radius: number; tint: RGBA };
  const groups = new Map<string, { z: number; first: number; items: Item[] }>();

  for (const e of entries) {
    if (!e.el.isConnected || (e.opts.layer ?? 0) !== layer) continue;
    const rect = e.opts.rect ? e.opts.rect() : e.el.getBoundingClientRect();
    if (rect.width < 0.5 || rect.height < 0.5) continue;
    if (rect.x > viewW + 60 || rect.y > viewH + 60 || rect.x + rect.width < -60 || rect.y + rect.height < -60) continue;
    const opacity = (e.opts.opacity ?? 1) * effectiveOpacity(e.el);
    if (opacity < 0.005) continue;
    const radius = e.opts.radius ?? cornerRadius(e.el, rect);
    const tint = effectiveTint(e.opts.tint === undefined ? material.tint : e.tint ?? NO_TINT, material);
    const key = e.opts.group ?? `\u0000${e.order}`;
    let g = groups.get(key);
    if (!g) groups.set(key, (g = { z: e.opts.z ?? 0, first: e.order, items: [] }));
    g.z = Math.min(g.z, e.opts.z ?? 0);
    g.first = Math.min(g.first, e.order);
    g.items.push({ e, rect, opacity, radius, tint });
  }

  const sorted = [...groups.values()].sort((a, b) => a.z - b.z || a.first - b.first);
  out.fill(0);
  const placed: Placed[] = [];
  let si = 0;
  let gi = 0;
  for (const g of sorted) {
    if (gi >= MAX_SHAPES || si >= MAX_SHAPES) break;
    g.items.sort((a, b) => a.e.order - b.e.order);
    const start = si;
    let k = 0;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const it of g.items) {
      if (si >= MAX_SHAPES) break;
      const { rect, e } = it;
      const hw = (rect.width / 2) * scale;
      const hh = (rect.height / 2) * scale;
      const o = si * DATA_W * 4;
      out[o + 0] = (rect.x + rect.width / 2) * scale;
      out[o + 1] = (rect.y + rect.height / 2) * scale;
      out[o + 2] = hw;
      out[o + 3] = hh;
      out[o + 4] = it.radius * scale;
      out[o + 5] = effectiveBlur(e.opts, material) * scale;
      out[o + 6] = (e.opts.refraction ?? 1) * (material.reduceTransparency ? 0.35 : 1);
      out[o + 7] = Math.min((e.opts.bevel ?? material.bevel) * scale, Math.min(hw, hh));
      out[o + 8] = it.tint[0];
      out[o + 9] = it.tint[1];
      out[o + 10] = it.tint[2];
      out[o + 11] = it.tint[3];
      out[o + 12] = e.opts.glow ?? 0;
      out[o + 13] = it.opacity;
      out[o + 14] = liftOf(e.opts);
      out[o + 15] = e.opts.dispersion ?? 1;
      k = Math.max(k, e.opts.merge ?? 0);
      x0 = Math.min(x0, rect.x);
      y0 = Math.min(y0, rect.y);
      x1 = Math.max(x1, rect.x + rect.width);
      y1 = Math.max(y1, rect.y + rect.height);
      placed.push({ entry: e, rect, tint: it.tint });
      si++;
    }
    // Pad the group box for the melt and the contact shadow.
    const pad = k + 34;
    const go = (MAX_SHAPES + gi) * DATA_W * 4;
    out[go + 0] = (x0 - pad) * scale;
    out[go + 1] = (y0 - pad) * scale;
    out[go + 2] = (x1 + pad) * scale;
    out[go + 3] = (y1 + pad) * scale;
    out[go + 4] = start;
    out[go + 5] = si;
    out[go + 6] = k * scale;
    out[go + 7] = g.z;
    gi++;
  }
  return { data: out, groups: gi, placed };
}
