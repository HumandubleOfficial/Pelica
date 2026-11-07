import { rasterize } from './raster';
import type { ContentQuad, GLRenderer } from './renderer';
import { effectiveOpacity, type Entry } from './shapes';

export interface ContentOptions {
  /** Raster resolution multiplier on top of the canvas scale (e.g. 1.6 for icons that magnify). */
  resolution?: number;
  /** Transparent margin captured around the element in CSS px (shadows, glows). Default 10. */
  pad?: number;
}

export interface ContentHandle {
  readonly el: Element;
  /** Re-capture now (e.g. after a change the observers can't see). */
  refresh(): void;
  remove(): void;
}

interface Item {
  el: HTMLElement | SVGElement;
  opts: ContentOptions;
  tex: WebGLTexture | null;
  w: number;
  h: number;
  pad: number;
  dirty: boolean;
  busy: boolean;
  last: number;
  owner: Entry | null;
  clips: Element[];
}

const sameClasses = (a: string | null, b: string | null) => {
  const norm = (s: string | null) => (s ?? '').split(/\s+/).filter((c) => c && c !== 'lg-content-hidden').sort().join(' ');
  return norm(a) === norm(b);
};

/** Largest refraction offset content can be pulled by, CSS px (sizes the drawn quads). */
const REACH = 70;
const MIN_INTERVAL = 50;
const MAX_INFLIGHT = 6;

/**
 * Keeps rasterized copies of HTML elements and places them each frame so the
 * renderer can draw them *through* the glass (refracted, fringed, and hidden
 * under higher glass). The real elements stay in the page, invisible
 * (`filter: opacity(0)`), so clicks, focus and screen readers still work.
 */
export class ContentManager {
  private items = new Map<Element, Item>();
  private inflight = 0;
  private mo: MutationObserver;
  private ro: ResizeObserver;
  private ownerMo: MutationObserver;
  private watchedOwners = new WeakSet<Element>();

  constructor(
    private gl: GLRenderer,
    private ownerOf: (el: Element) => Entry | null,
  ) {
    this.mo = new MutationObserver((records) => {
      for (const r of records) {
        if (r.type === 'attributes' && this.items.has(r.target as Element)) {
          // Root geometry/opacity (inline style) is read every frame, not captured.
          if (r.attributeName === 'style') continue;
          // Our own hide class toggling is not a content change.
          if (r.attributeName === 'class' && sameClasses(r.oldValue, (r.target as Element).getAttribute('class'))) continue;
        }
        this.dirtyFrom(r.target);
      }
    });
    this.ownerMo = new MutationObserver((records) => {
      for (const r of records) this.dirtyInside(r.target as Element);
    });
    this.ro = new ResizeObserver((entries) => entries.forEach((e) => this.mark(e.target)));
  }

  add(el: HTMLElement | SVGElement, opts: ContentOptions = {}): ContentHandle {
    let it = this.items.get(el);
    if (!it) {
      it = { el, opts, tex: null, w: 0, h: 0, pad: opts.pad ?? 10, dirty: true, busy: false, last: 0, owner: null, clips: [] };
      this.items.set(el, it);
      this.mo.observe(el, { subtree: true, childList: true, characterData: true, attributes: true, attributeOldValue: true, attributeFilter: ['class', 'style', 'value', 'checked', 'aria-checked', 'aria-pressed', 'aria-selected', 'src', 'd'] });
      this.ro.observe(el);
      for (const ev of ['pointerenter', 'pointerleave', 'pointerdown', 'pointerup', 'focusin', 'focusout', 'input', 'change']) {
        el.addEventListener(ev, () => setTimeout(() => this.mark(el), 0));
      }
      this.resolve(it);
    } else Object.assign(it.opts, opts);
    return { el, refresh: () => this.mark(el), remove: () => this.remove(el) };
  }

  remove(el: Element) {
    const it = this.items.get(el);
    if (!it) return;
    this.items.delete(el);
    this.ro.unobserve(el);
    el.classList.remove('lg-content-hidden');
    if (it.tex) this.gl.deleteContent(it.tex);
  }

  /** Mark every content element inside `root` for re-capture. */
  dirtyInside(root: Element) {
    for (const it of this.items.values()) if (root.contains(it.el)) it.dirty = true;
  }

  private mark(el: Element) {
    const it = this.items.get(el);
    if (it) it.dirty = true;
  }

  private dirtyFrom(node: Node) {
    for (let n: Node | null = node; n; n = n.parentNode) {
      const it = this.items.get(n as Element);
      if (it) {
        it.dirty = true;
        return;
      }
    }
  }

  /** Find the owning glass and the scroll containers that clip this element. */
  private resolve(it: Item) {
    it.owner = this.ownerOf(it.el);
    if (it.owner && !this.watchedOwners.has(it.owner.el)) {
      this.watchedOwners.add(it.owner.el);
      // Tone flips and state classes on the glass restyle the content inside it.
      this.ownerMo.observe(it.owner.el, { attributes: true, attributeFilter: ['class', 'data-glass-tone', 'aria-checked', 'aria-pressed'] });
    }
    it.clips = [];
    for (let n = it.el.parentElement; n && n !== document.body; n = n.parentElement) {
      const cs = getComputedStyle(n);
      if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') it.clips.push(n);
    }
  }

  /** Start re-captures for changed elements (throttled, a few at a time). */
  update(now: number) {
    for (const it of this.items.values()) {
      if (!it.dirty || it.busy || this.inflight >= MAX_INFLIGHT) continue;
      if (now - it.last < MIN_INTERVAL) continue;
      if (!it.el.isConnected) continue;
      it.dirty = false;
      it.busy = true;
      it.last = now;
      this.inflight++;
      const scale = Math.min(this.gl.scale * (it.opts.resolution ?? 1), 4);
      rasterize(it.el, scale, it.pad)
        .then((r) => {
          if (!r || !this.items.has(it.el)) return;
          it.tex = this.gl.uploadContent(it.tex, r.image);
          it.w = r.width;
          it.h = r.height;
          it.el.classList.add('lg-content-hidden');
        })
        .catch((err) => console.warn('[liquid-glass] content capture failed', err))
        .finally(() => {
          it.busy = false;
          this.inflight--;
        });
    }
  }

  /** Content placed for this frame, split into the backdrop/base/floating layers. */
  quads(scale: number): { base: ContentQuad[]; floating: ContentQuad[] } {
    const base: (ContentQuad & { order: number })[] = [];
    const floating: (ContentQuad & { order: number })[] = [];
    let order = 0;
    for (const it of this.items.values()) {
      order++;
      if (!it.tex || !it.el.isConnected) continue;
      if (!it.owner || !it.owner.el.isConnected) this.resolve(it);
      // While focused, the real element shows (with its focus ring) instead.
      if (it.el.matches(':focus-visible, :has(:focus-visible)')) continue;
      const opacity = effectiveOpacity(it.el as HTMLElement);
      if (opacity < 0.005) continue;
      const b = it.el.getBoundingClientRect();
      if (b.width < 0.5 || b.height < 0.5) continue;
      const sx = b.width / it.w;
      const sy = b.height / it.h;
      const rx = b.x - it.pad * sx;
      const ry = b.y - it.pad * sy;
      const rw = b.width + it.pad * 2 * sx;
      const rh = b.height + it.pad * 2 * sy;
      let x0 = rx - REACH, y0 = ry - REACH, x1 = rx + rw + REACH, y1 = ry + rh + REACH;
      for (const c of it.clips) {
        const cr = c.getBoundingClientRect();
        x0 = Math.max(x0, cr.left);
        y0 = Math.max(y0, cr.top);
        x1 = Math.min(x1, cr.right);
        y1 = Math.min(y1, cr.bottom);
      }
      if (x1 <= x0 || y1 <= y0) continue;
      const q = {
        tex: it.tex,
        rect: [rx * scale, ry * scale, rw * scale, rh * scale] as [number, number, number, number],
        geom: [x0 * scale, y0 * scale, x1 * scale, y1 * scale] as [number, number, number, number],
        opacity,
        z: it.owner?.opts.z ?? 0,
        order,
      };
      ((it.owner?.opts.layer ?? 0) === 1 ? floating : base).push(q);
    }
    const sort = (a: { z: number; order: number }, b: { z: number; order: number }) => a.z - b.z || a.order - b.order;
    return { base: base.sort(sort), floating: floating.sort(sort) };
  }

  destroy() {
    this.mo.disconnect();
    this.ownerMo.disconnect();
    this.ro.disconnect();
    for (const el of [...this.items.keys()]) this.remove(el);
  }
}
