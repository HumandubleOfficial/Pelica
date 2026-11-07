import './glass.css';
import { parseColor, luma } from './color';
import { ContentManager, type ContentHandle, type ContentOptions } from './content';
import { CSSFallback } from './fallback';
import { LiveLayers, liveSupported, type LiveLayer, type LiveLayerOptions } from './live';
import { GLRenderer, type LumaMap } from './renderer';
import { DATA_H, DATA_W, effectiveTint, liftOf, packShapes, type Entry, type Placed } from './shapes';
import { DEFAULT_MATERIAL, type BackgroundSpec, type GlassOptions, type Material } from './types';

export { Spring } from './spring';
export type { SpringConfig } from './spring';
export type { ContentHandle, ContentOptions } from './content';
export type { LiveLayer, LiveLayerOptions } from './live';
export { DEFAULT_MATERIAL } from './types';
export type { BackgroundSpec, GlassOptions, Material, RGBA, RectLike } from './types';

export interface SceneOptions {
  /** Canvas to draw into. One is created and prepended to <body> if omitted. */
  canvas?: HTMLCanvasElement;
  background?: BackgroundSpec;
  /** A canvas composited over the background (and refracted by the glass), e.g. a clock. */
  overlay?: HTMLCanvasElement | null;
  material?: Partial<Material>;
  /** Canvas resolution multiplier; defaults to min(devicePixelRatio, 2). */
  renderScale?: number;
  /** Force the CSS fallback even when WebGL2 works. */
  forceFallback?: boolean;
  /** Where to put a created canvas (default: the start of <body>). */
  parent?: HTMLElement;
}

export interface GlassHandle {
  readonly el: HTMLElement;
  readonly options: Readonly<GlassOptions>;
  /** Merge new options in (cheap; safe to call every frame). */
  set(opts: Partial<GlassOptions>): void;
  remove(): void;
}

export type FrameCallback = (dt: number, time: number) => void;

export class GlassScene {
  readonly mode: 'webgl' | 'css';
  readonly canvas: HTMLCanvasElement;
  private gl: GLRenderer | null = null;
  private css: CSSFallback | null = null;
  private content: ContentManager | null = null;
  private live: LiveLayers | null = null;
  private pendingDraw = false;
  private now = 0;
  private entries = new Map<HTMLElement, Entry>();
  private order = 0;
  private material: Material;
  private callbacks = new Set<FrameCallback>();
  private data = new Float32Array(DATA_W * DATA_H * 4);
  private data1 = new Float32Array(DATA_W * DATA_H * 4);
  private raf = 0;
  private last = 0;
  private time = 0;
  private frame = 0;
  private renderScale?: number;
  private baseLight: number;
  private pointer: [number, number] | null = null;
  private placed: Placed[] = [];
  private cssDirtyMaterial = false;
  /** Last drawn shape data: a frame where nothing moved isn't drawn again (in Pelica the GPU belongs to the model). */
  private prev = new Float32Array(DATA_W * DATA_H * 4);
  private prev1 = new Float32Array(DATA_W * DATA_H * 4);
  private force = true;
  /** Frames actually drawn (for tests). */
  drawn = 0;
  private onResize = () => this.resize();
  private onPointer = (e: PointerEvent) => (this.pointer = [e.clientX, e.clientY]);
  /** Rolling average frame time in ms. */
  frameMs = 16;

  constructor(opts: SceneOptions = {}) {
    this.material = { ...DEFAULT_MATERIAL, ...opts.material };
    this.baseLight = this.material.lightAngle;
    // Follow the OS accessibility setting unless the caller decided explicitly.
    if (opts.material?.reduceTransparency === undefined && typeof matchMedia === 'function') {
      const mq = matchMedia('(prefers-reduced-transparency: reduce)');
      this.material.reduceTransparency = mq.matches;
      mq.addEventListener?.('change', (ev) => this.setMaterial({ reduceTransparency: ev.matches }));
    }
    this.renderScale = opts.renderScale;
    this.canvas = opts.canvas ?? document.createElement('canvas');
    this.canvas.classList.add('lg-canvas');
    if (!this.canvas.isConnected) (opts.parent ?? document.body).prepend(this.canvas);

    let mode: 'webgl' | 'css' = 'css';
    if (!opts.forceFallback) {
      try {
        this.gl = new GLRenderer(this.canvas);
        mode = 'webgl';
      } catch (err) {
        console.warn('[liquid-glass] falling back to CSS:', err);
      }
    }
    if (this.gl?.supportsContent) this.content = new ContentManager(this.gl, (el) => this.ownerOf(el));
    if (!this.gl) {
      this.css = new CSSFallback(this.canvas);
      this.css.material = this.material;
    }
    this.mode = mode;
    document.documentElement.dataset.glassMode = mode;
    if (opts.background) this.setBackground(opts.background);
    if (opts.overlay) this.setOverlay(opts.overlay);
    this.resize();
    addEventListener('resize', this.onResize);
    addEventListener('pointermove', this.onPointer, { passive: true });
    this.raf = requestAnimationFrame(this.tick);
  }

  /** Register an element as glass. Calling again for the same element updates its options. */
  add(el: HTMLElement, opts: GlassOptions = {}): GlassHandle {
    let e = this.entries.get(el);
    if (e) {
      Object.assign(e.opts, opts);
      e.tint = parseColor(e.opts.tint);
    } else {
      e = { el, opts: { ...opts }, order: this.order++, tint: parseColor(opts.tint) };
      this.entries.set(el, e);
      el.dataset.glass ??= '';
      this.css?.add(e);
    }
    const entry = e;
    return {
      el,
      options: entry.opts,
      set: (o) => {
        const tintChanged = 'tint' in o && o.tint !== entry.opts.tint;
        Object.assign(entry.opts, o);
        if (tintChanged) entry.tint = parseColor(entry.opts.tint);
        this.css?.style(entry);
      },
      remove: () => this.remove(el),
    };
  }

  remove(el: HTMLElement) {
    const e = this.entries.get(el);
    if (!e) return;
    this.css?.remove(e);
    this.entries.delete(el);
    delete el.dataset.glassTone;
  }

  /**
   * Register every `[data-glass]` element under `root`. Options can be given as
   * attributes: data-glass-blur, -tint, -radius, -bevel, -refraction, -group, -merge, -z.
   */
  attach(root: ParentNode = document): GlassHandle[] {
    const out: GlassHandle[] = [];
    root.querySelectorAll<HTMLElement>('[data-glass]').forEach((el) => {
      const d = el.dataset;
      const num = (v?: string) => (v == null || v === '' ? undefined : Number(v));
      const o: GlassOptions = {};
      if (d.glassBlur) o.blur = num(d.glassBlur);
      if (d.glassTint) o.tint = d.glassTint;
      if (d.glassRadius) o.radius = num(d.glassRadius);
      if (d.glassBevel) o.bevel = num(d.glassBevel);
      if (d.glassRefraction) o.refraction = num(d.glassRefraction);
      if (d.glassGroup) o.group = d.glassGroup;
      if (d.glassMerge) o.merge = num(d.glassMerge);
      if (d.glassZ) o.z = num(d.glassZ);
      out.push(this.add(el, o));
    });
    return out;
  }

  /**
   * Draw an HTML element *through* the glass it sits in instead of on top of
   * it: it refracts and fringes with the glass, smears under floating lenses,
   * and hides under higher glass. The element is captured to a texture
   * (re-captured when it changes) and stays in the page, invisible, for input
   * and accessibility. Mark leaf-ish pieces (icons, labels, rows) rather than
   * whole panels, and never an element that contains glass. No-op in the CSS
   * fallback, where content simply stays HTML.
   */
  addContent(el: HTMLElement | SVGElement, opts?: ContentOptions): ContentHandle {
    // Live layers already draw everything in them through the glass.
    if (this.content && !this.live?.contains(el)) return this.content.add(el, opts);
    return { el, refresh() {}, remove() {} };
  }

  /** `addContent` for every `[data-glass-content]` element under `root`. */
  attachContent(root: ParentNode = document): ContentHandle[] {
    return [...root.querySelectorAll<HTMLElement>('[data-glass-content]')].map((el) => this.addContent(el));
  }

  /** Whether HTML content is drawn through the glass (WebGL2 + float render targets). */
  get contentSupported() {
    return !!this.content;
  }

  /**
   * Experimental: a viewport-size container whose contents are drawn through
   * the glass *live*, straight from the browser's own rendering (inputs,
   * carets, video, web fonts included), using Chromium's HTML-in-canvas API.
   * Returns null where that API isn't available (anything but Electron 45+ /
   * Chromium started with `--enable-blink-features=CanvasDrawElement`); use
   * `addContent` snapshots there. See live.ts.
   */
  liveLayer(opts?: LiveLayerOptions): LiveLayer | null {
    if (!this.liveSupported || !this.gl) return null;
    if (!this.live) {
      this.live = new LiveLayers(this.gl, this.canvas);
      this.live.onPaint = () => {
        if (!this.pendingDraw) return;
        this.pendingDraw = false;
        this.draw();
      };
    }
    return this.live.add(opts);
  }

  /** Whether `liveLayer` works here. */
  get liveSupported() {
    return !!this.gl?.supportsContent && liveSupported();
  }

  private ownerOf(el: Element): Entry | null {
    for (let n = el.parentElement; n; n = n.parentElement) {
      const e = this.entries.get(n);
      if (e) return e;
    }
    return null;
  }

  getMaterial(): Readonly<Material> {
    return this.material;
  }

  setMaterial(m: Partial<Material>) {
    Object.assign(this.material, m);
    if ('lightAngle' in m) this.baseLight = m.lightAngle!;
    this.force = true;
    if ('exposure' in m) {
      this.gl?.invalidate();
      this.css?.invalidate();
    }
    if (this.css) this.cssDirtyMaterial = true;
  }

  setBackground(spec: BackgroundSpec) {
    this.force = true;
    this.gl?.setBackground(spec);
    this.css?.setBackground(spec);
  }

  setOverlay(c: HTMLCanvasElement | null) {
    this.gl?.setOverlay(c);
    this.css?.setOverlay(c);
  }

  /** Call after drawing into the overlay canvas. */
  invalidateOverlay() {
    this.gl?.invalidateOverlay();
    this.css?.invalidate();
  }

  /** Runs every frame before glass geometry is read; move your DOM here. */
  onFrame(cb: FrameCallback): () => void {
    this.callbacks.add(cb);
    return () => this.callbacks.delete(cb);
  }

  private resize() {
    const w = innerWidth;
    const h = innerHeight;
    const s = this.renderScale ?? Math.min(devicePixelRatio || 1, 2);
    this.force = true;
    this.gl?.resize(w, h, s);
    this.css?.resize(w, h, Math.min(s, 1.5));
  }

  private tick = (now: number) => {
    this.raf = requestAnimationFrame(this.tick);
    // With live layers the frame is drawn from the canvas `paint` event, after
    // layout and the browser's own paint, so glass and content match. If that
    // event didn't come last frame, draw now rather than drop the frame.
    if (this.pendingDraw) {
      this.pendingDraw = false;
      this.draw();
    }
    this.now = now;
    const dt = this.last ? Math.min((now - this.last) / 1000, 0.1) : 1 / 60;
    this.last = now;
    this.time += dt;
    this.frameMs += (dt * 1000 - this.frameMs) * 0.05;
    this.frame++;
    for (const cb of this.callbacks) cb(dt, this.time);

    const m = this.material;
    if (m.lightFollowsPointer && this.pointer) {
      const target = (Math.atan2(this.pointer[1] - innerHeight * 0.5, this.pointer[0] - innerWidth * 0.5) * 180) / Math.PI;
      const diff =((target - m.lightAngle + 540) % 360) - 180;
      m.lightAngle += diff * Math.min(1, dt * 6);
    } else if (!m.lightFollowsPointer && m.lightAngle !== this.baseLight) {
      m.lightAngle = this.baseLight;
    }

    if (this.gl && this.live?.size) {
      this.pendingDraw = true;
      this.live.requestPaint();
    } else this.draw();
  };

  private draw() {
    const m = this.material;
    const now = this.now;
    if (this.gl) {
      const scale = this.gl.scale;
      const base = packShapes(this.entries.values(), scale, m, innerWidth, innerHeight, this.data, 0);
      const float = packShapes(this.entries.values(), scale, m, innerWidth, innerHeight, this.data1, 1);
      this.placed = base.placed.concat(float.placed);
      this.content?.update(now);
      const q = this.content?.quads(scale) ?? { base: [], floating: [] };
      const same = (a: Float32Array, b: Float32Array) => {
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
        return true;
      };
      if (!this.force && !this.gl.animating && !q.base.length && !q.floating.length && !this.live?.size && same(base.data, this.prev) && same(float.data, this.prev1)) return;
      this.force = false;
      this.prev.set(base.data);
      this.prev1.set(float.data);
      this.drawn++;
      if (this.live?.size) {
        this.live.upload();
        const lq = this.live.quads();
        q.base = q.base.concat(lq.base);
        q.floating = q.floating.concat(lq.floating);
      }
      this.gl.render(
        [
          { data: base.data, groups: base.groups, content: q.base },
          { data: float.data, groups: float.groups, content: q.floating },
        ],
        this.time,
        m,
      );
      if (this.drawn % 12 === 1) this.updateTones(this.gl.readLuma(), true);
    } else if (this.css) {
      if (this.cssDirtyMaterial) {
        this.css.restyleAll();
        this.cssDirtyMaterial = false;
      }
      this.css.render();
      if (this.frame % 20 === 0) {
        this.placed = [...this.entries.values()].map((entry) => ({
          entry,
          rect: entry.el.getBoundingClientRect(),
          tint: effectiveTint((entry.opts.tint === undefined ? m.tint : entry.tint) ?? [1, 1, 1, 0], m),
        }));
        this.updateTones(this.css.readLuma(), false);
      }
    }
  }

  /** Mark each shape light or dark from the backdrop brightness behind it. */
  private updateTones(map: LumaMap, flipY: boolean) {
    const { data, w, h } = map;
    const vw = innerWidth;
    const vh = innerHeight;
    for (const p of this.placed) {
      const e = p.entry;
      if (e.opts.adaptive === false) continue;
      let sum = 0;
      let n = 0;
      for (let j = 0; j < 4; j++) {
        for (let i = 0; i < 4; i++) {
          const x = (p.rect.x + ((i + 0.5) / 4) * p.rect.width) / vw;
          const y = (p.rect.y + ((j + 0.5) / 4) * p.rect.height) / vh;
          if (x < 0 || x > 1 || y < 0 || y > 1) continue;
          const tx = Math.min(w - 1, Math.floor(x * w));
          const ty = Math.min(h - 1, Math.floor((flipY ? 1 - y : y) * h));
          const k = (ty * w + tx) * 4;
          sum += luma([data[k] / 255, data[k + 1] / 255, data[k + 2] / 255]);
          n++;
        }
      }
      if (!n) continue;
      let l = sum / n;
      if (this.gl) {
        // Mirror the shader's legibility darkening and dark-backdrop lift so
        // the text tone matches what's drawn.
        const ss = (a: number, b: number, x: number) => {
          const s = Math.min(1, Math.max(0, (x - a) / (b - a)));
          return s * s * (3 - 2 * s);
        };
        const open = 1 - p.tint[3];
        l *= 1 - 0.38 * this.material.legibility * ss(0.45, 0.95, l) * open;
        l += (1 - l) * this.material.lift * liftOf(e.opts) * (1 - ss(0, 0.4, l)) * open;
      }
      l = l * (1 - p.tint[3]) + luma(p.tint) * p.tint[3];
      // iOS keeps white glyphs on glass that darkened over white (e.g. Safari's
      // toolbar), so only very bright glass flips to dark text.
      const tone = e.tone === 'light' ? (l < 0.64 ? 'dark' : 'light') : l > 0.76 ? 'light' : 'dark';
      if (tone !== e.tone) {
        e.tone = tone;
        e.el.dataset.glassTone = tone;
      }
    }
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    removeEventListener('resize', this.onResize);
    removeEventListener('pointermove', this.onPointer);
    this.entries.forEach((_, el) => this.remove(el));
    this.content?.destroy();
    this.live?.destroy();
    this.gl?.destroy();
    this.css?.destroy();
    delete document.documentElement.dataset.glassMode;
  }
}

export function createGlassScene(opts?: SceneOptions) {
  return new GlassScene(opts);
}
