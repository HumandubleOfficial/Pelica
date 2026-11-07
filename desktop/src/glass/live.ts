import type { ContentQuad, GLRenderer, LiveTexture } from './renderer';

/**
 * Live HTML through the glass, using Chromium's experimental HTML-in-canvas
 * API (Electron 45+ with `--enable-blink-features=CanvasDrawElement`).
 *
 * A live layer is a full-viewport element placed *inside* the glass canvas,
 * which has `layoutsubtree`: its children are laid out, hit-tested, focusable
 * and accessible like normal DOM, but not painted. Each time one changes,
 * the browser fires a `paint` event on the canvas and we copy its current
 * rendering straight into a texture (GPU side, no snapshot, no decode). That
 * texture is then drawn through the glass's refraction map, like snapshot
 * content, so everything in the layer (text, inputs, carets, focus rings,
 * video) bends and smears with the glass, always current.
 *
 * Direct canvas children are forced to static position, and only direct
 * children can be uploaded, so a layer is a viewport-size box; put your
 * fixed or absolutely positioned UI inside it.
 */

export interface LiveLayerOptions {
  /**
   * Glass depth of the content. Pixels covered by higher glass are hidden, so
   * give each layer the z of the topmost glass it sits on.
   */
  z?: number;
  /** 0 = base glass (default), 1 = floating glass. */
  layer?: 0 | 1;
  opacity?: number;
}

export interface LiveLayer {
  /** Put your UI in here. */
  readonly el: HTMLElement;
  readonly options: Readonly<LiveLayerOptions>;
  set(opts: Partial<LiveLayerOptions>): void;
  remove(): void;
}

type CanvasWithLive = HTMLCanvasElement & {
  layoutSubtree: boolean;
  requestPaint(): void;
  updateElementGeometry(el: Element, opts: { canvasTransform?: DOMMatrix }): void;
};

interface Item {
  el: HTMLElement;
  opts: LiveLayerOptions;
  tex: LiveTexture | null;
  dirty: boolean;
  order: number;
}

/** True when this browser can draw live DOM into WebGL (needs the flag above). */
export function liveSupported(): boolean {
  return (
    typeof HTMLCanvasElement !== 'undefined' &&
    'layoutSubtree' in HTMLCanvasElement.prototype &&
    'requestPaint' in HTMLCanvasElement.prototype &&
    typeof WebGL2RenderingContext !== 'undefined' &&
    'texElementSubImage2D' in WebGL2RenderingContext.prototype
  );
}

export class LiveLayers {
  private items: Item[] = [];
  private order = 0;
  private canvas: CanvasWithLive;
  private geomKey = '';
  /** Set by the scene: draws the frame. Called from the canvas `paint` event. */
  onPaint: (() => void) | null = null;

  constructor(
    private gl: GLRenderer,
    canvas: HTMLCanvasElement,
  ) {
    this.canvas = canvas as CanvasWithLive;
    this.canvas.layoutSubtree = true;
    this.canvas.classList.add('lg-live');
    this.canvas.addEventListener('paint', this.handlePaint);
  }

  get size() {
    return this.items.length;
  }

  add(opts: LiveLayerOptions = {}): LiveLayer {
    const el = document.createElement('div');
    el.className = 'lg-live-layer';
    const it: Item = { el, opts: { ...opts }, tex: null, dirty: true, order: this.order++ };
    this.items.push(it);
    this.canvas.append(el);
    this.geomKey = '';
    return {
      el,
      options: it.opts,
      set: (o) => Object.assign(it.opts, o),
      remove: () => {
        const i = this.items.indexOf(it);
        if (i < 0) return;
        this.items.splice(i, 1);
        if (it.tex) this.gl.deleteContent(it.tex.tex);
        el.remove();
      },
    };
  }

  /** Ask for a `paint` event this frame even if nothing in the layers changed. */
  requestPaint() {
    this.canvas.requestPaint();
  }

  private handlePaint = (e: Event) => {
    const changed = (e as Event & { changedElements?: readonly Element[] }).changedElements;
    for (const it of this.items) if (!changed || changed.includes(it.el)) it.dirty = true;
    this.onPaint?.();
  };

  /** Copy changed layers into their textures. Call while handling `paint`. */
  upload() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    for (const it of this.items) {
      if (!it.dirty && it.tex && it.tex.w === w && it.tex.h === h) continue;
      try {
        it.tex = this.gl.uploadElement(it.tex, it.el, w, h);
        it.dirty = false;
      } catch {
        // No snapshot recorded yet (first frame); the next paint brings one.
      }
    }
    // Hit testing needs to know where each layer is drawn: in place, scaled
    // from CSS px to canvas px.
    const cw = this.canvas.clientWidth || innerWidth;
    const ch = this.canvas.clientHeight || innerHeight;
    const key = `${w}x${h}/${cw}x${ch}/${this.items.length}`;
    if (key !== this.geomKey) {
      this.geomKey = key;
      const m = new DOMMatrix().scale(w / cw, h / ch);
      for (const it of this.items) this.canvas.updateElementGeometry(it.el, { canvasTransform: m });
    }
  }

  quads(): { base: ContentQuad[]; floating: ContentQuad[] } {
    const base: ContentQuad[] = [];
    const floating: ContentQuad[] = [];
    for (const it of this.items) {
      if (!it.tex) continue;
      const { w, h } = it.tex;
      const q: ContentQuad = { tex: it.tex.tex, rect: [0, 0, w, h], geom: [0, 0, w, h], opacity: it.opts.opacity ?? 1, z: it.opts.z ?? 0 };
      (it.opts.layer === 1 ? floating : base).push(q);
    }
    const byZ = (a: ContentQuad, b: ContentQuad) => a.z - b.z;
    return { base: base.sort(byZ), floating: floating.sort(byZ) };
  }

  /** Whether `el` lives in a live layer (and so needs no snapshot). */
  contains(el: Element) {
    return this.items.some((it) => it.el.contains(el));
  }

  destroy() {
    this.canvas.removeEventListener('paint', this.handlePaint);
    for (const it of [...this.items]) {
      if (it.tex) this.gl.deleteContent(it.tex.tex);
      this.canvas.after(...it.el.childNodes);
      it.el.remove();
    }
    this.items = [];
    this.canvas.layoutSubtree = false;
    this.canvas.classList.remove('lg-live');
  }
}
