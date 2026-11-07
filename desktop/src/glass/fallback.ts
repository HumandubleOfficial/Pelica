import { effectiveBlur, effectiveTint, liftOf, type Entry } from './shapes';
import type { BackgroundSpec, Material } from './types';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Chromium can run SVG filters inside backdrop-filter, which gives real (approximate) edge refraction. */
const supportsBackdropUrl = (() => {
  const ua = navigator.userAgent;
  return /Chrome\/|Chromium\/|Edg\//.test(ua) && !/Firefox\//.test(ua);
})();

/**
 * CSS-only material for when WebGL2 isn't available. The backdrop is a 2D
 * canvas behind the page; each glass element gets a ::before layer with
 * backdrop-filter, highlights and a shadow (see glass.css).
 */
export class CSSFallback {
  readonly ctx: CanvasRenderingContext2D;
  private bg: BackgroundSpec = { type: 'procedural', preset: 0 };
  private overlay: HTMLCanvasElement | null = null;
  private svg: SVGSVGElement;
  private filters = new WeakMap<HTMLElement, { filter: SVGFilterElement; image: SVGFEImageElement; disp: SVGFEDisplacementMapElement; key: string }>();
  private ro: ResizeObserver;
  private seq = 0;
  private dirty = true;
  private lumaCanvas = document.createElement('canvas');
  material!: Material;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!;
    this.svg = document.createElementNS(SVG_NS, 'svg');
    this.svg.setAttribute('aria-hidden', 'true');
    this.svg.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden';
    document.body.appendChild(this.svg);
    this.ro = new ResizeObserver((items) => items.forEach((i) => this.updateFilter(i.target as HTMLElement)));
    document.documentElement.classList.add('lg-css-mode');
  }

  private entries = new Map<HTMLElement, Entry>();

  add(e: Entry) {
    this.entries.set(e.el, e);
    e.el.classList.add('lg-css');
    if (getComputedStyle(e.el).position === 'static') e.el.style.position = 'relative';
    this.style(e);
    if (supportsBackdropUrl) {
      this.ro.observe(e.el);
      // A parent that just gained glass inside it loses its SVG refraction (see style()).
      const parent = e.el.parentElement?.closest<HTMLElement>('.lg-css');
      const pe = parent && this.entries.get(parent);
      if (pe) this.style(pe);
    }
  }

  remove(e: Entry) {
    this.entries.delete(e.el);
    e.el.classList.remove('lg-css');
    this.ro.unobserve(e.el);
    this.filters.get(e.el)?.filter.remove();
    this.filters.delete(e.el);
    for (const p of ['--lg-backdrop', '--lg-tint', '--lg-opacity', '--lg-glow']) e.el.style.removeProperty(p);
  }

  style(e: Entry) {
    const m = this.material;
    const blur = Math.max(effectiveBlur(e.opts, m), 0) * 1.4 + 2;
    const t = effectiveTint((e.opts.tint === undefined ? m.tint : e.tint) ?? [1, 1, 1, 0], m);
    const f = this.filters.get(e.el);
    // Chromium has two bugs with SVG url() inside backdrop-filter: nested ones
    // blow out colors, and the element's tint layer can stay painted beneath
    // the filtered backdrop. So only untinted leaf glass gets displacement
    // refraction (tinted glass is mostly opaque color anyway).
    const leaf = !e.el.querySelector('.lg-css');
    const tinted = !!t && t[3] > 0.04;
    const url = f && leaf && !tinted && (e.opts.refraction ?? 1) > 0 ? `url(#${f.filter.id}) ` : '';
    // Nested glass already sees a saturated backdrop; don't compound it.
    const nested = !!e.el.parentElement?.closest('.lg-css');
    const sat = nested ? 1 : m.saturation * 1.25;
    e.el.style.setProperty('--lg-backdrop', `${url}blur(${blur.toFixed(1)}px) saturate(${sat.toFixed(2)}) brightness(${(1 + (nested ? 0 : m.brightness)).toFixed(2)})`);
    e.el.style.setProperty('--lg-tint', t ? `rgba(${t[0] * 255 | 0},${t[1] * 255 | 0},${t[2] * 255 | 0},${(t[3] * 0.9).toFixed(3)})` : 'transparent');
    e.el.style.setProperty('--lg-lift', `rgba(255,255,255,${(m.lift * liftOf(e.opts) * 0.55).toFixed(3)})`);
    e.el.style.setProperty('--lg-opacity', String(e.opts.opacity ?? 1));
    e.el.style.setProperty('--lg-glow', String(e.opts.glow ?? 0));
    e.el.style.setProperty('--lg-shadow', String(m.shadow));
  }

  restyleAll() {
    this.entries.forEach((e) => {
      if (supportsBackdropUrl) this.updateFilter(e.el);
      this.style(e);
    });
  }

  /** Build a displacement map that pushes samples inward along a convex rim. */
  private updateFilter(el: HTMLElement) {
    const e = this.entries.get(el);
    if (!e) return;
    const w = el.offsetWidth;
    const h = el.offsetHeight;
    if (w < 2 || h < 2) return;
    const m = this.material;
    const raw = getComputedStyle(el).borderTopLeftRadius;
    let r = parseFloat(raw) || 0;
    if (raw.endsWith('%')) r = (r / 100) * Math.min(w, h);
    r = Math.min(r, Math.min(w, h) / 2);
    const bevel = Math.min(e.opts.bevel ?? m.bevel, Math.min(w, h) / 2);
    const strength = m.refraction * (e.opts.refraction ?? 1);
    const key = `${w}x${h}:${r.toFixed(1)}:${bevel.toFixed(1)}:${strength.toFixed(2)}`;
    let f = this.filters.get(el);
    if (f?.key === key) return;

    const ds = Math.max(1, Math.ceil(Math.max(w, h) / 320));
    const mw = Math.ceil(w / ds);
    const mh = Math.ceil(h / ds);
    const maxOff = bevel * strength * 1.4 + 1;
    const c = document.createElement('canvas');
    c.width = mw;
    c.height = mh;
    const cx = c.getContext('2d')!;
    const img = cx.createImageData(mw, mh);
    for (let y = 0; y < mh; y++) {
      for (let x = 0; x < mw; x++) {
        const px = (x + 0.5) * ds - w / 2;
        const py = (y + 0.5) * ds - h / 2;
        const qx = Math.abs(px) - w / 2 + r;
        const qy = Math.abs(py) - h / 2 + r;
        const mx = Math.max(qx, 0);
        const my = Math.max(qy, 0);
        const d = Math.hypot(mx, my) + Math.min(Math.max(qx, qy), 0) - r;
        let gx: number, gy: number;
        if (mx > 0 || my > 0) {
          const l = Math.hypot(mx, my) || 1;
          gx = (mx / l) * Math.sign(px);
          gy = (my / l) * Math.sign(py);
        } else if (qx > qy) {
          gx = Math.sign(px);
          gy = 0;
        } else {
          gx = 0;
          gy = Math.sign(py);
        }
        const t = Math.min(Math.max(-d / Math.max(bevel, 1), 0), 1);
        const u = 1 - t;
        const off = Math.min(u / Math.sqrt(Math.max(1 - u * u, 0.03)), 5) / 5;
        const mag = off * maxOff;
        const i = (y * mw + x) * 4;
        img.data[i] = Math.round(128 + (-gx * mag / (2 * maxOff)) * 255);
        img.data[i + 1] = Math.round(128 + (-gy * mag / (2 * maxOff)) * 255);
        img.data[i + 2] = 128;
        img.data[i + 3] = 255;
      }
    }
    cx.putImageData(img, 0, 0);
    if (!f) {
      const filter = document.createElementNS(SVG_NS, 'filter');
      filter.id = `lg-f-${++this.seq}`;
      filter.setAttribute('color-interpolation-filters', 'sRGB');
      filter.setAttribute('x', '0');
      filter.setAttribute('y', '0');
      filter.setAttribute('width', '100%');
      filter.setAttribute('height', '100%');
      const image = document.createElementNS(SVG_NS, 'feImage');
      image.setAttribute('preserveAspectRatio', 'none');
      image.setAttribute('result', 'map');
      const disp = document.createElementNS(SVG_NS, 'feDisplacementMap');
      disp.setAttribute('in', 'SourceGraphic');
      disp.setAttribute('in2', 'map');
      disp.setAttribute('xChannelSelector', 'R');
      disp.setAttribute('yChannelSelector', 'G');
      filter.append(image, disp);
      this.svg.appendChild(filter);
      f = { filter, image, disp, key };
      this.filters.set(el, f);
    }
    f.key = key;
    f.image.setAttribute('href', c.toDataURL());
    f.image.setAttribute('x', '0');
    f.image.setAttribute('y', '0');
    f.image.setAttribute('width', String(w));
    f.image.setAttribute('height', String(h));
    f.disp.setAttribute('scale', (2 * maxOff).toFixed(1));
    this.style(e);
  }

  setBackground(spec: BackgroundSpec) {
    this.bg = spec;
    this.dirty = true;
  }

  setOverlay(c: HTMLCanvasElement | null) {
    this.overlay = c;
    this.dirty = true;
  }

  invalidate() {
    this.dirty = true;
  }

  resize(cssW: number, cssH: number, scale: number) {
    const w = Math.round(cssW * scale);
    const h = Math.round(cssH * scale);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.dirty = true;
    }
  }

  /** Repaint the backdrop when needed (every frame for video). */
  render() {
    if (!this.dirty && this.bg.type !== 'video') return false;
    this.dirty = false;
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;
    ctx.save();
    ctx.clearRect(0, 0, w, h);
    if (this.bg.type === 'procedural') paintWallpaper(ctx, w, h, this.bg.preset ?? 0);
    else {
      const src = this.bg.source as HTMLImageElement & HTMLVideoElement;
      const iw = src.videoWidth || src.naturalWidth || (src as unknown as HTMLCanvasElement).width || 1;
      const ih = src.videoHeight || src.naturalHeight || (src as unknown as HTMLCanvasElement).height || 1;
      const s = Math.max(w / iw, h / ih);
      try {
        ctx.drawImage(src, (w - iw * s) / 2, (h - ih * s) / 2, iw * s, ih * s);
      } catch {
        /* not decoded yet */
      }
    }
    if (this.overlay) ctx.drawImage(this.overlay, 0, 0, w, h);
    ctx.restore();
    this.canvas.style.filter = this.material.exposure !== 1 ? `brightness(${this.material.exposure})` : '';
    return true;
  }

  readLuma() {
    const c = this.lumaCanvas;
    c.width = 96;
    c.height = Math.max(1, Math.round((96 * this.canvas.height) / Math.max(this.canvas.width, 1)));
    const cx = c.getContext('2d', { willReadFrequently: true })!;
    cx.drawImage(this.canvas, 0, 0, c.width, c.height);
    const d = cx.getImageData(0, 0, c.width, c.height);
    return { data: d.data as unknown as Uint8Array, w: c.width, h: c.height };
  }

  destroy() {
    this.ro.disconnect();
    this.svg.remove();
    this.entries.forEach((e) => this.remove(e));
    document.documentElement.classList.remove('lg-css-mode');
  }
}

/** 2D approximations of the procedural shader presets. */
export function paintWallpaper(ctx: CanvasRenderingContext2D, w: number, h: number, preset: number) {
  const u = h;
  if (preset === 1) {
    const sky = ctx.createLinearGradient(0, 0, 0, h);
    sky.addColorStop(0, '#6a4aae');
    sky.addColorStop(0.55, '#ff9e61');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);
    const sx = w * 0.68;
    const sy = h * 0.28;
    const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, u * 0.35);
    glow.addColorStop(0, 'rgba(255,190,120,.6)');
    glow.addColorStop(1, 'rgba(255,190,120,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#fff0c8';
    ctx.beginPath();
    ctx.arc(sx, sy, u * 0.11, 0, Math.PI * 2);
    ctx.fill();
    for (let i = 0; i < 7; i++) {
      const base = (0.62 - i * 0.085) * u;
      const k = i / 6;
      ctx.fillStyle = `rgb(${245 - 204 * k},${117 - 99 * k},${92 - 26 * k})`;
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let x = 0; x <= w; x += 8) {
        const p = x / u;
        const y = base + u * (0.05 * Math.sin(p * (2.2 + i * 0.6) + i * 1.7) + 0.03 * Math.sin(p * 3 + i) + 0.02 * Math.sin(p * 7.3 + i * 4));
        ctx.lineTo(x, h - y);
      }
      ctx.lineTo(w, h);
      ctx.fill();
    }
    return;
  }
  if (preset === 4) {
    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, '#010205');
    bg.addColorStop(1, '#050812');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
    const glow = ctx.createRadialGradient(w * 0.42, h * 0.82, 0, w * 0.42, h * 0.82, u * 0.5);
    glow.addColorStop(0, 'rgba(92,52,30,.9)');
    glow.addColorStop(1, 'rgba(92,52,30,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, 0, w, h);
    const boxes = [[0.3, 0.2, 0.26, 0.13], [0.78, 0.45, 0.2, 0.1], [0.18, 0.7, 0.14, 0.12], [0.62, 0.84, 0.24, 0.08], [0.95, 0.12, 0.18, 0.09]];
    ctx.strokeStyle = 'rgba(225,232,255,.9)';
    ctx.shadowColor = 'rgba(120,150,255,.9)';
    ctx.shadowBlur = u * 0.02;
    ctx.lineWidth = Math.max(1, u * 0.003);
    for (const [x, y, bw, bh] of boxes) {
      const hw = bw * u * (w / h) * 0.62;
      const hh = bh * u;
      ctx.beginPath();
      ctx.roundRect(x * w - hw, y * h - hh, hw * 2, hh * 2, u * 0.06);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    return;
  }
  if (preset === 2) {
    ctx.fillStyle = '#0d0f17';
    ctx.fillRect(0, 0, w, h);
    const cell = u / 7;
    for (let y = -1; y < h / cell + 1; y++) {
      for (let x = -1; x < w / cell + 1; x++) {
        const hue = ((Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1 + 1) % 1;
        ctx.fillStyle = `hsl(${hue * 360},75%,62%)`;
        const s = cell * 0.8;
        ctx.beginPath();
        ctx.roundRect(x * cell + cell * 0.1, y * cell + cell * 0.1, s, s, cell * 0.1);
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,.55)';
        ctx.lineWidth = cell * 0.04;
        ctx.beginPath();
        ctx.arc(x * cell + cell / 2, y * cell + cell / 2, cell * 0.18, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    return;
  }
  const bg = ctx.createLinearGradient(0, 0, 0, h);
  bg.addColorStop(0, '#290d42');
  bg.addColorStop(1, '#0a0d29');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  const colors = ['#fa6b4d', '#ffbd42', '#5285ff', '#f257ad', '#38d1c7', '#8f5cff'];
  const aspect = w / h;
  for (let i = 0; i < 6; i++) {
    const cx = aspect * (0.12 + 0.76 * ((0.37 * i + 0.11) % 1)) * u;
    const cy = h - (0.12 + 0.76 * ((0.61 * i + 0.3) % 1)) * u;
    const r = (0.21 + 0.07 * Math.sin(i * 3.1)) * u;
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,.45)';
    ctx.shadowBlur = u * 0.06;
    ctx.shadowOffsetX = u * 0.014;
    ctx.shadowOffsetY = u * 0.02;
    const g = ctx.createRadialGradient(cx - r * 0.4, cy - r * 0.4, r * 0.1, cx, cy, r);
    g.addColorStop(0, colors[i]);
    g.addColorStop(1, colors[i] + 'cc');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}
