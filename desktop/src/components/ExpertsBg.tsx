// The "live experts" background: every expert of every layer as a dot (one
// block per layer), lighting up when the model routes tokens to it and
// fading after.
// (For models without experts, or before any arrive, a quiet idle grid.)
// The dim grid is drawn once into its own canvas and copied each frame;
// only lit experts are drawn on top (a glow sprite, no gradients per dot),
// and when nothing is lit the canvas isn't touched at all.
import { useEffect, useRef } from "react";
import { experts, useApp } from "../state";

export function ExpertsBg() {
  const on = useApp((s) => s.settings.background === "experts" || s.settings.theme === "neural");
  const anim = useApp((s) => s.settings.animations);
  const theme = useApp((s) => s.settings.theme);
  const running = useApp((s) => s.running);
  const ref = useRef<HTMLCanvasElement>(null);
  const runRef = useRef(running);
  runRef.current = running;

  useEffect(() => {
    if (!on) return;
    const c = ref.current!;
    const g = c.getContext("2d")!;
    let raf = 0;
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--expert-hot").trim() || "255,140,80";
    const base = css.getPropertyValue("--expert-dim").trim() || "120,110,100";

    // a glow, drawn once and stamped wherever an expert is lit
    const glow = document.createElement("canvas");
    glow.width = glow.height = 64;
    const gg = glow.getContext("2d")!;
    const grd = gg.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, `rgba(${accent},0.85)`);
    grd.addColorStop(0.35, `rgba(${accent},0.35)`);
    grd.addColorStop(1, `rgba(${accent},0)`);
    gg.fillStyle = grd;
    gg.fillRect(0, 0, 64, 64);

    // the dim grid, redrawn only when the size or the model changes
    const grid = document.createElement("canvas");
    let key = "";
    let geo = { gx: 1, bc: 1, rows: 1, ox: 0, oy: 0, cell: 0, r: 1 };
    let wasLit = true;
    const layout = (W: number, H: number, L: number, E: number, dpr: number) => {
      const k = `${W}x${H}@${dpr}/${L}x${E}`;
      if (k === key) return;
      key = k;
      const bc = Math.ceil(Math.sqrt(E));
      const br = Math.ceil(E / bc);
      let gx = 1;
      let cell = 0;
      for (let n = 1; n <= L; n++) {
        const s = Math.min(W / (n * (bc + 1)), H / (Math.ceil(L / n) * (br + 1)));
        if (s > cell) (cell = s), (gx = n);
      }
      const gy = Math.ceil(L / gx);
      geo = { gx, bc, rows: br, ox: (W - gx * (bc + 1) * cell) / 2 + cell, oy: (H - gy * (br + 1) * cell) / 2 + cell, cell, r: Math.max(1, cell * 0.3) };
      grid.width = W * dpr;
      grid.height = H * dpr;
      const q = grid.getContext("2d")!;
      q.setTransform(dpr, 0, 0, dpr, 0, 0);
      q.fillStyle = `rgba(${base},0.1)`;
      q.beginPath();
      for (let l = 0; l < L; l++)
        for (let e = 0; e < E; e++) {
          const [x, y] = at(l, e);
          q.moveTo(x + geo.r, y);
          q.arc(x, y, geo.r, 0, 6.283);
        }
      q.fill();
      wasLit = true;
    };
    const at = (l: number, e: number): [number, number] => {
      const { gx, bc, rows, ox, oy, cell } = geo;
      return [ox + (l % gx) * (bc + 1) * cell + ((e % bc) + 0.5) * cell, oy + Math.floor(l / gx) * (rows + 1) * cell + (Math.floor(e / bc) + 0.5) * cell];
    };

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const dpr = window.devicePixelRatio || 1;
      const W = c.clientWidth;
      const H = c.clientHeight;
      const L = experts.layers || 24;
      const E = experts.experts || 64;
      const live = experts.peak.length === L * E;
      const idleAnim = !live && anim && runRef.current;
      // nothing lit for a while: the grid is already on screen
      const lit = idleAnim || (live && now - experts.last < 4000);
      if (c.width !== W * dpr || c.height !== H * dpr) {
        c.width = W * dpr;
        c.height = H * dpr;
        key = "";
      }
      const k0 = key;
      layout(W, H, L, E, dpr);
      if (!lit && !wasLit && key === k0) return;
      wasLit = lit;
      g.setTransform(1, 0, 0, 1, 0, 0);
      g.clearRect(0, 0, c.width, c.height);
      g.drawImage(grid, 0, 0);
      if (!lit) return;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      const r = geo.r;
      for (let l = 0; l < L; l++)
        for (let e = 0; e < E; e++) {
          const v = live ? experts.heat(l * E + e, now) : Math.max(0, Math.sin(now / 700 + l * 0.7 + e * 1.3) - 0.92) * 6;
          if (v < 0.03) continue;
          const [x, y] = at(l, e);
          const rr = r * (1.5 + v * 3);
          g.globalAlpha = Math.min(1, v);
          g.drawImage(glow, x - rr, y - rr, rr * 2, rr * 2);
        }
      g.globalAlpha = 1;
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [on, anim, theme]);

  if (!on) return null;
  return <canvas ref={ref} className="experts-bg" />;
}
