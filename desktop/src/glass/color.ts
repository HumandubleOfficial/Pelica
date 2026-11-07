import type { RGBA } from './types';

let ctx: CanvasRenderingContext2D | null = null;
const cache = new Map<string, RGBA>();

/** Parse any CSS color into [r, g, b, a] in 0..1. Alpha becomes tint strength. */
export function parseColor(input: string | RGBA | null | undefined): RGBA | null {
  if (input == null) return null;
  if (Array.isArray(input)) return input;
  const hit = cache.get(input);
  if (hit) return hit;
  ctx ??= document.createElement('canvas').getContext('2d');
  if (!ctx) return null;
  ctx.fillStyle = '#000';
  ctx.fillStyle = input;
  const s = ctx.fillStyle as string;
  let out: RGBA;
  if (s.startsWith('#')) {
    const n = parseInt(s.slice(1), 16);
    out = [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, 1];
  } else {
    const m = s.match(/[\d.]+/g)?.map(Number) ?? [0, 0, 0, 1];
    out = [m[0] / 255, m[1] / 255, m[2] / 255, m[3] ?? 1];
  }
  cache.set(input, out);
  return out;
}

export function luma(c: RGBA | [number, number, number]): number {
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}
