// Photos for a model that sees them (Qwen3.8 with its image encoder): resized
// the way Qwen's processor does it (sides in multiples of 32, at most 768 x 768
// pixels, the shape kept) and sent as raw RGB, so the engine needs no image
// decoder. Kept in memory per photo, since every turn sends them again.

export interface Pixels {
  w: number;
  h: number;
  /** RGB bytes, base64 */
  rgb: string;
}

const MAX = 768 * 768;
const MIN = 8 * 32 * 32;

/** Qwen's smart_resize (and the engine's q4x_vision::target_size). */
export function targetSize(w: number, h: number): [number, number] {
  const f = 32;
  let hb = Math.max(f, Math.round(h / f) * f);
  let wb = Math.max(f, Math.round(w / f) * f);
  if (hb * wb > MAX) {
    const beta = Math.sqrt((h * w) / MAX);
    hb = Math.max(f, Math.floor(h / beta / f) * f);
    wb = Math.max(f, Math.floor(w / beta / f) * f);
  } else if (hb * wb < MIN) {
    const beta = Math.sqrt(MIN / (h * w));
    hb = Math.ceil((h * beta) / f) * f;
    wb = Math.ceil((w * beta) / f) * f;
  }
  return [wb, hb];
}

async function toPixels(dataUrl: string): Promise<Pixels> {
  const blob = await (await fetch(dataUrl)).blob();
  const probe = await createImageBitmap(blob);
  const [w, h] = targetSize(probe.width, probe.height);
  probe.close();
  const bmp = await createImageBitmap(blob, { resizeWidth: w, resizeHeight: h, resizeQuality: "high" });
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff";
  g.fillRect(0, 0, w, h);
  g.drawImage(bmp, 0, 0);
  bmp.close();
  const rgba = g.getImageData(0, 0, w, h).data;
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, o = 0; i < rgba.length; i += 4, o += 3) {
    rgb[o] = rgba[i];
    rgb[o + 1] = rgba[i + 1];
    rgb[o + 2] = rgba[i + 2];
  }
  let bin = "";
  for (let i = 0; i < rgb.length; i += 0x8000) bin += String.fromCharCode(...rgb.subarray(i, i + 0x8000));
  return { w, h, rgb: btoa(bin) };
}

const cache = new Map<string, Pixels>();

/** Gets the photos ready (the conversation's, before a turn is sent). */
export async function preparePixels(urls: string[]) {
  for (const u of urls) {
    if (cache.has(u)) {
      // most recently used last
      const v = cache.get(u)!;
      cache.delete(u);
      cache.set(u, v);
      continue;
    }
    try {
      cache.set(u, await toPixels(u));
    } catch {}
    while (cache.size > 24) cache.delete(cache.keys().next().value!);
  }
}

/** A photo's pixels, once prepared. */
export const pixelsOf = (url: string): Pixels | undefined => cache.get(url);
