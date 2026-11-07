/** Global material parameters shared by every glass shape. */
export interface Material {
  /** Refraction strength, as a multiple of the bevel width. */
  refraction: number;
  /** Chromatic dispersion: 0 = none, 0.3 = strong rainbow rims. */
  dispersion: number;
  /** Index of refraction of the glass. */
  ior: number;
  /** Width of the curved rim in CSS px (the flat top sits inside it). */
  bevel: number;
  /** Default frost of "regular" glass (Gaussian sigma in CSS px). "clear" glass uses a fifth of it. */
  blur: number;
  /** Strength of the light-facing highlight. */
  specular: number;
  /** Strength of the Fresnel glow on steep parts of the rim. */
  fresnel: number;
  /** Saturation boost of the refracted backdrop (1 = unchanged). */
  saturation: number;
  /** Brightness lift of the refracted backdrop. */
  brightness: number;
  /** Contact shadow opacity. */
  shadow: number;
  /** 0..1: how much glass darkens over bright backdrops to keep content readable. */
  legibility: number;
  /** 0..1: how much frosted glass lifts toward grey over dark backdrops. */
  lift: number;
  /** How much glass bends the HTML content inside it, relative to the backdrop (content sits inside the slab). */
  contentRefraction: number;
  /**
   * Accessibility mode: heavy frost, a solid neutral tint and gentle
   * refraction. Follows the OS `prefers-reduced-transparency` setting by default.
   */
  reduceTransparency: boolean;
  /** Direction the light comes from, in degrees (0 = right, -90 = top). */
  lightAngle: number;
  /** Rotate the light toward the pointer. */
  lightFollowsPointer: boolean;
  /** Multiplier applied to the backdrop (dims or brightens the wallpaper). */
  exposure: number;
  /** Tint applied to every shape that doesn't set its own: [r, g, b, strength] 0..1. */
  tint: RGBA;
}

export type RGBA = [number, number, number, number];

export type BackgroundSpec =
  | { type: 'procedural'; preset?: number; animate?: boolean }
  | { type: 'image'; source: HTMLImageElement | HTMLCanvasElement | ImageBitmap }
  | { type: 'video'; source: HTMLVideoElement };

export interface RectLike {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Per-element options. Lengths are CSS px. */
export interface GlassOptions {
  /**
   * "regular" (default): frosted, lifts to grey over dark backdrops — panels,
   * bars, widgets, menus. "clear": barely frosted, no lift — lenses, thumbs and
   * controls floating over rich content.
   */
  variant?: 'regular' | 'clear';
  /** Chromatic dispersion multiplier (1 = material default). */
  dispersion?: number;
  /** Corner radius; defaults to the element's computed border-radius. */
  radius?: number;
  /** Frost amount (Gaussian sigma); defaults to material.blur. */
  blur?: number;
  /** CSS color or [r, g, b, strength]; the alpha is used as tint strength. */
  tint?: string | RGBA | null;
  /** Refraction multiplier (1 = material default). */
  refraction?: number;
  /** Rim width; defaults to material.bevel. Set it to the radius for a dome lens. */
  bevel?: number;
  /** Shapes that share a group melt together when close. */
  group?: string;
  /** Smoothness of the melt in CSS px (the group uses its largest value). */
  merge?: number;
  /** Stacking order; higher draws on top. Ties keep registration order. */
  z?: number;
  /** 0..1 extra highlight, e.g. while pressed. */
  glow?: number;
  /** 0..1 opacity of the glass (multiplied with the element's CSS opacity). */
  opacity?: number;
  /** Set `data-glass-tone="light|dark"` from the brightness behind the shape. Default true. */
  adaptive?: boolean;
  /**
   * 0 = base layer (refracts the backdrop). 1 = floating layer: drawn in a
   * second pass that refracts the backdrop *and* the base-layer glass, for
   * lenses, slider thumbs and selection pills that float over other glass.
   */
  layer?: 0 | 1;
  /** Custom geometry provider (viewport CSS px) instead of the element's bounding box. */
  rect?: () => RectLike;
}

export const DEFAULT_MATERIAL: Material = {
  refraction: 0.77,
  dispersion: 0.185,
  ior: 1,
  bevel: 6,
  blur: 0,
  specular: 0.33,
  fresnel: 0.78,
  saturation: 1.35,
  brightness: 0.07,
  shadow: 0.13,
  legibility: 0.83,
  lift: 0.17,
  contentRefraction: 0.85,
  reduceTransparency: false,
  lightAngle: -180,
  lightFollowsPointer: false,
  exposure: 1,
  tint: [1, 1, 1, 0],
};
