import { createProgram, createTarget, createTexture, deleteTarget, resizeTarget, type Program, type Target } from './gl';
import { DATA_H, DATA_W } from './shapes';
import type { BackgroundSpec, Material } from './types';
import vert from './shaders/fullscreen.vert?raw';
import sceneFrag from './shaders/scene.frag?raw';
import blurFrag from './shaders/blur.frag?raw';
import glassFrag from './shaders/glass.frag?raw';
import contentVert from './shaders/content.vert?raw';
import contentFrag from './shaders/content.frag?raw';
import presentFrag from './shaders/present.frag?raw';

/** Blur pyramid: each level is this fraction of the CSS-pixel viewport size. */
const LEVELS = [1 / 2, 1 / 4, 1 / 8];
const SPREAD = 1.5;
/** Approximate Gaussian sigma of each level, in CSS px (kernel sigma ~1.5 texels * spread, compounded). */
const LEVEL_SIGMA = [4.5, 10, 21];

export interface LumaMap {
  data: Uint8Array;
  w: number;
  h: number;
}

/** Experimental WebGL entry point of the HTML-in-canvas API. */
interface ElementUpload {
  texElementSubImage2D(
    target: number,
    level: number,
    x: number,
    y: number,
    el: Element,
    config?: { sx?: number; sy?: number; swidth?: number; sheight?: number; width?: number; height?: number },
  ): void;
}

/** Immutable-storage texture holding a live canvas child. */
export interface LiveTexture {
  tex: WebGLTexture;
  w: number;
  h: number;
}

/** A rasterized HTML element placed for this frame (canvas px, y down). */
export interface ContentQuad {
  tex: WebGLTexture;
  rect: [number, number, number, number];
  /** Drawn region: rect grown by the max refraction offset, clipped. x0, y0, x1, y1. */
  geom: [number, number, number, number];
  opacity: number;
  z: number;
}

export interface LayerInput {
  data: Float32Array;
  groups: number;
  content: ContentQuad[];
}

type Pyramid = { a: Target; b: Target }[];

/** Color target plus a refraction-map target sharing one MRT framebuffer. */
interface Comp {
  color: Target;
  off: WebGLTexture;
  mrt: WebGLFramebuffer;
}

export class GLRenderer {
  readonly gl: WebGL2RenderingContext;
  private scene: Program;
  private blur: Program;
  private glass: Program;
  private contentP: Program;
  private present: Program;
  private vao: WebGLVertexArrayObject;
  private sceneT: Target;
  private base: Pyramid;
  private floatPyr: Pyramid;
  private comps: [Comp, Comp];
  private dataTex: WebGLTexture;
  private imageTex: WebGLTexture;
  private overlayTex: WebGLTexture;
  private bg: BackgroundSpec = { type: 'procedural', preset: 0, animate: true };
  private imageSize: [number, number] = [1, 1];
  private imageDirty = false;
  private overlay: HTMLCanvasElement | null = null;
  private overlayDirty = false;
  private sceneDirty = true;
  private cssW = 1;
  private cssH = 1;
  scale = 1;
  private lumaBuf: Uint8Array = new Uint8Array(4);
  lost = false;
  /** Whether refraction maps can be rendered (needs float color buffers). */
  readonly supportsContent: boolean;

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.supportsContent = !!gl.getExtension('EXT_color_buffer_float');
    this.scene = createProgram(gl, vert, sceneFrag, 'scene');
    this.blur = createProgram(gl, vert, blurFrag, 'blur');
    this.glass = createProgram(gl, vert, glassFrag, 'glass');
    this.contentP = createProgram(gl, contentVert, contentFrag, 'content');
    this.present = createProgram(gl, vert, presentFrag, 'present');
    this.vao = gl.createVertexArray()!;
    this.sceneT = createTarget(gl, 1, 1);
    const pyr = () => LEVELS.map(() => ({ a: createTarget(gl, 1, 1), b: createTarget(gl, 1, 1) }));
    this.base = pyr();
    this.floatPyr = pyr();
    this.comps = [this.createComp(), this.createComp()];
    this.dataTex = createTexture(gl, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, DATA_W, DATA_H, 0, gl.RGBA, gl.FLOAT, null);
    this.imageTex = createTexture(gl);
    this.overlayTex = createTexture(gl);
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.lost = true;
    });
  }

  private createComp(): Comp {
    const gl = this.gl;
    return { color: createTarget(gl, 1, 1), off: createTexture(gl, gl.NEAREST), mrt: gl.createFramebuffer()! };
  }

  private resizeComp(c: Comp, w: number, h: number) {
    const gl = this.gl;
    resizeTarget(gl, c.color, w, h);
    gl.bindTexture(gl.TEXTURE_2D, c.off);
    if (this.supportsContent) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, c.mrt);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, c.color.tex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, c.off, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  resize(cssW: number, cssH: number, scale: number) {
    const w = Math.max(1, Math.round(cssW * scale));
    const h = Math.max(1, Math.round(cssH * scale));
    this.cssW = cssW;
    this.cssH = cssH;
    this.scale = w / cssW;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    const gl = this.gl;
    resizeTarget(gl, this.sceneT, w, h);
    for (const p of [this.base, this.floatPyr]) {
      LEVELS.forEach((f, i) => {
        resizeTarget(gl, p[i].a, cssW * f, cssH * f);
        resizeTarget(gl, p[i].b, cssW * f, cssH * f);
      });
    }
    this.comps.forEach((c) => this.resizeComp(c, w, h));
    this.sceneDirty = true;
  }

  setBackground(spec: BackgroundSpec) {
    this.bg = spec;
    if (spec.type !== 'procedural') this.imageDirty = true;
    this.sceneDirty = true;
  }

  setOverlay(canvas: HTMLCanvasElement | null) {
    this.overlay = canvas;
    this.overlayDirty = true;
    this.sceneDirty = true;
  }

  invalidateOverlay() {
    this.overlayDirty = true;
    this.sceneDirty = true;
  }

  invalidate() {
    this.sceneDirty = true;
  }

  private upload(tex: WebGLTexture, src: TexImageSource) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  /** Upload a rasterized HTML element (premultiplied, top row first). */
  uploadContent(tex: WebGLTexture | null, src: TexImageSource): WebGLTexture {
    const gl = this.gl;
    const t = tex ?? createTexture(gl);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    return t;
  }

  /**
   * Upload a child of this canvas as it currently renders (Chromium's
   * HTML-in-canvas API; see live.ts). Premultiplied, top row first, w×h canvas px.
   */
  uploadElement(tex: LiveTexture | null, el: Element, w: number, h: number): LiveTexture {
    const gl = this.gl as WebGL2RenderingContext & ElementUpload;
    let t = tex;
    if (!t || t.w !== w || t.h !== h) {
      if (t) gl.deleteTexture(t.tex);
      const handle = createTexture(gl);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
      t = { tex: handle, w, h };
    }
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texElementSubImage2D(gl.TEXTURE_2D, 0, 0, 0, el, { width: w, height: h });
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    return t;
  }

  deleteContent(tex: WebGLTexture) {
    this.gl.deleteTexture(tex);
  }

  private bindFb(fb: WebGLFramebuffer | null, w: number, h: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
  }

  private draw(p: Program, target: Target | null) {
    const gl = this.gl;
    gl.useProgram(p.program);
    this.bindFb(target ? target.fb : null, target ? target.w : this.canvas.width, target ? target.h : this.canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private bindTex(unit: number, tex: WebGLTexture) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }

  private renderScene(time: number, material: Material) {
    const gl = this.gl;
    const bg = this.bg;
    const u = this.scene.u;
    let preset = 3;
    if (bg.type === 'procedural') preset = bg.preset ?? 0;
    else {
      const src = bg.source;
      const isVideo = bg.type === 'video';
      const ready = !isVideo || (src as HTMLVideoElement).readyState >= 2;
      if (ready && (this.imageDirty || isVideo)) {
        this.upload(this.imageTex, src);
        const w = (src as HTMLVideoElement).videoWidth || (src as HTMLImageElement).naturalWidth || (src as HTMLCanvasElement).width;
        const h = (src as HTMLVideoElement).videoHeight || (src as HTMLImageElement).naturalHeight || (src as HTMLCanvasElement).height;
        this.imageSize = [w || 1, h || 1];
        this.imageDirty = false;
      }
    }
    if (this.overlay && this.overlayDirty) {
      this.upload(this.overlayTex, this.overlay);
      this.overlayDirty = false;
    }
    gl.useProgram(this.scene.program);
    gl.uniform2f(u.u_res, this.sceneT.w, this.sceneT.h);
    gl.uniform1f(u.u_time, time);
    gl.uniform1i(u.u_preset, preset);
    gl.uniform2f(u.u_imageSize, this.imageSize[0], this.imageSize[1]);
    gl.uniform1f(u.u_hasOverlay, this.overlay ? 1 : 0);
    gl.uniform1f(u.u_exposure, material.exposure);
    this.bindTex(0, this.imageTex);
    gl.uniform1i(u.u_image, 0);
    this.bindTex(1, this.overlayTex);
    gl.uniform1i(u.u_overlay, 1);
    this.draw(this.scene, this.sceneT);
  }

  /** Blur pyramid: each level blurs the previous one into a smaller target. */
  private blurInto(src: Target, pyr: Pyramid) {
    const gl = this.gl;
    gl.useProgram(this.blur.program);
    gl.uniform1i(this.blur.u.u_tex, 0);
    for (const lv of pyr) {
      this.bindTex(0, src.tex);
      gl.uniform2f(this.blur.u.u_dir, SPREAD / lv.a.w, 0);
      this.draw(this.blur, lv.a);
      this.bindTex(0, lv.a.tex);
      gl.uniform2f(this.blur.u.u_dir, 0, SPREAD / lv.b.h);
      this.draw(this.blur, lv.b);
      src = lv.b;
    }
  }

  /**
   * Draw the frame:
   *   backdrop → blur → base glass (+ refraction map) → base content through it
   *   → [re-blur → floating glass refracting all of that → floating content] → screen
   */
  render(layers: [LayerInput, LayerInput], time: number, material: Material) {
    if (this.lost) return;
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    const animated = this.bg.type === 'video' || (this.bg.type === 'procedural' && this.bg.animate !== false);
    if (this.sceneDirty || animated) {
      this.renderScene(time, material);
      this.blurInto(this.sceneT, this.base);
      this.sceneDirty = false;
    }

    const [c0, c1] = this.comps;
    this.glassPass(layers[0], material, this.sceneT, this.base, c0);
    this.contentPass(layers[0].content, c0);
    let out = c0;
    const floating = layers[1].groups > 0 || layers[1].content.length > 0;
    if (floating) {
      this.blurInto(c0.color, this.floatPyr);
      // The floating layer looks through glass that already got its vibrancy; don't stack it.
      const soft = { ...material, saturation: 1 + (material.saturation - 1) * 0.35, brightness: material.brightness * 0.4 };
      this.glassPass(layers[1], soft, c0.color, this.floatPyr, c1);
      this.contentPass(layers[1].content, c1);
      out = c1;
    }
    gl.useProgram(this.present.program);
    this.bindTex(0, out.color.tex);
    gl.uniform1i(this.present.u.u_tex, 0);
    this.draw(this.present, null);
  }

  private glassPass(layer: LayerInput, material: Material, backdrop: Target, pyr: Pyramid, out: Comp) {
    const gl = this.gl;
    this.bindTex(4, this.dataTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, DATA_W, DATA_H, gl.RGBA, gl.FLOAT, layer.data);

    const u = this.glass.u;
    gl.useProgram(this.glass.program);
    gl.uniform2f(u.u_res, this.canvas.width, this.canvas.height);
    gl.uniform1f(u.u_scale, this.scale);
    this.bindTex(0, backdrop.tex);
    this.bindTex(1, pyr[0].b.tex);
    this.bindTex(2, pyr[1].b.tex);
    this.bindTex(3, pyr[2].b.tex);
    gl.uniform1i(u.u_scene, 0);
    gl.uniform1i(u.u_b1, 1);
    gl.uniform1i(u.u_b2, 2);
    gl.uniform1i(u.u_b3, 3);
    gl.uniform1i(u.u_data, 4);
    gl.uniform1i(u.u_groups, layer.groups);
    gl.uniform3f(u.u_levels, LEVEL_SIGMA[0] * this.scale, LEVEL_SIGMA[1] * this.scale, LEVEL_SIGMA[2] * this.scale);
    gl.uniform1f(u.u_refraction, material.refraction);
    gl.uniform1f(u.u_dispersion, material.dispersion);
    gl.uniform1f(u.u_ior, material.ior);
    gl.uniform1f(u.u_specular, material.specular);
    gl.uniform1f(u.u_fresnel, material.fresnel);
    gl.uniform1f(u.u_saturation, material.saturation);
    gl.uniform1f(u.u_brightness, material.brightness);
    gl.uniform1f(u.u_shadow, material.shadow);
    gl.uniform1f(u.u_legibility, material.legibility);
    gl.uniform1f(u.u_lift, material.lift);
    gl.uniform1f(u.u_contentDepth, material.contentRefraction);
    gl.uniform2f(u.u_light, Math.cos((material.lightAngle * Math.PI) / 180), Math.sin((material.lightAngle * Math.PI) / 180));
    gl.useProgram(this.glass.program);
    this.bindFb(out.mrt, out.color.w, out.color.h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Composite rasterized HTML through the layer's refraction map. */
  private contentPass(quads: ContentQuad[], comp: Comp) {
    if (!quads.length) return;
    const gl = this.gl;
    const u = this.contentP.u;
    gl.useProgram(this.contentP.program);
    // Color-only framebuffer: the refraction map is read, never written, here.
    this.bindFb(comp.color.fb, comp.color.w, comp.color.h);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform2f(u.u_res, comp.color.w, comp.color.h);
    gl.uniform1f(u.u_useOff, this.supportsContent ? 1 : 0);
    this.bindTex(1, comp.off);
    gl.uniform1i(u.u_off, 1);
    gl.uniform1i(u.u_tex, 0);
    for (const q of quads) {
      this.bindTex(0, q.tex);
      gl.uniform4f(u.u_rect, q.rect[0], q.rect[1], q.rect[2], q.rect[3]);
      gl.uniform4f(u.u_geom, q.geom[0], q.geom[1], q.geom[2], q.geom[3]);
      gl.uniform1f(u.u_opacity, q.opacity);
      gl.uniform1f(u.u_z, q.z);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.disable(gl.BLEND);
  }

  /** Read back the most-blurred backdrop level (tiny) to judge brightness behind shapes. */
  readLuma(): LumaMap {
    const gl = this.gl;
    const t = this.base[2].b;
    const n = t.w * t.h * 4;
    if (this.lumaBuf.length !== n) this.lumaBuf = new Uint8Array(n);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    gl.readPixels(0, 0, t.w, t.h, gl.RGBA, gl.UNSIGNED_BYTE, this.lumaBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { data: this.lumaBuf, w: t.w, h: t.h };
  }

  /** The backdrop changes by itself (video, an animated wallpaper) or still needs drawing. */
  get animating() {
    return this.sceneDirty || this.bg.type === 'video' || (this.bg.type === 'procedural' && this.bg.animate !== false);
  }

  get cssSize() {
    return [this.cssW, this.cssH] as const;
  }

  destroy() {
    const gl = this.gl;
    deleteTarget(gl, this.sceneT);
    for (const p of [this.base, this.floatPyr]) p.forEach((l) => (deleteTarget(gl, l.a), deleteTarget(gl, l.b)));
    for (const c of this.comps) {
      deleteTarget(gl, c.color);
      gl.deleteTexture(c.off);
      gl.deleteFramebuffer(c.mrt);
    }
    gl.deleteTexture(this.dataTex);
    gl.deleteTexture(this.imageTex);
    gl.deleteTexture(this.overlayTex);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
