/**
 * Rasterize an HTML element into an image through SVG <foreignObject>.
 *
 * The element is deep-cloned with every computed style inlined (so it renders
 * the same without its ancestors), page stylesheets are embedded for
 * pseudo-elements like range-input thumbs, and resolved CSS variables are set
 * on the root. System fonts and inline SVG work; external images and web
 * fonts would need to be inlined as data URLs first.
 *
 * Chromium does not taint canvases or WebGL with these images, so they can be
 * uploaded as textures.
 */

const SKIP = /^(transition|animation|will-change|cursor|pointer-events|user-select|-webkit-user-select|touch-action|view-transition)/;
const XHTML = 'http://www.w3.org/1999/xhtml';

let pageCss: string | null = null;
let cssVars: string[] = [];

function collectCss() {
  if (pageCss != null) return;
  let css = '';
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      for (const r of Array.from(sheet.cssRules)) css += r.cssText + '\n';
    } catch {
      /* cross-origin sheet: skip */
    }
  }
  // Hidden-content rules must not hide the clone.
  pageCss = css.replace(/\.lg-content-hidden[^{]*\{[^}]*\}/g, '');
  cssVars = [...new Set(css.match(/--[\w-]+/g) ?? [])];
}

/** Forget cached page CSS (call after stylesheets change, e.g. on hot reload). */
export function invalidatePageCss() {
  pageCss = null;
}

function inline(src: Element, dst: Element) {
  const cs = getComputedStyle(src);
  let s = '';
  for (let i = 0; i < cs.length; i++) {
    const p = cs[i];
    if (SKIP.test(p)) continue;
    s += `${p}:${cs.getPropertyValue(p)};`;
  }
  dst.setAttribute('style', s);
  dst.classList?.remove('lg-content-hidden');
  if (src instanceof HTMLInputElement) {
    dst.setAttribute('value', src.value);
    if (src.checked) dst.setAttribute('checked', '');
    else dst.removeAttribute('checked');
  } else if (src instanceof HTMLTextAreaElement) {
    dst.textContent = src.value;
  } else if (src instanceof HTMLCanvasElement) {
    try {
      const img = document.createElementNS(XHTML, 'img');
      img.setAttribute('src', src.toDataURL());
      img.setAttribute('style', s);
      dst.replaceWith(img);
      return;
    } catch {
      /* tainted canvas: leave blank */
    }
  }
  const sc = src.children;
  const dc = dst.children;
  for (let i = 0; i < sc.length; i++) if (dc[i]) inline(sc[i], dc[i]);
}

const escapeXmlText = (t: string) => t.replace(/]]>/g, ']]]]><![CDATA[>');

export interface Raster {
  image: HTMLImageElement;
  /** Layout size of the element in CSS px (before transforms). */
  width: number;
  height: number;
  /** Transparent margin around the element, CSS px (for shadows and overflow). */
  pad: number;
}

/**
 * Untransformed border-box size with sub-pixel precision. offsetWidth rounds
 * (73.11 → 73), which is enough to wrap a label onto a second line in the clone.
 */
function borderBox(el: HTMLElement | SVGElement) {
  const cs = getComputedStyle(el);
  const px = (v: string) => parseFloat(v) || 0;
  let w = px(cs.width);
  let h = px(cs.height);
  if (!cs.width.endsWith('px') || !cs.height.endsWith('px')) {
    // Inline boxes report "auto": fall back to the layout box.
    const html = el as HTMLElement;
    w = html.offsetWidth ?? el.getBoundingClientRect().width;
    h = html.offsetHeight ?? el.getBoundingClientRect().height;
  } else if (cs.boxSizing !== 'border-box') {
    w += px(cs.paddingLeft) + px(cs.paddingRight) + px(cs.borderLeftWidth) + px(cs.borderRightWidth);
    h += px(cs.paddingTop) + px(cs.paddingBottom) + px(cs.borderTopWidth) + px(cs.borderBottomWidth);
  }
  return { w, h };
}

export async function rasterize(el: HTMLElement | SVGElement, scale: number, pad = 10): Promise<Raster | null> {
  collectCss();
  const { w, h } = borderBox(el);
  if (!w || !h) return null;
  const clone = el.cloneNode(true) as Element;
  inline(el, clone);
  // Detach the root from its context: no transform, opacity or offsets.
  const rootCs = getComputedStyle(el);
  let vars = '';
  for (const v of cssVars) {
    const val = rootCs.getPropertyValue(v);
    if (val) vars += `${v}:${val};`;
  }
  clone.setAttribute(
    'style',
    clone.getAttribute('style') +
      `position:relative;left:auto;top:auto;right:auto;bottom:auto;inset:auto;margin:0;transform:none;translate:none;scale:none;rotate:none;opacity:1;filter:none;box-sizing:border-box;width:${w}px;height:${h}px;` +
      vars,
  );

  const W = w + pad * 2;
  const H = h + pad * 2;
  const s = Math.max(1, scale);
  const body = new XMLSerializer().serializeToString(clone);
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${Math.ceil(W * s)}" height="${Math.ceil(H * s)}" viewBox="0 0 ${W} ${H}">` +
    `<foreignObject x="0" y="0" width="${W}" height="${H}">` +
    `<div xmlns="${XHTML}" style="width:${W}px;height:${H}px;padding:${pad}px;box-sizing:border-box;margin:0">` +
    `<style><![CDATA[${escapeXmlText(pageCss!)}]]></style>${body}</div></foreignObject></svg>`;
  const image = new Image();
  image.decoding = 'async';
  image.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
  try {
    await image.decode();
  } catch {
    return null;
  }
  return { image, width: w, height: h, pad };
}
