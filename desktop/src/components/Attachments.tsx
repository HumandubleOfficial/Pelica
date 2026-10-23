// Files attached to a message (dropped on the window, pasted, or picked with
// the paperclip). The models read text, so a file goes into the message as
// an <attachment> block; the chat shows it as a chip. Photos go along as
// images to a model that sees them, else as the text Windows OCR reads.
import { bridge } from "../lib/bridge";
import { app, toast, useApp, type Photo } from "../state";
import { Icon } from "./Icons";

export interface Attached {
  name: string;
  text: string;
  path?: string;
  /** a photo (data: URL, downscaled); `text` is what OCR read from it */
  image?: string;
}

const MAX_CHARS = 120_000;

export const attachmentBlock = (name: string, text: string, image = false) => `\n\n<attachment name="${name.replace(/"/g, "'")}"${image ? ' kind="image"' : ""}>\n${text}\n</attachment>`;

const BLOCK = /\n*<attachment name="([^"]*)"( kind="image")?>\n([\s\S]*?)\n<\/attachment>/g;

/** A message's own text and the attachments in it (photos matched to their images). */
export function splitAttachments(content: string, photos: Photo[] = []): { text: string; files: Attached[] } {
  if (!content.includes("<attachment ")) return { text: content, files: [] };
  const files: Attached[] = [];
  const text = content
    .replace(BLOCK, (_m, name: string, img: string | undefined, body: string) => (files.push({ name, text: body, image: img ? photos.find((p) => p.name === name)?.data || "" : undefined }), ""))
    .trim();
  return { text, files };
}

export const joinAttachments = (text: string, files: Attached[]) => text + files.map((f) => attachmentBlock(f.name, f.text, f.image !== undefined)).join("");

const IMAGE = /^image\/(png|jpeg|webp|gif|bmp)$/;
const MAX_SIDE = 1568;

/** A photo as a JPEG data: URL, at most MAX_SIDE pixels on its long side. */
async function shrink(f: Blob): Promise<string> {
  const bmp = await createImageBitmap(f);
  const k = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(bmp.width * k));
  c.height = Math.max(1, Math.round(bmp.height * k));
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff";
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close();
  return c.toDataURL("image/jpeg", 0.88);
}

const ipcErr = (e: any) => String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, "");

/** What goes into the message for a photo: a note for a model that sees it,
 *  else the text Windows reads in it (said plainly, so the model doesn't
 *  pretend to see the picture). */
export async function photoText(f: Attached): Promise<string> {
  if (f.text || f.image === undefined) return f.text;
  if (app.get().status?.vision) return "[A photo, attached as an image.]";
  let read = "";
  try {
    read = bridge && f.image ? await bridge.media.ocr(f.image) : "";
  } catch (e) {
    toast(`Couldn't read the text in ${f.name}: ${ipcErr(e)}`);
  }
  return read
    ? `[A photo. The loaded model can't see images, so this is the text Windows read from it (OCR); layout, pictures and colours are lost.]\n${read}`
    : "[A photo. The loaded model can't see images, and Windows found no text in it.]";
}

/** Reads files as text (photos as images); binary and huge files are refused with a note. */
export async function readFiles(list: FileList | File[]): Promise<Attached[]> {
  const out: Attached[] = [];
  for (const f of Array.from(list)) {
    if (IMAGE.test(f.type)) {
      if (f.size > 40 * 1024 * 1024) {
        toast(`${f.name} is too big to attach.`);
        continue;
      }
      try {
        const name = !f.name || f.name === "image.png" ? `photo-${new Date().toTimeString().slice(0, 8).replace(/:/g, "")}.jpg` : f.name;
        out.push({ name, text: "", image: await shrink(f) });
      } catch {
        toast(`${f.name} couldn't be opened as an image.`);
      }
      continue;
    }
    if (f.size > 8 * 1024 * 1024) {
      toast(`${f.name} is too big to attach (${(f.size / 1024 / 1024).toFixed(1)} MB).`);
      continue;
    }
    let text = await f.text();
    if (/\u0000/.test(text.slice(0, 8000)) || /^(image|audio|video)\//.test(f.type)) {
      toast(`${f.name} isn't text or a photo Pelica can read.`);
      continue;
    }
    if (text.length > MAX_CHARS) text = text.slice(0, MAX_CHARS) + `\n…[cut: the file has ${text.length.toLocaleString()} characters]`;
    let path: string | undefined;
    try {
      path = bridge?.pathFor(f) || undefined;
    } catch {}
    out.push({ name: f.name, text, path });
  }
  return out;
}

const size = (n: number) => (n < 1000 ? `${n} chars` : `~${Math.max(1, Math.round(n / 3.6 / 100) / 10)}K tokens`);

/** The chips: in the message box (removable) and in sent messages (click to view). */
export function AttachChips({ files, onRemove }: { files: Attached[]; onRemove?: (i: number) => void }) {
  if (!files.length) return null;
  return (
    <div className="attach-chips">
      {files.map((f, i) =>
        f.image !== undefined ? (
          <span key={i} className="attach-chip photo" title={f.text ? `${f.name}\n\n${f.text}` : f.name}>
            <button className="attach-open" onClick={() => f.image && app.set({ photo: f.image })}>
              {f.image ? <img src={f.image} alt={f.name} /> : <Icon name="image" size={18} />}
            </button>
            {onRemove && (
              <button className="attach-x" title="Remove" onClick={() => onRemove(i)}>
                <Icon name="x" size={11} stroke={2.2} />
              </button>
            )}
          </span>
        ) : (
          <span key={i} className="attach-chip" title={f.path || f.name}>
            <button className="attach-open" onClick={() => app.set({ artifact: { title: f.name, kind: /\.(md|markdown)$/i.test(f.name) ? "markdown" : /\.html?$/i.test(f.name) ? "html" : /\.svg$/i.test(f.name) ? "svg" : "code", code: f.text, lang: f.name.split(".").pop() } })}>
              <span className="attach-icon">
                <Icon name={f.name === "Clipboard" ? "clipboard" : "file"} size={14} />
              </span>
              <span className="attach-text">
                <span className="attach-name">{f.name}</span>
                <span className="attach-size">{size(f.text.length)}</span>
              </span>
            </button>
            {onRemove && (
              <button className="attach-x" title="Remove" onClick={() => onRemove(i)}>
                <Icon name="x" size={11} stroke={2.2} />
              </button>
            )}
          </span>
        ),
      )}
    </div>
  );
}

/** A photo at full size (click anywhere to close). */
export function PhotoViewer() {
  const p = useApp((s) => s.photo);
  if (!p) return null;
  return (
    <div className="modal-back photo-view" onClick={() => app.set({ photo: null })}>
      <img src={p} alt="" />
    </div>
  );
}
