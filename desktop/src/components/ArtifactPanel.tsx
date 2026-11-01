// The artifact panel: previews what a reply made (an HTML page, an SVG,
// Markdown) or shows a file in full, beside the chat. Pages run in a
// sandboxed frame with no access to Pelica and no network requests of
// their own (scripts and styles from https CDNs still load).
import { useMemo, useState } from "react";
import hljs from "highlight.js/lib/common";
import { bridge } from "../lib/bridge";
import { app, toast, useApp, type Artifact } from "../state";
import { Icon } from "./Icons";
import { Markdown } from "./Markdown";

const CSP = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' https:; style-src 'unsafe-inline' https:; img-src data: blob: https:; font-src data: https:; media-src data: blob: https:">`;

function srcDoc(a: Artifact): string {
  if (a.kind === "svg") return `<!doctype html><html><head>${CSP}<style>html,body{margin:0;height:100%;background:#fff}body{display:grid;place-items:center}svg{max-width:96vw;max-height:96vh}</style></head><body>${a.code}</body></html>`;
  // the policy goes first, before anything in the page can run
  return /<head[^>]*>/i.test(a.code) ? a.code.replace(/<head[^>]*>/i, (h) => h + CSP) : `<!doctype html><html><head>${CSP}<meta charset="utf-8"></head><body>${a.code}</body></html>`;
}

const EXT: Record<string, string> = { html: "html", svg: "svg", markdown: "md" };

export function ArtifactPanel() {
  const a = useApp((s) => s.artifact);
  const folder = useApp((s) => s.chat?.folder ?? null);
  const [tab, setTab] = useState<"preview" | "code">("preview");
  const [width, setWidth] = useState(560);
  const [nonce, setNonce] = useState(0);
  const code = useMemo(() => {
    if (!a) return "";
    try {
      const lang = a.kind === "code" ? a.lang : a.kind === "markdown" ? "markdown" : "xml";
      return lang && hljs.getLanguage(lang) ? hljs.highlight(a.code, { language: lang }).value : hljs.highlightAuto(a.code.slice(0, 200000)).value;
    } catch {
      return a.code.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    }
  }, [a]);
  if (!a) return null;
  const previewable = a.kind !== "code";
  const showing = previewable ? tab : "code";

  async function save() {
    if (!a || !folder || !bridge) return;
    const name = /\.\w+$/.test(a.title) ? a.title : `${a.title.replace(/[^\w.-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "artifact"}.${EXT[a.kind] || a.lang || "txt"}`;
    try {
      const r = await bridge.fs.write(folder, name, a.code);
      toast(`Saved ${r.path}`);
    } catch (e: any) {
      toast(String(e.message || e));
    }
  }

  return (
    <div className="artifact" style={{ width }}>
      <div
        className="browser-resize"
        onMouseDown={(e) => {
          const x0 = e.clientX;
          const w0 = width;
          const mv = (m: MouseEvent) => setWidth(Math.max(340, Math.min(window.innerWidth - 460, w0 + (x0 - m.clientX))));
          const up = () => (window.removeEventListener("mousemove", mv), window.removeEventListener("mouseup", up));
          window.addEventListener("mousemove", mv);
          window.addEventListener("mouseup", up);
        }}
      />
      <div className="artifact-bar">
        <span className="artifact-kind">
          <Icon name={a.kind === "code" ? "code" : "eye"} size={14} />
        </span>
        <span className="artifact-title" title={a.title}>
          {a.title}
        </span>
        {previewable && (
          <div className="seg small-seg">
            <button className={showing === "preview" ? "sel" : ""} onClick={() => setTab("preview")}>
              Preview
            </button>
            <button className={showing === "code" ? "sel" : ""} onClick={() => setTab("code")}>
              Code
            </button>
          </div>
        )}
        <span className="grow" />
        {showing === "preview" && a.kind === "html" && (
          <button className="ghost-btn" title="Run it again" onClick={() => setNonce(nonce + 1)}>
            <Icon name="reload" size={14} />
          </button>
        )}
        <button className="ghost-btn" title="Copy" onClick={() => (navigator.clipboard.writeText(a.code), toast("Copied."))}>
          <Icon name="copy" size={14} />
        </button>
        {folder && (
          <button className="ghost-btn" title={`Save into ${folder}`} onClick={save}>
            <Icon name="download" size={14} />
          </button>
        )}
        <button className="ghost-btn" title="Close" onClick={() => app.set({ artifact: null })}>
          <Icon name="x" size={15} />
        </button>
      </div>
      <div className="artifact-body">
        {showing === "code" ? (
          <pre className="artifact-code">
            <code className="hljs" dangerouslySetInnerHTML={{ __html: code }} />
          </pre>
        ) : a.kind === "markdown" ? (
          <div className="artifact-md">
            <Markdown text={a.code} />
          </div>
        ) : (
          <iframe key={nonce} className="artifact-frame" sandbox="allow-scripts allow-modals allow-forms allow-pointer-lock" srcDoc={srcDoc(a)} title={a.title} />
        )}
      </div>
    </div>
  );
}

/** What a code block or file can be previewed as, if anything. */
export function previewKind(lang: string, code: string): Artifact["kind"] | null {
  const l = lang.toLowerCase();
  if (l === "html" || l === "htm" || (l === "xml" && /<html/i.test(code))) return "html";
  if (l === "svg" || ((l === "xml" || !l) && /^\s*<svg[\s>]/i.test(code))) return "svg";
  if (l === "markdown" || l === "md") return "markdown";
  if (!l && /^\s*<!doctype html/i.test(code)) return "html";
  return null;
}

export function openArtifact(title: string, code: string, lang: string) {
  app.set({ artifact: { title, kind: previewKind(lang, code) || "code", code, lang } });
}
