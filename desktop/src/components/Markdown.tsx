// Markdown, rendered while it streams: unclosed fences, bold and code spans
// are closed for the moment so no raw ** or ``` shows. Raw HTML from the
// model is shown as text (this page can reach files and commands), links
// open in the default browser, shell blocks become run boxes.
import { memo, useMemo, useState } from "react";
import { Lexer, Parser, Renderer, type Token, type Tokens } from "marked";
import hljs from "highlight.js/lib/common";
import { RunBox } from "./RunBox";
import { Icon } from "./Icons";
import { openLink } from "../state";
import { openArtifact, previewKind } from "./ArtifactPanel";

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const renderer = new Renderer();
renderer.html = ({ text }: Tokens.HTML | Tokens.Tag) => esc(text);
renderer.link = function ({ href, title, tokens }: Tokens.Link) {
  const ok = /^(https?:|mailto:|#)/i.test(href || "");
  const inner = this.parser.parseInline(tokens);
  return ok ? `<a href="${esc(href)}"${title ? ` title="${esc(title)}"` : ""}>${inner}</a>` : inner;
};
renderer.image = ({ href, text }: Tokens.Image) => (/^https:/i.test(href || "") ? `<img src="${esc(href)}" alt="${esc(text || "")}" loading="lazy">` : esc(text || ""));

const SHELL = new Set(["bash", "sh", "shell", "zsh", "cmd", "bat", "batch", "powershell", "ps", "ps1", "pwsh", "console", "terminal"]);

/** Closes what a half-written reply leaves open. */
export function closeOpen(src: string): string {
  let s = src;
  const fences = (s.match(/^\s*(```|~~~)/gm) || []).length;
  if (fences % 2 === 1) return s + "\n```";
  const lastPara = s.slice(s.lastIndexOf("\n\n") + 1);
  const noCode = lastPara.replace(/`[^`]*`/g, "");
  if ((lastPara.match(/`/g) || []).length % 2 === 1) s += "`";
  if ((noCode.match(/\*\*/g) || []).length % 2 === 1) s += "**";
  else if ((noCode.match(/__/g) || []).length % 2 === 1) s += "__";
  // a lone trailing * or _ about to become emphasis
  s = s.replace(/(\s)[*_]{1,2}$/, "$1");
  return s;
}

const LANG_NAMES: Record<string, string> = {
  js: "JavaScript", javascript: "JavaScript", jsx: "JSX", ts: "TypeScript", typescript: "TypeScript", tsx: "TSX", py: "Python", python: "Python", rs: "Rust", rust: "Rust",
  html: "HTML", css: "CSS", scss: "SCSS", json: "JSON", md: "Markdown", markdown: "Markdown", yaml: "YAML", yml: "YAML", toml: "TOML", sql: "SQL", go: "Go", java: "Java",
  c: "C", cpp: "C++", "c++": "C++", cs: "C#", csharp: "C#", rb: "Ruby", ruby: "Ruby", php: "PHP", swift: "Swift", kt: "Kotlin", kotlin: "Kotlin", lua: "Lua", luau: "Luau",
  xml: "XML", svg: "SVG", wgsl: "WGSL", glsl: "GLSL", diff: "Diff", ini: "INI", dockerfile: "Dockerfile", makefile: "Makefile", text: "Text", txt: "Text", plaintext: "Text",
};
export const langName = (l: string) => LANG_NAMES[l] || (l ? l[0].toUpperCase() + l.slice(1) : "Code");

/** Lines past which a finished block starts folded. */
const FOLD = 28;

function CodeBlock({ lang, code, streaming }: { lang: string; code: string; streaming?: boolean }) {
  const [copied, setCopied] = useState(false);
  const [wrap, setWrap] = useState(false);
  const [unfold, setUnfold] = useState(false);
  const pk = streaming ? null : previewKind(lang, code);
  const body = code.replace(/\n$/, "");
  const lines = body.split("\n").length;
  const folded = !streaming && !unfold && lines > FOLD + 6;
  const html = useMemo(() => {
    try {
      return lang && hljs.getLanguage(lang) ? hljs.highlight(body, { language: lang }).value : lang ? esc(body) : hljs.highlightAuto(body).value;
    } catch {
      return esc(body);
    }
  }, [lang, body]);
  return (
    <div className={`code ${wrap ? "wrap" : ""} ${folded ? "folded" : ""}`}>
      <div className="code-head">
        <span className="code-lang">{langName(lang)}</span>
        {lines > 1 && <span className="code-lines">{lines} lines</span>}
        <span className="grow" />
        <button className={`ghost-btn ${wrap ? "on" : ""}`} title={wrap ? "Don't wrap long lines" : "Wrap long lines"} onClick={() => setWrap(!wrap)}>
          <Icon name="list" size={13} />
        </button>
        {pk && (
          <button className="ghost-btn preview-btn" onClick={() => openArtifact(pk === "html" ? "Page" : pk === "svg" ? "Image" : "Document", code, lang)}>
            <Icon name="eye" size={13} /> Preview
          </button>
        )}
        {!pk && !streaming && lines > 12 && (
          <button className="ghost-btn" title="Open beside the chat" onClick={() => openArtifact(`${lang || "code"} · ${lines} lines`, code, lang)}>
            <Icon name="external" size={13} />
          </button>
        )}
        <button
          className="ghost-btn"
          onClick={() => {
            navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
          }}
        >
          <Icon name={copied ? "check" : "copy"} size={13} /> {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="code-scroll">
        {lines > 3 && (
          <div className="code-gutter" aria-hidden="true">
            {Array.from({ length: folded ? FOLD : lines }, (_, i) => (
              <span key={i}>{i + 1}</span>
            ))}
          </div>
        )}
        <pre>
          <code className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
        </pre>
      </div>
      {folded && (
        <button className="code-unfold" onClick={() => setUnfold(true)}>
          <Icon name="down" size={13} /> Show all {lines} lines
        </button>
      )}
    </div>
  );
}

// A block re-renders only when its source changes: while a reply streams,
// the finished blocks above the last one are left alone.
const Block = memo(
  function Block({ tok, streaming }: { tok: Token; streaming: boolean }) {
  if (tok.type === "code") {
    const t = tok as Tokens.Code;
    const lang = (t.lang || "").trim().split(/\s/)[0].toLowerCase();
    if (SHELL.has(lang) && !streaming) return <RunBox command={t.text} lang={lang} />;
    return <CodeBlock lang={lang} code={t.text} streaming={streaming} />;
  }
  const html = Parser.parse([tok], { renderer, gfm: true, breaks: false });
  return <div className="md-block" dangerouslySetInnerHTML={{ __html: html }} />;
  },
  (a, b) => a.streaming === b.streaming && a.tok.type === b.tok.type && a.tok.raw === b.tok.raw,
);

export const Markdown = memo(function Markdown({ text, streaming = false }: { text: string; streaming?: boolean }) {
  const toks = useMemo(() => {
    try {
      return Lexer.lex(streaming ? closeOpen(text) : text, { gfm: true });
    } catch {
      return [{ type: "paragraph", raw: text, text, tokens: [{ type: "text", raw: text, text }] } as unknown as Token];
    }
  }, [text, streaming]);
  return (
    <div
      className="md"
      onClick={(e) => {
        const a = (e.target as HTMLElement).closest("a");
        if (a && a.getAttribute("href")) {
          e.preventDefault();
          const href = a.getAttribute("href")!;
          if (!href.startsWith("#")) openLink(href);
        }
      }}
    >
      {toks.map((t, i) => (t.type === "space" ? null : <Block key={i} tok={t} streaming={streaming && i === toks.length - 1} />))}
    </div>
  );
});
