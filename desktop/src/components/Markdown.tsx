// Markdown, rendered while it streams: unclosed fences, bold and code spans
// are closed for the moment so no raw ** or ``` shows. Raw HTML from the
// model is shown as text (this page can reach files and commands), links
// open in the default browser, shell blocks become run boxes.
import { memo, useLayoutEffect, useMemo, useRef, useState } from "react";
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
// [1] in a reply that has passages to cite becomes a button (Citations.tsx)
let citeNums: Set<number> | null = null;
const plainText = Renderer.prototype.text;
renderer.text = function (t: Tokens.Text | Tokens.Escape) {
  const out = plainText.call(this, t);
  if (!citeNums || ("tokens" in t && t.tokens)) return out;
  return out.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (all: string, list: string) => {
    const nums = list.split(",").map(Number);
    return nums.every((n) => citeNums!.has(n)) ? nums.map((n) => `<button class="cite" data-cite="${n}">${n}</button>`).join("") : all;
  });
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
/** Past this a block with no language stays plain: guessing tries every language. */
const AUTO_MAX = 3000;
/** How often a block being written is coloured again (the text after that shows plain). */
const COLOUR_MS = 150;

function CodeBlock({ lang, code, streaming }: { lang: string; code: string; streaming?: boolean }) {
  const [copied, setCopied] = useState(false);
  const [wrap, setWrap] = useState(false);
  const [unfold, setUnfold] = useState(false);
  const pk = streaming ? null : previewKind(lang, code);
  const body = code.replace(/\n$/, "");
  const lines = body.split("\n").length;
  const folded = !streaming && !unfold && lines > FOLD + 6;
  const done = useRef({ at: 0, len: 0, html: "" });
  const html = useMemo(() => {
    const known = !!lang && !!hljs.getLanguage(lang);
    if (streaming && !known) return esc(body);
    const now = performance.now();
    if (streaming && now - done.current.at < COLOUR_MS && body.length >= done.current.len) return done.current.html + esc(body.slice(done.current.len));
    try {
      const out = known ? hljs.highlight(body, { language: lang }).value : lang || body.length > AUTO_MAX ? esc(body) : hljs.highlightAuto(body).value;
      done.current = { at: now, len: body.length, html: out };
      return out;
    } catch {
      return esc(body);
    }
  }, [lang, body, streaming]);
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

const FADE_MS = 450;

/** New words fade in while a block streams. Each update replaces the block's
 *  HTML, so every render wraps the text that arrived in the last FADE_MS in
 *  spans whose (negative) delay carries their fade on from where it was. */
function useFadeIn(ref: React.RefObject<HTMLDivElement | null>, on: boolean, html: string) {
  const chunks = useRef<{ at: number; t: number }[]>([]);
  const seen = useRef(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !on) return;
    const now = performance.now();
    // (without the trailing newline the HTML ends with: it moves as text arrives)
    const len = el.textContent?.trimEnd().length || 0;
    if (len > seen.current) chunks.current.push({ at: seen.current, t: now });
    seen.current = len;
    chunks.current = chunks.current.filter((c) => now - c.t < FADE_MS && c.at < len);
    const live = chunks.current;
    if (!live.length) return;
    // the text nodes, with where each starts in the block's text
    const nodes: { n: Text; at: number }[] = [];
    const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    let at = 0;
    for (let n = walk.nextNode() as Text | null; n; n = walk.nextNode() as Text | null) {
      nodes.push({ n, at });
      at += n.data.length;
    }
    for (let i = live.length - 1; i >= 0; i--) {
      const from = live[i].at;
      const to = i + 1 < live.length ? live[i + 1].at : at;
      const delay = `${-(now - live[i].t)}ms`;
      for (const { n, at: s } of nodes) {
        const e = s + n.data.length;
        if (e <= from || s >= to || !n.parentNode || !n.data.trim()) continue;
        // (split off the part before and after the range, wrap the middle)
        let mid = n;
        if (from > s) mid = mid.splitText(from - s);
        if (to < e) mid.splitText(to - Math.max(from, s));
        const span = document.createElement("span");
        span.className = "tok-in";
        span.style.animationDelay = delay;
        mid.parentNode!.insertBefore(span, mid);
        span.appendChild(mid);
      }
    }
  }, [html, on]);
}

// A block re-renders only when its source changes: while a reply streams,
// the finished blocks above the last one are left alone.
const Block = memo(
  function Block({ tok, streaming, quiet, cites }: { tok: Token; streaming: boolean; quiet?: boolean; cites?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  citeNums = cites ? new Set(cites.split(",").map(Number)) : null;
  const html = tok.type === "code" ? "" : Parser.parse([tok], { renderer, gfm: true, breaks: false });
  citeNums = null;
  useFadeIn(ref, streaming, html);
  if (tok.type === "code") {
    const t = tok as Tokens.Code;
    const lang = (t.lang || "").trim().split(/\s/)[0].toLowerCase();
    if (SHELL.has(lang) && !streaming && !quiet) return <RunBox command={t.text} lang={lang} />;
    return <CodeBlock lang={lang} code={t.text} streaming={streaming} />;
  }
  return <div ref={ref} className="md-block" dangerouslySetInnerHTML={{ __html: html }} />;
  },
  (a, b) => a.streaming === b.streaming && a.quiet === b.quiet && a.cites === b.cites && a.tok.type === b.tok.type && a.tok.raw === b.tok.raw,
);

/** `quiet`: no run boxes for shell blocks (thinking, where commands are drafts).
 *  `cites`: the passage numbers this text can cite, "1,2,3". */
export const Markdown = memo(function Markdown({ text, streaming = false, quiet = false, className = "", cites }: { text: string; streaming?: boolean; quiet?: boolean; className?: string; cites?: string }) {
  const toks = useMemo(() => {
    try {
      return Lexer.lex(streaming ? closeOpen(text) : text, { gfm: true });
    } catch {
      return [{ type: "paragraph", raw: text, text, tokens: [{ type: "text", raw: text, text }] } as unknown as Token];
    }
  }, [text, streaming]);
  return (
    <div
      className={`md ${className}`}
      onClick={(e) => {
        const a = (e.target as HTMLElement).closest("a");
        if (a && a.getAttribute("href")) {
          e.preventDefault();
          const href = a.getAttribute("href")!;
          if (!href.startsWith("#")) openLink(href);
        }
      }}
    >
      {toks.map((t, i) => (t.type === "space" ? null : <Block key={i} tok={t} streaming={streaming && i === toks.length - 1} quiet={quiet} cites={cites} />))}
    </div>
  );
});
