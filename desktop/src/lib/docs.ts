// Chatting with a project's documents: the passages that match a question go
// into the message as a <documents> block, numbered, so the reply can cite
// them ([1], [2]) and the chat can show where each came from. The block stays
// in the message, so later turns don't change what the model already read.
import { bridge, type DocPassage } from "./bridge";
import type { Chat, Project } from "../state";

export interface Excerpt {
  n: number;
  file: string;
  page?: number;
  text: string;
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
const unesc = (s: string) => s.replace(/&quot;/g, '"').replace(/&amp;/g, "&");

export const DOCS_BLOCK = /\n*<documents>\n[\s\S]*?\n<\/documents>/;
const EXCERPT = /<excerpt n="(\d+)" file="([^"]*)"(?: page="(\d+)")?>\n([\s\S]*?)\n<\/excerpt>/g;

export function docsBlock(found: DocPassage[]): string {
  if (!found.length) return "";
  const body = found.map((p, i) => `<excerpt n="${i + 1}" file="${esc(p.file)}"${p.page ? ` page="${p.page}"` : ""}>\n${p.text}\n</excerpt>`).join("\n");
  return `\n\n<documents>\n${body}\n</documents>`;
}

/** The excerpts a message carries. */
export function excerptsOf(content: string): Excerpt[] {
  const m = DOCS_BLOCK.exec(content);
  if (!m) return [];
  return [...m[0].matchAll(EXCERPT)].map((x) => ({ n: +x[1], file: unesc(x[2]), page: x[3] ? +x[3] : undefined, text: x[4] }));
}

export const withoutDocs = (content: string) => content.replace(DOCS_BLOCK, "");

export const docName = (file: string) => file.split(/[\\/]/).pop() || file;

/** The folder of `folders` that holds `file` (to open it through the bridge). */
export const rootOf = (file: string, folders: string[]) => folders.find((d) => file.toLowerCase().startsWith(d.toLowerCase().replace(/[\\/]+$/, "") + "\\") || file.toLowerCase().startsWith(d.toLowerCase().replace(/[\\/]+$/, "") + "/"));

export const projectDocs = (chat: Chat | null, projects: Project[]) => {
  const p = chat?.project ? projects.find((x) => x.id === chat.project) : undefined;
  return p?.docs?.length ? p.docs : [];
};

/** The block to add to a question in a project with documents ("" if nothing matches). */
export async function excerptsFor(folders: string[], question: string, earlier = ""): Promise<string> {
  if (!bridge || !folders.length || !question.trim()) return "";
  // (a short follow-up like "and the second one?" searches with the question before it too)
  const q = question.length < 60 && earlier ? `${earlier}\n${question}` : question;
  const found = await bridge.docs.search(folders, q, 6).catch(() => []);
  return docsBlock(found);
}

/** The numbers a reply cites, in order of first use. */
export function cited(reply: string): number[] {
  const out: number[] = [];
  for (const m of reply.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) for (const n of m[1].split(",")) if (!out.includes(+n)) out.push(+n);
  return out;
}
