// Projects (as in the Claude app): chats that share instructions, files and
// a folder. The list of projects, and a project's page with its chats on the
// left and what every chat in it gets (instructions, files, folder) on the
// right. The agent puts a chat's project into its system prompt (agent.ts).
import { useEffect, useMemo, useRef, useState } from "react";
import { bridge, type DocsInfo } from "../lib/bridge";
import { app, deleteProject, newProject, openChat, openProject, projectOf, saveProject, startNewChat, toast, useApp, type Project } from "../state";
import { Icon } from "./Icons";
import { docName } from "../lib/docs";

/** How much file text a project keeps (it goes into every chat's prompt). */
const FILE_MAX = 300_000;
const ALL_MAX = 1_000_000;
const kb = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
const ago = (t: number) => {
  const d = Math.round((Date.now() - t) / 86400e3);
  return d < 1 ? "today" : d === 1 ? "yesterday" : d < 30 ? `${d} days ago` : new Date(t).toLocaleDateString();
};

export function ProjectsPage() {
  const projects = useApp((s) => s.settings.projects);
  const chats = useApp((s) => s.chats);
  const create = () => openProject(newProject().id);
  return (
    <div className="page projects-page">
      <div className="page-head">
        <h1>Projects</h1>
        <button className="btn primary" onClick={create}>
          <Icon name="plus" size={14} /> New project
        </button>
      </div>
      <p className="muted">A project keeps related chats together. Every chat in it gets the project's instructions and files, and can work in its folder.</p>
      <div className="proj-grid">
        {projects.map((p) => {
          const mine = chats.filter((c) => c.project === p.id);
          const last = Math.max(p.created, ...mine.map((c) => c.updated));
          return (
            <button key={p.id} className="proj-card" onClick={() => openProject(p.id)}>
              <b>{p.name}</b>
              <span className="muted small proj-card-desc">{p.instructions.trim().split("\n")[0] || "No instructions yet"}</span>
              <span className="muted small">
                {mine.length} chat{mine.length === 1 ? "" : "s"}
                {p.files.length ? ` · ${p.files.length} file${p.files.length === 1 ? "" : "s"}` : ""} · {ago(last)}
              </span>
            </button>
          );
        })}
        {!projects.length && (
          <div className="empty-note">
            No projects yet.{" "}
            <button className="btn small" onClick={create}>
              Create one
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/** Folders of documents the project's chats search: PDFs, Word, PowerPoint,
 *  notes. Read once on this PC (electron/docs.ts), then again only for what
 *  changed, each time the page opens. */
function DocsSection({ p, set }: { p: Project; set: (patch: Partial<Project>) => void }) {
  const docs = p.docs || [];
  const [info, setInfo] = useState<Record<string, DocsInfo>>({});
  const [now, setNow] = useState<Record<string, string>>({});
  const read = (d: string) =>
    bridge?.docs
      .index(d)
      .then((v) => (setInfo((x) => ({ ...x, [d]: v })), setNow((x) => ({ ...x, [d]: "" }))))
      .catch((e) => toast(String(e?.message || e).replace(/^Error invoking remote method '[^']+': (Error: )?/, "")));
  useEffect(() => {
    if (!bridge) return;
    const off = bridge.docs.onProgress((e) => setNow((x) => ({ ...x, [e.folder]: e.file ? `Reading ${e.file}… (${e.done + 1} of ${e.total})` : "" })));
    for (const d of docs) {
      bridge.docs.info(d).then((v) => setInfo((x) => ({ ...x, [d]: v }))).catch(() => {});
      read(d);
    }
    return off;
  }, [p.id, docs.join("|")]);
  return (
    <section>
      <div className="proj-side-head">
        Documents
        <button
          className="btn small ghost"
          disabled={!bridge}
          onClick={async () => {
            const d = await bridge?.folder.pick();
            if (!d || docs.some((x) => x.toLowerCase() === d.toLowerCase())) return;
            set({ docs: [...docs, d] });
          }}
        >
          <Icon name="plus" size={12} /> Add folder
        </button>
      </div>
      {docs.map((d) => {
        const v = info[d];
        return (
          <div key={d} className="proj-doc">
            <div className="proj-file">
              <Icon name="folder" size={13} />
              <span className="proj-file-name" title={d}>
                {docName(d)}
              </span>
              <button className="ghost-btn" title="Read it again" disabled={!!now[d]} onClick={() => read(d)}>
                <Icon name="reload" size={12} />
              </button>
              <button
                className="ghost-btn"
                title="Stop searching this folder (its files stay where they are)"
                onClick={() => {
                  set({ docs: docs.filter((x) => x !== d) });
                  bridge?.docs.forget(d);
                }}
              >
                <Icon name="x" size={12} />
              </button>
            </div>
            <div className="muted small proj-doc-state">
              {now[d] || (v ? (v.files ? `${v.files} file${v.files === 1 ? "" : "s"} · ${v.passages} passages${v.unreadable.length ? ` · ${v.unreadable.length} couldn't be read` : ""}` : "No documents found in it yet.") : "…")}
            </div>
          </div>
        );
      })}
      {!docs.length && <div className="muted small">Folders of PDFs, Word and PowerPoint files and notes. Chats in the project search them and cite what they use. Everything stays on this PC.</div>}
    </section>
  );
}

export function ProjectPage() {
  const id = useApp((s) => s.projectId);
  const p = useApp((s) => s.settings.projects.find((x) => x.id === s.projectId));
  const all = useApp((s) => s.chats);
  const chats = useMemo(() => all.filter((c) => c.project === id), [all, id]);
  const [instr, setInstr] = useState<string | null>(null);
  const [sure, setSure] = useState(false);
  const pick = useRef<HTMLInputElement>(null);
  if (!id || !p) {
    return (
      <div className="page">
        <div className="empty-note">
          That project isn't there anymore.{" "}
          <button className="btn small" onClick={() => app.set({ view: "projects" })}>
            All projects
          </button>
        </div>
      </div>
    );
  }
  const set = (patch: Partial<Project>) => saveProject({ ...projectOf(p.id)!, ...patch });
  const used = p.files.reduce((n, f) => n + f.text.length, 0);
  async function addFiles(list: FileList | null) {
    if (!list) return;
    let files = [...projectOf(p!.id)!.files];
    let total = files.reduce((n, f) => n + f.text.length, 0);
    const skipped: string[] = [];
    for (const f of [...list]) {
      const text = await f.text().catch(() => "");
      if (!text || text.includes("\u0000")) {
        skipped.push(`${f.name} (not a text file)`);
        continue;
      }
      if (text.length > FILE_MAX) {
        skipped.push(`${f.name} (over ${kb(FILE_MAX)})`);
        continue;
      }
      if (total + text.length > ALL_MAX) {
        skipped.push(`${f.name} (the project's files are limited to ${kb(ALL_MAX)})`);
        continue;
      }
      files = [...files.filter((x) => x.name !== f.name), { name: f.name, text }];
      total = files.reduce((n, x) => n + x.text.length, 0);
    }
    set({ files });
    if (skipped.length) toast(`Not added: ${skipped.join(", ")}`);
  }
  return (
    <div className="page project-page">
      <button className="link-btn proj-back" onClick={() => app.set({ view: "projects" })}>
        <Icon name="back" size={13} /> All projects
      </button>
      <div className="page-head">
        <input className="proj-name" value={p.name} onChange={(e) => set({ name: e.target.value })} onBlur={(e) => !e.target.value.trim() && set({ name: "Untitled project" })} spellCheck={false} aria-label="Project name" />
        <button className="btn primary" onClick={() => startNewChat(p.folder, p.id)}>
          <Icon name="plus" size={14} /> New chat
        </button>
      </div>
      <div className="proj-cols">
        <div className="proj-chats">
          {chats.map((c) => (
            <button key={c.id} className="proj-chat" onClick={() => openChat(c.id)}>
              <span className="proj-chat-title">{c.title}</span>
              <span className="muted small">{ago(c.updated)}</span>
            </button>
          ))}
          {!chats.length && <div className="empty-note">No chats in this project yet. Start one with New chat, or move a chat here from its menu (Move to project).</div>}
        </div>
        <aside className="proj-side">
          <section>
            <div className="proj-side-head">Instructions</div>
            <textarea
              className="text-input proj-instr"
              rows={7}
              value={instr ?? p.instructions}
              placeholder="What should every chat in this project know or do? E.g. “This is my Rust game engine. Answer briefly and show code.”"
              onChange={(e) => setInstr(e.target.value)}
              onBlur={() => {
                if (instr !== null) set({ instructions: instr });
                setInstr(null);
              }}
            />
          </section>
          <section>
            <div className="proj-side-head">
              Files <span className="muted small" title="Every chat in the project reads them first">{p.files.length ? `${kb(used)} · ~${Math.round(used / 3500)}K tokens` : ""}</span>
              <button className="btn small ghost" onClick={() => pick.current?.click()}>
                <Icon name="plus" size={12} /> Add
              </button>
            </div>
            <input ref={pick} type="file" multiple hidden onChange={(e) => (addFiles(e.target.files), (e.target.value = ""))} />
            {p.files.map((f) => (
              <div key={f.name} className="proj-file">
                <Icon name="file" size={13} />
                <span className="proj-file-name" title={f.name}>
                  {f.name}
                </span>
                <span className="muted small">{kb(f.text.length)}</span>
                <button className="ghost-btn" title="Remove from the project" onClick={() => set({ files: projectOf(p.id)!.files.filter((x) => x.name !== f.name) })}>
                  <Icon name="x" size={12} />
                </button>
              </div>
            ))}
            {!p.files.length && <div className="muted small">Text files (notes, docs, code) every chat in the project can read.</div>}
            {used > 100_000 && <div className="warn-note">That's a lot for a local model: files that don't fit in about 40% of its context are left out (by name) in its chats.</div>}
          </section>
          <DocsSection p={p} set={set} />
          <section>
            <div className="proj-side-head">Folder</div>
            {p.folder ? (
              <div className="proj-file">
                <Icon name="folder" size={13} />
                <span className="proj-file-name" title={p.folder}>
                  {p.folder}
                </span>
                <button className="ghost-btn" title="No folder" onClick={() => set({ folder: null })}>
                  <Icon name="x" size={12} />
                </button>
              </div>
            ) : (
              <div className="muted small">New chats in the project work in this folder: they can read and change its files and run commands there.</div>
            )}
            <button
              className="btn small ghost"
              disabled={!bridge}
              onClick={async () => {
                const d = await bridge?.folder.pick();
                if (d) set({ folder: d });
              }}
            >
              {p.folder ? "Change folder…" : "Choose a folder…"}
            </button>
          </section>
          <section>
            {sure ? (
              <div className="inline">
                <button className="btn small danger" onClick={() => deleteProject(p.id)}>
                  Delete the project
                </button>
                <button className="btn small ghost" onClick={() => setSure(false)}>
                  Cancel
                </button>
              </div>
            ) : (
              <button className="btn small ghost danger" onClick={() => setSure(true)} title="The project's chats stay, outside any project">
                <Icon name="trash" size={13} /> Delete project
              </button>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}
