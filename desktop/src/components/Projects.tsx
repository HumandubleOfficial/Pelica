// Projects (as in the Claude app): chats that share instructions, files and
// a folder. The list of projects, and a project's page with its chats on the
// left and what every chat in it gets (instructions, files, folder) on the
// right. The agent puts a chat's project into its system prompt (agent.ts).
import { useEffect, useMemo, useRef, useState } from "react";
import { bridge, type DocsInfo } from "../lib/bridge";
import { app, deleteProject, newProject, openChat, openProject, patchChat, projectOf, saveProject, startNewChat, toast, useApp, type Project } from "../state";
import { Icon } from "./Icons";
import { Menu, type MenuItem } from "./Menu";
import { send } from "../agent/agent";
import { docName } from "../lib/docs";

/** How much file text a project keeps (it goes into every chat's prompt). */
const FILE_MAX = 300_000;
const ALL_MAX = 1_000_000;
const kb = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
const ago = (t: number) => {
  const d = Math.round((Date.now() - t) / 86400e3);
  return d < 1 ? "Today" : d === 1 ? "Yesterday" : d < 30 ? `${d} days ago` : new Date(t).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
};

export function ProjectsPage() {
  const [over, setOver] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const projects = useApp((s) => s.settings.projects);
  const chats = useApp((s) => s.chats);
  const create = () => openProject(newProject().id);
  const needle = q.trim().toLowerCase();
  // what's been worked on lately first
  const rows = projects
    .map((p) => {
      const mine = chats.filter((c) => c.project === p.id);
      return { p, chats: mine.length, last: Math.max(p.created, ...mine.map((c) => c.updated)) };
    })
    .filter((r) => !needle || r.p.name.toLowerCase().includes(needle) || r.p.instructions.toLowerCase().includes(needle))
    .sort((a, b) => b.last - a.last);
  return (
    <div className="page projects-page">
      <div className="page-in">
        <div className="page-head">
          <h1>Projects</h1>
          <button className="btn primary" onClick={create}>
            <Icon name="plus" size={14} /> New project
          </button>
        </div>
        {projects.length > 0 && <p className="page-lead">Related chats in one place. Every chat in a project gets its instructions and files, and can work in its folder.</p>}
        {projects.length > 4 && (
          <label className="page-filter">
            <Icon name="search" size={15} />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search projects" spellCheck={false} />
          </label>
        )}
        <div className="proj-grid">
          {rows.map(({ p, chats: n, last }) => (
            <button
              key={p.id}
              className={`proj-card ${over === p.id ? "drop" : ""}`}
              onClick={() => openProject(p.id)}
              onDragOver={(e) => {
                if (!e.dataTransfer.types.includes("application/x-pelica-chat")) return;
                e.preventDefault();
                setOver(p.id);
              }}
              onDragLeave={() => setOver(null)}
              onDrop={(e) => {
                setOver(null);
                const id = e.dataTransfer.getData("application/x-pelica-chat");
                if (id) patchChat(id, (c) => ({ ...c, project: p.id })).then(() => toast(`Moved to ${p.name}.`));
              }}
            >
              <span className="proj-card-top">
                <span className="proj-mark">{p.name.trim()[0]?.toUpperCase() || "P"}</span>
                <b>{p.name}</b>
              </span>
              <span className="muted small proj-card-desc">{p.instructions.trim() || "No instructions yet"}</span>
              <span className="muted small proj-card-foot">
                <span>
                  {n} chat{n === 1 ? "" : "s"}
                  {p.files.length ? ` · ${p.files.length} file${p.files.length === 1 ? "" : "s"}` : ""}
                  {p.docs?.length ? ` · ${p.docs.length} document folder${p.docs.length === 1 ? "" : "s"}` : ""}
                </span>
                <span>{ago(last)}</span>
              </span>
            </button>
          ))}
          {needle && !rows.length && <div className="empty-note">No projects match “{q.trim()}”.</div>}
          {!projects.length && (
            <div className="empty-state">
              <span className="empty-icon">
                <Icon name="stack" size={20} />
              </span>
              <b>No projects yet</b>
              <span className="muted small">A project keeps related chats together, with its own instructions, files and documents to answer from.</span>
              <button className="btn primary small" onClick={create}>
                Create a project
              </button>
            </div>
          )}
        </div>
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
    <section className="proj-box">
      <div className="proj-side-head">
        <span>Documents</span>
        <button
          className="ghost-btn"
          title="Add a folder of documents"
          disabled={!bridge}
          onClick={async () => {
            const d = await bridge?.folder.pick();
            if (!d || docs.some((x) => x.toLowerCase() === d.toLowerCase())) return;
            set({ docs: [...docs, d] });
          }}
        >
          <Icon name="plus" size={15} />
        </button>
      </div>
      {docs.map((d) => {
        const v = info[d];
        return (
          <div key={d} className="proj-doc">
            <div className="proj-file">
              <Icon name="folder" size={14} />
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
              {now[d] || (v ? (v.files ? `${v.files} file${v.files === 1 ? "" : "s"} · ${v.passages} passages${v.unreadable.length ? ` · ${v.unreadable.length} couldn't be read` : ""}` : "No documents found in it") : "Reading…")}
            </div>
          </div>
        );
      })}
      {!docs.length && <div className="muted small proj-hint">Folders of PDFs, Word and PowerPoint files and notes. Chats search them and cite what they use. Everything stays on this PC.</div>}
    </section>
  );
}

/** A message that starts a new chat in the project. Without a model loaded
 *  it waits in that chat's message box. */
function ProjectStart({ p }: { p: Project }) {
  const [text, setText] = useState("");
  const enter = useApp((s) => s.settings.sendOnEnter);
  const go = () => {
    const t = text.trim();
    startNewChat(p.folder, p.id);
    if (t && app.get().status?.state === "ready") send(t);
    else if (t) setTimeout(() => window.dispatchEvent(new CustomEvent("pelica:compose", { detail: t })), 60);
  };
  return (
    <div className="proj-start">
      <textarea
        rows={1}
        value={text}
        placeholder={`Start a chat in ${p.name}`}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (enter ? !e.shiftKey : e.ctrlKey)) {
            e.preventDefault();
            go();
          }
        }}
      />
      <button className={`send-btn ${text.trim() ? "ready" : ""}`} title="Start the chat" onClick={go}>
        <Icon name="send" size={16} />
      </button>
    </div>
  );
}

export function ProjectPage() {
  const id = useApp((s) => s.projectId);
  const p = useApp((s) => s.settings.projects.find((x) => x.id === s.projectId));
  const all = useApp((s) => s.chats);
  const chats = useMemo(() => all.filter((c) => c.project === id), [all, id]);
  const [instr, setInstr] = useState<string | null>(null);
  const [sure, setSure] = useState(false);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const more = useRef<HTMLButtonElement>(null);
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
  const items: MenuItem[] = [
    ...(p.folder && bridge ? [{ label: "Show folder", icon: "folder", on: () => bridge!.sys.open(p.folder!, "").catch((e: any) => toast(e.message)) }] : []),
    { label: "Delete project", icon: "trash", danger: true, on: () => setSure(true) },
  ];
  return (
    <div className="page project-page">
      <div className="page-in">
        <button className="link-btn proj-back" onClick={() => app.set({ view: "projects" })}>
          <Icon name="back" size={13} /> All projects
        </button>
        <div className="page-head">
          <input className="proj-name" value={p.name} onChange={(e) => set({ name: e.target.value })} onBlur={(e) => !e.target.value.trim() && set({ name: "Untitled project" })} spellCheck={false} aria-label="Project name" />
          <button
            ref={more}
            className={`ghost-btn ${menu ? "menu-open" : ""}`}
            title="More"
            aria-haspopup="menu"
            onClick={() => {
              const r = more.current!.getBoundingClientRect();
              setMenu(menu ? null : { x: r.right - 200, y: r.bottom + 4 });
            }}
          >
            <Icon name="more" size={17} />
          </button>
          {menu && <Menu at={menu} ignore={more} minWidth={200} close={() => setMenu(null)} items={items} />}
        </div>
        {sure && (
          <div className="proj-sure">
            <span>Delete this project? Its chats stay, outside any project.</span>
            <button className="btn small danger" onClick={() => deleteProject(p.id)}>
              Delete project
            </button>
            <button className="btn small ghost" onClick={() => setSure(false)}>
              Cancel
            </button>
          </div>
        )}
        <div className="proj-cols">
          <div className="proj-main">
            <ProjectStart p={p} />
            {chats.length > 0 && (
              <>
                <div className="proj-label">
                  {chats.length} chat{chats.length === 1 ? "" : "s"}
                </div>
                <div className="proj-rows">
                  {chats.map((c) => (
                    <div key={c.id} className="proj-row">
                      <button className="proj-row-main" onClick={() => openChat(c.id)}>
                        <span className="proj-row-title">{c.title}</span>
                        <span className="muted small">{ago(c.updated)}</span>
                      </button>
                      <button className="ghost-btn proj-row-out" title="Take out of this project (the chat stays)" onClick={() => patchChat(c.id, (x) => ({ ...x, project: undefined }))}>
                        <Icon name="x" size={13} />
                      </button>
                    </div>
                  ))}
                </div>
              </>
            )}
            {!chats.length && <div className="muted small proj-none">No chats here yet. Start one above, or move a chat here from its menu (Move to project).</div>}
          </div>
          <aside className="proj-side">
            <section className="proj-box">
              <div className="proj-side-head">
                <span>Instructions</span>
              </div>
              <textarea
                className="proj-instr"
                value={instr ?? p.instructions}
                placeholder="What should every chat in this project know or do? E.g. “This is my Rust game engine. Answer briefly and show code.”"
                onChange={(e) => setInstr(e.target.value)}
                onBlur={() => {
                  if (instr !== null) set({ instructions: instr });
                  setInstr(null);
                }}
              />
            </section>
            <section className="proj-box">
              <div className="proj-side-head">
                <span>
                  Files <span className="muted small" title="Every chat in the project reads them first">{p.files.length ? `${kb(used)} · about ${Math.max(1, Math.round(used / 3500))}K tokens` : ""}</span>
                </span>
                <button className="ghost-btn" title="Add files" onClick={() => pick.current?.click()}>
                  <Icon name="plus" size={15} />
                </button>
              </div>
              <input ref={pick} type="file" multiple hidden onChange={(e) => (addFiles(e.target.files), (e.target.value = ""))} />
              {p.files.map((f) => (
                <div key={f.name} className="proj-file">
                  <Icon name="file" size={14} />
                  <span className="proj-file-name" title={f.name}>
                    {f.name}
                  </span>
                  <span className="muted small">{kb(f.text.length)}</span>
                  <button className="ghost-btn" title="Remove from the project" onClick={() => set({ files: projectOf(p.id)!.files.filter((x) => x.name !== f.name) })}>
                    <Icon name="x" size={12} />
                  </button>
                </div>
              ))}
              {!p.files.length && <div className="muted small proj-hint">Text files (notes, docs, code) every chat in the project can read.</div>}
              {used > 100_000 && <div className="warn-note small">That's a lot for a local model: files that don't fit in about 40% of its context are left out (by name) in its chats.</div>}
            </section>
            <DocsSection p={p} set={set} />
            <section className="proj-box">
              <div className="proj-side-head">
                <span>Folder</span>
                <button
                  className="ghost-btn"
                  title={p.folder ? "Change folder" : "Choose a folder"}
                  disabled={!bridge}
                  onClick={async () => {
                    const d = await bridge?.folder.pick();
                    if (d) set({ folder: d });
                  }}
                >
                  <Icon name={p.folder ? "edit" : "plus"} size={p.folder ? 14 : 15} />
                </button>
              </div>
              {p.folder ? (
                <div className="proj-file">
                  <Icon name="folder" size={14} />
                  <span className="proj-file-name" title={p.folder}>
                    {docName(p.folder)}
                  </span>
                  <button className="ghost-btn" title="No folder" onClick={() => set({ folder: null })}>
                    <Icon name="x" size={12} />
                  </button>
                </div>
              ) : (
                <div className="muted small proj-hint">New chats in the project work in this folder: they can read and change its files and run commands there.</div>
              )}
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}
