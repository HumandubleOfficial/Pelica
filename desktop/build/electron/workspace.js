"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.inside = inside;
exports.listFiles = listFiles;
exports.readFile = readFile;
exports.writeFile = writeFile;
exports.editFile = editFile;
exports.searchFiles = searchFiles;
exports.findFiles = findFiles;
// The models' file tools, confined to a workspace folder the user picked:
// every path is resolved (symbolic links included) and refused if it lands
// outside the folder.
const node_fs_1 = __importDefault(require("node:fs"));
const node_path_1 = __importDefault(require("node:path"));
const SKIP = new Set(["node_modules", ".git", "target", "dist", "build", "__pycache__", ".venv", "venv", ".next", ".cache", ".idea", ".vs"]);
function realish(p) {
    // the real path of the deepest part that exists, plus the rest
    let cur = p;
    const rest = [];
    for (;;) {
        try {
            return node_path_1.default.join(node_fs_1.default.realpathSync.native(cur), ...rest.reverse());
        }
        catch {
            const up = node_path_1.default.dirname(cur);
            if (up === cur)
                return p;
            rest.push(node_path_1.default.basename(cur));
            cur = up;
        }
    }
}
/** The absolute path of `rel` inside `root`; throws if it's outside. */
function inside(root, rel) {
    const r = realish(node_path_1.default.resolve(root));
    const p = realish(node_path_1.default.resolve(r, rel || "."));
    const a = p.toLowerCase();
    const b = r.toLowerCase();
    if (a !== b && !a.startsWith(b.endsWith(node_path_1.default.sep) ? b : b + node_path_1.default.sep))
        throw new Error(`${rel} is outside the workspace folder (${root})`);
    return p;
}
const relOf = (root, p) => node_path_1.default.relative(realish(node_path_1.default.resolve(root)), p).split(node_path_1.default.sep).join("/") || ".";
function listFiles(root, rel = ".", depth = 2) {
    const base = inside(root, rel);
    const out = [];
    const walk = (d, lvl, pre) => {
        let ents = [];
        try {
            ents = node_fs_1.default.readdirSync(d, { withFileTypes: true });
        }
        catch {
            return;
        }
        ents.sort((x, y) => Number(y.isDirectory()) - Number(x.isDirectory()) || x.name.localeCompare(y.name));
        for (const e of ents) {
            if (out.length >= 500)
                return;
            if (e.isDirectory()) {
                if (SKIP.has(e.name)) {
                    out.push(`${pre}${e.name}/ (skipped)`);
                    continue;
                }
                out.push(`${pre}${e.name}/`);
                if (lvl < depth)
                    walk(node_path_1.default.join(d, e.name), lvl + 1, pre + "  ");
            }
            else {
                let size = 0;
                try {
                    size = node_fs_1.default.statSync(node_path_1.default.join(d, e.name)).size;
                }
                catch { }
                out.push(`${pre}${e.name} (${size < 1024 ? size + " B" : (size / 1024).toFixed(1) + " KB"})`);
            }
        }
    };
    walk(base, 1, "");
    if (out.length >= 500)
        out.push("... (more entries not shown)");
    return `${relOf(root, base)}/\n` + (out.length ? out.join("\n") : "(empty)");
}
function isBinary(b) {
    const n = Math.min(b.length, 8000);
    for (let i = 0; i < n; i++)
        if (b[i] === 0)
            return true;
    return false;
}
function readFile(root, rel, offset = 1, limit = 500) {
    const p = inside(root, rel);
    const st = node_fs_1.default.statSync(p);
    if (st.isDirectory())
        throw new Error(`${rel} is a folder; use list_files`);
    if (st.size > 5_000_000)
        throw new Error(`${rel} is ${(st.size / 1e6).toFixed(1)} MB, too large to read`);
    const b = node_fs_1.default.readFileSync(p);
    if (isBinary(b))
        throw new Error(`${rel} is a binary file`);
    const all = b.toString("utf8").split(/\r?\n/);
    const start = Math.max(1, offset);
    const slice = all.slice(start - 1, start - 1 + Math.max(1, limit));
    const text = slice.map((l, i) => `${String(start + i).padStart(5)}  ${l.length > 2000 ? l.slice(0, 2000) + "…" : l}`).join("\n");
    const more = start - 1 + slice.length < all.length ? `\n[lines ${start}-${start - 1 + slice.length} of ${all.length}; read more with offset]` : "";
    return { text: text + more, lines: all.length, path: relOf(root, p) };
}
function writeFile(root, rel, content) {
    const p = inside(root, rel);
    let before = null;
    try {
        before = node_fs_1.default.readFileSync(p, "utf8");
    }
    catch { }
    node_fs_1.default.mkdirSync(node_path_1.default.dirname(p), { recursive: true });
    node_fs_1.default.writeFileSync(p, content, "utf8");
    return { path: relOf(root, p), before, after: content };
}
function editFile(root, rel, oldS, newS, all = false) {
    const p = inside(root, rel);
    const before = node_fs_1.default.readFileSync(p, "utf8");
    // the model may write \n where the file has \r\n
    const crlf = before.includes("\r\n") && !oldS.includes("\r\n");
    const o = crlf ? oldS.replace(/\n/g, "\r\n") : oldS;
    const n = crlf ? newS.replace(/\n/g, "\r\n") : newS;
    if (!o)
        throw new Error("old_string is empty; use write_file to create a file");
    const count = before.split(o).length - 1;
    if (count === 0)
        throw new Error(`old_string was not found in ${rel}. Read the file again and copy the text exactly, including indentation.`);
    if (count > 1 && !all)
        throw new Error(`old_string appears ${count} times in ${rel}; add more surrounding lines to make it unique, or set replace_all`);
    const after = all ? before.split(o).join(n) : before.replace(o, () => n);
    node_fs_1.default.writeFileSync(p, after, "utf8");
    return { path: relOf(root, p), before, after, count: all ? count : 1 };
}
function globRe(g) {
    const s = g
        .replace(/[.+^${}()|[\]\\]/g, "\\$&")
        .replace(/\*\*\//g, "\u0000")
        .replace(/\*/g, "[^/]*")
        .replace(/\?/g, "[^/]")
        .replace(/\u0000/g, "(?:.*/)?");
    return new RegExp(`(^|/)${s}$`, "i");
}
function searchFiles(root, pattern, rel = ".", glob) {
    const base = inside(root, rel);
    let re;
    try {
        re = new RegExp(pattern, "i");
    }
    catch {
        re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
    }
    const gre = glob ? globRe(glob) : null;
    const out = [];
    let files = 0;
    const walk = (d) => {
        let ents = [];
        try {
            ents = node_fs_1.default.readdirSync(d, { withFileTypes: true });
        }
        catch {
            return;
        }
        for (const e of ents) {
            if (out.length >= 200 || files > 5000)
                return;
            const p = node_path_1.default.join(d, e.name);
            if (e.isDirectory()) {
                if (!SKIP.has(e.name))
                    walk(p);
                continue;
            }
            const r = relOf(root, p);
            if (gre && !gre.test(r))
                continue;
            files++;
            let b;
            try {
                if (node_fs_1.default.statSync(p).size > 2_000_000)
                    continue;
                b = node_fs_1.default.readFileSync(p);
            }
            catch {
                continue;
            }
            if (isBinary(b))
                continue;
            const lines = b.toString("utf8").split(/\r?\n/);
            for (let i = 0; i < lines.length && out.length < 200; i++) {
                if (re.test(lines[i]))
                    out.push(`${r}:${i + 1}: ${lines[i].trim().slice(0, 240)}`);
            }
        }
    };
    walk(base);
    return out.length ? out.join("\n") + (out.length >= 200 ? "\n[first 200 matches]" : "") : "No matches.";
}
/** Files under `root` whose path matches the glob (for CQS and the UI). */
function findFiles(root, glob, max = 2000) {
    const base = inside(root, ".");
    const gre = globRe(glob);
    const out = [];
    const walk = (d) => {
        let ents = [];
        try {
            ents = node_fs_1.default.readdirSync(d, { withFileTypes: true });
        }
        catch {
            return;
        }
        for (const e of ents) {
            if (out.length >= max)
                return;
            const p = node_path_1.default.join(d, e.name);
            if (e.isDirectory()) {
                if (!SKIP.has(e.name))
                    walk(p);
            }
            else if (gre.test(relOf(root, p)))
                out.push(relOf(root, p));
        }
    };
    walk(base);
    return out;
}
