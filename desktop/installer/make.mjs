// Builds the Windows installer, release/Pelica-Setup-<version>.exe, with what
// is already on this PC: the Electron runtime from node_modules, the engine
// (target/release/pelica.exe), the built UI, and the C# compiler that ships
// with Windows. Nothing is downloaded.
//
//   npm run installer        (builds the app first, then this)
//
// 1. lays the app out as it will be installed (release/stage/Pelica):
//    Pelica.exe (electron.exe, renamed, with Pelica's icon and version info),
//    resources/app (the main process and UI), resources/pelica.exe (the engine)
// 2. compiles the uninstaller into it, packs it into payload.zip, and
// 3. compiles the installer around the payload.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const desktop = path.resolve(here, "..");
const engine = path.resolve(desktop, "../target/release/pelica.exe");
const electronDist = path.join(desktop, "node_modules/electron/dist");
const release = path.join(desktop, "release");
const stage = path.join(release, "stage", "Pelica");
const work = path.join(release, "work");
const csc = path.join(process.env.WINDIR || "C:\\Windows", "Microsoft.NET/Framework64/v4.0.30319/csc.exe");

const pkg = JSON.parse(fs.readFileSync(path.join(desktop, "package.json"), "utf8"));
const version = pkg.version;
const year = new Date().getFullYear();

function run(exe, args, what) {
  const r = spawnSync(exe, args, { stdio: "inherit", windowsHide: true });
  if (r.status !== 0) throw new Error(`${what} failed (${r.error?.message || "exit " + r.status})`);
}

function need(p, hint) {
  if (!fs.existsSync(p)) throw new Error(`missing ${p}${hint ? ` (${hint})` : ""}`);
}

function bytes(dir) {
  let n = 0;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) n += e.isDirectory() ? bytes(path.join(dir, e.name)) : fs.statSync(path.join(dir, e.name)).size;
  return n;
}

const cscCommon = ["/nologo", "/optimize+", "/platform:x64", "/nowarn:162,429", "/r:System.IO.Compression.dll"];
const cscUi = [...cscCommon, "/target:winexe", "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll", "/r:Microsoft.VisualBasic.dll", `/win32icon:${path.join(work, "pelica.ico")}`, `/win32manifest:${path.join(here, "setup.manifest")}`, `/resource:${path.join(work, "logo.png")},logo.png`];

need(engine, "build it with cargo build --release");
need(path.join(desktop, "build/electron/main.js"), "npm run build");
need(path.join(desktop, "dist/index.html"), "npm run build");
need(path.join(electronDist, "electron.exe"), "npm install");
need(csc, ".NET Framework 4 is part of Windows 10 and 11");
fs.mkdirSync(stage, { recursive: true });
fs.mkdirSync(work, { recursive: true });

console.log(`Pelica ${version}: laying out the app`);
// the runtime (electron.exe becomes Pelica.exe; Electron's default app stays out)
fs.cpSync(electronDist, stage, { recursive: true, force: true, filter: (src) => !/[\\/](electron\.exe|default_app\.asar)$/i.test(src) });
fs.copyFileSync(path.join(electronDist, "electron.exe"), path.join(stage, "Pelica.exe"));
// the app: main process, UI, and the engine beside them
const app = path.join(stage, "resources", "app");
fs.mkdirSync(app, { recursive: true });
fs.writeFileSync(path.join(app, "package.json"), JSON.stringify({ name: pkg.name, productName: pkg.productName, version, description: pkg.description, main: pkg.main }, null, 2));
fs.cpSync(path.join(desktop, "build/electron"), path.join(app, "build/electron"), { recursive: true, force: true });
fs.cpSync(path.join(desktop, "dist"), path.join(app, "dist"), { recursive: true, force: true });
fs.copyFileSync(engine, path.join(stage, "resources", "pelica.exe"));

console.log("icons");
run(path.join(electronDist, "electron.exe"), [path.join(here, "icons.cjs"), path.join(desktop, "public/logo.svg"), work], "rendering the icons");

console.log("build tool");
run(csc, [...cscCommon, "/target:exe", `/out:${path.join(work, "buildtool.exe")}`, path.join(here, "buildtool.cs")], "compiling the build tool");
const tool = path.join(work, "buildtool.exe");
const copyright = `© ${year} Pelica`;
run(tool, ["stamp", path.join(stage, "Pelica.exe"), path.join(work, "pelica.ico"), version, "Pelica", "Pelica", "Pelica", copyright, "Pelica.exe"], "stamping Pelica.exe");
run(tool, ["stamp", path.join(stage, "resources", "pelica.exe"), path.join(work, "pelica.ico"), version, "Pelica", "Pelica engine", "Pelica", copyright, "pelica.exe"], "stamping the engine");

const size = bytes(stage);
fs.writeFileSync(path.join(work, "version.cs"), `// written by installer/make.mjs\nstatic partial class Info\n{\n    public const string Version = "${version}";\n    public const long Bytes = ${size}L;\n}\n`);

console.log("uninstaller");
run(csc, [...cscUi, "/define:UNINSTALL", `/out:${path.join(stage, "Uninstall Pelica.exe")}`, path.join(here, "setup.cs"), path.join(work, "version.cs")], "compiling the uninstaller");

console.log("payload");
run(tool, ["zip", stage, path.join(work, "payload.zip")], "packing the app");

console.log("installer");
const out = path.join(release, `Pelica-Setup-${version}.exe`);
run(csc, [...cscUi, `/resource:${path.join(work, "payload.zip")},payload.zip`, `/out:${out}`, path.join(here, "setup.cs"), path.join(work, "version.cs")], "compiling the installer");
console.log(`\n${out}\n${(fs.statSync(out).size / 1048576).toFixed(1)} MB (installs ${(size / 1048576).toFixed(0)} MB)`);
