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
// 2. compiles the uninstaller into it and packs it into payload.zip,
// 3. compiles the installer around the payload, and
// 4. writes release/latest.json beside it: the update feed the app reads
//    (Settings → Updates; electron/updates.ts). Attach both files to
//    a GitHub release (gh release create vX.Y.Z <installer> release/latest.json):
//    the app reads the newest release's latest.json by default.
//
// Signing (optional): with a code-signing certificate, the exes are signed
// (Windows then names the publisher instead of warning about an unknown one)
// and the app only installs updates signed by the same publisher.
//   PELICA_SIGN_PFX=<file.pfx>  PELICA_SIGN_PASSWORD=<its password>, or
//   PELICA_SIGN_THUMBPRINT=<a certificate in your personal store>
//   PELICA_SIGN_TIMESTAMP=<timestamp server> (default DigiCert's; none: no timestamp)
//   PELICA_SIGN_TEST=1: also accept a certificate Windows doesn't trust (one
//   you made yourself, to try this out)
// PowerShell's Set-AuthenticodeSignature does it; nothing else is needed.
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
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
// every installer carries release notes for its version (src/changelog.json,
// shown in the app as What's new); bump.mjs stamps them before this runs
const notes = JSON.parse(fs.readFileSync(path.join(desktop, "src", "changelog.json"), "utf8"));
if (notes[0]?.version !== version || !notes[0]?.notes?.length) {
  console.error(`No release notes for ${version}: add a "next" entry at the top of src/changelog.json (see installer/bump.mjs), then run npm run installer again.`);
  process.exit(1);
}
const year = new Date().getFullYear();

function run(exe, args, what) {
  const r = spawnSync(exe, args, { stdio: "inherit", windowsHide: true });
  if (r.status !== 0) throw new Error(`${what} failed (${r.error?.message || "exit " + r.status})`);
}

const signing = process.env.PELICA_SIGN_PFX || process.env.PELICA_SIGN_THUMBPRINT;

/** Signs exes with the configured certificate (see the top), SHA-256 and timestamped. */
function sign(files) {
  if (!signing) return;
  const ps = `
$ErrorActionPreference = 'Stop'
if ($env:PELICA_SIGN_PFX) {
  $cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($env:PELICA_SIGN_PFX, $env:PELICA_SIGN_PASSWORD)
} else {
  $cert = Get-Item ("Cert:\\CurrentUser\\My\\" + $env:PELICA_SIGN_THUMBPRINT)
}
foreach ($f in $env:PELICA_SIGN_FILES.Split('|')) {
  if ($env:PELICA_SIGN_TS -eq 'none') { $r = Set-AuthenticodeSignature -LiteralPath $f -Certificate $cert -HashAlgorithm SHA256 }
  else { $r = Set-AuthenticodeSignature -LiteralPath $f -Certificate $cert -HashAlgorithm SHA256 -TimestampServer $env:PELICA_SIGN_TS }
  if ($r.Status -ne 'Valid' -and !$env:PELICA_SIGN_TEST) { throw ($f + ': ' + $r.Status + ' ' + $r.StatusMessage) }
  Write-Output ("signed " + (Split-Path $f -Leaf))
}`;
  const r = spawnSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", ps], {
    stdio: "inherit",
    windowsHide: true,
    env: { ...process.env, PELICA_SIGN_FILES: files.join("|"), PELICA_SIGN_TS: process.env.PELICA_SIGN_TIMESTAMP || "http://timestamp.digicert.com" },
  });
  if (r.status !== 0) throw new Error("signing failed");
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
// a fresh layout each time (an old one kept every earlier build's UI bundles)
fs.rmSync(stage, { recursive: true, force: true });
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
// Pelica's license (MIT) beside Electron's LICENSE and Chromium's LICENSES.chromium.html
fs.copyFileSync(path.resolve(desktop, "../LICENSE"), path.join(stage, "LICENSE.Pelica.txt"));

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
sign([path.join(stage, "Pelica.exe"), path.join(stage, "resources", "pelica.exe"), path.join(stage, "Uninstall Pelica.exe")]);

console.log("payload");
run(tool, ["zip", stage, path.join(work, "payload.zip")], "packing the app");

console.log("installer");
const out = path.join(release, `Pelica-Setup-${version}.exe`);
run(csc, [...cscUi, `/resource:${path.join(work, "payload.zip")},payload.zip`, `/out:${out}`, path.join(here, "setup.cs"), path.join(work, "version.cs")], "compiling the installer");
sign([out]);

// the update feed for this version
const sha256 = crypto.createHash("sha256").update(fs.readFileSync(out)).digest("hex");
const feed = path.join(release, "latest.json");
fs.writeFileSync(feed, JSON.stringify({ version, url: path.basename(out), sha256, date: notes[0].date, title: notes[0].title, notes: notes[0].notes }, null, 2) + "\n");
console.log(`\n${out}\n${(fs.statSync(out).size / 1048576).toFixed(1)} MB (installs ${(size / 1048576).toFixed(0)} MB), ${signing ? "signed" : "not signed (see the top of installer/make.mjs)"}\n${feed}`);
