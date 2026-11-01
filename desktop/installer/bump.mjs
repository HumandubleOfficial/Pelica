// Stamps a new version from the release notes. Every update gets notes first:
// add an entry at the top of src/changelog.json with "version": "next" (and
// "bump": "minor" for a bigger release; patch otherwise). This script (run by
// `npm run installer`, or alone with `npm run bump`) then
//   - picks the next version after package.json's (0.3.0 -> 0.3.1, or 0.4.0),
//   - writes it into the entry (with today's date), package.json and the
//     engine's Cargo.toml.
// With no "next" entry it changes nothing; make.mjs then refuses to build an
// installer whose version has no notes.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const logPath = path.join(desktop, "src", "changelog.json");
const pkgPath = path.join(desktop, "package.json");
const cargoPath = path.join(desktop, "..", "Cargo.toml");

const log = JSON.parse(fs.readFileSync(logPath, "utf8"));
const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
const top = log[0];

if (!top || top.version !== "next") {
  console.log(`version ${pkg.version} (no new release notes)`);
  process.exit(0);
}
if (!Array.isArray(top.notes) || !top.notes.length) {
  console.error('the "next" entry in src/changelog.json has no notes');
  process.exit(1);
}
const [ma, mi, pa] = pkg.version.split(".").map(Number);
const next = top.bump === "major" ? `${ma + 1}.0.0` : top.bump === "minor" ? `${ma}.${mi + 1}.0` : `${ma}.${mi}.${pa + 1}`;
const d = new Date();
const date = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

log[0] = { version: next, date, title: top.title || "", notes: top.notes };
fs.writeFileSync(logPath, JSON.stringify(log, null, 2) + "\n");

// (keep the file's own formatting: only the version line changes)
const pkgText = fs.readFileSync(pkgPath, "utf8").replace(/("version":\s*")[^"]+(")/, `$1${next}$2`);
fs.writeFileSync(pkgPath, pkgText);
if (fs.existsSync(cargoPath)) {
  const c = fs.readFileSync(cargoPath, "utf8").replace(/^version = "[^"]+"/m, `version = "${next}"`);
  fs.writeFileSync(cargoPath, c);
}
console.log(`version ${pkg.version} -> ${next}`);
