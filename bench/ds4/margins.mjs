// Read-ahead guesses vs router margin (PELICA_MARGINS dumps): hit rate of
// uncached predictions by margin, and what threshold rules would read.
// usage: node margins.mjs marg_a.csv [marg_b.csv ...]
import fs from "fs";

const rows = process.argv.slice(2).flatMap((f) =>
  fs.readFileSync(f, "utf8").trim().split("\n").map((l) => l.split(",").map(Number)),
);
const bins = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, Infinity];
for (const kind of [0, 1]) {
  console.log(kind ? "late" : "early", "(uncached only): hits/count by rank x margin bin");
  for (let r = 0; r < 6; r++) {
    const cells = bins.slice(0, -1).map((lo, i) => {
      const s = rows.filter((x) => x[0] === kind && x[1] === r && !x[4] && x[2] >= lo && x[2] < bins[i + 1]);
      const h = s.filter((x) => x[3]).length;
      return s.length ? `${((h / s.length) * 100).toFixed(0).padStart(3)}%/${String(s.length).padEnd(4)}` : "   -     ";
    });
    console.log(`  rank ${r}: ${cells.join(" ")}`);
  }
  console.log(`  bins: ${bins.slice(0, -1).join(" ")}`);
}
// rules: read rank r if r < kmin, or (r < kmax and margin >= t)
const ntok = rows.filter((x) => x[0] === 0 && x[1] === 0).length; // layer-tokens
for (const kind of [0, 1]) {
  const base = kind ? 3 : 4;
  console.log(`${kind ? "late" : "early"} rules (reads, hits, wasted per layer-token; now: top ${base})`);
  const rule = (kmin, kmax, t) => {
    const s = rows.filter((x) => x[0] === kind && !x[4] && (x[1] < kmin || (x[1] < kmax && x[2] >= t)));
    const h = s.filter((x) => x[3]).length;
    return [s.length / ntok, h / ntok, (s.length - h) / ntok];
  };
  const show = (name, v) => console.log(`  ${name.padEnd(26)} reads ${v[0].toFixed(3)} hits ${v[1].toFixed(3)} wasted ${v[2].toFixed(3)}`);
  for (let k = 2; k <= 6; k++) show(`top ${k}`, rule(k, k, 0));
  for (const kmin of [1, 2, 3])
    for (const t of [0.01, 0.02, 0.04, 0.08]) show(`top ${kmin} + margin >= ${t}`, rule(kmin, 6, t));
}
