// Replays decode expert picks (PELICA_PICKS traces) against cache policies and
// counts drive reads per token. Models the engine's two tiers: VRAM (moved by
// rebalance from heat) and the RAM cache (coldest-by-heat eviction), with dedup.
//
// usage: node cachesim.mjs usage.bak picks_a.bin [picks_b.bin ...]
import fs from "fs";

const NE = 256, V = +(process.env.V || 443), R = +(process.env.R || 1548);
const [usageFile, ...traces] = process.argv.slice(2);
const ub = fs.readFileSync(usageFile);
const usage = new Float64Array(ub.length / 4).map((_, i) => ub.readUInt32LE(i * 4));
const NK = usage.length;

function load(f) {
  const b = fs.readFileSync(f);
  const toks = [];
  let cur = [];
  for (let i = 0; i < b.length; i += 4) {
    const k = b.readUInt32LE(i);
    if (k === 0xffffffff) {
      if (cur.length) toks.push(cur);
      cur = [];
    } else cur.push(k);
  }
  if (cur.length) toks.push(cur);
  return toks;
}

// p: { hl, hl2, mix, every, step, margin, demote, ramScore }
function sim(toks, p) {
  const heat = new Float64Array(NK), last = new Float64Array(NK);
  const heat2 = new Float64Array(NK);
  let tick = 0;
  const d1 = Math.pow(0.5, 1 / p.hl), d2 = p.hl2 ? Math.pow(0.5, 1 / p.hl2) : 0;
  const eff = (k) => heat[k] * Math.pow(d1, tick - last[k]) + (p.mix ? p.mix * heat2[k] * Math.pow(d2, tick - last[k]) : 0);
  const bump = (k, by) => {
    const dt = tick - last[k];
    heat[k] = heat[k] * Math.pow(d1, dt) + by;
    if (p.mix) heat2[k] = heat2[k] * Math.pow(d2, dt) + by;
    last[k] = tick;
  };
  const order = [...Array(NK).keys()].sort((a, b) => usage[b] - usage[a]);
  const inV = new Set(order.slice(0, V));
  const inR = new Set(order.slice(V, V + R));
  let reads = 0;
  const ramInsert = (k) => {
    if (inR.size >= R) {
      let worst = -1, wh = Infinity;
      for (const x of inR) {
        const h = eff(x);
        if (h < wh) { wh = h; worst = x; }
      }
      inR.delete(worst);
    }
    inR.add(k);
  };
  let since = 0, moved = 0;
  for (const t of toks) {
    for (const k of t) {
      if (!inV.has(k) && !inR.has(k)) {
        reads++;
        ramInsert(k);
      }
    }
    for (const k of t) bump(k, 1);
    tick++;
    if (++since >= p.every) {
      since = 0;
      // rebalance: hottest V not in VRAM, in place of the coldest unwanted
      const all = [...Array(NK).keys()].map((k) => [k, eff(k)]).sort((a, b) => b[1] - a[1]);
      const want = new Set(all.slice(0, V).map((x) => x[0]));
      const add = all.slice(0, V).filter((x) => !inV.has(x[0])).slice(0, p.step);
      const free = [...inV].filter((k) => !want.has(k)).map((k) => [k, eff(k)]).sort((a, b) => a[1] - b[1]);
      for (let i = 0; i < add.length && i < free.length; i++) {
        const [k, h] = add[i], [old, oh] = free[i];
        if (p.margin && h < oh * (1 + p.margin) + (p.marginAbs || 0)) continue;
        if (!inR.has(k)) reads++;
        inR.delete(k);
        inV.delete(old);
        inV.add(k);
        moved++;
        if (p.demote) ramInsert(old);
      }
    }
  }
  return { reads, moved };
}

// Belady on one combined tier of V+R (a lower bound for any policy)
function opt(toks) {
  const seq = toks.flat();
  const next = new Int32Array(seq.length);
  const seen = new Map();
  for (let i = seq.length - 1; i >= 0; i--) {
    next[i] = seen.has(seq[i]) ? seen.get(seq[i]) : 1e9;
    seen.set(seq[i], i);
  }
  const order = [...Array(NK).keys()].sort((a, b) => usage[b] - usage[a]);
  const cache = new Map(order.slice(0, V + R).map((k) => [k, 1e9]));
  // next use of the warm set
  const firstUse = new Map();
  for (let i = seq.length - 1; i >= 0; i--) firstUse.set(seq[i], i);
  for (const k of cache.keys()) cache.set(k, firstUse.has(k) ? firstUse.get(k) : 1e9 + k);
  let reads = 0;
  for (let i = 0; i < seq.length; i++) {
    const k = seq[i];
    if (!cache.has(k)) {
      reads++;
      let worst = -1, wn = -1;
      for (const [x, n] of cache) if (n > wn) { wn = n; worst = x; }
      cache.delete(worst);
    }
    cache.set(k, next[i]);
  }
  return reads;
}

const sets = traces.map(load);
const configs = [
  ["now (hl16, every 4 step 8)", { hl: 16, every: 4, step: 8 }],
  ["hl 32", { hl: 32, every: 4, step: 8 }],
  ["hl 64", { hl: 64, every: 4, step: 8 }],
  ["hl 8", { hl: 8, every: 4, step: 8 }],
  ["demote", { hl: 16, every: 4, step: 8, demote: 1 }],
  ["margin 25%", { hl: 16, every: 4, step: 8, margin: 0.25 }],
  ["margin 50%", { hl: 16, every: 4, step: 8, margin: 0.5 }],
  ["margin +1", { hl: 16, every: 4, step: 8, margin: 1e-9, marginAbs: 1 }],
  ["step 2", { hl: 16, every: 4, step: 2 }],
  ["no rebalance", { hl: 16, every: 1e9, step: 0 }],
  ["mix hl16 + 0.5*hl256", { hl: 16, hl2: 256, mix: 0.5, every: 4, step: 8 }],
  ["mix hl16 + 0.25*hl128", { hl: 16, hl2: 128, mix: 0.25, every: 4, step: 8 }],
  ["hl 32 + demote", { hl: 32, every: 4, step: 8, demote: 1 }],
];
for (const [i, toks] of sets.entries()) {
  console.log(`${traces[i]}: ${toks.length} tokens, ${toks[0]?.length} picks/token, OPT ${(opt(toks) / toks.length).toFixed(1)} reads/token`);
  for (const [name, p] of configs) {
    const r = sim(toks, p);
    console.log(`  ${name.padEnd(28)} ${(r.reads / toks.length).toFixed(1)} reads/token, moved ${r.moved}`);
  }
}
