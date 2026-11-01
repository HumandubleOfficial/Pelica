// Line diffs for the file cards: an LCS over lines, shown as hunks with
// a few lines of context.

export interface DiffLine {
  kind: " " | "+" | "-";
  text: string;
  a?: number;
  b?: number;
}

/** A text's lines: an empty text has none, and a final newline doesn't start another. */
function lines(t: string): string[] {
  if (!t) return [];
  const l = t.split(/\r?\n/);
  if (l[l.length - 1] === "") l.pop();
  return l;
}

export function diffLines(before: string, after: string): DiffLine[] {
  const A = lines(before);
  const B = lines(after);
  // trim the common ends first: edits are usually small
  let s = 0;
  while (s < A.length && s < B.length && A[s] === B[s]) s++;
  let e = 0;
  while (e < A.length - s && e < B.length - s && A[A.length - 1 - e] === B[B.length - 1 - e]) e++;
  const a = A.slice(s, A.length - e);
  const b = B.slice(s, B.length - e);
  const out: DiffLine[] = [];
  for (let i = 0; i < s; i++) out.push({ kind: " ", text: A[i], a: i + 1, b: i + 1 });
  if (a.length * b.length > 4_000_000) {
    // too big for the table: all out, all in
    a.forEach((t, i) => out.push({ kind: "-", text: t, a: s + i + 1 }));
    b.forEach((t, i) => out.push({ kind: "+", text: t, b: s + i + 1 }));
  } else {
    const n = a.length;
    const m = b.length;
    const L = new Array(n + 1);
    for (let i = 0; i <= n; i++) L[i] = new Uint32Array(m + 1);
    for (let i = n - 1; i >= 0; i--) for (let k = m - 1; k >= 0; k--) L[i][k] = a[i] === b[k] ? L[i + 1][k + 1] + 1 : Math.max(L[i + 1][k], L[i][k + 1]);
    let i = 0;
    let k = 0;
    while (i < n || k < m) {
      if (i < n && k < m && a[i] === b[k]) {
        out.push({ kind: " ", text: a[i], a: s + i + 1, b: s + k + 1 });
        i++;
        k++;
      } else if (i < n && (k >= m || L[i + 1][k] >= L[i][k + 1])) {
        // (removed lines before added ones where either order works)
        out.push({ kind: "-", text: a[i], a: s + i + 1 });
        i++;
      } else {
        out.push({ kind: "+", text: b[k], b: s + k + 1 });
        k++;
      }
    }
  }
  for (let i = 0; i < e; i++) out.push({ kind: " ", text: A[A.length - e + i], a: A.length - e + i + 1, b: B.length - e + i + 1 });
  return out;
}

/** The changed lines with `ctx` lines around them; null marks a gap. */
export function hunks(lines: DiffLine[], ctx = 3): (DiffLine | null)[] {
  const keep = new Array(lines.length).fill(false);
  lines.forEach((l, i) => {
    if (l.kind !== " ") for (let j = Math.max(0, i - ctx); j <= Math.min(lines.length - 1, i + ctx); j++) keep[j] = true;
  });
  const out: (DiffLine | null)[] = [];
  let gap = false;
  lines.forEach((l, i) => {
    if (keep[i]) {
      if (gap && out.length) out.push(null);
      gap = false;
      out.push(l);
    } else gap = true;
  });
  return out;
}

export function diffStats(lines: DiffLine[]): { added: number; removed: number } {
  let added = 0;
  let removed = 0;
  for (const l of lines) {
    if (l.kind === "+") added++;
    else if (l.kind === "-") removed++;
  }
  return { added, removed };
}
