// Starter models: the ones Pelica's engines are tested with, and whether
// each runs well on this PC (its GPU's own memory, its RAM, free disk space).
// Sizes are from Hugging Face; a download takes the real ones from there.
import { bridge, type Download, type Hardware } from "./bridge";

export interface Starter {
  id: string;
  name: string;
  blurb: string;
  repo: string;
  /** the files to download, as the repository names them */
  files: string[];
  /** bytes, all files */
  size: number;
  tools: boolean;
  vision?: boolean;
  /** why it won't run well here, or null when it will */
  wont: (vram: number, ram: number) => string | null;
}

const GB = 1e9;
const gb = (n: number) => `${(n / GB).toFixed(n < 10 * GB ? 1 : 0)} GB`;
/** memory as it is sold (a 16 GB card reports 17.1e9 bytes) */
export const gib = (n: number) => `${Math.round(n / 2 ** 30)} GB`;

export const STARTERS: Starter[] = [
  {
    id: "gpt-oss-20b",
    name: "gpt-oss 20B",
    blurb: "OpenAI's open model. The whole model runs on the GPU: the fastest answers, with reasoning and tools.",
    repo: "ggml-org/gpt-oss-20b-GGUF",
    files: ["gpt-oss-20b-MXFP4.gguf"],
    size: 12.11 * GB,
    tools: true,
    // the whole model (12.1 GB) and its context live in the GPU's memory
    wont: (vram) => (vram >= 14.5 * GB ? null : `needs a GPU with 16 GB of memory (this one has ${gib(vram)})`),
  },
  {
    id: "qwen3-30b-a3b",
    name: "Qwen3 30B-A3B Instruct",
    blurb: "A strong all-rounder with tools. Experts that don't fit on the GPU wait in RAM, so it runs on mid-range GPUs too.",
    repo: "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
    files: ["Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"],
    size: 18.56 * GB,
    tools: true,
    wont: (vram, ram) => (vram < 6 * GB ? `needs a GPU with at least 6 GB of memory (this one has ${gib(vram)})` : vram + ram * 0.75 < 30 * GB ? `needs more GPU memory and RAM together (about 32 GB RAM with an 8 GB GPU)` : null),
  },
  {
    id: "qwen3.8-flash",
    name: "Qwen3.8 Flash Next",
    blurb: "The biggest and smartest here, and it sees pictures. 94 GB: the GPU, RAM and a fast SSD share it, so answers come slower.",
    repo: "unsloth/Qwen3.8-Flash-Next-GGUF",
    files: ["UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf", "UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf", "mmproj-F16.gguf"],
    size: (0.01 + 49.84 + 43.84 + 0.9) * GB,
    tools: true,
    vision: true,
    wont: (vram, ram) => (vram < 12 * GB ? `needs a GPU with at least 12 GB of memory (this one has ${gib(vram)})` : ram < 30 * GB ? `needs 32 GB of RAM (this PC has ${gib(ram)})` : null),
  },
  {
    id: "olmoe",
    name: "OLMoE 1B-7B",
    blurb: "Small and quick, for GPUs with little memory. Chat only: it can't use tools or work on files, and its context is short (4K).",
    repo: "allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
    files: ["olmoe-1b-7b-0924-instruct-q4_k_m.gguf"],
    size: 4.21 * GB,
    tools: false,
    wont: (vram) => (vram >= 5 * GB ? null : `needs a GPU with at least 6 GB of memory (this one has ${gib(vram)})`),
  },
];

/** The GPU Pelica uses (by the name the engine reports), else the one with the most memory. */
export function gpuOf(hw: Hardware, name?: string) {
  const byName = name ? hw.gpus.find((g) => name.toLowerCase().includes(g.name.toLowerCase()) || g.name.toLowerCase().includes(name.toLowerCase().replace(/\s*\(.*\)$/, ""))) : undefined;
  return byName || [...hw.gpus].sort((a, b) => b.vram - a.vram)[0] || { name: "no GPU found", vram: 0 };
}

/** The one to suggest first: the first in the list that runs well here (OLMoE last). */
export function bestFor(vram: number, ram: number): Starter | undefined {
  return STARTERS.find((s) => s.id !== "qwen3.8-flash" && !s.wont(vram, ram)) || STARTERS.find((s) => !s.wont(vram, ram));
}

/** Starts downloading `files` (with their sizes) when the download folder's
 *  drive has room for them and the downloads already going; else says why.
 *  Returns an error message, or null when it started. */
export async function getFiles(repo: string, files: { path: string; size: number }[], going: Download[]): Promise<string | null> {
  if (!bridge) return "Downloading needs the Pelica desktop app.";
  const hw = await bridge.lib.hardware();
  const dir = hw.downloadDir;
  const drive = await bridge.lib.space(dir);
  if (!drive) return `The download folder's drive (${dir}) isn't there. Choose another in Settings → Models.`;
  const root = drive.root.toLowerCase();
  // what running and paused downloads on that drive still have to write
  const pending = going.filter((d) => (d.state === "running" || d.state === "paused") && d.dest?.toLowerCase().startsWith(root)).reduce((n, d) => n + Math.max(0, d.total - d.got), 0);
  const need = files.reduce((n, f) => n + f.size, 0);
  if (drive.free - pending < need + 1 * GB) {
    const others = (await bridge.lib.drives()).filter((d) => d.root.toLowerCase() !== root && d.free > need + 2 * GB);
    return `Not enough space on ${drive.root.slice(0, 2)}: this needs ${gb(need)} and ${gb(Math.max(0, drive.free - pending))} is free.${others.length ? ` ${others.map((d) => d.root.slice(0, 2)).join(" and ")} ${others.length > 1 ? "have" : "has"} room: change the download folder in Settings → Models.` : " Free some space, or uninstall a model you don't use."}`;
  }
  for (const f of files) await bridge.lib.download(repo, f.path, dir);
  return null;
}
