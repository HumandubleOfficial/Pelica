// The Pelica engine's HTTP API (same origin): status, models, loading,
// token counts, and chat v2 streamed as server-sent events.

export interface Status {
  state: "empty" | "loading" | "ready" | "error";
  kind?: "moe" | "gpu" | "tiered";
  model?: string;
  path?: string;
  ctx?: number;
  max_ctx?: number;
  gpu?: string;
  arch?: string;
  efforts?: string[];
  effort?: string;
  progress?: number;
  busy?: boolean;
  error?: string;
  experts?: number;
  experts_ram?: number;
  ram_gb?: number;
  kv_vram?: number;
  kv_ram?: number;
  kv_drive?: number;
  note?: string;
  load_s?: number;
  tools?: boolean;
  /** the model was trained for tool calls (else it tends to explain instead of act) */
  agent?: boolean;
  /** the model takes images (else photos go in as the text Windows OCR reads) */
  vision?: boolean;
}

export interface LocalModel {
  path: string;
  name: string;
  gb: number;
  kind: string;
  max_ctx: number | null;
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: any;
}

export interface WireCall {
  id: string;
  name: string;
  arguments: any;
}

export interface WireMsg {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  /** photos (base64, no data: prefix), for a model that sees images */
  images?: string[];
  reasoning?: string;
  tool_calls?: WireCall[];
  tool_call_id?: string;
  name?: string;
}

export type ChatEvent =
  | { t: "start"; prompt: number; reused: number; ctx: number }
  | { t: "prefill"; done: number; total: number }
  | { t: "think"; d: string }
  | { t: "text"; d: string }
  | { t: "stats"; tps: number; generated: number; elapsed_s: number }
  | { t: "tool_call"; id: string; name: string; arguments: any; raw?: string; before?: string }
  | { t: "experts"; layers?: number; experts?: number; hits: number[] }
  | { t: "done"; prompt: number; generated: number; tps: number; prefill_s: number; decode_s: number; think_s?: number; first_s?: number; ctx_used: number; ctx: number; finish: "stop" | "length" | "tool_call" | "stopped"; reused: number }
  | { t: "error"; error: string };

async function j<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const v = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((v as any).error || `HTTP ${r.status}`);
  return v as T;
}

export const engine = {
  status: () => j<Status>("/api/status"),
  models: () => j<{ dir: string; models: LocalModel[] }>("/api/models"),
  load: (path: string, ctx: number) => j("/api/load", { path, ctx }),
  unload: () => j("/api/unload", {}),
  stop: () => j("/api/stop", {}),
  count: (parts: { system: string; messages: WireMsg[]; tools: ToolSpec[]; mcp_tools?: ToolSpec[] }) => j<{ system: number; messages: number; tool_results: number; tools: number; mcp: number; total: number; error?: string }>("/api/count", { parts }),
  gguf: (path: string) => j<any>("/api/gguf", { path }),
  /** The experts each token of `text` goes to, per layer: routes[token][layer] = [expert, weight][]. */
  trace: (text: string) => j<{ tokens: string[]; layers: number; experts: number; routes: [number, number][][][] }>("/api/trace", { text }),
  tensor: (path: string, name: string) => j<{ n: number; buckets: [number, number][]; mean: number; sd: number }>("/api/tensor", { path, name }),

  /** One v2 completion; events stream to `on`. Resolves with the `done` event (or throws). */
  async chat(req: { messages: WireMsg[]; tools: ToolSpec[]; effort?: string; max_tokens?: number; experts?: boolean }, on: (e: ChatEvent) => void, signal?: AbortSignal): Promise<Extract<ChatEvent, { t: "done" }>> {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ v: 2, ...req }), signal });
    if (!r.ok || !r.body) {
      const v = await r.json().catch(() => ({}));
      throw new Error((v as any).error || `HTTP ${r.status}`);
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let done: Extract<ChatEvent, { t: "done" }> | null = null;
    let err: string | null = null;
    for (;;) {
      const { value, done: end } = await reader.read();
      if (end) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const line = chunk.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        let ev: ChatEvent;
        try {
          ev = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        if (ev.t === "done") done = ev;
        if (ev.t === "error") err = ev.error;
        on(ev);
      }
    }
    if (err) throw new Error(err);
    if (!done) throw new Error("the engine stopped without finishing the reply");
    return done;
  },
};
