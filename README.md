# Pelica

A tiered **GPU + RAM + SSD** inference engine for giant Mixture-of-Experts
models, written in Rust. The aim is to run frontier-scale MoE checkpoints on a
mid-range gaming PC faster than [Colibrì](https://github.com/JustVugg/colibri).

The idea is the same one Colibrì proved: an MoE token only touches a few
experts, so the dense part of the model can stay resident while the experts
stream from an NVMe drive. Pelica's bet is that the winning engine treats
VRAM, RAM and disk as one cache hierarchy and keeps the disk busy every
millisecond, instead of reading experts only once a layer asks for them.

## The three tiers

```
 Layer 1  VRAM   the essentials and the hottest experts — the GPU computes here
 Layer 2  RAM    the main core: the next-hottest experts, evicted by heat
 Layer 3  SSD    the rest of the model; experts heat up here and get promoted
```

* **Layer 1 (GPU).** The dense weights that every token needs (attention
  projections, dense FFNs, shared experts) are uploaded to VRAM at start, and
  attention runs entirely on the GPU: a KDA layer (projections, short conv,
  gated delta rule, norm, output) or an MLA layer (projections, norms, latent
  cache, absorbed attention, output) is one GPU round trip, with its state
  kept in VRAM. Leftover VRAM becomes a fixed set of expert slots, filled with
  the historically most-used experts. The GPU runs those experts while the CPU
  runs the rest, in parallel.
* **Layer 2 (RAM).** A byte-budgeted cache of experts, sized to the free RAM
  after loading, that evicts the coldest expert. Heat is mostly a short memory
  (half-life 4 tokens) of how highly the router ranked each expert, over all
  288 of a layer's experts rather than just the chosen 8, so a near-miss is
  warm before it gets picked; a little of a long memory of real selections
  (half-life 64 tokens) keeps the steady favourites. Replaying recorded
  router traces, that reads ~4% less from disk than counting uses, and the
  parameters sit at the optimum of a sweep. Plain LRU is the wrong fit here:
  experts come round once per token in layer order, and LRU throws out
  exactly the ones whose layer is up next.
* **Layer 3 (SSD).** Everything else. Every routing adds heat to an expert;
  an expert that becomes clearly hotter than the coldest one in VRAM is
  promoted into its slot (and dropped from RAM, so no expert sits in two
  tiers). The upload happens on a background thread.

The SSD is driven by a small I/O scheduler. Reads are served in priority
order: reads a layer is blocked on first (while any is under way, prefetching
pauses), then lookahead guesses, nearest layer and most confident first. On
Windows one asynchronous engine (overlapped unbuffered reads on an I/O
completion port, `src/aio.rs`) issues them: it keeps at most 4 requests at the
drive, of which at most 2 are guesses, and everything else waits in Pelica's
own queue, where a blocked layer jumps ahead and a dropped guess costs
nothing. Measured on the test drive, a request costs a fixed ~0.1 ms of drive
time on top of its bytes and queue depth beyond two adds nothing (16 MB reads:
5.06 GB/s with one in flight, 5.80 with two, 5.62 with four; 1 MB reads top
out at 3.3 GB/s), so each weight span goes out as one read of up to 16 MB.
`PELICA_AIO=0` falls back to reader threads. The guesses come from running the next layer's router on the
current layer's input, which picks the right expert 99% of the time at rank
1 and about half the time at rank 8. When a layer has actually routed, the
guesses it did not pick stop reading. The first MoE layer has no MoE layer
before it to guess from, so the disk used to sit idle across every token
boundary (~11 ms); now its experts are guessed from the token's embedding
(~40% right) while the dense layers run, and again from the last dense
layer's output (~75%). While a layer waits for its misses, it already
computes the experts that were in RAM.

**Packed scales.** An int4 expert carries one f32 scale per 64 weights
(1.6 MB per expert), but each matrix uses at most a few hundred distinct
scale values. On first run Pelica writes `pelica_scales.bin` next to the
model (4.8 GB, one-off, ~20 s): per matrix, a table of its distinct values
and a one-byte index per group. Lookups give back the exact f32, so tokens
do not change, and every expert drops from 14.2 to 13.0 MB on disk, in RAM
and in VRAM: 9% more experts per GB and 8.5% fewer bytes per read.
`--no-pack` uses the scales as stored.

Prompts are processed 512 tokens at a time, so each expert is read once per
prompt rather than once per small chunk, and experts with several tokens are
streamed through spare VRAM slots and run on the GPU.

Use `--no-gpu` to run CPU only, and `--vram-gb <gb>` to cap VRAM use (the
default is free VRAM minus 0.75 GB; the driver's shared-memory counter stays
at zero, i.e. nothing spills into system RAM).

## Status

| | |
|---|---|
| Model | GLM-5.3-Flash (`glm5_next`, 320B-A18B), Colibrì's int4-gs64 container |
| Correctness | **Bit-identical next-token choices to Colibrì** on a 24-token teacher-forced prompt and a 16-token greedy continuation; the GPU path gives the same tokens as the CPU path |
| Resident weights | 8.5 GB (int8 per row), 7.9 GB of it in VRAM — or 4.6 GB with `--dense-bits 4` |
| Expert tiers | VRAM slots with heat-based promotion, rank-heat-evicted RAM tier, prioritised SSD reads with lookahead and cancellation, losslessly packed scales |
| GPU | wgpu (Vulkan / DX12): no CUDA toolkit or SDK needed |
| Speed | **~4× Colibrì's decode** (3.16 vs 0.76 tok/s) on the same machine and prompt; long prompts 4.8× faster than before |

Results on the dev machine (Ryzen 5 8400F, 32 GB DDR5, RTX 5060 Ti 16 GB,
PCIe 4 NVMe) are in [Benchmarks](#benchmarks).

## Usage

```bash
cargo build --release
```

```bash
./target/release/pelica --model C:/models/glm53_flash_i4 -p "Why is the sky blue?" --no-think -n 128
```

Useful options (`--help` lists all):

* `--no-gpu` CPU only; `--vram-gb <gb>` cap VRAM use
* `--cache-gb <gb>` RAM for cached experts (default: whatever is free)
* `--dense-bits 8|4|32` precision of the resident weights (4 frees ~3.5 GB of
  VRAM for experts: about 6% faster decode, slightly different tokens)
* `--chunk <n>` prompt tokens per forward pass (default 512)
* `--no-pack` read expert scales as stored instead of from `pelica_scales.bin`
* `--lookahead <n>` MoE layers ahead to prefetch experts for (1 is best measured)
* `--buffered` read experts through the OS file cache instead of unbuffered
* `--warm` preload the historically most-used experts (Pelica keeps its own
  routing history in `.pelica_usage` next to the model; it reads Colibrì's
  `.coli_usage` if that is all there is)
* `pelica bench-disk --model <dir>` measures expert read throughput
  (`PELICA_AIO_BENCH=1` instead sweeps overlapped request size x depth);
  `pelica gpu-test` checks the GPU kernels against the CPU ones
* `--teacher` print the argmax at every prompt position plus the generated
  ids, the same format as Colibrì's `glm53 --ids ... --greedy N`, for exact
  comparison

### Pelica Desktop

`desktop/` is the chat app: Electron + React + TypeScript, frameless, laid out
like the Claude desktop app. It starts `pelica.exe app` as its engine (chat API
v2: whole conversation + tools in, streamed reply out, stopping at tool
calls; see `desktop/ARCHITECTURE.md`) and adds the agent around it:

* web search and page reading for every model (never for small talk), MCP
  servers (Settings → Connectors); links open in your default browser. Searches go
  through Chromium's network stack (DuckDuckGo, then Bing, then DuckDuckGo
  rendered in a hidden window), so engines don't turn them away as scripts;
  pages built by scripts are rendered before reading
* file and command work needs a model trained for tool calls (Qwen3,
  gpt-oss, GLM). Older chat models such as OLMoE explain instead of doing;
  the app says so when one is loaded with a folder chosen
* a workspace folder: the models read, create and edit files there (shown
  with diffs) and run commands (each one asks first); run boxes with a play
  button for the commands a reply suggests; long tasks with a todo list
* live markdown and tok/s while it writes, effort levels with their own
  animations, a context meter (system, tools, MCP, messages, tool results),
  `/compact`, `/autocompact` and other slash commands (`/help`)
* CQS, the code quality scanner: checks changed code (syntax, brackets, tags,
  placeholders, the project's own compilers) and hands problems to a fixer
  subagent; `/cqs polish` also tidies code and UI
* a model library (models on this PC; GGUF models on Hugging Face to
  download, with pause and resume that survive closing Pelica; cancelling
  puts the partial file in the Recycle Bin), the Brain inspector (every tensor, what it does, expert heat),
  themes including Neural, whose background lights up the experts in use
* the Brain's neural map: every expert of every layer as a node in a 3D brain
  (or unfolded into layers) that fires live as the model routes tokens,
  with signals running along the synapses; "Watch it think" runs a prompt
  so you can see it, and a token trace (`POST /api/trace`) shows the exact
  experts each token of a text goes through, layer by layer
* Windows integration: a tray icon; a global quick-ask bar (Alt+Shift+Space
  by default, with an optional "with clipboard" chip) that answers right
  there, streaming (Enter asks, Esc stops, "Continue in Pelica" turns the
  exchange into a chat, Ctrl+Enter sends the question to the main window
  instead); notifications when a
  reply finishes or a command waits while you're elsewhere; taskbar progress
  for the prompt pass; a jump list (New chat, Quick ask); keep running in the
  tray; start with Windows (off by default); Mica or Acrylic backdrop
* attachments (drop text files anywhere on the window, paste, the
  paperclip, or type `@` and part of a name to pick a file from the
  workspace folder), export a chat as Markdown or copy it whole (`/export`,
  `/copy`, the palette, or a chat's menu in the sidebar), an artifact panel that previews HTML, SVG and Markdown from
  replies and written files in a sandboxed frame, the command palette
  (Ctrl+K) and shortcuts (Ctrl+N new chat, Ctrl+B sidebar, Ctrl+, settings,
  Ctrl+L message box, Esc stops a reply)
* computer tools: system_info (Windows, CPU, GPU, memory, displays, disks),
  clipboard_write, and clipboard_read / open_file, which ask first

```bash
cd desktop && npm install && npm run build
```

Then start `desktop/Pelica.cmd` (it finds `target/release/pelica.exe`).

To build the Windows installer (no downloads: it reuses the Electron in
`node_modules` and the C# compiler that ships with Windows):

```bash
cd desktop && npm run installer
```

It writes `desktop/release/Pelica-Setup-<version>.exe`: a per-user install
to `%LOCALAPPDATA%\Programs\Pelica` (no admin), Start menu and desktop
shortcuts, an entry in Installed apps, and an uninstaller that removes only
the files it installed (your chats and settings stay unless you tick the
box, and then go to the Recycle Bin). Flags: `/S` silent, `/D=<dir>`,
`/NODESKTOP`, `/NOLAUNCH`.

## How it works

```
token ─► embed ─► 4 mHC streams ─► 45 × [ hc_pre ─► KDA or MLA ─► hc_post
                                          hc_pre ─► dense FFN / shared + 8 of 288 experts ─► hc_post ]
                                ─► mean of streams ─► norm ─► lm_head
```

* `src/st.rs` — multi-shard safetensors index; dense weights are read
  through mmap once at load.
* `src/quant.rs` — F32 / int8-per-row / int4-gs64 matrices and AVX2 kernels.
  Experts are used in place in the container's int4 layout.
* `src/forward.rs` — the GLM5-Next layer stack: Kimi Delta Attention (gated
  delta rule with short convs), nope-MLA with the latent-absorbed cache, mHC
  Sinkhorn mixing, sigmoid routing with bias-corrected top-k, clamped SwiGLU.
* `src/experts.rs` — the expert store (layers 2 and 3): the heat-evicted RAM
  tier, the chunked priority I/O scheduler with cancellation, and in-flight
  deduplication so a layer waits on a prefetch rather than reading the same
  expert twice.
* `src/aio.rs` — overlapped unbuffered reads through an I/O completion port
  (Windows), used by the scheduler's engine and by `bench-disk`.
* `src/scalepack.rs` — builds and opens `pelica_scales.bin`, the packed
  (table + index) expert scales.
* `src/gpu.rs` — wgpu device; WGSL kernels for int8 / int4 matrix-vector
  (plain and per-head), SwiGLU, and attention (KDA conv and delta rule, RMS
  norm, MLA latent attention); a batch API that chains them with one readback.
* `src/vram.rs` — layer 1's expert slots, heat tracking and promotion.
* `src/prof.rs` — the per-part timing printed after decode.

Contexts up to 2,051 tokens use exact dense attention (the DSA indexer selects
every position at that length); the indexer itself is not implemented yet.

## Benchmarks

GLM-5.3-Flash, prompt "Write a short paragraph about how hummingbirds hover."
(23 tokens), 48 greedy tokens, each engine with its defaults in a fresh process
(`sh bench/vs_colibri.sh`). Ryzen 5 8400F (6c/12t), 32 GB DDR5, PCIe 4 NVMe
measured at ~5 GB/s for random expert reads (`pelica bench-disk`).

| engine | decode | s/token |
|---|---|---|
| Colibrì 2.0.0 `glm53` (int4 dense, CPU) | 0.683–0.760 tok/s | 1.32–1.46 |
| Pelica `--no-gpu` (int8 dense, CPU + RAM + SSD; before the I/O scheduler) | 1.369 tok/s | 0.73 |
| **Pelica** (int8 dense, VRAM + RAM + SSD, RTX 5060 Ti 16 GB) | **3.16 tok/s** (3.161, 3.170) | **0.32** |

Pelica is **~4× faster** while keeping the resident weights at higher
precision. With both engines on int8 dense weights (`GLM53_BITS=8`) the
generated tokens are identical, which is how correctness is checked.

Where the speed came from (same prompt, Pelica only):

| change | s/token |
|---|---|
| on-demand expert reads, LRU RAM tier | 1.60 |
| + lookahead prefetch of the next MoE layer's experts | 1.35 |
| + unbuffered reads (no double caching in the OS file cache) | 0.73 |
| + GPU tier: dense weights and ~400 hot experts in VRAM | 0.64 |
| + heat eviction, chunked priority reads, cancelling wrong guesses, buffer reuse, computing RAM experts while misses load | 0.48 |
| + attention on the GPU, dedicated readers for blocking reads, whole-span requests, background VRAM uploads | 0.39 |
| + packed scales (experts 9% smaller everywhere) | 0.36 |
| + rank-heat eviction, first-layer prefetch across the token boundary, ~60 more VRAM slots | 0.34–0.35 |
| + asynchronous I/O engine at a fixed drive depth | 0.32–0.33 |
| + ~41 more VRAM slots after prefill, bf16 routers, 2.5 GB RAM headroom (was 3) | **0.316** |

The engine was worth +2.5–6% in interleaved A/B runs with pinned tiers
(`sh bench/abc.sh`); the last row's runs also had a little more free RAM and
VRAM than the earlier ones (466 VRAM slots and ~1,985 RAM experts vs ~414 and
~1,860), which accounts for part of the step.

The last row: once prefill is over, its GPU scratch buffers are shrunk and the
freed VRAM becomes ~41 more expert slots (`PELICA_GROW=0` turns this off;
3.161 and 3.170 tok/s with it, 3.107 and 3.126 without, default tier sizes).
The router weights are bf16 values stored as f32 in the checkpoint; they are
now kept as bf16 with a dot product that gives bit-identical scores. The RAM
tier leaves 2.5 GB free for everything else (`PELICA_RAM_MARGIN_GB` changes
it; raise it if a game or other app needs the room). Letting a
guess share the drive with a lone blocking read (`PELICA_AIO_HOLD=2`) did not
help (2.82 vs 2.90 tok/s over three interleaved rounds), so prefetches still
wait.

About 30% of expert uses run from VRAM, 38% hit RAM, 21% arrive by prefetch
and 11% are read from the SSD on demand. The disk is busy 95% of the time at
about 4.9 GB/s (the drive's measured best is 5.25), so decode is limited by
the SSD: it moves about 1.65 GB per token, of which about 1.4 GB is used. The
rest is wrong guesses: a guess is still mid-read when its layer routes (a
12.6 MB read takes longer than a layer), so a wrong one costs all its bytes.
Run-to-run noise is about ±2%, and after an hour of back-to-back runs the
drive throttles (busy throughput falls from ~4.9 to as low as 3.8 GB/s): a
heatsink on the NVMe is worth more than most of the software tweaks here.
Decode is not bit-reproducible run to run (whether an expert runs on the GPU
or the CPU depends on when its VRAM upload lands, and the two kernels round
differently), so correctness is checked on the chosen tokens, which have
matched the reference in every run.
On a 160-token answer with smaller fixed tiers (`sh bench/fridge.sh`) decode
went from 0.61 to 0.38 s/token over these changes.

Tried and measured, no gain (so not kept): finishing unpicked guesses into
the cache instead of dropping them; re-guessing each next layer from this
layer's output (more accurate, 79–81% vs 75%, but the extra reads start too
late and the wrong ones cost more than the right ones save); smaller read
requests so cancellation stops sooner (saves bytes, but the drive is slower
on them); lookahead of 2 layers; narrower or wider guess lists; 1 or 3
prefetch readers; 8 MB blocking reads. Lossless entropy coding of the int4
weights would save at most ~12% (they carry 3.5 bits of information per
4-bit value), but would need a 134 GB recoded copy and fast decoding.

Also tried, with numbers:

* **Speculative decoding** (the checkpoint has an MTP draft layer). Replaying
  routing traces, verifying two tokens per pass costs 223.5 expert reads per
  step against 111.4 for one token: consecutive tokens share 2.5 of 8 experts
  per layer, and those are nearly always cached already. With no reads saved,
  any rejected draft is a loss on an SSD-bound engine.
* **A learned lookahead** (a trained correction to the next-layer guess).
  Trained on ~11,600 tokens of routing it overfits: expert use follows the
  topic, and corrections learned on a few documents hurt held-out ones
  (recall 0.54 -> 0.45-0.56 on unseen text, 0.69 -> 0.61-0.69 on real decode).
* **Lookahead two layers** with the asynchronous engine: it halves the
  drive's idle time between layers (0.9 -> 0.45 s per 48 tokens) but reads
  more and slows decode ~5% (2.75-2.77 vs 2.90 tok/s).
* **Splitting blocking reads in two** so the drive works on both halves at
  once (`PELICA_DCHUNK=6815744`): 2.89 vs 2.90 tok/s, no change.
* **A smaller VRAM margin** (`PELICA_VRAM_MARGIN_GB=0.5`, 18 more slots, no
  spill to shared memory): within noise (3.114 vs 3.129 tok/s), and it leaves
  less room for other apps.
* **Small cancellable requests** at high queue depth: the drive's per-request
  cost makes them slower than what cancellation saves (see above).
* **A contiguous (or compressed) hot copy** of the most-missed experts, to
  save the separate scale-block request: 1,500 experts (19.5 GB) would cover
  ~23% of disk reads for ~1.4% (uncompressed) to ~4% (compressed) less disk
  time; 40% coverage needs 39 GB of disk.

Prompt processing, 340-token prompt (`bench/long_prompt.txt`):

| | prefill |
|---|---|
| 32-token chunks, CPU experts | 158.5 s |
| 512-token chunks | 61.1 s |
| + experts with several tokens run on the GPU | 36.0 s |
| + packed scales, more VRAM slots | **33.0 s** |

### Models that fit in VRAM: gpt-oss-20b

Given a `.gguf` file, Pelica runs its GPU-resident engine instead of the tiers:
the whole model stays in VRAM (12.3 GB for gpt-oss-20b MXFP4) and the whole
token runs in kernels, routing and argmax included.

```bash
./target/release/pelica --model C:/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf -p "Why is the sky blue?" -n 256
```

* **Decode is exactly greedy.** Each pass checks the current token plus a
  prompt-lookup draft on the GPU and keeps what the model agrees with. The
  kernels for passes of 1, 2, 4 and 8 tokens do every token's arithmetic the
  same way, so drafting never changes the text (`PELICA_CHECK=2,5,8`: 0 of
  201,088 logits differ from one-token passes).
* **Prompt passes** of up to 512 tokens run as tiled products. Q8_0 attention
  weights and MXFP4 experts are decoded to f16 into workgroup memory and
  multiplied on the tensor cores (`coop_mat16x16`, f32 sums), and attention is
  a flash kernel. Experts are grouped into 64-token tiles from a GPU-built plan.
  These passes only fill the KV cache. The prompt's last token goes through
  the exact decode path, and its logits are within 4e-2 of all one-token
  passes (same argmax). Set `PELICA_TC=0` for the f32 path (6e-5, half the
  speed).
* **Context up to the model's 131,072 tokens, VRAM then RAM.** The
  sliding-window layers keep a small ring; the full-attention layers keep
  every position, 48 KB each (f32 keys and values of 12 layers). As many
  positions as fit go in VRAM (about 51K next to the weights on a 16 GB
  card); the rest go in pinned host RAM that the GPU reads in place over
  PCIe (about 12 GB/s, against 270 GB/s for VRAM) when decoding, and through
  a VRAM staging copy in prompt passes, which read every key 64 times. Pass
  attention for full layers is split over fixed key ranges and merged in a
  fixed order, so a token's arithmetic still doesn't depend on the pass size
  or on where its keys live: output is bit-identical with the cache all in
  VRAM or all in RAM (`PELICA_KV_RESIDENT=<n>` forces how many positions stay
  in VRAM). `--prompt-file` takes prompts longer than a command line allows.
  Deep in a long context, prompt passes go to the GPU a layer at a time
  (Windows resets a GPU whose single submission runs for seconds). The SSD
  is not a tier for the cache: every token reads all of it, and the full
  131K context needs at most ~4 GB past VRAM, which RAM holds.

RTX 5060 Ti 16 GB, Vulkan:

| | gpt-oss-20b |
|---|---|
| prefill, 341-token prompt (`bench/long_prompt.txt`) | **1,773 tok/s** (0.19 s); f32 path 830 tok/s |
| prompt pass of 512 tokens (GPU time) | 253 ms (2,020 tok/s): expert gate/up 42%, down 22%, Q8 products 18%, attention 13% |
| decode, plain greedy | 112–127 tok/s |
| decode with drafting (`sh bench/oss_spec.sh`) | 111–141 tok/s, up to 2.0 tokens per pass on copy-heavy prompts; text identical |
| 27K-token prompt (needle in source code, found) | prefill 553 tok/s (49 s); decode 51 tok/s all in VRAM, 13.7 tok/s with 23K positions in RAM (same text) |
| 96K-token prompt, `--ctx 131072` (52K positions in VRAM, 3.9 GB in RAM) | prefill 192 tok/s (8.3 min); decode 8.1 tok/s |

Long prompts are bound by the prompt passes' attention (87% of a 512-token
pass at position 30K, f32 on the shader cores): moving it to the tensor
cores is the next step for them.

### Any other MoE GGUF: Qwen3-MoE, OLMoE, Mixtral-style, ...

Other mixture-of-experts `.gguf` files run on a general engine (`src/moe.rs`,
kernels in `src/moe_kernels.rs`) that reads the architecture from the file:
head counts, grouped-query attention, q/k norms (per head as in Qwen3 or
over the whole projection as in OLMoE), RoPE (NeoX or adjacent pairs, linear
or YaRN scaling), softmax or sigmoid routing with or without renormalised
weights, shared experts, leading dense layers and tied output weights.
Weights stay in their GGUF formats (F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0,
Q4_K, Q5_K, Q6_K, IQ4_XS, MXFP4: per-layer mixes such as Q4_K_M are fine),
decoded in the kernels. The prompt goes through the model's own chat
template (Jinja, via minijinja), and its end-of-turn tokens stop the reply.

```bash
./target/release/pelica --model D:/Models/qwen3-30b-a3b/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf -p "Why is the sky blue?" -n 256 --ctx 8192
```

* **Experts that don't fit in VRAM stay in RAM**, in pinned host memory the
  GPU reads in place. Which experts get VRAM is decided by how often each was
  used: the engine counts every expert's use on the GPU and adds the counts
  to `<model>.gguf.pelica-usage` after each run (the app does this after
  every reply), so the next load puts the hottest ones in VRAM. Experts are
  packed into ~1 GB buffers (one buffer per layer wasted gigabytes in
  allocator padding). `PELICA_EXPERTS_GB=<gb>` caps the VRAM for experts.
* **Experts move between VRAM and RAM while it writes.** Experts specialise
  by topic, so placement learned from coding chats leaves a story's experts
  in RAM (Qwen3-30B-A3B: 34% of expert reads over PCIe, 21 tok/s). Every 8
  tokens the engine reads the use counters back without waiting, keeps a
  fading score per expert, and per layer swaps the host experts used more
  than the least used VRAM ones (GPU copies through a 32 MB buffer, the slot
  table rewritten in the same submission, so queued passes stay consistent;
  the first round after a prompt moves up to 1 GB). Same story: 0.3% from
  RAM, 62 tok/s, the same tokens. `PELICA_ADAPT=0` turns it off.
* **Decode kernels sized to the model.** Rows of a known width read all
  their quant blocks in one go (`rowdot_w`), the expert kernels take two rows
  per subgroup, the routing kernel builds its expert list with a parallel
  scan instead of one thread, and per-head q/k norms, RoPE and the cache
  write run as one dispatch (`qk_dec`). With the placement above, Qwen3
  decode went from 21 to 78 tok/s on a story prompt (69 to 87 on code),
  tokens identical. `PELICA_PROF=1 PELICA_PROF_REAL=1` profiles GPU time per
  kernel on a real generation and reports the share of expert reads from RAM.
* **Long contexts: the KV cache spills from VRAM to RAM to the drive.** The
  cache (f16) is split by position. The first positions go in VRAM: all of
  them if that takes at most 40% of the free VRAM, else a quarter of it, so
  the experts keep the rest. The next positions go in pinned host RAM, as
  much as fits while leaving 6 GB free. The rest goes in a scratch file in
  `%LOCALAPPDATA%\Pelica\kv` (`PELICA_KV_DIR` to move it; 5 GB stay free
  on that drive), mapped into memory. The OS deletes the file when the model
  is unloaded or Pelica exits. So Qwen3-30B-A3B runs at its full
  262,144-token context, about 25 GB of cache, on a 16 GB card.
* **Decode attention past VRAM runs on the CPU.** Each decoded token reads
  the whole cache, and the CPU reads RAM at ~45 GB/s where the GPU got ~15
  GB/s over PCIe. So for every layer the GPU sends the token's queries and
  new key/value row to the CPU (~150 µs round trip), the CPU attends over
  the RAM and drive positions on all its cores while the GPU does the VRAM
  ones, and its partial results go back for the GPU to merge. Drive
  positions are read straight from the mapped file (from the OS file cache
  when it holds them); a thread asks the OS to read the next layer's rows
  ahead (`PrefetchVirtualMemory`), so page faults don't stall the layer.
  Prompt passes stage the CPU's rows into VRAM a round of 16,384 positions
  at a time, and the pass's new rows go back to RAM or the file after it.
* **Where the cache lives doesn't change the output.** Decode attention
  works over fixed 256-key chunks, one workgroup per (KV head, chunk) with
  all the query heads that share that KV head, so each key is read once.
  Its arithmetic is spelled out so the CPU (`src/cpu_attn.rs`) reproduces it
  bit for bit: each lane's dot product in dim order with explicit fmas, the
  lanes summed by xor shuffles in a fixed order, a polynomial `exp` of adds,
  multiplies and fmas (the GPU's own `exp` is approximate in its own way),
  and fmas wherever a product meets a sum, so the shader compiler has
  nothing left to fuse. `PELICA_ATTN_CHECK=1` compares the two on every
  chunk of three layers: no bit differs. Prompt attention works over fixed
  1,024-key splits, folded in key order. So the greedy tokens are identical
  with the cache all in VRAM or split across VRAM, RAM and the drive
  (`PELICA_KV_VRAM` and `PELICA_KV_RAM`, in positions, force a split for
  testing; `PELICA_KV_DUMP=<file>` writes a few layers' cache rows to
  compare). Qwen3-30B-A3B's 27,426-token prompt plus 64 generated tokens
  gives the same cache, bit for bit, all in VRAM or split 8K VRAM / 8K RAM /
  16K drive. Which experts sit in VRAM does change the rounding of prompt
  passes, though, so compare runs with the same experts in VRAM: pin
  `PELICA_EXPERTS_GB` and restore `<model>.gguf.pelica-usage` between runs
  (each run adds its counts to it).
* Decoding never waits on the CPU while the cache is in VRAM: routing,
  top-k and the argmax run in kernels, and the next pass is recorded while
  the GPU works on this one. Past VRAM, each layer waits for the CPU's
  attention.
* **Prompts go in passes of 256 tokens on the tensor cores.** Each matrix
  product tiles 32 columns at a time of its weights (decoded from any of the
  formats above) and of its tokens into f16 in workgroup memory, and sums
  16x16 tiles in f32 (`coop_mat16x16`; `PELICA_TC=0` falls back to the
  vector kernels). Routed experts get tiles of their own tokens. Passes go
  to the GPU a layer at a time (one long submission gets the device reset).
* Checked against llama.cpp (`llama-server`, CUDA) on five chat prompts with
  150 generated tokens, teacher-forced: OLMoE agrees on 666 of 674 tokens and
  Qwen3-30B-A3B on 416 of 422 (with the tensor-core prompt passes; 664 and
  416 before). Every disagreement is a near-tie (llama.cpp's
  own top two tokens within 0.02-0.7 in log-probability), where its
  quantized-activation products and Pelica's f32 ones round differently.
  Other architectures load with a note that they haven't been checked.
* Not yet: fused-QKV and multi-head-latent attention (DeepSeek-V2/V3), and
  sliding-window or attention-sink models other than gpt-oss.

RTX 5060 Ti 16 GB, Vulkan, Q4_K_M files:

| | OLMoE-1B-7B (4.2 GB) | Qwen3-30B-A3B-Instruct-2507 (18.6 GB) |
|---|---|---|
| where | all in VRAM | 15.2 GB in VRAM; ~1,500 of 6,144 experts (4.2 GB) in RAM |
| load (file in the OS cache) | 5 s | 10 s |
| prefill | 2,155 tok/s (3,088-token prompt; was 572 before the tensor cores) | 524 tok/s (2,654-token prompt; was 185) |
| decode | 180 tok/s (158 at 3K tokens of context) | 40-54 tok/s |

Cache past VRAM, OLMoE at 4K context with a 2,935-token prompt, 200 tokens
generated (`PELICA_KV_VRAM` / `PELICA_KV_RAM` force the split; the rest of
the 4,096 positions is on the drive). "Before" is the GPU reading RAM over
PCIe and staging drive rows through VRAM; "after" is CPU attention. The
greedy tokens are the same in every row:

| cache (VRAM / RAM / drive positions) | decode before | decode after | prefill before | prefill after |
|---|---|---|---|---|
| all 4,096 in VRAM | 159 tok/s | 159 tok/s | 2,323 tok/s | 2,328 tok/s |
| 512 / 1,536 / 2,048 | 24.2 tok/s | 57.0 tok/s | 487 tok/s | 2,062 tok/s |
| 1,024 / 1,024 / 2,048 | 26.2 tok/s | 61.6 tok/s | 954 tok/s | 2,112 tok/s |
| 512 / 3,584 / 0 | 33.3 tok/s | 57.1 tok/s | 2,004 tok/s | 2,104 tok/s |

Past VRAM, each layer of a decoded token costs ~0.5 ms of GPU wait and
~0.5 ms of CPU attention, which runs at ~45 GB/s here (Ryzen 5 8400F,
DDR5) whatever the thread count, so it is bound by memory bandwidth.

Long context, Qwen3-30B-A3B at 32K context with a 27,426-token prompt
(source code with a "launch code" hidden in a comment; it finds it),
64 tokens generated:

| cache | prefill | decode before | decode after |
|---|---|---|---|
| all in VRAM (3.1 GB, so ~2,400 experts go to RAM) | 283 tok/s | (not rerun) | 14.4 tok/s |
| 8K VRAM, 8K RAM, 16K on the NVMe drive | 285 tok/s | 3.5 tok/s | 7.1 tok/s |

Loading a file the OS hasn't cached runs at the drive's speed: about
80 MB/s from the USB drive on the dev machine (`D:`), so 3.5 to 7.5
minutes for the 18.6 GB Qwen3 file.

#### Chat app

Double-click `Pelica.cmd` (or run `pelica app`). A window opens and the model
loads. It is a local page served by Pelica on `127.0.0.1:8642`; nothing
leaves the machine.

* **Models**: the menu under the model name lists every model under
  `C:\models` and `D:\Models` (or `--models-dir`, repeatable): gpt-oss
  `.gguf` files run fully on the GPU (gpt-oss-20b: loads in ~13 s, ~90-120
  tok/s); other MoE `.gguf` files show a **MoE** badge and run on the
  general engine, with the experts that don't fit in RAM (the sidebar then
  says "VRAM + RAM"; hover for how many); and tiered model folders
  (GLM-5.3-Flash: loads in ~25 s, streams experts from VRAM, RAM and SSD at
  ~2 tok/s, 2K context until the sparse-attention indexer exists). Switching
  frees the old model first; **Unload** frees the GPU and RAM.
* **Effort** sets the reasoning level per message: Low / Medium / High for
  gpt-oss, Off / Low / High / Max for GLM (Off skips thinking; changing it
  mid-chat makes GLM reread the chat, since it sits in the system header).
* **Thinking** streams under a "Thinking… 2.1s" timer, then folds into
  "Thought for 2.8s · 321 tokens". Before that, "Reading the conversation…"
  shows elapsed time and, for long prompts, a progress bar.
* **Context**: the ring in the header shows the tokens this chat uses; hover
  it for a breakdown including the draft you are typing (counted with the
  model's tokenizer). Settings picks the window for GPU models, 4K to 128K
  tokens (default 32K, which fits in VRAM); what doesn't fit in VRAM spills to RAM (the sidebar then says
  "VRAM + RAM"), which works but slows replies as a chat grows into it.
  Changing it reloads the model. MoE models default to 8K, since their
  cache shares VRAM with the experts. They go up to the model's maximum
  (262K for Qwen3-30B-A3B), with the cache spilling to RAM and then to a
  scratch file on the drive (the sidebar says "VRAM + RAM + drive"; hover it
  for how many tokens are where).
* **Admit uncertainty** (Settings, on by default) tells the model it is
  offline and should say when it isn't sure rather than invent names, dates,
  numbers or links. gpt-oss's identity is set to itself, not "ChatGPT". Its
  greedy output was checked against llama.cpp token by token: they match up
  to near-ties, so wrong facts are the 20B model's knowledge, not the engine.
* **Web search** (the **Web** button by the message box; MoE models, on by
  default). The model gets two tools, `web_search` (DuckDuckGo's plain HTML
  results) and `open_page` (a page's main text, with scripts, menus and
  footers dropped and long pages cut to fit the context). It decides when to
  use them. A reply can make up to 6 calls. Each one shows up in the reply as
  "Searched the web for …" with the result sites as links, or "Read <page>".
  The answer cites what it read. Qwen3-style templates carry the tools in
  their own format; for templates without tools, the format is described
  in the instructions (small models follow it less reliably). Requests go
  through the system `curl` (built into Windows 10 and later), so Pelica
  needs no TLS library. Only http(s) links are opened, never this computer
  or the local network (also checked after redirects and DNS). Searches and
  page loads leave your computer; nothing else does. Example: asked for the
  latest Rust release, Qwen3-30B-A3B searched, opened the official release
  notes and answered "1.99.1, October 15, 2026" with the link. Not yet:
  gpt-oss (its harmony tool format) and GLM-5.3.
* **Chats**: search, rename (double-click), delete, grouped by date; edit and
  resend any of your messages (or press ↑ in an empty box), regenerate the
  last reply, copy messages or code blocks, export a chat as Markdown, attach
  or drop text files. Chats live in the window's local storage.
* Each reply lists its model, tokens, speed, effort and how much of the
  prompt was read (follow-ups reuse the conversation's cache on both
  engines, so only the new text runs).
* Keys: Enter send, Shift+Enter new line, Esc stop, Ctrl+Shift+O new chat,
  Ctrl+K search, Ctrl+, settings.
* `pelica app --no-open` serves without a window. Without `--no-open`, the
  app runs in the background (log in `%TEMP%\pelica-app.log`) and quits
  about 3 minutes after its window closes.

Tuning hooks: `PELICA_TC=q8,gu,down` (which products use the tensor cores),
`PELICA_TC_Q8|GATEUP|DOWN=gc,wm,wn` (tile shapes), `PELICA_PREFILL=0` (prompt
through the decode kernels), `PELICA_PROF=1 PELICA_PROF_BATCH=512` (per-stage
GPU times of a prompt pass), `PELICA_DRAFT=0` (no drafting).

### DeepSeek V4 (`deepseek4` GGUF)

DeepSeek V4 files are far larger than VRAM + RAM, so they get an engine of
their own (`src/ds4.rs`, with `ds4_gpu.rs`, `ds4_io.rs`, `ds4_cpu.rs`) that
brings the tiers back to GGUF: the dense weights (attention, shared experts,
compressors, indexers, the head) sit in VRAM; routed experts sit in VRAM (the
most used, moved in as heat changes), in a RAM cache kept by heat, or on the
drive, read unbuffered when picked. The CLI and the app pick it up from the
file's architecture.

```bash
./target/release/pelica --model D:/Models/DeepSeek-V4/DeepSeek-V4-Q4_K_M.gguf -p "Why is the sky blue?" -n 128
```

* The math follows llama.cpp's `deepseek4` graph: 4 hyper-connection streams
  with Sinkhorn mixing, a sliding window of raw positions plus compressed
  entries (groups of 4 picked top-k by the indexer, or of 128, all of them),
  per-head sinks, the grouped low-rank output, sqrt(softplus) routing with
  hash-routed early layers, and clamped SwiGLU.
* The GPU does the matrix products; the CPU does the mixing, norms, rotary,
  compression, the indexer and attention (the cache lives in RAM), plus
  Q4_K experts from RAM that only a few tokens of a pass use.
* `PELICA_EXPERTS_GB` caps the VRAM for experts, `PELICA_RAM_GB` the RAM
  cache. Use history goes to `<model>.gguf.pelica-usage`.
* Status: it builds, and the kernels pass their tests, but it has not yet
  been run against a real DeepSeek V4 file or compared with llama.cpp's
  tokens. The MTP block, if the file has one, is skipped.

## Roadmap

1. ~~**GPU tier (wgpu)**~~ — done: dense weights and hot experts in VRAM.
2. ~~**Smarter tiers**~~ — done: heat eviction, I/O scheduler, cancellation.
   (Warm start from history exists but helps little on a cold prompt.)
3. ~~**GPU attention**~~ — done. ~~**Fewer bytes per token**~~ — packed
   scales, rank-heat eviction, asynchronous I/O at a fixed drive depth.
   Decode is SSD-bound; what is left is guess accuracy at long range (a wrong
   prefetch costs a whole expert) and faster storage.
4. **DSA indexer** for contexts beyond 2K, then MTP speculative decoding.
5. ~~More model families~~: any MoE GGUF (`src/moe.rs`), with tensor-core
   prompt passes and a KV cache that spills to RAM and the drive, attended
   over by the CPU past VRAM. Next: MLA attention (DeepSeek), an
   OpenAI-compatible server.
