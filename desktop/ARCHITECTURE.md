# Pelica Desktop — architecture and API contract

Pelica Desktop is an Electron + TypeScript app (`desktop/`) in front of the
Rust engine (`pelica.exe`). The split:

| Rust (`src/`) | TypeScript (`desktop/`) |
|---|---|
| loading models, inference, chat templates for every engine, tool-call parsing, token counts, engine stats | the window and UI, the agent loop, every tool (web, files, commands, MCP), approvals, compaction, CQS, chats storage |

The Electron main process starts `pelica.exe app --no-open --port <free> --ui <desktop/dist>`
and opens `http://127.0.0.1:<port>/` in a frameless `BrowserWindow`. The
Rust server serves the built UI from `--ui` (same origin as the API, so no
CORS). Without `--ui` it keeps serving the old `app.html`.

## Chat API v2 (Rust)

`POST /api/chat` keeps its v1 behaviour when `v` is absent. With `"v": 2` it is
a stateless, OpenAI-like completion that **stops at a tool call** and lets
the caller run the tool:

Request:
```json
{
  "v": 2,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "text before the call", "reasoning": "optional thinking",
     "tool_calls": [{"id": "c1", "name": "web_search", "arguments": {"query": "..."}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "result text"}
  ],
  "tools": [{"name": "web_search", "description": "...", "parameters": { "type": "object", ... }}],
  "effort": "off|low|medium|high|max|on",
  "max_tokens": 4096
}
```
- The server renders the conversation with the loaded model's own format:
  - gguf MoE: its Jinja chat template, with `tools` (OpenAI function schemas)
    when the template prints tools, else the tools described in the system
    message in the `<tool_call>{json}</tool_call>` format;
  - gpt-oss: harmony, tools as `functions` in the developer message, calls
    `to=functions.NAME`, results as `functions.NAME to=assistant` messages;
  - GLM (tiered): its template's tool format (or the words fallback).
- `system` content is used verbatim (the TS side owns all instructions,
  grounding text included). No server-side web tools in v2.
- Prefix reuse works across calls (the cache keeps the last rendered ids).

Server-sent events (`data: {...}\n\n`), in order:
| `t` | fields | meaning |
|---|---|---|
| `start` | `prompt`, `reused`, `ctx` | prompt tokens, how many came from the cache, context size |
| `prefill` | `done`, `total` | prompt reading progress |
| `think` | `d` | reasoning text |
| `text` | `d` | answer text (tool-call markup never appears here) |
| `stats` | `tps`, `generated`, `elapsed_s` | live speed, sent at most every 250 ms while decoding |
| `tool_call` | `id`, `name`, `arguments` (object) | the model called a tool; generation has stopped |
| `experts` | first `layers`, `experts`; then `hits`: flat `[layer, expert, picks, …]` | with `"experts": true` in the request (MoE): expert activity since the last event, for the brain map and the background |
| `done` | as v1, plus `finish`: `"stop"`, `"length"`, `"tool_call"`, `"stopped"` | end of the call |
| `error` | `error` | failure |

`POST /api/count` (extend): `{ "parts": {"system": "...", "messages": [...], "tools": [...]} }`
→ `{ "system": n, "messages": n, "tools": n, "total": n }` token counts with the
loaded tokenizer, for the context breakdown.

`POST /api/trace` `{ "text": "..." }` (MoE models): runs the text (at most
384 tokens) through the model's prompt pass and returns where each token
went: `{ "tokens": [str], "layers": L, "experts": E, "routes":
[[[expert, weight], …top-k] per layer] per token }`. The plan kernel writes
the routes into the engine's state buffer as it routes, so the trace is
the real routing, not a re-computation. It clears the prefix cache.

`GET /api/status` (extend): add `"tools": true` once v2 is available,
`"engine": "moe" | "oss" | "tiered"`, and `"agent": bool` once a model is
loaded: whether it was trained for tool calls (its chat template takes
tools; gpt-oss and GLM always). The app warns when it's false and a folder
is chosen, since such models describe tasks instead of doing them.

## TypeScript side

- `desktop/electron/` main + preload: window (frameless, custom title bar),
  spawning `pelica.exe`, folder picker, filesystem and command execution
  (sandboxed to the chosen workspace), links handed to the default browser,
  IPC.
- `desktop/src/` renderer (React + Vite): Claude-desktop-style UI (sidebar:
  Search, New chat, Models, Brain, Pinned, Recents with each chat's folder
  and a live dot on the chat being replied in; centered
  greeting + composer; settings modal with left nav), streaming markdown
  rendered live (no raw `**`), live tok/s, tool cards (searched / read /
  created file / edited file with diff / ran command with output), run boxes
  with a play button and mini output, context ring with a breakdown
  (system, messages, tools, MCP), slash commands (`/compact`,
  `/autocompact`, `/clear`, `/model`, `/context`, `/folder`, `/cqs`, ...).
- Agent loop (`desktop/src/agent/`): calls v2 until `finish != "tool_call"`,
  runs tools, enforces permissions, up to a high step limit for long tasks
  with a todo list tool. It works on its own copy of the chat (`beginLive` /
  `updateLive` / `endLive` in `state.ts`, `liveId` in the store), so the user
  can open other chats or start new ones while a reply is written; the other
  chats can be read and typed in, and send once the engine is free (one
  reply at a time: the engine answers one request).
- No model loads at start: main passes `--no-autoload` (without it the
  engine loads the first .gguf it finds), and adds `--model lastModel` only
  when `settings.loadOnStart` is on (Settings → Models, off by default).
- The Brain's map grows grander with the model's total parameters
  (`grandeur()` in `NeuralMap.tsx`, 0 at 2B to 1 at 500B on a log scale): a
  connection web from ~4B, drifting particles, a denser and brighter cortex,
  a cooler node tint, more signals at once, data rings from ~100B, and a
  nebula background per size class (`.nmap.grand-0..3`).
- The message box is one line: a + menu (files and photos, the chat's
  folder, web search, prompt maker, commands, compare), the text, send and
  the mic; under it the permission mode on the left and the model, effort
  and context ring on the right.
- The side panel (`components/SidePanel.tsx`, `settings.sidePanel`, title
  bar toggle, hidden under 1080 px) sits right of a chat: Environment (this
  computer and GPU, the chat's folder, permissions, web search), Context
  (the `/api/count` breakdown plus the auto-compact buffer and Compact
  session), Connectors (state and tool counts) and Outputs (files the chat
  wrote or edited, opened beside the chat). Folded sections are kept in
  `settings.panelFolded`.
- Folders are per chat (`chat.folder`, picked from the + menu or the side
  panel; `settings.recentFolders` keeps the last 8). The old global
  `settings.folder` only seeds the recent list.
- Permission modes (`settings.permMode`, composer chip, Shift+Tab, `/mode`):
  `ask` (file writes/edits/deletes and commands ask), `edits` (files
  automatic, commands ask; the default), `plan` (read-only tools only and a
  plan-mode system prompt), `auto` (nothing asks). Paths stay inside the
  chat's folder in every mode; deletes go to the Recycle Bin.
- Photos (`components/Attachments.tsx`): pasted, dropped or picked images are
  downscaled to 1568 px JPEG and kept on the user message (`images`). A
  model whose status says `vision: true` gets them as `images` (base64) on
  the v2 message; otherwise the `<attachment kind="image">` block carries the
  text Windows OCR reads (`media:ocr`, `electron/media.ts`,
  `Windows.Media.Ocr` through Windows PowerShell), labelled as OCR so the
  model doesn't claim to see the picture. The engine doesn't take images yet.
- Voice (`media:dictateStart` / `media:dictateStop`, events `media:dictate`):
  System.Speech dictation on the default microphone in a PowerShell child;
  each recognised phrase is appended to the message box.
- Prompt maker (`components/PromptMaker.tsx`): goal → the loaded model asks
  3–5 questions as JSON → answers → a tuned prompt (editable, tunable again,
  copy, use in chat, or append to the custom instructions).
- Model comparison (`components/Compare.tsx`): a local model next to GPT-6
  Astra, Claude Fable 5.1, Claude Opus 5.5, Kimi K3 and GLM 5.3. Scores are
  Vals AI's independent runs where they exist, else the vendor's own
  (striped, with the setup noted); missing scores show n/a. Update the
  tables there when new numbers are published.
- Connectors (`components/Connectors.tsx`): one-click MCP presets (files,
  memory, sequential thinking, fetch, git, time, SQLite, Playwright, GitHub,
  Brave Search) with the one field each needs, a custom-server form, per
  server on/off, restart (`mcp:restart`), tools and log, and the JSON under
  Advanced. `mcp:launchers` tells which of npx / uvx are installed;
  arguments with spaces are quoted for cmd.exe.
- Web search: `web_search` + `open_page` through Electron's `session.fetch`
  in the main process (DuckDuckGo, Bing, then a hidden rendered window), with
  local/private-address refusal. The system prompt and tool descriptions tell
  the model to search only when the request needs current or external facts
  (never for greetings or small talk).

## Windows integration (main process, `electron/main.ts`)

- One instance (`requestSingleInstanceLock`): a second start (the jump list's
  New chat / Quick ask, a shortcut) reaches the running one as
  `second-instance` with its flags (`--new-chat`, `--quick`, `--hidden`).
- Tray icon (the logo, cached in the user folder as `icon.png`) with Open,
  New chat, Quick ask, Brain, Models, Settings, Quit; its tooltip says what
  the model is doing. "Keep running in the tray" makes closing hide the window.
- The quick-ask bar is a second, transparent, always-on-top window loading
  the same UI at `#quick`; the global shortcut (`settings.hotkey`, default
  `Alt+Shift+Space`) toggles it. Enter answers in the bar itself (v2 chat,
  no tools, a short system prompt, `max_tokens` 1200, streamed Markdown);
  "Continue in Pelica" sends `quick:submit(text, reply)`, which reaches the
  main window as `app:action {kind:"ask", text, reply}` and becomes a new
  chat holding that exchange. Ctrl+Enter sends only the text, and the main
  window starts a new chat with it.
- `fs:raw` reads one workspace file as is (4 MB cap) for `@`-mentions in the
  message box (`fs:find "*"` lists the candidates, cached 15 s per folder).
- `sys:saveAs(name, text, dir)` shows a save dialog (the chat's workspace
  folder first, else Documents) and writes the exported chat; nothing is
  written without the dialog. `src/lib/exportChat.ts` builds the Markdown:
  messages in full, attachments as names, tool calls as one line each,
  thinking left out.
- Model downloads (`electron/library.ts`): `lib:pause` stops one and keeps
  `<file>.part` plus a `<file>.part.json` note (repo, file, size), so
  `lib:downloads(dir)` lists it as paused after a restart; `lib:resume`
  continues with an HTTP Range request (a run waits for the previous one to
  close its file); `lib:cancel` sends both files to the Recycle Bin.
- Workspace roots only grow through the folder picker: `settings:set`
  ignores `roots`, and every `fs:*` call checks its root against them.
- The renderer reports progress with `app:busy` (`setProgressBar`: the prompt
  pass as a fraction, generation as indeterminate) and asks for
  notifications with `app:notify`, shown only when the window isn't focused
  (with `flashFrame`); clicking one opens that chat.
- `backgroundMaterial` (`settings.material`: none / mica / acrylic) with a
  transparent window background; the CSS lets it through the sidebar and
  main area (`:root[data-material]`), `nativeTheme` follows the app theme.
- Start with Windows uses `setLoginItemSettings` with `--hidden` (off unless
  the user turns it on).
- `sys:*` IPC for the computer tools: `system_info` (no user name or paths),
  clipboard read and write, `open_file` (`shell.openPath`, inside the
  workspace). Reading the clipboard and opening files go through the same
  approval dialog as commands.

## Installer (`desktop/installer/`, `npm run installer`)

No downloads: everything comes from `node_modules` and Windows itself.

- `make.mjs` stages `release/stage/Pelica`: Electron's files with
  `electron.exe` renamed to `Pelica.exe`, `resources/app` (package.json,
  `build/electron`, `dist`) and `resources/pelica.exe` (the engine).
- `icons.cjs` runs in Electron: an offscreen window draws `public/logo.svg`
  at 16–256 px into `pelica.ico` (PNG entries) and `logo.png`.
- `buildtool.cs`, compiled with the .NET Framework 4 `csc.exe` that ships
  with Windows (C# 5: no `$""`, `?.` or `=>` members), stamps the icon and
  version info into both exes (`UpdateResource`) and zips the payload.
- `setup.cs` is the installer (WinForms, custom-painted, per-monitor DPI),
  with `payload.zip` embedded; built with `/define:UNINSTALL` it is the
  uninstaller. It installs per user to `%LOCALAPPDATA%\Programs\Pelica`,
  writes `uninstall.dat` (the files it put there, so uninstall and upgrades
  touch nothing else), shortcuts with AppUserModelID `Pelica.Desktop`, and
  `HKCU\...\Uninstall\Pelica`. A running Pelica from that folder is closed
  first. Flags: `/S`, `/D=<dir>`, `/NODESKTOP`, `/NOSHORTCUTS`, `/NOREG`,
  `/NOLAUNCH`, `/DATA` (uninstall: user data to the Recycle Bin),
  `/SHOT=<dir>` (renders every page to PNG, for checking the UI).
- The packaged app takes `--user-data-dir=<dir>` for a separate profile
  (settings, chats, its own single-instance lock), which is how it is
  tested beside an installed copy.
