# Contributing to Pelica

Thanks for wanting to help. Bug reports, benchmark results from your hardware and pull requests are all welcome.

## Reporting a problem

Use the [bug report form](https://github.com/HumandubleOfficial/Pelica/issues/new?template=bug_report.yml). Performance and loading problems almost always depend on the hardware, so please include:

- your graphics card, RAM and drive,
- the model file you were using,
- the Pelica version (Settings → About).

The engine log helps too: copy the end of it from Settings → Developer → Engine log.

## Sharing benchmarks

Numbers from PCs other than the dev machine help a lot, especially AMD and Intel graphics cards, which haven't been tested yet. Open an issue with your hardware, the model, and the tok/s Pelica shows under a reply.

## Pull requests

1. Open an issue first for anything bigger than a small fix, so we can agree on the approach.
2. Build and check before sending:

   ```bash
   cargo build --release
   cargo test --release
   cd desktop
   npm run typecheck
   npm run build
   ```

3. Changes to the engine must not change the model's output. Speedups have to be lossless: the same tokens as before for the same prompt. Say in the pull request how you checked.
4. If the change is something users will notice, add a line to `desktop/src/changelog.json` under the `"next"` entry, written for users ("Prompts are read about twice as fast"), not developers.

## Where things are

| Path | What |
|---|---|
| `src/` | The Rust engine: model engines, GPU kernels (`*_gpu.rs`), the drive scheduler (`aio.rs`), the app server and OpenAI API |
| `desktop/electron/` | The desktop app's main process: windows, settings, connectors, updates |
| `desktop/src/` | The app's interface (React) and the agent loop (`agent/`) |
| `desktop/installer/` | The Windows installer |
| `docs/ENGINE.md` | How the engine works, every option, full benchmarks |

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
