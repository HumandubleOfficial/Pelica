// When a part of the window fails to draw, that part says so and the rest
// carries on, instead of the whole window going blank.
import { Component, type ErrorInfo, type ReactNode } from "react";

/** `reset`: when it changes (another chat opened) it tries drawing again. */
export class ErrorBoundary extends Component<{ what: string; reset?: unknown; children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info.componentStack);
  }

  componentDidUpdate(before: { reset?: unknown }) {
    if (this.state.error && before.reset !== this.props.reset) this.setState({ error: null });
  }

  render() {
    const e = this.state.error;
    if (!e) return this.props.children;
    return (
      <div className="crash" role="alert">
        <h2>{this.props.what} couldn't be shown</h2>
        <p className="muted">Something went wrong in Pelica. Your chats are safe: reloading brings everything back.</p>
        <div className="crash-btns">
          <button className="btn small primary" onClick={() => location.reload()}>
            Reload Pelica
          </button>
          <button className="btn small" onClick={() => navigator.clipboard.writeText(`${e.message}\n${e.stack || ""}`).catch(() => {})}>
            Copy details
          </button>
        </div>
      </div>
    );
  }
}
