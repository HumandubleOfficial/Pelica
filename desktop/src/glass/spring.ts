export interface SpringConfig {
  stiffness?: number;
  damping?: number;
  mass?: number;
}

/** Damped harmonic spring, integrated with fixed substeps for stability. */
export class Spring {
  value: number;
  velocity = 0;
  target: number;
  stiffness: number;
  damping: number;
  mass: number;

  constructor(value = 0, cfg: SpringConfig = {}) {
    this.value = value;
    this.target = value;
    this.stiffness = cfg.stiffness ?? 320;
    this.damping = cfg.damping ?? 26;
    this.mass = cfg.mass ?? 1;
  }

  set(target: number) {
    this.target = target;
    return this;
  }

  snap(value: number) {
    this.value = this.target = value;
    this.velocity = 0;
    return this;
  }

  impulse(v: number) {
    this.velocity += v;
    return this;
  }

  step(dt: number): number {
    const h = 1 / 240;
    let t = Math.min(dt, 0.064);
    while (t > 0) {
      const s = Math.min(h, t);
      const a = (-this.stiffness * (this.value - this.target) - this.damping * this.velocity) / this.mass;
      this.velocity += a * s;
      this.value += this.velocity * s;
      t -= s;
    }
    return this.value;
  }

  get settled() {
    return Math.abs(this.value - this.target) < 1e-3 && Math.abs(this.velocity) < 1e-3;
  }
}
