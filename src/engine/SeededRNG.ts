/**
 * Deterministic Seeded Pseudo-Random Number Generator (Mulberry32).
 * Ensures 100% reproducible trajectories for simulation, RL rollouts, and physics.
 */
export class SeededRNG {
  private state: number;

  constructor(seed = 0) {
    this.state = (seed >>> 0) || 0x6d2b79f5;
  }

  public setSeed(seed: number): void {
    this.state = (seed >>> 0) || 0x6d2b79f5;
  }

  public getSeedState(): number {
    return this.state;
  }

  /**
   * Returns a pseudo-random float in [0, 1).
   */
  public next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * Returns a pseudo-random float in [min, max).
   */
  public nextRange(min: number, max: number): number {
    return min + this.next() * (max - min);
  }

  /**
   * Returns a pseudo-random integer in [min, max] inclusive.
   */
  public nextInt(min: number, max: number): number {
    return Math.floor(this.nextRange(min, max + 1));
  }
}
