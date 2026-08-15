import { Vector2D, Vector3D } from '../types/football';

export class Vec2 {
  static create(x = 0, y = 0): Vector2D {
    return { x, y };
  }

  static clone(v: Vector2D): Vector2D {
    return { x: v.x, y: v.y };
  }

  static add(a: Vector2D, b: Vector2D): Vector2D {
    return { x: a.x + b.x, y: a.y + b.y };
  }

  static sub(a: Vector2D, b: Vector2D): Vector2D {
    return { x: a.x - b.x, y: a.y - b.y };
  }

  static scale(v: Vector2D, s: number): Vector2D {
    return { x: v.x * s, y: v.y * s };
  }

  static length(v: Vector2D): number {
    return Math.sqrt(v.x * v.x + v.y * v.y);
  }

  static lengthSq(v: Vector2D): number {
    return v.x * v.x + v.y * v.y;
  }

  static distance(a: Vector2D, b: Vector2D): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  static normalize(v: Vector2D): Vector2D {
    const len = Vec2.length(v);
    if (len === 0) return { x: 0, y: 0 };
    return { x: v.x / len, y: v.y / len };
  }

  static dot(a: Vector2D, b: Vector2D): number {
    return a.x * b.x + a.y * b.y;
  }

  static angle(v: Vector2D): number {
    return Math.atan2(v.y, v.x);
  }

  static fromAngle(rad: number, len = 1): Vector2D {
    return { x: Math.cos(rad) * len, y: Math.sin(rad) * len };
  }

  static clampLength(v: Vector2D, maxLen: number): Vector2D {
    const len = Vec2.length(v);
    if (len <= maxLen || len === 0) return v;
    return Vec2.scale(v, maxLen / len);
  }

  static lerp(a: Vector2D, b: Vector2D, t: number): Vector2D {
    return {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
    };
  }
}

export class Vec3 {
  static create(x = 0, y = 0, z = 0): Vector3D {
    return { x, y, z };
  }

  static clone(v: Vector3D): Vector3D {
    return { x: v.x, y: v.y, z: v.z };
  }

  static add(a: Vector3D, b: Vector3D): Vector3D {
    return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
  }

  static sub(a: Vector3D, b: Vector3D): Vector3D {
    return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  }

  static scale(v: Vector3D, s: number): Vector3D {
    return { x: v.x * s, y: v.y * s, z: v.z * s };
  }

  static length(v: Vector3D): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  }

  static normalize(v: Vector3D): Vector3D {
    const len = Vec3.length(v);
    if (len === 0) return { x: 0, y: 0, z: 0 };
    return { x: v.x / len, y: v.y / len, z: v.z / len };
  }

  static lerp(a: Vector3D, b: Vector3D, t: number): Vector3D {
    return {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
      z: a.z + (b.z - a.z) * t,
    };
  }
}
