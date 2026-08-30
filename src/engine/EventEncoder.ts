/**
 * EventBitmask and binary encoding specification for high-throughput step events.
 * Provides a 16-byte fixed-header frame payload.
 */

export const EventBits = {
  GOAL_SCORED: 1 << 0, // 0x0001
  GOAL_CONCEDED: 1 << 1, // 0x0002
  SHOT_ON_TARGET: 1 << 2, // 0x0004
  PASS_COMPLETED: 1 << 3, // 0x0008
  DISPOSSESSED: 1 << 4, // 0x0010
} as const;

export interface StepEvents {
  goalScored: boolean;
  goalConceded: boolean;
  shotOnTarget: boolean;
  passCompleted: boolean;
  dispossessed: boolean;
  scorerId: number; // -1 if none
  passFromId: number; // -1 if none
  passToId: number; // -1 if none
  dispossessedId: number; // -1 if none
  ballOwnerId: number; // -1 if free ball
}

export class EventBinaryEncoder {
  public static readonly EVENT_STRUCT_SIZE = 16; // 16 bytes total

  /**
   * Packs event metadata and continuous ball tracking into a 16-byte DataView slice:
   * - Offset 0..1 (uint16): event bitmask
   * - Offset 2 (int8): ball_owner_id (-1 if none)
   * - Offset 3 (int8): scorer_id (-1 if none)
   * - Offset 4 (int8): pass_from_id (-1 if none)
   * - Offset 5 (int8): pass_to_id (-1 if none)
   * - Offset 6 (int8): dispossessed_id (-1 if none)
   * - Offset 7 (uint8): reserved/padding (0)
   * - Offset 8..11 (float32): ball_x (IEEE 754 float32, little-endian)
   * - Offset 12..15 (float32): ball_y (IEEE 754 float32, little-endian)
   */
  public static encodeEvents(
    view: DataView,
    offset: number,
    events: StepEvents,
    ballX: number,
    ballY: number
  ): number {
    let bitmask = 0;
    if (events.goalScored) bitmask |= EventBits.GOAL_SCORED;
    if (events.goalConceded) bitmask |= EventBits.GOAL_CONCEDED;
    if (events.shotOnTarget) bitmask |= EventBits.SHOT_ON_TARGET;
    if (events.passCompleted) bitmask |= EventBits.PASS_COMPLETED;
    if (events.dispossessed) bitmask |= EventBits.DISPOSSESSED;

    // Write uint16 bitmask (little-endian)
    view.setUint16(offset, bitmask, true);
    offset += 2;

    // Write int8 event metadata
    view.setInt8(offset++, events.ballOwnerId);
    view.setInt8(offset++, events.scorerId);
    view.setInt8(offset++, events.passFromId);
    view.setInt8(offset++, events.passToId);
    view.setInt8(offset++, events.dispossessedId);
    view.setUint8(offset++, 0); // Padding byte

    // Write ball positions for continuous tracking (float32 little-endian)
    view.setFloat32(offset, ballX, true);
    offset += 4;
    view.setFloat32(offset, ballY, true);
    offset += 4;

    return offset;
  }

  /**
   * Helper to allocate and create an ArrayBuffer / Uint8Array containing only the 16-byte event payload.
   */
  public static createEventBuffer(
    events: StepEvents,
    ballX: number,
    ballY: number
  ): Uint8Array {
    const buffer = new ArrayBuffer(this.EVENT_STRUCT_SIZE);
    const view = new DataView(buffer);
    this.encodeEvents(view, 0, events, ballX, ballY);
    return new Uint8Array(buffer);
  }
}
