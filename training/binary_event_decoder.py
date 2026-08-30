"""
Binary Event Decoder for High-Throughput Step Events.
Unpacks packed 16-byte fixed-header frame payloads using Python struct.
"""

import struct
from typing import Dict, Any, List, Optional, Tuple


class BinaryEventDecoder:
    """
    Decodes the 16-byte fixed-header binary event payload:
      Offset 0..1 (uint16): event_bitmask
      Offset 2    (int8)  : ball_owner_id (-1 if free ball)
      Offset 3    (int8)  : scorer_id (-1 if none)
      Offset 4    (int8)  : pass_from_id (-1 if none)
      Offset 5    (int8)  : pass_to_id (-1 if none)
      Offset 6    (int8)  : dispossessed_id (-1 if none)
      Offset 7    (uint8) : reserved padding (0)
      Offset 8..11(float32): ball_x
      Offset 12..15(float32): ball_y
    """

    # Format: uint16 (bitmask), 5x int8, 1x uint8 pad, 2x float32 (ball_x, ball_y)
    EVENT_STRUCT_FMT = "<Hbbbbbbff"
    EVENT_STRUCT_SIZE = struct.calcsize(EVENT_STRUCT_FMT)  # exactly 16 bytes

    def __init__(self, agent_id_map: List[str]):
        """
        agent_id_map: List mapping agent integer indices (0, 1, 2...) to agent ID strings (e.g. ['left_1', 'left_2', 'left_3']).
        """
        self.agent_id_map = list(agent_id_map)

    def _idx_to_agent(self, idx: int) -> Optional[str]:
        if 0 <= idx < len(self.agent_id_map):
            return self.agent_id_map[idx]
        return None

    def decode_step_payload(
        self, buffer: bytes, offset: int = 0
    ) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        """
        Unpacks event flags and state summary for CompositeRewardCalculator and RL loops.

        Returns:
            Tuple of (current_state, events, new_offset)
        """
        (
            bitmask,
            ball_owner_idx,
            scorer_idx,
            pass_from_idx,
            pass_to_idx,
            dispossessed_idx,
            _,
            ball_x,
            ball_y,
        ) = struct.unpack_from(self.EVENT_STRUCT_FMT, buffer, offset)

        events: Dict[str, Any] = {
            "goal_scored": bool(bitmask & (1 << 0)),
            "goal_conceded": bool(bitmask & (1 << 1)),
            "shot_on_target": bool(bitmask & (1 << 2)),
            "pass_completed": bool(bitmask & (1 << 3)),
            "dispossessed": bool(bitmask & (1 << 4)),
            "scorer_id": self._idx_to_agent(scorer_idx),
            "pass_from": self._idx_to_agent(pass_from_idx),
            "pass_to": self._idx_to_agent(pass_to_idx),
            "agent_id": self._idx_to_agent(dispossessed_idx),
        }

        current_state: Dict[str, Any] = {
            "ball_pos": (float(ball_x), float(ball_y)),
            "ball_owner": self._idx_to_agent(ball_owner_idx),
        }

        new_offset = offset + self.EVENT_STRUCT_SIZE
        return current_state, events, new_offset
