"""Canonical map-chunk progress tracker (v109).

The byte at 0xE110 ("area id" in the runner) is not a free-roaming screen id. Across 40 independent v85 workers the
ordered list of ids visited was identical, including an exact four-fold repeat (26, 66, 25, 106) in mission 2 and an
eight-fold (16, 17) repeat on the mission-2 ladders - independent agents cannot loop a maze identically. It is the
index of the current map chunk, and chunks come in a fixed order. Deviations are deaths (the mission restarts).

That makes the sequence a free, exact progress oracle:
  * position in CANON (not pixels, not novelty) measures progress;
  * "the next chunk" is the right goal for the navigation planner (novelty fails inside the repeated stretches);
  * the old "looping" detector (an id re-entered 3x) fires falsely inside every repeated stretch.

CANON is the first pass of v85 worker 15 (the deepest run) up to the mission-4 arena (86), minus the one-frame
241 blip after the first 15 (a saved-state artefact the live tracker never sees - v110 lost the opening doorway to it). Nobody has been past the
arena, so beyond the end the tracker reports `beyond=True` and callers fall back to novelty.
"""
from __future__ import annotations

from typing import List, Optional, Set

_RAW: List = [
    15, 48, 84, 48, 106, 89, 106, 89, 106, 36, 106, 73, 74, 13, 75, 76, 22, 255, 32, 66, 106, 45,
    42, 43, 15, 106, 30, 26, 66, 25, 106, 26, 66, 25, 106, 26, 66, 25, 106, 26, 66, 25, 8, 7, 8, 7,
    8, 106, 73, 74, 13, 75, 76, 4, 5, 106, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17, 16, 17,
    16, 17, 16, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
    11, 12, 49, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
    12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11,
    12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 90, 244, 245, 246, 244, 12, 11, 12, 245, 246, 244,
    245, 246, 244, 245, 246, 244, 245, 246, 244, 245, 246, 244, 245, 246, 52, 53, 46, 11, 12, 11, 12, 11, 12, 11,
    12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 49, 11, 12, 11, 12, 11, 12,
    11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
    11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12,
    11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 11, 12, 90, 244, 245, 246, 244, 12, 11, 12, 11, 244, 245, 246,
    244, 245, 246, 244, 245, 246, 244, 245, 246, 244, 245, 246, 244, 245, 246, 52, 53, 46, 11, 46, 31, {36, 73}, 249, 248,
    250, {71, 72}, {87, 88}, {231, 232}, 247, 28, {44, 45}, 106, 27, 106, 44, 106, 27, 106, 44, 106, 27, 106, 44, 106, 234, 84, 85, 86,
]
CANON: List[Set[int]] = [x if isinstance(x, set) else {x} for x in _RAW]
M2_START = next(i for i, s in enumerate(CANON) if 66 in s)          # mission-2 start chunk
M3_START = next(i for i, s in enumerate(CANON) if 12 in s)          # forest
M4_START = next(i for i, s in enumerate(CANON) if 31 in s) - 2      # "11, 46, 31" after the bruiser
RESPAWN_IDS = {12, 11, 15, 66, 31}    # mission starts: a death past the arena restarts mission 3 (12/11)
LOOKAHEAD = 3          # chunks may be skipped by a fast scroll; also lets a 2-chunk cycle be recognised from either id


class ProgressTracker:
    """Follow the player's index in CANON from the stream of chunk ids seen each step."""

    def __init__(self, area0: Optional[int] = None):
        self.pos = -1
        self.max_pos = -1
        self.beyond = False
        self.pos_changed_step = 0
        self.regressions = 0
        self.beyond_chunks = 0          # v122: distinct chunk transitions seen past the arena
        self._last_area = None
        if area0 is not None:
            self.update(area0, 0)

    def update(self, area: int, step: int, life_lost_recently: bool = True) -> int:
        p = self.pos
        if 0 <= p < len(CANON) and area in CANON[p]:
            return p
        # v122: past the arena the cave reuses river/bridge ids (246, 244, 245, 234, 233, 86 ...). Once the arena index is
        # reached, only a genuine respawn id (mission starts) regresses the tracker; everything else is new ground.
        # v138: the hideout corridor reuses 11/12 too (the mission-3 start ids), so past the arena a respawn id only counts
        # when the runner saw a life lost recently; a death inside the hideout respawns at chunk 6 (still beyond).
        if p >= len(CANON) - 1 and (area not in RESPAWN_IDS or not life_lost_recently):
            self.beyond = True
            if area != self._last_area:
                self.beyond_chunks += 1; self.pos_changed_step = step
            self._last_area = area
            return p
        new = None
        for j in range(p + 1, min(p + 1 + LOOKAHEAD, len(CANON))):     # forward (allowing skips)
            if area in CANON[j]:
                new = j; break
        if new is None:                                                 # death / walked back: nearest earlier index
            for j in range(min(p, len(CANON)) - 1, -1, -1):
                if area in CANON[j]:
                    new = j; break
            if new is not None:
                self.regressions += 1
        if new is None:                                                 # unknown id (past the arena, or a cut-scene branch)
            anywhere = [j for j, s in enumerate(CANON) if area in s]
            if anywhere:
                new = anywhere[0]
            else:
                self.beyond = self.pos >= len(CANON) - 4                # unknown ids right after the arena = new ground
                return p
        if new != p:
            self.pos = new; self.pos_changed_step = step
            self.max_pos = max(self.max_pos, new)
            if new < len(CANON) - 1:
                self.beyond = False; self.beyond_chunks = 0; self._last_area = None
        return self.pos

    def goal_areas(self) -> Optional[Set[int]]:
        """Chunk ids that count as progress from here, or None when CANON is exhausted (fall back to novelty)."""
        if self.pos + 1 >= len(CANON) or self.beyond:
            return None                 # novelty mode (the nav planner's recent-window goal) past the arena
        g: Set[int] = set(); cur = CANON[self.pos] if 0 <= self.pos < len(CANON) else set(); n = 0
        for j in range(self.pos + 1, len(CANON)):                     # v111: the next LOOKAHEAD *distinct* chunks ahead
            if CANON[j] == cur:
                continue
            g |= CANON[j]; n += 1
            if n >= LOOKAHEAD:
                break
        return g - cur

    def mission(self) -> int:
        if self.pos < M2_START:
            return 1
        if self.pos < M3_START:
            return 2
        if self.pos < M4_START:
            return 3
        return 4
