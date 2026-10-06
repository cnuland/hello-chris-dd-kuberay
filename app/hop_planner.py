"""Verified hop (v123): the mission-4 upper walkway is pillars separated by ~28 px of void + non-solid chain column.
The running jump kick (right+A+B) crosses only when launched 7-10 px before the edge, and the landing is marginal to
2-3 px and sensitive to frame alignment (the same launch point landed with one input path and fell with another).
So instead of a fixed window, try the hop on the live emulator: launch now, or after 1-4 more steps right, or after
1-2 steps left; keep the first launch whose flight lands standing on the far side. ~120 emulator steps, restores."""
from __future__ import annotations

import io
import os
from typing import List, Optional

from radar import sprite_radar

LIVES_ADDR = 0xC499
PRE_OPTIONS = tuple([["right"] * k for k in range(0, 13)] + [["left"], ["left"] * 2, ["left"] * 4 + ["right"] * 4, ["left"] * 2 + ["right"] * 2])   # v127: up to ~42 px of approach - a climb can end well short of the hole; v129: a step back and forth changes the frame alignment at the same launch point
# v129: the hideout's stepping stones are moving slabs (sprites, ~24 px wide, swinging ~24 px with a ~24-step period), so
# the SAME launch point lands or falls depending on WHEN you jump. Wait 0-22 steps (standing still on the slab) before a
# short run; also a step back and forth, which changed the outcome on the fixed pillars too (frame alignment).
PRE_OPTIONS_TIMED = tuple([["down"] * w + ["right"] * k for w in range(0, 24, 2) for k in (0, 2, 4)] + [["left"] * 4 + ["right"] * 4, ["left"] * 2 + ["right"] * 2])


class HopPlanner:
    def __init__(self, env, action_names: List[str]):
        self.env = env
        self.pb = env.pyboy
        self.idx = {n: action_names.index(n) for n in ("left", "right", "kick_right", "kick_left", "down", "a")}
        self.plans = 0

    def _snap(self) -> bytes:
        b = io.BytesIO(); self.pb.save_state(b); return b.getvalue()

    def _state(self):
        p = sprite_radar(self.pb).get("player")
        return (p and (self.pb.get_memory_value(0xFF43) + p["x"]), p and p["feet_y"])

    def _edge_dx(self, side: str = "right") -> int:
        """Distance to the nearest pit edge on the given side of the player (999 if none in view)."""
        r = sprite_radar(self.pb, cave=True)
        d = [abs(g["dx_edge"]) for g in r.get("pits", []) if g.get("side") == side and (g["dx_edge"] >= -4 if side == "right" else g["dx_edge"] <= 4)]
        return min(d) if d else 999

    def _right_edge_dx(self) -> int:
        return self._edge_dx("right")

    @staticmethod
    def _mirror(pre: List[str]) -> List[str]:
        return [{"left": "right", "right": "left"}.get(a, a) for a in pre]

    def plan(self, flight: int = 16, min_advance: int = 30, timed: bool = False, direction: str = "right", drop: bool = False) -> Optional[List[int]]:
        """v143: direction="left" mirrors everything (kick_left, walk left in flight, advance measured leftward); drop=True also accepts a
        landing 8-28 px LOWER than the launch floor (the maze's step-down ledges and the drop-kick from ledge C to the lower platform -
        walking off those edges is death, only the running kick reaches the lower floor)."""
        root = self._snap(); L0 = self.pb.get_memory_value(LIVES_ADDR); wx0, feet0 = self._state()
        found = None
        fwd, kick = ("right", "kick_right") if direction == "right" else ("left", "kick_left")
        sgn = 1 if direction == "right" else -1
        if timed:
            min_advance = min(min_advance, 16)       # a slab drifting left under the landing shortens the measured advance
        try:
            # v129: the fixed-window pass first; if nothing lands, the timed variants (waits change the frame alignment - the
            # pillar hop at the slab approach landed after a 6-step walk and failed from the same pixel without it)
            for pre0 in PRE_OPTIONS + PRE_OPTIONS_TIMED:
                pre = pre0 if direction == "right" else self._mirror(list(pre0))
                self.pb.load_state(io.BytesIO(root)); ok = True
                for a in pre:
                    self.env._run_action_on_emulator(self.idx[a])
                    wx, feet = self._state()
                    if wx is None or feet is None or feet > (feet0 or 80) + 6 or self.pb.get_memory_value(LIVES_ADDR) < L0:
                        ok = False; break                                   # walked off before launching
                if not ok:
                    continue
                if timed:
                    wx_l, _f = self._state(); wx0 = wx_l if wx_l is not None else wx0   # advance counts from the launch point (the slab may have carried us left while waiting)
                self.env._run_action_on_emulator(self.idx[kick]); seq = [self.idx[a] for a in pre] + [self.idx[kick]]
                prev_feet = None; airborne = True
                for i in range(flight + (8 if drop else 0)):
                    # v143b: the kick's flight does not depend on the held direction (identical trajectories with 2-10 presses), but
                    # pressing it after touchdown WALKS the player off a short ledge (ledge B is 16 px). Hold the direction only
                    # while airborne, then idle (punch) and read the landing.
                    press = fwd if airborne else "a"
                    self.env._run_action_on_emulator(self.idx[press]); seq.append(self.idx[press])
                    wx, feet = self._state()
                    if wx is None or feet is None or feet > (feet0 or 80) + (34 if drop else 8) or self.pb.get_memory_value(LIVES_ADDR) < L0:
                        break                                               # fell
                    if i >= 2 and feet >= (feet0 or 76) - 2:
                        airborne = False
                    stable = prev_feet is not None and abs(feet - prev_feet) <= 1
                    level = abs(feet - (feet0 or 76)) <= 2 and stable
                    lower = drop and 8 <= feet - (feet0 or 76) <= 28 and stable       # v143: a lower landing counts only once the height holds for two steps (the fall passes through every height in range)
                    prev_feet = feet
                    adv = ((wx - (wx0 or 0) + 128) % 256) - 128                 # v143: SCX is one byte - the drop from ledge C wraps 230 -> 8
                    if i >= 6 and (level or lower) and sgn * adv >= min_advance:
                        land_feet = feet
                        # standing on the far side: walk up to 3 more steps onto the platform (the landing spot is its edge - the
                        # model pressed back there and walked straight into the void), verifying each step
                        settled = True
                        for _ in range(3):
                            # v143: the ledge row past door 2 has 16 px ledges between 16-24 px gaps - three blind steps walked
                            # straight off the far edge (the fight planner then started from a falling player: "no kill found in 0s"
                            # at chunk 27 in every v141 maze arrival). Step on only while the next edge ahead is > 12 px away;
                            # otherwise stand still (punch is the only idle that neither drops nor walks).
                            # v143b: the tile pit reading is blind when the player stands on a ledge's last pixels (the ledge-row
                            # edge at x 113 showed no pits at all) - probe the step on the emulator instead: step, idle twice, and
                            # keep it only if the feet never move; otherwise stand still (punch).
                            probe = self._snap(); self.env._run_action_on_emulator(self.idx[fwd]); stable = True
                            for _k in range(3):
                                wx2, feet2 = self._state()
                                if wx2 is None or feet2 is None or abs(feet2 - land_feet) > 2:
                                    stable = False; break
                                if _k < 2:
                                    self.env._run_action_on_emulator(self.idx["a"])
                            self.pb.load_state(io.BytesIO(probe))
                            _step = fwd if stable else "a"
                            self.env._run_action_on_emulator(self.idx[_step]); seq.append(self.idx[_step])
                            wx2, feet2 = self._state()
                            if wx2 is None or feet2 is None or feet2 > land_feet + 4:
                                settled = False; break
                        for _ in range(2):                                      # v143: two idle steps - a landing on the last pixel shows its fall here, not in the settle
                            if not settled:
                                break
                            self.env._run_action_on_emulator(self.idx["a"]); seq.append(self.idx["a"])
                            wx2, feet2 = self._state()
                            if wx2 is None or feet2 is None or abs(feet2 - land_feet) > 2:
                                settled = False
                        if settled and timed:
                            # v153k: a slab landing is only safe if the player RIDES the slab afterwards - fullgame5 died 2 steps after a
                            # verified landing at the slab's edge (the pit reading showed the gap under the player right after touchdown)
                            # because the slab drifted on. Follow the slab's centre for 14 steps (half a swing); reject the launch if
                            # the player still falls.
                            for _ in range(14):
                                r = sprite_radar(self.pb); pp = r.get("player")
                                if not pp or abs(pp["feet_y"] - land_feet) > 4 or self.pb.get_memory_value(LIVES_ADDR) < L0:
                                    settled = False; break
                                under = [sl for sl in r.get("platforms", []) if -6 <= pp["feet_y"] - sl["y"] <= 8 and sl["left"] - 10 <= pp["x"] <= sl["right"] + 10]
                                if under:
                                    sl = min(under, key=lambda sl: abs((sl["left"] + sl["right"]) / 2 - pp["x"]))
                                    cx = (sl["left"] + sl["right"]) / 2
                                    _a = "right" if cx - pp["x"] > 4 else ("left" if pp["x"] - cx > 4 else "a")
                                else:
                                    _a = "a"
                                self.env._run_action_on_emulator(self.idx[_a]); seq.append(self.idx[_a])
                            if settled:
                                pp = sprite_radar(self.pb).get("player")
                                if not pp or abs(pp["feet_y"] - land_feet) > 4 or self.pb.get_memory_value(LIVES_ADDR) < L0:
                                    settled = False
                        if os.environ.get("DD_HOP_DEBUG"):
                            print("hop-debug pre=%s land i=%d feet=%s adv=%d settled=%s tail=%s" % ("".join(a[0] for a in pre) or "-", i, land_feet, adv, settled, [self._state()[1]]), flush=True)
                        if settled:
                            found = seq
                        break
                if found:
                    break
        finally:
            self.pb.load_state(io.BytesIO(root)); self.env.kick_penalty = False
        self.plans += 1
        print("hop-planner %s%s (plans so far %d)" % (("verified hop: %d steps" % len(found)) if found else "no landing launch found",
                                                      "" if (direction == "right" and not drop) else f" [{direction}{' drop' if drop else ''}]", self.plans), flush=True)
        return found
