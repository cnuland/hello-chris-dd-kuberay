"""Navigation planner (v108): escape a stall by searching coarse moves on the live emulator.

The fight planner showed that the deterministic emulator plus save states turns a hard sub-problem into a search.
Navigation stalls are the other sink: the mission-2 maze loops four areas forever, a ladder 8 px off-axis eats 3000
"up" presses, a wall gets 3000 "left" presses. When the runner's no-progress detector fires, this searches macro
moves (12-step walks / climbs) breadth-first from the current state for any area id that has not been visited
recently, without losing a life, and returns the action sequence to replay.

Search facts from the mission-2 loop states: from area 66 the only exit is a long ladder descent (U/D macros) then
a walk to area 45 and on to area 5 - 9 macros deep, unreachable by any left/right-only policy.
"""
from __future__ import annotations

import collections
import io
import os
import time
from typing import Dict, List, Optional, Set

from radar import read_health, sprite_radar

LIVES_ADDR = 0xC499
MACRO_LEN = int(os.environ.get("DD_NAV_MACRO_LEN", "12"))
NAV_DEPTH = int(os.environ.get("DD_NAV_DEPTH", "24"))
NAV_TIME_S = float(os.environ.get("DD_NAV_TIME_S", "120"))   # v109: 300 s failures were the novelty goal chasing unreachable ids
NAV_HP_LOSS = int(os.environ.get("DD_NAV_HP_LOSS", "3"))      # ticks of HP a path may cost (rooms with thugs are never free)
NAV_PARTIAL_PX = int(os.environ.get("DD_NAV_PARTIAL_PX", "24"))
NAV_OPEN_PX = int(os.environ.get("DD_NAV_OPEN_PX", "96"))        # v111: displacement by depth 3 that means "open terrain, stop searching"  # v109: no goal reached -> replay the path that changed floor/scroll the most, if at least this many px
ENEMY_HP_ADDRS = (0xC595, 0xC5DA)
MACROS = ("right", "left", "up", "down", "right4", "left4", "right8", "left8", "ladder_up", "ladder_down", "punch", "throw")   # v131: "throw" = B - a player holding a barrel ignores every other input (mission-1 warehouse doorway stalls in v128/v130)


class NavPlanner:
    def __init__(self, env, action_names: List[str], log=None):
        self.env = env
        self.pb = env.pyboy
        self.idx = {n: action_names.index(n) for n in ("right", "left", "up", "down", "a", "b")}
        self.log = log
        self.plans = 0
        self.last_was_exit = False
        self.chain = False                                  # v113: mission 4 - hanging chains are ladders

    def _snap(self) -> bytes:
        b = io.BytesIO(); self.pb.save_state(b); return b.getvalue()

    def _restore(self, s: bytes) -> None:
        self.pb.load_state(io.BytesIO(s))

    def _key(self):
        p = sprite_radar(self.pb, river=False, chain=self.chain).get("player")
        return (self.pb.get_memory_value(0xE110), self.pb.get_memory_value(0xFF43) // 8, self.pb.get_memory_value(0xFF42) // 8,
                (p["x"] // 8) if p else -1, (p["feet_y"] // 8) if p else -1, bool(p and p.get("climbing")),
                sum(self.pb.get_memory_value(a) for a in ENEMY_HP_ADDRS))   # v109: enemy HP in the key, else every punch is a "seen" state

    def _macro(self, m: str) -> List[int]:
        """Expand a macro from the CURRENT emulator state (ladder macros need the live radar)."""
        if m in ("right", "left", "up", "down"):
            return [self.idx[m]] * MACRO_LEN
        if m in ("right4", "left4"):
            return [self.idx[m[:-1]]] * 4
        if m in ("right8", "left8"):
            return [self.idx[m[:-1]]] * 8
        if m == "throw":
            return [self.idx["b"]] + [self.idx["down"]] * 4
        if m == "punch":
            r0 = sprite_radar(self.pb, river=False, chain=self.chain)
            if not any(e["reachable"] and abs(e["dx"]) <= 40 for e in r0.get("enemies", [])):
                return []                    # nothing to punch
            p0 = r0.get("player"); e0 = min((e for e in r0["enemies"] if e["reachable"]), key=lambda e: abs(e["dx"]))
            turn = [self.idx[e0["side"]]] if p0 and p0.get("facing") != e0["side"] else []
            return turn + [self.idx["a"]] * 8
        # ladder macros: walk to within 3 px of the nearest ladder's axis, then climb 20 steps
        r = sprite_radar(self.pb, river=False, chain=self.chain); p = r.get("player")
        lad = min((l for l in r.get("ladders", []) if abs(l["dx"]) <= 40), key=lambda l: abs(l["dx"]), default=None)
        if not p or lad is None:
            return []
        acts: List[int] = []
        dx = lad["dx"]; step = 3
        n_align = min(12, abs(dx) // step)
        acts += [self.idx["right" if dx > 0 else "left"]] * n_align
        acts += [self.idx["up" if m == "ladder_up" else "down"]] * 20
        return acts

    def plan(self, recent_areas: Set[int], river: bool = False, goal_areas: Optional[Set[int]] = None, max_hp_loss: Optional[int] = None) -> Optional[List[int]]:
        """Breadth-first over macro moves; goal = the next canonical chunk(s) when goal_areas is given (v109), else any
        area id not in recent_areas; reached alive with at most NAV_HP_LOSS ticks lost. Restores the emulator before
        returning. Returns action indices or None."""
        root = self._snap(); t0 = time.time()
        L0 = self.pb.get_memory_value(LIVES_ADDR); H0 = read_health(self.pb); area0 = self.pb.get_memory_value(0xE110)
        seen = {self._key(): ""}; frontier = [(root, [])]; found = None
        scx0, scy0 = self.pb.get_memory_value(0xFF43), self.pb.get_memory_value(0xFF42)
        partial = None                                                   # (displacement px, path) of the farthest-moved healthy state
        partials: List = []                                              # v133: top candidates, verified for survival before one is returned
        try:
            for depth in range(NAV_DEPTH):
                nxt = []
                for st, path in frontier:
                    if time.time() - t0 > NAV_TIME_S:
                        break                                                     # v111: the cap used to be checked once per depth level (174 s runs at a 120 s cap)
                    for m in MACROS:
                        self._restore(st)
                        acts = self._macro(m)
                        if not acts:
                            continue
                        dead = False
                        for ai in acts:
                            self.env._run_action_on_emulator(ai)
                            if self.pb.get_memory_value(LIVES_ADDR) < L0:
                                dead = True; break
                        if dead or read_health(self.pb) < H0 - (NAV_HP_LOSS if max_hp_loss is None else max_hp_loss):   # v153g: the hideout corridor passes 0 (stalactites)
                            continue
                        pm = sprite_radar(self.pb, river=False, chain=self.chain).get("player")
                        if pm is not None and pm.get("feet_y", 0) > 122:
                            continue                                                   # v118: below the floor = falling (life drop lags the fall)
                        # v129: the v128 "vanished in goal mode = fall" prune made the mission-1 doorway (chunk 84, player hidden
                        # in the door after the fight) unsolvable: 'no exit found (states 1)' 38 times, two workers spent all
                        # 12000 steps there. Falls are rejected by the survival check after the goal test instead.
                        if pm is None:
                            # v125: vanished = a fall OR a door/scene transition (the hideout door on the cave's third screen). Idle
                            # and see which: a life lost is a fall; the player reappearing or the chunk changing is a transition.
                            fell = True
                            for _w in range(60):
                                for _t in range(8): self.pb.tick()
                                if self.pb.get_memory_value(LIVES_ADDR) < L0:
                                    break
                                pm = sprite_radar(self.pb, river=False, chain=self.chain).get("player")
                                if pm is not None or self.pb.get_memory_value(0xE110) != area0:
                                    fell = False; break
                            if fell:
                                continue
                        k = self._key(); via_idle = False
                        if k in seen and k[0] == area0 and goal_areas is None:
                            # v125: a macro that changed nothing may still have started a scene transition (the hideout door needs
                            # ~20 held UP presses and fades during the following idle). Idle briefly and look for the chunk change.
                            for _w in range(40):
                                for _t in range(8): self.pb.tick()
                                if self.pb.get_memory_value(LIVES_ADDR) < L0 or self.pb.get_memory_value(0xE110) != area0:
                                    break
                            if self.pb.get_memory_value(LIVES_ADDR) < L0:
                                continue
                            k = self._key(); via_idle = self.pb.get_memory_value(0xE110) != area0
                        if k in seen:
                            continue
                        seen[k] = path + acts
                        dsx = ((self.pb.get_memory_value(0xFF43) - scx0 + 128) % 256) - 128
                        dsy = ((self.pb.get_memory_value(0xFF42) - scy0 + 128) % 256) - 128
                        disp = dsx + (3 * max(0, -dsy) if goal_areas is None else abs(dsy))   # v111: signed - no walking back left; v124: past the arena (novelty mode) climbing UP outranks walking - the cave's lower walkway ends in a void and its exit is the chain
                        if read_health(self.pb) >= H0 - 1 and (partial is None or disp > partial[0]):
                            partial = (disp, path + acts)
                        if read_health(self.pb) >= H0 - 1 and disp >= NAV_PARTIAL_PX:
                            partials.append((disp, path + acts))
                            if len(partials) > 12:
                                partials.sort(key=lambda t: -t[0]); del partials[6:]
                        if (k[0] in goal_areas) if goal_areas else (k[0] not in recent_areas and k[0] != area0):
                            # v128: a chunk change during a FALL also passes the goal test (the mission-2 lower floor of 25/26: a 4-step
                            # "exit" was walking off the ledge; the life is lost a second later). Idle and require the player alive,
                            # visible and on a floor before accepting.
                            ok = True
                            for _w in range(40):
                                for _t in range(8): self.pb.tick()
                                if self.pb.get_memory_value(LIVES_ADDR) < L0:
                                    ok = False; break
                            pm2 = sprite_radar(self.pb, river=False, chain=self.chain).get("player")
                            for _w in range(40):                                        # v129: a door fade keeps the sprite hidden a while longer; a fall costs the life
                                if pm2 is not None or not ok:
                                    break
                                for _t in range(8): self.pb.tick()
                                if self.pb.get_memory_value(LIVES_ADDR) < L0:
                                    ok = False; break
                                pm2 = sprite_radar(self.pb, river=False, chain=self.chain).get("player")
                            if ok and (pm2 is None or pm2.get("feet_y", 0) > 122 or read_health(self.pb) <= 0):
                                ok = False
                            if not ok:
                                continue
                            # v135: an exit that only appeared during the post-macro idle (door fade) needs the input HELD through
                            # the fade - v134's door-room worker replayed the 12 UP presses, the model's next action cancelled the
                            # entry, and nav reported the same 12-step exit three times
                            found = path + acts + (acts[-1:] * 24 if via_idle else []); break
                        nxt.append((self._snap(), path + acts))
                    if found:
                        break
                if found or not nxt or time.time() - t0 > NAV_TIME_S:
                    break
                if depth >= 2 and partial and partial[0] >= NAV_OPEN_PX and goal_areas:
                    break                                                     # v111: open terrain (a 3-macro walk already moved far, goal chunk out of range) - return the walk, do not burn 120 s
                frontier = nxt
        finally:
            self._restore(root)
            self.env.kick_penalty = False
        self.plans += 1
        seq = list(found) if found else None
        self.last_was_exit = bool(found)
        what = f"exit in {len(seq)} steps" if found else "no exit found"
        if seq is None and partials:
            # v133: a partial walk must also survive a 40-step idle (the cave's upper walkway: a 66 px rightward "partial"
            # ended on the last pixels of a pillar and the model walked into the gap; lower walkway: the void at its end)
            for disp_, path_ in sorted(partials, key=lambda t: -t[0])[:6]:
                self._restore(root)
                for ai in path_:
                    self.env._run_action_on_emulator(ai)
                ok = self.pb.get_memory_value(LIVES_ADDR) == L0
                for _w in range(40):
                    if not ok:
                        break
                    for _t in range(8): self.pb.tick()
                    ok = self.pb.get_memory_value(LIVES_ADDR) == L0
                pm3 = sprite_radar(self.pb, river=False, chain=self.chain).get("player")
                if ok and pm3 is not None and pm3.get("feet_y", 0) <= 122:
                    seq = list(path_); what = f"partial {len(seq)} steps (moved {disp_}px, no exit)"   # v109: e.g. climb to the thug floor and let the model fight
                    break
            self._restore(root); self.env.kick_penalty = False
        msg = "nav-planner area=%d %s -> %s in %.0fs (states %d, plans so far %d)" % (
            area0, ("goal=%s" % sorted(goal_areas)) if goal_areas else ("recent=%s" % sorted(recent_areas)[:8]), what, time.time() - t0, len(seen), self.plans)
        print(msg, flush=True)
        if self.log:
            self.log.info(msg)
        return seq
