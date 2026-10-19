"""Online fight planner (v82): beam search over the env's own single-step actions from the live emulator state.

The emulator is deterministic given a save state and an input sequence, and the enemies' HP live at fixed WRAM
addresses (slot bytes 0xC595 / 0xC5DA: 42 for the mission-3 bruiser, 12 for a thug). So when a worker meets the
bruiser it can search its exact situation offline for a few minutes, restore the pre-search state, and replay the
winning sequence through the normal env.step (traces and video stay intact). Offline the same search killed the
bruiser with <= 1 HP lost from 9/9 saved arrival states (the hand-written autopilot: 5/8 from the easiest one).

Usage from the runner:
    planner = FightPlanner(env)                      # once per worker
    seq = planner.plan(area_id)                      # None if no kill found within the budget
Then feed seq's action indices into env.step one per step.
"""
from __future__ import annotations

import io
import os
import time
from typing import Dict, List, Optional

from radar import read_health, sprite_radar

ENEMY_HP_ADDRS = (0xC595, 0xC5DA)
ALT_HP_ADDRS = (0xC57D, 0xC5C2)     # v112: the mission-4 arena's second wave (13/20-HP thugs) keeps HP here (same 0x45 stride, 0x18 below)
ALT2_HP_ADDRS = (0xC583, 0xC5C8)    # v145: the hall past the sealed room (chunks 97/96): its 10-HP thugs keep HP here (0x06 above the ALT slots) - a RAM diff on a landed hit found 10 -> 9 at both
ALT3_HP_ADDRS = (0xC588, 0xC5CD)    # v150: the hall's later waves / lock room 2 (0x0B above the ALT slots) - the discovery pass found them in run 52; lock room 2's thugs read as untracked without this
DISC_MIN_HP = 4                     # v153c: minimum HP a discovered (hit-verified) slot must show
ALT_MIN_HP = 10                     # v112: a real enemy shows >= 10 HP once active; walk-in counters at these bytes flicker 0..6
ALT_KNOWN_CHUNKS = {86}             # v115: the arena's second wave lives at ALT (13 -> 11 -> 9 on jabs); skip the stability test there. NOT chunk 48: the second boss's bytes there are an animation counter (2,4,0,5,9,10 while idle) and he is not meant to be killed - the room is left, not cleared
HP_SCAN = (0xC540, 0xC700)          # v111: enemy object memory; the mission-4 arena's second wave keeps HP at other offsets (0xC57D/0xC5C2)
LIVES_ADDR = 0xC499
PLAN_ACTIONS = ("left", "right", "up", "down", "a", "b", "jump_kick")
PLAN_BEAM = int(os.environ.get("DD_PLAN_BEAM", "24"))
PLAN_DEPTH = int(os.environ.get("DD_PLAN_DEPTH", "260"))
SAFE_EXIT_PX = int(os.environ.get("DD_PLAN_SAFE_EXIT_PX", "28"))
PARTIAL_STEPS = int(os.environ.get("DD_PLAN_PARTIAL_STEPS", "60"))
PARTIAL_AT_DEPTH = int(os.environ.get("DD_PLAN_PARTIAL_AT_DEPTH", "160"))
ESCAPE_PX = int(os.environ.get("DD_PLAN_ESCAPE_PX", "150"))          # escape mode: win once this many px of rightward progress are banked (or the area id changes)
ESCAPE_DEPTH = int(os.environ.get("DD_PLAN_ESCAPE_DEPTH", "420"))    # escape searches may span several waves   # v95: every real kill so far was found by depth ~150; stop and go partial after this   # how much of a non-winning best leaf to replay before re-planning
PLAN_TIME_S = float(os.environ.get("DD_PLAN_TIME_S", "1500"))
PLAN_HP_W = float(os.environ.get("DD_PLAN_HP_W", "12"))
SAFE_AFTER = int(os.environ.get("DD_PLAN_SAFE_AFTER", "10"))        # v113: a kill only counts if backing away for this many steps costs no HP (3 of 5 arena ticks landed the step a plan ended)              # v109: cost of one HP tick lost (vs 10 per enemy HP dealt); arena arrivals have ~4 HP for 4 fights      # safety cap only: the depth is the real limit (v88; 420 s failed under CPU contention)


class FightPlanner:
    def __init__(self, env, action_names: List[str], log=None):
        self.env = env
        self.pb = env.pyboy
        self.idx = {n: action_names.index(n) for n in PLAN_ACTIONS if n in action_names}
        self.log = log
        self.plans = 0
        self.slot = ENEMY_HP_ADDRS[0]
        self.slots = (ENEMY_HP_ADDRS[0],)
        self.scx0, self.x0 = 0, 0
        self.simple = False
        self.escape = False
        self.area0 = 0
        self.probe_max = 0

    # -- emulator helpers -------------------------------------------------------------------------------------
    def _snap(self) -> bytes:
        b = io.BytesIO(); self.pb.save_state(b); return b.getvalue()

    def _restore(self, s: bytes) -> None:
        self.pb.load_state(io.BytesIO(s))

    def _obs(self, river: bool) -> Dict:
        r = sprite_radar(self.pb, river=river)
        p = r.get("player")
        hv = [e for e in r["enemies"] if e.get("heavy")] or r["enemies"]
        e = min(hv, key=lambda e: abs(e["dx"])) if hv else None
        ehp = sum(self.pb.get_memory_value(a) for a in self.slots)
        nz = sum(1 for a in self.slots if self.pb.get_memory_value(a) > 0)
        area_now = self.pb.get_memory_value(0xE110)
        others = sum(self.pb.get_memory_value(a) for a in ENEMY_HP_ADDRS if a not in self.slots and set(self.slots) <= set(ENEMY_HP_ADDRS))   # v111: discovered slots -> no clearance against the (unrelated) known bytes
        scx = self.pb.get_memory_value(0xFF43)
        prog = ((scx - self.scx0 + 128) % 256) - 128 + ((p["x"] - self.x0) if p else 0)   # px gained to the right since the root
        near = min((abs(e["dx"]) for e in r["enemies"] if abs(e["dy"]) <= 24), default=None)
        return dict(x=p and p["x"], feet=p and p["feet_y"], facing=p and p["facing"], dx=e and e["dx"], dy=e and e["dy"],
                    visible=bool(r["enemies"]), ehp=ehp, near=near, prog=prog, others=others, nz=nz, area=area_now,
                    hp=read_health(self.pb), lives=self.pb.get_memory_value(LIVES_ADDR), score=self._score())

    def _score(self) -> int:
        return sum(self.pb.get_memory_value(0xC640 + i) * 10 ** i for i in range(6))

    def discover_slots(self, river: bool = False, max_steps: int = 140) -> tuple:
        """v111: find the HP bytes of the enemies on screen empirically. From a snapshot, walk at the nearest enemy and
        jab; at the first score gain, the bytes in the enemy window that dropped by a hit amount (2-6, from 6-64) are
        the HP bytes. Restores the emulator. Needed because the arena's second wave (and possibly other enemy types)
        does not use 0xC595/0xC5DA - the planner saw enemy_hp=0 and never fought them."""
        root = self._snap(); found = (); first = None; hits = 0
        lo, hi = HP_SCAN
        try:
            prev = {a: self.pb.get_memory_value(a) for a in range(lo, hi)}; s0 = self._score()
            L0 = self.pb.get_memory_value(LIVES_ADDR)
            since_hit = {}                                                            # max value of each candidate since the first hit (HP never rises)
            for _ in range(max_steps):
                # v134: discovery runs on a snapshot that is restored afterwards, so the player's HP byte (0xC498, HUD ticks =
                # value // 8) is held at 3 ticks during the probe: at 2 HP the first enemy hit ended the probe at once (0 s,
                # nothing found) and the door-room thugs stayed invisible to the planner
                if self.pb.get_memory_value(0xC498) < 24:
                    self.pb.set_memory_value(0xC498, 24)
                r = sprite_radar(self.pb, river=river); es = r["enemies"]; p = r.get("player")
                if not es:
                    for _t in range(8): self.pb.tick()
                else:
                    e = min(es, key=lambda e: abs(e["dx"]))
                    if abs(e["dx"]) > 14:
                        self.env._run_action_on_emulator(self.idx[e["side"]])
                    elif abs(e["dy"]) > 4:
                        self.env._run_action_on_emulator(self.idx["down" if e["dy"] > 0 else "up"])
                    else:
                        if p and p.get("facing") != e["side"]:
                            self.env._run_action_on_emulator(self.idx[e["side"]])
                        self.env._run_action_on_emulator(self.idx[("a", "a", "a", "b", "jump_kick")[_ % 5]])   # v115: vary attacks - jabs score on the second boss without damaging him
                now = {a: self.pb.get_memory_value(a) for a in range(lo, hi)}; s1 = self._score()
                if first:
                    for a in first:
                        since_hit[a] = max(since_hit.get(a, 0), now[a])
                if s1 > s0:
                    drops = tuple(a for a in range(lo, hi) if 2 <= prev[a] - now[a] <= 6 and 6 <= prev[a] <= 64)
                    if drops and first is None:
                        first = drops; hits = 1; since_hit = {a: now[a] for a in first}
                        self._first_vals = {a: prev[a] for a in first}
                    elif drops and first is not None:
                        # v111b: HP = dropped at both hits and never rose in between (stun timers and counters fail this)
                        both = tuple(a for a in first if a in drops and since_hit.get(a, 0) <= prev[a])
                        if both:
                            found = both; hits = 2; break
                prev, s0 = now, s1
                if self.pb.get_memory_value(LIVES_ADDR) < L0 or read_health(self.pb) <= 0:
                    break
        finally:
            self._restore(root); self.env.kick_penalty = False
        # v112: no single-hit fallback - one hit also drops stun timers and counters (0xC5BF, 0xC5CD, 0xC585 were picked)
        self.probe_max = max((getattr(self, "_first_vals", {}).get(a, 0) for a in found), default=0) if found else 0   # HP shown at the first hit (0 at the root while he walks in)
        if found:
            print("fight-planner discovered HP bytes %s (hp %d)" % ([hex(a) for a in found], self.probe_max), flush=True)
        return found

    def _stable(self, addrs, steps: int = 6) -> bool:
        """v115: HP does not change while nobody lands a hit; a timer does. Idle a few steps on a copy and require the
        bytes to hold (the mission-1 second boss made the ALT bytes read a 14/4 flicker that passed the >= 10 test, so
        the planner 'killed' a timer while he threw the player)."""
        root = self._snap()
        try:
            v0 = [self.pb.get_memory_value(a) for a in addrs]
            for _ in range(steps):
                for _t in range(8): self.pb.tick()
                if [self.pb.get_memory_value(a) for a in addrs] != v0:
                    return False
            return True
        finally:
            self._restore(root)

    def _activation_wait(self, addrs, max_wait: int = 48) -> bool:
        """v112: enemies walking in have no HP yet. Idle up to max_wait steps on a copy and report whether these bytes
        reach ALT_MIN_HP - cheap (no search) and avoids a 90 s search that can never see a kill."""
        root = self._snap(); ok = False
        try:
            for _ in range(max_wait):
                for _t in range(8): self.pb.tick()
                if max(self.pb.get_memory_value(a) for a in addrs) >= ALT_MIN_HP:
                    ok = True; break
        finally:
            self._restore(root)
        return ok

    def _grounded_after(self, river: bool, hp: int, steps: int = 16) -> bool:
        """v153p: from the current emulator state, idle (punch) for `steps`; True if the player stays on screen, never reads as
        falling (feet > 122), keeps the life and the HP. Escape wins by progress used to accept a player in mid-flight."""
        feet0 = None
        for _ in range(steps):
            self.env._run_action_on_emulator(self.idx["a"])
            p = sprite_radar(self.pb, river=river).get("player")
            if p is None or p.get("feet_y", 0) > 122 or self.pb.get_memory_value(LIVES_ADDR) < self._L0 or read_health(self.pb) < hp:
                return False
            feet0 = p["feet_y"]
        return True

    def _safe_after(self, river: bool, hp: int) -> bool:
        """v113: from the current emulator state, back away from the nearest enemy for SAFE_AFTER steps; True if no HP
        is lost. The caller restores state afterwards (the beam loop restores the parent before every expansion)."""
        _p0 = sprite_radar(self.pb, river=river).get("player"); _feet0 = _p0["feet_y"] if _p0 else None
        for _ in range(getattr(self, "safe_after", SAFE_AFTER)):   # v153m: the runner raises it on the forest ledge (two runs thrown into the river right after a planned kill)
            r = sprite_radar(self.pb, river=river); p = r.get("player")
            _floor = getattr(self, "feet0", None) or _feet0
            if not self.escape and p and _floor is not None and p["feet_y"] < _floor - 12 and not p.get("climbing"):   # v153ae: against the search's starting floor - the win state itself can be mid-jump-kick
                return False                     # v153ad: lifted off the floor = grabbed and thrown (ladder1/2: a kill plan ended with a second thug adjacent; the throw cost no HP inside the window, then the player flew off the catwalk)
            es = [e for e in r.get("enemies", []) if abs(e["dx"]) <= 48 and abs(e["dy"]) <= 24]   # v114: only a live, close enemy is worth backing away from (walking blindly fell off the chunk-16 ledge)
            live = sum(self.pb.get_memory_value(a) for a in ENEMY_HP_ADDRS + ALT_HP_ADDRS) > 0
            if es and p and live:
                e = min(es, key=lambda e: abs(e["dx"]))
                self.env._run_action_on_emulator(self.idx["left" if e["dx"] > 0 else "right"])
            else:
                self.env._run_action_on_emulator(self.idx["a"])
            if read_health(self.pb) < hp or self.pb.get_memory_value(LIVES_ADDR) < self._L0:
                return False
            if not self.escape and (p is None or p.get("feet_y", 0) > 122):
                return False                     # v118: vanished or below the floor during the safety window = a fall in progress
        self.settle_seq = []
        if not self.escape and not getattr(self, "long_settle", False):
            # v154c: every kill must END GROUNDED. f13 trace: a 43-step kill plan's last action was a running kick (the hit that
            # killed) - the player was airborne at the win, _safe_after's lift check used the lane the plan started on, and the
            # flight carried him off the forest ledge 10 steps after the plan ended. Idle until the feet hold still for 3 steps
            # (appended to the plan), fail on a fall / vanish / lost life or HP.
            still = 0; last_f = None
            for _ in range(16):
                p = sprite_radar(self.pb, river=river).get("player")
                if p is None or p.get("feet_y", 0) > 122 or self.pb.get_memory_value(LIVES_ADDR) < self._L0 or read_health(self.pb) < hp:
                    return False
                still = still + 1 if (last_f is not None and abs(p["feet_y"] - last_f) <= 1) else 0
                last_f = p["feet_y"]
                if still >= 3:
                    break
                self.env._run_action_on_emulator(self.idx["a"]); self.settle_seq.append(self.idx["a"])
            else:
                return False
        if not self.escape and getattr(self, "long_settle", False):
            # v131: the hideout's door-2 bruiser died with the player still airborne from his last throw; he landed on the
            # EDGE of a moving slab, stood there 9 steps, and fell when the slab drifted away (life lost 20 steps after the
            # kill). Wait until the player is standing; on slabs also walk to the slab's centre and survive a full 24-step
            # idle. The walk is appended to the returned plan (settle_seq) so the replay ends in the same safe spot.
            still = 0; last_f = None
            for _i in range(48):                     # v131: landing (~10 steps) + a full slab swing (24) + margin
                r = sprite_radar(self.pb, river=river, cave=True); p = r.get("player")
                if p is None or p.get("feet_y", 0) > 100 or self.pb.get_memory_value(LIVES_ADDR) < self._L0:
                    return False
                if read_health(self.pb) < hp:
                    return False
                # v136: the settle checks the LANDING, not the next fight - with another live thug closing in (hideout room 5)
                # every kill was rejected by the hit taken during the idle ("no kill found" at 6 HP vs a 10-HP thug). Once the
                # player stands on a floor and an enemy with HP is within 48 px, accept and let the next plan handle him.
                live = sum(self.pb.get_memory_value(a) for a in ENEMY_HP_ADDRS + ALT_HP_ADDRS) > 0
                if (_i >= 6 and live and abs(p["feet_y"] - getattr(self, "feet0", p["feet_y"])) <= 4
                        and any(e["reachable"] and abs(e["dx"]) <= 48 and abs(e["dy"]) <= 24 for e in r.get("enemies", []))):
                    return True
                slabs = r.get("slabs_visible")
                # v131: step away from any pit edge within 12 px (the door-2 room is a solid ledge with a void on its left; the
                # kill left the player at x 101-103 on a ledge starting at 96 - half his sprite over the void - and he dropped
                # the moment the bruiser's dying sprite stopped overlapping him). Slab gaps are pits too, so this also centres him on a slab.
                near = min(r.get("pits", []), key=lambda g: abs(g["dx_edge"]), default=None)
                if near is not None and abs(near["dx_edge"]) <= 20 and p["feet_y"] >= 80:   # 20: the tile edge reads ~8 px inside the real drop (x 103 fell on a "96" edge)
                    a = "right" if near["side"] == "left" else "left"
                    self.env._run_action_on_emulator(self.idx[a]); self.settle_seq.append(self.idx[a])
                else:
                    for _t in range(8): self.pb.tick()
                still = still + 1 if last_f is not None and abs(p["feet_y"] - last_f) <= 1 else 0
                last_f = p["feet_y"]
                # v131: no early exit - the slab under a thrown player can be invisible (OAM budget) and drift away 9 steps later
            p = sprite_radar(self.pb, river=river).get("player")
            f0 = getattr(self, "feet0", 90)
            # v131: the slab sprites are dropped from OAM while the big bruiser is on screen, so the void under a thrown
            # player is invisible here; the only reliable sign is the player NOT being back at the floor level the plan
            # started from (feet 99 and sinking passed the old > 100 test and the life was lost 2 steps later)
            if p is None or p.get("feet_y", 0) > 100 or self.pb.get_memory_value(LIVES_ADDR) < self._L0 or (f0 <= 95 and abs(p.get("feet_y", 0) - f0) > 4):
                return False
        return True

    # -- search ------------------------------------------------------------------------------------------------
    danger = None                       # v148b: optional callable (pyboy, x, feet) -> bool marking lethal standing zones; set by the runner

    def plan(self, area_id: int, river: bool = True, progress_w: float = 0.0, simple: bool = False, escape: bool = False, escape_px: Optional[int] = None, escape_depth: Optional[int] = None, beam_width: Optional[int] = None, long_settle: bool = False, escape_chunk_win: bool = True, max_hp_loss: Optional[int] = None, time_s: Optional[float] = None, require_verified: bool = False, alt_layouts: bool = True) -> Optional[List[int]]:
        """Beam search for a sequence that zeroes the target enemy's HP without losing a life. Restores the
        emulator to the pre-search state before returning. Returns action indices or None."""
        # v84: target the slot holding the most HP at the root (the bruiser/boss), not the mean of both slots: with an
        # empty second slot the mean reached 0 at 1 HP, so v83's plans ended with him alive (11 elbows, no kill).
        vals = {a: self.pb.get_memory_value(a) for a in ENEMY_HP_ADDRS}
        discovered = False
        if require_verified and not escape:
            # v153e: hideout rooms 6/5 - their thugs' bytes vary per visit and the main/ALT pairs there read flickering counters
            # (14, 9, 1, 16, 8, 0 while idling) that passed the static tests and drew 67-s searches (fullgame4, corr1/corr3,
            # even with 0xC595 nonzero at the root). Only a hit-verified slot that then holds still counts; otherwise the
            # runner's knockdown search (score-gain win) owns the room.
            disc = self.discover_slots(river)
            if disc and self.probe_max >= DISC_MIN_HP and self._stable(disc):
                vals = {a: self.pb.get_memory_value(a) for a in disc}; discovered = True
                print("fight-planner verified slots %s" % [hex(a) for a in disc], flush=True)
            else:
                return None
        elif max(vals.values()) == 0 and not escape:
            alt = {a: self.pb.get_memory_value(a) for a in ALT_HP_ADDRS}
            alt2 = {a: self.pb.get_memory_value(a) for a in ALT2_HP_ADDRS}
            if (max(alt.values()) >= ALT_MIN_HP or self._activation_wait(ALT_HP_ADDRS)) and (self.pb.get_memory_value(0xE110) in ALT_KNOWN_CHUNKS or self._stable(ALT_HP_ADDRS)):
                vals = alt; discovered = True                                          # v112: known layout of the arena's second wave
                self.probe_max = max(alt.values())
            elif alt_layouts and max(alt2.values()) >= ALT_MIN_HP and self._stable(ALT2_HP_ADDRS):   # v153j: hall layouts only (runner passes alt_layouts=maze_entered) - in the mission-3 forest 0xC583 read a stable counter and drew three 72-s searches before a knockback into a pit (fullgame5 step 4115)
                vals = alt2; discovered = True                                         # v145: the hall's third slot layout
                self.probe_max = max(alt2.values())
            elif alt_layouts and max(self.pb.get_memory_value(a) for a in ALT3_HP_ADDRS) >= ALT_MIN_HP and self._stable(ALT3_HP_ADDRS):
                vals = {a: self.pb.get_memory_value(a) for a in ALT3_HP_ADDRS}; discovered = True   # v150: the hall's fourth slot layout
                self.probe_max = max(vals.values())
            else:
                disc = self.discover_slots(river)                                     # v111: enemies on screen but the known slots are empty
                if disc and self.probe_max >= DISC_MIN_HP:
                    vals = {a: self.pb.get_memory_value(a) for a in disc}; discovered = True   # v153c: discovery already requires a hit-sized drop at a score gain, so a 10-HP thug seen at 7 after the probe jab counts (fullgame3: the mission-4 door-room thug pair at 0xC588/0xC5CD was found at hp 7 and rejected 8 times in a row while it took HP 6 -> 2)
                else:
                    print("fight-planner no HP slot for the enemy on screen (area %d) - knockdown fallback" % self.pb.get_memory_value(0xE110), flush=True)   # v153z: the silent return hid 13-step gaps
                    return None
        self.slot = max(vals, key=vals.get)
        # v89: a thug pair (both slots <= 20 HP) is one target - a plan that kills only one ends with the other
        # mid-swing and the next search starts from a doomed 1-HP state. Big enemies keep the single-slot target.
        if simple or escape:
            self.slots = tuple(a for a, v in vals.items() if v > 0) or (self.slot,)   # v98: any enemy dying is the win (the max-HP one may be unreachable); escape: damage to anyone counts
        else:
            self.slots = tuple(a for a, v in vals.items() if v > 0) if (max(vals.values()) <= 20 and sum(1 for v in vals.values() if v > 0) == 2) else (self.slot,)
        self.simple = simple                 # v95: rooms with respawning thugs (mission-1 warehouse): kill the strongest one, no clearance test
        self.long_settle = long_settle       # v133: the 48-step post-kill settle is for mission 4's ledges/slabs only - in the warehouse (respawning thugs) it rejected every kill and three v132 workers were beaten to death there
        self.escape = escape                 # v99: gauntlet rooms (mission-4 arena): the win is the screen scrolling on, not a kill
        self.area0 = self.pb.get_memory_value(0xE110)
        p0 = sprite_radar(self.pb, river=river).get("player") or {}
        self.scx0, self.x0 = self.pb.get_memory_value(0xFF43), p0.get("x", 0)
        self.feet0 = p0.get("feet_y", 90)        # v131: the floor level the plan starts from; the settle test wants the player back there
        root = self._snap(); o0 = self._obs(river); t0 = time.time()
        L0, H0, E0 = o0["lives"], o0["hp"], o0["ehp"]
        self._L0 = L0; S0 = o0["score"]
        if E0 <= 0 and discovered:
            E0 = max(self.probe_max, ALT_MIN_HP)                                   # v111: he has not activated yet; damage is measured against the max HP seen on each path
        elif E0 <= 0 and not escape:
            return None
        min_emax = (min(ALT_MIN_HP, max(self.probe_max, DISC_MIN_HP)) if discovered else 1)   # v153c: a discovered 7-HP thug is a kill at 0 too                                   # v112: alternate/discovered bytes must show a real enemy before a zero counts as a kill

        def score(o, gone, steps, emax, dmg):
            if o["lives"] < L0 or (o["hp"] <= 0 and (o["visible"] or not self.escape)) or (max_hp_loss is not None and o["hp"] < H0 - max_hp_loss):   # v138: corridor escapes must be damage-free (the boulder cost 2 HP per accepted plan)
                return None                      # v115: HUD reads 0 during the mission-end cut-scene (no enemy on screen) - not a death in escape mode
            if self.danger is not None and o["x"] is not None and self.danger(self.pb, o["x"], o["feet"]):
                return None                      # v148b: a runner-supplied lethal zone (the hall's ceiling shafts) - never plan through it
            if not self.escape and (o["x"] is None or (o["feet"] is not None and o["feet"] > 122)):
                return None                      # v118: the player sprite is gone or below the floor = falling into a pit (the life counter drops ~15 steps later; 10 v117 workers died in the forest gap at x 108 right after planned kills)
            s = dmg * 10.0 - (H0 - o["hp_eff"]) * PLAN_HP_W - steps * 0.02 + progress_w * o["prog"]   # v111b: only hit-sized drops count (a 13 -> 0 slot reset is not a kill)
            if self.escape:
                s += 0.05 * (o["score"] - S0)      # v115: in escape mode landed hits matter even without an HP byte (the mission-1 second boss leaves after one 200-point knockdown, then the room ends)
            if o["ehp"] == 0 and emax >= min_emax and dmg >= emax - 1:
                s += 30.0 + (min(o["near"], 60) if o["near"] is not None else 60) * 0.5   # target down: now get clear
            if not o["visible"] and o["ehp"] < emax:
                s += 2.0 * min(gone, 8)       # v109: off-radar bonus only after damage and capped - in the arena it paid 300 points for walking away
            if o["dy"] is not None and abs(o["dy"]) <= 2:
                s += 1.0
            return s

        def key(o):
            return (o["x"] // 4 if o["x"] is not None else -1, o["feet"], o["facing"], o["ehp"], o["hp"],
                    (o["dx"] // 4) if o["dx"] is not None else None, o["visible"])

        if os.environ.get("DD_PLAN_DEBUG"):
            print(f"fight-planner search start: area {self.area0} slot {hex(self.slot)} slots {[hex(a) for a in self.slots]} E0 {E0} vals {{{', '.join(hex(a)+':'+str(v) for a, v in vals.items())}}} discovered {discovered} verified {require_verified} escape {escape}", flush=True)
        beam = [dict(state=root, seq=[], gone=0, sc=0.0, emax=o0["ehp"], ehp=o0["ehp"], dmg=0, hp=H0)]        # v111: emax = max target HP seen on the path (0 until a walking-in enemy activates)
        best = None
        base_beam = beam_width or PLAN_BEAM                                       # v119: the runner widens the beam in mission 4 (bridge + arena kills were 1-tick at 24, 0-tick at 40 locally)
        beam_w = base_beam if E0 > 20 else max(8, base_beam // 2)       # v88: thugs (<= 12 HP) fall in ~40 steps; half the beam is plenty
        depth_max = (escape_depth or ESCAPE_DEPTH) if escape else (PLAN_DEPTH if E0 > 20 else min(PLAN_DEPTH, 140))   # v115: escape searches span the room's timer regardless of enemy HP
        partial_at = depth_max if escape else PARTIAL_AT_DEPTH
        try:
            for depth in range(depth_max):
                cand = {}
                for node in beam:
                    for name, ai in self.idx.items():
                        self._restore(node["state"])
                        self.env._run_action_on_emulator(ai)
                        o = self._obs(river)
                        o["hp_eff"] = o["hp"] if o["hp"] > 0 else node["hp"]          # v115: last valid reading while the HUD is blank
                        gone = node["gone"] + 1 if not o["visible"] else 0
                        emax = max(node["emax"], o["ehp"])
                        d = node["ehp"] - o["ehp"]
                        dmg = node["dmg"] + (d if 0 < d <= 6 and node["emax"] >= min_emax else 0)   # v112: count hits only once the enemy has shown real HP (pre-activation counters flicker 0..6)
                        sc = score(o, gone, depth + 1, emax, dmg)
                        if sc is None:
                            continue
                        escaped = self.escape and ((escape_chunk_win and o["area"] != self.area0) or o["prog"] >= (escape_px if escape_px is not None else ESCAPE_PX))   # v138: escape_chunk_win=False where the chunk id flips every few px (hideout corridor/stairs)
                        if escaped or (not self.escape and emax >= min_emax and dmg >= emax - 1 and ((o["ehp"] == 0 and (o["others"] == 0 or o["near"] is None or o["near"] > SAFE_EXIT_PX)) or (self.simple and o["nz"] < o0["nz"]))):
                            # v92: the clearance test only matters while an untargeted enemy still has HP - the dying
                            # target's own sprite lingers within 28 px and made bruiser searches run to full depth (520 s)
                            # v89: the kill only counts once nobody is within SAFE_EXIT_PX - a plan that ends with
                            # another thug mid-swing hands the model a doomed state
                            win_state = self._snap()
                            _by_progress = escaped and not (escape_chunk_win and o["area"] != self.area0)
                            if (escaped and (not _by_progress or self._grounded_after(river, o["hp_eff"]))) or (not escaped and self._safe_after(river, o["hp_eff"])):   # v153p: a progress win must end on the ground - corr7: a corridor escape ended mid-air over the slab gap (feet 58 -> 111 -> dead)
                                win = dict(seq=node["seq"] + [ai] + (list(getattr(self, "settle_seq", [])) if not escaped else []), hp=o["hp_eff"], steps=depth + 1)
                                if best is None or (win["hp"], -win["steps"]) > (best["hp"], -best["steps"]):
                                    best = win
                                continue
                            # v113: the kill lands but a hit follows within SAFE_AFTER steps - keep searching from here
                            self._restore(win_state)
                            o = self._obs(river)
                            o["hp_eff"] = o["hp"] if o["hp"] > 0 else node["hp"]     # v116: the re-read observation needs the field too (KeyError crashed v115 workers)
                        k = key(o)
                        if k in cand and cand[k]["sc"] >= sc:
                            continue
                        cand[k] = dict(state=self._snap(), seq=node["seq"] + [ai], gone=gone, sc=sc, ehp=o["ehp"], emax=emax, dmg=dmg, hp=o["hp_eff"], feet=o["feet"], near=o["near"])
                if not cand and depth < 3 and os.environ.get("DD_PLAN_DEBUG_SAVE"):
                    self._restore(root)
                    with open(os.environ["DD_PLAN_DEBUG_SAVE"] % self.plans, "wb") as fh:
                        self.pb.save_state(fh)                                     # v115 debug: a search that dies within 3 levels - what is the root?
                    print("plan-debug saved dead-root state (depth %d, hp %d, lives %d)" % (depth, read_health(self.pb), self.pb.get_memory_value(LIVES_ADDR)), flush=True)
                if not cand or (best and depth > best["steps"] + 3) or time.time() - t0 > (time_s if time_s is not None else PLAN_TIME_S) or (best is None and depth + 1 >= partial_at):   # v139: per-call budget (the arena escape at 2 HP needs ~10 min; the 600 s default cut it to a partial)
                    break
                beam = sorted(cand.values(), key=lambda n: -n["sc"])[:beam_w]
        finally:
            self._restore(root)
            self.env.kick_penalty = False
        partial = None
        if best is None and beam:
            # v90: no kill-and-clear within the budget. Replay the best leaf's first steps anyway (damage dealt, HP
            # kept) and let the runner re-plan from there instead of handing a 489 s search back as nothing.
            hurt = [n for n in beam if n.get("dmg", 0) > 0
                    and (n.get("feet") is None or abs(n["feet"] - self.feet0) <= 4)
                    and (self.danger is None or n.get("near") is None or n["near"] > 24)]   # v150: where lethal floor exists (the hall), a partial must not end with a thug in grab range - run 35 lost three lives to throws into shafts right after plans ended   # v109: a partial plan must have dealt damage (v85 arena arrivals retreated to chunk 44 on damage-free partials); v148: and end with the player ON THE FLOOR - run 29 lost all three lives to throws whose arc began inside a partial's last steps and landed in a hall hole after the plan ended
            leaf = max(hurt, key=lambda n: n["sc"]) if hurt else None
            if leaf and leaf["sc"] > 5.0 and len(leaf["seq"]) >= 8:
                partial = dict(seq=leaf["seq"][:PARTIAL_STEPS], sc=leaf["sc"])
        if partial and not best and os.environ.get("DD_PLAN_DEBUG"):
            traj = []
            for ai in partial["seq"]:
                self.env._run_action_on_emulator(ai); o = self._obs(river); traj.append((o["ehp"], o["x"], o["feet"], o["hp"], o["lives"]))
            self._restore(root); self.env.kick_penalty = False
            names = {v: k for k, v in self.idx.items()}
            print("plan-debug partial seq:", " ".join(names[a] for a in partial["seq"]), flush=True)
            print("plan-debug partial expected (ehp,x,feet,hp,lives):", traj, flush=True)
        if best and os.environ.get("DD_PLAN_DEBUG"):
            # re-simulate the winning sequence from the root and print the expected HP trajectory, then restore
            traj = []
            for ai in best["seq"]:
                self.env._run_action_on_emulator(ai); o = self._obs(river); traj.append((o["ehp"], o["x"], o["feet"]))
            self._restore(root); self.env.kick_penalty = False
            names = {v: k for k, v in self.idx.items()}
            print("plan-debug seq:", " ".join(names[a] for a in best["seq"]), flush=True)
            print("plan-debug expected (ehp,x,feet):", traj, flush=True)
        self.plans += 1
        msg = "fight-planner area=%d enemy_hp=%d player_hp=%d -> %s in %.0fs (plans so far %d) [slot 0x%X]" % (
            area_id, E0, H0, f"kill in {best['steps']} steps, hp left {best['hp']}" if best else
            (f"partial plan {len(partial['seq'])} steps (score {partial['sc']:.0f})" if partial else "no kill found"), time.time() - t0, self.plans, self.slot)
        print(msg, flush=True)                 # stdout: Ray forwards worker stdout to the job log; worker logging does not surface
        if best:
            names = {v: k for k, v in self.idx.items()}
            print("fight-plan-seq " + " ".join(names[a] for a in best["seq"]), flush=True)
        if self.log:
            self.log.info(msg)
        return best["seq"] if best else (partial["seq"] if partial else None)
