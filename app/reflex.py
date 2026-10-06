# reflex.py
"""Frame-level reflexes the decision model cannot time itself (it acts every 1-2 env steps with ~1s latency).

Mission 2's upper walkway is a chain of CONVEYOR BELTS separated by gaps. Measured on the emulator:
  * a belt drifts the player ~3.6 px per 8-frame step; walking adds ~6 px/step, so you can walk against it
  * the only way across a gap is a jump kick toward it (LEFT+A+B / RIGHT+A+B) pressed while the gap edge is
    1..8 px ahead: earlier lands you in the gap, later you are already falling
  * belt tiles animate by cycling tile ids, so direction is measured from the player's drift, not the tilemap
The BeltTracker infers belt speed from the player's world-x change minus the movement the input should have
produced; belt_reflex() then takes over the last ~20 px of an approach to a gap and fires the jump in the window.
"""
from __future__ import annotations

from collections import deque
import os
from typing import Dict, Optional

BELT_MIN_SPEED = 2.5        # px/step of unexplained drift before we call it a conveyor
WALK_PX = 6                 # px/step a left/right press moves the player on a still floor
# gap edge distance (px) at which a jump clears it, by gap width. Measured: mission-2 belt gaps (24 px) 1..8 onto a
# still floor but only 2..5 when the landing is a belt running back toward the gap (6 tipped in); mission-3 river
# gaps (8 px stepping stones) 5..13, and 3 falls in. Approach steps are 3-4 px (belt drift, or walking while the
# camera scrolls), so a 4-wide window is always entered; overshooting it triggers one step back.
JUMP_WINDOW_WIDE = (2, 5)          # gap >= 16 px (conveyor room, jumping left)
JUMP_WINDOW_NARROW = (9, 12)       # gap < 16 px, jumping RIGHT (mission-3 stones). v52: the running jump is ballistic
                                   # (35 px, no air control: 24 trials identical) and lands 27-d px into the 24 px island;
                                   # (5,8) landed 3 px from the far edge and fell on 4/4 wide-gap-state runs, (9,12) 4/4 ok
JUMP_WINDOW_NARROW_LEFT = (13, 20) # gap < 16 px, jumping LEFT: dx_edge is measured from the sprite centre to the pit's
                                   # right edge, and the left foot is ~8 px ahead of the centre (measured: 13/16/20 clear,
                                   # 12 and below has already fallen)
BACK_OFF_MAX = 3            # after this many nudges just jump (a 6 px-per-step floor would otherwise loop 2->8->2)
APPROACH_ZONE = 28          # within this distance of a gap the reflex owns the approach
# The camera follows the player with lag and the player's screen-space speed is capped, so a jump taken while the
# camera is still catching up (player left of its anchor when travelling left) lands ~15px short: re-centre first.
CAMERA_ANCHOR = {"left": 68, "right": 108}
ANCHOR_SLACK = 6

LEFTWARD = ("left", "kick_left")
RIGHTWARD = ("right", "kick_right")


def _wrap(d: int) -> int:
    return (d + 128) % 256 - 128


BELT_PX_PER_STEP = 3.6


def belt_tile_under(pyboy, radar: Dict) -> Optional[int]:
    """Kept for the debug log only: belt surface tiles (249-251) and rollers animate by cycling ids, but the phase is
    global and does NOT encode direction, so direction is inferred from drift instead."""
    p = radar.get("player")
    if not p:
        return None
    try:
        scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
        tm = pyboy.botsupport_manager().tilemap_background()
        return tm[((p["x"] + scx) // 8) % 32, ((p["feet_y"] + scy) // 8) % 32]
    except Exception:
        return None


class BeltTracker:
    """Which way the walkway under the player moves, from drift the input does not explain:
    * up/down/attack pressed and the player still moved >= 2px => belt in that direction
    * right pressed but moved < 3px right (or left) => belt left; mirror for left
    * left pressed and moved ~3.5px left is ambiguous (belt with you caps your speed) and is ignored
    * up/down/attack pressed and no movement at all => still floor
    Measured on the emulator: idle drift 3.6 px/step, walking against the belt nets ~1.8 px/step."""

    def __init__(self, window: int = 5, stale_after: int = 40):
        self.hist: deque = deque(maxlen=window)
        self.prev = None            # (world_x, feet_y, health)
        self.last_dx = 0
        self.dir: Optional[str] = None
        self.age = 10**6            # steps since the direction was last confirmed
        self.stale_after = stale_after
        self.cooldown = 0           # steps to ignore after landing (recovery frames swallow input)
        self.airborne = False
        self.jump_dir: Optional[str] = None   # direction of the jump in progress (air control pulls the player back)
        self.ground_feet: Optional[int] = None  # feet_y of the last floor stood on (the jump apex has |dfeet|<=2)
        self.grounded_steps = 0
        self.stable_steps = 0
        self.backoffs = 0           # consecutive back-off nudges at a gap edge (the reflex caps these)

    def reset(self):
        self.hist.clear(); self.prev = None; self.last_dx = 0; self.dir = None; self.age = 10**6

    def update(self, radar: Dict, scx: int, action: Optional[str], health: int, tile: Optional[int] = None) -> Optional[str]:
        p = radar.get("player")
        self.age += 1
        if not p:
            self.prev = None; return self.direction
        world_x = (p["x"] + scx) % 256
        cur = (world_x, p["feet_y"], health)
        if self.prev is not None:
            dx = _wrap(world_x - self.prev[0]); self.last_dx = dx
            moving_v = abs(p["feet_y"] - self.prev[1]) > 2
            # v43: at the apex of a jump the feet barely move for a step, which read as "landed" and released the
            # air-control hold; anything well above the last floor stood on is still in the air
            # only during a jump: a lane change (up) also puts the feet above the previous floor (v44: every player that
            # walked up a lane was "airborne" for the rest of the episode and the advice froze)
            above_floor = self.jump_dir is not None and self.ground_feet is not None and p["feet_y"] < self.ground_feet - 6
            airborne = moving_v or above_floor
            hurt = health < self.prev[2]
            if self.airborne and not airborne:
                self.cooldown = 2
                self.jump_dir = None
            elif self.cooldown:
                self.cooldown -= 1
            self.airborne = airborne
            if not airborne:
                self.grounded_steps += 1
                if self.grounded_steps >= 2:
                    self.ground_feet = p["feet_y"]
            else:
                self.grounded_steps = 0
            # v47: a jump that lands on a HIGHER platform (tree stumps, feet ~64 vs ground 84) kept above_floor true
            # forever ("IN THE AIR ... keep pressing left" for 200 decisions). Stable feet for 3 steps = standing.
            self.stable_steps = self.stable_steps + 1 if not moving_v else 0
            if airborne and self.stable_steps >= 3:
                airborne = False; self.airborne = False; self.jump_dir = None; self.ground_feet = p["feet_y"]; self.grounded_steps = 2
            if action in ("kick_left", "kick_right") and self.jump_dir is None:
                self.jump_dir = "left" if action == "kick_left" else "right"
            found = None
            if not airborne and not hurt and not self.cooldown and action not in ("kick_left", "kick_right", "jump_kick", None):
                if action not in ("left", "right"):
                    found = ("left" if dx < 0 else "right") if abs(dx) >= 2 else ("none" if dx == 0 else None)
                else:
                    # walking speed is not a discriminator (a scrolling camera pins you to ~3.5px/step just like a
                    # belt does); the only unambiguous horizontal-input evidence is going nowhere or backwards
                    fwd = dx if action == "right" else -dx
                    at_edge = p["x"] >= 140 or p["x"] <= 20
                    if fwd < 0 and not at_edge:   # pushed backwards while walking forward (a wall, the screen edge or a
                                                  # knockback at the edge is not a belt; v48 sidecars still showed CONVEYOR there)
                        found = "left" if action == "right" else "right"
            if found is not None:
                self.hist.append(found)
                votes = {v: list(self.hist).count(v) for v in set(self.hist)}
                top = max(votes, key=votes.get)
                if votes[top] >= 3 or (len(self.hist) == 2 and votes[top] == 2):
                    self.dir = None if top == "none" else top
                    self.age = 0
        self.prev = cur
        return self.direction

    @property
    def direction(self) -> Optional[str]:
        return self.dir if self.age <= self.stale_after else None

    @property
    def known(self) -> bool:
        return self.age <= self.stale_after

    @property
    def speed(self) -> float:
        return -BELT_PX_PER_STEP if self.direction == "left" else (BELT_PX_PER_STEP if self.direction == "right" else 0.0)


ATTACKS = ("a", "b", "jump_kick", "kick_left", "kick_right")


def belt_reflex(radar: Dict, belt: "BeltTracker", model_action: str, recent: Optional[list] = None) -> Optional[str]:
    """Override the model's action with a jump when the player is about to go over a gap edge, or None.
    Travel side = the model's left/right input, else the direction the player is drifting. The jump window is
    1..8 px from the edge; pressing 'up' to wait is NOT neutral (the camera stops following and the player ends
    up left of its anchor, which shortens the jump), so the approach itself is left to the model / the belt."""
    p = radar.get("player")
    if not p or belt.airborne:
        return None
    pits = [g for g in (radar.get("pits") or []) if not (g["left"] <= p["x"] <= g["right"]) and not g.get("edge")]
    if not pits:
        return None
    if model_action in LEFTWARD:
        goal = "left"
    elif model_action in RIGHTWARD:
        goal = "right"
    elif belt.direction and radar.get("conveyor_zone"):
        goal = belt.direction                # no horizontal input: the belt decides where you are heading
    else:
        return None                          # v52: knockbacks in a bank fight read as "drift" and launched kick_left
                                             # jumps into the stones gaps behind (3 per fight in the replay); a belt
                                             # goal needs belt tiles under the player
    ahead = [g for g in pits if g["side"] == goal]
    if not ahead:
        return None
    nearest = min(ahead, key=lambda g: abs(g["dx_edge"]))
    d = abs(nearest["dx_edge"])
    # a climbable ladder standing before the gap is the intended route (mission 2's first screen: the pit chain
    # under the ladder is not jumpable, the level continues on the floor above). Leave the decision to the model.
    if any(l.get("climbable") and (abs(l["dx"]) <= 10 or (l["side"] == goal and abs(l["dx"]) <= d + 8))
           for l in radar.get("ladders", [])):
        return None
    width = nearest["right"] - nearest["left"]
    x = p["x"]
    behind_camera = (x < CAMERA_ANCHOR["left"] - ANCHOR_SLACK) if goal == "left" else (x > CAMERA_ANCHOR["right"] + ANCHOR_SLACK)
    jump = "kick_left" if goal == "left" else "kick_right"
    if d > APPROACH_ZONE:
        belt.backoffs = 0
    on_belt_toward = belt.direction == goal
    # Only jump when the player is genuinely committed to crossing: a belt is carrying it toward the gap, or the model
    # has been walking toward it for most of the last few decisions. A single stray press must not launch a jump
    # (v21: models wandering left on mission 2's first floor were jumped into the pit chain). Wide gaps are only
    # known to be jumpable on the conveyor walkway, so without a belt they are left to the DANGER advice.
    toward = LEFTWARD if goal == "left" else RIGHTWARD
    intent = recent is not None and sum(1 for a in recent[-3:] if a in toward) >= 2
    conveyor = bool(radar.get("conveyor_zone"))
    lo, hi = JUMP_WINDOW_WIDE if width >= 16 else (JUMP_WINDOW_NARROW_LEFT if goal == "left" else JUMP_WINDOW_NARROW)
    back = "right" if goal == "left" else "left"
    if not on_belt_toward and d < lo and (width < 16 or conveyor):
        # already too close to take off (v41: after landing on the 24 px island a single stray 'left' put the player
        # past the leftward take-off point and it fell): step back first, whatever the model intended
        belt.backoffs += 1
        return back
    if width >= 16 and not (on_belt_toward or conveyor):
        return None                          # wide gap off the conveyor walkway: not known to be jumpable
    if not on_belt_toward and not intent:
        return None
    if lo <= d <= hi or (belt.backoffs >= BACK_OFF_MAX and d <= hi):
        if behind_camera and on_belt_toward and belt.backoffs < BACK_OFF_MAX:
            belt.backoffs += 1
            return back                      # jump would land short: back off against the belt, come round again
        belt.backoffs = 0
        return jump
    if d < lo:
        belt.backoffs += 1                   # overshot the window: one step back re-enters it from above
        return back
    if d <= APPROACH_ZONE and behind_camera and on_belt_toward and model_action != goal:
        return goal                          # keep walking so the camera re-centres before the jump
    return None


def stones_autopilot(radar: Dict, model_action: str) -> Optional[str]:
    """Forest ledge with a stepping-stone gap ahead (right side, within 48 px) and no enemy in reach: the crossing is a
    frame-precise sequence (take-off point, air control, 24 px islands) and five successive model-side fixes were not
    enough. The reflex owns it: walk right; belt_reflex() then fires the jumps. Hands back when no gap is ahead or an
    enemy is in reach."""
    p = radar.get("player")
    if not p or not radar.get("river_level"):
        return None
    enemies = radar.get("enemies", [])
    if 93 <= p["feet_y"] <= 106 and any(g["side"] == "right" and abs(g["dx_edge"]) <= 56 for g in radar.get("pits", [])):
        # v52: the grass bank (feet ~97) runs into the river while the stone tops (tile 352) continue at feet 88-92.
        # Walking, lane-holding and every jump timing from the bank died (11/11); the way on is UP onto the stone lane
        # first. The reproduced v49/v51 "x=163" deaths were this spot.
        return "up"
    if not (76 <= p["feet_y"] <= 92):
        return None
    if any(e["in_reach"] and e["side"] == "right" for e in enemies):
        return None                          # something ahead in reach: the model fights it
    near = [g for g in radar.get("pits", []) if abs(g["dx_edge"]) <= 48 and not g.get("edge")]
    if any(e.get("heavy") for e in enemies):
        return None                          # the bruiser must be fought (v46: all deep runs died pinned at the right edge
                                             # with him behind them because this rule kept them walking right)
    if not near:
        return None
    # gaps near (either side: a left jump back is the other way to die). v56: the old "thug closing from behind ->
    # keep walking right" clause walked players to the screen edge with the thug 16-18 px behind them, where nothing
    # lands (every pinned death at x=149 in v53); turning to fight a level thug beats walking on (v51 emulator).
    return "right"


def air_veto(radar: Dict, belt: "BeltTracker", model_action: str) -> Optional[str]:
    """Mid-jump, any input other than the jump direction shortens the arc (v42: 'rrrrrr ll' -> landed in the gap).
    Hold the jump direction until landing."""
    if belt.airborne and belt.jump_dir and model_action != belt.jump_dir:
        return belt.jump_dir
    return None


def no_freelance_jump_veto(radar: Dict, model_action: str, travel: Optional[str]) -> Optional[str]:
    """On the forest ledge the reflex owns jumping: a model-initiated kick/jump at the wrong distance is a fall
    (v42: jumps launched 30-40 px before the gap). With no enemy in reach, turn it into a walk."""
    if radar.get("river_level") and model_action in ("jump_kick", "kick_left", "kick_right") and radar.get("pits") \
            and not any(e["in_reach"] for e in radar.get("enemies", [])):
        return travel or "right"
    return None


def rung_veto(radar: Dict, model_action: str) -> Optional[str]:
    """Attacks are ignored on ladder rungs; the game still runs, so the thugs waiting at the foot get free hits the
    moment the player finally steps off. Turn a wasted attack into the climb step the advice asked for."""
    p = radar.get("player")
    if p and p.get("climbing"):
        # v103: a thug LEVEL with us on the rungs (mission-2 start, ladder top off-screen): the model, told to climb
        # down, pressed down while this veto turned its attacks into "up" - 4000 up/down pairs per episode at full
        # HP. A level enemy means we are at his floor: climb toward it (up if he is not below us) whatever the
        # model pressed, until the climbing pose ends; a 1-px tie-break on ladder ends must not decide this.
        near0 = [e for e in radar.get("enemies", []) if abs(e["dy"]) <= 40 and abs(e["dx"]) <= 80]
        if near0 and model_action in ("up", "down"):
            below0 = [e for e in near0 if e["dy"] > 4]
            above0 = [e for e in near0 if e["dy"] < -4]
            if above0 and not below0 and model_action == "down":
                return "up"                  # the only thug in reach is above: climbing down just hands him the top
            if below0 and not above0 and model_action == "up":
                return "down"
            if not above0 and not below0:    # level with us: we are at his floor, get off toward him
                return "up" if min(near0, key=lambda e: abs(e["dx"]))["dy"] <= 0 else "down"
    if p and p.get("climbing") and (model_action in ATTACKS or model_action in ("left", "right")):
        lad0 = next((l for l in radar.get("ladders", []) if l["on_it"]), None)
        level = [e for e in radar.get("enemies", []) if abs(e["dy"]) <= 6 and abs(e["dx"]) <= 80]
        if lad0 and level:
            return "up" if min(level, key=lambda e: abs(e["dx"]))["dy"] <= 0 else "down"
        if lad0 and model_action in ATTACKS:
            # v103: an attack on the rungs is wasted whatever we do; head for the thugs at ANY height (the mission-2
            # start has them 64 px below: the old 40-px band ignored them and the tie-break sent the attack "up")
            anyn = [e for e in radar.get("enemies", []) if abs(e["dx"]) <= 100]
            if anyn and all(e["dy"] > 4 for e in anyn):
                return "down"
            if anyn and all(e["dy"] < -4 for e in anyn):
                return "up"
    if p and p.get("climbing") and (model_action in ATTACKS or model_action in ("left", "right")):
        # the climbing pose persists one step after reaching the foot (feet = bottom+1): left/right/attacks are still
        # ignored there and one 'down' lets go (measured, v31: 200 wasted attacks next to a thug)
        lad = next((l for l in radar.get("ladders", []) if l["on_it"]), None) or \
              next((l for l in sorted(radar.get("ladders", []), key=lambda l: abs(l["dx"])) if abs(l["dx"]) <= 16), None)
        if lad is None:
            return None
        if p["feet_y"] >= lad["bottom_y"] - 4:
            # v32: in the locked room the pit is 15px right of the ladder foot and a 'down' on the floor walks off the
            # ledge (18/18 mission-2 starters died there). Only let go with 'down' when no drop is close.
            if any(abs(g["dx_edge"]) <= 40 for g in radar.get("pits", [])):
                return None
            return "down"
        # only enemies on a floor this ladder connects to matter (v33: a thug 80px below on the lower walkway turned
        # every 'right' at the top of room 3's ladder into 'down', and the player hovered on the rungs for 200 steps)
        near = [e for e in radar.get("enemies", []) if abs(e["dy"]) <= 40]
        below = [e for e in near if e["dy"] > 4]
        above = [e for e in near if e["dy"] < -4]
        if below and not above:
            return "down"
        if above and not below:
            return "up"
        return "down" if (lad["bottom_y"] + 1 - p["feet_y"]) <= (p["feet_y"] - lad["top_y"]) else "up"
    return None


HEAVY_LANE_TOL = 2          # px of vertical offset within which hits on the heavy enemies register
HEAVY_RANGE = (12, 24)      # horizontal window where punches land on them; under 12 is the grab/hurt zone


def heavy_autopilot(radar: Dict, model_action: str, area_id: int, last_heavy: Optional[Dict] = None) -> Optional[str]:
    """v58: the heavy fights (mission-1 boss, mission-3 bruiser) are won or lost on reaction time the model does not
    have. Harness, acting every step: align within 2 px, close to 14-24 px, punch, step out under 14 -> bruiser 4/4
    (19 hits, 1.0 health lost), boss 3/4; the same policy with each action held two steps: 1/4 and 0/4. v57 gave the
    model a decision every step within 40 px and the first bruiser fight was still lost (grabbed at 28-40 px on the
    way in). The reflex runs every env step, so it owns the fight from 64 px: the model keeps everything else."""
    p = radar.get("player")
    if not p or radar.get("on_ladder"):
        return None                          # (no airborne guard: a lane change trips it and handed the fight back mid-alignment)
    if area_id == 48 and not AUTO_BOSS:
        return None                          # v60: with the autopilot the mission-1 boss went 0/16 in v59 (21-45 hits over
                                             # three lives, never 20 in one); the model alone wins ~half. His room is the model's.
    heavy = [e for e in radar.get("enemies", []) if (e.get("heavy") or area_id == 48) and abs(e["dx"]) <= 160 and abs(e["dy"]) <= 40]
    if not heavy and last_heavy and last_heavy.get("steps_ago", 99) <= 16:
        # the heavy flag is a sprite height and drops during his crouch/punch frames; he is still the same enemy
        heavy = [e for e in radar.get("enemies", []) if abs(e["dx"]) <= 160 and abs(e["dy"]) <= 40]
    if not heavy:
        # knocked down he leaves the radar for a few steps; the model's wandering then (random stand-in: 1/3 of all
        # fight steps) handed him the lunge. Keep walking toward where he was, as the scripted policy did.
        if last_heavy and last_heavy.get("steps_ago", 99) <= 16 and not radar.get("enemies"):
            return last_heavy.get("side", "right")
        return None
    others = [e for e in radar.get("enemies", []) if e["in_reach"] and not any(e is h for h in heavy)]
    if others:
        return None                          # an ordinary thug in reach first: the model's fight
    combo_n = AUTO_COMBO_BOSS if area_id == 48 else AUTO_COMBO
    if last_heavy is not None and combo_n:
        # a knockdown comes from consecutive hits (3rd hit of an unbroken chain); a punch that just landed is followed
        # through for combo_n more steps whatever his distance / lane reads (v65 boss: 42 hits but 3 knockdowns in
        # 700 steps because every hit was followed by an alignment press on his jittering dy)
        if last_heavy.pop("just_hit", False):
            last_heavy["combo"] = combo_n
        e0 = min(heavy, key=lambda ee: abs(ee["dx"]))
        if last_heavy.get("combo", 0) > 0 and abs(e0["dy"]) <= 8 and abs(e0["dx"]) <= 28:
            last_heavy["combo"] -= 1
            return "a" if p.get("facing") == e0["side"] else e0["side"]
    e = min(heavy, key=lambda ee: abs(ee["dx"]))
    toward = e["side"]; away = "left" if toward == "right" else "right"
    if AUTO_BODY_DISTANCE and "back_x" in e:
        d = max(0, abs(e["back_x"] - p["x"]) - AUTO_BODY_HALF)   # distance to his body, immune to his punch animation
    else:
        d = abs(e["dx"])
    if AUTO_DODGE != "none" and area_id != 48 and e.get("w", 24) >= 36 and d <= 30 and abs(e["dy"]) <= 12:
        # env trace: his sprite is 24 px wide standing and 40 px wide for the one step his punch is out; every hit on
        # us landed in such a step, and his reach is ~28 px. Dodge that step.
        if AUTO_DODGE == "up":
            return "up" if (p["feet_y"] >= 82 or radar.get("on_ledge")) else "down"
        return away if not (away == "left" and p["x"] <= 8 or away == "right" and p["x"] >= 152) else "a"
    if area_id != 48 and AUTO_APPROACH == "counter" and last_heavy is not None and abs(e["dy"]) <= HEAVY_LANE_TOL:
        # bruiser (env trace): his sprite centre jumps 8-10 px toward us every third step while we are within ~26 px:
        # that is his punch, and it out-ranges ours (hits landed on us at 22-26 while we stepped). Hold just outside
        # his reach and counter into his recovery: the step his arm comes out, close in; the next step, punch.
        prev = last_heavy.get("prev_dx"); last_heavy["prev_dx"] = d
        phase = last_heavy.get("phase", 0)
        if phase == 1:
            last_heavy["phase"] = 0
            return "a" if not (p.get("facing") != toward) else toward
        if prev is not None and prev - d >= 6 and d <= 30:
            last_heavy["phase"] = 1
            return toward
        if d > 30:
            return toward
        if d >= 21:
            return "a" if p.get("facing") == toward else toward     # hold just outside his reach, facing him
        # inside 20: the close rules below
    if area_id != 48 and d > AUTO_OFFLANE_DROP and AUTO_APPROACH == "offlane18":
        # bruiser approach: he lunges at 28-32 px on his lane. Close in well above his lane, drop onto it late.
        if abs(e["dy"]) < AUTO_OFFLANE_DY:
            return "up" if (p["feet_y"] >= 82 or radar.get("on_ledge")) else "down"
        return toward
    if AUTO_FLIGHT == "chase" and area_id != 48 and e["dy"] < -8 and not e.get("heavy") and d > 19:
        return toward                        # he is flying from our knockdown toward the screen edge: close the gap now
    if abs(e["dy"]) > HEAVY_LANE_TOL:
        want = "down" if e["dy"] > 0 else "up"
        if want == "down" and radar.get("on_ledge") and p["feet_y"] + e["dy"] > 92:
            return None
        return want
    step_out = AUTO_STEP_OUT["boss" if area_id == 48 else "bruiser"]
    hi = AUTO_PUNCH_HI["boss" if area_id == 48 else "bruiser"]
    turned_away = p.get("facing") != toward
    # hysteresis (v59, real run): stepping out flips the facing, and the turn back costs ~4 px, so from 15 px the
    # pair "step out / turn" cycled for 30 steps without a punch. Facing away, keep stepping out until the turn will
    # land inside the window.
    # (tried: backing off or holding 21+ px during his short-sprite get-up frames, where the grab at 19 px comes from.
    #  Both were worse on the 8-offset env benchmark: 1/8 and 2/8 against 5/8 without.)
    if step_out and (d < step_out or (turned_away and d < step_out + 5)):
        if away == "left" and p["x"] <= 8 or away == "right" and p["x"] >= 152:
            return "a" if not turned_away else toward                # nowhere to step: fight from here
        if AUTO_ELBOW and turned_away and 7 <= d <= 20:
            return "a"                       # v78 probe: with our back to him A is the ELBOW (+200, 3 dmg, lands 7-20 px)
        return away
    if d > hi:
        if area_id != 48 and AUTO_APPROACH == "edgetimed":
            # real arrivals: he spawns pinned at the right edge (x ~154) and only lunges 8-10 px forward; cross his
            # 26-40 px lunge zone while he is back at the edge, back off the step his punch frame (w >= 36) shows
            if e.get("w", 24) >= 36 and d <= 32:
                return away if not (away == "left" and p["x"] <= 8) else "a"
            if e["x"] >= 148 or d > 34 or turned_away:
                return toward
            return "a"                       # he is forward of the edge and within 34: hold facing him
        if area_id != 48 and AUTO_APPROACH == "kick" and 24 <= d <= 34 and not turned_away:
            return "kick_" + toward          # jump kick into him from just outside his lunge range
        if area_id != 48 and AUTO_APPROACH == "alternate" and not turned_away and d % 3 == 0:
            return "a"                       # walk in with punches interleaved: his lunge lands on a fist
        return toward                        # punches at 21-24 px landed 0/18 in the lane probe: close in instead
    if turned_away:
        return toward
    return "a"


AUTO_STEP_OUT = {"bruiser": 16, "boss": 0}    # bruiser: his grab lands at 13-15 px; boss: 14 cleared 4/5 vs 1/5 at 16
AUTO_PUNCH_HI = {"bruiser": 20, "boss": 24}
AUTO_APPROACH = "walk"      # bruiser approach from beyond the window: walk | offlane18 | alternate | counter
AUTO_BODY_DISTANCE = False  # measure to his back edge minus half a body: his punch animation moves the sprite centre, not his body
AUTO_BODY_HALF = 12
AUTO_DODGE = "none"
AUTO_FLIGHT = os.environ.get("DD_AUTO_FLIGHT", "align")
AUTO_COMBO = 0              # follow-through punches after a landed hit (0 = off)
AUTO_COMBO_BOSS = int(os.environ.get("DD_AUTO_COMBO_BOSS", "0"))         # same for the mission-1 boss room
AUTO_ELBOW = os.environ.get("DD_AUTO_ELBOW", "0") == "1"      # elbow instead of a second step-out when already facing away
AUTO_MIN_SEEN = int(os.environ.get("DD_AUTO_MIN_SEEN", "6"))   # consecutive heavy sightings before a tall sprite counts as the bruiser
KNOWN_HEAVY_AREAS = {46}                                        # the mission-3 bruiser screen: trust the flag at once there


class HeavyGate:
    """v77: the radar flags any sprite group taller than 34 px as "heavy". Two thugs stacked vertically (one jumping
    over the other) merge into one such group for 1-9 steps: 168 false heavies in 3910 forest/bridge frames, always with
    enemy_count == 1. The real bruiser is flagged for hundreds of consecutive steps and only ever in area 46. Outside the
    known areas the flag is stripped until it has held for AUTO_MIN_SEEN consecutive steps, so the autopilot, the lane
    veto and the prompt text all stop treating a thug pair as the bruiser. Apply once per env step, before anything
    reads the radar."""

    def __init__(self):
        self.streak = 0
        self.gap = 0

    def apply(self, radar, area_id):
        hv = [e for e in radar.get("enemies", []) if e.get("heavy")]
        if hv:
            self.streak += 1
            self.gap = 0
        else:
            self.gap += 1
            if self.gap > 16:                # the knockdown gap is <= 16 steps; longer means he is gone
                self.streak = 0
        if hv and area_id not in KNOWN_HEAVY_AREAS and self.streak < AUTO_MIN_SEEN:
            for e in hv:
                e["heavy"] = False
        return radar
AUTO_BOSS = os.environ.get("DD_AUTO_BOSS", "0") == "1"            # let the autopilot fight the mission-1 boss (area 48)
AUTO_OFFLANE_DY = 12
AUTO_OFFLANE_DROP = 16       # while he flies from a knockdown: align (chase his dy) | chase (close the gap)         # react to his punch frame (sprite width >= 36): none | away | up


def boss_combo_veto(radar: Dict, model_action: str, area_id: int, last_heavy: Optional[Dict]) -> Optional[str]:
    """Mission-1 boss (v62). Traces: a knockdown comes exactly 4 steps after the previous hit with two hits in the 8
    steps before it, i.e. it is the third hit of an unbroken punch chain, and he only falls to knockdowns. v47 (10/12)
    kept punching in streaks of 3+ 77% of the time; v56-v61 58%, with knockdowns halved and 0/36 wins. Once a punch
    lands, the next BOSS_COMBO steps stay on 'a' facing him whatever the model wants (except when he is off the lane)."""
    p = radar.get("player")
    if area_id != 48 or not p or last_heavy is None or radar.get("on_ladder"):
        return None
    if last_heavy.pop("just_hit", False):
        last_heavy["combo"] = BOSS_COMBO
    if last_heavy.get("combo", 0) <= 0:
        return None
    es = [e for e in radar.get("enemies", []) if e["reachable"]]
    if not es:
        return None
    e = min(es, key=lambda ee: abs(ee["dx"]))
    if abs(e["dy"]) > 4 or abs(e["dx"]) > 26:
        last_heavy["combo"] = 0
        return None
    last_heavy["combo"] -= 1
    return "a" if p.get("facing") == e["side"] else e["side"]


BOSS_COMBO = 3


RIVER_EXIT_AREAS = {52, 53}
RIVER_EXIT_FEET = 90


def river_exit_veto(radar: Dict, model_action: str, area_id: int) -> Optional[str]:
    """v81: the last river screens (areas 52/53) end in water at the area-46 boundary for anyone on the bank:
    v78 mission-3 starters entering 46 with feet >= 97 fell 21/28 (twice each, once per life); feet <= 92 never
    fell (v76: 26/27 entered at 83-91, 1 fall). v76 only had the lane because thug pairs mis-read as "heavy" made
    the autopilot align upward - gone with the v77 HeavyGate. Deterministic replacement: on these screens, when
    no enemy is within punching reach, walk the upper lane (press up until feet <= RIVER_EXIT_FEET)."""
    p = radar.get("player")
    if not p or area_id not in RIVER_EXIT_AREAS or p.get("climbing") or radar.get("on_ladder"):
        return None
    if p["feet_y"] <= RIVER_EXIT_FEET:
        return None
    if any(e["reachable"] and abs(e["dx"]) <= 24 and abs(e["dy"]) <= 6 for e in radar.get("enemies", [])):
        return None                          # a thug is on us: fight first, the lane can wait
    return "up"


CAVE_HOP_WINDOW = (6, 12)      # px before a one-tile hole where right+A+B (the running jump kick) lands on the far side
CAVE_HOP_MAX_FEET = 80         # the mission-4 upper walkway (feet 75-76); the lower cave floor (108) is a dead end anyway


def cave_hop_veto(radar: Dict, model_action: str, river: bool) -> Optional[str]:
    """v87: mission-4 cave. The upper walkway (reached by climbing the chain) is cut by one-tile holes with
    non-solid ladder columns beyond them: ~28 px of nothing. A plain jump (A+B) carries 17 px and drops into the
    ladder column; the running jump kick (right+A+B) carries 35 px and lands standing when started 6-12 px before
    the edge (measured: triggers 6-12 land at x+84..+88, 13+ fall, <6 fall). Closer than the window: back off."""
    p = radar.get("player")
    if river or not p or p.get("climbing") or radar.get("on_ladder") or p["feet_y"] > CAVE_HOP_MAX_FEET:
        return None
    if p["feet_y"] < CAVE_HOP_MAX_FEET - 6:
        return "right"                       # airborne (walkway feet are 75-76): keep the momentum, never back off mid-hop
    holes = [g for g in radar.get("pits", []) if g["side"] == "right" and g.get("right", 0) - g.get("left", 0) <= 8 and g["dx_edge"] >= -4]
    if not holes:
        return None
    d = min(g["dx_edge"] for g in holes)
    lo, hi = CAVE_HOP_WINDOW
    if lo <= d <= hi:
        return "kick_right"
    if d < lo:
        return "left"
    if d <= 40:
        return "right"                       # v123: own the whole approach - a later reflex (gap jump) fired a plain A+B at 20 px and dropped into the column
    return None


def ladder_align_veto(radar: Dict, model_action: str) -> Optional[str]:
    """v107: 'up' at a ladder foot only climbs within ~5 px of the ladder's axis. The model pressed up 3000 times
    from 8 px off (mission-2 area 16). Nudge toward the axis instead."""
    p = radar.get("player")
    if not p or p.get("climbing") or model_action != "up":
        return None
    lad = next((l for l in radar.get("ladders", []) if l.get("under_it") and not l.get("on_it") and not l.get("at_top")), None)
    if lad and 5 < abs(lad["dx"]) <= 12 and p["feet_y"] >= lad["bottom_y"] - 6:
        return lad["side"]
    return None


def heavy_lane_veto(radar: Dict, model_action: str, area_id: int) -> Optional[str]:
    """v49: hits on the heavy enemies (mission-1 boss, mission-3 bruiser) depend on LANE alignment, not range. Emulator,
    bruiser: 6 px off his lane 0 of 286 attacks landed at any distance; within 2 px, punches land 18-41% at 16-20 px.
    Mission-1 boss traces (v46-v48, 14k steps): |dy|<=2 -> a 9-20%, b 20-33%; |dy| 3-5 -> 2-6%; |dy|>=6 -> 0-1%.
    A scripted policy with just this rule (align within 2 px, punch at 12-24 px, step away under 12) clears the
    mission-1 boss 3/3 and the bruiser 2/3 where the earlier 10 px-tolerance grid cleared neither. Enforce alignment
    and the window; the choice of attack and of when to dodge stays with the model."""
    p = radar.get("player")
    if not p or radar.get("on_ladder") or radar.get("airborne"):
        return None
    if area_id == 48:
        # v54: with the veto the mission-1 boss went 1/13 (v49, v53) against 16/31 without (v46-v48): hits/100 steps
        # were unchanged (4.8 -> 5.3) but knockdowns halved (1.6 -> 0.8 per 100). The fight stays with the model there.
        return None
    heavy = [e for e in radar.get("enemies", []) if e["reachable"] and e.get("heavy")]
    if not heavy:
        # v50: the same mechanic for ordinary thugs (v47+v49 traces, 80k mission-3 steps near an enemy): level within
        # 2 px punches land 16-24% and kicks 29-38%; 3-10 px off-lane 1-2%. Mission 3 spent 42k of those steps off-lane
        # and threw 21k punches there. Attacks off-lane become the alignment press; nothing else is touched.
        near = [e for e in radar.get("enemies", []) if e["reachable"] and abs(e["dx"]) <= 28]
        if not near:
            return None
        if radar.get("river_level") and any(abs(g["dx_edge"]) <= 40 for g in radar.get("pits", [])):
            return None                       # v51: turning / lining up next to the stepping stones walked players off them
        e = min(near, key=lambda ee: abs(ee["dx"]))
        if abs(e["dy"]) <= HEAVY_LANE_TOL:
            # v51: a punch thrown facing away from a level thug cannot land (v49 traces: 6% on thugs behind vs 24% ahead,
            # the difference being the facing); turn first. Scripted "turn and fight" beat "walk on" in every forest state.
            if model_action in ATTACKS and p.get("facing") and p["facing"] != e["side"] and abs(e["dx"]) >= 10:
                return e["side"]
            return None
        want = "down" if e["dy"] > 0 else "up"
        if want == "down" and radar.get("on_ledge") and p["feet_y"] + e["dy"] > 92:
            return None
        if model_action == want or model_action in ATTACKS:
            return want
        return None
    e = min(heavy, key=lambda ee: abs(ee["dx"]))
    d = abs(e["dx"]); toward = e["side"]; away = "left" if toward == "right" else "right"
    if abs(e["dy"]) > HEAVY_LANE_TOL:
        want = "down" if e["dy"] > 0 else "up"
        if want == "down" and radar.get("on_ledge") and p["feet_y"] + e["dy"] > 92:
            return None                       # lining up would mean stepping off the ledge: leave it to the model
        if model_action == want:
            return want                       # pass-through so the ledge veto (down -> a) cannot eat the alignment press
        if model_action in ATTACKS or model_action == toward or model_action == ("up" if want == "down" else "down"):
            return want                       # attacks and walking into him off-lane are wasted (and hurt): line up first
        return None
    if d < HEAVY_RANGE[0] and (model_action in ATTACKS or model_action == toward):
        return away                           # his grab range: step out (hurt 5-8%/step here vs 2-4% at 16-24)
    if d > HEAVY_RANGE[1] and model_action in ATTACKS:
        return toward                         # punching air from 25+ px: 0-4% hit, 5-11% hurt; close in instead
    if HEAVY_RANGE[0] <= d <= HEAVY_RANGE[1] and model_action == toward:
        return "a" if p.get("facing") == toward else toward   # walking into him from the window is the classic stall
    if d >= 10 and model_action in ATTACKS and p.get("facing") and p["facing"] != toward:
        return toward                         # v56: every v53 bruiser death at x=149 was 'a' spam with him 16-18 px BEHIND
    return None


def pit_back_veto(radar: Dict, model_action: str) -> Optional[str]:
    """Fighting with a drop at your back: every recorded death at the foot of the locked room's ladder (v21-v24) came
    from walking INTO the thugs and being thrown backwards into the pit, while punching from the spot beats them with
    no damage (measured). With a pit edge within 24 px behind and a reachable enemy within 40 px in front, a step
    toward the enemy becomes a punch."""
    p = radar.get("player")
    if not p or radar.get("conveyor_zone") or radar.get("on_ladder"):
        return None
    if model_action not in ("left", "right"):
        return None
    back = "right" if model_action == "left" else "left"
    pit_behind = any(g["side"] == back and abs(g["dx_edge"]) <= 32 and (g.get("edge") or g["right"] - g["left"] >= 16)
                     for g in radar.get("pits", []))
    ahead = [e for e in radar.get("enemies", []) if e["reachable"] and e["side"] == model_action and abs(e["dx"]) <= 40 and abs(e["dy"]) <= 24]
    if pit_behind and ahead:
        e = min(ahead, key=lambda e: abs(e["dx"]))
        if abs(e["dy"]) > 10:                # v37: thugs one lane lower at the room-3 walkway end; line up, do not close in
            return "down" if e["dy"] > 0 else "up"
        return "a"
    return None


HOLE_MARGIN = 36            # a grab-and-throw carries the player up to ~30 px (v56 traces: thrown from x 118-141 into the hole at 144)


def walk_in_veto(radar: Dict, model_action: str) -> Optional[str]:
    """v73: walking INTO a level thug is where the damage is. Bridge (v71, 9 episodes): adjacent to a thug ahead
    (<= 12 px) costs 9.2% health per step, 13-28 px 1.8%; forest and boss room show the same shape. A step toward an
    ordinary enemy that is already level and within 16 px becomes the punch (facing it) or the turn toward it."""
    p = radar.get("player")
    if not p or radar.get("on_ladder") or radar.get("airborne"):
        return None
    if model_action not in ("left", "right"):
        return None
    near = [e for e in radar.get("enemies", []) if e["reachable"] and not e.get("heavy") and e["side"] == model_action
            and abs(e["dx"]) <= WALK_IN_PX and abs(e["dy"]) <= 4]
    if not near:
        return None
    e = min(near, key=lambda ee: abs(ee["dx"]))
    return "a" if p.get("facing") == e["side"] else e["side"]


WALK_IN_PX = 16


def hole_margin_veto(radar: Dict, model_action: str) -> Optional[str]:
    """Mission-2 walkways: a thug's grab THROWS the player ~10-13 px away from the thug, no health lost; standing within
    20 px of the ladder hole with the thug on the other side puts the throw in the hole (v49: 15 falls at area 66,
    scx 176, the biggest mission-2 killer; v51 7; every trace shows x 137-140 -> 149-150 -> fall). Measured on the
    saved walkway state: fighting from the spot 6/6 falls, walking toward the thug 0/6. With a hole that close behind
    and an enemy ahead, every attack or step toward the hole becomes a step toward the enemy until the margin is back."""
    p = radar.get("player")
    if not p or radar.get("on_ladder") or radar.get("airborne"):
        return None
    if not (radar.get("multi_floor") or radar.get("river_level")):
        return None
    close = [g for g in radar.get("pits", []) if abs(g["dx_edge"]) <= HOLE_MARGIN - 4]
    if len({g["side"] for g in close}) > 1:
        return None                          # gaps on both sides: a stepping stone, the autopilot's business
    for g in close:
        hole_side = g["side"]; away = "left" if hole_side == "right" else "right"
        enemies = [e for e in radar.get("enemies", []) if e["side"] == away and abs(e["dx"]) <= 64 and abs(e["dy"]) <= 12]
        # (a thug 40-64 px out arrives within the few steps a ladder attempt takes: replay, 'up' at the ladder foot = 6/6 thrown)
        if not enemies:
            continue
        e = min(enemies, key=lambda ee: abs(ee["dx"]))
        # v58 replay (m2x-end-v17): walking INTO the thug to gain margin is what gets you grabbed (3/6 thrown into the
        # hole); punching it from 14-24 px with the hole 30+ px behind never did (0/6). So: close in only beyond 24 px,
        # otherwise fight it facing it, and never step toward the hole or try the ladder while it is on you.
        if abs(e["dx"]) > 24:
            return away if model_action != away else None
        if model_action in (hole_side, "up", "down") or (model_action in ATTACKS and p.get("facing") != e["side"]):
            return "a" if p.get("facing") == e["side"] else e["side"]
    return None


def ledge_veto(radar: Dict, model_action: str) -> Optional[str]:
    """On mission 3's river ledge 'down' walks off the path (measured). Turn it into a harmless input."""
    if model_action == "down" and radar.get("on_ledge") and radar.get("player"):
        return "a"                       # any 'down' on the ledge is a step toward the water
    return None


def belt_note(belt_dir: Optional[str], speed: float, radar: Dict) -> str:
    if not belt_dir:
        return ""
    pits = radar.get("pits") or []
    gap = (f" A GAP in the walkway is {min(abs(g['dx_edge']) for g in pits)}px away; gaps are jumped for you automatically "
           f"when you are within a few px of the edge, so do NOT press kick/jump yourself: keep pressing your travel direction.") if pits else ""
    return (f"CONVEYOR: this walkway is a moving belt carrying you {belt_dir} about {abs(speed):.0f}px per step even with no "
            f"input; pressing {'right' if belt_dir == 'left' else 'left'} walks against it. Keep choosing the direction you "
            f"want to travel (left/right) and it will be honoured.{gap}")
