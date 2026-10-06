# run-decision-dd.py
"""
Decision-model mode: no PPO, no learner, no gradient updates.

Runs N Double Dragon emulators in parallel (one Ray actor per env, 1 CPU each) and, at
every step, sends the *same observation the RL agent received* to a DiffusionGemma
decision-model replica (round-robined across the GPU pods) and executes whatever button
the model picks. DDEnv still computes the old shaped reward, which we log purely as a
comparison metric against the PPO runs.

Recording (all under s3://$S3_BUCKET_NAME/):
  ddenv_videos/decision/<run_tag>/...           full-frame MP4 per episode (DDEnv built-in)
  decision_runs/<run_tag>/traces/w<idx>_ep<n>.npz  obs + teacher action/probabilities per step
                                                  (the behaviour-cloning dataset for PPO pretraining)
  decision_runs/<run_tag>/episodes.jsonl        one line per finished episode
  decision_runs/<run_tag>/summary.json          run-level metrics

Usage (inside the Ray cluster, e.g. via the notebook's job client):
    python run-decision-dd.py --num-envs 100 --episodes-per-env 1
Quick single-process check without Ray:
    python run-decision-dd.py --local --num-envs 1 --max-steps 200
Print one request body (no server needed):
    python run-decision-dd.py --dry-run
"""
from __future__ import annotations

import argparse
import io
import glob
import json
import logging
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from decision_client import DecisionModelClient
from reflex import BeltTracker, belt_reflex, belt_note, belt_tile_under, ledge_veto, rung_veto, pit_back_veto, air_veto, no_freelance_jump_veto, stones_autopilot, heavy_lane_veto, hole_margin_veto, heavy_autopilot, boss_combo_veto, walk_in_veto, HeavyGate, river_exit_veto, cave_hop_veto, ladder_align_veto
from radar import sprite_radar, suggested_rule, ScrollTracker, read_health
import radar as radar_mod
from fight_planner import FightPlanner, ENEMY_HP_ADDRS as planner_slots
from nav_planner import NavPlanner
from hop_planner import HopPlanner   # v123: verified pillar hops on the mission-4 upper walkway
from progress import ProgressTracker, CANON, M2_START, M3_START, M4_START   # v109: canonical chunk sequence = exact progress oracle

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("run-decision-dd")

DEFAULT_INIT_STATES = [
    "boss.dd.gb.state", "lvl2-dd.gb.state", "lvl3.5-dd.gb.state",
    "dd.gb.state", "lvl-1.5.dd.gb.state", "lvl3-dd.gb.state",
]

# Level ids read from RAM 0xE110. Names come from loading each save state in ignored/ and reading
# the address (see README "Level ids"). DDEnv's reward map only knows 15/84/48/89/11, so mission 3
# (ids 12 and 5) is invisible to the shaped reward; we track distinct ids visited instead and leave
# "game cleared" detection to the ending-screen RAM signature once a winning run is recorded.
SAVE_AREAS = {int(x) for x in os.environ.get("DD_SAVE_AREAS", "").split(",") if x.strip()}
# ---------------------------------------------------------------------------------------------------------------------
# v143: maze autopilot (past hideout door 2). The maze is deterministic terrain mapped offline on 2026-10-03 (memory note
# dd-maze-route): ledge row -> drop-kick to a lower platform -> column -> top ledge (its own respawn point) -> leftward
# step-down ledges (kick_left drops) -> column -> SCY-120 ledge -> island column -> upper floor (SCY 72, fights at its left
# end open the way) -> column -> a new area at SCY 0 that continues RIGHT. The live runner needs: a travel direction per
# scroll height, "hold UP under a climbable column", direction-aware hops with lower landings, a look-ahead probe so the
# player never walks off an edge, and no fights on ledges too narrow to survive a knockback.
MAZE_CLIMB_TILES = {254, 255, 267, 268, 270, 271}      # climbable column tiles (dotted 316/317/319 are decoration); v143e: 270/271 = the column at the upper floor's far left (new tileset)


def maze_dir(scy: int, feet: int) -> str:
    if scy == 200 and feet >= 100:
        return "right"                       # the ledge above the lower platform: its column is ~20 px to the right
    if scy in (200, 72):
        return "left"                        # top ledge / step-down ledges / upper floor
    return "right"                           # shaft (56), ledge row (0), lower platform (0), SCY-120 ledge + island, the area past the upper floor (0)


def maze_column_above(pyboy, player):
    """Horizontal offset (px, multiples of 8) of the nearest climbable column whose tiles stand above the player's floor level,
    scanning 16 px behind to 56 px ahead on both sides; 0 = overhead (hold UP climbs). None = no column in range. (v143: UP 6 px
    beside the shaft column did nothing for 40 steps; the drop onto the lower platform lands 24-40 px left of its column.)"""
    try:
        tm = pyboy.botsupport_manager().tilemap_background()
        scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
        best = None
        for ddx in range(-56, 57, 8):
            tx = ((scx + player["x"] + ddx) // 8) % 32
            # v143d: the column's FOOT must be at this floor (a column above the next ledge up is not reachable - run 16 walked off the
            # feet-90 ledge toward one); v144: and it must be a vertical stack of 3 tiles (chunk 97 has a horizontal strip of tile 254
            # as decoration at head height - the autopilot held UP under it forever)
            if all(tm.tile_identifier(tx, ((scy + player["feet_y"] - dy) // 8) % 32) in MAZE_CLIMB_TILES for dy in (8, 16, 24)):
                if best is None or abs(ddx) < abs(best):
                    best = ddx
        return best
    except Exception:
        return None


def maze_probe_deadly(env, action_idx: int, feet0: int, steps: int = 12) -> bool:
    """Would `steps` presses of this action drop the player (feet +12) or hide the sprite? Restores the emulator."""
    pb = env.pyboy
    buf = io.BytesIO(); pb.save_state(buf); L0 = pb.get_memory_value(0xC499); dead = False
    try:
        for _ in range(steps):
            env._run_action_on_emulator(action_idx)
            p = sprite_radar(pb).get("player")
            if pb.get_memory_value(0xC499) < L0 or p is None or p["feet_y"] > feet0 + 12:
                dead = True; break
    finally:
        buf.seek(0); pb.load_state(buf); env.kick_penalty = False
    return dead


def maze_grounded(env, feet0: int, a_idx: int = 4) -> bool:
    """One idle step on a snapshot: the feet stay put only when the player is standing (not thrown, jumping or falling)."""
    pb = env.pyboy
    buf = io.BytesIO(); pb.save_state(buf)
    try:
        env._run_action_on_emulator(a_idx)                        # 'a' (punch) - the idle that neither walks nor drops
        p = sprite_radar(pb).get("player")
        return p is not None and abs(p["feet_y"] - feet0) <= 1
    finally:
        buf.seek(0); pb.load_state(buf); env.kick_penalty = False


def maze_knockdown_plan(env, action_names, feet0: int, depth: int = 20, beam: int = 8, time_s: float = 20.0, max_loss: int = 0, evade: bool = False):
    """v143d: beam search for an enemy with no HP byte (the top-ledge thug past door 2 leaves for good after ONE landed
    hit; the planner's slot discovery finds nothing and then blocks itself for 60 steps while he throws the player off).
    Win = score gain and no reachable enemy left within 48 px; prune = life lost, HP lost, or a fall. Restores the emulator."""
    pb = env.pyboy
    def snap():
        b = io.BytesIO(); pb.save_state(b); return b.getvalue()
    def score():
        return sum(pb.get_memory_value(a) * 10 ** (5 - i) for i, a in enumerate(range(0xC640, 0xC646)))
    root = snap(); t0 = time.time(); L0 = pb.get_memory_value(0xC499); H0 = read_health(pb); S0 = score()
    idx = {n: action_names.index(n) for n in ("left", "right", "a", "b", "kick_left", "kick_right", "jump_kick", "down")}
    nodes = [(0.0, [], root)]; best = None; best_partial = None
    try:
        for d in range(depth):
            cand = []
            for sc0, seq, st in nodes:
                for name, ai in idx.items():
                    pb.load_state(io.BytesIO(st)); env._run_action_on_emulator(ai)
                    p = sprite_radar(pb).get("player")
                    if pb.get_memory_value(0xC499) < L0 or read_health(pb) < H0 - max_loss or p is None or p["feet_y"] > feet0 + 12:
                        continue
                    r = sprite_radar(pb, cave=True); near = [e for e in r["enemies"] if e["reachable"] and abs(e["dx"]) <= 48 and abs(e["dy"]) <= 24]
                    gain = score() - S0
                    sc = gain / 100.0 - (min(abs(e["dx"]) for e in near) if near else 0) * 0.05 - d * 0.1
                    if maze_in_skull_zone(pb, p):
                        continue                                                 # v150: never through a skull-door kill zone (the shaft test flickers with the skull animation)
                    grounded = abs(p["feet_y"] - feet0) <= 4                  # v148: a win/partial must end standing (a throw's arc landed three of run 29's lives in holes after the plan ended)
                    if evade and grounded and len(seq) + 1 >= 4 and not any(e["reachable"] and abs(e["dx"]) <= 32 and abs(e["dy"]) <= 12 for e in r["enemies"]):
                        if best is None or len(seq) + 1 < len(best):
                            best = seq + [ai]                                  # v150 evade mode: just get out of grab range unhurt (every skull-room death was a grab in the gap between plans)
                        continue
                    if gain > 0 and not near and grounded:
                        if best is None or len(seq) + 1 < len(best):
                            best = seq + [ai]
                        continue
                    if gain > 0 and grounded and len(seq) + 1 >= 6 and (best_partial is None or gain > best_partial[0]):
                        best_partial = (gain, seq + [ai])                   # v147: a landed hit with no HP lost - replay its first steps when no clean knockdown exists (lock-room pairs ground run 26 from 7 to 3 HP while every search came back empty)
                    cand.append((sc, seq + [ai], snap(), (p["x"] // 8, p["feet_y"] // 8, p.get("facing"), (min(near, key=lambda e: abs(e["dx"]))["dx"] // 8) if near else None)))   # the thug's position keeps timing variants apart
                if time.time() - t0 > time_s:
                    break
            if best or not cand or time.time() - t0 > time_s:
                break
            _byc = {}
            for c in cand:                                                   # v150: one node per 8-px cell - a closeness-scored beam all walked into the same grab (run 47: 80 candidates, 0 survivors at depth 8)
                if c[3] not in _byc or c[0] > _byc[c[3]][0]:
                    _byc[c[3]] = c
            cand = sorted(_byc.values(), key=lambda c: -c[0]); nodes = [c[:3] for c in cand[:beam]]
    finally:
        pb.load_state(io.BytesIO(root)); env.kick_penalty = False
    if best is None and best_partial is not None:
        best = best_partial[1][:12]
        print("maze-knockdown partial %d steps (gain %d) in %.0fs" % (len(best), best_partial[0], time.time() - t0), flush=True)
        return best
    print("maze-knockdown %s in %.0fs" % (("plan %d steps" % len(best)) if best else "no plan", time.time() - t0), flush=True)
    return best


def maze_under_trap_shaft(pyboy, player) -> bool:
    """v148: the hall past the sealed room has dark ceiling shafts (tile 362 from the ceiling down to head height) that kill
    whatever stands beneath them within a few steps - four local runs died standing at scroll 96, x 78, feet 67. True when
    the three tile rows above the player's head at the player's column are all 362."""
    try:
        tm = pyboy.botsupport_manager().tilemap_background()
        scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
        # the shaft is 343/359/243 stacked over a 362 bottom at head height (tile rows y 48 and y 56 in the hall, SCY 0); its
        # kill zone is ~16 px wider than the two tile columns (x 90 died with the shaft at x 64-79)
        r56 = ((scy + 56) // 8) % 32; r48 = ((scy + 48) // 8) % 32
        for ddx in (-24, -16, -8, 0, 8, 16, 24):
            tx = ((scx + player["x"] + ddx) // 8) % 32
            if tm.tile_identifier(tx, r56) == 362 and tm.tile_identifier(tx, r48) == 243:
                return True
        return False
    except Exception:
        return False


def maze_trap_screen(pyboy):
    """v149: screen x of the ceiling shaft's left edge if one is on screen (tile 243 over 362 at rows y 48/56), else None.
    On such a screen everything right of (shaft_left - 28) killed a standing player within 10-30 steps in the idle tests
    (both lanes, x 105-144), so the safe place to wait for the room's thugs is the left third."""
    try:
        tm = pyboy.botsupport_manager().tilemap_background()
        scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
        r56 = ((scy + 56) // 8) % 32; r48 = ((scy + 48) // 8) % 32
        for c in range(21):
            tx = ((scx // 8) + c) % 32
            if tm.tile_identifier(tx, r56) == 362 and tm.tile_identifier(tx, r48) == 243:
                return c * 8 - (scx % 8)
    except Exception:
        pass
    return None


SKULL_TILES = set(range(284, 288)) | set(range(300, 304)) | set(range(316, 320)) | set(range(332, 336)) | set(range(348, 352)) | {343, 359, 243}
# the hall's skull-door blocks: 4 columns x 5 tile rows (y 24..56 at SCY 0); the barred/open variant animates its two middle
# columns through other ids (run 42: the middle skull's zone vanished after step 0), so blocks are built from any skull tile
# in any of the five rows and runs are merged across gaps of up to 2 columns


SKULL_STATE_ADDR, SKULL_X_ADDR, SKULL_TIMER_ADDR = 0xC55E, 0xC484, 0xC569
SKULL_RADIUS = 50


def maze_skull_blocks_present(pyboy) -> bool:
    """True when skull-door blocks are on screen (tile test on rows y 24..56)."""
    try:
        tm = pyboy.botsupport_manager().tilemap_background()
        scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
        rows = [((scy + y) // 8) % 32 for y in (24, 32, 40, 48, 56)]
        hits = sum(1 for c in range(0, 21) if any(tm.tile_identifier(((scx // 8) + c) % 32, r) in SKULL_TILES for r in rows))
        return hits >= 6
    except Exception:
        return False


def maze_skull_zones(pyboy, idle: bool = False):
    """v150 (RAM model, measured 2026-10-04): the hall's skull doors are a state machine - 0xC55E = 0 idle (0xC569 counts
    48 -> 0, ~17-40 steps), 1 opening, 2 open/aiming (~15 steps), 3 attack (a 24-px flying sprite drops in front of the
    skull); 0xC484 = the active skull's x (16 / 80 / 144 on the trap-room screen). Anyone within ~45 px of that x when
    the attack lands (11-17 steps into state 2) dies; x 129+ survived the middle skull, x 98-122 died. Idle = safe
    everywhere, walking through early in state 2 is fine. Returns [(X - 50, X + 50)] while a skull is active, else []."""
    try:
        if not maze_skull_blocks_present(pyboy):
            return []
        st = pyboy.get_memory_value(SKULL_STATE_ADDR); X = pyboy.get_memory_value(SKULL_X_ADDR)
        if st in (1, 2, 3) and 0 < X < 168:
            return [(X - SKULL_RADIUS, X + SKULL_RADIUS)]
        if not idle:
            return []                                             # idle zones only on request (run 57k: they froze the walk-on rule for 1700 steps)
        # idle: no skull kills, but the next one to open is unknown and the screen is too narrow for a spot 50 px from all
        # three - the midpoints between skulls (x ~48, ~112) are the only places every activation can be escaped from in
        # time (run 57j died 6 px from the skull that opened next). Narrow 40-px zones around each skull steer the
        # gap-centre wait and the planners' end positions there.
        tm = pyboy.botsupport_manager().tilemap_background()
        scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
        rows = [((scy + y) // 8) % 32 for y in (24, 32, 40, 48, 56)]
        cols = [c for c in range(-1, 22) if any(tm.tile_identifier(((scx // 8) + c) % 32, r) in SKULL_TILES for r in rows)]
        runs = []
        for c in cols:
            if runs and c <= runs[-1][1] + 3:
                runs[-1][1] = c
            else:
                runs.append([c, c])
        zones = []
        for a, b in runs:
            if b - a + 1 == 2:
                a, b = a - 1, b + 1
            if b - a + 1 >= 3:
                cx = (a * 8 + (b + 1) * 8) // 2 - (scx % 8)
                zones.append((cx - 20, cx + 20))
        return zones
    except Exception:
        return []


def maze_in_skull_zone(pyboy, player, zones=None) -> bool:
    zones = maze_skull_zones(pyboy) if zones is None else zones
    return player is not None and any(a <= player["x"] <= b for a, b in zones)


def maze_skull_exit(zones, x):
    """Direction out of the zone containing x: the nearer edge, never toward the screen border."""
    for a, b in zones:
        if a <= x <= b:
            left_ok, right_ok = a > 8, b < 152
            if left_ok and (not right_ok or x - a <= b - x):
                return "left"
            return "right"
    return None


def _detour_ok(zones, x, direction):
    """True when leaving the zone containing x in `direction` is at most 32 px longer than the nearer way out."""
    for a, b in zones:
        if a <= x <= b:
            via = (x - a) if direction == "left" else (b - x)
            return via <= min(x - a, b - x) + 32
    return True


def maze_gap_action(zones, x):
    """Walk to the centre of the nearest gap between skull zones (or stand there): the only place to wait for thugs."""
    edges = [0] + [v for z in zones for v in z] + [160]
    # (centres are clamped to 28..132 below - run 52 died cornered at x 149 by a 4-HP thug)
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] >= 20]
    if not gaps:
        return "a"
    centres = [min(132, max(28, (a + b) // 2)) for a, b in gaps]
    cx = min(centres, key=lambda c: abs(c - x))                          # nearest midpoint (run 57m: preferring the right-hand one walked the player across the middle skull's front as it opened)
    return "a" if abs(cx - x) <= 5 else ("right" if cx > x else "left")


def maze_throne_room(pyboy):
    """v150: (in the throne room, its door open). The final room's dragon pillars put tiles 244/245 at columns 5-6 of tile
    row y 8 (SCY 0); the shield door at columns 8-11 reads 232.. while closed and 362 once the gunman has come out."""
    try:
        tm = pyboy.botsupport_manager().tilemap_background()
        scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
        r1 = ((scy + 8) // 8) % 32; c0 = scx // 8
        here = (tm.tile_identifier((c0 + 5) % 32, r1) == 244 and tm.tile_identifier((c0 + 6) % 32, r1) == 245
                and tm.tile_identifier((c0 + 7) % 32, r1) == 331 and tm.tile_identifier((c0 + 12) % 32, r1) == 330
                and tm.tile_identifier((c0 + 8) % 32, r1) in (362, 232))     # run 56: a dragon-banner corridor (244-247 repeated) with a 24-px heavy fooled the looser test
        return here, here and tm.tile_identifier((c0 + 8) % 32, r1) == 362
    except Exception:
        return False, False


def plan_hp_prefix(env, seq, min_len: int = 8):
    """v153g: replay a plan in the emulator and return the prefix before the first HP drop (None if shorter than min_len).
    Nav plans are searched with an HP allowance and in the hideout corridor that allowance is stalactite damage. Restores."""
    pb = env.pyboy
    b = io.BytesIO(); pb.save_state(b); root = b.getvalue()
    h0 = read_health(pb); l0 = pb.get_memory_value(0xC499); cut = len(seq)
    try:
        for i, ai in enumerate(seq):
            env._run_action_on_emulator(ai)
            _pp = sprite_radar(pb).get("player")
            if read_health(pb) < h0 or pb.get_memory_value(0xC499) < l0 or _pp is None or _pp["feet_y"] > 122:   # v153j: a fall shows as feet > 122 / sprite gone before the life counter moves (fullgame5 step 4115: a forest pit, no HP lost)
                cut = max(0, i - 8); break                                              # v153w: stop well short - a cut 2-4 steps before the hit parked the player next to the hazard (fullgame10 step 4704: cut at 32 of 39 on the balcony, dead 40 steps later)
        else:
            for _ in range(24):                                                     # v153j: the life counter lags a fall by ~15 steps - idle tail
                env._run_action_on_emulator(env.action_names.index("a") if hasattr(env, "action_names") else 4)
                if pb.get_memory_value(0xC499) < l0 or read_health(pb) < h0:      # v153p: HP too - corr7: a 56-step nav walk ended with a stalactite directly overhead (dx 0, top 33) and the first free step took 2 ticks
                    cut = max(0, len(seq) - 12); break
    finally:
        pb.load_state(io.BytesIO(root)); env.kick_penalty = False
    if cut < len(seq):
        print(f"nav-plan cut at step {cut} of {len(seq)}: HP would drop", flush=True)
    return list(seq[:cut]) if cut >= min_len else None


def maze_plan_survives(env, seq, tail: int = 24) -> bool:
    """v150: replay a plan in the emulator and then idle `tail` steps - True if no life is lost. The hall's skull kill lands
    6-20 steps after the player stops in front of the open skull, so a plan that ends there looks fine to the searches
    (runs 57d/57e died exactly that way); the emulator, not a zone heuristic, is the judge. Restores the emulator."""
    pb = env.pyboy
    b = io.BytesIO(); pb.save_state(b); root = b.getvalue()
    L0 = pb.get_memory_value(0xC499); ok = True
    try:
        for ai in seq:
            env._run_action_on_emulator(ai)
            if pb.get_memory_value(0xC499) < L0:
                ok = False; break
        if ok:
            idle = env.action_names.index("a") if hasattr(env, "action_names") else 4
            for _ in range(tail):
                env._run_action_on_emulator(idle)
                if pb.get_memory_value(0xC499) < L0:
                    ok = False; break
    finally:
        pb.load_state(io.BytesIO(root)); env.kick_penalty = False
    return ok


def maze_plan_with_exit(env, action_names, seq):
    """v150: a plan that passes its fight but ends where the player dies within 24 idle steps (in front of the open skull)
    is salvaged by appending a short walk left or right (6-14 steps) and re-checking; None if nothing survives."""
    if maze_plan_survives(env, list(seq)):
        return list(seq)
    for n in (6, 10, 14):
        for d in ("left", "right"):
            cand = list(seq) + [action_names.index(d)] * n
            if maze_plan_survives(env, cand):
                print(f"maze: plan of {len(seq)} steps salvaged with {d} x{n}", flush=True)
                return cand
    return None


def maze_wall_escape(env, action_names, direction="right"):
    """v150: at a wall, try lane shifts in the emulator - DOWN x8 then walk, UP x8 then walk - and return the first that
    survives (no life lost, feet not falling) AND gains 12+ px of scroll+x. Run 57f: a blind DOWN at scroll 161 fell
    into a front-lane hole. Restores the emulator."""
    pb = env.pyboy
    b = io.BytesIO(); pb.save_state(b); root = b.getvalue()
    L0 = pb.get_memory_value(0xC499); p0 = sprite_radar(pb).get("player"); scx0 = pb.get_memory_value(0xFF43)
    if p0 is None:
        return None
    best = None
    try:
        for first, n in (("down", 8), ("up", 8), ("down", 5), ("up", 5), ("down", 11), ("up", 11), ("down", 3), ("up", 3)):
            pb.load_state(io.BytesIO(root))
            seq = [action_names.index(first)] * n + [action_names.index(direction)] * 8   # run 57p: from a mid-lane position neither 8-press shift worked - try several depths
            ok = True
            for ai in seq:
                env._run_action_on_emulator(ai)
                p = sprite_radar(pb).get("player")
                if pb.get_memory_value(0xC499) < L0 or p is None or p["feet_y"] > 95:
                    ok = False; break
            if not ok:
                continue
            p = sprite_radar(pb).get("player"); scx = pb.get_memory_value(0xFF43)
            adv = ((scx - scx0 + 128) % 256) - 128 + (p["x"] - p0["x"])
            if adv >= 12:
                best = seq; break
    finally:
        pb.load_state(io.BytesIO(root)); env.kick_penalty = False
    return best


def maze_timed_walk(env, action_names, direction="right", n=16):
    """v150: lock room 2's moving wall block hits for 8 HP and is invisible to the planners. Simulate waiting 0..16 steps
    and then walking `n` steps; return the first sequence that loses no HP and no life (else the least HP loss)."""
    pb = env.pyboy
    b = io.BytesIO(); pb.save_state(b); root = b.getvalue()
    H0 = pb.get_memory_value(0xC498); L0 = pb.get_memory_value(0xC499); best = None
    try:
        for wait in (0, 4, 8, 12, 16, 20):
            pb.load_state(io.BytesIO(root))
            seq = [action_names.index("a")] * wait + [action_names.index(direction)] * n
            ok = True
            for ai in seq:
                env._run_action_on_emulator(ai)
                if pb.get_memory_value(0xC499) < L0:
                    ok = False; break
            if not ok:
                continue
            loss = H0 - pb.get_memory_value(0xC498)
            p = sprite_radar(pb).get("player")
            if p is None or p["feet_y"] > 95:
                continue
            if best is None or loss < best[0]:
                best = (loss, seq)
            if loss <= 0:
                break
    finally:
        pb.load_state(io.BytesIO(root)); env.kick_penalty = False
    return best[1] if best else None


def maze_traverse_plan(env, action_names, direction="right", depth=40, beam=12, time_s=30.0):
    """v150: HP-aware traverse for the hall screens with the roaming wall block (lock room 2, lap-3 corridor): beam search
    over walk/lane/hop/wait, pruning any HP loss, life loss or fall; score = scroll+x advance. Win = 64 px of advance;
    else the best partial with >= 24 px. Restores the emulator."""
    pb = env.pyboy
    def snap():
        b = io.BytesIO(); pb.save_state(b); return b.getvalue()
    root = snap(); t0 = time.time(); L0 = pb.get_memory_value(0xC499); H0 = pb.get_memory_value(0xC498)
    p0 = sprite_radar(pb).get("player"); scx0 = pb.get_memory_value(0xFF43)
    if p0 is None:
        return None
    names = (direction, "up", "down", "a", "jump_kick") + (("kick_right",) if direction == "right" else ("kick_left",))
    idx = {n: action_names.index(n) for n in names if n in action_names}
    def adv():
        p = sprite_radar(pb).get("player"); scx = pb.get_memory_value(0xFF43)
        if p is None:
            return None, None
        return ((scx - scx0 + 128) % 256) - 128 + (p["x"] - p0["x"]), p
    nodes = [(0.0, [], root)]; best = None; best_partial = None
    try:
        for d in range(depth):
            cand = []
            for sc0, seq, st in nodes:
                for name, ai in idx.items():
                    pb.load_state(io.BytesIO(st)); env._run_action_on_emulator(ai)
                    if pb.get_memory_value(0xC499) < L0 or pb.get_memory_value(0xC498) < H0:
                        continue
                    a, p = adv()
                    if p is None or p["feet_y"] > 95:
                        continue                                      # run 57u: feet 97 is already falling into a hole (lanes end at 94)
                    sc = a - d * 0.05
                    if a >= 64 and abs(p["feet_y"] - p0["feet_y"]) <= 30 and p["feet_y"] <= 94:
                        if best is None or len(seq) + 1 < len(best):
                            best = seq + [ai]
                        continue
                    if a >= 24 and p["feet_y"] <= 94 and (best_partial is None or a > best_partial[0]):
                        best_partial = (a, seq + [ai])
                    cand.append((sc, seq + [ai], snap(), (p["x"] // 8, p["feet_y"] // 8, pb.get_memory_value(0xFF43) // 8)))
                if time.time() - t0 > time_s:
                    break
            if best or not cand or time.time() - t0 > time_s:
                break
            byc = {}
            for c in cand:
                if c[3] not in byc or c[0] > byc[c[3]][0]:
                    byc[c[3]] = c
            nodes = [c[:3] for c in sorted(byc.values(), key=lambda c: -c[0])[:beam]]
    finally:
        pb.load_state(io.BytesIO(root)); env.kick_penalty = False
    if best:
        return best
    return best_partial[1] if best_partial else None


def maze_boss(step_radar):
    """v150: the final gunman (hall throne room) - the only 24-px-wide enemy sprite in the game so far."""
    bs = [e for e in step_radar.get("enemies", []) if e.get("w", 16) >= 24]
    return min(bs, key=lambda e: abs(e["dx"])) if bs else None


def boss_finish_plan(env, action_names, time_s: float = 30.0, depth: int = 40, beam: int = 10):
    """v150: at bar 0 (0xC595 == 0) the gunman takes no damage but leaves after ONE standing hit (punch/kick; jump kicks
    never do it). Beam search over walking + standing attacks: win = the 24-px sprite gone for two expansions with the
    player alive; prune = life lost, > 24 raw HP lost, or the player gone while the boss is still there. Restores the emulator."""
    pb = env.pyboy
    def snap():
        b = io.BytesIO(); pb.save_state(b); return b.getvalue()
    def look():
        r = sprite_radar(pb, cave=True); bs = [e for e in r["enemies"] if e.get("w", 16) >= 24]
        return (min(bs, key=lambda e: abs(e["dx"])) if bs else None), r.get("player")
    root = snap(); t0 = time.time(); L0 = pb.get_memory_value(0xC499); H0 = pb.get_memory_value(0xC498)
    idx = {n: action_names.index(n) for n in ("left", "right", "up", "down", "a", "b")}
    nodes = [(0.0, [], root, 0)]; best = None
    try:
        for d in range(depth):
            cand = []
            for sc0, seq, st, gone in nodes:
                for name, ai in idx.items():
                    pb.load_state(io.BytesIO(st)); env._run_action_on_emulator(ai)
                    hp = pb.get_memory_value(0xC498)
                    if pb.get_memory_value(0xC499) < L0 or hp < H0 - 24:
                        continue
                    b, pl = look()
                    if b is None:
                        if gone + 1 >= 2 and pl is not None:
                            if best is None or len(seq) + 1 < len(best):
                                best = seq + [ai]
                            continue
                        cand.append((sc0 + 50.0, seq + [ai], snap(), gone + 1))
                        continue
                    if pl is None:
                        continue
                    sc = -abs(b["dx"]) * 0.1 - abs(b["dy"]) * 0.1 - (H0 - hp) * 1.0 - d * 0.05
                    if b["dy"] <= -15:
                        sc += 30.0                                        # launched: he is on his way out
                    if name in ("a", "b") and abs(b["dx"]) <= 26 and abs(b["dy"]) <= 10:
                        sc += 5.0
                    cand.append((sc, seq + [ai], snap(), 0))
                if time.time() - t0 > time_s:
                    break
            if best or not cand or time.time() - t0 > time_s:
                break
            cand.sort(key=lambda c: -c[0]); nodes = cand[:beam]
    finally:
        pb.load_state(io.BytesIO(root)); env.kick_penalty = False
    return best


def maze_narrow_ledge(step_radar) -> bool:
    """Both pit edges within 24 px: a knockback here is a fall (the 16-px ledge B of the ledge row)."""
    pits = step_radar.get("pits", [])
    l = [abs(g["dx_edge"]) for g in pits if g["side"] == "left" and abs(g["dx_edge"]) <= 24]
    r = [abs(g["dx_edge"]) for g in pits if g["side"] == "right" and abs(g["dx_edge"]) <= 24]
    return bool(l and r)


PLAN_FIGHTS = {int(x) for x in os.environ.get("DD_PLAN_FIGHTS", "").split(",") if x.strip()}   # v82: areas where the fight planner takes over (46 bruiser, 48 boss)
PLAN_ANY_ENEMY = {int(x) for x in os.environ.get("DD_PLAN_ANY_ENEMY", "").split(",") if x.strip()}   # v86: areas where ANY enemy within reach triggers a plan (mission-4 arena 86)
PLAN_PROGRESS_W = float(os.environ.get("DD_PLAN_PROGRESS_W", "1.0"))   # v90: score weight per px gained rightward during mission-4 fight plans
PLAN_ANY_ENEMY_ALWAYS = {int(x) for x in os.environ.get("DD_PLAN_ANY_ENEMY_ALWAYS", "").split(",") if x.strip()}   # v91: areas where any enemy triggers a plan in every mission
PLAN_ANY_ENEMY_PX = int(os.environ.get("DD_PLAN_ANY_ENEMY_PX", "90"))   # v96: plan before contact - the model lost more HP between plans than the plans did
PLAN_ESCAPE = {int(x) for x in os.environ.get("DD_PLAN_ESCAPE", "86").split(",") if x.strip()}   # v99: mission-4 areas planned in escape mode (arena 86: 4 waves, exits when both slots are empty)
NAV_PLAN = os.environ.get("DD_NAV_PLAN", "1") == "1"            # v108: search coarse moves out of a navigation stall (any area not seen in the recent window)
NAV_STALL_DECISIONS = int(os.environ.get("DD_NAV_STALL_DECISIONS", "30"))
NAV_WINDOW_STEPS = int(os.environ.get("DD_NAV_WINDOW_STEPS", "1500"))
NAV_STALL_STEPS = int(os.environ.get("DD_NAV_STALL_STEPS", "900"))
NAV_FIRST_ALL = os.environ.get("DD_NAV_FIRST", "0") == "1"                # v154i
REWIND = os.environ.get("DD_REWIND", "0") == "1"                         # v154: rewind-on-death teacher (local demonstration runs): reload a snapshot >= REWIND_BACK steps old when a life is lost, up to REWIND_MAX times per chunk
REWIND_MAX = int(os.environ.get("DD_REWIND_MAX", "3"))
REWIND_EVERY, REWIND_BACK = 50, 100
PLAN_MAX = int(os.environ.get("DD_PLAN_MAX", "300"))                    # v153: 80 was for a few owned fights; planning every mission-3+ fight needs ~150 per full game
PLAN_M3_ALL = os.environ.get("DD_PLAN_M3_ALL", "1") == "1"                 # v153: plan every fight from mission 3 on (see any_enemy_here)
PLAN_BEAM_M4 = int(os.environ.get("DD_PLAN_BEAM_M4", "40"))                  # v119: beam width for mission-4 fights (arrivals at the arena carry 4 HP; 1-tick kills at beam 24 were 0-tick at 40)                          # v117: fight plans per episode (forest thugs + arena stream need far more than 20)
HP_LOG = [] if os.environ.get("DD_HP_LOG") else None                       # v113 debug: per-step (step, chunk, hp, lives, plan_len, enemies, hp bytes) dumped to DD_HP_LOG at episode end     # v109/v110: steps with neither the chunk index advancing nor a new scroll maximum (v109 fired in the opening street, 40 workers x 180 s)
WALK_IN = os.environ.get("DD_WALK_IN", "1") == "1"
FIGHT_PER_STEP = os.environ.get("DD_FIGHT_PER_STEP", "0") == "1"   # v57 experiment; off since v61

LEVEL_NAMES = {15: "mission1", 84: "mission1-boss", 48: "unknown-48", 89: "mission2", 11: "unknown-11",
               12: "mission3", 5: "mission3-part2"}


def s3_client():
    bucket = os.environ.get("S3_BUCKET_NAME")
    if not bucket:
        return None, None
    import boto3
    params = {k: v for k, v in {
        "aws_access_key_id": os.environ.get("S3_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.environ.get("S3_SECRET_ACCESS_KEY"),
        "region_name": os.environ.get("S3_REGION_NAME"),
        "endpoint_url": os.environ.get("S3_ENDPOINT_URL"),
    }.items() if v}
    return boto3.client("s3", **params), bucket


def build_env_config(max_steps: int, save_video: bool, run_tag: str,
                     init_states: Optional[List[str]] = None) -> Dict[str, Any]:
    """Same knobs run-ray-dd.py used, minus anything that only mattered to PPO."""
    return {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 8,
        "init_state_dir": "ignored/",
        "available_init_states": init_states or DEFAULT_INIT_STATES,
        "max_steps": max_steps,
        "print_rewards": False,
        "save_video": save_video,
        "fast_video": True,
        "gb_path": "ignored/dd.gb",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "extra_buttons": False,
        "s3_bucket_name": os.environ.get("S3_BUCKET_NAME"),
        "s3_endpoint_url": os.environ.get("S3_ENDPOINT_URL"),
        "s3_access_key_id": os.environ.get("S3_ACCESS_KEY_ID"),
        "s3_secret_access_key": os.environ.get("S3_SECRET_ACCESS_KEY"),
        "s3_region_name": os.environ.get("S3_REGION_NAME"),
        "s3_video_prefix": f"ddenv_videos/decision/{run_tag}/",
        "s3_state_prefix": f"ddenv_states/decision/{run_tag}/",
    }


class EpisodeTrace:
    """Accumulates one episode and ships it as a compressed npz + a JSON summary line."""

    def __init__(self, action_names: List[str], record_obs: bool):
        self.action_names = action_names
        self.record_obs = record_obs
        self.obs: List[np.ndarray] = []
        # step, action, confidence, latency, ok, reward, score, lives, level,
        # player_x, player_feet_y, nearest_dx, nearest_dy, enemy_count, reachable_enemy_count, distance_px, health_ticks
        self.rows: List[List[float]] = []
        self.probs: List[List[float]] = []

    def add(self, obs, decision, reward, info, step, radar=None, distance_px=0, exec_action=-1):
        if self.record_obs:
            self.obs.append(obs)
        p = (radar or {}).get("player") or {}
        reach = [e for e in (radar or {}).get("enemies", []) if e.get("reachable", True)]
        n = reach[0] if reach else {}
        self.rows.append([step, decision.action, decision.confidence, decision.latency_s, int(decision.ok),
                          float(reward), int(info.get("score", 0)), int(info.get("lives", 0)), int(info.get("level", 0)),
                          p.get("x", -1), p.get("feet_y", -1), n.get("dx", 0), n.get("dy", 0),
                          (radar or {}).get("enemy_count", 0), (radar or {}).get("reachable_enemy_count", 0), distance_px,
                          (radar or {}).get("health_ticks", -1), exec_action])   # v59: column 17 = action actually executed (reflex)
        self.probs.append([decision.probabilities.get(n, 0.0) for n in self.action_names])

    def to_npz_bytes(self, meta: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        arrays = {
            "steps": np.asarray(self.rows, dtype=np.float32),
            "teacher_probs": np.asarray(self.probs, dtype=np.float32),
            "action_names": np.asarray(self.action_names),
            "meta": np.asarray(json.dumps(meta)),
        }
        if self.record_obs and self.obs:
            arrays["obs"] = np.stack(self.obs).astype(np.uint8)   # (T, 128, 40, 3) - what PPO would consume
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()


class EnvWorker:
    """One emulator + one round-robin client. Wrapped as a Ray actor below."""

    def __init__(self, idx: int, env_config: Dict[str, Any], client_kwargs: Dict[str, Any], episodes: int,
                 run_tag: str, record_traces: bool, record_obs: bool, decision_interval: int = 1):
        from ddenv import DDEnv  # import inside the worker so the module resolves in the Ray working_dir

        self.idx = idx
        self.episodes = episodes
        self.run_tag = run_tag
        self.env = DDEnv(env_config)
        self.client = DecisionModelClient(start_offset=idx, **client_kwargs)
        self.include_native = self.client.include_native_frame
        self.action_names = [n for n, _ in self.client.action_options]
        self.s3, self.bucket = s3_client() if record_traces else (None, None)
        self.record_obs = record_obs
        self.decision_interval = max(1, int(decision_interval))
        self.progress = {"episode": 0, "step": 0, "return": 0.0, "score": 0, "lives": 0, "stage": 0, "distance": 0, "vertical": 0}

    def get_progress(self) -> Dict[str, Any]:
        return {"idx": self.idx, **self.progress, **self.client.stats}

    def _save_state(self, ep: int, steps: int, reason: str, saved: List[str]) -> None:
        """Snapshot the emulator (PyBoy save state) to S3 so a situation can be reloaded locally for diagnosis."""
        if os.environ.get("DD_SAVE_LOCAL_DIR") and len(saved) < 400:                      # v153r: 140 periodic saves per 14000-step run                      # v125: local fixtures without S3
            try:
                path = os.path.join(os.environ["DD_SAVE_LOCAL_DIR"], f"w{self.idx:03d}_ep{ep:02d}_step{steps:05d}_{reason}.state")
                with open(path, "wb") as fh:
                    self.env.pyboy.save_state(fh)
                saved.append(path)
            except Exception as exc:
                log.warning("worker %d: local save_state failed: %s", self.idx, exc)
            return
        if not self.s3 or len(saved) >= 16:   # v135: 8 slots were used up by mission-1/2 events, so no arena or cave state was ever saved
            return
        try:
            buf = io.BytesIO()
            self.env.pyboy.save_state(buf)
            key = f"states/w{self.idx:03d}_ep{ep:02d}_step{steps:05d}_{reason}.state"
            self._put(key, buf.getvalue(), "application/octet-stream")
            saved.append(key)
        except Exception as exc:
            log.warning("worker %d: save_state failed: %s", self.idx, exc)

    def _put(self, key: str, body: bytes, content_type: str) -> None:
        if not self.s3:
            return
        try:
            self.s3.put_object(Bucket=self.bucket, Key=f"decision_runs/{self.run_tag}/{key}", Body=body, ContentType=content_type)
        except Exception as exc:
            log.warning("worker %d: upload of %s failed: %s", self.idx, key, exc)

    def run(self) -> List[Dict[str, Any]]:
        results = []
        try:
            for ep in range(self.episodes):
                obs, info = self.env.reset()
                trace = EpisodeTrace(self.action_names, self.record_obs)
                prev_name: Optional[str] = None
                ep_return, steps, conf_sum, lat_sum, ok_steps, queried = 0.0, 0, 0.0, 0.0, 0, 0
                action_hist: Dict[str, int] = {}
                start_level = int(info.get("level", 0))
                start_lives = int(info.get("lives", 0))
                levels_visited: List[int] = [start_level]
                max_score = int(info.get("score", 0))
                t_ep = time.time()
                done = False
                d = None
                scroll = ScrollTracker(self.env.pyboy)
                recent: List[str] = []
                pos_hist: List[Tuple[int, int]] = []
                radar = None
                saved_states: List[str] = []
                min_health = 7
                stuck_run = 0
                door_mode = False
                progress_sig = None          # (distance, vertical, score) for generic no-progress detection
                progress_age = 0
                noprog_saved_at = -10**9
                last_enemy: Dict[str, Any] = {}     # side/dx of the most recent reachable enemy + when
                enemy_hist: List[Tuple] = []        # (frozenset of enemy positions, health) per decision, for passivity
                belt = BeltTracker()                # conveyor detection from unexplained drift (mission 2 walkway)
                reflex_steps = 0                    # env steps where the conveyor reflex overrode the model
                exec_name: Optional[str] = None     # action actually sent to the emulator last step
                noprog_hist: Dict[str, int] = {}    # model actions since progress last changed (for the stall sidecar)
                last_advice = ""
                travel_dir: Optional[str] = None    # sticky "the way on is LEFT/RIGHT" for the current walkway (mission 2+)
                travel_scy = None                   # SCY when it was set; a different floor (SCY changed) forgets it
                prev_lives = int(info.get("lives", 0)); last_life_lost_step = -10**9
                rewind_ring, rewind_counts, rewind_hold_until, rewinds_total = [], {}, -1, 0   # v154
                rewind_deaths, rewind_last_used = [], None                                     # v154d
                last_gain_step, lives_lost, last_score, streak = 0, 0, int(info.get("score", 0)), 0
                boss_hits = 0                # score increments while a heavy enemy / the mission-1 boss is up; he needs 20 in one life
                saved_areas = set()
                last_heavy = None            # where the heavy enemy was last seen (he leaves the radar while knocked down)
                heavy_gate = HeavyGate()     # v77: strips the heavy flag from thug pairs mis-read as the bruiser
                planner = FightPlanner(self.env, self.action_names, log)          # v114: always present (the stalled-fight trigger needs it in any chunk); PLAN_FIGHTS only gates the chunk-based trigger
                nav = NavPlanner(self.env, self.action_names, log) if NAV_PLAN else None
                hop = HopPlanner(self.env, self.action_names); hop_block_until = -1
                ckpt, ckpt_prev, ckpt_step = None, None, -1          # v129: in-memory emulator checkpoints for soft-lock recovery
                ckpt_ring = []                                        # v143: (step, state) of the last 6 checkpoints - reload #N goes N checkpoints back (v142: one worker reloaded the same re-freezing checkpoint three times)
                ring_unco = []                                       # v154m3: unconditional healthy-checkpoint ring (balcony lock has a permanent on-screen thug)
                hideout_entered = bool(os.environ.get("DD_FORCE_HIDEOUT"))   # v138; v141: local tests from hideout fixtures
                door2_block_until = -1                              # v141
                maze_entered = bool(os.environ.get("DD_FORCE_MAZE"))   # v141
                room7_block_until = -1
                hdoor_block_until = -1
                blind_hold_until = -1                                                   # v153ac                                                   # v153aa                                 # v144: sealed-room door script throttle
                trap_block_until = -1                                  # v148b: planners off while walking out from under a ceiling shaft
                boss_seen_here, boss_gone_steps, finish_block_until, ending_steps, game_complete = False, 0, -1, 0, False   # v150: final-boss scripts
                boss_fail_here = 0                                                              # v154g
                skull_kd_until = -1                                    # v150: skull-room knockdown-search block
                maze_fail_streak = 0                                   # v150: escalating hall search budget
                maze_prev_x, maze_retreat_dir = None, None             # v150: blocked-retreat reversal in a skull zone
                kd_saved_areas = set()                                 # v145: one knockdown-situation save per area
                door2_bruiser_seen = False                             # v145: door-2 script only after the room's bruiser has activated
                maze_lane_until, maze_lane_dir = -1, "up"              # v144: lane shift after a failed hop (front-lane holes in the hall)
                maze_stuck_n, maze_prev_scx = 0, None                   # v150: wall detector
                _zones, _in_zone, _live_here, _enemy_close = [], False, False, False   # defined here for the DD_MAZE_DEBUG print (run 57 crashed on it during a plan replay)
                maze_backoff = False                                   # v143d: retreating from the screen border toward mid-screen
                maze_prev_hp, maze_prev_feet, plan_from_nav = 7, None, False   # v143d: nav-replay abort bookkeeping
                maze_prev_slots = {}                                   # v143d: enemy HP slots last step (a value counting down 9..1 is a knockout timer = camera lock)
                frozen_pos, frozen_since, softlock_reloads, softlock_block_until = None, 0, 0, -1
                nav_count, nav_block_until, no_progress_run, area_hist = 0, -1, 0, []   # v108
                nav_guard_until, nav_recent = -1, set()                                 # v108b: after an exit, undo the model's regression into the escaped region
                tracker = ProgressTracker(); nav_exit_pos = -1; plan_saved_areas = set()   # v109
                if os.environ.get("DD_FORCE_BEYOND"):                                       # v133: local cave tests - pretend the arena was passed
                    tracker.pos = len(CANON) - 1; tracker.max_pos = tracker.pos; tracker.beyond = True
                fight_fail_streak, fight_fail_chunk = 0, -1                                 # v115: consecutive planner misses in the current chunk -> leave instead of fighting
                boss1_dead = False                                                          # v115: mission-1 boss killed (chunk 48)
                nav_best_px, nav_gain_step = -1, 0                                        # v110: last step that set a new scroll maximum
                plan, plan_block_until, plan_count = [], -1, 0      # v82: replayed search result (action indices)
                fight_armed, fight_last_seen = False, -999          # v84: once a plan was made, keep planning wherever the fight scrolls
                fight_kills = 0
                mission4 = False             # v87: set by the mission-4 intro areas; cleared on the mission-3 respawn (11/12)
                seen46 = False
                bruiser_hp_min, bruiser_hp_at_life_lost, bruiser_hp_seen_step = None, [], -999
                stal_dodge_dir, stal_dodge_until = None, -1                              # v153o: latched stalactite dodge
                _hp_trace_prev = None                                                   # v153: per-step HP-drop trace (where the ticks go between planner calls)   # v78: enemy HP byte 0xC595 (42 fresh)
                while not done:
                    # mission 3 (the forest): reached after mission 2's climb, or started there. Its river tiles double as
                    # floor tiles in the mission-1 boss room, so the radar is told rather than left to guess.
                    in_forest = ((start_level in (11, 12)) or (scroll.vertical_px >= 300 and radar is not None and not radar.get("multi_floor"))) and not mission4   # v99: the river rules (stone hops, pit jumps) hopped a cave-floor player into the void
                    step_radar = sprite_radar(self.env.pyboy, river=in_forest, chain=False, cave=(mission4 and tracker.beyond), stalactites=(hideout_entered and not maze_entered))   # ~0.5ms; v123: past the arena the chain columns are pits (and never ladders)
                    heavy_gate.apply(step_radar, int(info.get("level", 0)))
                    _hp_now = step_radar.get("health_ticks", -1)
                    if _hp_trace_prev is not None and 0 <= _hp_now < _hp_trace_prev:
                        _pl = step_radar.get("player") or {}
                        _en = [(int(e.get("dx", 0)), int(e.get("dy", 0)), bool(e.get("reachable")), bool(e.get("heavy"))) for e in step_radar.get("enemies", [])][:4]
                        log.info("hp-drop step=%d area=%d hp %d->%d x=%s feet=%s scy=%d scx=%d plan_left=%d last=%s enemies(dx,dy,reach,heavy)=%s stal=%s",
                                 steps, int(info.get("level", 0)), _hp_trace_prev, _hp_now, _pl.get("x"), _pl.get("feet_y"), self.env.pyboy.get_memory_value(0xFF42), self.env.pyboy.get_memory_value(0xFF43),
                                 len(plan) if plan else 0, (recent[-1] if recent else None), _en,
                                 [(st["x"] - int(_pl.get("x", 0) or 0), st["bottom"]) for st in step_radar.get("stalactites", [])])   # v153k: stalactite offsets (dx, bottom y) in the trace
                    _hp_trace_prev = _hp_now
                    tracker.update(int(info.get("level", 0)), steps, life_lost_recently=(steps - last_life_lost_step <= 240))   # v138
                    mission4 = tracker.mission() == 4
                    if mission4 and tracker.beyond and int(info.get("level", 0)) in (6, 5) and self.env.pyboy.get_memory_value(0xFF42) == 0:
                        hideout_entered = True                                        # v138: rooms 6/5 at SCY 0 = inside the hideout
                    if hideout_entered and int(info.get("level", 0)) in (73, 72):
                        maze_entered = True                                           # v141: past door 2
                    if maze_entered and not radar_mod.EXTRA_VOID:
                        radar_mod.EXTRA_VOID, radar_mod.EXTRA_BELOW = {354}, {312, 313}   # v144: hall holes past the sealed room (354 over 362/312); maze-only to keep the cave/hideout pit reading unchanged
                        if planner is not None:
                            def _maze_danger(pb, x, feet):
                                if pb.get_memory_value(0xE110) == 7 and pb.get_memory_value(0xFF43) >= 160:
                                    return False                                            # v153t: the sealed room's door tiles read as a skull zone - the hook pruned the search around the 52-HP heavies (searches "exhausted" in 46-66 s where the v144 build killed them in 180 s)
                                return maze_in_skull_zone(pb, {"x": x}) or maze_under_trap_shaft(pb, {"x": x, "feet_y": feet})   # v150: skull-door fronts (both lanes) are lethal to stand in
                            planner.danger = _maze_danger                                   # v148b: the fight search never ends or passes through lethal floor
                    # v129: soft-lock recovery. Two v128 workers spent all 12000 steps in the mission-1 warehouse doorway (chunk 84)
                    # with the player sprite frozen at one pixel: no input and no idle changes anything (1500 varied presses
                    # tested offline) - a game bug, not a policy problem. Keep a checkpoint every 200 healthy steps; when the
                    # player has not moved a pixel for 300 steps with no enemy on screen, confirm the freeze on the emulator
                    # (snapshot, 40 varied presses, restore) and reload the checkpoint taken before it. Logged and counted.
                    _pl = step_radar.get("player")
                    if _pl is not None:
                        _pos = (int(info.get("level", 0)), self.env.pyboy.get_memory_value(0xFF43), _pl["x"])   # v132: x only - the frozen sprite's feet toggle 85/86 every step, which reset the freeze clock in v131 (11 nav plans, no reload)
                        if _pos != frozen_pos:
                            frozen_pos, frozen_since = _pos, steps
                        _new_chunk = bool(area_hist) and area_hist[-1][1] != int(info.get("level", 0))
                        if ((steps - ckpt_step >= 200 and steps - frozen_since < 100 and _pl["feet_y"] <= 100 and not step_radar.get("enemies"))
                                or (_new_chunk and steps - ckpt_step >= 50 and steps - frozen_since < 20 and _pl["feet_y"] <= 100)):   # v131: also on entering a chunk (the warehouse has thugs on screen most of the time)
                            _b = io.BytesIO(); self.env.pyboy.save_state(_b); ckpt_prev, ckpt, ckpt_step = ckpt, _b.getvalue(), steps
                            ckpt_ring.append((steps, ckpt)); del ckpt_ring[:-6]
                        if (steps % 200 == 0 and steps - frozen_since < 40 and _pl["feet_y"] <= 110
                                and not any(e["reachable"] and abs(e["dx"]) <= 36 for e in step_radar.get("enemies", []))):
                            _rb = io.BytesIO(); self.env.pyboy.save_state(_rb); ring_unco.append((steps, _rb.getvalue())); del ring_unco[:-8]   # v154m3: a healthy-moving checkpoint every 200 steps even with a far enemy on screen (the balcony lock always has the doorway thug visible, so the enemy-free ring was never populated there)
                        if (steps - frozen_since >= 300 and steps >= softlock_block_until and softlock_reloads < 3
                                and (not step_radar.get("enemies") or steps - frozen_since >= 500)   # v154m: fullgame17 froze on the mission-2 balcony (x 47, no input moves the player, a thug frozen in the doorway 60 px away) - the enemy-free requirement kept the v129 reload from ever firing; after 500 frozen steps an enemy on screen is part of the lock, not a fight
                                and (int(info.get("level", 0)) == 84 or steps - frozen_since >= 500)   # v142: the warehouse doorway froze most often; v154m2: also any chunk after a 500-step freeze (fullgame17 locked in chunk 16) - the emulator test below confirms a real freeze before reloading, so a false positive just costs one verificationspect in the mild warehouse-death regression since v132)
                                and (ckpt_prev is not None or (ckpt is not None and ckpt_step < frozen_since))):
                            _s = io.BytesIO(); self.env.pyboy.save_state(_s); _moved = False; _L = int(info.get("lives", 0)); _lv = int(info.get("level", 0))
                            # v131: movement presses only - attack animations (A, B, jump kick) shift the sprite bbox 6-11 px and made
                            # the v130 test call a frozen warehouse player "moved" every time (one worker idle there for 9 nav plans)
                            for _a in ("left", "right", "up", "down", "b", "left", "right"):
                                for _k in range(10):
                                    self.env._run_action_on_emulator(self.action_names.index(_a))
                                _p2 = sprite_radar(self.env.pyboy).get("player")
                                if _a == "b":
                                    continue                                   # throw a held object, then see whether walking works again
                                if _p2 is None or abs(_p2["x"] - _pl["x"]) > 6 or self.env.pyboy.get_memory_value(0xE110) != _lv or self.env.pyboy.get_memory_value(0xC499) != _L:
                                    _moved = True; break
                            _s.seek(0); self.env.pyboy.load_state(_s); self.env.kick_penalty = False
                            softlock_block_until = steps + 300
                            if not _moved:
                                self._save_state(ep, steps, f"softlock{softlock_reloads}-a{_lv}", saved_states)   # v143e: the frozen state itself (two v143 workers re-froze after every reload - cause unknown)
                                _older = [c for c_step, c in (ckpt_ring + ring_unco) if c_step < frozen_since - 20]          # v143: checkpoints taken before the freeze, oldest first; v154m3: include the unconditional ring
                                _older = [c for _, c in sorted([(cs, cc) for cs, cc in (ckpt_ring + ring_unco) if cs < frozen_since - 20])]
                                _use = _older[-1 - min(softlock_reloads, len(_older) - 1)] if _older else (ckpt_prev if (ckpt is None or ckpt_step >= frozen_since) else ckpt)
                                if _use is not None:
                                    _frozen_for = steps - frozen_since
                                    self.env.pyboy.load_state(io.BytesIO(_use)); softlock_reloads += 1
                                    plan, fight_armed, pos_hist = [], False, []
                                    frozen_pos, frozen_since = None, steps
                                    msg = f"softlock-reload #{softlock_reloads}: player frozen at chunk {_lv} x={_pl['x']} for {_frozen_for} steps; reloaded checkpoint {softlock_reloads} back"
                                    print(msg, flush=True); log.warning("worker %d: %s", self.idx, msg)
                                    step_radar = sprite_radar(self.env.pyboy, river=in_forest, chain=False, cave=(mission4 and tracker.beyond), stalactites=(hideout_entered and not maze_entered))
                    if int(info.get("level", 0)) == 15:
                        boss1_dead = False                                                 # back at the mission-1 street (respawn): the boss is alive again        # v112: from the canonical chunk index (a death regresses it to mission 3); replaces the seen46 heuristic, which a run starting in the arena never satisfied
                    any_enemy_here = (int(info.get("level", 0)) in PLAN_ANY_ENEMY and (mission4 or int(info.get("level", 0)) == 86)) \
                        or int(info.get("level", 0)) in PLAN_ANY_ENEMY_ALWAYS \
                        or (mission4 and tracker.beyond) \
                        or (PLAN_M3_ALL and tracker.mission() >= 1) \
                        or (PLAN_M3_ALL and not (int(info.get("level", 0)) == 48 and boss1_dead) and any(e.get("heavy") and abs(e["dx"]) <= 120 for e in step_radar.get("enemies", [])))   # v153d: not the scripted second boss (fullgame4: three 100-s escape searches found nothing and the room cost 2 ticks; fullgame3 left it at 0 loss with the model surviving)   # v153b: every heavy (mission-1 boss, bruisers) in every mission - fullgame2 lost 5 ticks to the boss with no plan (the stall fallback never fires in a moving fight)   # v153r: mission 2 too; v153x: mission 1 too (fullgame11: the street took 7 -> 3 before the warehouse, model-fought) - every fight in the game is planned from first contact   # v153: missions 3+4 share ONE 7-tick HP budget (no refill until the hideout); the model-fought thug pairs and the chunk-46 bruiser cost 1-4 ticks each before the stall fallback handed them to the planner (fullgame1: 3 deaths, every one a mission-3/4 attrition death) - plan every fight from first contact
                    # v113/v114: no planner owned "stalled next to an enemy" (mission-2 ladder top with a thug on the rungs held 37/40
                    # v112 workers). no_progress_run resets while an enemy is reachable, so use position history directly: the
                    # player has not moved for 12 decisions, the score has not moved for 60 steps, and an enemy is within 60 px.
                    stalled_fight = (len(pos_hist) >= 12 and abs(pos_hist[-1][0] - pos_hist[0][0]) <= 4 and abs(pos_hist[-1][1] - pos_hist[0][1]) <= 4
                                     and steps - last_gain_step >= 60
                                     and any(abs(e["dx"]) <= 60 and abs(e["dy"]) <= 40 for e in step_radar.get("enemies", [])))
                    # v120: the arena (chunk 86) is four waves (two 21-HP thugs, the 32, the 42) and then the screen scrolls on to the cave.
                    # Escape mode with the chunk change as the only win and the rightward weight clears it from the entry fixture at
                    # 5 -> 5 HP (351 steps at beam 24, 259 at beam 40). Kill mode fought fillers forever; 150-px wins walked into them.
                    arena_escape = mission4 and int(info.get("level", 0)) in PLAN_ESCAPE
                    target_visible = any((e.get("heavy") or int(info.get("level", 0)) == 48 or fight_armed or stalled_fight or arena_escape
                                          or (any_enemy_here and abs(e["dx"]) <= PLAN_ANY_ENEMY_PX))
                                         and abs(e["dx"]) <= 120 and abs(e["dy"]) <= 40
                                         for e in step_radar.get("enemies", []))
                    if fight_armed and target_visible:
                        fight_last_seen = steps
                    if fight_armed and all(self.env.pyboy.get_memory_value(a) == 0 for a in planner.slots):
                        bruiser_hp_min = 0; fight_kills += 1                              # the target's HP byte hit zero (the sprite is gone, so the radar path never sees 0)
                        if int(info.get("level", 0)) == 48 and set(planner.slots) <= set(planner_slots):
                            boss1_dead = True                                              # v115: the real boss is down; what follows is survived, not fought
                    if fight_armed and (steps - fight_last_seen > 48 or self.env.pyboy.get_memory_value(0xC595) == 0 == self.env.pyboy.get_memory_value(0xC5DA)
                                        or all(self.env.pyboy.get_memory_value(a) == 0 for a in planner.slots)):
                        fight_armed = False                                                # he is dead or long gone
                    # v115: the mission-1 "second boss" cannot be killed (his bytes are an animation counter) and the room ends by
                    # itself ~150 steps after the first boss dies (tested: thrown around, HP 6 -> 2, then the mission-1 ending
                    # cut-scene with all lives). 21 of 38 v114 deaths were players fighting him. Survive instead: escape mode.
                    survive_48 = int(info.get("level", 0)) == 48 and self.env.pyboy.get_memory_value(0xC595) == 0 == self.env.pyboy.get_memory_value(0xC5DA) and boss1_dead
                    # v138: hideout corridor autopilot. Past room 5 the corridor (ids 96/80/81/64/65/68/24/26/25/11/12/9/58, SCY 0)
                    # descends a staircase with a rolling boulder whose sprite the radar cannot see (low tile ids); the v134
                    # hideout workers went 6 -> 1 HP there. A progress-only escape search (chunk flips do not count as wins)
                    # threads it: 45-48 steps, 5 HP kept, ~20 s per plan on the staircase fixtures.
                    CORRIDOR_IDS = {96, 80, 81, 64, 65, 68, 24, 26, 25, 11, 12, 9, 58} | ({6, 5} if hideout_entered else set())   # v140: the 6/5 stretch right after room 5 is corridor too (a v138 worker went 6 -> 1 HP there)
                    if (planner and not plan and mission4 and tracker.beyond and int(info.get("level", 0)) in CORRIDOR_IDS and not maze_entered   # v141: the maze past door 2 reuses 26/25/24 too
                            and self.env.pyboy.get_memory_value(0xFF42) < 112      # the corridor and the staircase; at SCY 112 (slabs, door-2 room) ids 80/81 recur and the hop hook owns the ground
                            and steps >= plan_block_until and plan_count < PLAN_MAX and step_radar.get("player")
                            and not any(e["reachable"] and abs(e["dx"]) <= 90 and abs(e["dy"]) <= 40 for e in step_radar.get("enemies", []))
                            and not (int(info.get("level", 0)) in (5, 6) and step_radar.get("enemies"))):   # v153g: rooms 6/5 are locked while their thugs live - three 68-s corridor searches per visit found nothing (corr5)
                        seq = planner.plan(int(info.get("level", 0)), river=False, progress_w=1.0, escape=True, escape_px=120, escape_depth=160,
                                           beam_width=24, escape_chunk_win=False, max_hp_loss=0)
                        if not seq:                                                        # v153f: a shorter damage-free hop first (corr2 states: 120 px failed where 48 px was found in 9 s)
                            seq = planner.plan(int(info.get("level", 0)), river=False, progress_w=1.0, escape=True, escape_px=48, escape_depth=160,
                                               beam_width=24, escape_chunk_win=False, max_hp_loss=0)
                        if not seq:                                                        # v142: no damage-free line from here - accept one tick rather than leave the model alone with the boulder
                            seq = planner.plan(int(info.get("level", 0)), river=False, progress_w=1.0, escape=True, escape_px=120, escape_depth=160,
                                               beam_width=24, escape_chunk_win=False, max_hp_loss=1)
                        if seq:
                            plan = list(seq); plan_count += 1
                        else:
                            plan_block_until = steps + 30
                    # v141: door-2 autopilot. The second hideout door (door-2 room, ids 80/81 at SCY 112) unlocks ~20-60 steps after
                    # the 52-HP bruiser dies and opens on ~30-40 held UP presses at screen x ~142; UP sooner does nothing, and nav
                    # at 1 HP finds no exit there (every macro that takes a hit is pruned). Scripted: walk to x 142, punch air
                    # 30 steps (a safe no-op on that ledge - DOWN drops off it, LEFT walks into the void), hold UP 40, punch 40.
                    if hideout_entered and int(info.get("level", 0)) in (80, 81) and self.env.pyboy.get_memory_value(0xFF42) == 112 \
                            and any(self.env.pyboy.get_memory_value(a) >= 40 for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2)):
                        door2_bruiser_seen = True                                      # v145: the room's 52-HP bruiser has activated
                    if (not plan and hideout_entered and int(info.get("level", 0)) in (80, 81) and self.env.pyboy.get_memory_value(0xFF42) == 112
                            and door2_bruiser_seen                                     # v145: v143 fired the script on arrival, before the pair/bruiser activated (all slots < 10), then blocked itself for 400 steps through the real fight - 4 workers, 0 shaft arrivals
                            and steps >= door2_block_until and step_radar.get("player") and step_radar["player"]["x"] >= 100
                            and all(self.env.pyboy.get_memory_value(a) < 10 for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2))):   # v142: the knocked-out pair's sprites lie there as "reachable enemies" for 100+ steps - the HP slots are the real test (v141: bruiser dead, autopilot never armed)
                        _dx = 142 - step_radar["player"]["x"]; _walk = ["right" if _dx > 0 else "left"] * min(14, abs(_dx) // 3)
                        plan = [self.action_names.index(a) for a in _walk + ["a"] * 30 + ["up"] * 40 + ["a"] * 40]
                        door2_block_until = steps + 400
                        print(f"door2-autopilot: {len(plan)} steps from x={step_radar['player']['x']}", flush=True)
                    # v144: the sealed room past the maze (chunk 7 at SCX >= 160): after its 52/35-HP heavies die, 20-HP thugs respawn while
                    # the player stands mid-room and the only exit is a door on the upper platform at x ~105 (climb from the floor with UP,
                    # then ~34 more UP presses -> chunk 97, HP refilled). Any thug on screen throws the player off the climb, so the script
                    # runs only with an empty radar; the fight planner / knockdowns clear the room first.
                    # v153aa: hideout door autopilot (door room, chunk 106 past the arena). Once the room is clear the back-wall door opens on
                    # ~27 held UP presses at x 72-89 (probe from fullgame11's step-7400 state) - the run stood at the right pillar with the
                    # hop hook firing at the edge pit instead. Closed-loop align to x 80, then hold UP 40.
                    hdoor_align = None
                    if (not plan and mission4 and tracker.beyond and not hideout_entered and int(info.get("level", 0)) == 106
                            and self.env.pyboy.get_memory_value(0xFF42) == 0 and steps >= hdoor_block_until and step_radar.get("player")
                            and not step_radar.get("enemies") and all(self.env.pyboy.get_memory_value(a) < 10 for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2))):
                        _dx = 80 - step_radar["player"]["x"]
                        if abs(_dx) > 3:
                            hdoor_align = "right" if _dx > 0 else "left"
                        else:
                            plan = [self.action_names.index("up")] * 40; plan_from_nav = False
                            hdoor_block_until = steps + 200
                            print(f"hideout-door: 40 UP from x={step_radar['player']['x']} feet={step_radar['player']['feet_y']}", flush=True)
                    room7_align = None
                    if (not plan and maze_entered and int(info.get("level", 0)) == 7 and self.env.pyboy.get_memory_value(0xFF43) >= 160
                            and self.env.pyboy.get_memory_value(0xFF42) == 0 and steps >= room7_block_until and step_radar.get("player")
                            and not step_radar.get("enemies") and all(self.env.pyboy.get_memory_value(a) < 10 for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2))):   # v153h: tested from fg4-room7-stuck.state - the door stays shut while the respawned thugs exist (even with their HP bytes zeroed), so the room must be cleared: see room7_approach
                        _dx = 105 - step_radar["player"]["x"]
                        if abs(_dx) > 2:
                            room7_align = "right" if _dx > 0 else "left"                   # v153i: closed-loop alignment - the old open-loop walk (|dx|//3 steps at ~2.25 px/step) left the player at x 112-125, where 48 UP presses do nothing (fullgame4 stood there 2600 steps; r7fix3 drifted to x 125). The door needs x 105 (room7.log: x 105, feet 113 -> 93 -> 75 -> chunk 97)
                        else:
                            plan = [self.action_names.index("up")] * 48; plan_from_nav = False
                            room7_block_until = steps + 120
                            print(f"room7-door: 48 UP from x={step_radar['player']['x']} feet={step_radar['player']['feet_y']}", flush=True)
                    if (planner and not plan and steps >= plan_block_until and plan_count < PLAN_MAX and not ending_steps   # v154f: no fight searches during the ending
                            and (int(info.get("level", 0)) in PLAN_FIGHTS or any_enemy_here or fight_armed or stalled_fight or arena_escape or survive_48)   # v153l: the second boss from first contact (fullgame8: 5 -> 3 while waiting for the stall trigger)
                            and step_radar.get("player") and (stalled_fight or not step_radar.get("player", {}).get("climbing")) and target_visible
                            and not (maze_entered and maze_narrow_ledge(step_radar))    # v143: on the ledge row's 16-px ledge every plan ends in a knockback fall - hop on instead
                            and not (maze_entered and ((steps < trap_block_until and not maze_skull_zones(self.env.pyboy)) or maze_under_trap_shaft(self.env.pyboy, step_radar["player"])
                                                       or maze_in_skull_zone(self.env.pyboy, step_radar["player"])))   # v150: on a skull screen the danger hook handles the window - no post-exit block
                            and not (maze_entered and boss_seen_here and maze_throne_room(self.env.pyboy)[0] and self.env.pyboy.get_memory_value(0xC595) == 0)
                            and not (maze_entered and boss_seen_here and boss_fail_here >= 2 and self.env.pyboy.get_memory_value(0xC595) >= 12 and step_radar["player"]["x"] > 16)   # v154g: reposition first
                            and not (maze_entered and not any(e["reachable"] and abs(e["dx"]) <= 96 for e in step_radar.get("enemies", [])))
                            and not (maze_entered and int(info.get("level", 0)) == 7 and self.env.pyboy.get_memory_value(0xFF43) >= 160 and step_radar.get("enemies")
                                     and (abs(min(step_radar["enemies"], key=lambda e: abs(e["dx"]))["dy"]) > 8
                                          or not (16 <= step_radar["player"]["x"] + min(step_radar["enemies"], key=lambda e: abs(e["dx"]))["dx"] <= 144)))):   # v153s; v153v: nor while it sits at a screen edge: sealed room - change lane first (room7_approach), the back-lane searches on the edge thug never find a kill   # v150: no search for a target that is not on screen (stale slot values)   # v150: bar 0 / Marian: the finish and ending scripts own the room   # v148/v149: walk clear of lethal floor first: under a ceiling shaft the autopilot walks out first (run 31: plan -> dropped -> re-plan loop for 3 steps in the kill zone)
                        if int(info.get("level", 0)) not in plan_saved_areas:                  # v109: one pre-plan state per area (boss-room re-plans ate the 8-state budget)
                            plan_saved_areas.add(int(info.get("level", 0)))
                            self._save_state(ep, steps, f"plan{plan_count}-a{int(info.get('level', 0))}", saved_states)   # v86: exact pre-plan state for offline diagnosis
                        _plan_kw = dict(river=in_forest,
                                           require_verified=((hideout_entered and not maze_entered and int(info.get("level", 0)) in (5, 6))
                                                             or (maze_entered and int(info.get("level", 0)) == 7 and self.env.pyboy.get_memory_value(0xFF43) >= 160)),   # v153e: hideout rooms 6/5; v153u: the sealed room's respawned pair read 20 at 0xC583 with the main slots at 0 and no kill was ever found - hit-verified slots only
                                           alt_layouts=bool(maze_entered),                 # v153j: the static ALT2/ALT3 slot layouts belong to the hall
                                           progress_w=(PLAN_PROGRESS_W if (any_enemy_here and mission4 and not tracker.beyond) else 0.0),   # v90/v94: only mission-4 bridge fights reward rightward progress (the mission-1 warehouse exit is LEFT); v134: not past the arena - in the door room the weight turned a 42-step kill (3 HP kept) into 60-step partials toward the right-edge pit
                                           simple=(not mission4 and (int(info.get("level", 0)) in PLAN_ANY_ENEMY_ALWAYS or int(info.get("level", 0)) == 84)),   # v95: warehouse thugs respawn - single kill, no clearance; v153x: chunk 84 always
                                           escape=survive_48 or arena_escape,   # v99/v120: gauntlet rooms - the win is the screen scrolling on (arena: 4 waves then exit to chunk 246)
                                           escape_px=(9999 if (survive_48 or arena_escape) else None),   # v115/v120: only the chunk change counts (150 px of progress was a false win into the thug stream)
                                           escape_depth=(240 if survive_48 else None),  # knockdown -> he leaves -> ending takes ~120 steps; failed 420-step searches cost 180 s each
                                           beam_width=(PLAN_BEAM_M4 if mission4 else None),   # v119: wider beam where every tick counts (bridge + arena)
                                           long_settle=(mission4 and tracker.beyond),      # v133/v135: ledge/slab landing checks only past the arena (on the bridge they stretched 32-HP plans to 11 min)
                                           time_s=(1200.0 if arena_escape else None))      # v139: the arena escape gets 20 min (v138: a 2-HP arrival hit the 600 s cap and got a partial)
                        seq = None
                        planner.safe_after = 16 if (in_forest and not mission4) else 12    # v153m; v153ae: 12 everywhere (6 missed the catwalk grab-throw three times): forest ledge - a kill only counts if 16 idle/back-away steps cost nothing (fullgame5 4115, fullgame8 4206: thrown into the river right after a planned kill)
                        _boss_fight = maze_entered and boss_seen_here and maze_boss(step_radar) is not None and self.env.pyboy.get_memory_value(0xC595) >= 12   # v150: the gunman's bar - a 600-s beam-40 search found a 117-step damage-free drain offline where 150 s/beam 24 found nothing
                        _small = maze_entered and max(self.env.pyboy.get_memory_value(a) for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2, 0xC583, 0xC5C8, 0xC588, 0xC5CD)) <= 12   # v147: hall thugs (10 HP) fly off after one hit - long searches never finish them (run 28: 4 x 2 min for nothing); the knockdown search handles them
                        if maze_entered and not (survive_48 or arena_escape):
                            # v146: past door 2 the runs die of HP attrition across the hall's fight-lock rooms, not of any single fight -
                            # look for a damage-free line first (2 min), then fall back to the normal search
                            if _boss_fight:
                                print(f"boss-fight: bar {self.env.pyboy.get_memory_value(0xC595)} hp {read_health(self.env.pyboy)} - long damage-free search (fails here so far {boss_fail_here})", flush=True)
                            seq = planner.plan(int(info.get("level", 0)), **{**_plan_kw, "max_hp_loss": (8 if _boss_fight else 0), "time_s": (900.0 if _boss_fight else (45.0 if maze_fail_streak == 0 else 120.0) if _small else 120.0), **({"beam_width": 80} if _boss_fight else {})})   # v154g: beam 80 (60 exhausted at x 62; 80 found a line)   # v150: the gunman - beam 60 / 900 s / 1-tick allowance found a 114-step drain from the live 6-HP arrival where beam 40 exhausted its candidates   # v150: a failed 45-s search on a hall pair escalates to 120 s next time (run 57q: 20 failures in a row in lock room 2)
                        elif PLAN_M3_ALL and tracker.mission() >= 3 and not survive_48:
                            # v153: missions 3+4 share one HP budget (no refill before the hideout) - damage-free pass first here too:
                            # cave2/cave3 replays cleared the chunk-46 bruiser at 0 loss but paid 1-2 ticks in the arena and the door room
                            seq = planner.plan(int(info.get("level", 0)), **{**_plan_kw, "max_hp_loss": 0, "time_s": (600.0 if arena_escape else 120.0)})
                        if not seq:
                            _room7_heavy = maze_entered and int(info.get("level", 0)) == 7 and max(self.env.pyboy.get_memory_value(a) for a in (0xC595, 0xC5DA)) >= 40
                            if _room7_heavy and not seq:
                                seq = planner.plan(int(info.get("level", 0)), **{**_plan_kw, "time_s": 300.0, "beam_width": 40, "max_hp_loss": 8})   # v153q: the sealed room's 52-HP heavies - four 90-128 s searches at beam 24 found nothing (probe_lane)
                            if not seq: seq = planner.plan(int(info.get("level", 0)), **({**_plan_kw, "time_s": 45.0} if _small else {**_plan_kw, "time_s": 600.0, "beam_width": 60, "max_hp_loss": 16} if _boss_fight else _plan_kw))
                        if maze_entered:
                            maze_fail_streak = 0 if seq else maze_fail_streak + 1
                        if maze_entered and seq and maze_skull_zones(self.env.pyboy):
                            _seq2 = maze_plan_with_exit(self.env, self.action_names, seq)
                            if _seq2 is None:
                                print(f"maze: plan of {len(seq)} steps discarded - the player dies within 24 idle steps after it (skull)", flush=True)
                            seq = _seq2
                        if seq:
                            plan = list(seq); plan_count += 1; fight_armed = True; fight_last_seen = steps; fight_fail_streak = 0; plan_from_nav = False
                            if _boss_fight: boss_fail_here = 0
                        else:
                            fight_fail_streak = (fight_fail_streak + 1) if fight_fail_chunk == int(info.get("level", 0)) else 1
                            if _boss_fight: boss_fail_here += 1                                    # v154g
                            fight_fail_chunk = int(info.get("level", 0))
                        if seq:
                            pass
                        elif any(e["reachable"] and abs(e["dx"]) <= 40 and abs(e["dy"]) <= (8 if int(info.get("level", 0)) == 7 else 24) for e in step_radar.get("enemies", [])):   # v153z: every mission - f12 trace: a planner-blind thug (no HP slot found) stood 9-16 px away for 13 silent steps while the model punched with a pit 30 px behind, then threw the player into it (fullgame11 step 4552); the knockdown search prunes falls   # v153q: sealed room - no knockdowns across lanes (the 4-step loop)   # v153l: one landed knockdown sends the second boss off (dd-room-mechanics) - the escape search keeps failing there   # v153c: mission 4 too - a door-room thug the slot discovery rejects otherwise punches freely for 60-step blocks
                            if int(info.get("level", 0)) not in kd_saved_areas:
                                kd_saved_areas.add(int(info.get("level", 0))); self._save_state(ep, steps, f"kd-a{int(info.get('level', 0))}-x{self.env.pyboy.get_memory_value(0xFF43)}", saved_states)   # v145: the state of an enemy the planner cannot see (one per area)
                            _kd = maze_knockdown_plan(self.env, self.action_names, step_radar["player"]["feet_y"], depth=36, beam=10, time_s=45.0)   # v143d: no HP byte found - one landed hit sends this enemy off for good; v147: deeper search (the hall's lock-room pairs flicker between slot layouts and the fight planner rarely locks on)
                            if _kd:
                                plan = list(_kd); plan_from_nav = False
                            plan_block_until = steps + (len(_kd) if _kd else 4)                 # v153ac: a failed knockdown search usually had a bad root (airborne / mid-throw) - retry in 4 steps, not 12
                            if not _kd:
                                blind_hold_until = steps + 12                                      # v153ac: and hold a punching stance meanwhile (ladder1: the model walked into a throw off the catwalk end)
                        elif max(self.env.pyboy.get_memory_value(a) for a in planner_slots) == 0:
                            plan_block_until = steps + (24 if maze_entered else 6)         # v102: enemy on radar but its HP slot not yet live (boss walking in): retry soon; v150: in the hall a failed 45-s search every few steps crawled through lock room 2 (run 57q: 11 searches for 20 steps)
                        else:
                            plan_block_until = steps + 60                                  # do not burn another budget right away
                    area_hist.append((steps, int(info.get("level", 0))))
                    # v109: the chunk ids come in a fixed order (identical across 40 independent v85 workers), so "the
                    # index has not advanced for 600 steps" is the stall test; the v108 re-entry test fired falsely in
                    # every repeated stretch (26,66,25,106 x4; 16,17 x8; 11,12 x50) and searched for unreachable ids.
                    if scroll.distance_px > nav_best_px:                  # v113: horizontal only - vertical_px counts both directions, so ladder bouncing never looked stalled
                        nav_best_px, nav_gain_step = scroll.distance_px, steps
                    stalled = tracker.pos >= 0 and steps - max(tracker.pos_changed_step, nav_gain_step) >= NAV_STALL_STEPS
                    regressed = steps < nav_guard_until and tracker.pos < nav_exit_pos
                    # v143d: in the maze no nav plan next to a live thug - a 128-step "exit" planned beside one (HP in an unknown slot) was replayed blindly while he threw the player off the top ledge
                    # v153n: nav never runs with a reachable enemy within 60 px (everywhere, not just the maze) - fullgame8 step 4674: a stall with two thugs at 2-12 px ran nav, whose 28-step walk was thrown off the forest ledge (third forest death today); nav_force (planner blind to the enemy) still passes
                    nav_force = fight_fail_streak >= 2 and fight_fail_chunk == int(info.get("level", 0)) and steps >= nav_block_until   # v115: the fight planner cannot see this enemy's HP (mission-1 second boss): leave the room instead
                    m2_navfirst = ((tracker.mission() == 2 or NAV_FIRST_ALL) and not tracker.beyond and not hideout_entered and not maze_entered and not plan and steps >= nav_block_until   # v154i: DD_NAV_FIRST=1 extends navigation-first to missions 1-4 before the arena (fall-checked, HP-strict walks instead of the model's free steps - fullgame17 lost two lives to unplanned walks in missions 1-2)   # v154h: fullgame16 - after a slab-row rewind the tracker regressed to mission 2's 5/6 and nav-first fired 220 "no exit" searches at the slab row's left edge
                                   and not any(e["reachable"] and abs(e["dx"]) <= 60 and abs(e["dy"]) <= 40 for e in step_radar.get("enemies", [])))   # v154b: mission 2's ladder/conveyor climb is nav-driven - fall-checked nav plans walk it, the model only fills gaps (rewind1: five deaths in 26/66/25/106 on the model's own steps)
                    if nav and not plan and not fight_armed and nav_count < 400 and step_radar.get("player") \
                            and not (any(e["reachable"] and abs(e["dx"]) <= 60 and abs(e["dy"]) <= 40 for e in step_radar.get("enemies", [])) and not nav_force) \
                            and (((no_progress_run >= NAV_STALL_DECISIONS or stalled) and steps >= nav_block_until or regressed
                                  and not any(e["reachable"] and abs(e["dx"]) <= 60 for e in step_radar.get("enemies", []))) or nav_force or m2_navfirst):
                        recent_areas = {a for s_, a in area_hist if steps - s_ <= NAV_WINDOW_STEPS} | nav_recent
                        if tracker.mission() == 3 and not tracker.beyond:
                            recent_areas |= {11, 12, 49}                                  # v153y: the forest's ids flip 11/12 every few px - a 12-step "exit" from a 2-state search was the last plan before the river deaths of fullgame9 (x2) and fullgame11 (step 4552); only real progress counts here
                        if tracker.beyond:
                            recent_areas |= {246, 244, 245, 234, 233, 84, 85, 86, 106}   # v133: the cave reuses these ids every few px - a flip is not an exit (4-step 'exits' burned the nav budget and a partial walked into the void)
                        if hideout_entered:
                            recent_areas |= {6, 5, 96, 80, 81, 64, 65, 68, 24, 26, 25, 11, 12, 9, 58, 89, 90, 91}   # v138: inside the hideout the ids flip every ~8 px too; only a door (a new id such as 73) is an exit
                        if maze_entered:
                            recent_areas |= {73, 72, 56, 57, 27, 26, 30, 31, 106, 88, 89, 63, 28, 29, 7, 97, 96}   # v141: past door 2 the shaft/ledge ids flip as well - climb/walk partials, not 12-step 'exits'; v143d: + top-ledge/step-down/island/new-area ids
                        self._save_state(ep, steps, f"nav{nav_count}-a{int(info.get('level', 0))}", saved_states)
                        nav.chain = False
                        _goals = None if (tracker.mission() == 3 and not tracker.beyond) else ((tracker.goal_areas() or set()) - {a_ for s_, a_ in area_hist if steps - s_ <= 120})   # v154j: a goal id seen in the last 120 steps is a flip, not an exit (fullgame17 step 2522); v154j-fix: goal_areas() is None past the arena - guard it (the raw None - set crashed fullgame19/21 at the cave walkway)
                        seq = nav.plan(recent_areas, river=in_forest, goal_areas=(_goals or None),   # v153y2: in the forest a goal-area hit is just the 11/12 flip (nav_planner line ~154) - novelty mode with 11/12/49 marked recent needs real progress
                                       max_hp_loss=(0 if (hideout_entered and not maze_entered) else None))   # v153g: corr4 - a 137-step nav "exit" through the stalactite corridor cost 7 -> 2
                        if seq:
                            _full = len(seq)
                            seq = plan_hp_prefix(self.env, seq)                     # v153g: emulator-verified - keep only the prefix before the first HP drop
                            if seq is None or len(seq) < _full:
                                nav_block_until = max(nav_block_until, steps + 300)    # v153w: the way on is guarded - leave the next 300 steps to the fight planner and the reflexes, not another nav walk
                        nav_count += 1; nav_block_until = steps + (24 if (tracker.mission() == 2 or (NAV_FIRST_ALL and not tracker.beyond)) else 300); no_progress_run = 0; tracker.pos_changed_step = steps; nav_gain_step = steps; fight_fail_streak = 0   # v154b: mission 2 re-plans as soon as a plan ends (+24 after a miss)
                        if seq:
                            plan = list(seq); nav_recent = set(recent_areas); plan_from_nav = True
                            nav_exit_pos = tracker.pos + 1 if nav.last_was_exit else -1      # v111: a partial walk is not an exit - do not re-plan the moment it ends
                            nav_guard_until = steps + len(seq) + 600
                        else:
                            nav_guard_until = -1                              # no way out from here: let the model try
                    if (int(info.get("level", 0)) == 46 or fight_armed) and any(e.get("heavy") for e in step_radar.get("enemies", [])):
                        bhp = self.env.pyboy.get_memory_value(0xC595)        # measured: jab -2, kick -3, knockdown blow / elbow -3..-5
                        step_radar["bruiser_hp"] = bhp
                        bruiser_hp_min = bhp if bruiser_hp_min is None else min(bruiser_hp_min, bhp)
                        bruiser_hp_seen_step = steps
                    belt.update(step_radar, self.env.pyboy.get_memory_value(0xFF43), exec_name, step_radar.get("health_ticks", -1),
                                belt_tile_under(self.env.pyboy, step_radar))
                    # v57: fights need a decision every step. The scripted aligned policy clears the bruiser 4/4 with one
                    # action per step and 1/4 when each action is held for two (boss: 3/4 -> 0/4): the action-repeat is
                    # what makes the fights unwinnable, so it is suspended while an enemy is within 40 px.
                    # v61: the per-step fight decisions (v57-v60) took the mission-1 boss from 4/7 (v56) to 0/16 and
                    # 0/8; with interval 2 a chosen punch is held for two steps, and that rhythm is what wins there.
                    # The heavy autopilot acts every env step regardless, so the fights it owns lose nothing.
                    fight = FIGHT_PER_STEP and any(e["reachable"] and abs(e["dx"]) <= 40 for e in step_radar.get("enemies", []))
                    if d is None or (not plan and (steps % self.decision_interval == 0 or fight)):   # v82: no model calls while a plan replays
                        native = self.env.screen.screen_ndarray() if self.include_native else None
                        radar = step_radar
                        last_dir = next((a for a in reversed(recent) if a in ("left", "right")), None)
                        # stuck = same direction pressed for the last 6 decisions while the player did not move
                        stuck_dir = None
                        if radar.get("player"):
                            pos_hist.append((radar["player"]["x"], radar["player"]["feet_y"], scroll.distance_px, scroll.vertical_px))
                            if len(pos_hist) > 12:
                                pos_hist.pop(0)
                            if len(pos_hist) >= 12 and len(recent) >= 12:
                                dx_move = abs(pos_hist[-1][0] - pos_hist[0][0])
                                dscroll = pos_hist[-1][2] - pos_hist[0][2]
                                # v80: on a ladder through a scrolling section the sprite parks at a fixed screen position
                                # while the screen scrolls vertically (SCY 196 -> 168 over 24 "up" presses, mission-2
                                # area 16). v77 flagged that as stuck-up, told the model "do NOT press up again", and
                                # 3 of 5 true-start runs stalled there until the step cap. Vertical motion counts too.
                                dy_move = abs(pos_hist[-1][1] - pos_hist[0][1])
                                dvert = pos_hist[-1][3] - pos_hist[0][3]
                                counts = {d: recent[-12:].count(d) for d in ("up", "down", "left", "right")}
                                dominant = max(counts, key=counts.get)
                                # pressed one direction for at least half of the last 12 decisions, yet neither the
                                # player nor the screen moved horizontally => blocked (doorway, wall, edge)
                                if counts[dominant] >= 6 and dx_move <= 4 and dscroll == 0 and dy_move <= 4 and dvert == 0 and not [e for e in radar["enemies"] if e["reachable"]]:
                                    stuck_dir = dominant
                        if radar.get("health_ticks", -1) >= 0:
                            min_health = min(min_health, radar["health_ticks"])
                        stuck_run = stuck_run + 1 if stuck_dir else 0
                        sig_now = (scroll.distance_px, scroll.vertical_px, int(info.get("score", 0)))
                        progress_age = progress_age + 1 if sig_now == progress_sig else 0
                        progress_sig = sig_now
                        if progress_age == 200 and steps - noprog_saved_at > 1000:   # frozen for 200 decisions: snapshot it
                            self._save_state(ep, steps, "no-progress", saved_states)
                            noprog_saved_at = steps
                            # sidecar: what the model was doing and being told, so the stall can be diagnosed without guessing
                            try:
                                self._put(f"states/w{self.idx:03d}_ep{ep:02d}_step{steps:05d}_no-progress.json", json.dumps({
                                    "recent_inputs": recent[-8:], "action_hist_last_200": dict(noprog_hist),
                                    "last_advice": last_advice, "player": radar.get("player"), "pits": radar.get("pits"),
                                    "ladders": [{k: l[k] for k in ("x", "on_it", "at_top", "climbable")} for l in radar.get("ladders", [])],
                                    "enemies": [(e["dx"], e["dy"]) for e in radar["enemies"]], "conveyor": belt.direction,
                                }).encode(), "application/json")
                            except Exception as exc:
                                log.warning("worker %d: no-progress sidecar failed: %s", self.idx, exc)
                        if progress_age == 0:
                            noprog_hist = {}
                        if stuck_run == 15:            # stuck for 15 decisions: snapshot the situation once per streak
                            self._save_state(ep, steps, f"stuck-{stuck_dir}", saved_states)
                        radar["airborne"] = bool(belt.airborne)      # from the per-step tracker: feet moved >2px since last step
                        radar_for_model = dict(radar)
                        reach_now = [e for e in radar["enemies"] if e["reachable"]]
                        if reach_now:
                            last_enemy = {"side": reach_now[0]["side"], "dx": reach_now[0]["dx"], "step": steps}
                        recent_enemy = ({"steps_ago": steps - last_enemy["step"], "side": last_enemy["side"]}
                                        if last_enemy else None)
                        no_progress = (len(pos_hist) >= 12 and pos_hist[-1][2] - pos_hist[0][2] == 0 and not reach_now
                                       and not (recent_enemy and recent_enemy["steps_ago"] <= 8))
                        no_progress_run = (no_progress_run + 1) if (no_progress and (len(pos_hist) < 12 or pos_hist[-1][3] - pos_hist[0][3] == 0)) else 0   # v108: vertical motion counts as progress too
                        pinned = bool(stuck_dir) or no_progress      # position alone is not evidence: the camera parks x~120 while scrolling
                        # attacks that never move the score (enemy across a gap / at the screen edge / on another plane)
                        futile_attacks = (bool(reach_now) and sum(1 for a in recent[-8:] if a in ("a", "b", "jump_kick", "kick_left", "kick_right")) >= 5
                                          and steps - last_gain_step >= 24)
                        # passive enemies: same absolute positions and no health loss for 8 decisions => decor / inactive
                        enemy_hist.append((frozenset((e["x"], e["feet_y"]) for e in radar["enemies"]), radar.get("health_ticks", -1)))
                        if len(enemy_hist) > 8:
                            enemy_hist.pop(0)
                        if radar.get("on_ladder"):
                            enemy_hist.clear()          # enemies never engage a climber: standing still here is not passivity
                        enemies_passive = (len(enemy_hist) == 8 and bool(reach_now) and len({h[0] for h in enemy_hist}) == 1
                                           and len({h[1] for h in enemy_hist}) == 1 and progress_age >= 8)
                        if radar.get("doors") and pinned and not reach_now:
                            door_mode = True                    # exit identified: stay in door mode until the scene changes
                        if not radar.get("doors"):
                            door_mode = False
                        if os.environ.get("DGEMMA_DOOR_ADVICE", "0") != "1" and not (pinned or door_mode):
                            radar_for_model.pop("doors", None)   # only show doors when the exit matters
                        extra = {
                            "radar": radar_for_model,
                            "radar_advice": (belt_note(belt.direction, belt.speed, radar) if (radar.get("conveyor_zone") and not any(e.get("heavy") for e in radar["enemies"])) else "") + suggested_rule(
                                radar, last_dir, stuck_dir, door_mode, recent_enemy, int(info.get("level", 0)), enemies_passive, belt.direction,
                                travel_dir, futile_attacks,
                                boss_hits=(boss_hits if (int(info.get("level", 0)) == 48 or any(e.get("heavy") for e in radar["enemies"])) else None)),
                            "enemies_passive": enemies_passive,
                            "conveyor": ({"direction": belt.direction, "speed_px_per_step": round(belt.speed, 1)} if belt.direction else None),
                            "area_id": int(info.get("level", 0)),
                            "recent_enemy": recent_enemy,
                            "stuck_warning": (f"pressing {stuck_dir} repeatedly is not moving you" if stuck_dir else None),
                            "stage_progress_px": scroll.distance_px,
                            "vertical_progress_px": scroll.vertical_px,
                            "recent_inputs_oldest_first": recent[-8:],
                            "same_input_streak": streak,
                            "steps_since_score_rose": steps - last_gain_step,
                            "lives_lost_this_episode": lives_lost,
                        }
                        # stuck: let the model think for a step (server-side reasoning) instead of the reflex answer
                        overrides = {"think": int(os.environ.get("DGEMMA_THINK_WHEN_STUCK", "1"))} if (stuck_dir and os.environ.get("DGEMMA_THINK_WHEN_STUCK", "1") != "0") else None
                        d = self.client.choose_action(obs, info, native, prev_name, extra, overrides)
                        fresh = True
                        last_advice = extra["radar_advice"]
                        # remember a direction decision so positional rules cannot flip it one step later (v24/v27/v28
                        # limit cycles); forget it once the screen has actually scrolled 48px, i.e. we are somewhere new
                        scy_now = self.env.pyboy.get_memory_value(0xFF42)
                        if radar.get("scrolled_vertically"):
                            if travel_dir and travel_scy != scy_now:
                                travel_dir = None                       # new floor: decide afresh
                            if "the way on is LEFT" in last_advice or "continues to the left: walk left" in last_advice:
                                travel_dir, travel_scy = "left", scy_now
                            elif "continues to the right: walk right" in last_advice:
                                travel_dir, travel_scy = "right", scy_now
                        else:
                            travel_dir = None
                        noprog_hist[d.name] = noprog_hist.get(d.name, 0) + 1
                    else:
                        fresh = False   # repeat the last decision (action-repeat, cuts model traffic by 1/interval)
                    # frame-level reflex: the conveyor gap jump has an 8px window the model cannot time at 1-2 steps/decision
                    auto = stones_autopilot(step_radar, d.name)
                    intent_src = ["right", "right", "right"] if auto else recent
                    hv = [e for e in step_radar.get("enemies", []) if e.get("heavy") or int(info.get("level", 0)) == 48]
                    if hv:
                        if last_heavy is None:
                            last_heavy = {}
                        last_heavy.update(side=min(hv, key=lambda e: abs(e["dx"]))["side"], steps_ago=0, seen=last_heavy.get("seen", 0) + 1)   # in place: keeps just_hit / combo
                    elif last_heavy:
                        last_heavy["steps_ago"] += 1
                        if last_heavy["steps_ago"] > 16:
                            last_heavy["seen"] = 0           # the knockdown gap is <= 16 steps; longer means he is gone
                    if last_heavy is None and int(info.get("level", 0)) == 48:
                        last_heavy = {"side": "left", "steps_ago": 99}
                    on_slab = bool(step_radar.get("slabs_visible"))             # v129: the hideout's moving stepping stones (slab sprites at feet ~90) - timed launches whenever slabs are in view
                    # v154k: island escape (any mission). fullgame18 stood 4000 steps on a one-tile lift between two gaps on a mission-2 belt
                    # (chunk 66, SCY 152): every nav macro died instantly (states 5), walking either way fell, idling was safe. From that
                    # state a running kick LEFT lands on the catwalk (x 83, feet 76). With a pit edge within 20 px on BOTH sides and no
                    # plan, try verified hops right then left.
                    if (not plan and not maze_entered and steps >= hop_block_until and tracker.mission() >= 2 and step_radar.get("player") and not step_radar["player"].get("climbing")   # v154k4: mission 2+ only - mission 1's street sewer gratings read as two-sided gaps but are not real drops (fullgame22/23 hopped 5x on the street); v154p: never in the maze/hall - the tile-354 holes read as two-sided gaps and island-escape ping-ponged with stalled-hop between x=52 and x=107 forever (fullgamevid4 stuck 23 min in chunk 96/97); the hall has its own traverse_hall + lane logic
                            and len(pos_hist) >= 12 and abs(pos_hist[-1][0] - pos_hist[0][0]) <= 4 and abs(pos_hist[-1][2] - pos_hist[0][2]) <= 4   # v154k3: only when the player has not moved for 12 decisions (the canonical-position stall was trivially true in the long first chunk - fullgame21 hopped at x 11 and 45 in mission 1 again)
                            and any(g["side"] == "left" and -20 <= g["dx_edge"] <= 4 for g in step_radar.get("pits", []))
                            and any(g["side"] == "right" and -4 <= g["dx_edge"] <= 20 for g in step_radar.get("pits", []))
                            and not any(e["reachable"] and abs(e["dx"]) <= 48 for e in step_radar.get("enemies", []))):
                        _isl = hop.plan(timed=False, direction="right") or hop.plan(timed=False, direction="left")
                        hop_block_until = steps + (len(_isl) if _isl else 48)
                        if _isl:
                            plan = list(_isl); plan_from_nav = False
                            print(f"island-escape: hop of {len(_isl)} steps from x={step_radar['player']['x']}", flush=True)
                            self._save_state(ep, steps, f"island-a{int(info.get('level', 0))}", saved_states)
                    # v154o: stalled-hop fallback. fullgame-vid3 stalled 500+ steps at the hideout slab row (chunk 6, SCY 112) at the
                    # far-left (x 16) with the gaps 90+ px right: the hop hook needs a gap within 40 px, nav loops "no exit", and the
                    # walk-and-hop never fires. When mission 4 past the arena, stalled 12 decisions, feet on the floor, a right-side
                    # gap anywhere on screen and no close thug - let the hop planner walk-and-hop (it found a 30-step crossing offline).
                    if (not plan and (mission4 and tracker.beyond or hideout_entered or maze_entered) and steps >= hop_block_until and step_radar.get("player") and not step_radar["player"].get("climbing")   # v154o2: latches, not tracker.beyond (a hideout respawn regresses the canonical position and flipped beyond False, the live vid3 stall); v154p: stalled-hop STAYS enabled in the maze - the hall's tile-354 holes must be hopped rightward and this is the only maze hop mechanism (the line-1286 hook is gated not-maze); it was island-escape hopping back LEFT that created the fullgamevid4 ping-pong, so only island-escape is maze-suppressed (above)
                            and len(pos_hist) >= 12 and abs(pos_hist[-1][0] - pos_hist[0][0]) <= 4 and abs(pos_hist[-1][2] - pos_hist[0][2]) <= 4
                            and any(g["side"] == "right" and (g["right"] - g["left"]) >= 12 for g in step_radar.get("pits", []))
                            and not any(e["reachable"] and abs(e["dx"]) <= 48 for e in step_radar.get("enemies", []))):
                        _sh = hop.plan(timed=bool(step_radar.get("slabs_visible")), direction="right") or hop.plan(timed=True, direction="right")
                        hop_block_until = steps + (len(_sh) if _sh else 48)
                        if _sh:
                            plan = list(_sh); plan_from_nav = False
                            print(f"stalled-hop: {len(_sh)} steps from x={step_radar['player']['x']}", flush=True)
                            self._save_state(ep, steps, f"stalledhop-a{int(info.get('level', 0))}", saved_states)
                    if (mission4 and not plan and not maze_entered and steps >= hop_block_until and step_radar.get("player") and (step_radar["player"]["feet_y"] <= (95 if hideout_entered else 80) or on_slab)   # v138: hideout ledges stand at feet ~89; v143: the maze autopilot owns hops there
                            and not step_radar["player"].get("climbing")
                            and any(g["side"] == "right" and g.get("right", 0) - g.get("left", 0) <= (64 if on_slab else 28) and -4 <= g["dx_edge"] <= 40 for g in step_radar.get("pits", []))):
                        hop_seq = hop.plan(timed=on_slab)                      # v123: try the hop on the emulator; replay only a launch that lands; v129: on slabs also vary the launch time
                        hop_block_until = steps + (len(hop_seq) if hop_seq else 24)   # v133: a landed hop re-arms the hook as soon as its replay ends (the 24-step block walked the player into the next gap 17 px away - the same death at scx 105 in every local run)
                        if hop_seq:
                            plan = list(hop_seq)
                            self._save_state(ep, steps, f"hop{hop.plans}-a{int(info.get('level', 0))}", saved_states)   # v143: the launch state of every replayed hop (the ledge-row fall was invisible without it)
                    left_edge = (mission4 and d.name == "left" and step_radar.get("player") and step_radar["player"]["feet_y"] <= 80
                                 and any(g["side"] == "left" and g["dx_edge"] >= -10 for g in step_radar.get("pits", [])))   # v123: upper-walkway pillar edge behind the player
                    # v133: upper-walkway autopilot. Offline the planners alone cross the cave (climb, two verified hops, door room) in
                    # under a minute; live, v128's workers died or idled 4000 steps there because the model's own kicks and lefts on the
                    # pillars go uncontrolled between hops. On the upper walkway (past the arena, feet <= 80, no enemy within 60 px)
                    # the movement is "right" and the hop hook owns the gaps.
                    walkway_auto = (mission4 and tracker.beyond and not maze_entered and step_radar.get("player") and step_radar["player"]["feet_y"] <= (95 if hideout_entered else 80)
                                    and not step_radar["player"].get("climbing") and int(info.get("level", 0)) != 106
                                    and not any(e["reachable"] and abs(e["dx"]) <= 60 and abs(e["dy"]) <= 40 for e in step_radar.get("enemies", [])))
                    if walkway_auto and steps < hop_block_until and any(g["side"] == "right" and -4 <= g["dx_edge"] <= 14 for g in step_radar.get("pits", [])):
                        walkway_auto = "down"                                   # v133: at a gap edge with the hop hook still blocked: wait, do not step off (executed as a punch - DOWN drops off the hideout ledges)
                    # v141: door-2 room guard - on that ledge DOWN drops the player off and LEFT walks into the void; with no live enemy
                    # the model's only useful moves are right/up/punch (the local test died 15 steps after a nav plan ended there)
                    door2_guard = (hideout_entered and int(info.get("level", 0)) in (80, 81) and self.env.pyboy.get_memory_value(0xFF42) == 112
                                   and d.name in ("down", "left", "kick_left", "jump_kick")
                                   and all(self.env.pyboy.get_memory_value(a) < 10 for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2)))
                    # v141: shaft autopilot past door 2 (chunk 73/72, SCY >= 20): the start ledge has spikes below and the only way on is
                    # the climbing column at its right end - walk right until the ledge ends, then hold UP (a local test lost three
                    # lives in 84 steps to the model's own moves there)
                    shaft_auto = None
                    if maze_entered and plan and plan_from_nav and step_radar.get("player") and maze_prev_feet is not None \
                            and (read_health(self.env.pyboy) < maze_prev_hp or abs(step_radar["player"]["feet_y"] - maze_prev_feet) > 12):
                        plan = []                                                  # v143d: hit or thrown during a nav replay: the rest of the sequence is meaningless (and walked into the collapsing floor)
                        print(f"maze: nav replay aborted at step {steps} (hit/thrown)", flush=True)
                    if maze_entered and step_radar.get("player") and plan and plan_from_nav and maze_in_skull_zone(self.env.pyboy, step_radar["player"]):
                        # v150: only NAV replays are dropped in a skull zone - fight/knockdown plans were simulated damage-free through it
                        # (run 48: a verified 65-step kill was cut at its zone edge and the leftover thugs killed the player)
                        plan = []                                                          # v148: a replayed plan parked the player under a ceiling shaft - drop it, the autopilot walks on
                        trap_block_until = steps + 16                                      # v148b: and keep the planners off for 16 steps (run 32: plan -> one step out of the zone -> re-plan -> back in, 40 s per loop)
                        print(f"maze: plan dropped under a ceiling shaft at step {steps}", flush=True)
                    if maze_entered and step_radar.get("player") and not plan:
                        # v143: maze autopilot (replaces v141's shaft-only rule; see maze_dir). Priority: enemy close -> model/fight
                        # planner; climbing or a column overhead -> UP; edge ahead or a deadly look-ahead -> verified hop (level, then
                        # lower landing), else stand (punch); otherwise walk the region's direction.
                        _pl = step_radar["player"]; _scy = self.env.pyboy.get_memory_value(0xFF42)
                        _dir = maze_dir(_scy, _pl["feet_y"])
                        _sl = {a: self.env.pyboy.get_memory_value(a) for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2)}
                        _enemy_close = (any(e["reachable"] and abs(e["dx"]) <= 60 and abs(e["dy"]) <= 40 for e in step_radar.get("enemies", []))
                                        and (any(self.env.pyboy.get_memory_value(a) >= 10 for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2, 0xC574))
                                             or any(1 <= _sl[a] <= 9 and _sl[a] == maze_prev_slots.get(a, -1) for a in _sl)))   # v143c: knocked-out sprites (slot = a timer counting down) stalled the autopilot on the top ledge; v144: a STATIC 1-9 is a damaged thug's HP (runs 20/21: a 9-HP thug threw the autopilot onto a lethal spot twice)
                        _col = maze_column_above(self.env.pyboy, _pl)
                        _zones = maze_skull_zones(self.env.pyboy)
                        _in_zone = maze_in_skull_zone(self.env.pyboy, _pl, _zones)
                        _live_here = any(self.env.pyboy.get_memory_value(a) >= 10 or (1 <= self.env.pyboy.get_memory_value(a) <= 9 and self.env.pyboy.get_memory_value(a) == maze_prev_slots.get(a, -1))
                                         for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2, 0xC583, 0xC5C8, 0xC588, 0xC5CD))   # v150: a damaged thug (static 1-9) is still alive and grabbing (run 48: 8 and 2 HP read as "nobody here")
                        if maze_skull_zones(self.env.pyboy) and any(e["reachable"] and abs(e["dx"]) <= 60 and abs(e["dy"]) <= 24 for e in step_radar.get("enemies", [])):
                            _live_here = True                                              # run 49: a skull-room thug with no tracked HP byte walked the player into the window
                        _throne, _door_open = maze_throne_room(self.env.pyboy)
                        _boss = maze_boss(step_radar) if _throne else None; _bar = self.env.pyboy.get_memory_value(0xC595)   # run 40: two overlapping skull-room thugs read as a 24-px sprite
                        if _boss is not None and not boss_seen_here and not (_bar >= 56 and self.env.pyboy.get_memory_value(0xC5DA) >= 56):
                            _boss = None                                                   # run 58: lock room 3's 42-HP bruisers are 24 px wide too - the gunman is the one arriving with a 64 bar in both main slots
                        if _boss is not None:
                            if not boss_seen_here:
                                print(f"boss-room: the gunman is on screen at step {steps} (bar {_bar}, hp {read_health(self.env.pyboy)}, lives {self.env.pyboy.get_memory_value(0xC499)})", flush=True)
                                self._save_state(ep, steps, f"boss-room-a{int(info.get('level', 0))}", saved_states)
                            boss_seen_here = True; boss_gone_steps = 0
                        elif boss_seen_here:
                            boss_gone_steps += 1
                        _marian = _throne and any(64 <= e["x"] <= 96 and e["feet_y"] <= 60 for e in step_radar.get("enemies", []))   # a sprite standing IN the throne doorway, above floor level
                        if _throne and _boss is None and ((boss_seen_here and boss_gone_steps >= 20) or _marian) \
                                and max(self.env.pyboy.get_memory_value(a) for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2)) < 10:   # the door tiles read closed again once he is out, and the hall's ALT2 slots keep a stale 10/10 here
                            # v150: the ending - Marian waits in the throne doorway (x 64-95); walk to x 80 and hold UP (offline: the
                            # player climbs into the doorway, the screen fades, STAFF credits, THE END)
                            if ending_steps == 0:
                                print(f"ending: the gunman is gone at step {steps} - walking to the doorway", flush=True)
                            ending_steps += 1
                            shaft_auto = ("left" if _pl["x"] > 84 else "right") if (abs(_pl["x"] - 80) > 4 and _pl["feet_y"] >= 66) else "up"
                            plan = []
                        elif _boss is not None and _bar == 0 and _bar == self.env.pyboy.get_memory_value(0xC5DA):
                            # v150: bar empty - one STANDING hit sends him off (jump kicks do not); search for it, else walk in and punch
                            if steps >= finish_block_until:
                                _seq = boss_finish_plan(self.env, self.action_names, time_s=30.0)
                                if _seq:
                                    print(f"boss-finish: {len(_seq)}-step plan at step {steps}", flush=True)
                                    plan = list(_seq); plan_from_nav = False
                                finish_block_until = steps + (len(_seq) if _seq else 6)
                            if not plan:
                                shaft_auto = ("b" if abs(_boss["dx"]) <= 26 and abs(_boss["dy"]) <= 10
                                              else (("down" if _boss["dy"] > 0 else "up") if abs(_boss["dy"]) > 10 and 66 <= _pl["feet_y"] <= 94 else ("left" if _boss["dx"] < 0 else "right")))
                        elif _in_zone and not _pl.get("climbing"):
                            # v150: in front of a skull door (or under its open shaft): never stand here. With thugs alive leave by the nearer
                            # edge (toward a gap); with none alive keep walking the region's way (passing through is safe). If the walk is
                            # blocked (a thug in the way: |dx| < 2 between steps) reverse - any exit beats standing still (run 37 death 332).
                            _exit = maze_skull_exit(_zones, _pl["x"]) or _dir       # run 49: with "nobody live" it walked _dir deeper into the open skull's window and died - always take the nearer way out
                            if _live_here and maze_retreat_dir is None:
                                _other = "right" if _exit == "left" else "left"
                                _thug_side = [e for e in step_radar.get("enemies", []) if e["reachable"] and abs(e["dy"]) <= 24
                                              and 0 < (e["dx"] if _exit == "right" else -e["dx"]) <= 40]
                                _zn = next(((a, b) for a, b in _zones if a <= _pl["x"] <= b), None)
                                _detour = (abs((_pl["x"] - _zn[0]) - (_zn[1] - _pl["x"])) if _zn else 99)
                                if _thug_side and _detour <= 16 and not ((_other == "left" and _pl["x"] < 44) or (_other == "right" and _pl["x"] > 116)):
                                    _exit = _other                                             # run 44: leave AWAY from the thug - but only when that way out is about as short (run 51: an 80-px detour through the window killed in 14 steps)
                            _act = _exit
                            _down_timer = any(1 <= self.env.pyboy.get_memory_value(a) <= 9 for a in (0xC57D, 0xC5C2, 0xC583, 0xC5C8))
                            if maze_prev_x is not None and abs(_pl["x"] - maze_prev_x) < 2 and maze_retreat_dir is not None:
                                _exit = maze_retreat_dir
                                _blk = [e for e in step_radar.get("enemies", []) if e["reachable"] and abs(e["dy"]) <= 6
                                        and 0 < (e["dx"] if _exit == "right" else -e["dx"]) <= 32]
                                _flip = "right" if _exit == "left" else "left"
                                if _blk:                                                       # run 54: a standing thug blocked the way out while ANOTHER thug's knockdown timer ran - kick him regardless
                                    _act = "kick_right" if _exit == "right" else "kick_left"   # run 39: a thug blocks the exit - hit him (hall thugs fly off after one hit) instead of reversing
                                elif not ((_flip == "left" and _pl["x"] < 44) or (_flip == "right" and _pl["x"] > 116)) and _detour_ok(_zones, _pl["x"], _flip):
                                    _exit = _act = _flip                                       # reverse, but never toward the screen border (run 39 walked off the left edge and the room reset)
                                else:
                                    _act = _exit
                            maze_retreat_dir = _exit
                            shaft_auto = _act
                            plan = []
                            trap_block_until = max(trap_block_until, steps + 16)
                        elif step_radar.get("wall_block") and not _pl.get("climbing") and not _enemy_close:
                            _tw = maze_traverse_plan(self.env, self.action_names, _dir)
                            _kind = "traverse"
                            if not _tw:
                                _tw = maze_timed_walk(self.env, self.action_names, _dir); _kind = "least-loss walk"   # the block is already on us: take the cheapest way out
                            if _tw:
                                plan = list(_tw); plan_from_nav = False
                                print(f"hall: wall block on screen at step {steps} - {_kind} plan {len(_tw)} steps", flush=True)
                                shaft_auto = None
                            else:
                                print(f"hall: wall block on screen at step {steps} - no plan, waiting", flush=True)
                                shaft_auto = "a"
                        elif _zones and not _pl.get("climbing") and any(a <= _pl["x"] + (20 if _dir == "right" else -20) <= b for a, b in _zones) \
                                and not any(e["reachable"] and abs(e["dx"]) <= 40 and abs(e["dy"]) <= 24 for e in step_radar.get("enemies", [])):
                            shaft_auto = "a"                                               # v150: the way on leads into the open skull's window - wait for the bars to move (~45 steps), never walk through
                        elif (_zones or maze_skull_zones(self.env.pyboy, idle=True)) and _live_here and not _pl.get("climbing") \
                                and not (int(info.get("level", 0)) == 7 and self.env.pyboy.get_memory_value(0xFF43) >= 160):   # v153s: the sealed room is not a skull room - its door tiles read as a skull zone and the block looped 4-step knockdowns/evades on the inert back-lane thug (fullgame8, r7fix5/6)
                            # (run 57l: with the next pair still off-screen nothing was "close", the walk-on rule took the player to the right
                            #  wall and the right skull opened 4 px away - while the room's thugs live, wait at a midpoint whatever their distance)
                            # run 52: this room is NOT a lock - thugs keep spawning while the player stays (3 pairs + a 32 in 460 steps, then
                            # death at the wall); fight only what is close, otherwise fall through and walk on past the window
                            # v150: a skull screen with thugs alive: wait for them in the middle of the right-hand gap; a thug closing in gets the
                            # emulator-backed knockdown search (zones pruned, must end grounded with nobody within 48 px) - every death here
                            # followed a grab, so the fight has to be grab-free; the model never gets a step here (its walks ended in zones)
                            maze_retreat_dir = None
                            _ne0 = min((e for e in step_radar.get("enemies", []) if e["reachable"] and abs(e["dy"]) <= 24), key=lambda e: abs(e["dx"]), default=None)
                            if _ne0 is not None and abs(_ne0["dx"]) <= 56 and steps >= skull_kd_until:   # run 57c: the 16-step block after a window exit covered exactly the thugs' arrival - the search prunes the window itself
                                _cornered = _pl["x"] <= 24 or _pl["x"] >= 136
                                _kd = maze_knockdown_plan(self.env, self.action_names, _pl["feet_y"], depth=36, beam=16, time_s=45.0,
                                                          max_loss=(1 if (_cornered and read_health(self.env.pyboy) >= 3) else 0))   # run 52: cornered at the wall every damage-free line was pruned and the thug beat him down
                                if not _kd and abs(_ne0["dx"]) <= 36:
                                    _kd = maze_knockdown_plan(self.env, self.action_names, _pl["feet_y"], depth=14, beam=16, time_s=15.0, evade=True)
                                    if _kd:
                                        print(f"skull-room: evade plan {len(_kd)} steps at step {steps} (thug at {_ne0['dx']} px)", flush=True)
                                if _kd:
                                    _kd2 = maze_plan_with_exit(self.env, self.action_names, _kd)
                                    if _kd2 is None:
                                        print(f"skull-room: knockdown plan of {len(_kd)} steps discarded - dies within 24 idle steps after it", flush=True)
                                    _kd = _kd2
                                if _kd:
                                    plan = list(_kd); plan_from_nav = False
                                    print(f"skull-room: knockdown plan {len(_kd)} steps at step {steps} (thug at {_ne0['dx']} px)", flush=True)
                                else:
                                    print(f"skull-room: no knockdown plan at step {steps} (thug at {_ne0['dx']} px)", flush=True)
                                skull_kd_until = steps + (len(_kd) if _kd else 6)       # its own block: the fight planner's 60-step block kept this search from ever running (run 46)
                            shaft_auto = maze_gap_action(_zones or maze_skull_zones(self.env.pyboy, idle=True), _pl["x"])   # idle: wait at a midpoint between skulls
                            _ne = min((e for e in step_radar.get("enemies", []) if e["reachable"] and abs(e["dy"]) <= 16), key=lambda e: abs(e["dx"]), default=None)
                            if _ne is not None and abs(_ne["dx"]) <= 30:
                                shaft_auto = "kick_right" if _ne["dx"] > 0 else "kick_left"   # in reach: a standing kick sends a hall thug flying - even mid-walk (run 57g walked to the gap centre while a thug closed to 16 px and grabbed him)
                            elif shaft_auto == "a" and _ne is not None and abs(_ne["dx"]) <= 56 and _pl.get("facing") != _ne["side"]:
                                shaft_auto = _ne["side"]                                       # face him so the punch lands
                        elif (not _zones and not _pl.get("climbing") and steps >= skull_kd_until and steps >= plan_block_until - 40
                              and any(e["reachable"] and abs(e["dx"]) <= 48 and abs(e["dy"]) <= 24 for e in step_radar.get("enemies", []))
                              and not any(self.env.pyboy.get_memory_value(a) >= 10 for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2, 0xC583, 0xC5C8, 0xC588, 0xC5CD))):
                            # v150: a hall thug with no tracked HP byte (run 57g, scroll 237: the planners never engaged, the autopilot walked
                            # into him and a throw dropped the player into a hole) - the knockdown search needs no HP byte
                            _ne1 = min((e for e in step_radar.get("enemies", []) if e["reachable"] and abs(e["dy"]) <= 24), key=lambda e: abs(e["dx"]))
                            _kd = maze_knockdown_plan(self.env, self.action_names, _pl["feet_y"], depth=36, beam=16, time_s=45.0)
                            if _kd:
                                _kd = maze_plan_with_exit(self.env, self.action_names, _kd)
                            if _kd:
                                plan = list(_kd); plan_from_nav = False
                                print(f"hall: knockdown plan {len(_kd)} steps at step {steps} (untracked thug at {_ne1['dx']} px)", flush=True)
                            else:
                                print(f"hall: no knockdown plan at step {steps} (untracked thug at {_ne1['dx']} px)", flush=True)
                                shaft_auto = ("kick_right" if _ne1["dx"] > 0 else "kick_left") if abs(_ne1["dx"]) <= 30 else ("a" if _pl.get("facing") == _ne1["side"] else _ne1["side"])
                            skull_kd_until = steps + (len(_kd) if _kd else 8)
                        elif steps < trap_block_until and not _pl.get("climbing"):
                            maze_retreat_dir = None
                            shaft_auto = _dir                                              # v148: keep walking for the block period after a shaft
                            plan = []
                        elif _enemy_close and not _pl.get("climbing") and d.name in ("left", "right", "kick_left", "kick_right", "jump_kick", "down") \
                                and any(g["side"] == ("left" if d.name in ("left", "kick_left") else "right") and -4 <= abs(g["dx_edge"]) <= 20 for g in step_radar.get("pits", [])):
                            shaft_auto = "a"                                               # v150: the model's walk/kick toward a pit edge while a thug is adjacent (run 36 death 144: plan ended, model pressed right into a 16-px shaft)
                        elif _enemy_close and steps < plan_block_until and not _pl.get("climbing"):
                            # v144: an enemy the planners cannot see (no HP byte, knockdown search empty) - do not hand the step to the model
                            # (runs 20-22: it walked into the same throw three times). Face him and punch; the planners retry when unblocked.
                            _ne = min((e for e in step_radar.get("enemies", []) if e["reachable"]), key=lambda e: abs(e["dx"]), default=None)
                            shaft_auto = ("a" if (_ne is None or abs(_ne["dx"]) > 56 or _pl.get("facing") == _ne["side"]) else _ne["side"])
                        elif _pl.get("climbing") or (_col is not None and abs(_col) <= 4):
                            shaft_auto = "up"
                        elif not _enemy_close and _col is None and any(1 <= _sl[a] <= 9 and _sl[a] != maze_prev_slots.get(a, _sl[a]) for a in _sl):
                            shaft_auto = "a"            # v143d: a knockout timer is counting down = the camera is locked; walking now reaches the screen border (and lingers on the collapsing floor) - stand until it clears (static 1-9 values are leftovers of vanished enemies, not timers)
                        elif not _enemy_close and _col is None and (((_dir == "left" and _pl["x"] < 48) or (_dir == "right" and _pl["x"] > 124))
                                                                    or (maze_backoff and ((_dir == "left" and _pl["x"] < 76) or (_dir == "right" and _pl["x"] > 96)))):
                            # v143c/d: at the screen border with the camera not following (locked during a knockout animation): a kick there leaves
                            # the screen, and waiting at x < 48 stood on the collapsing floor - back off toward mid-screen, but only while the
                            # ground behind is solid (run 17: the feet-90 ledge ends ~70 px in, the back-off walked off it)
                            _back = "right" if _dir == "left" else "left"
                            if maze_probe_deadly(self.env, self.action_names.index(_back), _pl["feet_y"], 4):
                                maze_backoff = False; shaft_auto = "a"
                            else:
                                maze_backoff = True; shaft_auto = _back
                        elif not _enemy_close and _col is None and maze_backoff:
                            maze_backoff = False; shaft_auto = "a"                # mid-screen: wait one step, then resume (the camera unlocks when the knockout timers end)
                        elif not _enemy_close:
                            _walk = (("right" if _col > 0 else "left") if _col is not None else _dir)   # a column in range outranks the region direction
                            _edge = any(g["side"] == _walk and -4 <= abs(g["dx_edge"]) <= 40 and (_col is None or abs(g["dx_edge"]) < abs(_col))
                                        and not (g["left"] <= 0 if _walk == "left" else g["right"] >= 160)      # v143c: a "pit" touching the screen border is usually unscrolled map (the camera lags the player); the probe decides there
                                        for g in step_radar.get("pits", []))   # a column before the edge: walk to it, no hop
                            _probe_steps = 12 if _col is None else max(2, min(12, abs(_col) // 3))   # toward a column: look only as far as the column
                            if steps >= hop_block_until and (_edge or maze_probe_deadly(self.env, self.action_names.index(_walk), _pl["feet_y"], _probe_steps)) and not maze_grounded(self.env, _pl["feet_y"], self.action_names.index("a")):
                                shaft_auto = "a"                                   # v143c: airborne (thrown / mid-jump): no hop plan from here, wait for the landing
                            elif steps >= hop_block_until and (_edge or maze_probe_deadly(self.env, self.action_names.index(_walk), _pl["feet_y"], _probe_steps)):
                                hop_seq = hop.plan(direction=_walk) or hop.plan(direction=_walk, drop=True)
                                hop_block_until = steps + (len(hop_seq) if hop_seq else 24)
                                if hop_seq:
                                    plan = list(hop_seq); plan_from_nav = False
                                    self._save_state(ep, steps, f"mhop{hop.plans}-a{int(info.get('level', 0))}", saved_states)
                                else:
                                    # v144: nothing lands. The hall past the sealed room is a 32-px-tall lane floor: its 32-px holes are in the
                                    # front lane only, so step to the back lane (UP x4 moved feet 89 -> 77 and the hole left the pit reading)
                                    # - or to the front lane if already at the back - and let the next step re-evaluate; else stand.
                                    maze_lane_until = steps + 4
                                    maze_lane_dir = "up" if _pl["feet_y"] >= 72 else "down"
                                    shaft_auto = maze_lane_dir
                                    self._save_state(ep, steps, f"mhopfail{hop.plans}-a{int(info.get('level', 0))}", saved_states)
                            elif steps < maze_lane_until:
                                shaft_auto = maze_lane_dir                                 # v144: finishing the lane shift
                            elif steps < hop_block_until and _edge:
                                shaft_auto = "a"
                            else:
                                # v150: walking into a wall (run 55 pressed right for 150 steps at scroll 153 where the back lane ends) - after
                                # five motionless steps switch lanes (the front lane continues there); UP x8 / DOWN x8 crosses the 24-px lane gap
                                if maze_prev_x is not None and abs(_pl["x"] - maze_prev_x) < 1 and maze_prev_scx == self.env.pyboy.get_memory_value(0xFF43):
                                    maze_stuck_n += 1
                                else:
                                    maze_stuck_n = 0
                                if maze_stuck_n >= 5:
                                    maze_stuck_n = 0
                                    _esc = maze_wall_escape(self.env, self.action_names, _walk)
                                    if _esc:
                                        plan = list(_esc); plan_from_nav = False
                                        print(f"maze: wall ahead at step {steps} (x {_pl['x']}, feet {_pl['feet_y']}) - simulated lane shift {self.action_names[_esc[0]]}", flush=True)
                                        shaft_auto = None
                                    else:
                                        print(f"maze: wall ahead at step {steps} (x {_pl['x']}, feet {_pl['feet_y']}) - no safe lane shift, standing", flush=True)
                                        shaft_auto = "a"
                                else:
                                    shaft_auto = _walk
                    if REWIND and steps % REWIND_EVERY == 0:
                        _b = io.BytesIO(); self.env.pyboy.save_state(_b); rewind_ring.append((steps, _b.getvalue())); del rewind_ring[:-16]   # v154; v154d: 16 (800 steps) so skipping doomed snapshots leaves candidates
                    if steps % 100 == 0 and steps > 0 and os.environ.get("DD_SAVE_LOCAL_DIR"):
                        self._save_state(ep, steps, f"{'hall' if maze_entered else 'run'}-s{steps}-a{int(info.get('level', 0))}-scx{self.env.pyboy.get_memory_value(0xFF43)}", saved_states)   # v150: periodic hall saves; v153r: everywhere (local only) - fullgame9's three mission-2 deaths had no fixture within 150 steps
                    if maze_entered and step_radar.get("player") and not plan and shaft_auto in ("left", "right"):
                        # v150: global wall detector - any walking action that moves nothing for 6 steps gets the simulated lane shift
                        # (run 57o: a stale HP slot kept the midpoint-wait branch walking into the wall at scroll 161 for 800 steps)
                        _pl2 = step_radar["player"]
                        if maze_prev_x is not None and abs(_pl2["x"] - maze_prev_x) < 1 and maze_prev_scx == self.env.pyboy.get_memory_value(0xFF43):
                            maze_stuck_n += 1
                        else:
                            maze_stuck_n = 0
                        if maze_stuck_n >= 6:
                            maze_stuck_n = 0
                            _esc = maze_wall_escape(self.env, self.action_names, shaft_auto)
                            if _esc:
                                plan = list(_esc); plan_from_nav = False
                                print(f"maze: stuck walking {shaft_auto} at step {steps} (x {_pl2['x']}, feet {_pl2['feet_y']}) - simulated lane shift {self.action_names[_esc[0]]}", flush=True)
                                shaft_auto = None
                            else:
                                print(f"maze: stuck walking {shaft_auto} at step {steps} - no safe lane shift", flush=True)
                                shaft_auto = "a"
                    if maze_entered:
                        if os.environ.get("DD_MAZE_DEBUG") and step_radar.get("player"):
                            print(f"maze-step {steps} scx={self.env.pyboy.get_memory_value(0xFF43)} scy={self.env.pyboy.get_memory_value(0xFF42)} x={step_radar['player']['x']} feet={step_radar['player']['feet_y']} slots={[self.env.pyboy.get_memory_value(a) for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2)]} hp={read_health(self.env.pyboy)} auto={shaft_auto} plan={len(plan)} model={d.name} en={[(e['dx'], e['dy']) for e in step_radar.get('enemies', [])]} zones={_zones} inz={int(_in_zone)} live={int(_live_here)} close={int(bool(_enemy_close))} tb={trap_block_until}", flush=True)
                        maze_prev_slots = {a: self.env.pyboy.get_memory_value(a) for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2)}
                        maze_prev_hp = read_health(self.env.pyboy); maze_prev_feet = step_radar["player"]["feet_y"] if step_radar.get("player") else None
                        maze_prev_x = step_radar["player"]["x"] if step_radar.get("player") else None
                        maze_prev_scx = self.env.pyboy.get_memory_value(0xFF43)
                    if maze_entered and ending_steps > 0 and not game_complete and not step_radar.get("player") and self.env.pyboy.get_memory_value(0xFF43) == 0:
                        game_complete = True
                        print(f"*** GAME COMPLETE *** step {steps}: the doorway fade started (hp {read_health(self.env.pyboy)}, lives {self.env.pyboy.get_memory_value(0xC499)}) - credits next", flush=True)
                        self._save_state(ep, steps, "GAME-COMPLETE", saved_states)
                    if game_complete:
                        plan = []; shaft_auto = "a"                                   # let the credits roll
                    stal_dodge = None
                    if hideout_entered and not maze_entered and step_radar.get("player") and step_radar.get("stalactites"):
                        _px = step_radar["player"]["x"]
                        _near = [st for st in step_radar["stalactites"] if abs(st["x"] - _px) <= 22 and st["bottom"] < step_radar["player"]["feet_y"] - 4]
                        if _near:
                            _sx = min(_near, key=lambda st: abs(st["x"] - _px))["x"]
                            stal_dodge = "left" if _sx >= _px else "right"            # v153f
                            if steps >= stal_dodge_until:
                                stal_dodge_dir, stal_dodge_until = stal_dodge, steps + 8   # v153o: latch the first choice for 8 steps (~18 px) - re-deciding every step flip-flopped the player under the stalactite (probe: x 108 <-> 105, hit at t=9)
                    if stal_dodge_dir and steps < stal_dodge_until:
                        stal_dodge = stal_dodge_dir
                    # v153ab: finish a climb when stalled on a ladder. fullgame12 step 3500 (mission 2, chunk 94): the player hung mid-ladder
                    # with a thug 59 px left / 45 up - outside the stalled-fight reach - and three 120-s nav searches found nothing. From
                    # that state 20 held UPs reach the catwalk with the thug level, where the fight planner can take him.
                    rewind_hold = "a" if (REWIND and steps < rewind_hold_until) else None   # v154
                    pit_step = None
                    if not plan and step_radar.get("player") and d.name in ("left", "right") and not step_radar["player"].get("climbing"):
                        for g in step_radar.get("pits", []):
                            if g["side"] == d.name and -2 <= g["dx_edge"] <= 8 and (g["right"] - g["left"]) >= 16:
                                pit_step = "a"                                           # v153af: ladder4 - after a verified hop landed at x 103 the model pressed LEFT twice into the pit it had just crossed (edge 1 px away); a model step onto a pit edge becomes a punch in every mission
                                break
                    boss_repos = None
                    if (maze_entered and boss_seen_here and not plan and boss_fail_here >= 2 and step_radar.get("player")
                            and self.env.pyboy.get_memory_value(0xC595) >= 12 and step_radar["player"]["x"] > 16):
                        boss_repos = "left"                                                 # v154g: the drain exists from the far-left spot (fullgame14 arrival at x 62: 3 searches, none; same state walked to x 10: 118-step drain in 337 s) - walk there before searching again
                    far_approach = None
                    if (not plan and not maze_entered and not hideout_entered and step_radar.get("player") and step_radar.get("enemies")
                            and steps - max(tracker.pos_changed_step, nav_gain_step) >= 60 and fight_fail_streak >= 2):
                        _fe = min(step_radar["enemies"], key=lambda e: abs(e["dx"]))
                        if 48 < abs(_fe["dx"]) <= 110 and abs(_fe["dy"]) <= 24:
                            _dir = "right" if _fe["dx"] > 0 else "left"
                            if not any(g["side"] == _dir and -2 <= g["dx_edge"] <= 24 and (g["right"] - g["left"]) >= 16 for g in step_radar.get("pits", [])):
                                far_approach = _dir                                       # v154l: fullgame17 stood 300+ steps on the mission-2 balcony with a thug waiting in the doorway 60 px away - the kill searches failed from range, nav had no exit, the model idled. Walk at him (no pit ahead) until the planner can reach him.
                    blind_hold = None
                    if (not plan and steps < blind_hold_until and step_radar.get("player") and not ending_steps   # v154f: Marian in the doorway is not an enemy - the stance preempted the ending's UP (fullgame13 stood at the doorway 1500 steps; 27 held UPs from its live state fade the screen)
                            and any(e["reachable"] and abs(e["dx"]) <= 28 and abs(e["dy"]) <= 12 for e in step_radar.get("enemies", []))):
                        _ne0 = min(step_radar["enemies"], key=lambda e: abs(e["dx"]))
                        blind_hold = "a" if step_radar["player"].get("facing") == _ne0["side"] else _ne0["side"]   # v153ac: face the thug and punch - no walking, no model
                    ladder_finish = None
                    if (not plan and step_radar.get("player") and step_radar["player"].get("climbing") and steps - max(tracker.pos_changed_step, nav_gain_step) >= 40
                            and not any(e["reachable"] and e["dy"] > 10 and abs(e["dx"]) <= 40 for e in step_radar.get("enemies", []))):
                        ladder_finish = "up"
                    room7_approach = None
                    if (maze_entered and not plan and int(info.get("level", 0)) == 7 and self.env.pyboy.get_memory_value(0xFF43) >= 160
                            and self.env.pyboy.get_memory_value(0xFF42) == 0 and step_radar.get("player") and step_radar.get("enemies")):
                        _ne = min(step_radar["enemies"], key=lambda e: abs(e["dx"]))
                        _ex = step_radar["player"]["x"] + _ne["dx"]
                        _far = abs(_ne["dx"]) > 56 or _ex < 16 or _ex > 144                # v153v: a thug at the screen edge is unhittable (r7fix9: player at x 11, thug at x 8 with 3 HP, two searches found nothing) - treat it as far and go hold the door column so it follows inward
                        if _far and abs(105 - step_radar["player"]["x"]) > 6:
                            room7_approach = "right" if step_radar["player"]["x"] < 105 else "left"   # v153u: hold the door column while the wave is far - the pair respawns ~18 steps after a wave dies (r7fix8: room empty for steps 42-59, the align walk from x 34 never reached x 105 in time), so the fights happen next to the door
                        elif _far and step_radar["player"]["feet_y"] < 100:
                            room7_approach = "down"                                       # v153u: and on the front lane, where the thugs engage
                        elif _far:
                            room7_approach = "a"                                           # v153v: at the door column on the front lane - wait
                        elif abs(_ne["dy"]) > 8:
                            room7_approach = "down" if _ne["dy"] > 0 else "up"           # v153q: the room's thugs fight on their own lane - from the back lane the edge thug is inert (12 kicks: nothing; fullgame8 looped 4-step knockdowns on it for 400 steps) while on the front lane idle punches killed the pair in 40 steps
                    override = (rewind_hold or stal_dodge or boss_repos or room7_approach or room7_align or hdoor_align or ladder_finish or blind_hold or far_approach or pit_step or ("right" if (left_edge or walkway_auto is True) else ("a" if (walkway_auto == "down" or door2_guard) else shaft_auto)) or air_veto(step_radar, belt, d.name) or heavy_autopilot(step_radar, d.name, int(info.get("level", 0)), last_heavy)
                                or river_exit_veto(step_radar, d.name, int(info.get("level", 0)))
                                or cave_hop_veto(step_radar, d.name, in_forest or not mission4)   # v100: mission-4 cave only (it hopped mission-2 players off area 66)
                                or belt_reflex(step_radar, belt, auto or d.name, intent_src)
                                or heavy_lane_veto(step_radar, auto or d.name, int(info.get("level", 0)))   # the stones autopilot outranks turning/lining up
                                or hole_margin_veto(step_radar, d.name)
                                or (walk_in_veto(step_radar, d.name) if WALK_IN else None)
                                or ledge_veto(step_radar, d.name) or ladder_align_veto(step_radar, d.name) or rung_veto(step_radar, d.name)
                                or pit_back_veto(step_radar, d.name) or no_freelance_jump_veto(step_radar, d.name, travel_dir)
                                or auto)
                    # v23's boss_veto (forced kick / step-back from correlational hit rates) went 0/21 and was dropped. v49's
                    # heavy_lane_veto enforces an emulator-verified mechanic instead (hits need lane alignment within 2 px).
                    if os.environ.get("DD_STEP_DEBUG"):
                        _pp = step_radar.get("player") or {}
                        print(f"step {steps} a={int(info.get('level', 0))} scx={self.env.pyboy.get_memory_value(0xFF43)} x={_pp.get('x')} feet={_pp.get('feet_y')} hp={read_health(self.env.pyboy)} lives={self.env.pyboy.get_memory_value(0xC499)} src={'plan' if plan else ('ovr:' + str(override) if override else 'model')} act={self.action_names[plan[0]] if plan else (override or d.name)} en={[(e['dx'], e['dy']) for e in step_radar.get('enemies', [])][:3]} pits={[(g['left'], g['right']) for g in step_radar.get('pits', [])][:3]}", flush=True)
                    if os.environ.get("DD_HIDEOUT_DEBUG") and hideout_entered and not maze_entered:
                        _pp = step_radar.get("player") or {}
                        print(f"hideout-step {steps} a={int(info.get('level', 0))} scx={self.env.pyboy.get_memory_value(0xFF43)} scy={self.env.pyboy.get_memory_value(0xFF42)} x={_pp.get('x')} feet={_pp.get('feet_y')} hp={read_health(self.env.pyboy)} stal={[(st['x'] - int(_pp.get('x', 0) or 0), st['top'], st['bottom']) for st in step_radar.get('stalactites', [])]} dodge={stal_dodge} ovr={override} model={d.name} plan={len(plan)} pits={[(g['left'], g['right']) for g in step_radar.get('pits', [])]} slabs={step_radar.get('slabs_visible')}", flush=True)
                    if override in ("left", "right") and not plan and step_radar.get("player") and not step_radar["player"].get("climbing"):
                        for g in step_radar.get("pits", []):
                            if g["side"] == override and -2 <= g["dx_edge"] <= 8 and (g["right"] - g["left"]) >= 16:
                                override = "a"; break                                 # v154e2: the plain pit-edge veto (v153af) for autopilot steps too, every mission
                    if maze_entered and override in ("left", "right") and not plan:
                        _hv = hole_margin_veto(step_radar, override)
                        if _hv:
                            override = _hv                                        # v154e: the hall autopilots' walk-on steps respect the floor holes too - h13 trace: after a kill plan ended, "ovr:right" walked the player into the hole 3 px ahead (feet 68 -> 88 -> 111 -> gone) while two thugs dropped in from the upper lane
                    if plan:
                        override = self.action_names[plan.pop(0)]            # v82: deterministic replay of the searched fight
                        if os.environ.get("DD_PLAN_DEBUG"):
                            _pp = step_radar.get("player") or {}
                            print(f"plan-step {steps} {override} ehp={self.env.pyboy.get_memory_value(0xC595)} x={_pp.get('x')} feet={_pp.get('feet_y')} d={d.name} d.action={d.action}", flush=True)
                    if override and override != d.name and override in self.action_names:
                        exec_name, exec_idx = override, self.action_names.index(override)
                        reflex_steps += 1
                    else:
                        exec_name, exec_idx = d.name, d.action
                    if os.environ.get("DD_REFLEX_DEBUG") and (hv or int(info.get("level", 0)) in (46, 48)):
                        _p = step_radar.get("player") or {}; _e = min(step_radar.get("enemies", []), key=lambda e: abs(e["dx"]), default=None)
                        log.info("heavy-debug step=%d model=%s exec=%s auto=%s n=%d player=(%s,%s,%s) enemy=%s others=%s hp=%s", steps, d.name, exec_name,
                                 heavy_autopilot(step_radar, d.name, int(info.get("level", 0)), last_heavy), len(step_radar.get("enemies", [])),
                                 _p.get("x"), _p.get("feet_y"), _p.get("facing"), _e and (_e["dx"], _e["dy"], _e.get("heavy"), _e["reachable"], _e["in_reach"]),
                                 [(e["dx"], e["dy"], e.get("heavy"), e["in_reach"]) for e in step_radar.get("enemies", []) if e is not _e], step_radar.get("health_ticks"))
                    if os.environ.get("DD_REFLEX_DEBUG") and (belt.direction or step_radar.get("pits")):
                        _p = step_radar.get("player") or {}
                        log.info("reflex-debug step=%d model=%s exec=%s belt=%s/%.1f player=(%s,%s) pits=%s", steps, d.name, exec_name,
                                 belt.direction, belt.speed, _p.get("x"), _p.get("feet_y"), [(g["x"], g["dx_edge"]) for g in step_radar.get("pits", [])])
                    next_obs, reward, terminated, truncated, info = self.env.step(exec_idx)
                    scroll.update()
                    trace.add(obs, d, reward, info, steps, step_radar, scroll.distance_px, exec_idx)   # per-step radar, not the decision's
                    obs = next_obs
                    ep_return += reward
                    steps += 1
                    if HP_LOG is not None:
                        _p = step_radar.get("player") or {}
                        if HP_LOG and read_health(self.env.pyboy) < HP_LOG[-1][2] and os.environ.get("DD_HP_SHOTS") and len(glob.glob(os.environ["DD_HP_SHOTS"] + "/hpdrop_*.png")) < 24:
                            from PIL import Image
                            _img = self.env.pyboy.botsupport_manager().screen().screen_ndarray()
                            Image.fromarray(_img).resize((480, 432), Image.NEAREST).save(f"{os.environ['DD_HP_SHOTS']}/hpdrop_{steps:05d}.png")
                            with open(f"{os.environ['DD_HP_SHOTS']}/hpdrop_{steps:05d}.state", "wb") as _fh:
                                self.env.pyboy.save_state(_fh)
                        HP_LOG.append((steps, int(info.get("level", 0)), read_health(self.env.pyboy), int(info.get("lives", 0)), len(plan), len(step_radar.get("enemies", [])), [self.env.pyboy.get_memory_value(a) for a in (0xC595, 0xC5DA, 0xC57D, 0xC5C2)], _p.get("x"), _p.get("feet_y"), bool(_p.get("climbing")), int(info.get("score", 0)), self.action_names[exec_idx], (d.name if d else None), override, self.env.pyboy.get_memory_value(0xFF43)))   # v113 debug (+ v123: executed/model/override actions, scx)
                    cur_score = int(info.get("score", 0))
                    if SAVE_AREAS and int(info.get("level", 0)) in SAVE_AREAS and (steps % 60 == 0 or int(info.get("level", 0)) not in saved_areas):
                        saved_areas.add(int(info.get("level", 0)))      # study hook: states of a level we have no fixtures for
                        self._save_state(ep, steps, f"area{int(info.get('level', 0))}", saved_states)   # v84: uploaded like life-lost states
                    if cur_score > last_score:
                        last_gain_step = steps
                        if cur_score - last_score == 30 and last_heavy is not None:
                            last_heavy["just_hit"] = True          # a punch landed: the combo veto follows through
                        if int(info.get("level", 0)) == 48 or any(e.get("heavy") for e in step_radar.get("enemies", [])):
                            boss_hits += 1
                    last_score = cur_score
                    lives_lost = max(0, start_lives - int(info.get("lives", 0)))
                    if int(info.get("lives", 0)) < prev_lives:
                        boss_hits = 0            # measured: the boss recovers all his health when you lose a life
                        plan = []                # a lost life invalidates the replayed sequence
                        fight_armed = False; last_life_lost_step = steps   # v138
                        _did_rewind = False
                        if REWIND:
                            _lv = int(info.get("level", 0)); _k = rewind_counts.get(_lv, 0)
                            _cap = 10 if (maze_entered and _lv in (96, 97)) else REWIND_MAX   # v154n: the hall's skull room (96/97) needs many timing-shifted attempts; other chunks keep 3
                            rewind_deaths.append(steps)
                            _cands = [(s_, st) for s_, st in rewind_ring if s_ <= steps - REWIND_BACK
                                      and not any(d_ - 40 <= s_ <= d_ for d_ in rewind_deaths)             # v154d: a snapshot taken in the 40 steps before ANY death is doomed (fullgame13: the step-10550 snapshot, 2 steps before a death, was reloaded twice and died 2 steps later each time)
                                      and s_ != rewind_last_used]                                          # v154d: never the same snapshot twice in a row
                            if _k < _cap and _cands:
                                s_, st = _cands[-1]; rewind_last_used = s_
                                self._save_state(ep, steps, f"death-before-rewind{rewinds_total + 1}", saved_states)
                                self.env.pyboy.load_state(io.BytesIO(st)); self.env.kick_penalty = False
                                rewind_counts[_lv] = _k + 1; rewinds_total += 1; _did_rewind = True
                                rewind_hold_until = steps + 8 * (_k + 1)                 # a timing shift so the deterministic model takes another line
                                plan = []; nav_block_until = max(nav_block_until, steps + 60); plan_block_until = steps
                                info = dict(info); info["lives"] = prev_lives
                                print(f"rewind #{rewinds_total}: life lost at step {steps} in chunk {_lv} - reloaded the step-{s_} snapshot, holding {8 * (_k + 1)} steps", flush=True)
                        if steps - bruiser_hp_seen_step <= 48 and bruiser_hp_min is not None:
                            bruiser_hp_at_life_lost.append(int(step_radar.get("bruiser_hp", bruiser_hp_min)))
                        self._save_state(ep, steps, "life-lost", saved_states)
                        if REWIND and not _did_rewind:
                            rewind_counts = {}                                              # v154n: a real death respawns at the hall start - a fresh attempt gets the full per-chunk rewind budget again
                        if boss_seen_here and not game_complete:
                            print(f"boss-room: life lost at step {steps} - respawn is the hall start, scripts reset", flush=True)
                        boss_seen_here, boss_gone_steps, ending_steps, finish_block_until = False, 0, 0, -1   # v150: a death in the throne room respawns at the hall start
                    prev_lives = int(info.get("lives", 0))
                    streak = streak + 1 if d.name == prev_name else 1
                    recent.append(d.name)
                    if len(recent) > 12:          # 12 kept for stuck detection; the model is shown the last 8
                        recent.pop(0)
                    prev_name = d.name
                    action_hist[d.name] = action_hist.get(d.name, 0) + 1
                    cur_level = int(info.get("level", 0))
                    if cur_level != levels_visited[-1]:
                        levels_visited.append(cur_level)
                    max_score = max(max_score, int(info.get("score", 0)))
                    if d.ok:
                        ok_steps += 1
                        if fresh:
                            conf_sum += d.confidence
                            lat_sum += d.latency_s
                            queried += 1
                    done = terminated or truncated
                    if steps % 25 == 0:
                        self.progress.update(episode=ep, step=steps, score=info.get("score", 0),
                                             lives=info.get("lives", 0), stage=len(set(levels_visited)) - 1,
                                             distance=scroll.distance_px, vertical=scroll.vertical_px, canon=tracker.max_pos, mission=tracker.mission())
                        self.progress["return"] = ep_return
                levels_completed = int(info.get("levels_completed", 0))
                if HP_LOG is not None:
                    with open(os.environ["DD_HP_LOG"], "w") as fh:
                        json.dump(HP_LOG, fh)
                    HP_LOG.clear()
                if truncated and not terminated:
                    self._save_state(ep, steps, "truncated", saved_states)   # where did it stall?
                summary = {
                    "run_tag": self.run_tag, "worker": self.idx, "episode": ep, "steps": steps,
                    # DDEnv names its MP4 rollout_reset{reset-1}_session{session}_id{instance_id}_vid*.mp4
                    "env_instance_id": info.get("instance_id"), "env_session": info.get("session"),
                    "video_prefix": f"ddenv_videos/decision/{self.run_tag}/rollout_reset{self.env.reset_count - 1}_session{info.get('session')}_id{info.get('instance_id')}_",
                    "return": round(ep_return, 3), "final_score": info.get("score"), "max_score": max_score,
                    "distance_px": scroll.distance_px, "screens_scrolled": round(scroll.distance_px / 160, 2),
                    "vertical_px": scroll.vertical_px, "total_progress_px": scroll.total_px,
                    "min_health_ticks": min_health,
                    "bruiser_hp_min": bruiser_hp_min, "bruiser_hp_at_life_lost": bruiser_hp_at_life_lost,
                    "fight_plans": plan_count, "fight_kills": fight_kills, "nav_plans": nav_count,
                    "canon_pos_max": tracker.max_pos, "canon_pos_end": tracker.pos, "canon_regressions": tracker.regressions,   # v109: index into the canonical chunk sequence (336 = arena)
                    "mission_max": 4 if tracker.max_pos >= M4_START else (3 if tracker.max_pos >= M3_START else (2 if tracker.max_pos >= M2_START else 1)),
                    "reflex_steps": reflex_steps, "softlock_reloads": softlock_reloads,   # v129
                    "rewinds": rewinds_total,                                                   # v154
                    "final_lives": info.get("lives"), "start_level": start_level,
                    "start_level_name": LEVEL_NAMES.get(start_level, f"unknown-{start_level}"),
                    "levels_visited": levels_visited,
                    "levels_visited_names": [LEVEL_NAMES.get(l, f"unknown-{l}") for l in levels_visited],
                    "new_levels_reached": len(set(levels_visited)) - 1,
                    "levels_completed": levels_completed,   # DDEnv's own count (only its 5 known ids)
                    "game_cleared": None,                   # TODO: ending-screen RAM signature
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "model_ok_steps": ok_steps, "fallback_steps": steps - ok_steps,
                    "model_queries": queried, "decision_interval": self.decision_interval,
                    "mean_confidence": round(conf_sum / queried, 4) if queried else None,
                    "mean_latency_s": round(lat_sum / queried, 4) if queried else None,
                    "steps_per_s": round(steps / max(time.time() - t_ep, 1e-6), 3),
                    "wall_s": round(time.time() - t_ep, 1),
                    "actions": action_hist,
                    "trace_key": f"decision_runs/{self.run_tag}/traces/w{self.idx:03d}_ep{ep:02d}.npz" if self.s3 else None,
                    "saved_states": [f"decision_runs/{self.run_tag}/{k}" for k in saved_states],
                }
                if self.s3:
                    self._put(f"traces/w{self.idx:03d}_ep{ep:02d}.npz", trace.to_npz_bytes(summary), "application/octet-stream")
                    self._put(f"episodes/w{self.idx:03d}_ep{ep:02d}.json", json.dumps(summary).encode(), "application/json")
                results.append(summary)
                log.info("worker %d episode %d done: %s", self.idx, ep, summary)
                # per-episode record right away (episodes.jsonl is only assembled when the whole run ends, so a run
                # that is stopped early would otherwise leave no per-episode metrics behind)
                self._put(f"episodes/w{self.idx:03d}_ep{ep:02d}.json", json.dumps(summary).encode(), "application/json")
        finally:
            try:
                self.env.close()
            except Exception as exc:  # never let cleanup mask results
                log.warning("worker %d env.close failed: %s", self.idx, exc)
        return results


def summarize(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not all_results:
        return {"episodes": 0}
    rets = [r["return"] for r in all_results]
    scores = [r["max_score"] or 0 for r in all_results]
    lat = [r["mean_latency_s"] for r in all_results if r["mean_latency_s"] is not None]
    conf = [r["mean_confidence"] for r in all_results if r["mean_confidence"] is not None]
    total_steps = sum(r["steps"] for r in all_results)
    fallback = sum(r["fallback_steps"] for r in all_results)
    actions: Dict[str, int] = {}
    for r in all_results:
        for k, v in r["actions"].items():
            actions[k] = actions.get(k, 0) + v
    level_hist: Dict[str, int] = {}
    for r in all_results:
        for l in set(r["levels_visited"]):
            name = LEVEL_NAMES.get(l, f"unknown-{l}")
            level_hist[name] = level_hist.get(name, 0) + 1
    dist = [r.get("distance_px", 0) for r in all_results]
    vert = [r.get("vertical_px", 0) for r in all_results]
    return {
        "episodes": len(all_results),
        "total_steps": total_steps,
        "distance_px_mean": round(statistics.mean(dist), 1),
        "distance_px_max": max(dist),
        "screens_scrolled_max": round(max(dist) / 160, 2),
        "vertical_px_max": max(vert),
        "canon_pos_max": max(r.get("canon_pos_max", -1) for r in all_results),
        "canon_pos_mean": round(statistics.mean(r.get("canon_pos_max", -1) for r in all_results), 1),
        "episodes_reaching_mission": {m: sum(1 for r in all_results if r.get("mission_max", 1) >= m) for m in (2, 3, 4)},
        "return_mean": round(statistics.mean(rets), 3),
        "return_max": round(max(rets), 3),
        "score_mean": round(statistics.mean(scores), 1),
        "score_max": max(scores),
        "levels_completed_total": sum(r["levels_completed"] for r in all_results),
        "levels_completed_max": max(r["levels_completed"] for r in all_results),
        "new_levels_reached_max": max(r["new_levels_reached"] for r in all_results),
        "episodes_reaching_level": level_hist,
        "episodes_game_cleared": sum(1 for r in all_results if r.get("game_cleared")),
        "died_fraction": round(sum(1 for r in all_results if r["terminated"]) / len(all_results), 3),
        "model_fallback_fraction": round(fallback / total_steps, 4) if total_steps else None,
        "latency_mean_s": round(statistics.mean(lat), 4) if lat else None,
        "confidence_mean": round(statistics.mean(conf), 4) if conf else None,
        "action_distribution": dict(sorted(actions.items(), key=lambda kv: -kv[1])),
    }


def upload_summary(summary: Dict[str, Any], results: List[Dict[str, Any]], run_tag: str) -> None:
    s3, bucket = s3_client()
    if not s3:
        return
    try:
        s3.put_object(Bucket=bucket, Key=f"decision_runs/{run_tag}/summary.json", ContentType="application/json",
                      Body=json.dumps(summary, indent=2).encode())
        s3.put_object(Bucket=bucket, Key=f"decision_runs/{run_tag}/episodes.jsonl", ContentType="application/x-ndjson",
                      Body="".join(json.dumps(r) + "\n" for r in results).encode())
        log.info("summary uploaded to s3://%s/decision_runs/%s/", bucket, run_tag)
    except Exception as exc:
        log.warning("summary upload failed: %s", exc)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-envs", type=int, default=int(os.environ.get("NUM_ENVS", "100")))
    p.add_argument("--episodes-per-env", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2048, help="steps per episode before truncation")
    p.add_argument("--init-states", default=os.environ.get("INIT_STATES"),
                   help="comma-separated save states to start from (default: all six); "
                        "use dd.gb.state alone for full-game attempts")
    p.add_argument("--endpoints", default=None, help="comma-separated decision server base URLs (overrides DGEMMA_* env)")
    p.add_argument("--include-native-frame", action="store_true",
                   help="ALSO send the full 160x144 frame (goes beyond what the RL agent saw)")
    p.add_argument("--image-scale", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None,
                   help="sample actions from the model's probabilities at this temperature (0/omitted = argmax)")
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--decision-interval", type=int, default=int(os.environ.get("DECISION_INTERVAL", "1")),
                   help="ask the model every K env steps and repeat its action in between (1 = every step)")
    p.add_argument("--no-video", action="store_true", help="skip the per-episode MP4 upload")
    p.add_argument("--no-traces", action="store_true", help="skip per-step trace (npz) upload")
    p.add_argument("--no-obs-in-traces", action="store_true", help="traces without observation frames (much smaller)")
    p.add_argument("--local", action="store_true", help="run in-process without Ray")
    p.add_argument("--dry-run", action="store_true", help="build one request from a fresh env and print it")
    p.add_argument("--run-tag", default=time.strftime("%Y%m%d-%H%M%S"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    client_kwargs: Dict[str, Any] = {}
    if args.endpoints:
        client_kwargs["endpoints"] = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    if args.include_native_frame:
        client_kwargs["include_native_frame"] = True
    if args.image_scale is not None:
        client_kwargs["image_scale"] = args.image_scale
    if args.timeout is not None:
        client_kwargs["timeout_s"] = args.timeout
    if args.temperature is not None:
        client_kwargs["sample_temperature"] = args.temperature
    init_states = [x.strip() for x in args.init_states.split(",") if x.strip()] if args.init_states else None
    env_config = build_env_config(args.max_steps, save_video=not args.no_video, run_tag=args.run_tag,
                                  init_states=init_states)
    record_traces = not args.no_traces and bool(os.environ.get("S3_BUCKET_NAME"))
    record_obs = not args.no_obs_in_traces
    worker_args = (env_config, client_kwargs, args.episodes_per_env, args.run_tag, record_traces, record_obs,
                   args.decision_interval)

    if args.dry_run:
        from ddenv import DDEnv
        env_config.update(save_video=False)
        env = DDEnv(env_config)
        obs, info = env.reset()
        client = DecisionModelClient(**client_kwargs)
        body = client.build_request(obs, info, env.screen.screen_ndarray(), None)
        shown = dict(body, images=[f"<{len(i)} chars data URL>" for i in body["images"]])
        print(json.dumps(shown, indent=2))
        print(f"\nobservation shape={obs.shape} payload_bytes={len(json.dumps(body))} endpoints={client.endpoints}")
        env.close()
        return 0

    log.info("run_tag=%s num_envs=%d episodes_per_env=%d max_steps=%d decision_interval=%d traces=%s obs_in_traces=%s",
             args.run_tag, args.num_envs, args.episodes_per_env, args.max_steps, args.decision_interval,
             record_traces, record_obs)
    t0 = time.time()
    if args.local:
        workers = [EnvWorker(i, *worker_args) for i in range(args.num_envs)]
        results = [r for w in workers for r in w.run()]
    else:
        import ray
        if not ray.is_initialized():
            ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
        log.info("ray resources: %s", ray.cluster_resources())
        Actor = ray.remote(num_cpus=1, max_concurrency=2)(EnvWorker)
        actors = [Actor.remote(i, *worker_args) for i in range(args.num_envs)]
        pending = {a.run.remote(): a for a in actors}
        results = []
        last_report = 0.0
        while pending:
            done, _ = ray.wait(list(pending), num_returns=1, timeout=30)
            for ref in done:
                actor = pending.pop(ref)
                try:
                    results.extend(ray.get(ref))
                except Exception as exc:
                    log.error("worker failed: %s", exc)
                try:
                    ray.kill(actor)      # release the CPU slot: finished workers held 1 CPU each until the whole job ended
                except Exception:        # (v51 waited 27 min for CPUs behind 100 idle actors of two earlier runs)
                    pass
            if time.time() - last_report > 60 and pending:
                snaps = ray.get([a.get_progress.remote() for a in list(pending.values())[:200]])
                steps = sum(s["step"] for s in snaps)
                reqs = sum(s["requests"] for s in snaps)
                errs = sum(s["errors"] for s in snaps)
                lat = sum(s["latency_sum"] for s in snaps) / max(reqs - errs, 1)
                best_stage = max((s["stage"] for s in snaps), default=0)
                n = max(len(snaps), 1)
                mean_score = sum(s.get("score", 0) for s in snaps) / n
                best_score = max((s.get("score", 0) for s in snaps), default=0)
                mean_lives = sum(s.get("lives", 0) for s in snaps) / n
                mean_dist = sum(s.get("distance", 0) for s in snaps) / n
                best_dist = max((s.get("distance", 0) for s in snaps), default=0)
                best_vert = max((s.get("vertical", 0) for s in snaps), default=0)
                canon_mean = sum(s.get("canon", 0) for s in snaps) / n
                canon_best = max((s.get("canon", 0) for s in snaps), default=0)
                missions = {m: sum(1 for s in snaps if s.get("mission", 1) >= m) for m in (2, 3, 4)}
                log.info("progress: %d/%d workers running, steps=%d, requests=%d, errors=%d, mean_latency=%.3fs, "
                         "score mean=%.0f best=%d, lives mean=%.2f, distance mean=%.0fpx best=%dpx, vertical best=%dpx, chunk mean=%.0f best=%d/%d, missions>=2/3/4=%d/%d/%d, done=%d, elapsed=%.0fs",
                         len(pending), args.num_envs, steps, reqs, errs, lat, mean_score, best_score, mean_lives,
                         mean_dist, best_dist, best_vert, canon_mean, canon_best, len(CANON) - 1, missions[2], missions[3], missions[4], len(results), time.time() - t0)
                last_report = time.time()

    summary = summarize(results)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    summary["run_tag"] = args.run_tag
    print("\n=== decision-model run summary ===")
    print(json.dumps(summary, indent=2))
    upload_summary(summary, results, args.run_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
