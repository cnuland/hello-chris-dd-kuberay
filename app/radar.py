"""
Sprite "radar" for Double Dragon (Game Boy), read straight from the OAM sprite table via PyBoy.

Empirically (see README "Sprite radar"): the player is drawn with OAM slots 0-6 using tile ids < 128,
every enemy is an 8-sprite block (2 columns x 4 rows of 8x8 tiles, tile ids >= 128). Sprite (x, y) is
the top-left of each 8x8 cell; screen is 160x144. We cluster non-player sprites into objects and report
each one relative to the player so the decision model gets exact geometry instead of guessing from pixels.
"""
from __future__ import annotations

import os
from typing import Optional, Any, Dict, List, Tuple

PLAYER_SLOTS = range(0, 7)
CLIMB_TILES = {13, 28, 29, 44, 45, 60, 61}   # player sprite tiles of the ladder-climbing pose (head 13, body 28/29,
                                             # 44/45, legs 60/61); standing / walking use 0, 16/17, 32-37, 48-53
REACH_X = 22        # px: roughly one body width; punch/kick connects inside this
REACH_Y = 10        # px: must be on (almost) the same street lane
LANE_MIN_FEET = 86  # measured: player's feet_y spans 91 (top of street) .. 119 (bottom); +-5 slack
LANE_MAX_FEET = 124 # anything outside is on a raised platform / background object: unreachable
DOOR_TILE = 362     # the solid-dark doorway tile; doors are >=2 tiles wide blocks whose bottom row touches the street top
DOOR_BOTTOM_ROWS = (9, 10, 11)   # tile rows (y 72..95) where a doorway's bottom edge lands


def _cluster(cells: List[Tuple[int, int]], dx: int = 12, dy: int = 10) -> List[List[Tuple[int, int]]]:
    """Group 8x8 sprite cells into objects: cells within dx/dy of any cell already in the group."""
    groups: List[List[Tuple[int, int]]] = []
    for c in sorted(cells):
        placed = False
        for g in groups:
            if any(abs(c[0] - o[0]) <= dx and abs(c[1] - o[1]) <= dy for o in g):
                g.append(c)
                placed = True
                break
        if not placed:
            groups.append([c])
    # merge groups that touch after growth
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(abs(a[0] - b[0]) <= dx and abs(a[1] - b[1]) <= dy for a in groups[i] for b in groups[j]):
                    groups[i] += groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    return groups


def _bbox(cells: List[Tuple[int, int]]) -> Dict[str, int]:
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    return {"x": (min(xs) + max(xs) + 8) // 2, "feet_y": max(ys) + 8, "top_y": min(ys), "w": max(xs) - min(xs) + 8}


HEALTH_TICK_TILE = 378
HEALTH_COLS = range(4, 12)      # window-layer row 0, the "1P" health bar: 7 ticks when full


def read_health(pyboy) -> int:
    """Player health in HUD ticks, 0..7. v137: from RAM 0xC498 (ticks = value // 8; 35/31/23/15 read 4/3/2/1 on the HUD,
    56-63 = full). The HUD tile count is blank during cut-scenes and in parts of the hideout (the live v134 hideout worker
    read 0 HP through the whole corridor, so the planner refused every fight there); the HUD is the fallback only."""
    try:
        v = pyboy.get_memory_value(0xC498)
        if 0 <= v <= 63:
            return max(1, v // 8) if v > 0 else 0      # v139: 1-7 points show no HUD tick but the player is alive - the planner must not treat it as dead (v137 refused every fight at "0 HP" in the hideout corridor)
        tw = pyboy.botsupport_manager().tilemap_window()
        return sum(1 for tx in HEALTH_COLS if tw[tx, 0] == HEALTH_TICK_TILE)
    except Exception:
        return -1


LADDER_TILES = (260, 261, 294, 295)   # v86: 294/295 = the mission-4 cave chain (climbable 4-tile-wide face); no false hits on m1-m3 fixtures
CHAIN_TILES = (345, 346, 233, 234)    # v113: the mission-4 hanging chains (arena exit at x 104-119, pillar chains on the upper walkway). Wall decor in m1-m3 (false ladders in chunks 15/46/66/106), so only with chain=True
STALACTITE_TILES = frozenset({233, 234, 235, 236, 251, 252, 231, 232, 247, 248, 249, 250})   # v153o: the sprite alternates two tile sets frame to frame (odd env steps 235/251/233, even steps 231/247/249) - with one set the radar saw it every other step and read it as a thug in between   # v153f: hideout corridor - a 16x24 stalactite spawns at the ceiling ~8-16 px ahead of the player and falls in ~12 steps (2 ticks on contact); the radar read it as a thug at dy -22..-26
WALL_BLOCK_TILES = frozenset(range(224, 230)) | frozenset(range(240, 243))   # v150: hall lock room 2's moving wall block (untouchable, 8 HP per contact)
PLATFORM_TILES = (253, 254, 255)      # v129: the hideout's moving stepping-stone slabs are SPRITES (3 tiles wide, top at y 88), not tilemap


def _visible_grid(pyboy):
    tm = pyboy.botsupport_manager().tilemap_background()
    scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
    grid = [[tm[(tx + scx // 8) % 32, (ty + scy // 8) % 32] for tx in range(21)] for ty in range(18)]
    return grid, scx, scy


def find_ladders(pyboy, chain: bool = False) -> List[Dict[str, int]]:
    """Ladders (mission 2+): vertical runs of tiles 260/261. Returns [{x, left, right, top_y, bottom_y}] in screen px.
    Climb by standing with your x within ~4px of the ladder centre and HOLDING up."""
    try:
        grid, scx, scy = _visible_grid(pyboy)
    except Exception:
        return []
    fx, fy = scx % 8, scy % 8
    tiles = LADDER_TILES + CHAIN_TILES if chain else LADDER_TILES
    cols = []
    for tx in range(20):                       # visible columns only (col 20 is off-screen)
        rows = [ty for ty in range(18) if grid[ty][tx] in tiles]
        if len(rows) >= 4:
            cols.append((tx, min(rows), max(rows)))
    ladders, run = [], []
    for c in cols + [None]:
        if c is not None and (not run or c[0] == run[-1][0] + 1):
            run.append(c); continue
        if run:
            left = run[0][0] * 8 - fx; right = (run[-1][0] + 1) * 8 - fx
            ladders.append({"x": (left + right) // 2, "left": left, "right": right,
                            "top_y": min(r[1] for r in run) * 8 - fy, "bottom_y": (max(r[2] for r in run) + 1) * 8 - fy})
        run = [c] if c is not None else []
    return ladders


VOID_TILES = {362, 286, 287, 314, 315, 317}
EXTRA_VOID, EXTRA_BELOW = set(), set()      # v144: the runner sets these once the maze past door 2 is entered (hall holes: 354 at the feet row, 362/312/313 below)
RIVER_VOID = {240, 241, 304}                                   # mission-3 river surface and bank edge: void only where water is on screen
RIVER_BELOW = VOID_TILES | RIVER_VOID | {305, 267, 268}        # row under a river edge (305 is also the stepping-stone tile: never void)


LEDGE_TILES = {282, 283}               # v133: hideout spiked-ledge tops


def find_pits(pyboy, feet_y: int, river: bool = False, cave: bool = False, player_x: Optional[int] = None) -> List[Dict[str, int]]:
    """Gaps in the floor the player stands on: columns where the two tile rows just below the player's feet are
    the solid-dark tile 362 (no floor). Returns [{x, left, right}] in screen px."""
    try:
        grid, scx, scy = _visible_grid(pyboy)
    except Exception:
        return []
    fx = scx % 8
    row = min(max((feet_y + scy % 8) // 8, 1), 16)
    # a column is a drop when the feet row shows void (the dark tile, or mission 2's first-floor background decor
    # 286/287/314/315/317 which is just as fatal) and the row below is the dark tile too
    void = VOID_TILES | RIVER_VOID if river else VOID_TILES     # v37: 240/241 also tile the mission-1 boss room floor
    below = RIVER_BELOW
    if cave:                                                    # v123: the mission-4 cave's hanging chain columns are non-solid - the real gap is void + chain (~24 px)
        void = void | set(CHAIN_TILES) | EXTRA_VOID; below = below | set(CHAIN_TILES) | EXTRA_BELOW   # v144: the hall past the sealed room (chunks 97/96) marks its floor holes with 354 over 362/312
    cols = [tx for tx in range(21) if grid[row][tx] in void and grid[min(row + 1, 17)][tx] in below
            and grid[row - 1][tx] not in LADDER_TILES]
    # v133: the hideout's spiked-ledge rows (past door 2): the only floor is the ledge-top tile pair 282/283; the rock
    # texture 312/313 beside it is a drop and 254/255 (+267/268 below) are spikes. When the player stands on a ledge
    # tile, every other column of the feet row is a pit.
    if player_x is not None:
        pcol = min(max((player_x + fx) // 8, 0), 20)
        if grid[row][pcol] in LEDGE_TILES:
            cols = [tx for tx in range(21) if grid[row][tx] not in LEDGE_TILES]
    pits, run = [], []
    for c in cols + [None]:
        if c is not None and (not run or c == run[-1] + 1):
            run.append(c); continue
        if len(run) >= 1:
            pits.append({"x": (run[0] * 8 - fx + (run[-1] + 1) * 8 - fx) // 2, "left": run[0] * 8 - fx, "right": (run[-1] + 1) * 8 - fx})
        run = [c] if c is not None else []
    return pits


WATER_TILES = {326, 340, 341, 342, 356, 357, 230, 244, 245, 246, 293}   # mission 3 river below the forest ledge


def _river_visible(pyboy) -> bool:
    """Mission 3's forest: water tiles anywhere on screen. Its lower bank ends in the river; the way on is the ledge."""
    try:
        grid, _, _ = _visible_grid(pyboy)
        return sum(1 for row in grid for t in row if t in WATER_TILES) >= 10     # true water tiles only (240/241 also floor the boss room)
    except Exception:
        return False


def _ledge_below(pyboy, p: Dict[str, int], river: bool = False) -> bool:
    """True when the ground two/three tile rows under the player (own column and its neighbours) is water or river
    bank edge: pressing down walks off the ledge into the river (v38: 'down' next to the stepping stones killed
    every mission-3 starter twice). Where the row below is grass, stepping down is fine."""
    try:
        grid, scx, scy = _visible_grid(pyboy)
        col = min(max((p["x"] + scx % 8) // 8, 0), 20)
        row = (p["feet_y"] + scy % 8) // 8
        if not (76 <= p["feet_y"] <= 92):
            return False                # only the ledge path itself; platforms above it (feet ~56-64) step down onto it safely
        bad = WATER_TILES | (RIVER_VOID if river else set())
        if scy != 0 and not river:
            # v64: mission 4's bridge (SCY 92-96) has void (362) two rows under the walkway; 'down' past feet ~86 is a
            # fall (every first death there). Same rule as the river ledge, keyed on the void tile instead of water.
            bad = bad | VOID_TILES
        cells = [grid[r][c] for r in (row + 2, row + 3) for c in (col - 1, col, col + 1) if 0 <= r < len(grid) and 0 <= c <= 20]
        return bool(cells) and any(t in bad for t in cells)
    except Exception:
        return False


BELT_SURFACE_TILES = {249, 250, 251, 252, 253, 254, 352, 353, 354}   # conveyor belt surface + roller ends (animated)


def _belt_tiles_in_row(pyboy, p: Dict[str, int]) -> bool:
    """True when the walkway row the player stands on contains conveyor tiles anywhere on screen: the gaps of this
    walkway are the only wide gaps known to be jumpable, and the belt's direction may still be unknown."""
    try:
        grid, scx, scy = _visible_grid(pyboy)
        row = min(max((p["feet_y"] + scy % 8) // 8, 0), len(grid) - 1)
        return any(t in BELT_SURFACE_TILES for t in grid[row])
    except Exception:
        return False


WALKWAY_TILES = {256, 257, 260, 261} | BELT_SURFACE_TILES      # mission 2+ floors: walkway, ladder, conveyor


def find_floor_ends(pyboy, p: Dict[str, int]) -> List[Dict[str, int]]:
    """Mission 2+ only: the walkway is a uniform tile pattern, so scan the feet row outward from the player and treat
    every column that is not walkway/ladder/belt as a drop. find_pits only knows the dark tile 362; the first floor
    of mission 2 ends in decor tiles that are just as fatal (v21/v22: walked or jumped into them)."""
    try:
        grid, scx, scy = _visible_grid(pyboy)
    except Exception:
        return []
    fx = scx % 8
    row = min(max((p["feet_y"] + scy % 8) // 8, 1), 16)
    col = min(max((p["x"] + fx) // 8, 0), 20)
    if grid[row][col] not in WALKWAY_TILES:
        return []                    # not standing on a recognised walkway: no opinion
    ends = []
    for step in (-1, 1):
        c = col + step
        while 0 <= c <= 20 and grid[row][c] in WALKWAY_TILES:
            c += step
        if 0 <= c <= 20:
            left = c * 8 - fx if step == 1 else 0
            right = 160 if step == 1 else (c + 1) * 8 - fx
            ends.append({"x": (left + right) // 2, "left": left, "right": right})
    return ends


def _merge_pits(pits: List[Dict[str, int]], ends: List[Dict[str, int]]) -> List[Dict[str, int]]:
    out = list(pits)
    for e in ends:
        if not any(g["left"] <= e["left"] <= g["right"] or g["left"] <= e["right"] <= g["right"] for g in out):
            out.append(e)
    return out


def find_doors(pyboy) -> List[Dict[str, int]]:
    """Doorways (level exits) from the visible background tilemap. Returns [{x, left, right, width_px}] in
    screen px. Enter one by standing under it at the top of the street and pressing UP."""
    try:
        tm = pyboy.botsupport_manager().tilemap_background()
        scx, scy = pyboy.get_memory_value(0xFF43), pyboy.get_memory_value(0xFF42)
    except Exception:
        return []
    fine_x = scx % 8
    cols_with_door = []
    for tx in range(21):
        col = (tx + scx // 8) % 32
        hit = False
        for ty in DOOR_BOTTOM_ROWS:
            if tm[col, (ty + scy // 8) % 32] == DOOR_TILE:
                hit = True
                break
        if hit:
            cols_with_door.append(tx)
    doors = []
    run: List[int] = []
    for tx in cols_with_door + [None]:
        if tx is not None and (not run or tx == run[-1] + 1):
            run.append(tx)
            continue
        if 3 <= len(run) <= 5:      # real doorways are 4 tiles wide; 2-wide = decor, >5 = dark background regions
            left = run[0] * 8 - fine_x
            right = (run[-1] + 1) * 8 - fine_x
            doors.append({"x": (left + right) // 2, "left": left, "right": right, "width_px": right - left})
        run = [tx] if tx is not None else []
    return doors


def _platforms(cells: List[Tuple[int, int]]) -> List[Dict[str, int]]:
    """v129: cluster slab sprite cells (8 px wide, same y, adjacent x) into [{left, right, y}] in screen px."""
    out: List[Dict[str, int]] = []
    for y in sorted(set(c[1] for c in cells)):
        xs = sorted(set(c[0] for c in cells if c[1] == y))
        run: List[int] = []
        for x in xs + [None]:
            if x is not None and (not run or x - run[-1] <= 9):
                run.append(x); continue
            if run:
                out.append({"left": run[0], "right": run[-1] + 8, "y": y})
            run = [x] if x is not None else []
    return out


def sprite_radar(pyboy, river: bool = None, chain: bool = False, cave: bool = False, stalactites: bool = False) -> Dict[str, Any]:
    bs = pyboy.botsupport_manager()
    player_cells: List[Tuple[int, int]] = []
    player_tiles: List[int] = []
    other_cells: List[Tuple[int, int]] = []
    wall_block_cells: List[Tuple[int, int]] = []
    stal_cells: List[Tuple[int, int]] = []
    platform_cells: List[Tuple[int, int]] = []
    player_flip = None
    for i in range(40):
        s = bs.sprite(i)
        if i in PLAYER_SLOTS and s.tile_identifier in CLIMB_TILES:
            player_tiles.append(s.tile_identifier)
        if not s.on_screen:
            continue
        if i in PLAYER_SLOTS:
            player_cells.append((s.x, s.y))
            if player_flip is None and i == 0:
                player_flip = bool(s.attr_x_flip)   # measured: flip set => facing right, cleared => facing left
        elif s.tile_identifier in PLATFORM_TILES:
            platform_cells.append((s.x, s.y))      # v129: moving slabs - never enemies (they read as 3 reachable thugs with 0 HP)
        elif stalactites and s.tile_identifier in STALACTITE_TILES:
            stal_cells.append((s.x, s.y))          # v153f: falling stalactite - a hazard, not an enemy
        elif s.tile_identifier in WALL_BLOCK_TILES:
            wall_block_cells.append((s.x, s.y))    # v150: the hall's lock room 2 has a moving lattice wall block (3x3 cells of the wall pattern) that hits for 8 HP and cannot be hurt - not an enemy, time the walk past it
        elif s.tile_identifier >= 128:
            other_cells.append((s.x, s.y))
        # else: dropped weapons / items also use low tile ids; they are not the player and not enemies

    out: Dict[str, Any] = {"player": None, "enemies": [], "enemy_count": 0, "any_enemy_in_reach": False, "wall_block": bool(wall_block_cells),
                           "stalactites": [{"x": int(sum(c[0] for c in g) / len(g)) + 4, "bottom": max(c[1] for c in g) + 8, "top": min(c[1] for c in g)} for g in _cluster(stal_cells)],   # v153f
                           "doors": (find_doors(pyboy) if pyboy.get_memory_value(0xFF42) == 0 else []), "health_ticks": read_health(pyboy), "health_max": 7,
                           "scrolled_vertically": pyboy.get_memory_value(0xFF42) != 0,
                           "multi_floor": pyboy.get_memory_value(0xFF42) != 0 or bool(find_ladders(pyboy))}
    if not player_cells:
        out["note"] = "player sprite not on screen (cutscene, death animation or transition)"
        return out
    p = _bbox(player_cells)
    scrolled_vertically = pyboy.get_memory_value(0xFF42) != 0
    ladders_visible = bool(find_ladders(pyboy))       # room 3's top sits at SCY=0, so SCY alone does not mark mission 2
    # the climbing pose uses its own sprite tiles: the only reliable sign that the player is actually on a ladder's
    # rungs rather than standing on a floor the ladder passes through
    climbing = len(player_tiles) >= 3
    out["player"] = {"x": p["x"], "feet_y": p["feet_y"], "facing": "right" if player_flip else "left", "climbing": climbing,
                     "screen_edge": "right" if p["x"] >= 136 else ("left" if p["x"] <= 24 else None)}
    enemies = []
    projectiles = []
    def _split_wide(g):
        xs = sorted(set(c[0] for c in g)); ys = sorted(set(c[1] for c in g))
        if not xs or xs[-1] - xs[0] <= 16 or (ys[-1] - ys[0] + 8) > 34:   # tall = the boss, keep whole
            return [g]
        parts, cur, start = [], [], xs[0]
        for c in sorted(g):
            if c[0] - start > 16:
                parts.append(cur); cur, start = [], c[0]
            cur.append(c)
        parts.append(cur)
        return parts
    groups = [part for g in _cluster(other_cells) for part in _split_wide(g)]
    for g in groups:
        if len(g) < 3:      # 1-2 cell objects: thrown knives / items. Report as projectiles.
            b = _bbox(g)
            dx, dy = b["x"] - p["x"], b["feet_y"] - p["feet_y"]
            projectiles.append({"dx": dx, "dy": dy, "side": "right" if dx > 0 else "left",
                                "on_my_lane": abs(dy) <= 12, "close": abs(dx) <= 48})
            continue
        b = _bbox(g)
        dx, dy = b["x"] - p["x"], b["feet_y"] - p["feet_y"]
        # Unreachable = far above the player while the player already stands at the top of its walkable
        # band (can't go further up). Relative on purpose: mission 1 has stairs, so absolute lane bounds lie.
        at_top = p["feet_y"] <= 94
        reachable = not (at_top and dy < -24)
        if (scrolled_vertically or ladders_visible) and abs(dy) > 40:
            reachable = False           # multi-floor level (mission 2+): a different floor, not a lane you can step onto
        enemies.append({
            "x": b["x"], "feet_y": b["feet_y"], "w": b.get("w", 16),
            # v60: the far edge of the sprite (his back) does not move when he punches; the centre lurches 8-10 px
            "back_x": b["x"] + b.get("w", 16) // 2 if dx > 0 else b["x"] - b.get("w", 16) // 2,
            "dx": dx, "dy": dy,
            "side": "right" if dx > 0 else "left",
            "reachable": reachable,
            "on_platform": not reachable,
            "in_reach": reachable and abs(dx) <= REACH_X and abs(dy) <= REACH_Y,
            "same_lane": reachable and abs(dy) <= REACH_Y,
            "heavy": (b["feet_y"] - b["top_y"]) > 34,      # 5-row sprite: the big bruiser (mission 3+), many hits, grabs
        })
    # reachable first, then by distance; platform enemies are listed but never targeted
    enemies.sort(key=lambda e: (not e["reachable"], abs(e["dx"]) + abs(e["dy"])))
    out["enemies"] = enemies
    out["projectiles"] = projectiles
    out["projectile_threat"] = any(pj["on_my_lane"] and pj["close"] for pj in projectiles)
    out["enemy_count"] = len(enemies)
    out["reachable_enemy_count"] = sum(1 for e in enemies if e["reachable"])
    out["platform_enemy_count"] = sum(1 for e in enemies if not e["reachable"])
    out["any_enemy_in_reach"] = any(e["in_reach"] for e in enemies)
    reach = [e for e in enemies if e["reachable"]]
    if reach:
        n = reach[0]
        out["nearest_enemy_summary"] = (
            f"{abs(n['dx'])}px to the {n['side']}, "
            + ("same lane" if n["same_lane"] else f"{abs(n['dy'])}px {'lower' if n['dy'] > 0 else 'higher'}")
            + (", IN REACH" if n["in_reach"] else "")
        )
    elif enemies:
        out["nearest_enemy_summary"] = f"{len(enemies)} enemy(ies) on a raised platform only: unreachable, keep moving"
    else:
        out["nearest_enemy_summary"] = "no enemies on screen"
    out["ladders"] = find_ladders(pyboy, chain=chain)      # v113: mission-4 chains count as ladders
    for l in out["ladders"]:
        l["dx"] = l["x"] - p["x"]
        l["side"] = "right" if l["dx"] > 0 else "left"
        # ladder is 16px wide; the player centre can sit anywhere on it. While in the climbing pose the game owns the
        # x position (measured 8px off the tile-run centre), so be lenient: losing under_it mid-climb freezes advice.
        l["under_it"] = abs(l["dx"]) <= (12 if climbing else 8)
        # measured: left/right and attacks are ignored until the feet are within 2px of the ladder top row (feet 67 on a top_y=64 ladder is still on the rungs)
        l["at_top"] = l["under_it"] and p["feet_y"] <= l["top_y"] + 2
        l["on_it"] = l["under_it"] and not l["at_top"] and p["feet_y"] <= l["bottom_y"] and climbing
        # a ladder that passes THROUGH the walkway you stand on (foot well below your feet, and you are not in the
        # climbing pose) cannot be climbed up from here. Measured: the right-hand ladder of mission 2's third room
        # only leads down; the left one, whose foot is at your floor, is the way up.
        # v33: at 17px from the decoy the old |dx|<=16 test made it "climbable" again -> left/right cycle (100/100).
        # A ladder whose foot is well below the floor you stand on leads DOWN from here, whatever your x.
        l["through"] = not climbing and l["bottom_y"] - p["feet_y"] > 12
        # climbable from here = the ladder's foot is at/below your floor. A ladder whose top is at your feet is
        # the one you just came up: never navigate back to it.
        l["climbable"] = l["bottom_y"] >= p["feet_y"] - 12 and not l["at_top"] and not (l["top_y"] >= p["feet_y"] - 6 and l["top_y"] <= p["feet_y"] + 6 and not l["on_it"])
        # a ladder whose top row is the top of the screen leads to the NEXT screen: the game only scrolls up (and
        # lets you finish the climb) once every enemy on this screen is dead
        l["off_top"] = l["top_y"] <= 0
    out["on_ladder"] = any(l["on_it"] for l in out["ladders"])
    for l in out["ladders"]:
        if l["through"]:
            l["climbable"] = False
    # the runner knows the mission (start state / progress history) and passes it; the tile heuristic is a fallback
    out["river_level"] = _river_visible(pyboy) if river is None else bool(river)
    out["pits"] = find_pits(pyboy, p["feet_y"], river=out["river_level"], cave=cave, player_x=p["x"])
    out["platforms"] = _platforms(platform_cells)
    on_slabs = [s for s in out["platforms"] if -4 <= p["feet_y"] - s["y"] <= 6]      # slab tops at the player's feet
    if on_slabs:
        # v129: between/after slabs there is nothing (the background tiles under the slab row are uniform rock); the
        # gaps move with the slabs, so they are recomputed every step. Each gap is a pit; past the last slab the floor ends.
        slabs = sorted(on_slabs, key=lambda s: s["left"])
        gaps = []
        if slabs[0]["left"] > 0:
            gaps.append({"x": slabs[0]["left"] // 2, "left": 0, "right": slabs[0]["left"]})
        for a, b in zip(slabs, slabs[1:]):
            if b["left"] - a["right"] >= 4:
                gaps.append({"x": (a["right"] + b["left"]) // 2, "left": a["right"], "right": b["left"]})
        if slabs[-1]["right"] < 160:
            gaps.append({"x": (slabs[-1]["right"] + 160) // 2, "left": slabs[-1]["right"], "right": 160})
        over = [s for s in slabs if s["left"] - 6 <= p["x"] <= s["right"] + 6]
        out["slabs_visible"] = True
        if over:
            out["pits"] = gaps                       # standing on a slab: the gaps are the only pits that matter
            out["on_slab"] = True
        else:
            # on tile ground with slabs in view (the section's first/last slab entering from the screen edge): keep the tile
            # pits and add only slab gaps ahead of the player - v129's first cut replaced everything and a slab at x 150
            # became a 150 px "pit" on the left, which the left-edge guard then pushed the player away from, into the real hole
            out["pits"] = out["pits"] + [g for g in gaps if g["left"] >= p["x"]]
    for g in out["pits"]:
        g["edge"] = g["right"] >= 152 or g["left"] <= 8      # runs into the screen edge: the floor ends, nothing to land on
    out["on_ledge"] = _ledge_below(pyboy, p, river=out["river_level"])
    out["conveyor_zone"] = _belt_tiles_in_row(pyboy, p) and not out["river_level"]   # v52: the stone tops are tile 352 too
    for g in out["pits"]:
        g["dx_edge"] = (g["left"] - p["x"]) if g["left"] > p["x"] else (g["right"] - p["x"])
        g["side"] = "right" if g["left"] > p["x"] else "left"
        g["close"] = abs(g["dx_edge"]) <= 40
    # an enemy on the far side of a pit, or beyond a ladder you still have to climb, is not reachable on foot
    for e in enemies:
        beyond_pit = any(g["side"] == e["side"] and abs(g["dx_edge"]) < abs(e["dx"]) - 8 for g in out["pits"])
        beyond_ladder = any(l.get("climbable") and l["side"] == e["side"] and abs(l["dx"]) < abs(e["dx"]) - 8 for l in out["ladders"])
        if e["reachable"] and (beyond_pit or beyond_ladder):
            e["reachable"] = False; e["in_reach"] = False; e["same_lane"] = False; e["beyond_obstacle"] = True
    out["reachable_enemy_count"] = sum(1 for e in enemies if e["reachable"])
    out["any_enemy_in_reach"] = any(e["in_reach"] for e in enemies)
    for d in out["doors"]:
        d["dx"] = d["x"] - p["x"]
        d["side"] = "right" if d["dx"] > 0 else "left"
        d["under_it"] = d["left"] <= p["x"] <= d["left"] + 16   # verified entry zone: left part of the opening
        d["player_at_top"] = p["feet_y"] <= 94
    return out


DOOR_ADVICE = os.environ.get("DGEMMA_DOOR_ADVICE", "0") == "1"


BOSS_TEXT = os.environ.get("DD_BOSS_TEXT", "hybrid")   # hybrid (v70: v47 dodge rhythm, punch only) | lane | v47

AREA_NOTES = {
    48: ("You are in the MISSION 1 BOSS ROOM. The boss is big, takes many hits, and after each knockdown he lies "
         "off-screen for a moment (the radar shows no enemy). Do not walk away. Tactic that works: when he is in reach, "
         "hit him 2-3 times (a; b hits harder), then step DOWN or UP off his lane to dodge his counter-punch, then line "
         "up again and repeat. Never stand still trading blows on his lane.") if BOSS_TEXT == "v47" else (
        "You are in the MISSION 1 BOSS ROOM. The boss is big, takes many hits, and after each knockdown he lies "
        "off-screen for a moment (the radar shows no enemy). Do not walk away. Tactic that works: get LEVEL with him "
        "(same lane within 2px) 14-22px away, PUNCH him 2-3 times (a only: kicks score points but never knock him "
        "down and he only falls to knockdowns), then step DOWN or UP off his lane to dodge his counter-punch, line up "
        "again and repeat. Never stand still trading blows on his lane. Losing a life heals him completely.") if BOSS_TEXT == "hybrid" else (
        "You are in the MISSION 1 BOSS ROOM. The boss is big, takes many hits, and after each knockdown he lies "
        "off-screen for a moment (the radar shows no enemy). Do not walk away. Your hits ONLY land when you are LEVEL "
        "with him (same lane within 2px) and 14-24px away: line up first (up/down), then punch (a). He only FALLS to "
        "the third punch of an unbroken chain: once a punch lands, keep pressing a every step without moving until he "
        "is down (a step or a kick between punches resets the chain; kicks never knock him down). Losing a life heals "
        "him completely."),
    84: "You are past the first doorway (boss street / warehouse). The warehouse exit is the big dark doorway on "
        "the LEFT: walk into it along the TOP lane (press up first). Walking left along the BOTTOM lane leaves the "
        "room the wrong way.",
}


def suggested_rule(radar: Dict[str, Any], last_dir: str = None, stuck_dir: str = None, door_mode: bool = False,
                   recent_enemy: Dict[str, Any] = None, area_id: int = None, enemies_passive: bool = False,
                   belt_dir: str = None, travel_dir: str = None, futile_attacks: bool = False, boss_hits: int = None) -> str:
    """A one-line reading of the radar, phrased as advice, so the model doesn't have to do the geometry.
    last_dir  = the most recent left/right input (the player faces the way it last walked).
    stuck_dir = a direction that has been pressed repeatedly without the player moving (set by the runner)."""
    if not radar.get("player"):
        return "player not visible: press a to skip/continue"
    if radar.get("airborne"):
        # v39: decisions taken mid-jump saw no pits (the feet row is wrong in the air), so an enemy across the gap
        # looked reachable and the model turned back into the gap it had just cleared. Nothing else applies in the air.
        hold = last_dir or "right"
        return f"you are IN THE AIR (jumping or falling): keep pressing {hold} until you land; do not change direction or attack"
    reach = [e for e in radar["enemies"] if e["reachable"]]
    platform = [e for e in radar["enemies"] if not e["reachable"]]
    prefix = ""
    # never call enemies "inactive" while climbing, or on a screen whose ladder runs off the top: those enemies are the
    # scroll lock and simply wait for you to come down
    edge_clipped = [e for e in radar["enemies"] if e["x"] >= 150 or e["x"] <= 10]
    if enemies_passive and edge_clipped and len(edge_clipped) == len(radar["enemies"]) and radar["player"].get("screen_edge") \
            and not radar.get("on_ladder"):
        # v49 sidecars: 200-decision stalls punching a thug half off the screen edge 3-11 px off your lane
        side = "right" if radar["player"]["screen_edge"] == "right" else "left"; other = "left" if side == "right" else "right"
        return prefix + (f"the enemy is half off the {side} screen edge and not on your lane, so nothing you do reaches it. Walk "
                         f"{other} 30-40px (5-6 presses) so it follows you into the open, then get LEVEL with it (within 2px) and punch (a)")
    if enemies_passive and reach and not edge_clipped and not radar.get("on_ladder") and not any(l.get("off_top") for l in radar.get("ladders", [])) \
            and not any(e.get("heavy") for e in reach):      # the bruiser stands still at the screen edge: not decor
        prefix += (f"NOTE: the {len(reach)} figure(s) near you have not moved or hurt you for a while and do not react to "
                   f"attacks: they are inactive. Ignore them and keep moving. ")
        reach = []
    any_in_reach = any(e["in_reach"] for e in reach)
    if area_id in AREA_NOTES:
        prefix += AREA_NOTES[area_id] + " "
    h = radar.get("health_ticks", -1)
    if 0 <= h <= 2 and reach:
        prefix = (f"LOW HEALTH ({h}/7): one or two more hits cost a life. Do not walk into enemies; keep them at arm's "
                  f"length, step away (left/right) when one closes in, and attack only the instant it is in reach. ")
    if stuck_dir in ("up", "down") and not [e for e in radar["enemies"] if e["reachable"]] \
            and not any(l["under_it"] for l in radar.get("ladders", [])):
        if radar.get("multi_floor"):
            prefix += (f"STUCK: you keep pressing {stuck_dir} but are not moving: there is no ladder here. Stop pressing "
                       f"{stuck_dir}; walk along the walkway (away from any drop) to the next ladder. ")
        else:
            prefix += (f"STUCK: you keep pressing {stuck_dir} but are not moving. If this area has a ladder or stairs you must "
                       f"be lined up with it: take 2-3 steps left or right, then press {stuck_dir} again; if there is no ladder, walk right. ")
    elif stuck_dir in ("left", "right") and not [e for e in radar["enemies"] if e["reachable"]] \
            and not any(l["under_it"] for l in radar.get("ladders", [])):
        if radar.get("multi_floor"):
            other = "left" if stuck_dir == "right" else "right"
            prefix += (f"BLOCKED: pressing {stuck_dir} no longer moves you or the screen. There are NO doorways on this level "
                       f"(the doors you see are decoration): the way on is a LADDER or the walkway in the other direction. "
                       f"Walk {other}; do not hold up unless you are directly under a ladder. ")
        else:
            prefix += (f"BLOCKED: pressing {stuck_dir} no longer moves you or the screen and no enemy is on the street. "
                       f"The sidewalk lane has ended (a building corner): press down 3-4 consecutive steps to drop to the road "
                       f"lane, then walk {stuck_dir} again. If a DOORWAY is listed in the radar, that is the exit instead: "
                       f"stand under it and hold up. ")
    elif stuck_dir:
        prefix += f"STUCK: you keep pressing {stuck_dir} but are not moving (edge of the street or a wall); do NOT press {stuck_dir} again. "
    if radar.get("projectile_threat"):
        pj = next(pj for pj in radar["projectiles"] if pj["on_my_lane"] and pj["close"])
        prefix += (f"a thrown object is flying at you from the {pj['side']} on your lane: step up or down ONCE to dodge it, "
                   f"then keep travelling; do not dodge again and again. ")
    if futile_attacks and reach:
        e = min(reach, key=lambda e: abs(e["dx"]))
        way = travel_dir or ("right" if e["side"] == "right" else "left")
        return prefix + (f"your attacks are NOT landing on the enemy {abs(e['dx'])}px to your {e['side']} (the score has not moved "
                         f"in many attacks): it stands where you cannot hit it. Stop attacking and walk {way} past it; "
                         f"if it blocks you, keep walking {way} anyway")
    facing = radar["player"].get("facing") or last_dir
    if radar.get("on_ledge") and any(e.get("heavy") for e in reach):
        prefix += ("LEDGE: you are on a narrow path above water and the HEAVY thug stands ON it. Lining up with his lane "
                   "(a few px up or down) is safe; never walk below his lane. ")
    elif radar.get("on_ledge"):
        # v51 (emulator, 5 forest states x 3 timings): walking on past thugs cost 3-9 health per 220 steps; turning to
        # fight every LEVEL thug within 28px, including those behind, cost 0-1.7 and landed 11-25 punches
        near_gap = any(abs(g["dx_edge"]) <= 40 for g in radar.get("pits", []))
        prefix += ("LEDGE: you are on a narrow path above a drop (water or a gap) with no lane below you: stay on it and move "
                   "RIGHT along it (down is disabled here). ")
        if not near_gap:
            prefix += ("Do NOT walk on with a thug behind you: a thug level with you within 28px behind hits you in the back; "
                       "turn (press left), get level, and punch it from 12-24px, then continue right. ")
    boss = [e for e in reach if area_id == 48 or e.get("heavy")]
    if boss and not radar.get("on_ladder"):
        # v49, emulator + 14k boss-room steps: hits on him depend on LANE alignment, not range. 6 px off his lane
        # nothing lands (0/286); level with him (within 2 px) punches land 18-41% at 16-20 px, kicks up to 50%.
        # Under 12 px is his grab range (5-8% hurt per step vs 2-4% at 16-24). Scripted: m1 boss 3/3, bruiser 2/3.
        e = min(boss, key=lambda e: abs(e["dx"]))
        away = "left" if e["side"] == "right" else "right"
        who = "BOSS" if area_id == 48 else "HEAVY thug"
        d = abs(e["dx"]); dy = e["dy"]
        if area_id == 48 and BOSS_TEXT == "hybrid":
            if not e["same_lane"]:
                return prefix + (f"BOSS: he is {abs(dy)}px {'lower' if dy > 0 else 'higher'}; press "
                                 f"{'down' if dy > 0 else 'up'} to line up while staying 14-22px away horizontally")
            if d <= 12:
                return prefix + (f"BOSS is {d}px away: TOO CLOSE, his grab range. Step {away} ONE step now, do not attack")
            if d <= 22:
                return prefix + (f"BOSS is {d}px to your {e['side']}: YOUR range. Punch (a) now and again next step; after 2-3 "
                                 f"punches step DOWN or UP off his lane to dodge his counter, then line up and punch again. "
                                 f"Never kick: kicks do not knock him down")
            if d <= 34:
                return prefix + (f"BOSS is {d}px to your {e['side']}: just out of range. Take ONE step {e['side']} toward him, "
                                 f"then punch (a) when he is within 22px")
            return prefix + (f"BOSS is {d}px to your {e['side']}. Walk {e['side']} toward him until he is about 20px away, then "
                             f"punch (a). Never linger under 12px from him")
        if area_id == 48 and BOSS_TEXT == "v47":
            # v63: the v47 wording verbatim (mined v19-v22 rates). v67: switchable; the lane-first text below won the two
            # low-latency local fights (v62 code) where this text lost two.
            if not e["same_lane"]:
                return prefix + (f"BOSS: he is {abs(dy)}px {'lower' if dy > 0 else 'higher'}; press "
                                 f"{'down' if dy > 0 else 'up'} to line up while staying 13-22px away horizontally")
            if d <= 12:
                return prefix + (f"BOSS is {d}px away: TOO CLOSE, this is his grab range (attacks from here: 5% hit, 25% hurt). "
                                 f"Step {away} ONE step now, do not attack")
            if d <= 22:
                return prefix + (f"BOSS is {d}px to your {e['side']}: this is YOUR range. Attack now: b (kick) landed 31% of the time in "
                                 f"recorded fights, a 21%; mix them, and step back if he gets under 12px")
            if d <= 34:
                return prefix + (f"BOSS is {d}px to your {e['side']}: just out of range. Do NOT punch (4% hit, 11% hurt). Take ONE "
                                 f"step {e['side']} toward him, then kick with b when he is within 22px")
            return prefix + (f"BOSS is {d}px to your {e['side']}. Walk {e['side']} toward him until he is about 20px away, then kick (b). "
                             f"He hits hardest when you stand under 12px from him: never linger there")
        if radar.get("bruiser_hp") is not None:
            # v78: his HP byte (0xC595) read live; fresh he has 42. Jab 2, kick 3, the knockdown blow 5.
            prefix += f"The bruiser has {radar['bruiser_hp']}/42 health left (each punch 2, the knockdown blow 5). "
        elif boss_hits is not None:
            # measured: he goes down after exactly 20 hits and recovers ALL of them when you lose a life
            prefix += f"{who} needs 20 hits in ONE life (losing a life heals him fully); hits this life: {boss_hits}/20. "
        if abs(dy) > 2:
            return prefix + (f"{who} is {abs(dy)}px {'lower' if dy > 0 else 'higher'} than you: your hits ONLY land when you are "
                             f"LEVEL with him (within 2px). Press {'down' if dy > 0 else 'up'} to line up"
                             + (" (stay on the path: he is on it too)" if radar.get("on_ledge") else "")
                             + f", keeping 14-24px away horizontally. Do not attack and do not walk into him until level")
        if d < 12:
            return prefix + (f"{who} is {d}px away and level with you: TOO CLOSE, this is his grab range. Step {away} ONE step "
                             f"now (do not attack from here), then punch from 14-24px")
        if d <= 24:
            return prefix + (f"{who} is {d}px to your {e['side']} and LEVEL with you: this is YOUR range. Face him and punch (a) "
                             f"every step; b also lands. If he drifts off your lane, line up again before attacking. "
                             f"Do not walk into him and do not back off"
                             + (" (you are at the screen edge: that is fine, he is in range right here)" if radar["player"].get("screen_edge") else ""))
        if d <= 34:
            return prefix + (f"{who} is {d}px to your {e['side']}: just out of range. Do NOT punch from here. Take ONE step "
                             f"{e['side']} toward him, then punch (a) when he is within 24px and still level with you")
        return prefix + (f"{who} is {d}px to your {e['side']}. Walk {e['side']} toward him until he is about 20px away and level "
                         f"with you, then punch. He hits hardest when you stand under 12px from him: never linger there")
    locked = [l for l in radar.get("ladders", []) if l["on_it"] and l.get("off_top")]
    if locked and radar["enemies"] and all(e["dy"] < -40 for e in radar["enemies"]):
        return prefix + ("you are on a ladder and the only enemy is on a floor ABOVE you: this ladder does not lead there. "
                         "Press up to get back to the walkway you came from, then walk to the other ladder")
    if locked and [e for e in radar["enemies"] if e["dy"] >= -40]:
        l = locked[0]
        p_feet = radar["player"]["feet_y"]
        below = [e for e in radar["enemies"] if e["dy"] > 8]
        n_down = max(1, (l["bottom_y"] + 1 - p_feet + 2) // 3)
        pit_side = {g["side"] for g in radar.get("pits", []) if g["close"]}
        pit_note = (" There is a PIT just to your " + "/".join(sorted(pit_side)) + " of the ladder foot: never walk that way and never "
                    "press down once you are standing on the floor.") if pit_side else ""
        if True:   # any live enemy while you hang on an off-top ladder: the climb cannot finish, get down and fight
            level_here = [e for e in radar["enemies"] if abs(e["dy"]) <= 4]
            if level_here and min(level_here, key=lambda e: abs(e["dx"]))["dy"] <= 0:
                return prefix + (f"SCREEN LOCKED: a thug is LEVEL with you {abs(level_here[0]['dx'])}px to your {level_here[0]['side']} on this upper walkway. "
                                 f"Press up a few steps to step off the ladder onto the walkway, then punch (a) him there. "
                                 f"Do not climb down while he is up here")
            return prefix + (f"SCREEN LOCKED: the screen will not scroll and you cannot finish this climb while enemies are alive "
                             f"({len(radar['enemies'])} on this screen, {(str(abs(below[0]['dy'])) + 'px below you') if below else 'waiting at the ladder foot'}). Climb DOWN: "
                             f"press down about {n_down} more steps until your feet are on the floor at the ladder's foot (y={l['bottom_y'] + 1}), "
                             f"then punch (a) the thugs as they come to you. Do NOT walk left/right through them: fight from the ladder foot.{pit_note}")
    if radar["player"].get("climbing") and not radar.get("on_ladder") and reach and \
            not any(abs(g["dx_edge"]) <= 40 for g in radar.get("pits", [])) and \
            any(abs(l["dx"]) <= 16 and radar["player"]["feet_y"] >= l["bottom_y"] - 4 for l in radar.get("ladders", [])):
        return prefix + ("you are still HOLDING the ladder at its foot (attacks and left/right are ignored in this pose): press "
                         "down ONCE to let go, then attack the enemy next to you")
    if radar.get("on_ladder") and reach and not any(l.get("off_top") for l in radar.get("ladders", []) if l["on_it"]):
        # on the rungs only up/down register. Go to whichever end of the ladder is the enemy's floor, then fight.
        l = next(l for l in radar["ladders"] if l["on_it"])
        p_feet = radar["player"]["feet_y"]
        e = min(reach, key=lambda e: abs(e["dx"]) + abs(e["dy"]))
        enemy_floor = p_feet + e["dy"]
        if abs(enemy_floor - l["top_y"]) <= abs(enemy_floor - (l["bottom_y"] + 1)):
            n = max(1, (p_feet - l["top_y"] + 2) // 3)
            return prefix + (f"you are on the ladder rungs (left/right and attacks are ignored here) and an enemy is on the platform "
                             f"above: press up {n} more step(s) to finish the climb, then turn to it and punch")
        n = max(1, (l["bottom_y"] + 1 - p_feet + 2) // 3)
        return prefix + (f"you are on the ladder rungs and CANNOT attack or step sideways from here: press down {n} more step(s) "
                         f"until your feet are on the floor (y={l['bottom_y'] + 1}), then press a to punch the thug next to you")
    hole = [g for g in radar.get("pits", []) if abs(g["dx_edge"]) <= 32 and radar.get("multi_floor") and not radar.get("river_level")]
    if hole and reach and any(e["side"] != hole[0]["side"] and abs(e["dx"]) <= 40 for e in reach):
        g = hole[0]
        return prefix + (f"a HOLE in the walkway is {abs(g['dx_edge'])}px to your {g['side']} and a thug is on your other side: "
                         f"its throw carries you up to 30px backwards, into the hole. Walk {'left' if g['side'] == 'right' else 'right'} "
                         f"toward the thug to get 36px clear of the hole BEFORE attacking")
    if any_in_reach:
        e = next(e for e in reach if e["in_reach"])
        if abs(e["dy"]) > 2 and not (radar.get("on_ledge") and e["dy"] > 0 and radar["player"]["feet_y"] + e["dy"] > 92):
            # v50 (80k steps): hits land 16-38% when level within 2px, 1-2% when 3-10px off; the model punched air 21k times
            return prefix + (f"enemy {abs(e['dx'])}px to your {e['side']} but {abs(e['dy'])}px {'lower' if e['dy'] > 0 else 'higher'}: "
                             f"hits ONLY land when you are LEVEL with it (within 2px). Press {'down' if e['dy'] > 0 else 'up'} to line up, "
                             f"then punch (a). Do not attack until level")
        if facing and facing != e["side"]:
            return prefix + f"enemy in reach on your {e['side']} but you are facing {facing}: press {e['side']} once to turn, then punch (a)"
        if radar.get("on_ledge"):
            # mined from v46 (54k forest steps): a thug on your LEFT within 12px hurts 6% per step whether you attack or turn
            # into it; one step right (0% hurt) and kicking at 13-22px (1.4% hurt, 7% hit) is the safe pattern. Thugs on
            # the right cost ~0.1% per step at any range.
            if e["side"] == "left" and abs(e["dx"]) < 10 and radar["player"].get("screen_edge") != "right":
                return prefix + (f"thug {abs(e['dx'])}px BEHIND you (left) on the LEDGE, inside grab range: step right ONCE to open "
                                 f"12-24px, then face it (press left) and punch (a) every step")
            return prefix + (f"enemy in reach on your {e['side']} on this LEDGE and level with you: face it and punch NOW (a), keep "
                             f"punching every step; step back one step if it gets under 10px")
        return prefix + (f"enemy in reach on your {e['side']}, level with you and you face it: punch NOW (a) and keep punching every "
                         f"step while it stays level; step back one step if it gets under 10px")
    ladders = radar.get("ladders", [])
    # a pit beyond a climbable ladder on the same side is not in your way: the ladder comes first
    def _shielded(g):
        return any(l.get("climbable") and (abs(l["dx"]) <= 10 or (l["side"] == g["side"] and abs(l["dx"]) < abs(g["dx_edge"])))
                   for l in ladders)
    # a pit that runs into the screen edge on a horizontally scrolling level is just the camera boundary (v38: two
    # mission-3 stalls pacing at "floor ENDS 36px right" / "walk right"); on multi-floor walkways it is a real end
    pits = [g for g in radar.get("pits", []) if g["close"] and not _shielded(g)
            and not (g.get("edge") and not radar.get("multi_floor"))]
    if pits and not any_in_reach and (belt_dir or radar.get("conveyor_zone")):
        g = min(pits, key=lambda g: abs(g["dx_edge"]))
        way = travel_dir or belt_dir or "left"
        prefix += (f"a GAP in the moving walkway is {abs(g['dx_edge'])}px to your {g['side']}: on this conveyor walkway gaps are "
                   f"jumped for you automatically when you reach the edge. Do NOT turn back and never press jump yourself: keep "
                   f"pressing {way} every step. ")
    elif pits and not any_in_reach and len({g["side"] for g in pits}) == 2 and all(abs(g["dx_edge"]) <= 32 for g in pits):
        L = min((g for g in pits if g["side"] == "left"), key=lambda g: abs(g["dx_edge"]))
        R = min((g for g in pits if g["side"] == "right"), key=lambda g: abs(g["dx_edge"]))
        way = travel_dir or "right"
        prefix += (f"you are on a SMALL ISLAND: a drop {abs(L['dx_edge'])}px to your LEFT and the next gap {abs(R['dx_edge'])}px to your "
                   f"RIGHT. Only move {way}; the jump is automatic, and one step the wrong way is fatal. ")
    elif pits and not any_in_reach and radar.get("river_level") and radar["player"]["feet_y"] >= 92 \
            and any(g["side"] == "right" and (g["right"] - g["left"] >= 16 or g.get("edge")) for g in pits):
        g = min((g for g in pits if g["side"] == "right"), key=lambda g: abs(g["dx_edge"]))
        return prefix + (f"the lower bank ENDS in the river {abs(g['dx_edge'])}px to your right. Do not walk into it: press UP "
                         f"3-4 times to climb onto the ledge path above (your feet at y~84), then continue right along it")
    elif pits and not any_in_reach:
        g = min(pits, key=lambda g: abs(g["dx_edge"]))
        if g["right"] - g["left"] < 16 and not g.get("edge"):
            prefix += (f"a narrow GAP in the path is {abs(g['dx_edge'])}px to your {g['side']}: it is jumped for you automatically "
                       f"when you reach its edge, so just keep walking {g['side']} (never press jump yourself). ")
        else:
            prefix += (f"DANGER: the floor ENDS {abs(g['dx_edge'])}px to your {g['side']} (a pit: falling costs a life). "
                       f"Do NOT walk {g['side']}. ")
    # v34: at the top of the screen with a thug alive on the floor below, the camera will not scroll up (measured:
    # kill him, climb again, the screen follows and the next floor appears). The ladder top itself is a dead end.
    below_far = [e for e in radar["enemies"] if e["dy"] > 40]
    if radar["player"]["feet_y"] <= 20 and below_far and ladders:
        l = min(ladders, key=lambda l: abs(l["dx"]))
        where = "you are on it: " if abs(l["dx"]) <= 8 else f"walk {l['side']} to the ladder at x={l['x']} ({abs(l['dx'])}px), then "
        return prefix + (f"SCREEN LOCKED: the screen cannot scroll up while the thug {abs(below_far[0]['dy'])}px BELOW you is alive; "
                         f"this top ledge is a dead end. {where}hold down until you are standing on his floor, punch him, then "
                         f"climb up again and the next floor will appear")
    at_top = [l for l in ladders if l["at_top"]]
    if not at_top and stuck_dir == "up":
        # v30: at the top of room 3's right ladder the feet settle 2-4px below the top row; up does nothing but the
        # on-rungs rule kept asking for it (135 presses). Stuck pressing up that close to the top = at the top.
        at_top = [l for l in ladders if l["on_it"] and radar["player"]["feet_y"] <= l["top_y"] + 6]
    if at_top and not reach:
        if travel_dir:
            return prefix + (f"you are at the TOP of the ladder. Up does nothing here and never go back down. Walk {travel_dir} "
                             f"along the platform to find the next ladder (never toward a pit)")
        pit_sides = {g["side"] for g in pits}
        # v26: at the conveyor room's ladder top the camera sits so that the drop on the right is off-screen and the
        # player is pressed against the right screen edge: "walk right" then means 200 wasted presses. A screen edge
        # counts as a blocked side just like a visible pit.
        if radar["player"].get("screen_edge"):
            pit_sides.add(radar["player"]["screen_edge"])
        if radar.get("multi_floor") and radar["player"]["x"] >= 128:
            pit_sides.add("right")          # nothing to the right but the screen edge on these walkways
        safe = "right" if "right" not in pit_sides else ("left" if "left" not in pit_sides else "right")
        return prefix + (f"you are at the TOP of the ladder you just climbed. Up does nothing here and never go back down. "
                         f"The platform continues to the {safe}: walk {safe} (never toward a pit) to find the next ladder")
    climbable = [l for l in ladders if l["climbable"] or l["on_it"]]
    if climbable and not reach:
        l = min(climbable, key=lambda l: abs(l["dx"]))
        if l["on_it"]:
            more = max(1, (p_feet - l["top_y"] + 3) // 3) if (p_feet := radar["player"]["feet_y"]) else 1
            return prefix + (f"you are on the ladder ({p_feet - l['top_y']}px below its top): keep pressing up EVERY step, about {more} more "
                             f"steps, until you are fully on the top platform; left/right are ignored while you are on the rungs")
        if l["under_it"] and abs(l["dx"]) <= 5:
            return prefix + "you are at the foot of the ladder: HOLD up (press up on consecutive steps) to climb"
        if l["under_it"]:
            # v107: the climb only registers within ~5 px of the ladder's axis (measured: x=40 vs ladder 32 never
            # climbed, x=37 did). "HOLD up" from 8 px off held true starts for 3000 steps in mission-2 area 16.
            return prefix + (f"you are {abs(l['dx'])}px off the ladder's axis: step {l['side']} ONE step to line up directly under it, "
                             f"then hold up to climb (up does nothing from here)")
        return prefix + (f"a LADDER is {abs(l['dx'])}px to your {l['side']} (x={l['x']}): walk {l['side']} until you are directly under it "
                         f"(within 4px), then hold up to climb. This is how you progress in this level")
    if not reach and recent_enemy and recent_enemy.get("steps_ago", 99) <= 8:
        side = recent_enemy.get("side", "left")
        return prefix + (f"no enemy on the radar right now, but one was {recent_enemy['steps_ago']} steps ago on your {side}: "
                         f"it is knocked down or just off-screen and WILL come back. Do not walk off: face {side} (tap {side} once) "
                         f"and wait, attacking the moment it is in reach")
    if reach:
        e = reach[0]
        if not e["same_lane"]:
            want = "down" if e["dy"] > 0 else "up"
            if stuck_dir == want:
                return prefix + f"nearest enemy is {abs(e['dy'])}px {'lower' if e['dy'] > 0 else 'higher'} but you cannot go further {want}: walk {e['side']} toward it and let it come to you, attack when in reach"
            return prefix + f"nearest enemy is {abs(e['dy'])}px {'lower' if e['dy'] > 0 else 'higher'}: press {want} to line up"
        return prefix + f"nearest enemy {abs(e['dx'])}px to the {e['side']}: walk {e['side']} toward it"
    if platform:
        return prefix + f"{len(platform)} enemy(ies) stand on a raised platform you cannot reach (they may throw things): ignore them, walk right, and only attack enemies that are on the street"
    # Doorway = level exit. Verified: stand under it (player x in [left, right-8]) and HOLD up ~6 steps.
    # Only advised when the street is clear AND the player is pinned at the right edge or stuck, which is
    # exactly the "walked past the exit" situation; this avoids chasing dark decor elsewhere.
    pinned = bool(stuck_dir) or DOOR_ADVICE or door_mode
    doors = radar.get("doors") if pinned else []
    if doors:
        d = min(doors, key=lambda d: abs(d["dx"]))
        target = d["left"] + 6
        health_note = prefix if prefix.startswith("LOW HEALTH") else ""
        if d["under_it"]:
            return health_note + ("EXIT: you are under the doorway. HOLD up: press up on 6 consecutive steps with no other input. "
                                  "If the screen has not changed after that, walk into the opening along the TOP lane (up, then "
                                  f"{d['side'] if d['dx'] else 'left'} a few steps)")
        lane = "first press up to get to the TOP lane, then " if radar["player"]["feet_y"] > 96 else ""
        if d["side"] == "left":
            return health_note + (f"EXIT: pressing right does nothing here; the exit is the DOORWAY {abs(d['dx'])}px to your LEFT. {lane}"
                                  f"walk left until your x is about {target} (between {d['left']} and {d['left'] + 14}), then hold up for 6 consecutive steps")
        return health_note + (f"EXIT: the doorway is {abs(d['dx'])}px to your right: {lane}walk right and STOP with x about {target}, then hold up for 6 consecutive steps")
    # v24: 40/40 stalls were a limit cycle at the conveyor landing: "walk right" beyond 60px from the drop, "walk left"
    # inside it, so the model paced between x=81 and x=87 forever. Any visible floor edge on the right decides it.
    if travel_dir:
        return prefix + f"screen clear: keep walking {travel_dir.upper()} (that is the way on along this walkway; do not turn back)"
    if radar.get("river_level") and radar["player"]["feet_y"] >= 92 and any(g["side"] == "right" for g in radar.get("pits", [])):
        return prefix + ("the bank ends in the river ahead: press UP 3-4 times to climb onto the ledge path above (feet y~84) "
                         "and continue RIGHT along it. Never turn back left here")
    if radar.get("multi_floor") and any(g.get("edge") and g["side"] == "right" for g in radar.get("pits", [])):
        if radar["player"]["x"] <= 16 or stuck_dir == "left":
            # v107: mission-2 area 5 - blocked at the left edge, this rule still said "LEFT" (3000 presses into the
            # wall); walking right there reaches area 106 before the drop
            return prefix + "you are against the LEFT wall: the way on is RIGHT. Walk right along this walkway (the drop is beyond the exit)"
        return prefix + "screen clear, but the floor ENDS on the right (a drop): the way on is LEFT. Keep walking left"
    # v27: one step left of a ladder top the ladder rule stops applying and this default said "right" again (106 right /
    # 94 left per 200 decisions). On vertically-scrolled walkways the camera pins a right-walking player near x=108, so
    # being far right of that means the way on is left.
    if radar.get("multi_floor") and (radar["player"]["x"] >= 120 or radar["player"].get("screen_edge") == "right"):
        return prefix + "screen clear: you are at the right end of this walkway, the way on is LEFT. Keep walking left"
    return prefix + ("screen clear: walk right to advance" if stuck_dir != "right" else
                     "screen clear but you cannot move right: try up or down to find the path, or left briefly")


class ScrollTracker:
    """Cumulative scroll from the LCD registers: SCX (0xFF43) horizontal, SCY (0xFF42) vertical. The stage
    scrolls only after the on-screen enemies are cleared, so distance is a direct measure of progress.
    Registers wrap at 256; horizontal deltas count in both directions (the game never scrolls back, but mission 2's
    conveyor walkway runs LEFT), vertical counts screen-up only (decreasing SCY)."""

    SCX = 0xFF43
    SCY = 0xFF42

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.last_x = pyboy.get_memory_value(self.SCX)
        self.last_y = pyboy.get_memory_value(self.SCY)
        self.distance_px = 0          # horizontal progress
        self.vertical_px = 0          # vertical progress (screen scrolled up)
        self.max_distance_px = 0

    def update(self) -> int:
        cx = self.pyboy.get_memory_value(self.SCX)
        d = (cx - self.last_x) % 256
        if 0 < d < 128:
            self.distance_px += d
        elif 128 < d < 256:
            self.distance_px += 256 - d       # mission 2's conveyor walkway scrolls LEFT: that is progress too
        self.last_x = cx
        cy = self.pyboy.get_memory_value(self.SCY)
        dy = (self.last_y - cy) % 256      # screen moving up => SCY decreases
        if 0 < dy < 128:
            self.vertical_px += dy
        elif 128 < dy < 256:
            self.vertical_px += 256 - dy      # mission 2 also descends (elevator / lower walkway): count it
        self.last_y = cy
        self.max_distance_px = max(self.max_distance_px, self.distance_px)
        return self.distance_px

    @property
    def total_px(self) -> int:
        return self.distance_px + self.vertical_px
