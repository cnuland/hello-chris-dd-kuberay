"""Scripted mission-2 explorer that follows the same rules radar.suggested_rule encodes. Maps areas, reports
stalls/deaths, saves a state at the first stall. Usage: python tests/m2_explorer.py <state> [iterations]"""
import sys, collections, hashlib
from pyboy import PyBoy
from pyboy.utils import WindowEvent
sys.path.insert(0, ".")
from radar import sprite_radar, ScrollTracker, find_doors
from PIL import Image
P = {"right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT), "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
     "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP), "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN), "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A)}
state = sys.argv[1]; iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
pb = PyBoy("ignored/dd.gb", window_type="headless", debugging=False, disable_input=False); pb.set_emulation_speed(0)
pb.load_state(open(state, "rb")); [pb.tick() for _ in range(2)]
bs = pb.botsupport_manager(); sc = ScrollTracker(pb)
def hold(k, n): pb.send_input(P[k][0]); [pb.tick() for _ in range(n)]; pb.send_input(P[k][1]); sc.update()
def bg(): tm = bs.tilemap_background(); scx, scy = pb.get_memory_value(0xFF43), pb.get_memory_value(0xFF42); return hashlib.md5(bytes(tm[(tx + scx // 8) % 32, (ty + scy // 8) % 32] % 256 for ty in range(4, 13) for tx in range(20))).hexdigest()[:6]
areas = []; recent = collections.deque(maxlen=120); L = pb.get_memory_value(0xC499); last = None; shots = 0
for k in range(iters):
    r = sprite_radar(pb)
    if not r["player"]: [pb.tick() for _ in range(30)]; sc.update(); continue
    a = pb.get_memory_value(0xE110)
    if not areas or areas[-1] != a:
        areas.append(a); print(f"k={k} AREA {a} dist={sc.distance_px} vert={sc.vertical_px} player={(r['player']['x'], r['player']['feet_y'])} lives={pb.get_memory_value(0xC499)} bg={bg()}", flush=True)
    reach = [e for e in r["enemies"] if e["reachable"]]; pits = [g for g in r["pits"] if g["close"]]
    at_top = [l for l in r["ladders"] if l["at_top"]]; climb = [l for l in r["ladders"] if l["climbable"] or l["on_it"]]
    def blocked(side):
        return any(g["side"] == side and not any(l["climbable"] and l["side"] == side and abs(l["dx"]) < abs(g["dx_edge"]) for l in r["ladders"]) for g in pits)
    locked = [l for l in r["ladders"] if l["on_it"] and l.get("off_top")]
    if r["enemies"] and locked:
        hold("down", 8); last = "down"          # screen lock: get back on the floor and clear the room first
    elif r.get("on_ladder") and reach:
        l = next(l for l in r["ladders"] if l["on_it"]); e = min(reach, key=lambda e: abs(e["dx"]) + abs(e["dy"]))
        ef = r["player"]["feet_y"] + e["dy"]
        d = "up" if abs(ef - l["top_y"]) <= abs(ef - (l["bottom_y"] + 1)) else "down"
        hold(d, 8); last = d                    # only up/down register on the rungs: go to the enemy's floor
    elif reach:
        e = reach[0]
        if e["in_reach"]:
            if r["player"]["facing"] != e["side"]: hold(e["side"], 2)
            hold("a", 6); [pb.tick() for _ in range(6)]; last = "a"
        else:
            if not blocked(e["side"]): hold(e["side"], 8); last = e["side"]
            if e["dy"] > 10 and not any(g["close"] for g in r["pits"]): hold("down", 8)
            elif e["dy"] < -10: hold("up", 8)
    elif at_top:
        safe = "right" if not blocked("right") else "left"; hold(safe, 8); last = safe
    elif climb:
        l = min(climb, key=lambda l: abs(l["dx"]))
        if l["on_it"] or l["under_it"]: hold("up", 8); last = "up"
        elif not blocked(l["side"]): hold(l["side"], 8); last = l["side"]
        else: hold("up", 8); last = "up"
    else:
        if not blocked("right"): hold("right", 16); last = "right"
        elif not blocked("left"): hold("left", 16); last = "left"
        else: hold("up", 8); last = "up"
    recent.append((a, sc.distance_px, sc.vertical_px))
    if k < int(__import__("os").environ.get("M2X_VERBOSE", "0")):
        print(f"  k={k} p={(r['player']['x'], r['player']['feet_y'], r['player']['facing'])} last={last} reach={[(e['dx'], e['dy'], e['in_reach']) for e in reach]} on_ladder={r.get('on_ladder')} at_top={bool(at_top)} climb={[(l['x'], l['dx']) for l in climb]}", flush=True)
    if pb.get_memory_value(0xC499) != L:
        print(f"k={k} LIFE LOST area={a} dist={sc.distance_px} vert={sc.vertical_px} player={(r['player']['x'], r['player']['feet_y'])} last={last} pits={[(g['x'], g['dx_edge']) for g in r['pits']]} ladders={[(l['x'], l['dx']) for l in r['ladders']]}", flush=True); L = pb.get_memory_value(0xC499)
    if k % 250 == 0 and shots < 6: Image.fromarray(bs.screen().screen_ndarray()).resize((320, 288), Image.NEAREST).save(f"/tmp/m2x_{k:04d}.png"); shots += 1
    if len(recent) == 120 and len(set(recent)) == 1 and not reach:
        print(f"k={k} STALL area={a} dist={sc.distance_px} vert={sc.vertical_px} player={r['player']} last={last} ladders={[(l['x'], l['dx'], l['top_y'], l['bottom_y'], l['at_top'], l['climbable'], l['on_it']) for l in r['ladders']]} pits={[(g['x'], g['dx_edge'], g['close']) for g in r['pits']]} doors={find_doors(pb)} bg={bg()}", flush=True)
        Image.fromarray(bs.screen().screen_ndarray()).resize((320, 288), Image.NEAREST).save("/tmp/m2x_stall.png")
        with open("ignored/m2x-stall.state", "wb") as f: pb.save_state(f)
        break
    if pb.get_memory_value(0xC499) == 0: print("DEAD"); break
with open("ignored/m2x-end.state", "wb") as f: pb.save_state(f)
print("END areas", areas, "dist", sc.distance_px, "vert", sc.vertical_px, "lives", pb.get_memory_value(0xC499)); pb.stop()
