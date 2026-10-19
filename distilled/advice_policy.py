"""Rule-based stand-in for the decision model (v154r).

When the DiffusionGemma service does not answer, DecisionModelClient used to return a RANDOM legal action. Between
planner plans that random walk is what killed every "planner-only" run on the mission-2 conveyor belts (fullgamevid5/6/7:
lives lost at full HP, falling through belt gaps at chunk 66/26 SCY 152). The runner already writes the same
natural-language instruction the model reads (`radar_advice` from radar.suggested_rule + reflex.belt_note), and the
model mostly follows it - so the fallback follows it too: take the first NON-negated imperative and map it to a button.
"""
import re
from typing import Optional

ACTIONS = ("down", "left", "right", "up", "a", "b", "kick_left", "kick_right", "jump_kick")

# imperative phrase -> action; {d} = left/right/up/down, (b)/(a) attack buttons
_PATTERNS = [
    (re.compile(r"(?<!you )keep pressing (left|right|up|down|a|b)\b"), lambda m: m.group(1)),
    (re.compile(r"\bpress (?:the )?(left|right|up|down|a|b)\b"), lambda m: m.group(1)),
    (re.compile(r"\bpress (down|up) \d"), lambda m: m.group(1)),
    (re.compile(r"\bturn \(press (left|right)\)"), lambda m: m.group(1)),
    (re.compile(r"\b(?:walk|step|move|go|head) (?:ONE step |one step |\d+(?:-\d+)?px |)?(left|right|up|down)\b", re.I), lambda m: m.group(1).lower()),
    (re.compile(r"\bstep (DOWN|UP)\b"), lambda m: m.group(1).lower()),
    (re.compile(r"\bpunch\b(?: \(a\))?"), lambda m: "a"),
    (re.compile(r"\bkick (?:with )?\(?b\)?"), lambda m: "b"),
    (re.compile(r"\battack\b"), lambda m: "a"),
    (re.compile(r"\bjump[- ]kick\b"), lambda m: "jump_kick"),
]
_NEGATION = re.compile(r"(?:\bnot\b|\bnever\b|\bstop\b|\bdon'?t\b|\bno longer\b|\bavoid\b|\bwithout\b)[^.;:]{0,24}$", re.I)


def advice_to_action(advice: str, travel_dir: Optional[str] = None, last_dir: Optional[str] = None) -> str:
    """Map the runner's radar advice to one action name from ACTIONS."""
    text = advice or ""
    best = None
    for pat, fn in _PATTERNS:
        for m in pat.finditer(text):
            if _NEGATION.search(text[max(0, m.start() - 40):m.start()]):
                continue                                  # "do NOT press kick", "Stop pressing right", "never walk below"
            act = fn(m)
            if act in ACTIONS and (best is None or m.start() < best[0]):
                best = (m.start(), act)
            break                                         # first non-negated hit of this pattern is enough
    if best:
        return best[1]
    if "CONVEYOR" in text or not text:
        return travel_dir or last_dir or "right"          # belts: keep the travel direction, gaps are auto-jumped
    return travel_dir or last_dir or "right"
