"""
Round-robin client for DiffusionGemma "decision model" replicas.

Talks to the vLLM structured_diffusion example server (the /v1/systemone "Jev-style"
endpoint described in the Red Hat article on running decision models on vLLM).
Each call sends the *same observation the RL policy received* (the DDEnv observation
image plus the numeric info dict) and asks one Choice question: "which button next?".

Endpoint discovery, in priority order:
  1. DGEMMA_ENDPOINTS  - comma-separated base URLs, e.g. "http://10.0.1.5:8011,http://10.0.1.6:8011"
  2. DGEMMA_HEADLESS_SERVICE (+ DGEMMA_PORT) - a headless k8s Service; every A record is a replica
  3. DGEMMA_URL - single base URL (default http://localhost:8011)

Round robin is client-side and stateless: caller N starts at offset N so 100 env
workers spread evenly across the 4 replicas instead of all hammering replica 0.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

log = logging.getLogger("decision_client")

# Must stay in the same order as DDEnv.valid_actions:
#   DOWN, LEFT, RIGHT, UP, A, B, 97 (LEFT+A+B), 98 (RIGHT+A+B), 99 (A+B)
DEFAULT_ACTION_OPTIONS: List[Tuple[str, str]] = [
    ("down", "D-pad DOWN: step toward the bottom of the street (use to line up vertically with an enemy)."),
    ("left", "D-pad LEFT: walk left / face left."),
    ("right", "D-pad RIGHT: walk right / face right. Only useful when no enemy is close."),
    ("up", "D-pad UP: step toward the top of the street (use to line up vertically with an enemy)."),
    ("a", "A button: attack (punch) the enemy directly in front of you."),
    ("b", "B button: attack (kick) the enemy directly in front of you."),
    ("kick_left", "Hold LEFT + press A and B: strong attack toward the left."),
    ("kick_right", "Hold RIGHT + press A and B: strong attack toward the right."),
    ("jump_kick", "A and B together: jump kick, hits enemies slightly ahead."),
]

DEFAULT_INSTRUCTIONS = (
    "You are playing Double Dragon on the Nintendo Game Boy, choosing exactly one controller input per step. "
    "It is a side-scrolling beat-em-up: thugs walk in from the left and right edges and you must knock each one "
    "out before the screen scrolls further (this also applies to ladders: a climb cannot finish while any enemy on "
    "the screen is alive). Walking into an enemy without attacking loses health and eventually "
    "a life. The state carries a 'radar' read from the console's sprite table: your position and facing, and "
    "for every enemy its horizontal offset dx (positive = to your right), vertical offset dy (positive = lower "
    "on the street), whether it is on your lane and whether it is in reach. Trust the radar over the picture. "
    "Decision rule: (1) an enemy in_reach on the side you are facing -> a, b or jump_kick; in reach but behind "
    "you -> kick_left/kick_right toward it or turn with left/right; (2) enemies on screen but none in reach -> "
    "close the gap: left/right toward the nearest, and up/down until dy is within 10px; (3) no enemies -> "
    "right to advance; (4) if the score has not risen for a long time and the same input repeats, change input. "
    "'radar_advice' is a literal reading of these rules; deviate only if the recent inputs show it is not working. "
    "Image 1 is the compact agent observation (three stacked 36x40 frames under three bar graphs); a second "
    "image, if present, is the full 160x144 screen."
)


def load_action_options() -> List[Tuple[str, str]]:
    """Allow overriding option descriptions without code changes (JSON list of [name, description])."""
    path = os.environ.get("DGEMMA_ACTION_OPTIONS_JSON")
    if not path:
        return list(DEFAULT_ACTION_OPTIONS)
    with open(path) as fh:
        data = json.load(fh)
    opts = [(str(n), str(d)) for n, d in data]
    if len(opts) != len(DEFAULT_ACTION_OPTIONS):
        raise ValueError(
            f"{path} defines {len(opts)} actions but the env action space has {len(DEFAULT_ACTION_OPTIONS)}"
        )
    return opts


def resolve_endpoints() -> List[str]:
    raw = os.environ.get("DGEMMA_ENDPOINTS")
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    host = os.environ.get("DGEMMA_HEADLESS_SERVICE")
    port = int(os.environ.get("DGEMMA_PORT", "8011"))
    if host:
        try:
            ips = sorted({ai[4][0] for ai in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
            if ips:
                return [f"http://{ip}:{port}" for ip in ips]
        except socket.gaierror as exc:
            log.warning("could not resolve headless service %s: %s", host, exc)
    return [os.environ.get("DGEMMA_URL", "http://localhost:8011").rstrip("/")]


def ndarray_to_data_url(arr: np.ndarray, scale: int = 1) -> str:
    """PNG-encode an HxWx3 uint8 array. Nearest-neighbour upscaling adds no information,
    it just keeps a 40px-wide observation from being crushed by the vision encoder."""
    img = Image.fromarray(np.ascontiguousarray(arr).astype(np.uint8))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class Decision:
    action: int
    name: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    latency_s: float = 0.0
    endpoint: str = ""
    ok: bool = True
    error: Optional[str] = None
    raw: Optional[dict] = None


class DecisionModelClient:
    def __init__(
        self,
        endpoints: Optional[Sequence[str]] = None,
        model: str = None,
        timeout_s: float = None,
        start_offset: int = 0,
        include_native_frame: bool = None,
        image_scale: int = None,
        refresh_interval_s: float = 60.0,
        instructions: str = None,
        action_options: Optional[Sequence[Tuple[str, str]]] = None,
        seed: Optional[int] = None,
        max_attempts: int = 2,
        samples: Optional[int] = None,
        sample_temperature: Optional[float] = None,
    ):
        self.model = model or os.environ.get("DGEMMA_MODEL", "dgemma")
        self.timeout_s = float(timeout_s if timeout_s is not None else os.environ.get("DGEMMA_TIMEOUT_S", "20"))
        self.include_native_frame = (
            include_native_frame
            if include_native_frame is not None
            else os.environ.get("DGEMMA_INCLUDE_NATIVE_FRAME", "0") == "1"
        )
        self.image_scale = int(image_scale if image_scale is not None else os.environ.get("DGEMMA_IMAGE_SCALE", "4"))
        self.instructions = instructions or os.environ.get("DGEMMA_INSTRUCTIONS", DEFAULT_INSTRUCTIONS)
        self.action_options = list(action_options) if action_options else load_action_options()
        self.name_to_index = {n: i for i, (n, _) in enumerate(self.action_options)}
        self.seed = seed
        # Number of diffusion reads the server averages per answer. 1 = one forward pass (~0.07s on an
        # L40S with the observation image); the server default of 4 is ~10x slower for little gain here.
        self.samples = int(samples if samples is not None else os.environ.get("DGEMMA_SAMPLES", "1"))
        # 0 = argmax (deterministic). >0 = sample the action from probabilities**(1/T); gives diverse
        # trajectories across the 100 emulators and a richer distillation dataset.
        self.sample_temperature = float(
            sample_temperature if sample_temperature is not None else os.environ.get("DGEMMA_TEMPERATURE", "0"))
        self.max_attempts = max_attempts
        self.refresh_interval_s = refresh_interval_s
        self._fixed_endpoints = list(endpoints) if endpoints else None
        self._endpoints: List[str] = []
        self._last_refresh = 0.0
        self._rr = start_offset
        self._sessions: Dict[str, requests.Session] = {}
        self.stats = {"requests": 0, "errors": 0, "fallbacks": 0, "latency_sum": 0.0}
        self._refresh_endpoints(force=True)

    # ------------------------------------------------------------------ endpoints
    def _refresh_endpoints(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_refresh < self.refresh_interval_s:
            return
        eps = self._fixed_endpoints or resolve_endpoints()
        if eps != self._endpoints:
            log.info("decision endpoints: %s", eps)
        self._endpoints = eps
        self._last_refresh = now

    def _next_endpoint(self) -> str:
        self._refresh_endpoints()
        # v105: random replica per request. Round-robin from per-worker offsets drifted into alignment (plan replays
        # pause requests for ~100 steps at a time), and one replica took 52 concurrent requests while eleven idled.
        self._rr += 1
        return random.choice(self._endpoints)

    def _session(self, endpoint: str) -> requests.Session:
        s = self._sessions.get(endpoint)
        if s is None:
            s = requests.Session()
            self._sessions[endpoint] = s
        return s

    @property
    def endpoints(self) -> List[str]:
        return list(self._endpoints)

    # ------------------------------------------------------------------ request
    def build_request(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
        native_frame: Optional[np.ndarray] = None,
        previous_action: Optional[str] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> dict:
        images = [ndarray_to_data_url(obs, scale=self.image_scale)]
        image_notes = ["image 1: the agent observation (see instructions)"]
        if self.include_native_frame and native_frame is not None:
            images.append(ndarray_to_data_url(native_frame, scale=2))
            image_notes.append("image 2: the current full-resolution Game Boy frame (160x144)")

        state = {
            "game": "Double Dragon (Game Boy)",
            "score": int(info.get("score", 0)),
            "lives": int(info.get("lives", 0)),
            "level_id": int(info.get("level", 0)),
            "step": int(info.get("steps", 0)),
            "levels_completed_this_episode": int(info.get("levels_completed", 0)),
            "previous_action": previous_action,
            "images": image_notes,
        }
        if extra_state:
            state.update(extra_state)
        body = {
            "model": self.model,
            "instructions": self.instructions,
            "state": state,
            "images": images,
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "Which single button input should the player press for the next step?",
                    # Jev contract: choice criteria is an ordered map of option name -> description
                    "criteria": {n: d for n, d in self.action_options},
                }
            },
        }
        if self.samples > 0:
            body["samples"] = self.samples
        if self.seed is not None:
            body["seed"] = self.seed
        if request_overrides:
            body.update(request_overrides)   # e.g. {"think": 1} to let the model reason before answering
        return body

    def _pick(self, probs: Dict[str, float], argmax_name: Optional[str]) -> Optional[str]:
        if self.sample_temperature <= 0 or not probs:
            return argmax_name
        names = [n for n in probs if n in self.name_to_index]
        if not names:
            return argmax_name
        w = np.array([max(probs[n], 1e-9) for n in names], dtype=np.float64) ** (1.0 / self.sample_temperature)
        w /= w.sum()
        return names[int(np.random.choice(len(names), p=w))]

    def _parse(self, payload: dict) -> Tuple[int, str, Dict[str, float], float]:
        ans = (payload.get("answers") or {}).get("action") or {}
        probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
        name = self._pick(probs, ans.get("choice"))
        if name not in self.name_to_index and probs:
            name = max(probs, key=probs.get)
        if name not in self.name_to_index:
            raise ValueError(f"model returned unknown action {name!r}; answer={ans}")
        confidence = float(ans.get("confidence", probs.get(name, 0.0)))
        return self.name_to_index[name], name, probs, confidence

    def choose_action(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
        native_frame: Optional[np.ndarray] = None,
        previous_action: Optional[str] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        body = self.build_request(obs, info, native_frame, previous_action, extra_state, request_overrides)
        last_err: Optional[str] = None
        for _ in range(self.max_attempts):
            ep = self._next_endpoint()
            t0 = time.perf_counter()
            self.stats["requests"] += 1
            try:
                resp = self._session(ep).post(f"{ep}/v1/systemone", json=body, timeout=self.timeout_s)
                resp.raise_for_status()
                idx, name, probs, conf = self._parse(resp.json())
                dt = time.perf_counter() - t0
                self.stats["latency_sum"] += dt
                return Decision(idx, name, probs, conf, dt, ep, True, None, resp.json())
            except Exception as exc:  # network, HTTP, JSON or schema problems all fall through to retry
                self.stats["errors"] += 1
                last_err = f"{ep}: {type(exc).__name__}: {exc}"
                log.warning("decision request failed (%s)", last_err)
        # Every attempt failed: keep the emulator moving with a random legal action.
        self.stats["fallbacks"] += 1
        idx = random.randrange(len(self.action_options))
        return Decision(idx, self.action_options[idx][0], {}, 0.0, 0.0, "", False, last_err, None)

    def health(self) -> Dict[str, bool]:
        self._refresh_endpoints(force=True)
        out = {}
        for ep in self._endpoints:
            try:
                out[ep] = self._session(ep).get(f"{ep}/health", timeout=5).ok
            except Exception:
                out[ep] = False
        return out
