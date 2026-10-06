# Application Documentation

## Run Locally

To run this application locally, follow these steps:

1. Create a Python virtual environment:
   ```bash
   python3.11 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create the required directory structure:
   ```bash
   mkdir -p ignored
   ```

5. Place the required ROM files in the ignored directory:
   - Copy `dd.gb` to `ignored/dd.gb`
   - Copy `dd.gb.state` to `ignored/dd.gb.state`

6. Run the application:
   ```bash
   python3 run-ray-train.py
   ```

Note: Make sure you have Python 3.11 installed as this version is required for compatibility with all dependencies.

## Building the Container Image

### Prerequisites

1. OpenShift Cluster with Tekton installed
2. `oc` CLI tool installed and logged into the cluster
3. Access to build and push images in your namespace

### Setting Up the Build Environment

1. Create the required Kubernetes resources:
   ```bash
   # Create the Service Account and PVC
   oc apply -f .tkn/hello-chris-sa.yaml
   oc apply -f .tkn/pvc.yaml

   # Apply Tekton Tasks
   oc apply -f .tkn/buildah-task.yaml
   oc apply -f .tkn/git-clone-task.yaml
   ```

2. Set up the RPM serving infrastructure:
   ```bash
   # Create an upload pod for temporary file storage
   oc create -f - <<EOF
   apiVersion: v1
   kind: Pod
   metadata:
     name: upload-pod
   spec:
     containers:
     - name: upload-pod
       image: nginx
       ports:
       - containerPort: 80
     volumes:
     - name: shared-data
       emptyDir: {}
   EOF

   # Wait for the pod to be ready
   oc wait --for=condition=ready pod/upload-pod

   # Copy the RPM file to the pod
   oc cp path/to/your.rpm upload-pod:/usr/share/nginx/html/

   # Create a service to expose the nginx server
   oc expose pod upload-pod --port=80
   ```

3. Configure the pipeline:
   ```bash
   # Apply the pipeline definition
   oc apply -f .tkn/hello-chris-pipeline.yaml
   ```

### Running the Build

1. Start the pipeline:
   ```bash
   oc apply -f .tkn/run.yaml
   ```

2. Monitor the build progress:
   ```bash
   tkn pipelinerun logs -f
   ```

### Pipeline Structure

The Tekton pipeline (.tkn/hello-chris-pipeline.yaml) consists of the following stages:
- Git clone of the source repository
- Building the container image using Buildah
- Pushing the image to the specified registry

### Build Assets

The `.tkn` directory contains all necessary Tekton resources:
- `buildah-task.yaml`: Task definition for building container images
- `git-clone-task.yaml`: Task for cloning Git repositories
- `hello-chris-pipeline.yaml`: Main pipeline definition
- `hello-chris-sa.yaml`: Service Account configuration
- `pvc.yaml`: Persistent Volume Claim for build workspace
- `run.yaml`: PipelineRun definition to trigger the build

### Image Build Process

The image is built using a multi-stage Dockerfile located in the `image/` directory. It:
1. Installs system dependencies
2. Sets up Python environment
3. Installs Python packages from Pipfile.lock
4. Configures the runtime environment for the Ray application

Note: The build process requires access to both PyPI and the temporary nginx server hosting the RPM file.


## Decision-model mode (DiffusionGemma instead of PPO)

The RL loop can be replaced by a *decision model*: at every emulator step the exact
observation the PPO agent received (`DDEnv._get_observation()`, a 128x40x3 image) plus the
numeric `info` dict is sent to a DiffusionGemma replica, which answers one **Choice**
question over the existing 9-action space (`down, left, right, up, a, b, kick_left,
kick_right, jump_kick`). No learner, no gradients; the old shaped reward is still logged
as a comparison metric. Background: <https://developers.redhat.com/articles/2026/09/28/run-decision-model-vllm-and-red-hat-ai>

### Pieces

| Path | What |
|---|---|
| `k8s/rosa/create-cluster.sh` | ROSA HCP cluster: 2x m6i.4xlarge default, 4x m6i.8xlarge `cpu-workers`, 4x g6e.2xlarge `gpu` (L40S 48 GB), then NFD + NVIDIA GPU Operator |
| `k8s/dgemma/` | 4-replica Deployment: vLLM (`RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`, :8000) + structured_diffusion example server sidecar (`/v1/systemone`, :8011); headless + ClusterIP services; `deploy.sh` |
| `app/decision_client.py` | Round-robin client. Resolves every A record of the headless service, offsets each worker's starting replica, retries on the next replica, falls back to a random legal action if all fail |
| `app/run-decision-dd.py` | Runner: N Ray actors (1 CPU each) x DDEnv, step-by-step decisions, progress + JSON summary (also to S3) |
| notebook | "Decision-model mode" cells: CPU-only CodeFlare cluster, job submit, follow |

### Run

```bash
# 1. cluster + GPUs (needs `rosa login` and AWS creds in profile rosa-rl)
k8s/rosa/create-cluster.sh

# 2. decision-model service
k8s/dgemma/deploy.sh

# 3. local sanity checks
cd app
python run-decision-dd.py --dry-run                       # prints one request body, no server needed
python run-decision-dd.py --local --num-envs 1 --max-steps 200 --endpoints http://localhost:8011

# 4. at scale: use the notebook cells, or from inside the Ray cluster
python run-decision-dd.py --num-envs 100 --episodes-per-env 1 --max-steps 2048
```

Client knobs (env vars): `DGEMMA_ENDPOINTS` (explicit list) or `DGEMMA_HEADLESS_SERVICE`/`DGEMMA_PORT`,
`DGEMMA_MODEL`, `DGEMMA_TIMEOUT_S`, `DGEMMA_IMAGE_SCALE` (nearest-neighbour upscale of the 40px-wide
observation), `DGEMMA_INCLUDE_NATIVE_FRAME=1` (also send the raw 160x144 frame; goes beyond RL parity),
`DGEMMA_ACTION_OPTIONS_JSON` (override the per-action descriptions the model sees).

### Metrics and recording

Every decision run writes to `s3://sessions/decision_runs/<run_tag>/`:

| Key | Contents |
|---|---|
| `traces/wNNN_epMM.npz` | `obs` (T,128,40,3) uint8, `teacher_probs` (T,9), `steps` (step, action, confidence, latency, ok, reward, score, lives, level), `action_names`, `meta` |
| `episodes/*.json`, `episodes.jsonl` | one record per episode: return, max score, lives, levels visited (ids + names), DDEnv `levels_completed`, fallback steps, latency, action histogram, wall time |
| `summary.json` | run-level aggregates: return/score stats, `episodes_reaching_level`, `new_levels_reached_max`, fallback fraction, latency, confidence, action distribution |

Full-frame MP4s go to `s3://sessions/ddenv_videos/decision/<run_tag>/` (DDEnv built-in).

#### Level ids (RAM 0xE110) seen in the bundled save states

| State | level id | name used in metrics |
|---|---|---|
| dd.gb.state, lvl-1.5.dd.gb.state | 15 | mission1 |
| boss.dd.gb.state | 84 | mission1-boss |
| lvl2-dd.gb.state | 89 | mission2 |
| lvl3-dd.gb.state | 12 | mission3 |
| lvl3.5-dd.gb.state | 5 | mission3-part2 |

`DDEnv`'s reward map only knows 15/84/48/89/11, so mission 3 progress is invisible to the shaped reward
and to `levels_completed`. `game_cleared` is left `null` until the ending screen's RAM signature is
captured from a winning run; reproduce the table with the snippet in `git log -p` for this README entry
or by loading each state in PyBoy and reading 0xE110.

### Training PPO from the decision model (distillation)

1. Run decision mode with traces on (default). Each step stores the teacher's full 9-way probability vector.
2. Distil into the PPO policy net:
   ```bash
   python bc_pretrain.py --s3-prefix decision_runs/<run_tag>/traces/ --epochs 5 \
       --out bc_policy.pt --upload-key models/bc_policy.pt      # KL to soft targets; --hard for argmax CE
   ```
3. Warm-start PPO from it and fine-tune with the normal reward:
   ```bash
   BC_INIT_WEIGHTS=s3://sessions/models/bc_policy.pt python run-ray-dd.py
   ```
   `BCWarmStartCallbacks` loads `conv_layers.*` + `fc.*` into `CustomCNNModel` on algorithm init and syncs
   the weights to all env runners; the value head trains from scratch.

### Node pools on the RHDP cluster

`rosa-rrvnz` is ROSA HCP inside RHDP's Red Hat org: no MachineSet API in the guest cluster and not visible
to your own `rosa` login. Add pools from the RHDP bastion with `k8s/rosa/bastion-machinepools.sh`
(4x m6i.8xlarge `workload=ray-cpu`, 4x g6e.2xlarge `workload=gpu`). Then `oc apply -f k8s/ray/raycluster-decision.yaml`.

Compare runs from MinIO: `python app/analyze_runs.py` (table of all runs) or `--tag <run_tag>` for the
best episodes of one run, with the level names each episode reached.

### Iterating on the decision policy

Knobs that changed behaviour in practice (see `decision_runs/*/summary.json` via `analyze_runs.py`):

| Flag / env | Effect |
|---|---|
| `DGEMMA_INCLUDE_NATIVE_FRAME=1` | also send the 160x144 frame. The 36x40 RL stack is too coarse for the model to see enemies; baseline without it pressed RIGHT 99% of the time and died in 74% of episodes |
| `--temperature T` / `DGEMMA_TEMPERATURE` | sample the action from the model's probabilities (0 = argmax). Breaks the identical-trajectory problem across 100 emulators and diversifies the distillation dataset |
| `--init-states dd.gb.state` | start every episode from the beginning of the game (full-game attempts) instead of the six mixed save states |
| `--decision-interval K` | ask the model every K steps, repeat in between; trades reactivity for throughput |
| `DGEMMA_SAMPLES` (default 1) | diffusion reads averaged per answer; the server default of 4 is ~10x slower |

The client also sends short-term memory in `state`: last 8 inputs, current same-input streak, steps since the
score last rose, lives lost this episode. Instructions live in `decision_client.DEFAULT_INSTRUCTIONS`
(override with `DGEMMA_INSTRUCTIONS`).

### Sprite radar (app/radar.py)

The vision path was the bottleneck: on the 36x40 stack the model pressed RIGHT 99% of the time; with the
full frame it switched to punching air 89% of the time. PyBoy exposes the Game Boy OAM sprite table, and
in Double Dragon the player is OAM slots 0-6 (tile ids < 128) while each enemy is an 8-sprite block
(tile ids >= 128). `sprite_radar(pyboy)` clusters non-player sprites into enemies and reports, per enemy,
`dx`/`dy` relative to the player, `same_lane` and `in_reach` (|dx| <= 22px, |dy| <= 10px).
`suggested_rule(radar, last_dir)` turns that into one line of advice the model can follow or override.
Both go into the request `state` every step. RAM 0xE100/0xE210 (the env's "player x/y") read constant
values, so the env's position-change reward never fires; the radar is the reliable position source.

### Progress metric: scroll distance

Level/area ids from RAM 0xE110 are noisy and the score digits at 0xC640-0xC644 occasionally read
garbage mid-episode, so the primary progress metric is **`distance_px`**: cumulative forward scroll of the
LCD SCX register (0xFF43), tracked by `radar.ScrollTracker`. The stage only scrolls after the on-screen
enemies are down, so distance measures real advancement; `screens_scrolled = distance_px / 160`. It is in
every episode record, the run summary (`distance_px_mean/max`), the live progress line, and the request
`state` (`stage_progress_px`).

**Score fix (2026-09-29):** `DDEnv._get_current_score` used to read 0xC640..0xC644 most-significant-first;
the game stores 6 digits least-significant-first (0xC645 is the top digit). The old reading was
non-monotonic garbage that plateaued near 9990, which is why every pre-fix `score_*` metric (baseline,
v2, v3 runs) is meaningless. Compare those runs on `died_fraction`, action mix and (from v4 on) `distance_px`.

### Health, doors, stuck detection (radar.py additions, 2026-09-29)

- **Health**: the HUD is the Game Boy window layer; row 0 columns 4-10 are the health ticks (tile 378 = full,
  7 when full). `read_health()` exposes it as `health_ticks`; the advice switches to a defensive rule at <= 2.
- **Doors**: level exits are blocks of BG tile 362 (>= 3 tiles wide) sitting on the street top. `find_doors()`
  reports them relative to the player; walking past a door locks the player at the right edge (one-way), so
  the advice tells the model to stop under it and press up.
- **Stuck**: same direction for 6 decisions with no player movement => `stuck_warning` in state and
  ladder/edge-aware advice.
- **Reachability** is relative (far above the player while the player is at the top of its band), because
  mission 1 has stairs.
- **State capture**: PyBoy save states are uploaded to `decision_runs/<tag>/states/` on life loss and on stuck
  streaks (max 4 per episode); load one locally with `pyboy.load_state(io.BytesIO(blob))` to replay a situation.

**Held directions (2026-09-29):** `DDEnv._run_action_on_emulator` used to release D-pad inputs after 4 of the
8 frames per step. Doorways/climbs need a sustained press, so pulsed UP never entered anything. Directions are
now held for the whole step (consecutive identical steps form a continuous hold); A/B remain taps. This changes
the env for PPO as well (arguably for the better). Stuck detection is position-based: >= 6 of the last 12
inputs the same direction with no player/scroll movement and no reachable enemy => `stuck_warning` plus
"blocked at an exit: hold up 5-6 steps" advice. Emulator states are also saved at truncation.

### Mission 2: screen lock, conveyor walkway, and the frame-level reflex layer

Diagnosing v16's stall (21 of 41 "no-progress" snapshots were the same scene) exposed three mechanics the
radar now encodes:

* **Screen lock.** Double Dragon does not scroll (or let a ladder climb finish) while any enemy on the screen is
  alive. Enemies never engage a player on ladder rungs, so a player who flees up the right-edge ladder of the
  first indoor room freezes the game: the two thugs stand still at the door, the earlier "inactive enemies"
  heuristic then told the model to ignore them and keep climbing. `radar.sprite_radar` now tags ladders that run
  off the top of the screen (`off_top`) and `suggested_rule` answers "SCREEN LOCKED: climb down, punch them from
  the ladder foot" (punching from the foot kills both with no damage in ~45 steps; walking through them = thrown
  into the pit on the right). Passive-enemy detection is disabled while `on_ladder` or on a locked screen, and
  the runner clears its passivity history while climbing.
* **Rungs swallow input.** On a ladder only up/down register; `at_top` now means feet within 2 px of the top
  row (measured), `on_it` ends at the ladder's bottom row, and there is a dedicated "you are on the rungs: press
  up/down N more steps to the floor the enemy is on, then fight" rule.
* **Conveyor walkway** (`app/reflex.py`). The upper walkway after that room is a chain of belts: the surface
  carries the player ~3.6 px/step (walking against it nets ~1.8 px/step), the belts are separated by 24 px gaps,
  the only way across is a jump kick toward the gap (LEFT+A+B) pressed when the edge is **2-5 px** away (1-8 px
  works onto a still floor, but the third segment runs back toward its gap and pushes a late lander in during the
  post-landing recovery frames), and jumping while the camera is still catching up (player left of screen x≈68)
  lands ~15 px short. Mission 3 opens with 8 px river gaps between 24 px stepping stones: there the window is
  **5-8 px** (3 px falls in), so the reflex picks its window from the gap width read off the tilemap. A 1.2 s-latency model acting every 16 frames cannot time that, so the runner has a reflex:
  `BeltTracker` infers belt direction from drift the input does not explain (belt tile ids animate globally and
  do not encode direction), `belt_reflex()` overrides the model's action only at the edge (jump in the window,
  one nudge back at 1 px, keep walking to re-centre the camera), and `radar_advice` gets a CONVEYOR note telling
  the model to keep choosing its travel direction and never jump itself. `reflex_steps` is recorded per episode.
  Scroll tracking now counts leftward and downward scrolling as progress (this walkway runs left, and the level
  later descends).
* Ladders that pass *through* the floor you stand on are decoys (the right-hand ladder of the third indoor room only
  leads down; the left one, whose foot is at your floor, is the way up). The radar now reads the player's climbing
  pose from its sprite tiles (head tile 13, legs 60/61) instead of guessing from position, marks pass-through
  ladders unclimbable, and tolerates the 8 px x-offset the game applies while climbing.
* Scripted mapping with these rules (a fight/climb/walk policy plus the reflex) now completes **mission 2** from
  its start with no life lost: pits → ladder → locked room (fight) → conveyor walkway (3 gaps) → ladder (+72 px) →
  walkway right with thugs → ~160 px descent → lower walkway → left ladder (+84 px) → right-edge ladder to the top
  → exit into **mission 3** (forest / river: area ids 11, 12, 49, 244-246, 52, 53, 46), and crosses mission 3's
  first ~800 px of stepping stones and thugs. Save states for each waypoint live in app/ignored/.

* **Decor doors (v19 diagnosis).** Mission 2's walkways are lined with doorway-shaped tiles; when a run stalled there,
  the "pinned ⇒ door mode" logic told the model to hold UP under one forever (about 20 of v19's 36 no-progress
  snapshots). Doors are now only reported while the screen has never scrolled vertically (mission-1 style
  screens), the "blocked ⇒ try a doorway" advice becomes "no doorways on this level: turn around or find a
  ladder", and a pit that runs into the screen edge is flagged as the end of the floor (not a gap to jump), which
  also flips the default "walk right" to "walk left" on the conveyor room's landing.
* Per-episode records are now uploaded as each episode finishes (`decision_runs/<tag>/episodes/w###_ep##.json`), and
  `analyze_runs.py --tag` falls back to them when `episodes.jsonl` does not exist yet, so a run that is stopped early
  still leaves its metrics behind.
* The mission-1 boss and mission-3 bruiser are the same heavy enemy type. A 32-policy grid of scripted fighting
  styles (engage distance × attack × retreat × lane dodge, three timings each) clears neither from a saved state,
  while the model has beaten the mission-1 boss in roughly half of its attempts since v10, so boss fights stay with
  the model plus the mined advice; scripted reflexes are reserved for frame-precise mechanics (gaps, ledges).
* **v20 diagnosis.** (a) Mission-1 street: the sidewalk lane ends at the B1 building corner and the road lane below
  continues; the "blocked walking right" advice now says step down 3-4 steps then walk right (6 no-progress
  snapshots). (b) The gap reflex jumped into mission 2's first pit chain because only 8 px of it was visible at the
  time; a climbable ladder underfoot or before a gap now vetoes the jump and silences the gap note, so the model
  takes the ladder (13 deaths). (c) Hanging on a locked ladder with thugs at its foot now always gets the
  climb-down advice (the in-reach attack rule used to win and attacks are ignored on rungs).
* **v21 diagnosis (first run with per-episode records).** 12 mission-1 starters: 6 beat the boss (50%), 6 died in the
  boss room. All 6 mission-2 starters died in mission 2, and the traces show two mechanisms: (a) after climbing the
  first ladder the model sometimes wandered LEFT, and the gap reflex, seeing 8 px of the pit chain, jumped it into
  the pit (8 of 18 deaths) — jumps now require commitment (a belt carrying the player, or 2 of the last 3 decisions
  toward the gap) and wide gaps are only jumped on the conveyor walkway, which the radar recognises from its belt
  tiles; (b) in the locked room the model punched while still on the rungs (ignored), stepped off late and was
  thrown into the pit — attacks on rungs are now converted into the climb step (`rung_veto`).
* **v22 (first full run under the reflex layer, 50 episodes, 60 min):** distance mean 888 px (v21 at the same point:
  594), max 1738, vertical max 312, 34% of episodes ended by game over. Mission-1 starters beat the boss 4/10 (v21
  6/12), but every winner arrived in mission 2 with one life left because the fight cost 12-28 health.
* **Boss fights mined from 46 recorded fights (v19-v22)**, hit% / hurt% within two steps of the input, on his lane:
  at 0-12 px `a` 5/25, `b` 6/10, stepping away 0/0; at 13-22 px `a` 21/5, **`b` 31/4**; at 23-34 px `a` 4/11,
  stepping toward 13/7. The boss/heavy-thug advice now states exactly this (kick at 13-22, back off under 12, one
  step in from 23-34) and a `boss_veto` reflex enforces the two unambiguous facts: an attack at <=12 px becomes a
  step away, a punch at 13-22 px becomes a kick. Winners and losers both walked into him ~75% of the time, so the
  old "hold your ground" wording was not being followed and is gone.
* Mission 2's first floor ends in background decor tiles (286/287/314/315/317) that are as fatal as the dark tile;
  pit detection now treats them as void, so the pit chain reads as one 56 px drop (not jumpable) instead of an
  8 px gap the reflex would hop into.
* **v23 (boss reflex) was a regression and is reverted:** with `boss_veto` forcing kicks at 13-22 px and step-backs
  under 12 px, 0 of 21 mission-1 starters beat the boss (v22: 4 of 10) and the reflex fired 120-215 times per fight.
  The mined hit/hurt rates are conditional on the model's own timing, not causal per input. Boss fights stay with
  the model; the advice keeps the ranges as information only.
* **v24 (no boss reflex) at 30 min:** boss wins back to 5/9, vertical best 392 px and distance best 1796 px (both
  records). All six deaths of the mission-2 starters were in the locked room: the model punches at the ladder foot,
  then walks INTO the thugs and is thrown backwards into the pit. `pit_back_veto` (reflex.py) turns a step toward a
  reachable enemy into a punch when a pit edge is within 24 px behind; a trace-like random fighter clears the room
  7/8 with it versus 5/8 without.
* **v24 stall (40/40 no-progress snapshots):** a limit cycle of my own making at the conveyor landing. The default
  direction said "walk right" while the floor edge was more than 60 px away and "walk left" inside 60 px, so the model
  paced between x=81 and x=87 for the rest of the run. Any visible floor edge on the right now decides "the way on is
  left", with no distance threshold.
* **v26 stall (38 of 39 no-progress snapshots, read from the new sidecars: 200/200 inputs were "right"):** at the
  conveyor room's ladder top the camera parks so the drop is off-screen and the player is pressed against the right
  screen edge; the ladder-top rule saw no pit on the right and said "walk right". The screen edge now counts as a
  blocked side, and on vertically-scrolled walkways x >= 128 always sends the player left.
* **v27 stall (90 no-progress snapshots, sidecars: 106 right / 94 left):** one step left of the ladder top the
  ladder rule no longer applied and the default direction said "right" again. On vertically-scrolled walkways the
  default now sends the player left whenever x >= 120 or it touches the right screen edge.
* **v28 stall (cycle moved to x=120, the newest threshold):** positional thresholds cannot fix a pacing cycle, they
  only move it. The runner now remembers the walkway's travel direction the moment any rule says "the way on is
  LEFT/RIGHT" and repeats it in the default advice until the screen has scrolled 48 px (mission 2+ only).
* **v29:** the ladder-landing cycle is gone (vertical best 456 px after 10 minutes, a record). The next stall was the
  16 px island between the conveyor room's second and third gaps: the DANGER text ("floor ends left ... floor ends
  right") made the model alternate left/right (100/100) instead of committing to the jump. On the conveyor walkway
  the gap text now says the jump is automatic and to keep pressing the travel direction, and the sticky travel
  direction is now kept per floor (cleared when SCY changes) instead of after 48 px of scroll, so the belt's own
  scrolling no longer erases it. From the island state a committed-left model now jumps the gap and reaches the
  ladder floor with no life lost.
* **v30 (island fix): vertical best 648 px, distance best 2320 px, score best 18880 — all records; model runs now
  reach room 3's top.** Three room-3 misreads fixed from the sidecars: (a) at the right ladder's top the feet settle
  2-4 px below the top row and the on-rungs rule demanded "up" 135 times — stuck-pressing-up within 6 px of the top
  now counts as at the top; (b) on the decoy (downward) ladder the lock rule said "climb down to the enemy at the
  foot" although the only enemy was a floor above — enemies more than 40 px above/below on ladder levels are now
  unreachable and the advice sends the player back up; (c) the ladder-top safe side flipped between x=128 and x=124 —
  the remembered travel direction now applies inside the ladder-top rule too.
* **v31 diagnosis.** (a) Stuck detection had been dead for many runs: the runner kept only 8 recent inputs while the
  check required 12, so no STUCK/BLOCKED prefix (and none of the rules keyed on it) ever fired in production. Fixed.
  (b) The climbing pose persists one step after reaching a ladder's foot; left/right/attacks are ignored there and one
  `down` lets go — the rung veto and the advice now cover that pose (200 wasted attacks next to a thug in room 3).
  (c) The decoy-ladder test needed the player directly under it; it now applies within 16 px, so the advice points
  to the real ladder across the room.
* **v32 regression, caught after 11 minutes:** the new ladder-foot rule ("press down once to let go") walked every
  mission-2 starter off the ledge next to the locked room's ladder (18/18 died at the same spot). The rule and the
  matching veto now apply only when no drop is within 40 px; at that ladder foot attacks land without letting go.
* **v33: boss wins 6/10 (stuck detection alive again), vertical best 856 px, distance best 2292 px.** Stalls fixed:
  (a) decoy-ladder detection no longer depends on the player's x (17 px away it flipped back to "climbable" and the
  model paced 100/100 between the two ladders); (b) the rung veto only weighs enemies within 40 px vertically (a thug
  on the lower walkway was turning every "right" at the top of room 3's ladder into "down"); (c) futile-attack
  detection: five or more attacks in the last eight decisions with no score change for 24 steps yields "your attacks
  are not landing, walk past" (knife thrower across a gap, bruiser at the screen edge); (d) the projectile dodge text
  says to dodge once and keep travelling (the model was dodging 160 times in a row).

### Milestone (v34, 2026-09-30 15:05 UTC): the model completed mission 2

Three v34 episodes (workers 15, 42, 44) reached the mission-3 forest — confirmed by replaying a life-lost snapshot
(area id 11, score 20280, respawn at the mission-3 start with two lives). Records at that point: vertical 1208 px,
score 23880, distance mean 1427 px at 21 minutes. The respawn is saved as `app/ignored/m3-start-model.state` so
future runs can start a share of their emulators in mission 3.
* **Room-3 stack (v33/v34 stalls, 27 states):** the right ladder's top is a dead-end ledge; the screen only scrolls
  up once the thug on the floor below is dead (measured: kill him, climb again, the camera follows 88 px and the next
  floor with its own thug appears). A top-of-screen lock rule now says so and sends the player back down the ladder.
  Multi-floor detection uses "ladders visible" as well as SCY, because room 3's top sits at SCY=0.
* **v35 mission-3 starters (18/18 dead within 900 steps):** the forest's lower bank ends in the river at ~290 px and
  again at ~615 px; the river surface tiles (240/241 over 304/305) were not in the void set, so no pit was detected
  and the model walked in on both lives. They are void now, `river_level` marks the forest, and at the bank's end
  the advice is "press up onto the ledge path (feet ~84) and continue right"; a scripted follower then survives
  500 steps to the bruiser screen with 3 automatic stone-hops.
* **v36:** the forest bank has lanes down to feet ~112; my river fix only saw the top lane (feet 97), so 9/9
  mission-3 starters still walked into the water at 290 px and 615 px. The bank-edge tile 304 is void now (305 stays
  walkable: it is also the stepping-stone tile), and the "press up onto the ledge" advice appears from every lane.
* **Distillation on v30-v35 traces (87 traces, 108k frames):** 65% argmax agreement with the decision model after 8
  epochs (`models/bc_policy_v30-v35.pt`); a PPO warm start from it (`ppo-v30-v35-*`) runs alongside the decision run
  on ~10 CPUs. `bc_pretrain.py --s3-prefix` now accepts a comma-separated list of runs.
* **v37:** mission-3 starters now survive the bank (none finished in 20 min, versus all dead by 900 steps before), but
  the river void tiles also tile the mission-1 boss-room floor and broke boss fights (0/7 wins). The runner now tells
  the radar whether the episode is in the forest (started there, or 300+ px of climbing followed by a ladder-less
  screen) instead of guessing from tiles; the "bank ends, press up" advice applies only from the low lanes; and the
  pit-at-your-back veto reaches 32 px and lines up lanes before punching (4 deaths at the room-3 walkway end, thugs
  one lane lower, thrown off the right end).
* **v38:** mission-3 starters pass the bank (deaths moved to 344 px and 775 px) but died stepping DOWN from the ledge
  beside the stepping stones into the water: ledge detection only looked at the player's own column and only for true
  water tiles. It now checks the neighbouring columns and river-bank tiles too, and any `down` on a ledge is vetoed.
  Screen-edge "pits" on horizontally scrolling levels are the camera boundary, not drops (two left/right pacing stalls).
* **v39 (full run, 50 episodes):** mission-1 starters beat the boss **19/23** (v34: 6/10; the forest-flag fix removed
  the phantom boss-room pit), distance mean 1266 px and max 2396 px (records), vertical max 768 px; 26 episodes hit
  the 7000-step cap still alive, so the cap is now 12000. All 18 mission-3 starters died at the second stepping-stone
  gap: decisions taken mid-jump see no pits (wrong feet row), so the thug left behind looked reachable and the model
  walked back into the gap. The runner now flags airborne frames and the only advice in the air is "keep pressing
  your direction". Ledge detection is limited to the path itself (feet 76-92): the tree-stump platforms above it were
  read as ledge and the down veto trapped 8 runs on top of them.
* PPO warm start `ppo-v30-v35-1139` trained (130k+ env steps, episode length ~1330) but the driver script's final
  report crashed on a metric name (`episode_reward_mean` is `env_runners/episode_return_mean` in this RLlib);
  checkpoints are unaffected (3 under `ray_tune_results/dd_ppo_experiment_final/.../PPO_dd_env_165bc_*`, 78 min of training). Fixed in run-ray-dd.py (`env_runners/episode_return_mean`). Evaluating that PPO policy in the env with the decision-run metrics (distance / vertical / boss wins) is the next step on this track.
* **v41:** the mid-air rule held, but 13/13 mission-3 starters still died on the 24 px island after the second stone
  gap: the landing leaves the next gap only 4 px away (too close to take off), the reflex nudges back, and a couple
  of stray `left` presses then walk off the island's other side. Measured: leftward take-off over a stone gap works
  from 13-20 px (the left foot leads the sprite centre by ~8 px), rightward from 5-8 px. The reflex now uses
  direction-aware windows, steps back whenever the player is closer than the take-off point regardless of intent,
  and the advice names the island ("drop 20 px LEFT, gap 4 px RIGHT, only move right"). The ledge sentence was
  rephrased to lead with what to do (a literal reader latched onto "press down" inside "NEVER press down").
* **v42:** mission-3 starters now reach ~790-860 px (past the island) but die in the air: the traces show 'rrrrrr ll'
  — a stray left mid-jump pulls the arc back into the gap (air control), and some jumps were launched by the model
  itself 30-40 px early. Two vetoes: `air_veto` holds the jump direction until landing, and on the forest ledge with a
  gap ahead and no enemy in reach, model-initiated kicks/jumps become a walk (the reflex owns take-off there).
  With an advice-obeying stand-in both are neutral; they target the observed model mistakes specifically.
* **v43:** air control was still killing the island jump because airborne detection read the arc's apex (feet barely
  moving for one step) as a landing and released the hold. Airborne now also means "well above the last floor stood
  on"; an adversarial stand-in that presses left whenever airborne now clears both stone gaps.
* **v44 regression (10 min):** the apex fix made any upward lane change look "above the floor", so players were
  airborne forever and the advice froze on "keep pressing <last direction>" (mission-3 starters walked to the left
  edge and died at 0 px). The above-floor test now applies only while a jump is in progress.
* **v45 / new technique for the forest ledge:** after five model-side fixes the island still killed every mission-3
  starter, and the traces revealed one more mechanic: a thug that catches you from behind on the ledge GRABS and
  throws you into the river, and no input escapes a grab (measured). The stone crossing is now an autopilot segment
  (`stones_autopilot`): while on the ledge with a gap within 48 px on either side, or a thug closing from behind, the
  reflex walks right and fires the jumps itself; the model keeps control only for enemies ahead. Advice-obeying,
  adversarial (left on every airborne frame) and uniformly random stand-ins now all cross the stones from the
  mission-3 start without losing a life (18/18 runs).
* **v46 (ledge autopilot):** mission-3 starters now die at 1168-1432 px instead of 775-860, i.e. the stepping stones
  are solved; score best 31780 (record). Their deaths are now fights on the ledge against thugs AHEAD (grab -> throw
  into the river, health irrelevant) and the bruiser screen.
* **v46 deep runs:** a mission-1 starter (w003) played through mission 2 and the whole forest to the bruiser screen
  (area 46): distance 4264 px, score 34920; distance best in the run 4716 px, score best 38890. Every deep run ended
  pinned at the right screen edge with the bruiser behind it — the ledge pursuit rule kept them walking right, so
  heavy enemies are now exempt from it. Mining the 15 bruiser fights: hits land only at 0-12 px (a 9%, b 10%, ~1%
  hurt) and never at 13-22 px, the opposite of the mission-1 boss, so the heavy-thug advice is range-flipped in the
  forest. 6 of 15 fights were survived.
* **Forest attrition mined from v46 (54k mission-3 steps):** every run reaches the bruiser with 0-3 of 7 health. 191 of
  259 health losses are melee, almost all from thugs on the LEFT (behind): within 12 px they cost ~6% per step whether
  the model attacks or turns into them; thugs on the right cost ~0.1% per step at any range and get hit 22-72% of the
  time. New ledge advice for a thug within 12 px behind: step right once, then kick at 13-22 px. No bruiser fight was
  won in v46 (19 fights, 0-8 hits each); the earlier "6 survived" counted respawns elsewhere.
* **v47 (30 min):** vertical best 1336 px (record), mission-1 boss 4/6, mission-3 starters die at 1185-1754 px (ledge
  fights, then the bruiser at 1754 with 1 health). Sidecars at the bruiser screen showed two heuristics misfiring: the
  passive-enemy note called the motionless bruiser "inactive", and a false CONVEYOR note came from being pushed back at
  the edge. Both are suppressed when a heavy enemy is on screen, and a pinned-at-the-edge rule tells the model to walk
  left into the open first (hits do not register at the edge; he follows).
* **v47 (65 min):** mission-1 boss 8/10, vertical best 1824 px (record), mean distance 2076 px at 60 min (record), five
  bruiser fights all lost on arrival health 0-2. New stall from the sidecars: a jump that lands on a HIGHER platform
  (tree stumps) kept the airborne flag set and the "IN THE AIR, keep pressing left" advice for 200 decisions; feet
  stable for three steps now count as standing. v48 carries this, the pinned-at-edge bruiser rule and the note
  suppressions near heavy enemies.
* **v48 (45 min) → v49 "lane":** mission-1 boss 2/10 (v46 4/9, v47 10/12) with identical health-loss and score-gain
  rates per step in all three runs: the fight is a thin-margin grind, not a broken rule. Both v48 bruiser fights were
  lost from the right screen edge: 80-95 "left" presses that never moved the player (he blocks the lane) and kicks at
  16 px that never landed. Emulator probe of the bruiser by lane offset × distance: 6 px below his lane 0 of 286 attacks
  landed at ANY distance; level with him (within 2 px) punches land 18-41% and kicks up to 50% at 16-20 px. The
  mission-1 boss traces (14k steps) say the same: |dy|<=2 a 9-20% / b 20-33%, |dy| 3-5 2-6%, |dy|>=6 0-1%. So the
  v48 "hits only within 12 px" forest rule and the earlier 32-policy grid (10 px lane tolerance) were both wrong about
  the mechanic. A scripted policy with just "align within 2 px, punch at 12-24 px, step out under 12" clears the
  mission-1 boss 3/3 and the bruiser 2/3 (113 steps, 17 hits). v49: `heavy_lane_veto` (reflex) enforces alignment and
  the window for heavy enemies and area 48, leaving attack choice and dodges to the model; a random stand-in with the
  model's boss-room action mix goes 0/8 → 2-3/8 on the boss and 0/8 → 2/8 on the bruiser. It runs ahead of the ledge
  veto so the alignment "down" press on the bruiser's path is not turned into a punch; the LEDGE text says so. The
  heavy advice is one lane-first rule placed before the generic in-reach rule (which used to pre-empt it). Also: a
  knockback at the screen edge no longer reads as a conveyor, and the CONVEYOR note needs belt tiles under the player.
* **v49 (50 min, 20 finished) → v50 "level":** the lane veto raised boss-room alignment from 69% to 73-81% of steps
  but the mission-1 boss still went 1/6 with 29-41 hits per fight. Emulator: the boss falls after exactly 20 hits and
  RECOVERS ALL of them when the player loses a life (scripted fights: hits per life [4, 12, 20] → cleared on the
  third life, [20] → cleared on the first). The model lands ~12 per life, so it almost never wins; the advice now
  carries a per-life hit counter ("hits this life: N/20") and says health matters more than score. Four bruiser
  fights were lost on arrival health 0-1. The right-edge stalls (148, 91-105) were thugs 3-11 px off the player's
  lane half off the screen: aligned punches land there too (+60/+160 in the emulator), so the passive-enemy note no
  longer fires for edge-clipped enemies and the advice is to pull them into the open. The general finding: in 80k
  mission-3 steps near an enemy, 42k were 3-10 px off-lane, where punches land 1-2% (level: a 16-24%, b 29-38%); the
  model punched air 21k times there. v50 extends the lane veto to every reachable enemy within 28 px (off-lane attacks
  become the alignment press; nothing else is touched), the in-reach advice leads with "get level, then kick (b)",
  and the mission-1 boss note explains the 20-hits-in-one-life rule.
* **v50 → v51 "turn":** emulator check of the forest advice (5 saved states × 3 timings, 220 steps): walking on past
  thugs ("run") cost 3-9 health; fighting only enemies ahead 1.3-2.3; turning to fight EVERY level thug within 28 px,
  including those behind, 0-1.7 health with 11-25 hits. Punches beat kicks in every state (e.g. 21 vs 12 hits, 0 vs
  1.7 health lost), as they did in the boss harness, so the trace statistic that favoured b (38% vs 24%) is an
  attribution artefact of the two-tap decision interval and v50's kick-first text was wrong. v49 traces: 58% of
  mission-3 damage comes from thugs BEHIND and level with the player at 13-28 px, and punches at them land 6% vs 24%
  ahead: the player is facing the wrong way. v51: punch-first advice everywhere, the LEDGE text says turn and fight a
  level thug behind you instead of walking on, and the lane veto turns the player toward a level enemy before an
  attack thrown facing away. v50 (15 min) stopped in favour of v51; v49 and v47 keep running for comparison.
* **v51 → v52 "stones":** v51's mission-3 starters died early (5 of 6 deaths were falls) and the traces showed one
  deterministic signature at x=163 that v49 also had. Reproduced in the emulator: the grass bank (feet ~97) runs into
  the river while the stone tops (tile 352) continue at feet 88-92; the stones autopilot only armed at feet 76-92, so
  the player walked into the water (every bank-level strategy died, 11/11). Then the first hop: the running jump is
  ballistic (35 px; 24 trials with different mid-air inputs landed on the same pixel) and lands 27-d px into the 24 px
  island, so the (5,8) take-off window put the player 3 px from the far edge. A jump after a back-step is shorter, so
  only d <= ~11 is safe after one. v52: the autopilot presses UP onto the stone lane when a gap is ahead at bank level,
  the right-hand take-off window is (9,12) (wide-gap state 0/4 → 4/4), and a screen-edge "cliff" veto tried this tick
  was removed after it oscillated against the autopilot (unverified premise). The lane veto no longer turns or lines
  up within 40 px of a gap and sees the autopilot's intent instead of the model's. Finished workers now release their
  CPU (v51 waited 27 min behind 100 idle actors). v51 (40 min) stopped in favour of v52.
* **v52 (12 min) → v53 "bank":** v52's first mission-3 deaths were still on the stones screen, now on the far bank
  (scx 112) during the fight with the two thugs there. Replaying the override chain from the real mission-3 start
  state found the cause: the stone tops are tile 352, which is also a conveyor-surface tile, so the bank counted as a
  conveyor zone; the knockbacks of the fight then read as belt drift and the belt reflex fired kick_left jumps into
  the gaps behind (three per fight in the replay, one death). The conveyor zone is now off on the river level and a
  belt-driven jump needs belt tiles under the player. The replay now reaches the bank with no deaths. The mission-2
  area-66 falls seen in v52 occur at the same rate in v49/v51 (not new).
* **v53 → v54 "hole" (prepared):** the biggest mission-2 killer is a throw, not a hit. Three v49 traces at area 66
  (scx 176) show the same signature: player at x 137-140 on the top walkway with a thug 7-14 px to the left, then x
  jumps to 149-150 while the feet rise and fall (a grab-and-throw, no health lost) and the player drops through the
  8 px ladder hole at 144-152. 15 falls in v49, 7 in v51, 1 in v53's first minutes. New `hole_margin_veto`: on a
  multi-floor screen with a hole within 20 px on one side and a reachable thug on the other, attacks and steps toward
  the hole become a step toward the thug until 24 px of margin is back; the advice says the same. (The pit-behind veto
  deliberately ignores 8 px holes, and its "punch from the spot" answer is wrong here because the throw, not a
  knockback, does the damage.) Not yet verified in the emulator: the saved walkway state respawns on the lower floor.
* **v53 (30 min, 24 finished) → v54 "range":** mission 3 transformed: all 11 mission-3 starters reached the bruiser
  screen (1392-1976 px; v51's died at 840-1392), falls 3 of 22 deaths (v51: 5 of 6), forest hit rate 19% (v49 12%).
  Bruiser fights still lost: arrival health 0-4 with 1-2 lives, 0-12 hits; every death at x=149 has the enemy 16-18 px
  behind, level, and punches landing nothing. Emulator (4 forest states, 600 punches): at a level thug BEHIND you,
  12-17 px lands 0-24%, 18-23 px 83%; ahead, 12-17 px 19%, 18-23 px 43%. Mission-1 boss 0/7 (v49 1/6) against
  16/31 before the lane veto: hits per 100 steps unchanged (4.8 → 5.3) but knockdowns halved (1.6 → 0.8), so the
  veto is off in the boss room (advice stays). v54: thug attacks at 10-17 px open the distance first (step away when
  there is room, lane change when pinned at the right edge), the hole-margin veto for walkway holes and the river
  bank, and the boss room back to model-only. v53 and v49 keep running alongside.
* **v54 (20 min) → v55:** the thug distance rule was a regression and is gone: mission-3 starters died at 482-1202 px
  (v53: 1392-1976), forest damage 1.03 per 100 steps (v53 0.67), hit rate 13% (19%). Stepping away from a thug at
  10-17 px hands it free hits; the emulator's 83% at 18-23 px was measured with the thug walking into the punch, not
  the player backing off. v55 = v53 + boss room model-only + hole-margin veto. The pinned-at-x=149 deaths (enemy 16-18
  px behind, nothing lands) remain the open bruiser-screen problem.
* **v55 → v56 "turn-heavy":** two more mechanical causes of the pinned x=149 deaths. (1) The heavy-enemy branch of the
  lane veto never turned the player: with the bruiser 16-18 px BEHIND and the player facing the screen edge, 'a' spam
  punched the edge for 86-97 steps (0 hits, v53 w000/w032/w048); an attack facing away from him is now the turn.
  (2) The stones autopilot's v46 clause "thug closing from behind → keep walking right" delivered players to the
  screen edge with the thug 16-18 px behind them; it is removed (the gap-ahead clause stays). Replays from four
  mission-3 states survive; the stones window check is unchanged. v55 (5 min) replaced by v56.
* **v56 (30 min, 35 finished) → v57 "react":** v56 is the best mission-3 run: 20 starters averaged 2109 px (v53
  1741), best 2320 (record from the mission-3 start), pinned deaths 13 → 0, falls 3 → 0, forest hit rate 23%, and
  every player reached the bruiser with 2-5 health and 2 lives. Mission-1 boss 4/7 with the room back on model-only.
  All 20 bruiser fights still lost (1-31 hits): the traces show the model walking into his 22-30 px strike range and
  punching during its own get-up frames. Harness: the scripted aligned policy clears the bruiser 4/4 (19 hits, 1.0
  health lost) when it acts every step, 1/4 when each action is held for two steps (5.5 hits, 5.0 lost) and 0/4 for
  three; the boss 3/4 → 0/4 → 1/4. The decision interval of 2 is what makes the fights unwinnable. v57 queries the
  model every step while an enemy is within 40 px (interval 2 elsewhere). The mission-2 hole falls (area 66, scx 176)
  did not improve with the hole-margin veto (18 of 104 deaths); still open.
* **PPO warm-start evaluated (ppo165bc-1001-0633, `app/eval_ppo.py`, `k8s/ray/submit-ppo-eval.sh`):** the BC-initialised
  PPO checkpoint (`PPO_dd_env_165bc_*/checkpoint_000002`, 78 min of training on v30-v35 traces) was played greedily
  from the three decision-run start states, 6 episodes, 6000 steps each. Mission-1 start: 240 px, score 0, all lives
  lost in 1246 steps. Mission-2 start: 74 px, score 8890, 3 lives, timed out. Mission-3 start: 32 px, score 21930,
  2 lives, timed out. The decision model covers 1000-1300 px from mission 1 and ~2100 px from mission 3 in the same
  budget. The distilled policy mostly stands and trades punches; the signal-producer path needs far more teacher data
  (the newer runs' traces, with the lane and reflex behaviour) and longer PPO fine-tuning before it is worth the CPU.
  Episode records are in `decision_runs/ppo165bc-1001-0633/episodes/` in the decision-run format.
* **v57 (35 min) → v58 "autopilot":** per-step decisions did not save the first v57 bruiser fight (grabbed at 28-40 px
  on the way in: the fight switch only armed within 40 px, and the lunge comes from 31). New `heavy_autopilot`
  reflex owns the heavy fights the way the stones reflex owns the crossing: align within 2 px, close to 16-24 px
  (boss 14-24), punch, step out when he is closer, keep walking toward where he was while he is down. Found and
  fixed on the way: his heavy flag (a sprite height) drops during his crouch frames, which made the "ordinary thug in
  reach" guard see the bruiser himself and hand the fight back to the model; the heavy identity is now sticky for 16
  steps. Harness with a random stand-in model deciding every other step: bruiser from the m3b state 5/5 cleared
  (19 hits, 0.8 health lost); a deterministic approach from 28 px: 20 hits, 0 damage, 63 steps; mission-1 boss 4/5.
  Waiting for him (he stands still) and an off-lane approach were both worse and are off. The model keeps everything
  outside heavy fights; the boss-room advice and the per-step fight decisions stay.
* **Walkway hole, reproduced (for v59):** `ignored/m2x-end-v17.state` is the area-66 top walkway with the ladder at
  dx 5, the hole at 144-152 and a thug arriving from the left. Replay, 6 timings each: fighting the thug from 14-24 px
  (which puts the player at x 106-112, 30+ px from the hole) 0 falls, 8 hits; pressing UP at the ladder as it arrives
  6/6 thrown into the hole; the v56 hole veto's "walk toward the thug for margin" 3/6 thrown (walking INTO it is what
  gets you grabbed). v56 traces show throws launched from x 118-141, so the carry is up to 30 px, not 12. The veto
  now arms for a thug within 64 px on the far side with the hole within 32 px: beyond 24 px it closes in, inside it
  fights facing the thug, and never steps toward the hole or tries the ladder; with an 'up' intent every step it is
  0/6 thrown and lands 8 hits.
* **v58 (1 h) → v59 "autopilot-2":** v58's autopilot lost its first 11 bruiser fights in the real env although the
  harness cleared 5/5. Running the real runner locally against the cluster model from the bruiser state (now possible:
  `--local --endpoints` + port-forward) and logging every reflex verdict found three gaps between harness and runner:
  (1) a step-out / turn-back limit cycle at exactly 15 px (the turn costs ~4 px) — hysteresis: facing away, keep
  stepping out until the turn lands inside the window; (2) the runner stamps the belt tracker's airborne flag onto the
  radar on fresh-decision steps, and a lane change trips it, so the autopilot handed the fight back mid-alignment —
  the guard is gone; (3) punches at 21-24 px (0/18 in the lane probe) — the window is 16-20 (boss 14-24), beyond it
  close in. New env-path benchmark (`/tmp/env_bench.py`, DDEnv.step, 8 start offsets): bruiser 5/8 cleared with
  16 hits and 1.9 health lost; the three losses are a grab at 19 px right after his get-up (his short-sprite frames).
  Backing off or holding 21+ px during those frames was worse (1/8, 2/8), so it is left as is. Step-out 18 was 0/8.
  Traces now carry the executed action as column 17. Also in: the walkway-hole veto rework (see above).
* **v59 (45 min): the bruiser is beatable, the approach is not solved.** The real runner, run locally from the
  bruiser state with the v59 autopilot, killed him (12-16 hits + 4 knockdowns, ~110 steps) and the model then walked
  on through new areas (71, 31, 73, 249, 248, 250, 72, 87, 231, 247, 28, 45, 27, 44, 234, 84-86) with a life to
  spare: the first time anything has passed area 46. In the cluster runs every arrival still dies there. The
  difference is the arrival geometry: from the forest the player meets him at 30-46 px mid-screen, and he lunges
  8-10 px every third step (his whole sprite moves, not an arm) with a reach that beats ours at 20-26 px; the
  autopilot reads the lunge as "too close", steps out, and never gets inside. From the close state, where he stands
  pinned at the screen edge, the same code clears him 5/8 (env benchmark, 8 offsets). Tried on two synthetic far
  states (he follows to ~30 px and stops), 8 offsets each, all 0/8: plain walk-in, off-lane approach dropping onto
  his lane at 18 px, walk interleaved with punches, counter-punch keyed on his lunge, jump-kick from 24-34 px, body
  distance anchored on his back edge, punch window 16-19 / 16-18, step-out 18. v59 traces (with the new executed
  action column) show the real arrival fights hovering at 20-23 px: punch at 20, his lunge, step out, walk back in.
  Open idea: his pinned position at the screen edge is what makes the close fight winnable, so the approach should
  aim to drive him there (he is knocked back by hits) or meet him there.
* **Bruiser punch tell found, not yet exploitable:** his sprite is 24 px wide standing and 40 px wide on the step
  his punch is out (the centre shifts 8 px toward the player); every hit taken in the env traces lands on such a
  step, and a fight held at 20 px while he stays 24 px wide landed 4 hits for no damage. The radar now exposes `w`
  and `back_x` per enemy. Dodging on `w >= 36` (step away, or step off his lane) with or without body-anchored
  distance stays 0-1/8 from the far states: by the step the width is visible the hit has already landed, and one env
  step (8 frames) is the reflex's reaction time. A knockdown throws him to the screen edge (x 153), which is exactly
  the pinned geometry the close fight wins from, but he gets up before the player can cover the 50+ px.
* **v59 (1 h, 39 finished) → v60:** mission-3 starters 2099 px mean, 2373 best (record); 18 bruiser fights, all
  lost, but with real hits now (1-20 per fight, four with 13-20) where v56 had 1-31 with most at 1-5. The mission-1
  boss with the autopilot: 0/16 (21-45 hits over three lives, 19-29 health lost per fight, never 20 in one life)
  against ~50% for the model alone, so the autopilot is off in area 48 again; the harness 4/5 there did not transfer.
  v60 = v59 minus the boss-room autopilot.
* **Bruiser approach, second round (all off by default):** from an arrival-like state (he stands at x 150 near the
  edge, player 35 px away) the plain walk-in is 1/8 and every alternative is 0/8 on the 8-offset env benchmark:
  off-lane approaches at 8, 10 and 12 px above his lane dropping onto it at 16-20 px (he hits across lanes), closing
  the gap while he flies from a knockdown, punch-frame dodges. The close fight (he pinned at the edge, player inside
  20 px) stays 5/8. The remaining lever is the 8-frame reaction time of the reflex itself; a finer action grain for
  heavy fights would need a per-step action_freq change in DDEnv and a re-tune of the stones windows.
* **v60 (45 min) → v61 "rhythm":** without the boss-room autopilot the mission-1 boss is still 0/8 (hits per life
  17/12/11, 13/15/4, 1/17/13: never 20). It won 4/7 in v56, 5/12 in v53, 10/12 in v47, all with the model's chosen
  action held for two env steps; v57 introduced a fresh decision every step within 40 px of an enemy and the boss has
  not been beaten since (0/16, 0/8). The per-step fight decisions are off (env `DD_FIGHT_PER_STEP=1` re-enables them);
  the heavy autopilot runs every env step on its own, so the bruiser fight is unaffected.
* **Bruiser, third round (off by default):** two autopilot decisions per 8-frame step (4-frame presses) is worse
  (close 4/8 vs 6/8, far 0/8), so reaction time is not the limit. In both long env runs he died on exactly the 4th
  knockdown, and a knockdown follows a run of consecutive hits, so a follow-through rule (keep punching 2-3 steps
  after a landed hit whatever his distance reads) was tried: no change (close 5/8, arrival 1/8, far 0/8) because the
  arrival fights die before the first hit lands. The approach from 30+ px is the whole problem; every rule tried so
  far loses to his lunge there. `AUTO_COMBO`, `AUTO_FLIGHT`, `AUTO_DODGE`, `AUTO_APPROACH`, `AUTO_BODY_DISTANCE`
  are the switches (all at the shipped v59 behaviour).
* **v61 (30 min): the boss collapse is explained.** With interval 2 restored and no autopilot the mission-1 boss is
  still 0/12, so the per-step decisions were not it. Boss-room comparison v47 / v56 / v61: arrivals identical (3
  lives, 4-5 health), action mix identical, per-punch hit rate identical (10-11% level in 12-24 px), but knockdowns
  per 100 steps 1.6 / 0.8 / 0.6 and wins 10/12 / 4/7 / 0/12. Trace timing: a knockdown lands exactly 4 steps after
  the previous hit with two hits in the 8 steps before it, i.e. it is the THIRD hit of an unbroken punch chain; and
  the emulator shows he only dies to knockdowns (punch-only policy clears 3/3 with 2-5 knockdowns; kick-only, 24-28
  kick hits, never clears; kick hit = +50, kick knockdown = +300). v47's model kept pressing 'a' in streaks of 3+ 77%
  of the time, v56-v61 58%, which halves the knockdowns. v62: `boss_combo_veto` holds 'a' for the 3 steps after a
  landed punch (runner flags the +30), the area-48 note states the chain mechanic and drops "protect your health",
  and the bruiser autopilot is unchanged (its wins come from the same chains). Random stand-in cannot test this (7
  hits per 500 steps); validated with the real model locally instead.
* **v62 "combo" launched:** two real-model boss fights run locally with `boss_combo_veto` both won (boss down, on into
  mission 2 with a life to spare, 28 and 42 vetoed steps), against 0/12 in v61 and a plain-v61 local fight lost in
  449 steps. v62 = v61 + the combo veto + the chain-mechanic boss note.
* **v62 (30 min) → v63 "v47-text":** the combo veto did not transfer to the cluster: mission-1 boss 0/11, veto active on
  0-6% of boss-room steps, 7-25 hits and 1-5 knockdowns per fight (the two local wins were luck or the full-health
  start). The only configuration that ever won the room reliably is v47's (10/12), and its advice was the opposite of
  every rewrite since: "hit him 2-3 times (a; b hits harder), then step DOWN or UP off his lane to dodge his
  counter-punch, then line up again", with the mined dx-bucket text (13-22 px: b 31% / a 21%, mix them). v63 restores
  both verbatim for area 48 (no 20-hit counter there), drops the combo veto from the chain, and changes nothing else.
  v62 keeps running for its mission-3 starters.
* **v62 (1 h): first cluster runs past the bruiser.** Workers 38 and 45 (mission-3 starters, 2333 and 2331 px) killed
  him (12-13 punches + 3-4 knockdowns, then a 6680-8040 point bonus) and went through a one-step-per-area transition
  (31, 73, 249, 248, 250, 71, 87, 231, 247, 28) into a new multi-floor level (areas 45/106/27/44 alternating, SCY 96).
  Both fell there within 60-230 steps pressing DOWN near a drop (area 85 scx 144 and area 106 scx 149), respawned
  at the bruiser screen and lost the second fight. Five of 18 bruiser fights produced the kill bonus. The two
  death states are saved as `ignored/m3post-w038-a85.state` and `ignored/m3post-w045-a106.state`.
* **The boss-room collapse is load, not code.** Mission-1 boss win rate against the mean model latency measured inside
  the boss room: v46 4/8 @1.09 s, v47 15/17 @1.26, v56 5/8 @1.23, v53 6/13 @1.34, v49 6/11 @1.32, v48 4/12 @1.52,
  v59 2/18 @1.53, v61 0/14 @1.64, v62 0/11 @1.68, v63 0/17 @1.80 (p90 up to 3.2 s). The two local fights that won
  ran at 0.5 s. Confidence and request success are flat (0.92-0.94, 100% ok), so the model server degrades its
  decisions under concurrency rather than failing; since v57 two 50-worker jobs have usually shared the four
  replicas. The v47 advice text (v63) did not help on its own: at 1.8 s the model followed the kick advice (20-25%
  'b') and landed almost nothing. New rule: one decision run at a time, 40 workers, and watch boss-room latency as a
  health metric (target <= 1.3 s). v62 stopped now to unload v63.
* **v63 (40 min) → v64 "bridge":** the local fixture run beat the bruiser a third time and captured save states of
  the new level (`ignored/save-a{45,106,27,44,234,84,85,86}-s*.state`): mission 4's bridge, SCY 92-96, a walkway at
  feet 78-86 with void (tile 362) two rows below; holding DOWN from it is a fall in 6 steps (both v62 deaths there).
  The ledge detector now also keys on the void tile when the screen is vertically scrolled, so the bridge reads as
  a ledge: 'down' is vetoed, the LEDGE text applies (now worded for any drop), and mission-2 walkways (feet 66-74)
  and mission-1 streets are unaffected (checked on 13 saved states). v64 is the first run under the new load rule:
  one job, 40 workers; v63 stopped (0/17 boss at 1.8 s latency).
* **v64 (25 min) → v65 "boss-autopilot":** at 1.49 s latency with 40 workers alone the boss is 0/9, and with the v47
  wording the model now kicks 90-150 times per fight; kicks never kill him. The knockdown chain needs uninterrupted
  punches, which neither the model nor the v59 autopilot (step-out at 14 px) sustained. Env benchmark, three lives,
  stand-in model every other step: autopilot with NO step-out and the 14-24 px window clears the boss 5/8 (26 hits,
  3.8 knockdowns, 2.0 lives lost), step-out 14: 3/8, window 20: 2/8. v65 turns the heavy autopilot on in area 48
  with `AUTO_STEP_OUT["boss"] = 0`; bruiser settings unchanged; bridge ledge from v64 kept. Single 40-worker job.
* **v65 (30 min):** boss autopilot without step-outs 0/6 on the cluster (42 hits but only 3 knockdowns in 700 steps;
  half the steps were approach and lane-alignment presses on his jittering dy) against 5/8 in the env benchmark.
  A follow-through rule (keep punching 3-5 steps after a landed hit regardless of his dy) was worse in the bench
  (2/8) once a bookkeeping bug was fixed: the heavy-tracking dict was rebuilt every step, so the "just hit" flag
  never survived to the next step (this also silently disabled v62's combo veto). The dict is now updated in place.
  Note on run design: every run mixes three start states (mission 1, mission 2 and a mission-3 save) so that later
  missions get iteration time without replaying the first 45 minutes; the end-to-end measure is the mission-1 start.
* **v65 (45 min) → v66 "bridge-start":** mission-3 starters 2101 px mean, 2403 best (record). The bruiser dies more
  than it looked: 12 of 20 v65 fights end with the 4th knockdown and the ~7000 bonus (per-life gains such as
  {30:16, 200:4, 7000:1}); the players then lose their remaining 2-5 health to the thugs that spawn on the same
  screen and never reach the bridge (v62's two did). Env replay from the bruiser state: kill by step ~80, bridge
  (areas 27/86) by step 160, then pinned at the right wall (x 149) by two thugs and a second heavy. Mission-1 boss
  1/7 with the no-step-out autopilot. v66 adds `save-a45-s00117.state` (bridge entrance, 5 health, 1 life) to the
  start pool so mission 4 gets iteration time; 40 workers, four start states.
* **Mission 4 bridge, first look (v66 bridge-start workers + replays):** from the entrance save (5 health, 1 life)
  every run dies by melee within 100-400 steps, model and scripted alike. The walkway scrolls freely through areas
  45/106/27/44/234/84 and locks at area 86 (scx 152) in front of a stone wall with an arched opening: thugs come two
  at a time without end (25% hit rate on them, 1.8-3.6% damage per step from the one ahead) and a tall enemy
  (sprite 30-33 px, the heavy flag) joins in, so it is very likely mission 4's boss arena. Pressing UP at the wall
  does nothing; the player cannot even reach x 128 before being knocked back. Extra life: the game grants one at
  20,000 points (seen 11 times in traces; none at 40,000; max score so far 47,130). Health never refills except on
  death, so a full run's budget is 3 lives + the 20k life, and the mission-1 boss currently costs ~2 of them.
* **Correction (v66):** the ~7000-point bonus at area 46 is awarded on ENTERING the bruiser screen (mission-3 clear
  bonus), not for the kill: v66 arrivals with a single hit show it too. Real kills are the fights with 16+
  increments followed by new areas: v62 2/18, v65 0/20 (two fights reached 20 and 24 increments and still died to the
  follow-up thugs), v66 1/12 so far (w006: 19 increments, then areas 36, 249, 248, 72, 88, 247, 28 and the bridge).
  Mission-3 starters 2131 px mean, 2397 best. Bridge-start workers spend 60-130 steps at the wall and die with 28-41
  hits landed; the heavy autopilot was active for 14-33% of those steps.
* **v66 (55 min) → v67 "lane-text, true-start-heavy":** eight low-latency local boss fights settle the boss-room
  configuration: shipped autopilot 0/2, autopilot + follow-through 0/2, model-only with the v47 wording 0/2,
  model-only with the lane-first wording and the chain-mechanic note 1/2 today and 2/2 earlier (v62 code): 3/4.
  v67: autopilot off in area 48 (`DD_AUTO_BOSS=0`), `DD_BOSS_TEXT=lane` default, bruiser autopilot unchanged, bridge
  ledge kept. Start mix 40% true start (mission 1), 20% each mission 2, mission-3 save, bridge entrance; one job,
  40 workers. v66 stopped (15 deep workers left) to keep latency down. The boss remains latency-sensitive: v56 won 5/8
  with this wording at 1.23 s; the cluster alone runs ~1.5 s with 40 workers.
* **v67 (30 min) → v68 "20 workers":** the lane-first wording that won 3/4 local boss fights at 0.5 s latency is 0/10 on
  the cluster at 1.38-1.47 s (same arrivals: 3 lives, 4-5 health). Third configuration whose boss outcome flips with
  latency alone; wins so far: 15/17 @1.26, 5/8 @1.23, 3/4 @0.5; losses: 0/10 @1.44, 0/60 @>=1.5. v68 is identical to
  v67 at 20 workers to push the four replicas under ~1.3 s. If the boss comes back, the end-to-end plan runs at
  20-25 workers and the next lever is more model replicas rather than more game logic.
* **v68 (20 workers, 4 replicas):** mission-1 boss 2/6 at 1.34-1.39 s boss-room latency — the first cluster wins since
  v56, with the lane-first wording and no autopilot. Confirms the latency envelope (wins at <= ~1.4 s, none above).
  GPU pool being scaled 4 -> 12 g6e.2xlarge (user approved the budget); the vLLM deployment follows the GPU node count.
* **v69 "scaled" (12 GPU nodes, 11-12 vLLM replicas):** same code as v67/v68 (lane-first boss text, no boss autopilot,
  bruiser autopilot, bridge ledge), 50 workers, start mix 40% true start / 20% mission 2 / 20% mission-3 save / 20%
  bridge entrance. First run with the headroom to keep latency ~1.0 s at full width; the boss win rate and true-start
  distance are the metrics to watch.
* **v69 (12 replicas, 50 workers) → v70 "hybrid text":** at 0.92 s boss-room latency the lane-first wording is still only
  1/12 (v68 at 1.35 s: 2/6), so latency was necessary but not sufficient. The three local wins all had hits/100 >= 5
  and hurt/100 ~3.0; the losses hurt 3.6-4.9. v47 (15/17) hurt 2.0 with 8% up/down presses: its note told the model to
  dodge off the lane after 2-3 punches. The v63 revival of that text failed because its "b hits harder" wording made
  the model kick (kicks never knock him down). v70 uses a hybrid: v47's punch-then-dodge rhythm, punches only, lane
  alignment within 2 px, no 20-hit counter (`DD_BOSS_TEXT=hybrid`, default). Otherwise identical to v69.
* **v70 (12 replicas, 50 workers, hybrid boss text): true-start record 6077 px.** Mission-1 boss 4/16 at 0.85-0.96 s
  (hits/100 2.9, knockdowns/100 1.1 — up from 0.3-0.6 — hurt/100 3.1): the punch-then-dodge rhythm doubled the
  knockdown rate. Worker 41 (mission-1 start) cleared the boss without losing a life, banked the 20k extra life in
  mission 2, crossed the forest and reached the bruiser with four lives; it then lost all four between the bruiser
  screen (2874, 5040 px) and the forest re-runs (3969, 6077 px). Worker 47: 4386 px, boss cost one life, two bruiser
  deaths. Mission-2 starter best 4746, mission-3 best 2463 (record). Video of worker 41 saved to
  `~/Documents/double-dragon-v70-w041-true-start-6077px.mp4`. The bruiser is now the gate for the complete run.
  Operational: the default oc context was switched to another cluster mid-session; all cluster commands now use
  `KUBECONFIG=~/.config/rosa-rl/kubeconfig-rrvnz`.
* **Model-service probe (12 replicas):** a real decision request served by an idle pod takes 55-170 ms; identical
  requests are answered from cache in ~64 ms with identical probabilities (which is why identical local fights
  replay identically); the same boss-room request sent 16 times sequentially and 16 times concurrently to one pod
  returns the same choice every time ('a'), with the concurrent batch finishing in 1.2 s. So the service does not
  degrade its answers under load; the run-level "mean latency" (cumulative over the run, inflated late by stuck
  workers in think mode and by queueing behind other workers' requests) was a weak proxy and the latency-to-win
  correlation across runs was most likely coincidence on small samples. With 12 replicas the queueing is gone
  (0.9-1.2 s at 50 workers). The mission-1 boss is a ~25% fight with the hybrid text; the four-life budget makes
  that workable. The sidecar forwards to its own pod's vLLM, so per-pod imbalance comes only from client round-robin
  timing.
* **v71 "volume":** v70 configuration unchanged (hybrid boss text, bruiser autopilot, bridge ledge, 40% true starts),
  50 workers, launched alongside v70's deep workers to accumulate true-start runs; the gates now are the bruiser
  (each attempt ~15-25%) and the mission-4 arena.
* **v72 "chase" (A/B vs v71's mission-3 starters):** the record run's two "forest" deaths were on the bruiser screen too
  (area id 11 there): the player ends pinned at the LEFT edge (x 10-18) with him 18-40 px ahead — his knockbacks and
  the autopilot's step-outs (always away from him) drift the player ~100 px left over a fight, the mirror of the
  winning geometry (him pinned at the right edge). Also: dying at the bruiser respawns a few screens before him, not
  at the forest start (re-run ~1100 px). v72 runs 30 mission-3 starters with `DD_AUTO_FLIGHT=chase` (close the gap
  while he flies from a knockdown, which also re-centres the player) against v71's align default. Submit script
  now passes DD_AUTO_FLIGHT / DD_BOSS_TEXT / DD_AUTO_BOSS through to the workers.
* **v73 "walk-in":** v71's nine bridge-start episodes all died pinned at the left edge of the walkway (x 11-21) with a
  thug 14-19 px ahead; being adjacent to a thug ahead (<= 12 px) costs 9.2% health per step there against 1.8% at
  13-28 px, and the model presses RIGHT half the time (the LEDGE text says move right). New `walk_in_veto`: a step
  toward an ordinary enemy that is already level and within 16 px becomes the punch (or the turn to face it).
  Stand-in benchmark, 6 seeds × 500 steps: forest ledge health lost 3.2 → 0.8 (distance 234 → 315), forest d500
  6.5 → 3.7 (118 → 229), bridge entrance 3.2 → 1.8. v73 = v71 + walk_in (`DD_WALK_IN=1`), 30 workers, true-start
  heavy; v71 (align control) and v72 (chase) keep running.
* **18:11 UTC outage:** growing the Ray worker group 4 -> 6 while jobs ran made Kueue recreate the RayCluster (head
  restart, workers deleted); v71, v72 and v73 died after 5-10 minutes (partial episodes are in MinIO). Cluster now 6
  workers / 168 CPUs. Relaunched as v74 (volume, 50 workers, walk-in on), v75 (30 mission-3 starters, chase) and
  v76 (30 mission-3 starters, align control), all on 12 model replicas.
* **v77 (heavy-flag gate):** v74's bridge-start episodes showed no walk-in benefit (hurt/100 2.24 vs 1.50) because
  `heavy_autopilot` fired 180 `(right, down)` overrides there. Cause: the radar flags any sprite group taller than
  34 px as "heavy", and two thugs stacked vertically (one jumping over the other) merge into one such group for 1-9
  steps: 168 false heavies in 3910 forest/bridge frames (areas 11, 12, 27, 44, 84, 86, 106), always with
  `enemy_count == 1`; the real bruiser is flagged for hundreds of consecutive steps and only in area 46. Fix:
  `reflex.HeavyGate` strips the flag outside area 46 until it has held 6 consecutive steps (`DD_AUTO_MIN_SEEN`),
  applied once per env step before the prompt text and every reflex read the radar. A first attempt (gate inside the
  autopilot only, no area bypass) cost bruiser kills (close-state bench 5/8 -> 3/8, deterministic seeds); with the
  area-46 bypass the bench is back to 5/8. Stand-ins with the autopilot in the chain, walk-in off -> on: bridge start
  hp lost 4.0 -> 3.3 (dist 129 -> 143), forest ledge 2.5 -> 1.0 (280 -> 312), forest d500 6.3 -> 5.0 (145 -> 270).
  v77 = 40 workers, 20 bridge starts / 10 mission-3 / 10 true starts, alongside v74-v76 (81 CPUs were free).
* **Interim results (19:10 UTC):** v75 (chase) 1/22 and v76 (align) 2/24 mission-3 starters got past the bruiser
  into mission 4 (detector: areas after the first 46; the earlier `levels_visited` intersection over-counted because
  areas 44/45/106 also occur in missions 1-2) - chase vs align is a wash, align stays. The three mission-4 entries
  died on the bridge end (84/85) or in the arena (86). v74 true starts so far: 3/12 past the mission-1 boss, 2/12
  reached the bruiser (4211 and 4118 px, both died there). **Bruiser HP byte found:** `0xC595` (mirror `0xC5DA`),
  42 at every fresh bruiser state; jab = 2, kick (b) = 3, third-hit knockdown = 5, so a kill is ~14 landed hits.
  **Elbow found:** pressing A with the bruiser behind you lands +200 / 3 damage at 7-20 px every 13-18 steps (the
  lane veto had been converting every back-turned attack into a turn, so it was never used); damage rate ~ equal to
  the jab chain, safety not better in the crude probe (3-5 hp lost per 8 elbows).
* **v78 (bruiser HP telemetry):** the runner now reads the bruiser's HP byte (`0xC595`, 42 fresh) every step he is
  on radar in area 46, puts `bruiser_hp` into the radar so the prompt says "The bruiser has N/42 health left (each
  punch 2, the knockdown blow 5)", and records `bruiser_hp_min` and `bruiser_hp_at_life_lost` per episode. Kill /
  no-kill was too coarse to compare fight tactics at 5-15% kill rates; damage dealt per attempt is the metric from
  here. Player HP byte is `0xD098` (248 = 5 HUD ticks; HUD reader kept). HP persistence across a player death is
  still unmeasured: the scripted walker could not get back through the river on the respawn life (saved
  `ignored/m3b-second-life.state`, lives patched to 2 for the test). v78 = 40 workers, 30 mission-3 starters + 10
  true starts, HeavyGate + HP text, align autopilot.
* **Elbow autopilot probe (`DD_AUTO_ELBOW=1`, off by default):** elbow instead of a second step-out when already
  facing away. Env bench: close 5/8 -> 5/8 (never triggers there), arrival 1/8 -> 1/8 (hits 2.4 -> 11.8), far 0/8 ->
  0/8 (one offset landed 74 elbows for +14,800 points without a kill). Each elbow does take 3 HP (42 -> 39 -> 36 ->
  33 verified on the byte), so the kill problem is still reaching and surviving him, not the move. Kept as an option.
* **v77 read-out (33/40 done, 19:30 UTC):** true starts 5/5 past the mission-1 boss (v74: 3/12), all five truncated
  at the 4096-step cap with 2 lives left at 2564 px mean (best 3018) - the cap, not the game, ended them. Bridge
  starts live 304 steps mean vs 184 in v74 (dist 201 vs 156) but still all die; mission-3 starters 0/13 past the
  bruiser. v79 = 40 true starts with `--max-steps 10000` to find out how far an uncut run goes.
* **v80 (false "stuck-up" on scrolling ladders):** v77 true starts w008/w019/w023 all parked in mission-2 area 16 on
  the ladder that runs through a vertically scrolling section: the sprite stays at (123, 99) while SCY scrolls
  196 -> 168, so the stuck detector (x and horizontal scroll only) fired "stuck-up", the prompt said "do NOT press up
  again", and the model obeyed until the step cap. Emulator check: 24 "up" presses from the saved state climb off
  the ladder. The detector now also requires no vertical sprite motion and no vertical scroll. v79 (launched 19:30
  without the fix) stopped; v80 = 40 true starts, 10000-step cap.
* **v81 (river-exit lane):** v78's bruiser telemetry showed 15/28 mission-3 starters never met him: they lose a
  life 3 steps after entering area 46 with full health and no enemy on screen, at x~106 - a fall at the boundary,
  once per life. Entry-fall counts per run: v62 0, v75 1, v76 1, v77 16 (13 eps), v78 38 (28 eps). Lane at area-46
  entry decides it: feet <= 92 never fell, feet >= 97 always; v76 entered at 83-91 in 26/27 episodes, v78 at 91-105.
  v76 only had the lane because thug pairs mis-read as "heavy" made the autopilot align upward through areas
  246/52/53 (exec_up > model_up in v76, equal in v78) - the v77 HeavyGate removed that accidental help. Fix:
  `river_exit_veto` (areas 52/53, no enemy within 24 px -> press up until feet <= 90). v81 = 30 mission-3 starters
  with it plus `DD_SAVE_AREAS=52,53` to capture fixtures; v80 (true starts, launched before this) will still fall
  at 46.
* **Bruiser fight solved offline by search (20:30 UTC):** `/tmp/bruiser_search.py` - beam search (width 24) over
  the 7 single-step actions from a saved state, deterministic emulator, objective = 10 x his HP damage (byte
  0xC595) - 12 x our HP loss, win = HP byte 0 with no life lost. Kills with ZERO damage taken from m3b-far (139
  steps), m3b-right-end (112), m3b-arrival (~86) and m3b-pinned (~75); m3b-far2 reached 12/42 within depth 121
  (deeper run pending). Tactic in the winning sequences: jab (2) or JUMP KICK (4 damage - the strongest move, never
  used by the autopilot) at 15-22 px, then retreat left 10-25 steps while he recovers, repeat; occasional elbow (3).
  The scripted autopilot's best is 5/8 from the close state with damage taken; the search is 4/4 clean.
* **v82/v83 (online fight planner, `DD_PLAN_FIGHTS=46,48`):** `app/fight_planner.py` runs the beam search on the
  worker's own emulator (save/restore around `env._run_action_on_emulator`, so timing is identical to env.step),
  restores the pre-search state and the runner replays the sequence through normal steps (traces/video intact, no
  model calls meanwhile). Through a real DDEnv: bruiser from m3b-far killed in 101 steps / 0 damage (149 s search),
  mission-1 boss from area48.state killed in 78 steps / 0 damage (103 s). The boss uses the same HP bytes (44 HP,
  jab 2, knockdown 5; refills on a lost life). Guards: trigger only with the heavy/boss within 120 px, max 6 plans
  per episode, 60-step back-off after a failed search, plan dropped on a lost life, `fight_plans` in the episode
  record. This is lookahead planning with save states, not learning - flagged as such; the searched sequences are
  also expert trajectories for the PPO/BC track. v82 = 40 true starts (10000 cap), v83 = 30 mission-3 starters
  with arena fixtures saved (`DD_SAVE_AREAS=86,84,85`).
* **v81 result:** entry falls 0/30 (v78: 21/28); all 30 entered area 46 at feet 87-90 and met the bruiser. Damage
  dealt per attempt without the planner: 1 kill, one left at 1 HP, median 11 of 42, 7 never landed a hit.
* **v83 planner post-mortem:** plans fired (2 per bruiser episode) and the replays were exact (verified locally
  step-for-step against the planned trajectory, 98/98 states), but no kills: the objective averaged the two enemy
  HP slots with integer division, so with an empty second slot "HP 0" was reached at 1 real HP and the plan ended
  with him alive; the fight also scrolls the area id from 46 to 11, so neither a re-plan nor the HP telemetry
  fired. v84: the planner targets the slot with the most HP at the root; the runner stays "armed" after the first
  plan and re-plans wherever the fight scrolls until the target's byte is 0 (`fight_kills` recorded) or the life is
  lost. Worker-side Python logging never reaches the Ray job log; the planner prints to stdout instead.
* **v84/v85 first hour (22:40 UTC):** corrected planner on the cluster: v84 19/19 bruiser searches found kills
  (93-128 steps, ~220 s each, no re-plans); v85 14+ boss searches all found kills (89-102 steps, ~200 s). The
  "second plan at 44 HP" lines in the boss room are not failures: area 48 holds TWO 44-HP bosses in sequence
  (local runner reproduction from area48.state: both plans replayed exactly, `fight_kills: 2`). Pre-plan states are
  now uploaded (`plan<N>-a<area>` save states) and each plan's action sequence is printed, so any divergent fight
  can be replayed offline. Client and env action orders verified identical.
* **v84 result (first 11 episodes, 23:00 UTC): the bruiser gate is closed.** 22/22 planned fights killed him
  (2 per episode: kill, die in the mission-4 arena, respawn at the mission-3 start, kill again). Every episode then
  died in area 86 with 1 life and <= 5 HP. Arena mapping with an invulnerable scripted brawler: the arena is NOT a
  lock - after two thug kills the screen rolls on into further bridge areas (106, 44, 27, 234, 244-246) and the
  brawler fell at scroll ~233 (no jumps in the script). Its enemies: thug pairs (12 HP, slot 1 cycling), a 32-HP
  tall enemy (slot 0) and, on re-entry, a 42-HP bruiser. The planner kills the 32-HP enemy from the fixtures
  (147/135 steps, <= 1 HP lost). v86 adds `DD_PLAN_ANY_ENEMY=86`: in listed areas any enemy within 60 px triggers a
  plan (the planner's max-HP-slot target handles whichever enemy is strongest).
* **Mission-4 cave mapped (23:20 UTC):** past the arena the walkway (feet 108, areas 244/245/234 at SCY 68) ends
  at a one-column void against a rock wall: every jump window and every lane falls there (5/24 v84 episodes died
  in 245). The way on is UP: a 4-tile-wide chain (background tiles 294/295) at screen x 91-123 is climbable (up
  x30: SCY 68 -> 0, player lands at feet 75 on an upper walkway). The radar's `LADDER_TILES` now includes 294/295
  (detected at x=107, span 20-92; no false ladders on mission 1-3 fixtures), so the model gets the LADDER text and
  the ladder reflexes apply. The upper walkway has its own hole ~60 px on (invulnerable brawler fell at feet 123).
* **Cave route + v87 (23:50 UTC):** the mission-4 cave past the arena is a vertical maze. Route found and
  scripted: climb the chain (tiles 294/295, now a radar ladder) to the upper walkway; cross its one-tile holes
  with the RUNNING jump kick `kick_right` started 6-10 px before the edge (`cave_hop_veto`; plain A+B carries only
  17 px and drops into the non-solid ladder column; the reflex holds right while airborne); the next screen is a
  fight lock (thug pairs, 8-10 HP) and a thug's grab-and-throw at the walkway edge is a fall death - the same
  lesson as mission 2's hole. So fights in mission 4 go to the planner: `DD_PLAN_ANY_ENEMY=84,85,86,106,233,234,
  44,45,27` gated by a `mission4` flag (set by the intro areas after the bruiser, cleared on the mission-3 respawn)
  because those area ids recur in missions 1-2. Plan cap 20 per episode. v87 = 30 mission-3 starters with all of it.
* **Housekeeping (23:20 UTC):** the cluster token expired (re-login via the stored admin creds). Jobs v74, v77,
  v80, v82, v83 were still "RUNNING" hours after their useful episodes finished because a few workers never
  terminate (stuck episodes that do not reach the step cap - to investigate); stopped them, CPU use 124 -> 96/168.
  Finished episodes of all of them are in MinIO.
* **v88 (planner budget):** two v87 bruiser searches returned "no kill found in 421s" - the 420 s wall-clock cap,
  not the search: with v85+v86+v87 planning concurrently each search ran ~2x slower than the 220 s kills of v84.
  The depth (260) is now the real limit and the time cap is a 1500 s safety. Small targets (<= 20 HP) use half the
  beam and depth 140 (arena thug: 14 s instead of ~70 s, same plan). v87 stopped (25 min in), relaunched as v88.
* **00:05 UTC interim:** v88 (26 min): 19/19 bruiser kills, 12/13 kills of the 32-HP bridge enemy in area 106,
  1 arena miss; lives mean at the bruiser 1.67 (was 1.00 - fewer deaths on the way). v85 true starts: 33 boss
  kills, 15 bruiser kills, best 5797 px; its 5 finished episodes all stalled in the mission-2 loop 26 -> 66 -> 25
  -> 106 (the elevator/lower-walkway maze) until the lives ran out - the next navigation gate for true starts.
* **v89 (pair targets + safe exit, 00:40 UTC):** v88's first 4 cave episodes: 11 of 13 mission-4 life losses were
  fights (HP to 0 at the screen's left edge), not falls; the plans themselves all killed, but a kill plan ended the
  instant the target died - with the second thug mid-swing - so the next search began from a doomed 1-HP state
  ("no kill found in 0s") and the model-controlled steps in between bled the remaining HP. Planner changes: a thug
  pair (both slots <= 20 HP) is a single target (sum of both slots), and a kill only counts once no enemy on the
  lane is within 28 px (`DD_PLAN_SAFE_EXIT_PX`). Lock-screen fixture: both thugs dead in 33 steps / 19 s / 0 HP
  lost, exit at 28 px. v89 = 30 mission-3 starters with it; v88 keeps running as the baseline.
* **v90 (no-retreat plans + partial plans, 01:20 UTC):** finished mission-4 episodes were treadmills: each kill
  plan retreats left for dozens of steps (the search's kiting habit), the model walks right, thugs respawn, repeat -
  bridge areas 44/106/27/234/84/85/86 cycle with no net progress; 2 true starts hit the 10000-step cap that way.
  Planner objective gets `+0.4 x px gained rightward` for mission-4 fights only (bruiser/boss keep kiting): arena
  32-HP fixture 149 -> 59 steps, 166 -> 58 s, HP 5->4 -> 5->5, backward scroll 41 px -> 0; lock-screen pair same
  HP, longer plan. Also: when no kill-and-clear is found within the depth (v89: 489 s, 30-HP pair at 4 HP), the
  planner returns the best leaf's first 60 steps as a partial plan and the runner re-plans from there. v88 stopped
  (baseline recorded: 103 kills / 107 searches, all misses at 0 HP); v90 = 30 mission-3 starters; v89 continues.
* **01:30 UTC interim:** v89 (pair+safe-exit, no progress weight): 96 kills / 99 searches, but its 7 finished
  episodes are the same bridge treadmill (tails 44/106/27/234/84, 5 plans each, no cave entry). v90 (progress
  weight 0.4) is 15 min in. v85 true starts: 15 done, 5 truncated at 10000 steps (2 mission-3/4 cycles, 3
  navigation stalls in missions 1-2: warehouse exit area 84, area 16/17 loop).
* **v91 (true starts, full kit, 01:40 UTC):** the mission-1 "warehouse exit" stall (v85 w030: 8956 steps in area
  84, 7877 punches) is a 7-HP thug the model cannot land hits on and cannot walk past. The planner kills him in 6
  steps / 4 s (`DD_PLAN_ANY_ENEMY_ALWAYS=84`: any-enemy planning in area 84 in every mission). v91 = 40 true starts
  with planner on 46/48, mission-4 any-enemy planning with progress weight 0.4, pair targets, safe exit, partial
  plans, chain ladder, cave hops, 10000-step cap.
* **v92/v93 (clearance fix, 01:55 UTC):** the safe-exit test counted the dying target's own sprite (it lingers
  within 28 px), so no state qualified as "kill and clear" and searches ran to full depth: v91 warehouse thugs
  150-200 s (4 s locally), v90 arena 400-500 s with partial plans. The clearance test now applies only while an
  untargeted enemy still has HP. Local: bruiser 99 steps / 100 s, boss 89 / 85 s, both kills. v90 and v91 stopped;
  v92 = 30 mission-3 starters, v93 = 40 true starts, both with the full kit.
* **v94 (02:05 UTC):** v93's first warehouse plan ran 462 s to a partial with score 706: area 84 planning carried
  the mission-4 progress weight, so the search preferred walking right over killing, while the warehouse exit is
  to the left. Progress weight now applies only when the mission-4 flag is set. v93 stopped at 10 min, relaunched
  as v94 (40 true starts).
* **v95 (02:40 UTC):** v94's warehouse plans were all ~620 s partials: the room's thugs respawn, so "kill the pair
  and be clear" is rarely reachable. Added a `simple` planner mode for the always-on areas (strongest single
  target, no clearance test): warehouse thug dead in 4 steps / 3 s. Unsuccessful searches now stop at depth 160
  and return the partial. v94 stopped, relaunched as v95 (40 true starts).
* **v96/v97 (03:00 UTC):** v92's 9 finished episodes: 6 die in the arena (86) at 1-4 HP against its 42-HP bruiser
  (4 partial plans ~500 s), 3 at the mission-3 respawn; 2 climbed the chain. Trace split: 67% of mission-4 steps
  are plan replays and those are 52% "left" / 18% "right" even at progress weight 0.4 (model-controlled steps: 56%
  right) - the planner still kites backwards; HP lost 20 ticks in plans vs 26 in the model's third of the steps.
  Changes: mission-4 progress weight 1.0 (bridge fixture: 145 -> 57 steps, 2 -> 1 HP lost), any-enemy plans start
  at 90 px instead of 60. v92 and v95 stopped; v96 = 30 mission-3 starters, v97 = 40 true starts.
* **v98 (03:25 UTC):** v97 warehouse plans: 35 kills at 34 s but 13 failed searches and 5 partials - simple mode
  targeted the max-HP slot, which can belong to an enemy the player cannot reach (upper lane). Simple mode now
  counts ANY enemy dying as the win. Fixtures unchanged (warehouse 4 steps, bruiser 99, lock pair 138). v97
  stopped, relaunched as v98 (40 true starts). v96 (mission-3 starters) continues: 16 bruiser kills, mission-4
  plans now 40-145 s.
* **v96 read-out (04:05 UTC):** 11 episodes: 8 entered the arena with 5-6 HP, 7 died there (~200 steps each, 4
  fights: two 21-HP thugs, the 32, the 42); the one survivor (w019) left with 1 HP and died on the cave's lower
  floor because `in_forest` stays True for mission-3 starters, so the river-gap jump reflexes fired in the cave and
  hopped him into the void. Fixed: `in_forest` is off once the mission-4 flag is set. Arena loop test with the
  planner + a walking harness: all variants died in the spawn gaps between fights, not in the fights -> an
  "escape" planner mode (win = the screen scrolls on / 150 px banked, depth 420, damage to anyone counts) is
  being tested on the arena fixtures.
* **v99/v100 (04:20 UTC):** v98 lost 39/40 true starts in area 66 (mission-2 start) at full HP: area 73 is in BOTH the
  mission 3->4 cut-scene list and the mission 1->2 transition, so the mission-4 flag switched on in mission 2 and
  enabled `cave_hop_veto`, which hopped players off the right edge of area 66 (model "left" overridden to
  "right"/"kick_right"). Fixes: the flag now also requires having met the bruiser (area 46) in this life, 73 is
  dropped from the list, and the cave hop is gated on the flag. Escape mode result: from the arena fixture one
  330-step plan (320 s) clears all four waves and exits to area 246 at 5 -> 3 HP (second fixture: partial). Escape
  mode is on for area 86 (`DD_PLAN_ESCAPE=86`). v96 read-out: 17 done, 10 died in the arena, 6 at the respawn.
  v96 stopped; v99 = 30 mission-3 starters, v100 = 40 true starts, both with escape mode + the river/flag fixes.
* **v101 (05:50 UTC):** v100 true starts all looped warehouse (84) <-> boss room (48) for 40+ min: after the planner
  kills the warehouse thug the model walks out the bottom-left into 48, a 36-44 HP boss respawns, gets planned and
  killed (~150 s), the model walks back into 84, repeat (7-8 plans per worker, zero distance). A live local run
  against the cluster model did pass (84 -> 48 -> 106 -> 89 -> 36 -> cut-scene -> 66 in 1200 steps), and v85
  without warehouse planning passed 37/40. Reverted: no planning in area 84 outside mission 4 for true starts, and
  escape mode off (v99: 13 exits vs 35 partials + 17 misses at ~500 s each; 8/11 finished episodes still died at
  the arena). v100 stopped; v101 = 40 true starts. v99 continues to finish its escape-mode sample.
* **v102 (07:05 UTC):** v101 (no warehouse planning) still stalled at ~1390 px: its two finished episodes show
  repeated silent planner misses in the boss room (`plan1-a48` saved 3x with no log line) - the dead first boss's
  sprite keeps the trigger firing while both HP slots read 0, the planner returns nothing, and the 60-step back-off
  leaves the model fighting the second boss alone (3 deaths in area 48 per episode). Only two enemy HP slots exist
  (third address reads 0) and the boss's slot is live on entry, so the fix is a 6-step retry when no slot has HP.
  v101 stopped, relaunched as v102 (40 true starts).
* **v103 (09:30 UTC) - the true-start stall was a ladder oscillation:** v102 finished with all 40 true starts
  truncated alive at 1310 px mean; 68% of all their steps were in area 66 (mission-2 start) at x~128, feet 64-72,
  pressing up and down ~4000 times each at full HP. Cause: the model, told "SCREEN LOCKED ... climb DOWN", pressed
  down, while `rung_veto` turned its attacks into "up" (a thug level with the player, ladder top off-screen, 1-px
  tie-break). Fixes: on the rungs, a level thug means climb toward his floor (up if dy <= 0) regardless of the
  model's up/down; up/down are overridden when the only thugs in reach are on the other side; wasted attacks head
  for the thugs at any height; the prompt now says "press up a few steps and punch him on this walkway" in that
  case. All five ladder fixtures resolve (climb off, kill, full HP). v103 = 40 true starts with the fix.
* **v104 (10:05 UTC) - model-service hot-spotting:** v103 (ladder fix) got past area 66 (mean 1619 px, vertical
  520, lives 2.36 at 39 min) but its step rate collapsed to ~400 steps/min for 40 workers with the model service
  healthy on paper (errors 0, cumulative latency 0.99 s). Real requests to an idle replica took 3.4-4.3 s; one
  pod's vLLM log showed 31 running + 21 waiting requests while eleven pods idled. The client round-robins the
  resolved replicas but every worker starts its counter at 0 and the workers step in lockstep (deterministic plan
  replays re-synchronize them), so all 40 hit one replica at a time. Fix: random per-client round-robin offset.
  v89 finished meanwhile: 30 episodes, mean 2100 px, 73% truncated (the bridge treadmill, pre-progress-weight).
  v103 stopped, relaunched as v104.
* **v105 (10:10 UTC):** correction to the v104 note - the client already offsets each worker's round-robin counter
  by its index; the pile-up (52 concurrent requests on one replica, eleven idle) must come from the counters
  drifting into alignment, since plan replays pause a worker's requests for ~100 steps at a time. Fresh requests to
  an idle replica take 0.22-0.32 s, so the service is fine. The client now picks a random replica per request. v104
  (launched with the unchanged client) stopped at 10 min; v105 = 40 true starts.
* **v106 (10:40 UTC) - the real slowdown:** py-spy on live workers shows every one blocked in the HTTP post to the
  model while probe requests take 0.3 s and all twelve replicas run at ~0.9 cores. The workers' own requests are
  slow because the runner switches the model into "think" mode whenever the stuck detector fires
  (`DGEMMA_THINK_WHEN_STUCK`, default 1), and mission 2's ladders trip it constantly; think-mode generations take
  seconds and saturate the GPUs for everyone (v103 and v105 both collapsed to <100 steps/min right as their true
  starts entered mission 2). v106 = 40 true starts with think-when-stuck off (now forwarded by the submit script).
* **v85 final (11:20 UTC):** the 40-true-start run with the first planner (boss/bruiser only, 10000-step cap):
  mean 3285 px, best 7078; 38/40 passed the boss, 14 reached the bruiser, 13 entered mission 4; 30/40 truncated
  alive - 18 of them in mission-2 areas 16 and 5 (the true-start sink), 6 in the cave (244).
* **v107 (11:45 UTC) - two mission-2 stalls from v85/v106 traces:** (a) area 16: the player stands 8 px off a
  ladder's axis, the prompt says "HOLD up", up does nothing (climbing registers within ~5 px: x=37 climbs, x=40
  does not) - 3000 presses per episode; now the advice says "step <side> ONE step to line up" and
  `ladder_align_veto` turns the model's up into that step when 5-12 px off. (b) area 5: blocked at the left wall
  while the "floor ENDS on the right -> way on is LEFT" rule kept saying left (3000 presses); walking right
  reaches area 106 before the drop, so the rule now says RIGHT when the player is against the left wall. v106
  (think-mode off) confirmed throughput: ~4100 steps/min vs <100 before; its 3 finished episodes died falling off
  the lower floor of areas 25/26 (walking left at feet 110-115, no enemy) - still open. v106 stopped at 55 min;
  v107 = 40 true starts with all of the above.
* **v108 (12:30 UTC) - navigation planner:** the mission-2 true-start sink is a closed loop of areas (25 -> 106 ->
  26 -> 66 -> 25; also 13/73/74/15/45/42/43) whose only exit is a long ladder descent in area 66 followed by a walk
  to area 45 and area 5 - unreachable by any left/right policy, missed by the model and by the scripted explorer.
  `app/nav_planner.py` applies the fight planner's trick to navigation: when the runner's no-progress detector has
  fired for 30 decisions (no horizontal or vertical progress, no enemy within 60 px), it breadth-first searches
  macro moves (12/8/4-step walks, ladder align+climb, punch) on the live emulator for any area not visited in the
  last 1500 steps, pruning death and >3 HP ticks lost, and replays the path. Saved loop states: area 25 -> area 5 in
  36 steps (3 s), area 66 -> area 5 in 76 steps (20 s), area 5 (left-wall stall) -> area 16 (112 steps); area 16's
  off-axis ladder is handled by `ladder_align_veto` instead. 300-step cooldown, 12 nav plans per episode,
  `nav_plans` in the episode record, pre-plan state uploaded as `nav<N>-a<area>`.
* **v108 launched (12:35 UTC):** local smoke test from the loop state: 5 nav plans, each exiting to a new area (5,
  8/7, 4), no crash after fixing a variable clash - but the model walked back into the loop after every exit. Added a
  guard: for 600 steps after an exit, re-entering any area of the escaped set triggers an immediate re-plan (cap 20
  per episode). v107 (ladder + area-5 fixes, no nav planner) sat at 1583 px mean for an hour and was stopped;
  v108 = 40 true starts with the navigation planner.
* **v109 (13:30 UTC) - the "area id" is a map-chunk index, and that is an exact progress oracle:** v108's first
  hour (40 true starts, no errors, 2067 px mean at 56 min vs v107's 1583) showed the navigation planner firing 212
  times: 119 exits, 93 failures, the failures burning up to 300 s each in areas 26, 106, 13, 15, 73, 84. Reading the
  ordered `levels_visited` of all 40 v85 episodes explained it: the lists are identical across workers, including the
  exact four-fold repeat 26, 66, 25, 106 in mission 2, the eight-fold 16, 17 on the mission-2 ladders and ~50
  alternations of 11, 12 through the mission-3 forest. Independent stochastic agents cannot loop a maze identically,
  so byte 0xE110 is the index of the current map chunk, chunks come in a fixed order, and every deviation in the
  lists is a death (the mission restarts: 89, 106, 36, 73, 74, 13 ... 66). Consequences: (1) the "mission-2 loop
  region" never existed - saved "loop" states advance to the next chunk in 4 steps; (2) v108's looping trigger
  (an id re-entered 3x) fired falsely inside every repeated stretch and then searched for a *novel* id that was 10+
  chunks away; (3) progress can be measured exactly as the index into the canonical sequence instead of pixels.
  `app/progress.py`: `CANON` (336 chunks, v85 worker 15's first pass up to the mission-4 arena, with the branch
  alternatives 36/73, 71/72, 87/88, 231/232, 44/45 seen in other workers) and `ProgressTracker` (follows the index
  with 3-chunk lookahead, falls back to the nearest earlier index on death, `goal_areas()` = the next 1-3 chunks,
  `mission()`); validated: an ordered walk never regresses, and it classifies v85 as 14/40 reaching mission 4.
  Runner: the nav planner's goal is now the next canonical chunk (novelty only past the arena); stall = the index
  unchanged for 600 steps (`DD_NAV_STALL_STEPS`) or the old 30-decision immobility test; the post-exit guard
  re-plans when the index drops below the exit target; pre-plan states are saved once per area (219 of v108's 359
  saved states were boss-room re-plans, so no nav state was ever kept). NavPlanner: enemy HP joins the search key
  (punching used to produce a "seen" state and was pruned - the area-66 ladder-with-a-thug state now exits in 76
  steps), a partial plan (the explored state that changed floor/scroll the most, >= 24 px, no more than 1 HP lost)
  replays when no goal is reachable (area 16: climb to the thug floor, then the model fights), time cap 120 s.
  Telemetry: `canon_pos_max/end`, `canon_regressions`, `mission_max` per episode; summary `canon_pos_max/mean`
  and `episodes_reaching_mission`; the progress line shows `chunk mean/best` and `missions>=2/3/4` counts.
  Fixtures: all 8 mission-2 states (areas 26/66/25/5/16/66-ladder) now yield a plan; previously 3 returned none.
* **v109 launched (13:47 UTC), v108 stopped at 75 min (2112 px mean, best 4432, 0 errors).** Smoke test from the
  area-16 stall state: 1200 steps, 4 nav plans (two area-16 partials, then two forest exits), no exception, the new
  summary fields present. Arena finding while testing offline on `save-a86-s00175` (5 HP, tall 32 + thug 21): the
  fight planner returned a damage-free "partial plan" of 50 lefts that walked out of the arena backwards into
  bridge chunk 44 (and then down into a pit). Two causes, both fixed: the off-radar bonus (2/step while the target
  is not visible, meant for a knocked-down bruiser) paid ~300 points for walking away - now capped at 8 steps and
  granted only after damage; partial plans now require damage dealt. v108 also ran with `DD_PLAN_ESCAPE=86`
  (escape mode counts *any* chunk change as a win, including the exit behind the player) - this explains the
  "86 -> 106, 44, 27 -> death" tail of every v85 arena arrival; v109 runs with escape mode off. `DD_PLAN_HP_W`
  knob added (default 12) for an HP-conserving arena search; `DD_PLAN_DEBUG` now also prints partial plans.
  Resolved: `DD_PLAN_DEBUG` re-simulation of the partial matched a fresh replay step for step (enemy HP 32
  throughout) - the 307-point leaf dealt its damage *after* step 60, so the 60-step partial replays only the
  retreat phase (walk to the left edge of the arena, wait for the enemies to follow). Save/restore round trips and
  cross-process determinism were verified byte-exact along the way. The planner is sound; whether plan/re-plan from
  5 HP clears the four-enemy gauntlet is being measured offline (`/tmp/arena_gauntlet.py`, HP weight 12 vs 40).
* **v109 stopped after 8 min, v110 launched (13:57 UTC):** all 40 workers froze at the same step: the new chunk-stall
  trigger (index unchanged for 600 steps) fired in the opening street, whose first chunk takes far longer than 600
  steps to cross, and each worker then spent ~180 s searching for chunk 241 (unreachable in 24 macros). v110: the
  stall test also requires no new scroll maximum (distance + vertical) for the same window, default 900 steps.
  150-step local run clean. Everything else as v109.
* **Arena gauntlet offline (14:00 UTC), `save-a86-s00175` (5 HP, 1 life, tall 32 + thug 21 on screen):** plan /
  replay / re-plan killed both with **zero HP lost** at HP weight 12 and at 40 (partial retreat to chunk 44, then
  kill in 112 steps, then kill in 34 steps). The HP was lost afterwards by the harness's naive "walk right and wait"
  loop with no enemy on screen (hp 5 -> 1 / a life, i.e. bridge holes). So the mission-4 "HP wall" is not the fights:
  it was escape mode's retreat reward plus unsafe walking between waves. Next measurement: return to the arena with
  the nav planner (goal chunk 86) and wait in place for the second wave (bruiser 42 + thug 21).
* **v110 stopped at 18 min, v111 launched (14:25 UTC):** v110 ran at ~1400 steps/min (v108: 4100). Cause: the
  no-progress trigger fires at step ~196 of every start (the model idles in the opening street; v108 saved 55
  `nav-a15` states for the same reason) and the canonical goal there was {241, 48}: CANON began "15, 241, 15, 48,
  84" and 241 is a one-frame saved-state blip the live tracker never sees, so the real next chunk (the doorway's
  84) was never a goal and every worker burned 120-180 s. Fixes: 241 dropped from CANON (334 chunks), goal lookahead
  = the next 3 *distinct* chunks, the nav time cap enforced inside the level loop, an open-terrain early exit (by
  depth 3 a walk already moved >= 96 px toward the right and the goal is out of range -> return that walk: 13 s
  instead of 120 s on the saved doorway state), partial displacement is signed (no walking back left), and a partial
  nav plan no longer sets the exit guard (it re-planned the moment the walk ended).
  **Arena wave 2 (the real mission-4 wall):** returning to chunk 86 after the first two kills, the second wave's
  two enemies walked in and beat the player to death while *both tracked HP bytes read 0* - the planner never saw
  a target, so the model fought alone. WRAM diff on scoring hits: their HP sits at 0xC57D / 0xC5C2 (13 HP thugs;
  same 0x45 stride, 0x18 below the known bytes - the object table is laid out differently for this wave). Rather than
  hard-code a third address set, `FightPlanner.discover_slots()` finds HP bytes at fight time: from a snapshot, walk
  at the nearest enemy and jab; at the first score gain the bytes in 0xC540-0xC700 that dropped by 2-6 (from 6-64)
  are the HP bytes; restore. Used automatically when the known slots read 0 with enemies on screen. `DD_PLAN_HP_W`
  12 vs 40 made no difference (both killed wave 1 with 0 HP lost), left at 12.
* **v112 launched (14:45 UTC), v111 stopped at 20 min (1171 px mean, 7/40 in mission 2, 0 errors):** the arena's
  second wave is beaten offline. Four hardening passes on the fight planner after the first discovery version:
  (1) `ALT_HP_ADDRS = (0xC57D, 0xC5C2)` tried before discovery (v111 on the cluster found the same pair in the
  boss room between phases, so the layout is not arena-specific); (2) discovery needs two consistent hits (bytes
  that drop at both and never rise between) - single-hit picks were stun timers (0xC5BF, 0xC5CD, 0xC585); (3) a
  walking-in enemy has no HP yet, so damage is measured against the max HP seen on each search path, a kill needs
  the enemy to have shown >= 10 HP (counters at those bytes flicker 0..6 and produced 2-step "kills"), and hits
  only count once he has activated (a flickering counter had produced a 1558-point phantom partial); (4) a cheap
  48-step activation pre-check before searching. Result from `arena-wave2-entry.state` (4 HP, 1 life): 4 kills in
  171 steps, 69 s of search, HP 4 -> 2, no phantom plans; wave 1 earlier: 2 kills, 0 HP lost. Bruiser regression
  (`m3b-arrival.state`): kill in 88 steps, 0 HP lost, unchanged. Beyond the arena the chunk sequence is unknown
  (CANON ends at 86); a novelty-mode exploration from the cleared state is running to extend it.
* **Live arena run (local runner + model, `save-a86-s00175`, 5 HP, 14:50 UTC):** with the v112 planner and
  `mission4` now derived from the canonical index (the old "seen area 46" heuristic was never true for a run that
  starts in the arena, so planning stopped after one chunk change), the runner fought **12 planned fights in a
  row inside chunk 86 with no retreat**: tall 32, thugs 19/22/10/9/2/16, then the arena bruiser (42) appeared.
  The player reached him at 1 HP because four kill plans each spent one tick (weight 12 per tick vs 10 per enemy
  HP: a tick is cheaper than a dozen steps) and died in his fight at step 607. The planner's wave-1 harness had
  shown no difference between HP weight 12 and 40 because those kills were HP-free either way; over a dozen
  fights the weight is the HP budget policy. Testing weights 40 and 80 live next. The old harness also retreated
  to chunk 44 on its first plan because it ran with progress weight 0; the runner uses 1.0 in mission 4 and the
  retreat disappears. The arena is a long stream of thugs ending in the bruiser, not a fixed four; what follows the
  bruiser is still unobserved.
* **Arena mechanics (15:20 UTC, offline):** the arena (chunk 86) is the end of the bridge against a rock tower.
  The dark band beside the player at x 104-119 is a hanging chain (tiles 345/346 over 233/234 - the same chains
  hang from the pillars in `m4-upper-landed`), but it is NOT the exit: up, jump and up+jump under it do nothing
  from the entry state and from `m4-post-arena-k180` (same chunk, 2400 points later). The room ends with a kill
  lock: `k305` is a black-screen transition into chunk 245 at a score ~290 points (one thug) above k180, and the
  "upper walkway" fixture is a new map (scx 84, scy 0) that reuses chunk id 86. The live run died after 11 kills
  (2 heavies + 9 thugs), so it was within one or two kills of the lock. Dying there restarts mission 3 (the v85
  lists show the full forest/river/bruiser sequence after every arena death), so the arena must be cleared in one
  life from the ~5 HP a bridge crossing leaves. HP weights 40 and 80 reproduced the weight-12 plans exactly: the
  one-tick losses inside four kill plans are forced at beam 24. Measuring beam 24 vs 40 live with a per-step HP log
  (`DD_HP_LOG`) to separate in-plan losses from between-plan losses. Radar: `CHAIN_TILES` added as ladders when
  `chain=True` (mission 4 only - they are wall decor in missions 1-3 and gave false ladders in chunks 15/46/66/106);
  the runner passes `chain=mission4` and the nav planner carries `nav.chain`, so the upper-walkway pillar chains
  are climbable for the planner later.
* **v113 launched (15:27 UTC), v112 stopped at 50 min (1949 px mean, 37/40 in mission 2, 0 in mission 3, 0 errors):**
  v112's best chunk sat at 72 (the last mission-2 ladder chunk) from minute 30 to 50. Saved states showed the
  stall: chunk 16, player on a ladder at the top with a thug on the rungs beside him (`v112-w039_..._no-progress`).
  Nobody owned that state: the player cannot attack while climbing, the nav planner is suppressed by an enemy within
  60 px, and the fight planner only fires in listed chunks. v113: (1) `stalled_fight` - the 30-decision immobility
  counter with an enemy within 60 px calls the fight planner in any chunk, climbing allowed; offline from that
  state it climbs down and kills the thug pair in 46 steps with 0 HP lost. (2) The nav stall timer reset counted
  `vertical_px`, which accumulates in both directions, so ladder bouncing never looked stalled - horizontal scroll
  only now. (3) `mission4` from the canonical index. (4) `chain=mission4` radar ladders. (5) `DD_HP_LOG` debug.
  The second v112 stall type (chunk 25, left wall next to a ladder) is already handled by the nav planner (34 exits).
* **Arena HP budget, measured (15:35 UTC):** the per-step HP log of the beam-24 live run puts the five lost ticks at
  steps 66, 142, 215, 271 and 582: three landed with NO plan active, each at the step a kill plan had just ended,
  and the next search then started from a state where the hit was already unavoidable (that is the "hp left 4" the
  planner reports). The win test declared a kill as soon as the target was dead and the next enemy was beyond
  28 px, with no check that the player survives the following steps. Fix (`DD_PLAN_SAFE_AFTER`, default 6): a
  candidate kill is accepted only if backing away from the nearest enemy for 6 steps costs no HP; otherwise the
  state stays in the beam and the search continues. Beam 40 vs 24 (same start): tall + first thug killed with 0 HP
  lost and the bruiser met at 5 HP (beam 24: at 1 HP after 5 more thugs), at ~4x the search time (250 s vs 65 s
  per heavy fight).
* **Arena, continued (15:50 UTC):** dying in the arena with lives left respawns at the mission-3 start (chunk 12,
  tested from `m4-post-arena-k180` with 5 lives), so the arena cannot be skipped by a death; the `m4-upper-*`
  fixtures did not come from the arena. The beam-40 live run killed 16 enemies (score 41790, above the `k305`
  fixture's 41270) without the lock opening, so the lock is not a simple kill count near 2700 points either, and
  from the post-kill harness state walking right is blocked while left returns to bridge chunk 44. A WRAM scan for
  a remaining-enemies counter (bytes that change exactly at kill steps) is running on the planner gauntlet.
  Solo full-game run `solo-1002-1540` (1 worker, 30000-step cap, beam 40, video on) launched at the user's request
  alongside v113; its MP4 and the deepest v113 episode's MP4 go to the user when they land.
* **v114 launched (16:15 UTC), v113 stopped at 50 min (1808 px mean, 30/40 in mission 2, best chunk 72 again):**
  v113's stalled-fight trigger never fired: it was built on `no_progress_run`, which resets whenever an enemy is
  reachable - and the ladder-top thug is 3 px away. The ladder state is a true freeze (same pixel, climbing flag
  set, score flat for 240 steps with the model acting). v114: the trigger reads position history directly (no move
  for 12 decisions, no score gain for 60 steps, enemy within 60 px) and the fight planner is instantiated
  unconditionally (the chunk list only gates the chunk trigger; the local smoke runs had no planner at all, which
  hid the problem twice). Safety lookahead refined: back away only from a live enemy within 48 px, otherwise jab in
  place - the blind retreat walked off the chunk-16 ledge and made the planner reject its own 46-step kill. Bruiser
  regression unchanged (88 steps, 0 HP lost). The solo video run keeps the earlier planner; if it freezes at the
  ladders it will be restarted with v114's code.
* **Videos for the user (16:35 UTC):** `~/Desktop/dd-videos/` holds the deepest run's MP4 (v85 worker 15, 7078 px,
  8856 steps; one frame per step at 60 fps = ~8x speed, 2:28 long). Milestones from its trace: mission 2 at step
  1592 (0:26), mission 3 at 2965 (0:49), river 3800 (1:03), bruiser 4084 (1:08), mission-4 intro 5586 (1:33),
  arena 5655 (1:34); the rest is arena deaths and mission-3 replays. Per-episode MP4s live under
  `s3://sessions/ddenv_videos/decision/<run_tag>/` and upload when an episode ends.
* **v114 at 41 min (16:57 UTC): the ladder gate is open.** 4/40 workers in mission 3, best chunk 175 (mid-forest),
  26 in mission 2, 7 episodes finished. The stalled-fight trigger fired in chunk 16 (kills in 34 and 61 steps, 0 HP
  lost) and also picks up stalled street/warehouse fights (chunks 15, 7, 76, 84). The solo video run (beam 40)
  lost a life in the ladder section, respawned at the mission-2 start and is fighting back through it; beam 40
  costs up to 316 s per thug-pair fight.
* **Solo video run finished (18:00 UTC):** `solo2-1002-1620` (1 worker, beam 40, video) reached chunk 183 (mission-3
  forest), 4655 px, 7177 steps, 20 fight plans, 14 nav plans; lives lost at steps 2082 (mission-2 lower floor),
  3786/4304 (ladder section) and 7176 (forest). MP4 in `~/Desktop/dd-videos/`. v114 (40 workers): 6 reached the
  arena with 1-4 HP, none cleared it; bruiser 13/14; the ladder gate is open (18/40 into mission 3).
  **Mission-1 boss room, resolved differently:** from a saved second-boss state the runner never fought him - the
  player was thrown around (HP 6 -> 2) and at step ~149 the chunk sequence ran through the mission-1 ending
  (106, 89, 106, 36, 106, 73 ... 66) with all three lives. His "HP bytes" are an animation counter (2, 4, 0, 5, 9,
  10 while idle), jabs score +30 without damage, and a knockdown makes him walk off the right edge. So 21/38 v114
  deaths were players fighting an unkillable scripted enemy. Escape-mode searches (survive until the chunk changes,
  depth 420, with and without a rightward incentive) did NOT find the exit from that state, so the trigger is not a
  timer alone; being investigated. Chunk 48 removed from `ALT_KNOWN_CHUNKS`; a failed-fight nav override added.
* **Mission-1 boss room solved (18:20 UTC, v115):** the exit condition is one 200-point knockdown on the second
  boss: he and his companion then walk off the right edge and ~25 steps later the mission-1 ending plays. Escape-mode
  search found it once two things were fixed: (1) escape searches now use the 420-step escape depth regardless of
  enemy HP, and landed hits (score gain x 0.05) count in the escape objective, since this enemy has no HP byte;
  (2) the HUD blanks during the ending cut-scene, so the health reading drops to 0 and every node was pruned as dead
  before the chunk change could be seen - nodes now carry the last valid HP reading and, in escape mode with no enemy
  on screen, a 0 reading is not a death. Result from `boss2-entry.state`: 86 steps, 28 s, HP 6 -> 6, mission 2
  reached with all three lives. Runner: `boss1_dead` is set when a kill plan zeroes the known slots in chunk 48,
  and `survive_48` then calls the planner in escape mode with `escape_px=9999` (chunk change only).
* **v115 launched (18:35 UTC), v114 stopped (36/40 done; 4 stuck at the ladder top):** v114's deepest episode
  (worker 33, 6275 px, 9274 steps) reached the arena at step 6606 with two lives and died three times there; its MP4
  is in `~/Desktop/dd-videos/` (boss room 0:17, mission 2 0:29, mission 3 1:26, bruiser 1:45, arena 1:50). v115 =
  v114 + the boss-room survive mode (`escape_depth` 240 there; the runner test cleared mission 1 with 3 lives and
  full HP, though five of six survive searches missed at 180 s because the second boss was out of reach from those
  roots) + the failed-fight nav override + chunk 48 removed from the ALT shortcut.
* **v115 crashed workers, v116 launched (18:45 UTC):** `KeyError: 'hp_eff'` - after a kill rejected by the safety
  lookahead the planner re-reads the observation, and that fresh observation lacked the field added for the blank-HUD
  fix. The bruiser regression never took that branch; a test that forces the first three wins to be rejected
  exercises it 218 times and now plans normally. v116 = v115 + that fix.
* **v116 at 65 min (19:48 UTC):** 35/40 alive, 20 in mission 3, 7 in mission 4, lives mean 2.60 (v114: 2.38 at the
  same point) - the boss-room survive mode is saving lives (21 escape wins at 32 s; the 23 misses at 219 s are the
  second boss out of reach). First three arena arrivals: 5, 3 and 4 HP; opening kills cost 1, 0 and 3 ticks. Arrival
  HP is unchanged from v114, so the next lever is the mission-4 bridge (HP 7 at the mission start, 3-5 at the arena)
  and the in-arena cost per kill (a 268-step kill that loses 3 ticks is still accepted at HP weight 12).
* **Config bug found (20:05 UTC):** `submit-decision-run.sh` used `${DD_PLAN_ESCAPE:-86}`, which substitutes the
  default for an EMPTY value too, so every cluster run since v109 fought the arena in escape mode (win = 150 px of
  rightward progress or a chunk change) while the local tests ran kill mode. That is why cluster arena "kills" were
  300-375 steps long and cost 2-3 ticks each (they were walks into the thug stream) where local kills took 60 steps
  at 0 HP. Default is now off (`${DD_PLAN_ESCAPE-}`); the mission-4 planner runs kill mode with the rightward
  progress weight. The boss-room survive mode is unaffected (it sets escape explicitly).
* **v117 launched (20:15 UTC), v116 stopped at 95 min:** v116 reached the arena with 10/40 workers at 2-5 HP, lives
  mean 2.3-2.6 (boss-room fix working), none cleared it - all its arena plans were escape-mode walks (config bug
  above). v117 = escape mode genuinely off (kill mode + rightward weight in mission 4), the fight planner on every
  forest thug (`DD_PLAN_ANY_ENEMY_ALWAYS=11,12`: local test from the mission-3 start, 900 steps, 7 kills, 0 HP lost,
  ~36 s per fight; the model alone lost 7-16 ticks there, which is where arena arrival HP went), and `DD_PLAN_MAX=80`
  plans per episode (forest + arena exceed the old cap of 20).
* **v117 at 60 min (21:01 UTC):** forest planner on the cluster: 107 thug kills with 0 HP lost, 2 with one tick,
  13 misses (182 s each), 4 partials; 21/40 in mission 3, lives mean 2.14, 4 done. Wall-clock is slower (chunk best
  183 at 60 min vs v116's 263 at 51 min) at ~51 s per forest fight; the payoff shows at the arena arrival HP.
* **v118 launched (21:30 UTC), v117 stopped at 85 min:** 10 of v117's first 15 finished episodes lost every life in
  the early forest at the same spot - full HP, no enemy on screen, x 108, feet ~90, then the sprite gone: falls
  into the stepping-stone gap right after planned kills. The fight planner pruned only a dropped life counter,
  which lags a fall by ~15 steps, and the 6-step safety window did not look for a vanished player, so kill plans
  that ended walking into the gap were accepted (and reflexes are bypassed while a plan replays). v116 had no forest
  planner and passed 25 workers through. v118: a vanished player sprite or feet below the floor band is a death in
  kill-mode search and fails the safety window; the nav planner prunes the same. Regressions unchanged (ladder 46,
  boss-room survive 86, bruiser 88 steps, nav 4). Everything else as v117 (escape off, forest planner, cap 80).
* **v118 at 60 min (22:20 UTC):** 37/40 alive, lives mean 2.62 (v117: 2.14; v116: 2.60), 21 in mission 3, 2 in
  mission 4, first arena arrival; forest planner 114 kills / 123 plans and no forest-gap deaths (fall guard working).
  User decision (22:10 UTC): after the teacher clears the game, distill into PPO via DAgger with the planners as
  the labeller, then PPO fine-tune on the RAM rewards (memory: dagger-after-teacher-clear).
* **HP trail resolved (22:35 UTC):** mission starts refill HP to 7 (the m3-start HUD reads 0 only because of the
  cut-scene blank). v118: forest fights 167 clean / 4 one-tick, bruiser clean, but the mission-4 bridge kills in
  chunk 106 cost one tick each, so arena arrivals carry 4 HP; arena kills at beam 24 cost 0-2 ticks (one worker:
  4 -> 2 -> 2 -> 2 -> 1 -> 1 over five kills). Prepared for v119: `DD_PLAN_BEAM_M4` (default 40) - the runner passes
  a wider beam for every mission-4 fight (bridge + arena), where the local tests turned 1-tick kills into 0-tick
  ones at 2-4x the search time. Holding the launch until v118's first real kill-mode arena attempts finish.
* **v119 launched (22:45 UTC), v118 stopped at 85 min:** v118's arena attempts (first genuine kill-mode fights on
  the cluster): arrivals at 3-4 HP, kills at 0-2 ticks each, one worker made ~8 kills before dying at 1 HP; the
  lock did not open. v119 = v118 + `DD_PLAN_BEAM_M4=40` (bridge and arena fights searched at beam 40).
* **v120 launched (23:20 UTC), v119 stopped at 42 min (no arena arrivals yet):** the arena is solved offline. Last
  night's notes (v96/v99) record the room as exactly four fights - two 21-HP thugs, the 32, the 42 - after which the
  screen scrolls on to the cave (one v96 survivor reached chunk 244; an escape plan reached 246). Today's WRAM scans
  confirm no byte counts kills: the extra thugs in the idle harness were fillers. Escape mode with the chunk change
  as the ONLY win (`escape_px=9999`) and the rightward weight clears the entry fixture at **5 -> 5 HP** (351 steps
  at beam 24 in 332 s; 259 steps at beam 40 in 410 s). Kill mode (v118) fought fillers forever; the 150-px win
  (v109-v116 by the config bug) walked into them. v120: chunk 86 planned in escape mode on arrival, `DD_PLAN_ESCAPE`
  default back to 86 (both runner and submit script), beam 40 for mission 4, everything else as v119.
* **The arena is passed live (23:30 UTC).** Local runner from the entry fixture with v120's code: the escape plan
  cleared all four waves in 332 steps (5 -> 4 HP) and the player walked into the cave (chunks 246, 244, 245). v120
  crashed on start (the new flag was used in the trigger before its definition - the runner smoke was skipped;
  fixed, and the trigger condition now includes the flag), v121 launched, then replaced by **v122 (23:45 UTC)**
  with a beyond-arena tracker mode: the cave reuses river/bridge ids (246, 244, 245, 234, 233, 84, 85, 86), which
  snapped the tracker back to mission 3 and aimed the nav planner at the river's successors; past the arena index
  only a respawn id (12/11/15/66/31) regresses it, mission stays 4, nav goals fall back to novelty. Local cave run:
  four fast novelty exits along the walkway, the chain climbed (feet 109 -> 75), then a fall through the upper
  walkway's hole ~60 px on right after a nav plan ended (the v87 `cave_hop_veto` route). Testing the hole from the
  `m4-upper-landed` fixture. Cluster token expired again (re-login).
* **v123 launched (00:25 UTC), v122 stopped at 50 min:** the cave past the arena, worked through on the
  `m4-upper-landed` fixture with the live runner. Findings: (1) the chain tiles must not be ladders - the model
  climbed the chain above the walkway into the void (chain radar off); (2) the "upper walkway" is pillars about
  24 px wide separated by an 8 px void plus a 16 px non-solid chain column, and the radar counted only the void as
  a pit, so the hole read 25 px further left than the true edge - past the arena the chain columns are pits
  (`cave` flag); (3) the running kick lands only from a 2-launch-point window (world x 206-209 from the raw fixture,
  208-212 after a reset - frame alignment shifts it), so a fixed reflex window cannot be trusted; `app/hop_planner.py`
  tries the kick after 0-7 steps right or 1-2 left on the live emulator and replays the first launch that lands,
  plus 3 settling steps (the landing spot is the far pillar's left edge and the model walked straight back off it);
  (4) a left-edge guard turns "left" into "right" when a pit is within 10 px behind on the walkway. Result: hop at
  world x 212, the whole upper walkway crossed, a new screen reached in chunk 106 (the "third screen fight lock"
  from the v87 notes) where the planner killed a 20-HP enemy and lost to a 42-HP heavy at 3 HP. Also found: the
  env's action order is down, left, right, up, a, b, kick_left, kick_right, jump_kick (the harnesses mapped by
  name, so their results stand). v123 carries everything (arena escape, cave pits, verified hop, edge guard).
* **First arena clears on the cluster (01:25 UTC, v123):** escape plans from 4 HP (224 steps, 0 lost, 693 s), 6 HP
  (294 steps, 0 lost, 893 s) and 4 HP (256 steps, 1 lost, 841 s), with "repeated 3-4x" tags on two of them. The
  first worker through walked the lower cave walkway (chunks 245/233) on a novelty nav partial (166 px right) to
  its dead-end void and died, respawning at mission 3 with 7 HP. Known from the v84 notes: that walkway's exit is
  UP the 294/295 chain at screen x 91-123; the nav planner's displacement fallback prefers 166 px of walking over
  68 px of climbing. Next: climb preference in beyond-arena mode, designed against v123's saved cave states.
* **v124 launched in parallel (01:35 UTC, 20 workers):** v123 at 80 min: 4/4 arena clears (3-6 HP arrivals, 0-1
  tick lost), 12-14 workers in mission 4, 37/40 alive; every cave arrival walked the lower walkway to its void on
  a nav partial. Fix: past the arena (novelty mode) the nav fallback's displacement weights upward scroll x3, so
  the chain climb (68 px) outranks the 166 px walk; from the `m4-post-arena-k270/k295` fixtures the plan now ends
  on the upper walkway (scy 0, feet 75). v123 keeps running for its arena/cave sample and deepest video.
* **Third screen = the hideout door (02:45 UTC):** v123 at 2h20: 25 arena clears out of 28 attempts; one worker
  made two verified pillar hops on the cluster and reached the third screen (chunk 106 beyond the arena). Local
  fixtures captured there (`DD_SAVE_LOCAL_DIR`, new): a room with a double door in the back wall at screen x 64-95
  (tiles 327/328 top, 359/358 panels), a 42-HP bruiser at the known HP slots, thug pairs at 10-24 HP, no pits
  besides one at the right edge. Holding up under the door at x 72-73 walks the player to the top of the floor
  (feet 44) and nothing happens; an earlier "vanish" was the thug's grab. So the door most likely opens when the
  bruiser dies, and arrivals have 2-3 HP. The nav planner now treats a vanished player as a possible transition
  (idles up to 30 steps; a life lost is a fall) instead of pruning it. v124 (cave climb preference) is 10/20 in
  mission 3; its third-screen arrivals will show the HP budget there.
* **The hideout door opens (03:05 UTC):** the third-screen bruiser (42 HP) dies to the beam-40 planner even from
  2 HP (79 steps, 0 lost). With him dead, standing anywhere under the door (x 64-93) and holding UP for ~20 steps
  (12 was not enough) takes the player into **chunk 6** - the hideout interior, never recorded before - at 5 HP.
  The door is locked while he lives. Fixtures: `ignored/hideout-post-bruiser.state`, `ignored/hideout-inside.state`.
* **v125 launched (03:15 UTC, 20 workers, 12000-step cap) alongside v124:** the nav planner can now find the
  hideout door by itself (24 steps, 1 s from the post-bruiser state): a macro that leaves the search key unchanged
  is idled up to 40 steps in novelty mode to catch a pending scene transition, and a vanished player is idled up to
  30 steps to tell a fall (life lost) from a transition. Inside the hideout (chunk 6, HP refilled to 5): a cave
  with a dark doorway in the left wall (void tiles - the radar calls it a pit at x 24-48), a boulder, and a heavy
  leaping in from the right. Probing the fight and the exit offline.
* **v126 launched (03:40 UTC, 20 workers), v125 stopped at 15 min:** hideout room 6's 24-HP heavy dies to the
  planner (40 steps, 0 lost) and the chunk advances to room 5 (two 10-HP thugs at the known slots). Those chunk ids
  were not in the any-enemy list, so nothing planned the room-5 fight; past the arena every fight is now planned.
  v124's first cave arrival produced the climb plan live (306 px nav partial in chunk 233 = the x3 vertical weight).
* **v127 launched (04:05 UTC, 20 workers), v126 stopped at 25 min:** v124's three cave climbers all died on the
  upper walkway right after the climb with no hop line; the first hop attempt that did fire found no landing. A
  climb can end 30+ px short of the hole, beyond the hop planner's 7-step approach (~25 px); the approach now goes
  to 12 steps (~42 px). Local check: hop still verified from the upper-walkway fixture, player alive on the far
  pillar. (A local run from a cave fixture cannot test this path end to end: the tracker only sets the beyond-arena
  flag after a real arena pass, so cave pits stay off there.)
* **04:30 UTC tallies:** v123 (no climb/door logic) 40/46 arena clears; three workers made BOTH pillar hops and
  reached the hideout door room (one then fired the hop planner 35 times at that room's right-edge pit - harmless).
  v124 10/11 arena clears. v127 (full chain: climb, 12-step hop approach, door transition, hideout fights planned) is
  15 min in. Chunks 5/6 also occur in mission 2, so "hideout fight" counts need the beyond flag, not the chunk id.
* **v127 regression and an old suicide (05:15 UTC):** v127's first 8 finished episodes lost all lives in mission 2
  (chunks 25/26/66 and 16/17) or the early forest within an hour; v123 had no deaths at that point. Two nav-planner
  defects: (1) the v125 vanished-player tolerance (idle 30 steps, accept if no life lost) let a lower-floor fall -
  which costs the life only after > 30 steps - pass as a "transition" in goal mode, so the fatal plan was replayed
  every life; it is now novelty-mode-only (past the arena) with a 60-step window. (2) Older and worse: the mission-2
  lower-floor "exit in 4 steps" from chunks 25/26 was the player walking off the ledge - the chunk id flips during
  the fall and the goal test fired before the life was lost. That has been killing lower-floor players since v108.
  A reached goal now has to survive a 40-step idle (alive, visible, on a floor) before it counts.
* **v123 final (05:30 UTC):** 40 true starts: 40 reached mission 2, 33 mission 3, 30 mission 4; distance mean
  5204 px (max 6350), canonical position mean 280/333; 40 arena clears in 46 attempts, 6 verified pillar hops. The
  deepest episode (worker 39, 10000 steps, 1 life left) is in `~/Desktop/dd-videos/v123-best-*.mp4`: arena at 1:47,
  arena cleared and cave entered at 1:54 - the first recorded footage past the arena.
* **v124 final (05:00 UTC):** 20 starts: 20 reached mission 2, 15 mission 3, 13 mission 4; 12 arena arrivals;
  distance max 6488 px (best so far), 1 verified hop, no hideout entry. v128 (nav survival checks) at 1 h: lives
  tracking v123 through mission 2, but two workers spent all 12000 steps in the mission-1 warehouse doorway (chunk 84)
  pressing left 9800 times: the player sprite is FROZEN there - 1500 varied inputs and 150 idle steps change nothing
  (a game soft-lock, not a policy problem; the fight plan that preceded it replays cleanly, so the trigger is later).
  The v128 goal-mode "vanished = fall" prune also made nav report "no exit (states 1)" 38 times in that chunk.
* **Hideout mapped offline (05:30 UTC, with the player HP byte frozen - 0xC498, HUD ticks = value // 8):** room 5's
  two 10-HP thugs must die, then the corridor scrolls right freely: ~250 px of empty cave (chunk ids flip every
  8 px of scroll: 6/5 alternating, then 96, 80, 81, 64, 65, 68, 24, 26, 25, 11, 12, 9, 58, 84, 89, 90, 91 - the byte is
  a column pointer here, still monotone), a staircase down with a rolling boulder (SCY 0 -> 112), then stepping
  stones: the slabs are SPRITES (tiles 253/254/255, 3 tiles = 24 px wide, top at y 88) that swing ~24 px sideways with
  a ~24-step period, so the background tilemap under them is uniform rock and the tile pit detector saw nothing. The
  same launch point lands or falls depending on WHEN you jump (a fixed-window hop found no landing, a 4-step back-and-
  forth then kick crossed). Past the slabs: a 20-HP thug pair and a 52-HP bruiser in front of a second back-wall door
  at screen x ~110-130; losing a life there respawned at the hideout entrance (chunk 6, SCY 0), not at mission 3.
* **v129 (code, 05:45 UTC):** (1) radar: slab sprites are platforms, never enemies (they read as three reachable 0-HP
  thugs); when the player stands on a slab row the pits are the gaps between slabs and the void past the last one,
  recomputed every step (`platforms`, `on_slab`). (2) hop planner `timed=True` on slabs: wait 0-22 steps, then 0-4
  steps right, then the kick; advance measured from the launch point (min 16 px). Runner triggers it on slabs
  (feet ~90, gap up to 64 px). (3) nav planner: the v128 vanished-player prune is gone (falls are rejected by the
  40-step survival check instead, which now also waits up to 40 more steps for a door fade to show the player).
  (4) soft-lock recovery: a checkpoint every 200 healthy steps; a player frozen at one pixel for 300 steps with no
  enemy on screen is confirmed on the emulator (40 varied presses on a snapshot) and the pre-freeze checkpoint is
  reloaded, max 3 per episode, logged as `softlock-reload` and counted in the episode record.
* **v129 stopped after 3 min, v130 launched (05:25 UTC, 20 workers, 12000 steps, start state `dd.gb.state`):** two
  defects caught between the two: (1) the first slab-pit cut replaced the tile pits whenever any slab sprite was on
  the feet row, so a slab entering from the screen edge with the player on tile ground became a 150 px "pit" on the
  left, which the left-edge guard pushed the player away from, toward the real hole. Now slab gaps replace the tile
  pits only while the player stands over a slab; otherwise the tile pits stay and only slab gaps ahead are added.
  (2) the submit script stages every `app/ignored/*.state`, and without `--init-states` the runner spreads workers
  over its six default start states; v129 started 8 of 20 workers in mission 2+. Offline, with the fix, timed hops
  cross the whole stepping-stone section from all three fixtures (pillar edge: 6 hops; slab fixtures: 4 each), and
  the pillar-section pits are correct across the SCX wrap. The hop planner now always falls back to the wait-phase
  launch variants when the fixed-window pass finds no landing.
* **v131 launched (06:00 UTC, 20 workers; v130 kept running):** the doorway "freeze" is the player stuck in the
  warehouse's right wall after a throw: walking, climbing and idling do nothing, but attacks still animate and shift
  the sprite box 6-11 px - the v129/v130 freeze test counted that as movement, so the checkpoint reload never fired
  (one v130 worker idled there through 9 nav plans). The test now holds each direction for 10 steps and ignores
  attack frames; both captured doorway states read "frozen", the healthy fixtures "moves". Checkpoints are also
  taken on entering a chunk (the warehouse has thugs on screen most of the time). Nav planner: a `throw` (B) macro.
  Fight planner: after a kill the player must be back at the plan's starting floor height after a full 48-step
  settle; on visible slabs the settle walks to the slab centre and appends that walk to the plan. Found at the
  hideout's second door: the 52-HP bruiser dies to the planner (121-124 steps, 0 HP lost) but every accepted plan
  left the player landing on the edge of a moving slab that the sprite table drops while the bruiser is on screen,
  and he fell 20-30 steps later. Bruiser regression unchanged (88 steps, HP 5 kept).
* **Second hideout door opened offline (06:15 UTC):** the door-2 room (chunk 80 past the slabs) is a solid ledge
  whose real left edge is ~8 px right of where the tile pit detector puts it; a planned kill that ends at x 101-108
  drops the player into the void the moment the bruiser's dying sprite stops overlapping him (standing at x >= 106
  is safe). The post-kill settle now steps away from any pit edge within 20 px. Holding UP at x ~142 after the
  bruiser dies fades to chunk 106 and then a new vertical area: chunk 73, SCY 56, HP refilled to 7, a small ledge
  with spikes below and a wall on the left. Exploring it.
* **v132 launched (06:30 UTC, 20 workers):** v131's freeze recovery never fired either (one worker frozen in the
  doorway through 11 nav plans). Cause: the frozen sprite's feet toggle 85/86 every step and the position key
  included feet//2, so the "unchanged for 300 steps" clock reset constantly. The key is now (chunk, SCX, x). The
  v131 local check could not reproduce the freeze (the model walked on), so this is verified on the fixtures only.
  Offline past door 2 (god mode): a shaft climbed on the dotted column (chunk 73, SCY 56 -> 1), a novelty nav exit
  to chunk 56, spiked ledges hopped with timed kicks (56 -> 27 -> 26 -> 57 -> 30), an alcove in chunk 30 that is
  NOT a door (UP does nothing) - the way on is a jump kick right into chunk 106's lower floor (feet 113), then
  another climbing column. The planner finds no kill for the 20-HP ledge thugs (fall pruning on narrow ledges).
* **v132 stopped, the live bottleneck located (07:15 UTC):** v132's freeze recovery fired live (worker 13, chunk
  84, checkpoint reloaded) but three of its first five finished episodes were beaten to death in the warehouse
  (chunk 48, canon index 3) - zero such wipeouts in the 35 earlier episodes of v128/v130/v131. Cause: the 48-step
  post-kill settle rejected every warehouse kill (respawning thugs hit during the idle), so the model fought alone.
  The long settle is now mission-4 only. v128's finished episodes show where live runs actually stall: eight of
  fourteen reached the arena and then spent 3700-4100 steps in the first cave chunks (246, 233) until the step cap.
  Reproduced locally with a new `DD_FORCE_BEYOND` knob (treat the arena as passed) from the post-arena fixture: the
  novelty nav found 4-step "exits" every time the reused cave id flipped (burning the 20-plan budget), then a
  66 px partial walked the player into the void. Fixes: past the arena the cave's reused ids count as recent, a
  partial walk must survive a 40-step idle, and on the upper walkway (feet <= 80, no enemy within 60 px) the model's
  movement is overridden with "right" so the hop hook owns the gaps. With the first two fixes the planners alone
  cross the cave offline: climb (325 px partial), two verified hops, door room reached, under a minute.
* **v133 launched (07:35 UTC, 20 workers):** cave fixes (reused ids are not exits, partial walks must survive 40
  idle steps, upper-walkway autopilot, hop hook re-armed as soon as a landed hop's replay ends - the 24-step block
  walked every local run into the next gap at the same pixel), long post-kill settle mission-4 only, freeze key
  (chunk, SCX, x). Local check from the post-arena fixture with the live model: climb (331 px partial), two verified
  hops, door room reached alive at step 211 - the first time the model pipeline has done that end to end. Arrived
  with 3 HP; the door-room thug pair (20 HP) only yields partial plans at that HP (4 min each). Next: the door room
  at low HP (fixture `cave-doorroom-hp3.state`).
* **v133 stopped at 6 min, v134 launched (07:45 UTC):** the door-room fight's live "partial plan" (vs a 42-step kill
  offline from the same state) was the mission-4 rightward progress weight: past the arena the fights are in rooms
  and the weight pulled the search toward the right-edge pit. The weight now applies only before the arena.
* **Door room at low HP (08:00 UTC):** from the local run's door-room state (3 HP) the planner kills the 20-HP
  pair (42 steps, 3 HP kept) once the progress weight is off. After that two sprites remain at dx -16/-65 with
  every known HP byte at 0; slot discovery ended in 0 s because the player's first hit taken ended the probe
  (HP 2 -> 0). Discovery now holds the HP byte at 3 ticks during its snapshot probe; it then finds only a 6 -> 3
  timer byte: jabs on these sprites score but no byte in C540-C700 drops, so either they are invulnerable
  (dying) and the damage comes from elsewhere, or their HP lives outside the scan window. Idling there at 2 HP
  loses the life within ~110 steps. Open; fixture `doorroom-after-hpw12-beam40.state`.
* **v128 final (08:35 UTC):** 20 episodes: 18 reached mission 2, 15 mission 3, 13 mission 4; 18 arena plans, 2
  verified hops, no hideout entry; distance mean 4585 px (max 6079). Eight workers reached the arena and then spent
  3700-4100 steps in the first cave chunks (the stall v133/v134 target). Two workers froze in the warehouse doorway.
  **v130 final:** 20 episodes: 19 / 16 / 10; distance mean 4786 px (max 7121, best so far); its arena arrivals came
  in at 2 HP and died there without a fight plan (2 arena plans in the whole run).
* **v134 live milestone (09:10 UTC):** with the cave fixes, worker 242759 escaped the arena at 3 HP, climbed the
  chain (331 px nav partial), made both verified hops, reached the door room, killed the 20-HP pair (57 steps),
  the remaining 10-HP thug (28 steps) and the 42-HP bruiser (86 steps, 598 s search) - all with 3 HP kept. Five
  arena escapes in the run so far (HP 2-5 kept). Two other door-room arrivals at 2 HP only got 20-minute partial
  plans. Mission-4 plan wall-clock is now the throughput limit (bridge fights 6-11 min each): next version limits
  the long settle to past the arena and caps a plan at 600 s.
* **v135 launched (09:20 UTC, 20 workers; v134 kept running):** nav exits found only during the post-macro idle
  (door fades) now hold the last input for 24 more steps - v134's door-room worker replayed 12 UP presses, the
  model's next action cancelled the entry, and the same 12-step exit was reported three times. Door fixture: exit in
  48 steps straight into the hideout (HP refilled to 6), robust to 24 noisy actions afterwards. Also: the long
  post-kill settle only past the arena, plan time cap 600 s (was 1500), 16 saved states per episode (was 8).
* **First live hideout entry (09:30 UTC, v134 worker 242759):** after the bruiser kill the nav planner found the
  door (two more plans), and the worker entered the hideout with HP refilled to 7, killed room 6's 24-HP heavy
  (58 steps, 6 HP kept), then got "no kill found" twice in room 5 (10-HP thug, 6 HP). Cause: the 48-step post-kill
  settle rejected every kill because the second thug hit the player during the idle. The settle now accepts early
  once the player stands on a floor and a live enemy is within 48 px (the landing check is what it is for). Room-5
  fixture: pair killed in 41 steps, 4 HP kept, 33 s. v135 stopped at 35 min, **v136 launched** with that fix.
* **v136 stopped at 10 min, v137 launched (09:35 UTC):** the v134 hideout worker read 0 HP through the whole
  corridor (chunks 96/80) and the planner refused every fight there ("no kill found in 0s"); the HUD tick count is
  blank in parts of the hideout and in cut-scenes. Player health now comes from RAM 0xC498 // 8 (checked against
  the HUD on 100 fixtures: identical wherever the HUD is drawn; the HUD lags a frame after damage and shows the old
  value during the respawn). Bruiser regression unchanged. v134 so far: 7 arena escapes, 6 verified hops, 4 room-6
  and 3 room-5 plans; the hideout worker died in the corridor and respawned at the hideout entrance with full HP.
* **Corridor boulder (09:55 UTC):** two more v134 workers entered the hideout (three in total); worker 239850 left
  room 5 with 6 HP and reached the corridor with 1. The staircase boulder's sprite uses low tile ids, so the radar
  never lists it and the fight planner has nothing to kill; walking right loses HP after 10-20 steps on the
  staircase fixtures. Escape mode "won" instantly there because the corridor's chunk id flips every few px, so the
  planner now takes `escape_chunk_win=False` (progress-only win): a 45-48-step plan threads the boulder with 5 HP
  kept in ~20 s. Runner: a corridor autopilot plans that escape in the corridor ids (96/80/81/64/65/68/24/26/25/
  11/12/9/58) whenever no enemy is near. Local test from the corridor fixture running.
* **Hideout fixes staged for v138 (10:15 UTC):** (1) the corridor reuses the mission-3 start ids 11/12, so entering
  it looked like a death respawn to the tracker and switched every past-the-arena behaviour off; a respawn id now
  counts only within 240 steps of a lost life. (2) Inside the hideout (rooms 6/5 seen at SCY 0) the room/corridor ids
  are treated as visited, so only a door is a nav exit. (3) The hop hook and the hold-right autopilot arm at ledge
  height (feet <= 95) inside the hideout - the ledges after the stairs stand at 89, not 76. (4) Corridor escape
  plans must be damage-free (`max_hp_loss=0`; the accepted plans had been trading 2 HP to the boulder each).
  Local run from room 5 with the live model: corridor plans, then five verified hops in a row across the ledges and
  the moving slabs (34/18/13/25/25 steps).
* **v138 launched (10:20 UTC, 20 workers; v137 kept running as control):** the hideout fixes above plus the corridor autopilot limited to SCY < 112 (at the slab level ids 80/81 recur and belong to the hop hook).
* **v137 stopped (2 h 10 min), v139 launched (11:40 UTC):** v137 confirmed the held-input door exit (two workers
  entered the hideout on their first nav exit) and showed two more gaps: (1) a player with 1-7 HP points has zero
  HUD ticks but is alive - the planner read 0 and refused every fight through the hideout corridor; health now reads
  at least one tick while the byte is nonzero. (2) a 2-HP arena arrival hit the new 600 s plan cap and got a
  partial; the arena escape now has its own 1200 s budget. v138 (hideout fixes) keeps running as the comparison.
* **v134 final (11:45 UTC):** 20 episodes: 12 reached mission 4; distance mean 4396 px (max 6322); 14 verified hops,
  12 arena plans, four hideout entries. Videos of the two deepest episodes are in `~/Desktop/dd-videos/` (worker 0:
  arena at 11:51, hideout door at 13:19, then four corridor deaths at chunks 80/24/80/64 - the boulder staircase;
  worker 1: arena 13:43, hideout 15:25). Worker 0's copy is also at `~/Documents/double-dragon-hideout-run-v134-w000.mp4`.
* **v139 stopped at 25 min, v140 launched (11:55 UTC):** v138's first three hideout entries cleared rooms 6 and 5
  first try (the settle fix), then two of them went 6 -> 1 HP and 5 -> dead in the alternating 6/5 stretch right
  after room 5, before the corridor ids the autopilot covered. The boulder rolls there too. The corridor autopilot
  now includes 6/5 once the hideout has been entered (rooms with enemies near are still the fight planner's).
* **v131 final (12:00 UTC):** 20 episodes: 19 / 13 / 11 reached missions 2/3/4; distance mean 5103 px (max 7690,
  new best); 37 arena plans, 5 verified hops, two hideout entries (door-hold fix). Ran 5.5 h because its last
  workers spent most steps inside 25-minute planner searches (the 600 s cap came later).
* **Door 2 live and its lock timing (12:15 UTC):** two v138 workers crossed the whole hideout (rooms, corridor plan
  at 1 HP, five to seven slab hops) and killed the door-2 bruiser (52 HP) at 1 HP kept; the nav planner then found
  no exit there (at 1 HP every macro that takes a hit is pruned). Offline the door unlocks only ~20-60 steps after
  the bruiser dies and opens on ~30-40 held UP presses at x ~142; pressing UP earlier does nothing, waiting 90 steps
  lets the next thug arrive. On that ledge DOWN drops the player off and LEFT walks into the void, so the only safe
  idle is punching the air. Runner: a scripted door-2 autopilot (walk to x 142, punch 30, UP 40, punch 40) when the
  room's HP slots are all zero. Local test with the live model running.
* **v141 launched (12:40 UTC, 20 workers; v138 and v140 kept):** door-2 autopilot (walk to x 142, punch 30, UP 40,
  punch 40, armed when no HP slot reads >= 10), a door-2 room guard (DOWN/LEFT/kicks become a punch there - DOWN drops
  off the ledge, LEFT is the void), the gap-edge wait is a punch instead of DOWN, the corridor autopilot covers the
  6/5 stretch and stays out of the maze, maze ids are visited once past door 2, and a shaft autopilot (chunk 73/72:
  walk right to the ledge end, then hold UP). Locally the live-model pipeline went through door 2 and up the shaft
  to the ledge row. v138 live: two workers killed the door-2 bruiser at 1 HP and one is in the shaft (chunk 73).
  Offline the maze past the shaft still dead-ends: the upper ledge's chimney is not enterable by walking (the player
  falls at x 137 even holding UP), its floor collapses at x ~67, and its right end drops to the lower platform.
* **Maze note (12:50 UTC):** from the upper ledge past door 2 a macro search with wait/hold macros found the way
  past the collapsing alcove floor: jump-kicking LEFT crosses it and the ledge continues leftward, stepping down
  (feet 58 -> 74 -> 89 -> 113 over ~150 px, ids 88/89/63). The search saturated there (393 states) without a new
  area; the chimney above the upper ledge cannot be entered by walking. Mapping continues offline when live runs
  reach that point; for now the live frontier is the shaft and ledge row right after door 2.
* **v142 launched (14:05 UTC, 20 workers; v140/v141 kept):** v141 reached door 2 with 3-5 HP (first cleanly), killed
  the bruiser at 5 HP kept, but the door-2 autopilot never armed: the knocked-out pair's sprites stay on the radar as
  reachable enemies; the trigger now uses only the HP slots and starts from x >= 100. Also: the corridor autopilot
  falls back to a one-tick plan when no damage-free line exists, and the freeze test runs only in the doorway chunk.
* **Maze status (14:15 UTC):** live, v141 worker 257423 went through door 2 with the scripted entry, climbed the
  shaft (nav partial) and died on the ledge row at full HP (the planner returns "no kill found" in 1 s against the
  20-HP ledge pair there; cause not yet reproduced - no saved state, the 16-state budget was spent earlier in the
  episode). The chunk-30 alcove is not a door with the ledge thugs alive (UP at x 96-108 does nothing; x 112 is the
  drop to the lower platform). Next approach for the maze: stitch the background tilemaps of the explored states
  into one map and plan over tiles (ladders, chimneys, ledges, spikes) instead of macro search.
* **Maze map (14:25 UTC):** `app/ignored/maze_map_v1.txt` is a first stitched tile map of the area past door 2
  (13 saved states placed by scroll-register continuity; `app/ignored/stitch_map.py`). Symbols: L/l ledge top and
  underside, S/s spikes, # rock, '.' and '?' dotted-column tiles, space = dark. The dotted columns drawn above the
  ledge row (world x 192-216) are NOT climbable from it (UP at every x from 189 to 222 does nothing), so the
  climbable shaft column must use other tiles; next step is to identify the climbable tile ids from the shaft
  state and re-read the map for a real route.
* **Maze tiles identified (14:40 UTC):** the climbable column is tiles 254/255 (top) and 267/268 (shaft) - the
  ones the first map legend called spikes; the dotted "chimney" tiles 316/317/319 with 284/285/300/301 interiors
  are decoration and never climbable. The shaft column (world x 176-208) tops out at the ledge row (y 88); the
  ledge there ends at x ~170 with the next column (x 128-160) across a gap that neither kick nor jump kick reaches
  (kick_left from x 184/178 falls, from 187/192 lands back at 220). Candidate routes left: the ledges at y 32
  (x 112-152, 176-240) and the column x 112-144 above them, reachable from somewhere not yet found.
* **Ledge-row death reproduced and fixed (15:00 UTC, v143 code):** a local run from the shaft respawn state (`--local`,
  `DD_FORCE_MAZE=1`) showed the "no kill found in 0-1 s" at chunk 27: the hop planner's three blind "settle" steps
  right after a verified landing walked straight off the 16-px ledge B of the ledge row, so the fight planner started
  from a falling player. The settle now steps on only while the next edge is > 12 px away (else punches) and adds two
  idle checks. With that fix the player stands, but the planner still cannot beat the 20-HP pair on a 16-px ledge
  (both settle modes fail in ~70 s: every contact is a knockback fall) - fights are now skipped on ledges with both
  pit edges within 24 px and the autopilot hops on instead (the pair's slots then read 6/6 and the thugs follow later).
* **Maze route found offline (15:00 UTC):** the "dead-end" upper ledge was the way. Verified on save states
  (fixtures `app/ignored/gdl*`, `ls-onC`, `dc-landed-*`, `pc-climbed-r0`, `tt-top-trace`): ledge row A->B->C by hops;
  from C's right end only a running kick reaches the lower platform (chunk 106, feet 114, 24 px lower - walking off
  is death); UP at x ~108 climbs its column; right 6, UP again -> top ledge (SCY 200, feet 58), which is a RESPAWN
  POINT (a death there respawns at (73,58), not in the shaft). Leftward: the collapsing floor is crossed by a kick-left
  drop onto the feet-74 ledge; kick-left drop again onto the 64-px feet-90 ledge (10-HP pair; punch twice on landing,
  the planner kills them in 31 steps with HP kept); left 2 + kick-left onto the leftmost ledge; hold UP -> 80 px up
  (column tiles 254/255/267/268 at x 16-47) -> SCY-120 ledge; right, hop the 24-px gap -> island; right 4, UP -> 48 px
  up to the upper floor (SCY 72). Its left end is a fight gauntlet (14+14, 28, 28+14; HP byte 0xC574 also used);
  after the kills the left wall opens, a column at the far left climbs to SCY 0: a NEW area (ids 106/7, different
  tileset) that continues RIGHT with kick-over gaps, a thug pair and a 52-HP bruiser (exploration continuing).
  Memory note `dd-maze-route` holds the step list.
* **v143 runner (not yet launched):** hop planner takes `direction` ("left" mirrors the launch options and the kick)
  and `drop` (accepts a landing 8-28 px lower once the height holds for two steps; SCX wrap at 256 fixed - the drop
  from ledge C wrapped 230 -> 8 and read as negative advance); maze autopilot replaces the shaft-only rule past door 2:
  travel direction by scroll height (`maze_dir`: SCY 200/72 left, the ledge above the lower platform right, else right),
  hold UP whenever a climbable column tile is over the player's head or the climbing pose is on, hop (level, then drop)
  at a pit edge ahead or when a 12-step look-ahead on a snapshot would drop the player, punch instead of stepping when
  no hop lands, no fight plans on narrow ledges, and the old hop hook / walkway rule are off inside the maze. Also:
  soft-lock checkpoints are a 6-deep ring and reload #N goes N checkpoints back (v142 worker 4 reloaded the same
  re-freezing checkpoint three times); the reload message reports the real frozen-step count; a save state is written
  at every replayed hop (`hop*/mhop*`). Local end-to-end test from the shaft respawn state in progress.
* **v143 launched (16:07 UTC, 20 workers; v141/v142 still running):** maze autopilot live. Eighteen local runs from the
  shaft respawn state drove the fixes, in order: hop settle walked off ledge B -> settle probes each step on the emulator
  and the flight presses the direction only while airborne (the kick's trajectory does not depend on the held direction;
  pressing after touchdown walked off); lower landings need two stable readings (the fall passes through every height);
  SCX wrap; "UP beside the shaft column does nothing" -> walk under a column before climbing, columns found by scanning
  +-56 px with the column FOOT at floor level (a column above the next ledge up is unreachable - one death); a near
  column outranks the pit-edge hop and the look-ahead probe stops at it; a "pit" touching the screen border is unscrolled
  map (the camera lags the player, and locks during knockout animations) -> the probe decides there and the autopilot
  stands still while a knockout timer counts down (static 1-9 leftovers are not timers); back-off from the border only
  over solid ground; no hop plans while airborne (thrown); knocked-out sprites (slot < 10) no longer yield to the model;
  the nav planner never plans next to a live thug in the maze and a nav replay aborts on a hit/throw (a 128-step "exit"
  replayed blindly while a thug threw the player off the top ledge); the top-ledge thug has NO HP byte (one landed hit
  sends him off for good) -> `maze_knockdown_plan` beam search (win = score gain and nobody within 48 px) when the fight
  planner's slot discovery returns nothing; maze ids 88/89/63/28/29/7 added to the nav planner's known set. Local run 18:
  shaft -> ledge row -> drop -> platform fights -> top ledge (knockdown) -> three left hops -> pair kill -> leftmost hop
  -> 80-px climb -> SCY-120 ledge -> island hop -> climb -> upper floor fights (14+14 kills, HP 6->5), all on one life.
* **Run finals (16:30 UTC):** v138 20 eps, m2/m3/m4 = 20/16/12, distance max 7822; v140 19/14/7, max 7380; v141 18/11/10,
  max 8155 px (new record - the door-2 / shaft arrivals). v142 still running (2 workers), v143 at 18/20 after 20 min with
  the first soft-lock reloads reporting real counts; two v143 workers re-froze at the warehouse doorway after every
  reload (checkpoints 1-3 back) - v144 saves the frozen state (`softlock*-a84`) for diagnosis.
* **Videos (16:35 UTC):** `~/Desktop/dd-videos/v141-door2-2JCU.mp4` (147 s at 60 fps = 1 frame per env step; the first
  live scripted door-2 entry, the shaft climb and the ledge-row deaths), `v141-door2-C22K.mp4` and `v141-door2-I9X0.mp4`
  (the other two door-2 workers). Deepest live footage so far; the maze-autopilot runs (v143+) should extend it.
* **New-area dead end (16:50 UTC):** local run 19 (from the upper floor's far left) climbed the 270/271 column into the
  new area, walked its lower floor right and reached the framed room at chunk 7 / scx 176: the planner killed its 52-HP
  and 35-HP heavies at 5 HP (first time - the explorer's HP cheat had confused the planner there), after which 20-HP
  thug pairs respawn forever (600 steps with HP frozen: no room change, score keeps rising). The room is sealed: floor
  x 16-149, an upper platform x 34-126 between two 292/293 pillars that are walls (UP beside them is a throw, not a
  climb), and the back-wall emblem is not a door (UP at x 56-104 does nothing). Maze exit from the new area is still
  unknown; candidates: the new area's upper floor (the explorer's path at feet 91, kicks over gaps) and whatever lies
  between scx 7 and 176 on either floor. `app/ignored/maze-bruiser-room.state` / `br3-boss-room-end.state` are the
  fixtures.
* **Sealed room solved (17:15 UTC):** run 19's own nav planner found it once the room was empty: at chunk 7 / SCX 176 the
  exit is a door on the upper platform at x ~105 - from the floor hold UP (climbs to feet 75 in ~20 steps) and keep
  holding ~34 more -> chunk 97, HP refilled (1 -> 5). Any thug on screen throws the player off the climb, so the
  scripted `room7-door` (walk to x 105, UP x48) runs only with an empty radar; thugs spawn only while the player stands
  mid-room, so clearing a wave and going straight to x 105 works (verified from three saved states: empty room = door at
  UP 34; rooms with a thug = thrown every time). Chunk 97/96 is a new hall that scrolls right (12/24-HP thugs); its
  head-height strip of tile 254 is decoration - columns now need a 3-tile vertical stack. **v144 launched (17:17 UTC)**
  with: 270/271 climb tiles, nav ids 88/89/63/28/29/7/97/96, the soft-lock state save, the room-7 door, the column stack
  rule. Local run 20 explores chunk 97 onward.
* **Hall past the sealed room (17:45 UTC):** chunks 97/96 scroll right (12/24-HP pairs; kills at 5-7 HP) and have their
  own respawn point (a death respawns at the hall start, chunk 97, HP 7). Three levels: an upper walkway at feet 67, the
  floor at feet 90, a lower walkway at feet 121. 16-px shafts (tile 362 through every row) cut the floor band and are
  hopped (three verified 16-step hops live); the floor also has 32-px holes marked by tile 354 at the feet row over
  362/312 - the pit detector did not see them (two run-20 deaths) -> `radar.EXTRA_VOID/EXTRA_BELOW` ({354}, {312,313}),
  set by the runner once the maze is entered. The running kick does not clear 32 px from the readable edge (the true edge
  is ~10 px inside the tile), so those holes must be bypassed on another level - unmapped. Fixtures: `maze-hall96*.state`.
* **v145 launched (18:25 UTC):** v143's door-2 script fired on ARRIVAL in the door-2 room (all HP slots read < 10 before the
  pair/bruiser activate), walked to x 142 and held UP at a locked door, then blocked itself for 400 steps through the
  real fight: four workers, zero shaft arrivals. The script now also requires the room's bruiser (a slot >= 40) to have
  been seen. Also in v145: hall holes in the pit reading (`EXTRA_VOID`), the lane shift after a failed hop, the punch
  fallback for enemies the planners cannot see (runs 20-22 died to the same throw at scroll 96; run 23 killed that thug),
  static sub-10 slots counted as live thugs. Local run 23 is the hall test.
* **Hall thugs' HP found (18:45 UTC):** the scroll-96 room's thugs that threw the autopilot to its death in runs 20-24 keep
  their HP (10) at 0xC583/0xC5C8 - a third slot layout, 6 bytes above the arena's alternate slots (RAM diff on a landed
  hit: 10 -> 9 at both). `ALT2_HP_ADDRS` added to the fight planner; it now plans against them (partials - they fly off
  screen after a hit and come back at full HP). Live: the first v143 worker is deep in the maze - door 2, ledge row,
  drop-kick, platform kills, top-ledge knockdown, two leftward drop hops, the feet-90 pair killed at HP 7, the leftmost
  hop - all on the cluster. v144 at 75 min: 14 alive; v145 (door-2 gate) launched 18:21 UTC.
* **v146 launched (19:25 UTC):** adds the hall thugs' HP slots (`ALT2_HP_ADDRS` 0xC583/0xC5C8) and a per-area save of
  "enemy the planner cannot see" situations (`kd-a*`). Local run 25 (from the scroll-96 room state): with the slots
  visible the planner killed the room's pair (61 steps, HP kept), a 32-HP wave, more pairs and a 10-HP thug past the
  room, then lost the life to a 42-HP bruiser at 1 HP and finished at the 1500-step cap on its second life. The room is
  a fight lock: the camera releases after the waves and a staircase structure sits just past its right edge (tiles
  224-226/240-242 at x 160-231 of the locked screen) - the hall's probable exit, not yet walked. Live: v143's deep
  worker climbed the leftmost column; v144 workers are at door 2; v145 at 1 h with 18 alive.
* **Deepest live run (19:35 UTC):** v143 worker pid 289662 went through the ENTIRE mapped maze on the cluster - door 2,
  ledge row, drop-kick, platform kills, top-ledge knockdown, three leftward hops, the feet-90 pair (HP 7 kept), the
  leftmost hop and column, the SCY-120 ledge and island, the upper-floor gauntlet, the far-left column - and is in the
  new area (chunks 106/7 at SCY 0) at 6 HP, killing its 20-HP pair and knocking thugs down toward the sealed room. v143
  has no room-7 door script (v144+), so it will probably stall there; v143's distance best is 7448 px with 3 workers left.
* **Hall = chain of fight-lock rooms (20:05 UTC):** with HP frozen offline, after the scroll-96 room's waves the camera
  released, the hall ran on (scroll wrapped past 255) into a second lock at scroll 192 (pairs + a 10-HP thug, the 14
  "door frames" there are wall decoration - UP at every column does nothing), then released again into a third lock
  at the next scroll 96 holding TWO 42-HP bruisers plus pairs (slots 0xC595/0xC5DA 42/42). Deaths respawn at the hall
  start with HP 7. The autopilot/planners handle each room (ALT2 slots make the pairs visible); HP attrition is the
  limiter, and the double-bruiser room is the current frontier. Fixtures: `maze-hall-postroom.state` (past room 1),
  `doors-at-doors.state` (lock 2), `hall3-end.state` (lock 3 mid-fight, HP cheat on).
* **Damage-free planning in the maze (20:15 UTC, for v147):** past door 2 the fight planner first searches 2 min for a
  plan with `max_hp_loss=0`, then falls back to the normal search. On the hall fixtures: the entrance pair dies in 18
  steps with HP kept (25 s); the scroll-96 room's pair yields a damage-free 60-step partial (72 s). v147 launches when
  v143/v144 finish (cluster at 47 envs now; the decision server's latency rises past ~60).
* **Deepest live run complete (20:25 UTC):** v143 worker pid 289662 (session LQNI) ran to the 12000-step cap having gone
  door 2 -> the whole maze -> the new area -> the sealed room (its nav planner found the platform door by itself: "area=7
  exit in 40 steps") -> the hall past it (chunk 97 fight at HP 7). Video: `~/Desktop/dd-videos/v143-maze-deepest-LQNI.mp4`
  (200 s at 60 fps = the full 12000 steps). **v147 launched (20:22 UTC)** with the damage-free first planning attempt
  for maze fights; v143 is down to its last worker, v144 to 3, v145 6, v146 13.
* **Lock room 3 (20:50 UTC):** after its two 42-HP bruisers die, 10-20-HP pairs keep spawning (their HP flickers between
  the alternate slot layouts, so the fight planner rarely locks on; runs 26/27 went 7 -> 2 HP there). No door: UP at
  every 12 px in both lanes with HP frozen does nothing; the back wall has a raised 342-filled platform with 254 tiles at
  its ends and more decorative frames. Knockdown search now returns damage-free partials and gets a deeper budget
  (depth 36, beam 10, 45 s) for maze fights. The hall's end is still unmapped; HP attrition across its rooms is the
  limiter for a 3-life run.
* **21:30 UTC:** v143 final: 20 eps, m4 reached by 9, distance max 7528 (one worker crossed the whole maze). v144 on its last
  worker (two `room7-door` entries, two shaft arrivals), v145 on 2 (one deep in the maze after the gated door-2 script),
  v146 on 5 (all five in mission 4), v147 (damage-free planning) at 14/20 after 66 min. Local run 28 is walking the hall
  at 4 HP but burned 4 x 2 min of planning on a single 10-HP thug that flies off after every hit - maze fights against
  enemies of <= 12 HP (any slot layout) now get 45 s per pass so the knockdown search takes over sooner (v148).
* **Hall attrition profile (22:00 UTC, runs 25/27/28):** the hall is entered at 5 HP (the sealed-room door's refill);
  room 1 (pairs, a 32) costs 0-1, the stretch to room 2 and room 2 (pairs, 10s, a 42) another 1-2, so room 3 (two 42s,
  pairs, then a 64-HP enemy - the biggest seen) is reached at 2-4 HP and the 64 is unbeatable from there (only partials).
  Deaths respawn at the hall start with 7 HP and must redo rooms 1-2. Levers: arrive with more HP (does the sealed-room
  door SET 5 or ADD 4? untested), lose none in rooms 1-2 (damage-free passes in v147), and faster kills of fly-off thugs.
* **Correction (22:05 UTC):** the sealed-room door REFILLS HP to full (byte 56 = 7 ticks) whatever the entry HP (tested at
  1, 3 and 6 ticks). The hall fixture used by runs 25-28 (`maze-chunk97.state`) was captured after the entrance pair had
  already cost 2 HP, so those runs started at 5; a live run enters the hall at 7 and should reach room 3 with ~4-6.
* **Hall deaths are throws into holes (22:40 UTC):** run 29 (true 7-HP entry) lost all three lives with HP never below 6:
  each time a thug grabbed the player during the last steps of a replayed partial plan, and the throw's arc landed in a
  16-px shaft after the plan ended (the search horizon saw no HP loss, which lands later). Fight-planner partials and
  the knockdown search's wins/partials now require the player to end ON THE FLOOR (feet within 4 px of the start).
  Run 30 retests from the full-HP entry. Run 28 killed room 3's 64-HP enemy at 2 HP; pairs keep spawning there.
* **v144 final (22:55 UTC):** 20 eps, mission 4 reached by 13 (best so far), distance max 8362 px (new record; v141 was
  8155). Two scripted sealed-room door entries and two shaft arrivals despite its early-firing door-2 script.
* **Ceiling-shaft trap (23:20 UTC):** the hall's lethal spot is a ceiling shaft (tiles 343/359/243 over a 362 bottom at
  head height, two tile columns wide at x 64-79 of the second scroll-96 screen): standing on the back lane within ~24 px
  of it kills in 1-6 steps (x 78: 1 step; x 90: 6). Four local runs and probably the early live hall deaths died there,
  thrown or parked under it by a replayed plan. `maze_under_trap_shaft` now drops any plan and walks on when the
  player is within 24 px of such a shaft (fixture `maze-hall-trap-room.state`).
* **v148 launched (23:25 UTC):** ceiling-shaft trap rule, grounded partials (fight planner and knockdown search),
  knockdown partials and the deeper knockdown budget, 45-s planning caps for small maze enemies, damage-free first pass.
* **Run 30 (23:45 UTC):** entered the hall at 7 HP, never dropped below 7, and lost all three lives at the ceiling shaft
  (x 61-78, back lane, second scroll-96 screen) - the trap, not fights, ends hall runs. Run 31 carries the trap rule.
* **v149 launched (23:10 UTC):** the fight planner takes a runner-supplied `danger(pyboy, x, feet)` hook and prunes any
  search state under a ceiling shaft (the knockdown search too); after a plan is dropped under a shaft the planners stay
  off for 16 steps while the autopilot walks clear (run 32/33: plan -> one step out of the zone -> re-plan -> back under
  it, 40 s per loop). Walking THROUGH the shaft zone is safe (run 34 crossed it at full speed); only standing kills.
  Run 34's later deaths at x 105-140 of that screen were throws by the live pair (HP 5 -> 0), not a second trap.
* **Trap screen = lethal right side (23:40 UTC):** on the second scroll-96 screen, idle tests at x 105-144 (both lanes)
  died within 10-30 steps and run 34 died standing against the locked room's right wall at x 138 with HP intact, while
  x 40-60 survived indefinitely and walking through at speed is safe. Rule (v150): when the ceiling-shaft pattern is on
  screen and a live enemy slot exists, retreat left of (shaft - 28 px) and fight there; the fight planner's danger hook
  prunes the shaft zone and the whole right side; no fight plans start from the lethal side. Run 35 tests it.
* **v147 final (23:55 UTC):** 20 eps, mission 4 reached by 7, distance max 7156 px. Mission-4 rates across the day's builds:
  v138 12, v140 7, v141 10, v143 9, v144 13, v147 7 of 20 - the spread is run variance, not the maze changes (which only
  act past door 2). v145/v146/v148/v149 still running.
* **00:00 UTC Oct 4:** run 35 (safe-zone rule) retreated out of the trap room, fought the pair on safe floor at full HP,
  then lost all three lives to throws into 16-px shafts with HP never below 7. Partial plans in lethal-floor areas now
  must also end with no enemy within 24 px (grab range). Live: v145/v146 on their last worker each (a v146 worker is
  in the maze), v148 13/20, v149 19/20. v147 final 7/20 to mission 4.
* **Blocked retreat (00:40 UTC Oct 4):** run 36 died when the model stepped into a 16-px shaft the moment a plan ended
  (fix: the model's move toward a pit edge is vetoed while an enemy is within reach); run 37 died at step 332 INSIDE the
  trap zone during the leftward retreat - a thug on the left pushed the player back (x 73 -> 68 -> 71 -> 76 -> 78) so
  the player lingered under the shaft ~14 steps. Rule (v150): while under the shaft, if the retreat is not moving the
  player (|dx| < 2 between steps) reverse its direction - any exit beats standing still; the retreat direction persists
  per episode until blocked again. Run 38 (from the trap-room save) retreated left to the previous screen at full HP,
  the camera unlocked (scroll 96 -> 0), the 12-HP pair died on safe floor, and it walked back right with no lives lost.
* **FINAL BOSS FOUND (01:00 UTC Oct 4):** a cheat-mode probe (HP frozen) kick-spammed through hall lock 3 and, once
  the pairs died (~2700 steps), a 64-HP boss (main slots 0xC595/0xC5DA = 64) walked out of the big throne-room door
  on the same scroll-96 screen: a gunman (sprite 24 px wide, everything else is 16). Facts from `bl3-boss.state`:
  - bullets fly along his lane and hit for 16 HP (2 ticks); his gun-butt melee hits for 6 every ~16 steps when the
    player is pinned beside him; a jump kick does 4 (16 hits to kill). Trading jump kicks at the right wall reaches
    boss 24 HP with the player dead from 63 HP - a straight trade loses by ~2x.
  - the floor is a trapezoid: walking DOWN past feet ~100 or UP past the back edge is a pit (two probe policies died
    on step 11-12 that way). Lanes feet 72-94 are solid.
  - death in the boss room respawns at the HALL START (chunk 97, scroll 0, HP 7, lives -1) with the boss gone; so the
    boss must die in one life from whatever HP survives the hall (max 56). Nothing persists across lives.
  Running now: the real runner from `bl3-boss.state` (boss-auto1) to see what the fight planner finds.
* **Skull doors, not shafts (01:40 UTC Oct 4):** run 38 (blocked-retreat reversal) retreated out of the trap room at full
  HP, killed the pair on the previous screen, walked back and died at step 670 at x 64 with 56 HP on solid floor - the
  "ceiling shaft" rule never fired because that screen's barred shaft was over a different skull. The lethal thing is
  the SKULL DOOR itself: 4-column blocks (row-y48 tiles 332-335, or 243 when barred) at x 0-23, 56-87, 120-151 on these
  screens; standing in front of one, on either lane, kills within a few steps (the lower lane has a 32-px hole under
  each skull as well). The 24-px gaps between skulls are the only safe spots, and the thugs' throws land the player
  in front of a skull (every idle test from the trap-room save died that way). Rule (v150): `maze_skull_zones` from
  the tilemap; in a zone never stand - leave by the nearer edge toward a gap while thugs live, else keep walking
  through; on a skull screen with thugs alive wait in the nearest gap's centre and let the planners (danger hook =
  the zones) do the fighting; the model gets no step there. Run 39 tests it from the trap-room save.
* **Boss "kill" was a bar reset (01:40 UTC):** the 600-s offline planner's 117-step zero-loss "kill" only emptied
  0xC595 (64 -> 0); the gunman stayed on screen shooting (the player then died in 70 steps at the right wall) and a
  second bar sits at 0xC574 = 60 (the gauntlet slot). Testing whether jump kicks drain it; if so the fight is 124 HP
  = 31 jump kicks, and the planner needs 0xC574 in its target slots for this room. The real runner from
  `bl3-boss.state` (150-s plans, beam 24) drove the first bar 64 -> 6 with partials at 1 HP, then died.
* **THE GAME HAS AN ENDING AND WE REACHED IT OFFLINE (02:30 UTC Oct 4):** correction to the two entries above - 0xC574 is
  a timer, not a second bar, and the HP cheat was corrupting the probes (a frozen player + vanished boss = the player's
  own death with HP restored mid-animation). With real HP from the bar-empty save the sequence is:
  1. drain the gunman's 64 bar (0xC595/0xC5DA; jump kick = 4, so 16 kicks). At bar 0 he takes no more damage
     (RAM diff across a landed kick: nothing changes) but keeps shooting (16 HP) and gun-butting (6 HP);
  2. land ONE standing attack (punch `b` or kick `a`) at bar 0 - he flies off up-left and is gone ~40 steps later;
     jump kicks never trigger this (400 steps of them did nothing; the planner's bar-0 "kills" all ended that way);
  3. Marian appears in the throne doorway (radar: a 16-px "enemy" at the door, feet ~52). Walk to x 76-84 and hold UP
     ~30 steps: the player climbs into the doorway (feet 71 -> 54), the screen fades (scroll 96 -> 0, player sprite
     gone), then STAFF credits roll and the screen ends on THE END (`app/ignored/end4-ending-end.state`).
  Saves: `endg-boss-left.state` (boss gone, Marian waiting), `end4-ending-start.state` (the fade). The runner now needs
  (a) a damage-free bar drain - the 600-s beam-40 planner found a 117-step zero-loss one from `bl3-boss.state`,
  (b) a finishing standing hit, (c) the doorway walk. Building these as v150 boss scripts.
* **Run 39 (skull-zone rule, 02:45 UTC Oct 4):** 700 steps, no lives lost (first hall run ever with zero deaths), but no
  progress either: every entry into the skull room spawned its 10-HP pair, a throw landed the player by the left skull,
  the zone exit oscillated against the blocking thug and walked off the left screen edge, the camera scrolled back and
  the pair respawned - a loop. v150 fixes: (a) a thug blocking the zone exit gets a standing kick instead of a
  reversal (hall thugs fly off after one hit), reversals never head into the screen border; (b) on a skull screen with
  thugs alive the gap-centre wait has priority and kicks anything within 30 px; (c) the knockdown search prunes skull
  zones. Run 40 tests it from the trap-room save; boss-auto2 tests the throne-room scripts from `bl3-boss.state`
  (600-s beam-40 damage-free drain -> finishing-hit search -> doorway walk -> GAME COMPLETE marker + saved state).
* **The runner finishes the game from the throne room (03:30 UTC Oct 4):** boss-auto2 from `bl3-boss.state`: the 600-s
  beam-40 damage-free search found a 121-step bar drain (313 s), the finishing-hit search a 19-step plan, and the
  gunman left with HP 7/7 intact. boss-auto4 from `endg-boss-left.state`: "ending: the gunman is gone at step 29" ->
  "*** GAME COMPLETE *** step 65: the doorway fade started" (saved as `GAME-COMPLETE` state). Gates learned on the way:
  the gunman is the only 24-px sprite but two overlapping skull-room thugs read as one (run 40) -> boss scripts need
  the throne room's pillar tiles (244/245 at row y8, columns 5-6); the door tiles read closed again once he is out and
  the hall's ALT2 slots keep a stale 10/10 there -> the ending fires on "boss gone 20 steps or a sprite standing in the
  doorway (x 64-96, feet <= 60)" with main/ALT slots < 10. boss-auto5 re-tests the whole chain.
* **Skull room, runs 42-44:** the barred skull animates its middle columns through other tile ids, so the zone (and the
  old shaft rule) flickered on alternate steps -> zones now come from all five skull rows with gap-tolerant runs and the
  barred variant's dark outer columns. Run 44: no death but the player kicked two knocked-down (invulnerable) thugs in
  the middle zone until they got up and threw him off the left screen edge (camera scrolled back, pair respawned).
  v150: leave a zone away from the nearest thug, never kick while a knockdown timer runs, wait in the RIGHT gap.
* **v148 final (03:40 UTC Oct 4):** 20 eps, 9 reached mission 4, distance max 6483 px (trap rule + grounded partials;
  pre-skull-zone build). v145/v146/v149 still finishing.
* **The skull-room "reset" is a death (04:00 UTC Oct 4):** runs 38/39/44/45 all ended a skull-room visit with the player
  placed at (x 44, feet 81) on the scroll-0 screen with a fresh 12/12 pair and no `life-lost` line - offline the same
  move costs a life (3 -> 2): this room respawns at ITS OWN start, not the hall start, and HP is kept. Every one of
  those deaths followed a grab-and-throw into a skull zone (airborne/knocked-down players cannot walk out in time), so
  the fight has to be grab-free: v150 runs the emulator-backed knockdown search (zones pruned, must end grounded with
  nobody within 48 px) as soon as a thug closes to 56 px, drops a replaying plan that strays into a zone, and only
  falls back to the gap-centre wait + 30-px kicks when the search finds nothing. Run 46 tests it.
* **boss-auto5 = full throne-room chain in the runner (04:10 UTC Oct 4):** from `bl3-boss.state`: 121-step damage-free
  bar drain (314 s), 19-step finishing hit at step 123, "ending: the gunman is gone at step 161", "*** GAME COMPLETE ***
  step 204" with HP 7/7 and 3 lives. The throne room is solved; what remains between the live runs and the credits is
  the hall (skull rooms + lock rooms). **v150 launched** (20 envs x 12000 steps, full game from power-on) with the
  throne-room scripts, skull zones, grab-free knockdown search and the hall rules of this morning.
* **v145 final (04:40 UTC Oct 4):** 20 eps, 12 reached mission 4, distance max 8252 px (door-2 gate build).
* **Skull mechanics, measured (04:40 UTC):** on the trap-room screen's upper lane, x 92-100 in front of the BARRED skull
  (block x 56-87) die within 6-9 idle steps with no thug nearby, x >= 102 survive, and x 125-135 in front of a CLOSED
  skull survived 16 steps - only the open/barred skull kills. Its bars blink off for ~6 frames every ~45 steps (the
  zone detector keeps the last reading). The dying sprite is drawn at x ~20 (that is the "thrown to the left edge"
  I kept seeing in logs), and this room respawns at its own start. Even standing in the right gap with HP frozen, a
  thug threw the player to x 61 (in front of the barred skull) at step 67 -> death; the fight must stay grab-free.
  The knockdown search also went blind: it pruned every node after depth 8-16 because the closeness score put the
  whole beam in the same grab, so v150 dedupes nodes by (cell, facing, nearest-thug cell) and widens the beam to 16.
  Zones are now the barred block only ((x0+4, x1+12)); run 48 tests it.
* **The open skull MOVES (05:10 UTC Oct 4):** clean test from run 49's post-kill save (both thugs dead): the bars sat on
  skull A, then B, then C over ~45-step phases (the "blink" is the move). Standing in front of a CLOSED skull is safe
  (x 78-90 survived 30 steps with the open one at x 12); within ~40 px of the open one dies in 6-20 steps, faster when
  closer (x 92-100: 6-9 steps, x 110: 20 steps, open skull centred at 72). Rule (v150): the zone is a 96-px window
  centred on the open skull; the gap-centre wait relocates as it moves. Run 49 (fight plans trusted through zones,
  damaged thugs counted live): the 65-step damage-free kill replayed fully, both thugs died, HP 7/7 kept.
* **Run 52 - the skull room is being survived (05:50 UTC Oct 4):** moving-window zones + nearer-side exits + grab-free
  planning: stepped out of the open skull's window, killed the first pair (50-step damage-free plan), a second pair
  (51 steps), the 32-HP enemy (59 steps, fought at x 8-29 while the window sat on the right skull), then a third pair
  with knockdown plans; 430+ steps in the room, HP 7 -> 6, no life lost (every earlier run died within 60-80 steps).
  The fight planner also discovered the third wave's HP slot (0xC588/0xC5CD) on its own. Open question: how many waves
  the room has before the camera unlocks (earlier runs did get past it to lock rooms 2 and 3).
* **Skull room, runs 52-53 (06:30 UTC Oct 4):** the room IS camera-locked while its spawned pair lives and a new pair
  spawns when one dies; run 28 (the only earlier pass) fought 430 steps before the camera moved on, run 30 passed it
  because it came back with the previous screen's 12/12 pair still alive (the two-enemy limit stopped the room's own
  pair from spawning). Run 52 died cornered at the right wall on wave 4 (step 459); run 53 stood 40 steps 8-17 px LEFT
  of the open right-hand skull unharmed and then died 9 px right of the open middle one - the kill reaches ~24 px to
  the right of the open block and only a few px to its left. v150: asymmetric window (x0-8, x1+26), waiting spots
  clamped to x 28-132, a cornered knockdown search may trade one HP tick, and the player walks on whenever the window
  is not ahead and no thug is within 40 px. Run 54 tests it.
* **SKULL ROOM CLEARED (07:05 UTC Oct 4, run 55):** asymmetric window + nearer-side exits + kick a blocking thug +
  grab-free planning: pair (45-step partial + 20-step kill), second pair, the 32-HP enemy (60-step damage-free kill,
  111 s) - and at step 347 the camera scrolled on (scroll 97 -> 120 ...) with HP 7/7 and no life lost. The lock
  releases when the 32 dies. Run 55 continues toward lock rooms 2 and 3 and the throne room.
* **Past the skull room (07:30 UTC Oct 4):** run 55 then pressed right for 150 steps at scroll 153 - the back lane ends
  in a wall there (pillar tiles 304/305) while the front lane continues (offline: DOWN x8 + RIGHT x20 reached scroll
  219). v150: a wall detector - five motionless walking steps switch lanes. Run 56 restarts from that save
  (`hall-scx153-stuck.state`); run 57 restarts the whole hall from `maze-hall-entry7.state` (full HP) end to end.
* **Past lock room 2 (07:50 UTC Oct 4):** run 56 (from the scroll-153 save) shifted lanes at the wall, scrolled to 215,
  fought lock room 2 and reached a NEW screen at scroll 16 on the next lap: a dragon-banner corridor (tiles 244-247
  repeated along the back wall) with a 24-px heavy. The loose throne-room test (244/245 at columns 5-6) matched it and
  the finishing/ending scripts misfired (HP 36). v150 fix: the test also needs 331 at column 7, 330 at column 12 and
  the door tiles (232 closed / 362 open) at column 8 - still true on all three throne saves. Run 58 restarts from the
  corridor save (`hall-dragon-corridor.state`), run 57c the whole hall from the entry.
* **Run 57c (full hall from the entry, 08:15 UTC Oct 4):** reached the skull room at step 311 with HP 7/7 (all earlier
  fights damage-free), stepped out of the window to the right, then got grabbed at x 120 because the 16-step
  post-exit planner block covered exactly the pair's arrival; thrown into the window and killed at step 335. v150: no
  post-exit block on skull screens (the searches prune the window themselves). Run 57d restarts the full hall.
* **Lock room 3 reached again (08:40 UTC Oct 4, run 58):** from the dragon corridor save the run cleared the corridor
  and arrived in the throne room's lock phase (two 42-HP bruisers + pairs) at HP 3. The bruisers are 24 px wide like
  the gunman, so the boss scripts armed on them; v150 arms only on a 24-px sprite seen with BOTH main slots >= 56 (the
  gunman arrives at 64/64, the bruisers read 42). Run 59 restarts from that save (`hall-lock3-live.state`).
* **Run 57d (full hall, 09:10 UTC Oct 4):** entry -> lock room 1 -> skull room all damage-free (HP 7/7 for 600 steps,
  three skull-room waves killed), then a 26-step knockdown plan landed the player in front of the middle skull in the
  very step its bars opened; stuck in hit-stun he died at 611. The bars walk left -> right, so v150's zone now covers
  the open skull AND the next one; the just-closed skull is the waiting spot. Run 57e restarts the hall. Run 59 (lock
  room 3 from the save) killed the first 42-HP bruiser (71-step plan) at HP 3 -> 2 and faces the second at HP 2.
* **Run 57e / survival check (09:40 UTC Oct 4):** zoning the next skull too covered two thirds of the screen and the
  run died at 388 bouncing between exits. Reverted. New principle: the emulator is the judge - every hall fight or
  knockdown plan is replayed with 24 idle steps appended and discarded if a life is lost (the skull kill shows up in
  that window). Run 57f restarts the full hall with it.
* **Run 57f + v151 launched (10:25 UTC Oct 4):** with the emulator survival check the full-hall run went entry -> lock
  room 1 -> skull room (two pairs + the 32 killed; two plans discarded because they ended in front of the open skull)
  -> camera released -> wall at 153 -> lane shift, reaching lock room 2 at HP 2 (the 32 fight cost 4 ticks). **v151**
  (20 envs x 12000 steps, full game) carries everything learned today: throne-room scripts with the 64-bar arming
  gate, barred-skull window (asymmetric) with the moving-bars cache, grab-free knockdown search with positional
  diversity and the 24-step survival check + exit salvage, nearer-side window exits, kick a blocking thug, wait off the
  walls, walk on when the window is not ahead, wall detector, dragon-corridor-safe throne test.
* **Run 57f second life (11:00 UTC Oct 4):** from the hall-start respawn it cleared lock room 1 and the whole skull room
  again at HP 7/7 (pairs + the 32, three plans discarded by the survival check), then died at the wall: the camera sat
  at scroll 161 (not 153), the blind DOWN lane shift dropped into a front-lane hole. v150: the lane shift is now
  simulated first (DOWN x8 + walk, else UP x8 + walk; must survive and gain 12 px) - `maze_wall_escape`. Run 57g
  restarts the full hall with every late fix (survival check + exit salvage + simulated lane shift).
* **Run 57g (11:40 UTC Oct 4):** entry -> lock room 1 -> skull room (two pairs + the 32, two plans salvaged with a short
  walk left, HP 7/7 throughout) -> wall passed by the simulated lane shift -> lock room 2 at HP 6 - the deepest,
  healthiest hall state yet. Died at scroll 237 to a thug with no tracked HP byte: no planner engaged, the autopilot
  walked into him, a throw dropped the player into a right-edge hole. v150: the HP-agnostic knockdown search now runs
  for any close enemy anywhere in the hall when no slot reads live; kicks fire in reach even while walking to a gap.
  Run 57h restarts the full hall.
* **Run 57h (12:20 UTC Oct 4):** reached the skull room at HP 6, died at 452 - again a grab in the gap between plans
  (the knockdown search found nothing at 23-33 px, the standing kicks missed, a throw landed in the open skull's
  window). v150: the knockdown search gets an EVADE mode (14-step beam, win = out of grab range unhurt, no hit needed)
  used whenever no hitting plan exists and a thug is within 36 px. Run 57i restarts the full hall.
* **Skull room solved by RAM (13:10 UTC Oct 4):** run 57i died at full HP with no thug near while my tile detector read
  "no open skull" - so I scanned RAM around the transitions. The skull doors are a state machine: 0xC55E = 0 idle
  (0xC569 counts 48 -> 0 over ~17-40 steps), 1 opening, 2 open/aiming (~15 steps), 3 attack (a 24-px flying sprite,
  tiles 144-149, drops in front of the skull); 0xC484 = the active skull's x (16 / 80 / 144 on the trap-room screen),
  reset to 0 at the kill. Clean idle tests with the middle skull active: x 98-122 died 11-17 steps into state 2,
  x 129+ survived the attack; idle = safe everywhere. v150: `maze_skull_zones` = [(X-50, X+50)] while the state is
  1-3 (guarded by skull tiles on screen), nothing else. Run 57j restarts the full hall with it.
* **Run 57j (13:40 UTC Oct 4):** the RAM zone worked while a skull was active, but in the idle phase the player chased a
  thug to x 138 and the right skull (x 144) opened next; the exit was blocked by hit-stun and he died at 356. The
  screen is too narrow for any spot 50 px from all three skulls, so during idle the only spots every activation can
  be escaped from are the midpoints (x ~48, ~112). v150: idle-phase zones = 40 px around each skull centre (from the
  tiles), so the gap wait and the planners' end positions land on the midpoints. Run 57k restarts the hall.
* **Run 57k deadlock (14:20 UTC Oct 4):** the skull room's skulls are already on screen from scroll 23, and the new
  idle zones fed the "wait until the window moves" rule, which held the player still for 1700 steps (HP 7, no deaths,
  no progress). Idle zones now only choose the waiting midpoint; walking, exits and the planners use the RAM-active
  zone only. Run 57l restarts the hall.
* **Run 57l (14:50 UTC Oct 4):** survived the point that killed 57j (an idle-phase knockdown plan engaged the pair), killed
  it at 7/7, then - with the next pair still off-screen - walked to the right wall and the right skull opened 4 px
  away (death at 448, within 5 steps: the kill is faster the closer you stand). v150: while the room's thugs live the
  midpoint wait applies whatever their distance. Run 57m restarts the hall.
* **Cluster finals (15:30 UTC Oct 4):** v146 20 eps, 11 to mission 4, max 7371 px; v149 7 to mission 4, max 7674 px;
  v150 (first throne-room/skull build) 12 to mission 4, max 7102 px - none reached the hall's end. v151 (today's
  full build) still running.
* **Run 57m (15:50 UTC Oct 4):** died at 396 in the skull room: after a knockdown plan ended at x 51 the waiting rule
  preferred the right-hand midpoint (x 112) and walked across the middle skull's front; it opened with the player 15 px
  away and killed him in 5 steps (the kill is faster the closer you stand). v150: the waiting spot is the NEAREST
  midpoint. Run 57n restarts the hall.
* **Run 57n (16:40 UTC Oct 4):** entry -> lock room 1 -> skull room (two pairs + the 32, nearest-midpoint waiting, HP
  7 -> 6, no death) -> wall (simulated lane shift) -> lock room 2 at HP 6 - then worn down to HP 1 by a thug whose HP
  lives in a FOURTH slot pair (0xC588/0xC5CD): untracked, so only the (failing) knockdown search engaged. v150: that
  pair is now a known layout in the fight planner (ALT3) and in the hall's live checks. Run 57o restarts the hall.
* **Run 57o (17:30 UTC Oct 4):** cleared lock room 1 and the whole skull room at HP 7/7 (two pairs, the 32, a further
  pair - the fourth slot layout now tracked), then stalled 800 steps at the scroll-161 wall: a stale slot kept the
  midpoint-wait branch walking right, bypassing the wall detector. v150: global stuck detector - any walk that moves
  nothing for 6 steps gets the simulated lane shift. Run 57p restarts the hall.
* **Run 57p (18:20 UTC Oct 4):** HP 7/7 through lock room 1 and the skull room again; at the scroll-161 wall the new
  stuck detector fired but the 8-press lane shifts both failed from the mid-lane position (DOWN x8 drops into the
  hole, UP x8 does not move). Offline from that save: DOWN x5 + RIGHT advances to scroll 194. v150: the escape tries
  depths 8/5/11/3 both ways. Run 57q restarts the hall.
* **Run 57q (19:00 UTC Oct 4):** entry -> lock room 1 -> skull room (HP 7/7, three waves) -> wall passed by the
  multi-depth simulated lane shift (DOWN x5) -> lock room 2 at HP 6 with three lives - the deepest clean state yet.
  There the planner burned eleven 45-s searches in 20 steps on a 20-HP target that was not even on screen (stale slot
  reading) while a back-lane thug wore the player to HP 4. v150: in the hall no search without a reachable enemy on
  screen, a 24-step block after a failed one, and the planner now logs the slot it targets.
* **Run 57q end (19:50 UTC Oct 4):** reached scroll 237 (lock room 2's second wave) at HP 3 with three lives but was
  crawling one step per failed 45-s search (20 failures). v150: a failed hall pair search escalates to 120 s next time,
  the planner logs its target slot, no search without a reachable target on screen, and local runs save a hall state
  every 100 steps for offline work. Run 57r restarts the hall.
* **Lock room 2 is a trap, not a fight (20:40 UTC Oct 4):** the periodic hall save at scroll 219 shows the "untracked
  thug" is a MOVING LATTICE WALL BLOCK (3x3 sprite cells of the wall pattern, tiles 224-229/240-242) that drifts
  around the screen and hits for 8 HP; nothing changes in RAM when it is kicked. Walking right past it on the back
  lane cost no HP and scrolled straight on (219 -> 255 -> lap 3). Runs 57g/57q/57r lost lives here because the radar
  called it an enemy: fights, no camera movement, and a push into the right-edge hole. v150: the radar ignores those
  sprite tiles. Also noted: lap 3 around scroll 17 has a hole at x ~106 on both lanes (the blind walk fell).
  Run 57s restarts the hall.
* **Run 57s (21:20 UTC Oct 4):** HP 7/7 to the wall again, then lost a life to the wall block itself at scroll 219:
  with the block invisible to the radar the border back-off rule bounced the player left and right on the front lane
  inside the block's path (eight hits, HP 6 -> 0). v150: the radar flags the block (`wall_block`) and the runner
  simulates waiting 0-20 steps then walking 16, taking the first sequence with no HP loss (offline from the save:
  walk now -> scroll 255 with HP intact). Run 57t restarts the hall.
* **Run 57t (22:00 UTC Oct 4):** HP 7/7 through the skull room and the wall, then the timed walks carried it past the
  wall block (scroll 219 -> 255 -> lap 3) but the block roams onto the lap-3 corridor too and, with a 16-px hole at
  x 128-143 blocking the walk-on, the least-loss walks bled HP 7 -> 0 at scroll 1-2. v150: an HP-aware traverse
  search (walk/lane/hop/wait, any HP loss pruned, scroll progress rewarded) for block screens - offline from the
  lock-room-2 save it crossed the block screen AND the hole into lap 3 in 30 steps with HP intact; the timed walk is
  the fallback when the block is already on the player. Run 57u restarts the hall.
* **v151 final (22:30 UTC Oct 4):** 20 eps, 8 reached mission 4, distance max 7472 px - no worker reached the hall's deep
  rooms. The cluster builds keep landing 7-13 of 20 in mission 4 with ~7-8k px; everything learned since (skull RAM
  model, lock room 2's wall block, traverse search) is local-only until a hall run passes lock room 2 reliably.
* **Run 57u (23:10 UTC Oct 4):** skull room cleared at HP 6 (four waves), wall passed, and the traverse search crossed
  lock room 2's block screen in one 40-step damage-free plan (scroll 219 -> lap 3); a least-loss walk on the lap-3
  corridor then cost 3 ticks (block already on the player). Now at lap-3 scroll 27, HP 3, three lives - the first
  full-hall run to pass lock room 2 without losing a life.
* **Run 57u end (23:30 UTC Oct 4):** on the lap-3 corridor (scroll 35) a traverse plan ended on the lip of a hole with
  the player already dropping (feet 97, under my 100-px fall test); the wait fell in. v150: feet above 95 counts as
  falling in every simulated check, and traverse plans must end on a lane (feet <= 94). Run 57v restarts the hall.
* **Run 57v (00:20 UTC Oct 5):** the first full-hall run to pass EVERYTHING up to lap 3 without losing a life: lock
  room 1, skull room (four waves), the wall, lock room 2 and its roaming block (three traverse plans), the lap-3
  corridor's holes - all at HP 7 -> 6 - into a new balustrade room at lap-3 scroll 144-192 (railing with posts, pairs
  of ordinary thugs, no trap). Still alive there at step 930 with HP 6 and three lives.
* **THRONE ROOM REACHED LIVE WITH 6 HP AND 3 LIVES (00:50 UTC Oct 5, run 57v):** the balustrade room at lap-3 scroll
  144-192 was the approach to the throne room (lap-4 scroll 96). The first 42-HP bruiser died to damage-free plans
  (42 -> 29 -> 0, HP 6 kept); pairs continue. Every earlier arrival here had 1-3 HP; this is the first with a real
  chance at the gunman (whose scripts need 56 HP for a bullet-free drain, but a 6-tick arrival leaves margin for
  the 600-s search).
* **The gunman, live (01:10 UTC Oct 5, run 57v):** arrived at the throne room with 6 HP and three lives; the gunman
  came out at step 1372 (`boss-live-arrival-hp6.state` saved). Both boss searches (600 s damage-free, 300 s with
  2-tick allowance, beam 40) exhausted their candidates (~275 s each) from the left-side position - no drain line.
  Offline search from that save with beam 60 / 900 s / 1-tick allowance running in parallel.
* **Offline: the 6-HP arrival is winnable (01:30 UTC Oct 5):** from `boss-live-arrival-hp6.state` the planner with beam
  60 / 900 s / 1-tick allowance found a 114-step bar drain in 298 s with HP 48 kept (bar 64 -> 0). The live planner's
  beam 40 had exhausted its candidates. v150: boss passes = beam 60, 900 s with 1 tick, then 600 s with 2 ticks.
  Run 57w restarts the full hall with it; the throne-room finish + ending from the drained live state is being
  re-verified (`boss-live-r0.state`).
* **The live arrival state ends the game (01:50 UTC Oct 5):** the runner from `boss-live-arrival-hp6.state` (run 57v's
  exact throne-room arrival, 6 HP, 3 lives) with the beam-60 budget: 107-step damage-free drain (309 s), 10-step
  finishing hit at step 112, "ending: the gunman is gone at step 150", "*** GAME COMPLETE *** step 187" with HP 6
  and 3 lives. Run 57w is replaying the whole hall with this budget - if it arrives as 57v did, the credits roll.
* **Run 57w (02:20 UTC Oct 5):** replaying 57v's deterministic track - skull room, wall, lock room 2 (three traverse
  plans) and the lap-3 corridor all at HP 7 -> 6, now in the balustrade room at step 931 with three lives, on course
  for the throne room (~step 1370) with the beam-60 boss budget that completed the game offline from that exact state.
* **GAME COMPLETE IN A LIVE RUN (03:10 UTC Oct 5, run 57w):** from the full-HP hall entry (`maze-hall-entry7.state`),
  one continuous run: lock room 1 -> skull room (four waves) -> wall -> lock room 2 and its roaming block (three
  traverse plans) -> lap-3 corridor and holes -> balustrade room -> throne room lock phase (two 42-HP bruisers +
  pairs) -> the gunman (110-step damage-free drain found live in 273 s, 10-step finishing hit at step 1488, "the
  gunman is gone at step 1526") -> doorway -> "*** GAME COMPLETE *** step 1562" with HP 6 and all three lives.
  No life lost anywhere in the hall. State saved as `ignored/run57w-GAME-COMPLETE.state`. What remains for a
  power-on clear is the walk to the hall entry, which the cluster builds have reached before; **v152** launches now
  with this complete build, and a local power-on run starts as a second check.
* **Power-on run fullgame1 (11:40 UTC Oct 4, local, this complete build):** 10147 steps, 6563 px, three deaths, all of
  them mission-3/4 attrition: the mission-3 start and the hideout entrance share ONE 7-tick HP budget (no refill in
  between - a death anywhere in mission 4 before the hideout respawns at the mission-3 start, chunk 12). A new per-step
  `hp-drop` trace replayed from the saved cave state shows where the ticks went: the chunk-46 bruiser took 5 -> 1 in
  40 steps with `plan_left=0` - the scripted heavy autopilot (v58) was fighting him, and the planner only fired via the
  stall fallback after the damage. Same for every 20-HP pair in areas 11/12: the model fights until it stalls.
  **v153:** `DD_PLAN_M3_ALL=1` (default) plans every fight from first contact once the tracker is in mission 3, as
  v126 already did past the arena; `DD_PLAN_MAX` default 80 -> 300. v152 (cluster, 20 workers) runs without this:
  at 94k steps, 12/20 alive, best 6176 px, lives mean 1.7 - the same attrition.
* **v153 replays (12:20 UTC Oct 4):** from the two saved cave states (`fg1-cave244-hp4`, `fg1-cave245-life1`) the
  planner-first rule cleared the chunk-46 bruiser at 4 -> 4 and 5 -> 5 HP (69 and 94-step kills) where the heavy
  autopilot had paid 4 ticks; the remaining losses were the door room (32 HP, 0-1 tick) and the arena (1-2 ticks,
  319/367-step escapes). v153 also gives missions 3+4 the hall's damage-free first pass (arena 600 s, others 120 s)
  before the normal search. Launched: local power-on run `fullgame2` (14000 steps) and cluster **v153** (20 workers,
  14000 steps, `dd.gb.state`). v152 still running alongside.
* **fullgame2 (12:40 UTC Oct 4):** the mission-1 boss took HP 5 -> 1 with `plan_left=0` on every drop: in missions
  1-2 the planner's only trigger is the stall test, and a moving fight never stalls, so the 44-HP boss was fought by
  the model alone. Lives never refill, so a mission-1 death costs the clear as much as a mission-4 one. **v153b:** any
  heavy enemy (boss, bruisers) within 120 px is planner-first in every mission. fullgame2 stopped, `fullgame3`
  restarted from power-on with it. v153 (cluster) carries v153 without this.
* **fullgame3 (13:40-14:00 UTC Oct 4, local, v153b):** power-on -> mission-1 boss (planned, 89 steps, 0 loss) ->
  mission 2 -> mission 3 pairs and the chunk-46 bruiser (94 steps, 0 loss) -> door room -> arena (309-step
  damage-free escape, HP 6) -> mission-4 cave door room at step 6000 with HP 6 and all lives. There the two 10-HP
  thugs live at 0xC588/0xC5CD: the slot discovery found them at hp 7 and `plan()` rejected the result eight times
  (`ALT_MIN_HP` 10) while they took HP 6 -> 1 at x 143. **v153c:** a discovered (hit-verified) slot counts from
  `DISC_MIN_HP` 4; the hall's knockdown fallback (score-gain win, no HP byte needed) is enabled in mission 4 too.
  Replay `door1` from `fg-cave84-hp2.state` tests it.
* **door1 replay / v154 (14:20 UTC Oct 4):** from `fg-cave84-hp2.state` (past the arena, HP 2) the v153c build cleared
  the door room (20-HP thug, 42-HP bruiser in two plans) and entered the hideout **at HP 7 - the hideout entrance is a
  refill point**, so the shared budget is mission-3 start -> hideout door. Cluster **v154** (20 workers, 14000 steps,
  power-on) launched with the full v153c build; local `fullgame4` runs the same build from power-on.
* **v153d (14:35 UTC Oct 4):** the v153b heavy trigger also fired on the mission-1 second boss (unkillable, scripted):
  three 100-s escape searches found nothing and the room cost 2 ticks (fullgame4) where fullgame3 had left it at 0
  with the model surviving. The trigger now skips chunk 48 once the real boss is dead. Not restarting fullgame4/v154
  for it (mission 2 refills HP).
* **fullgame4 reaches the hideout with everything (15:45 UTC Oct 4, local, v153c):** power-on -> mission 1 (boss
  planned, 95 steps) -> mission 2 -> mission 3 (every pair, the chunk-46 bruiser and the door room at 7 -> 7) ->
  arena (one tick in a partial, then a 237-step escape) -> cave hops -> door room (20-HP thug, 42-HP bruiser, a
  23-step knockdown plan for the slotless thug) -> hideout room 6 at step ~6400 with HP 7 and all three lives. The
  first power-on run to enter the hideout without a death. Hall entry (refill) is ~2000 steps further.
* **fullgame4 hideout (16:10 UTC Oct 4):** room 5's thugs have NO HP byte (probe: the ALT pair reads 14, 9, 1, 16, 8, 0
  while idling and 90 jabs make both leave) - the static ALT test accepted the counter and drew 67-s searches twice,
  the knockdown search found nothing, and the model walked up into a throw (7 -> 5). The corridor escape accepted a
  1-tick line (no damage-free one from chunk 68), the staircase boulder took two more with no plan active (chunk 9
  at x 112 and chunk 106 at x 146 - invisible to the radar), door-2 room reached at HP 2: the 52-HP bruiser could
  not be killed from 2 HP (five searches, knockdowns chipped him 52 -> 10) and the run died there at step 7332 (its
  second death - the first was the mission-1 second boss at step 1965, before v153d). The door-2 room respawn keeps
  the room (HP 7, bruiser at 10). **v153e:** hideout rooms 6/5 accept only hit-verified HP slots
  (`require_verified`). Corridor/boulder: replay corr2 saved states every 60 steps (`fg4-corr-s*.state`); searching
  them offline for a damage-free corridor line (escape_px 48/80/120, depth 160-300, beam 24-40).
* **The corridor "boulder" is falling stalactites (16:30 UTC Oct 4):** screenshots and OAM dumps of the corr2 states
  show a 16x24 sprite (tiles 235/251/233 + 236/252/234) spawning at the ceiling 8-16 px ahead of the player, falling
  in ~12 steps and shattering (231/232/240-244). The radar read it as a thug at dy -22..-26, so the fight planner
  searched for it ("no kill found in 0s") and the escape search started from already-doomed roots. Offline, the
  corridor states have damage-free lines when the search is not launched mid-hit (chunk 96: 51 steps at the default
  settings; chunk 12 before the stairs: 48-px goal found in 9 s where the 120-px goal failed). **v153f:** the radar
  classifies stalactites in the hideout (`stalactites`, not enemies), a dodge reflex steps out from under one within
  22 px while it is above head height (plans still take precedence), and the corridor search tries a 48-px
  damage-free hop before accepting a 1-tick line. Replay corr3 tests it from the room-5 state.
* **corr5: the hideout at 7 -> 7 (16:30 UTC Oct 4):** from the room-5 state with v153e-g (verified slots in rooms
  6/5, stalactite class + dodge, 48-px corridor pass, nav plans HP-strict in the hideout with an emulator-verified
  prefix - corr4 had a 137-step nav "exit" march through the stalactites for 7 -> 2), the replay crossed room 5, the
  corridor, the stairs and the slabs with NO HP drop and planned the door-2 pair at 7 -> 7. Also: the corridor search
  no longer runs inside locked rooms 6/5 (three 68-s misses per visit). fullgame4 (old build, last life) is in the
  maze shaft at HP 5. Launched local `fullgame5` and cluster **v155** (20 workers, 14000 steps) with the full build.
* **fullgame4 deadlock in the sealed room (17:00 UTC Oct 4):** on its last life the run reached the sealed room (chunk 7,
  SCX 176) at step ~10000 with HP 5, climbed onto the door platform, and then stood there 2600 steps: two respawned
  20-HP thugs idled at the left screen edge (dx -115, slots 10/10), the door script wanted an empty radar and slots
  < 10, and the fight planner refuses targets beyond 96 px. Offline from `fg4-room7-stuck.state`: 80 held UP presses at x 76-102 open nothing, even with the
  thugs' HP bytes zeroed - the door stays shut while they exist. **v153h:** in the sealed room the runner walks at
  the nearest enemy while it is > 80 px away so the planner can engage; the strict door condition stays.
* **Sealed-room door geometry (17:40 UTC Oct 4):** the room has lanes, not a platform - feet 76 is the back lane and
  feet 113 the front. The door opens only from x 105 (room7.log: x 105, feet 113 -> 93 -> 75 -> chunk 97 on ~34 held
  UP). The v144 script's open-loop walk (|dx|//3 steps at ~2.25 px/step) left the player at x 112-125 where 48 UP
  presses do nothing; fullgame4 stood there 2600 steps and r7fix3 drifted to x 125 the same way. **v153i:** the
  door script aligns to x 105 +-2 in closed loop (one step per env step), then holds UP 48; room7_approach (v153h)
  walks at far thugs first since the door stays shut while they live. Test r7fix4 from the stuck state.
* **fullgame5 (18:00 UTC Oct 4):** 7 -> 7 through mission 3, the chunk-46 bruiser, the door room and the arena
  (254-step damage-free escape), now in the mission-4 cave - but one life lost at step 4115 in the forest with NO HP
  drop: a knockback into the pit at x 92 while the fight planner had spent three 72-s searches on 0xC583, a stable
  counter accepted as the hall's ALT2 layout. **v153j:** the static ALT2/ALT3 layouts are hall-only
  (`alt_layouts=maze_entered`); the nav-plan prefix check also stops at a fall (feet > 122 / sprite gone) and idles
  24 steps for the lagging life counter. The sealed-room alignment (v153i) fires correctly from x 105 (r7fix4) but
  that fixture's room still had a 52-HP wave to clear at HP 5. Launched local `fullgame6` with the complete build;
  `fullgame5` (without h/i/j) continues as a second sample.
* **fullgame5's slab death (18:30 UTC Oct 4):** third life lost at step 8209 (after the forest pit and a door-2 heavy
  hit at HP 3): a verified 18-step slab hop landed on a slab's edge, the hop search for the next gap found nothing,
  and the slab drifted out from under the standing player within 2 steps (reproduced exactly from
  `fg5-slab-hop16.state`: standing after the landing dies at t=2). **v153k:** in timed (slab) mode the hop planner
  appends a 14-step "ride" after the landing - follow the slab's centre from the radar's platform sprites - and
  rejects the launch if the player still falls. From the same state the hop becomes 41 steps, lands centred, and the
  next hop is found at once (56 steps). Replay slab2 tests the full crossing; fullgame5/6 stopped, `fullgame7`
  launched with the complete build (c-k).
* **Local vLLM port-forward died (~18:40 UTC Oct 4):** the oc port-forward to svc/dgemma (18011 -> 8011) was gone; the
  decision client timed out 20 s per request and fell back, so `fullgame7` crawled through mission 1 at HP 5 with
  fight searches reported at 1239 s (model stalls inside the plan window) and the slab2 replay's door-2 searches at
  1266 s. Port-forward restored, a watchdog now re-creates it, `fullgame8` restarted from power-on (complete build).
  slab2 did verify the slab crossing before the stall: hops of 56/44/30 steps, door-2 room reached at HP 5.
* **v153l (19:25 UTC Oct 4):** the mission-1 second boss is still a leak (fullgame4: a life; fullgame8: 5 -> 3 while
  the stall trigger waited and a 142-s escape search found nothing). The planner now engages him from first contact
  (trigger includes `survive_48`) and, when the escape search fails, the knockdown search runs there too (one landed
  knockdown sends him off). Test boss2a from `fg8-boss1-arrival.state`.
* **boss2a (19:35 UTC Oct 4):** from fullgame8's boss-room arrival (`fg8-boss1-arrival.state`, HP 5) the v153l build
  killed the boss (91 steps, 0 loss), lost one tick to the second boss, then a 23-step knockdown plan plus a 39-step
  escape ended the room - mission 2 at HP 7, no life lost. fullgame8 (without v153l) lost a life there at the same
  spot minutes earlier. Cluster **v156** launches with the complete build (c-l).
* **Forest ledge deaths (20:10 UTC Oct 4):** fullgame8 lost its second life at step 4206 exactly like fullgame5 at
  4115 - thrown off the forest ledge into the river right after a planned kill, no HP lost, the radar sees no pit
  there (the river is below the ledge, not beside it), and the nav search from the thrown state has one state. The
  planner's post-kill check backs away for only 6 steps (`SAFE_AFTER`), shorter than a thug's throw. **v153m:** on
  the forest ledge the planner's safe-after window is 16 steps (`planner.safe_after`), so a kill that is followed by
  a throw within 16 steps is rejected and the search continues.
* **v152 final (control, pre-v153 build, 20:11 UTC Oct 4):** 20 episodes, best distance 8715 px (one worker into the
  maze), no clear. v153 final: 7055 px. Both lacked the planner-first fights and the hideout/sealed-room fixes.
* **fullgame8's third death (20:30 UTC Oct 4, step 4686):** forest ledge again - a stall with two thugs 2-12 px away
  started a NAV plan (the nav trigger excluded nearby enemies only in the maze), and its 28-step walk was thrown off
  the ledge 12 steps later. **v153n:** nav never runs with a reachable enemy within 60 px anywhere (nav_force, the
  planner-blind fallback, still passes). Replay forest1 from `fg8-forest-nav9.state` checks the state.
* **v154 final (v153c build, 20:35 UTC Oct 4):** 20 episodes, best 7666 px, no clear (hideout/sealed-room fixes came
  later). forest1 (v153m, old nav trigger) from `fg8-forest-nav9.state`: the nav plan was cut at step 6 by the prefix
  check and the player still died within 150 steps - the fight planner never took the thugs because nav had the
  turn; forest2 tests v153n from the same state.
* **forest2 (20:40 UTC Oct 4):** `fg8-forest-nav9.state` (step 4674) is already mid-throw - both replays lose the life
  at replay step 12 regardless of build, then play on cleanly (kills at 7 -> 7, one nav plan cut by the prefix
  check). No fixture exists for the 60 steps before the throw, so v153m/n stand as the mitigation: planner kills on
  the ledge must survive 16 idle steps, and nav never runs with a thug within 60 px.
* **Stalactites flicker (21:25 UTC Oct 4):** fullgame8 (last life) still took stalactite ticks in the corridor with the
  v153f dodge active (HP 6 -> 3). A per-step probe shows why: the sprite alternates two tile sets every env step
  (235/251/233 on odd steps, 231/247/249 on even), so the radar saw it every other step and read it as a thug in
  between, and the dodge, re-decided each step, flip-flopped the player under it (x 108 <-> 105, hit at t=9).
  **v153o:** both tile sets are stalactite tiles, and the dodge direction latches for 8 steps once chosen. Replay
  corr6 from the room-5 state tests the corridor again.
* **corr7 debug trace (21:35 UTC Oct 4):** both remaining hideout leaks explained step by step. (1) A 56-step nav walk
  ended with a stalactite directly overhead (dx 0, top 33); the dodge wanted "left" for ten steps but plans outrank
  reflexes, and the first free step took 2 ticks - the nav-plan tail check only watched the life counter.
  **v153p-a:** the 24-step idle tail also cuts on an HP drop. (2) The corridor escape plan ended in MID-AIR over the
  slab gap (feet 83 -> 65 -> 58 at its last steps; 68 -> 86 -> 111 -> dead after it): escape wins by progress never
  checked the landing. **v153p-b:** `_grounded_after` - a progress win must survive 16 idle steps on screen, not
  falling, same HP (chunk-change wins exempt: the mission-1 cut-scene hides the sprite). Replay corr8 tests both.
* **corr8: hideout clean (21:45 UTC Oct 4):** from the room-5 state with v153e-p: room 5, the stalactite corridor
  (one nav plan cut at step 44 of 56 by the HP tail), the stairs and six verified slab hops with NO HP drop and no
  death, HP 7 on the slabs at the 450-step cap. Launched local `fullgame9` and cluster **v157** (20 workers, 14000
  steps) with the complete build (c-p). fullgame8 (older build, last life) is in the maze shaft at HP 7.
* **v155 final (v153c-g build, 22:25 UTC Oct 4):** 20 episodes, best 7278 px, no clear. fullgame8 (last life) is in
  the sealed room at HP 6: the respawning 20-HP thugs draw "no kill found" searches (46-122 s) and 4-step knockdowns;
  the door script fired once from x 105 on the floor but a thug was back before it opened. Its live state
  (`fg8-room7-live.state`, step 9400) is the fixture for r7fix5 on the complete build.
* **Sealed room mechanics, from fullgame8's live state (22:50 UTC Oct 4):** the room has two lanes and its thugs
  fight on the FRONT one. From the back lane the thug at the left screen edge is inert (12 kicks changed nothing,
  it never follows) and the knockdown search "wins" 4-step plans on it forever - fullgame8 has been looping there
  since step 9400. On the front lane, idle punches killed the 10-HP pair in 40 steps at no cost; then two 52-HP
  heavies spawn (slots 52/52) and the door stays shut until they are dead; four 90-128 s beam-24 searches found no
  kill on them. **v153q:** in the sealed room the runner changes lane toward the nearest enemy when |dy| > 8 before
  anything else, knockdowns there need a level enemy, and the heavies get a 300 s / beam 40 / 1-tick second pass.
  Replay r7fix6 from the live state tests the whole room.
* **fullgame9 over (22:17 UTC Oct 4):** game over at step 2405 - three lives lost at steps 2029, 2198 and 2405, each
  ~150 steps after the respawn at chunk 89 (end of mission 1), each a fall with no HP trace, in the mission-2 opening
  (106 -> 36 -> 73/74 -> ...); the replay from `fg9-a89-nav5.state` passed the same stretch untouched, so it is the
  model's own moves, not a fixed trap. **v153r:** planner-first fights from mission 2 on (not 3), and a periodic
  local save every 100 steps everywhere (cap 400) so the next death has a fixture. `fullgame10` launched.
* **v153s (23:20 UTC Oct 4):** r7fix6 showed why the lane change never happened: the sealed room's door tiles read as
  a skull zone, so the hall's skull-room block owned every step (4-step knockdown/evade plans outrank reflexes) and
  the fight searches kept targeting the inert back-lane thug. The sealed room is now excluded from the skull-room
  block, and its fight searches wait until the player is on the enemy's lane. Replay r7fix7 from the live state.
* **v153t (23:35 UTC Oct 4):** r7fix7 killed the sealed room's pair from the front lane at 7 -> 6 (lane change
  working) but its 52-HP heavies drew "no kill found" searches that ENDED in 46-66 s - candidates exhausted, not a
  timeout - and took 3 ticks. The v144 build killed them from HP 5 with 180-s searches (room7.log: 52 -> kill in 97,
  35 -> kill in 103). The difference is the v148b danger hook: the room's door tiles read as a skull zone, so every
  state near the door was pruned. The hook now returns False in the sealed room. Replay r7fix8.
* **v153u (00:05 UTC Oct 5):** r7fix8 killed the pair AND both 52-HP heavies at 6 -> 5 (danger hook off), the room
  was empty with every slot at 0 for 18 steps (steps 42-59) - and the door-column walk from x 34 at ~2.3 px/step
  did not reach x 105 before the next pair respawned; that pair read 20 at 0xC583 with the main slots at 0 and no
  kill was ever found on it. Now, between waves, the runner holds the door column (x 105) on the front lane and
  waits for the thugs to come, so the door script fires within a few steps of a wave dying; and the sealed room
  uses hit-verified slots only. Replay r7fix9.
* **fullgame10's catwalk fall (00:10 UTC Oct 5):** life lost at step 2168 in mission 2 (chunk 66, a catwalk with a
  ladder opening the radar reads as a pit) at HP 7 - a fall. The fixture 68 steps earlier (`fg10-a66-s2100.state`)
  replays through 66 -> 106 -> 45 -> ... -> 8 with no death, like fullgame9's three mission-2 falls: the model's own
  moves between plans, not a fixed trap. No change; planner-first in mission 2 (v153r) is the mitigation in place.
* **v153v (00:20 UTC Oct 5):** r7fix9 walked to the LEFT screen edge after a thug (x 11, thug at x 8 with 3 HP) and
  two verified-slot searches found nothing - a thug at the screen edge is unhittable. In the sealed room an enemy
  with screen x < 16 or > 144 now counts as far: go hold the door column on the front lane and let it follow inward;
  the fight search also waits while the nearest enemy sits at an edge. Replay r7fix10.
* **Mission-2 balcony (00:35 UTC Oct 5):** fullgame10's second death (step 4704, chunk 16: a balcony open at both
  screen ends, a thug leaping in) came 40 steps after a nav plan was cut at 32 of 39 by the HP check. Three replays
  from the fixture 104 steps earlier are IDENTICAL (the model is deterministic) and all survive with planned kills
  at 7 -> 6 - the live run differed only in runner state (its stall counters fired nav where the fresh runner
  fought). **v153w:** a cut nav plan stops 8 steps short instead of 2-4, and any cut blocks nav for 300 steps.
* **SEALED ROOM CLEARED FROM THE LIVE STATE (00:40 UTC Oct 5, r7fix10):** from `fg8-room7-live.state` (HP 6, last life,
  the state fullgame8 had looped in for 400 steps) with v153q-v: lane change to the front lane, the 10-HP pair
  killed, the first 52-HP heavy killed (98 steps, 1 tick), the second worn down on partial plans from HP 2,
  `room7-door: 48 UP from x=105 feet=90`, and the next trace line reads chunk 97 / SCX 0 / HP 7 - the hall entry,
  with its refill. Every room from power-on to the credits has now been passed in some run. fullgame10 (build c-r,
  last life, mission 2) stopped; `fullgame11` and cluster **v158** launched with the complete build (c-w).
* **r7fix10 final (00:45 UTC Oct 5):** confirmed - chunk 97 reached (hall entry, HP 7), lock room 1's 24-HP thug
  killed at 7 -> 7, run ended at its 700-step cap at SCX 68 inside the hall with no life lost.
* **v153x (00:40 UTC Oct 5):** fullgame11's mission-1 street went 7 -> 3 before the warehouse with the model fighting;
  planner-first now covers mission 1 too (every fight in the game is planned from first contact), with single-kill
  mode in the respawning warehouse (chunk 84). Applies from the next run.
* **v153y (01:30 UTC Oct 5):** fullgame11 lost its first life at step 4552 in the forest at HP 7 (a fall), ten steps
  after `nav-planner area=11 goal=[12] -> exit in 12 steps in 0s (states 2)` - the same 2-state 12-step "exit" that
  preceded fullgame9's river deaths. The forest ids flip 11/12 every few px, so a flip is not an exit there: in
  mission 3 the nav search now treats 11/12/49 as already visited and must find real progress. Replay f11 from
  `fg11-forest-s4500.state` (52 steps before the death) checks the old behaviour.
* **v153z (01:45 UTC Oct 5) - the forest deaths explained:** replay f12 from `fg11-forest-s4500.state` with a per-step
  trace: after a planned kill, a second thug stood 9-16 px away for 13 steps while the planner returned silently
  (no HP slot found for it) and the model punched in place with the river gap 30 px behind; step 41-44 the player
  flew backwards (feet 67 -> 58 -> 63 -> 83, x 119 -> 96) into the gap and vanished - a grab-throw, no HP lost. The
  knockdown search (score-gain win, prunes falls) is now the fallback for planner-blind enemies in EVERY mission,
  not just mission 4 and the hall. Replay f13 tests the same fixture.
* **f13 (01:55 UTC Oct 5): forest fixture survived.** Same deterministic state, v153z build: the blind thug got a
  10-step knockdown plan exactly where the throw happened, then two planned kills at 7 -> 7, no death in 200 steps.
* **f14 (02:05 UTC Oct 5):** forest traversal with v153y2 (novelty-mode nav, 11/12/49 marked recent): 400 steps,
  296 px of progress, HP 7 throughout, no death; nav yields real partials (77 steps / 126 px) instead of 12-step
  flip "exits". The complete build is now c-z; `fullgame11` (c-w) is in the mission-4 door room at HP 7, two lives.
* **v153aa (02:25 UTC Oct 5):** fullgame11 reached the mission-4 door room at 7 -> 4 and then stood at the right
  pillar while the hop hook fired at the edge pit ten times - nothing walked it to the door. From its step-7400
  state the door opens on 27 held UP presses at any x in 72-89 (both lanes). New hideout-door script: once the room
  is clear (no enemies, main/ALT slots < 10), closed-loop align to x 80, then hold UP 40. Test hdoor1.
* **hdoor2 (02:35 UTC Oct 5):** from fullgame11's stuck door-room state (with the past-arena flag the live run has) the
  hideout-door script aligned to x 83, held UP, entered the hideout at HP 7 and planned rooms 6/5 at 7 -> 7.
  fullgame11 (build c-w, parked at the pillar) stopped; `fullgame12` and cluster **v159** launched with the
  complete build (c-aa: planner-first everywhere, forest novelty nav, knockdown fallback everywhere, hideout door).
* **v153ab (03:25 UTC Oct 5):** fullgame12 (complete build) stalled at step 3500 in mission 2 (chunk 94, SCY 152):
  hanging mid-ladder with a thug 59 px left / 45 up - just outside the stalled-fight reach - while three 120-s nav
  searches found nothing. Offline, 20 held UPs reach the catwalk with the thug level (then he must be fought at
  once: 40 blind UPs died). New rule: climbing + no progress for 40 steps + no enemy below -> hold UP; the fight
  planner takes over at the top. Test ladder1 from `fg12-a94-s3500.state`.
* **v153ac (03:35 UTC Oct 5):** ladder1 climbed to the catwalk (chunk 30) but died there: the thug has no discoverable
  HP byte, the knockdown search failed in 1 s on an airborne root and blocked 12 steps, and the model's "right"
  walked into a throw off the catwalk's left end (feet 51 -> 36 -> 113 -> gone). Now a failed knockdown retries in
  4 steps and, for 12 steps, a planner-blind thug within 28 px gets a punching stance (face it, punch) instead of
  the model. Test ladder2.
* **v153ad (03:50 UTC Oct 5):** ladder2 died identically: the 34-step KILL plan ended with its jump kick adjacent to a
  second (slot-less) thug; the planner's 6-step safe-after check passed because a grab-throw costs no HP inside the
  window, and the player then flew off the catwalk. `_safe_after` now also fails when the player is lifted more than
  10 px off the floor (thrown) and its default window is 10 steps. Test ladder3.
* **v153ae/af (04:10 UTC Oct 5):** ladder3 died identically (the lift check had measured against a mid-jump-kick win
  state, and the runner's 6-step window overrode the new default); now the lift is measured against the search's
  starting floor and the window is 12 steps everywhere. ladder4 then PASSED the catwalk (the throw-ending kill became
  a partial) and ran the mission-2 ladder loop 26/66/25/106 - and died walking LEFT off a ledge right after a
  verified hop (model move, pit edge 1 px away). New generic veto: a model step onto a wide pit edge (within 8 px)
  becomes a punch, every mission. Test ladder5 (400 steps).
* **v154 - rewind-on-death teacher (04:25 UTC Oct 5):** ladder5 passed the catwalk and the pit edge, then died in
  the mission-2 conveyor stretch (chunk 66: pressing LEFT, carried RIGHT into a pit) - the fourth distinct way to
  die in that fixture, each patched one at a time. New technique for local demonstration runs (`DD_REWIND=1`): the
  runner snapshots the emulator every 50 steps (ring of 8) and, when a life is lost, reloads the snapshot taken
  >= 100 steps earlier, holds a punching stance for 8 x attempt steps (a timing shift so the deterministic model
  takes another line), and resumes; at most 3 rewinds per chunk; every rewind is logged and counted in the summary
  (`rewinds`), and the pre-rewind death state is saved. A clear produced this way is a STITCHED TEACHER
  DEMONSTRATION, not a live run - the hall clear of Oct 5 (run 57w) remains the live result. Test rewind1 from the
  mission-2 fixture.
* **rewind1 / v154b (04:45 UTC Oct 5):** the rewind works mechanically (5 rewinds logged, pre-rewind deaths saved)
  but the mission-2 ladder stretch (26/66/25/106) killed the model's own walk three times from the same step-300
  snapshot despite 8/16/24-step timing shifts, so the per-chunk cap ran out. The model's free steps are the hazard
  there, not any one fight. **v154b:** mission 2 is navigation-first - whenever no plan runs and no thug is within
  60 px, the nav planner (fall-checked, HP-strict prefix) produces the next walk, re-planning as soon as one ends
  (+24 after a miss); nav plan cap 20 -> 400. Test rewind2 on the same fixture.
* **rewind2 (05:05 UTC Oct 5): mission 2 crossed.** Nav-first + rewind on the same fixture: one rewind (the conveyor
  death at step 277 -> back to the step-150 snapshot), then nav plans walked the 26/66/25/106 ladder loop and on to
  chunks 8/7/5/4 (the mission-2 exit stretch) at HP 6 with all three lives. fullgame12 (parked on the ladder, old
  code) stopped; **fullgame13** launched from power-on with the complete build (c-af + v154/v154b) and
  `DD_REWIND=1`, 16000 steps - the first stitched-teacher attempt at a power-on clear.
* **v158 final (build c-w, 05:00 UTC Oct 5):** 20 episodes, distance_px_max": 8053, no clear.
* **v154c (06:00 UTC Oct 5):** fullgame13 (rewind teacher) used three rewinds in the forest's second half, where
  every thug is planner-blind and only knockdowns land. The f13 trace shows the actual death: a 43-step KILL plan
  whose last action was the running kick that killed - the player was airborne at the win, the lift check compared
  against the lane the plan had started on (lanes changed mid-plan), and the flight carried him off the ledge 10
  steps after the plan ended while two lane-align reflexes fought over him. Every kill now has to end grounded: idle
  until the feet hold still for 3 steps (appended to the plan), fail on fall / vanish / lost life or HP. Test f14b.
* **f14b (06:30 UTC Oct 5): forest second half crossed.** From `fg13-forest-s5400.state` (the state that killed
  fullgame13 three times) the grounded-kill build ran the rest of the forest, the door room (32-HP, partial) and
  reached the ARENA at HP 7 with no death in 200 steps - the planner-blind thugs were handled by knockdowns, and no
  kill ended in the air. Launched **fullgame14** from power-on with the complete build (c-af + v154/b/c) and
  `DD_REWIND=1`; fullgame13 (without v154c, two lives) continues as a second sample.
* **HALL ENTRY FROM POWER-ON (06:55 UTC Oct 5, fullgame13, rewind teacher):** power-on -> missions 1-4 -> hideout ->
  door 2 -> shaft -> sealed room (pair, both heavies at 6 -> 6, a respawned pair, then `room7-door: 48 UP from
  x=106`) -> chunk 97 at HP 7 and two lives, step ~9900. Four rewinds were used on the way (one at the mission-2
  ladders, three in the forest's second half) and one real life was lost in the forest; this is a stitched teacher
  demonstration. The hall ahead was cleared live from this exact entry on Oct 5 (run 57w).
* **v154d (07:00 UTC Oct 5):** in the hall's first room fullgame13 died at step 10552, rewound to the step-10450
  snapshot, died again at 10662, and then rewound TWICE to the step-10550 snapshot - taken 2 steps before the first
  death, i.e. already mid-throw - dying 2 steps after each reload; the chunk's three rewinds were gone and the next
  death was real (one life left). Rewind now skips any snapshot taken within 40 steps before a recorded death, never
  reuses the last snapshot, and keeps 16 snapshots (800 steps). `fullgame15` launched with it; fullgame13 (one
  life, hall) and fullgame14 (mission 2, old rewind logic) continue.
* **v154e (07:30 UTC Oct 5):** the hall-entry deaths of fullgame13 explained by replay h13: after a kill plan ended,
  the hall autopilot's walk-on step ("ovr:right") carried the player into a floor hole 3 px ahead (feet 68 -> 88 ->
  111 -> gone) while two thugs dropped in from the upper lane. The hole-margin veto applied only to the model's
  moves; it now also filters the autopilots' left/right steps in the hall. Test h14 from `fg13-hall-s10500.state`.
* **h15 (07:40 UTC Oct 5): hall entry room passed.** With the plain pit-edge veto applied to autopilot steps the
  replay from `fg13-hall-s10500.state` ran 200 steps at HP 7, 352 px of progress, no death (h14, with only the
  mission-2 hole-margin veto, died). fullgame15 (20 min old, without it) replaced by **fullgame16** on the complete
  build (c-af + v154a-e2) with rewind; fullgame13 (hall, last life) and fullgame14 (mission 3/4) continue.
* **fullgame13 at the ending (09:20 UTC Oct 5):** the gunman went down at step 12840 on the last life at HP 4 - but the
  run then stood at the throne doorway for 1500 steps: Marian reads as a planner-blind "enemy", so every 4 steps the
  knockdown search ran and the v153ac punching stance preempted the ending script's UP. From the live state
  (`fg13-ending-live.state`) 27 consecutive UPs fade the screen. **v154f:** no fight searches and no blind stance
  once the ending has started. Replay end13 from the live state to finish the demonstration.

## POWER-ON TO CREDITS (stitched teacher demonstration) - 05 Oct 2026, 09:30 UTC

**fullgame13** (local, `DD_REWIND=1`, complete build c-af + v154/b/c/d): power-on -> mission 1 (street, warehouse,
boss, second boss) -> mission 2 (ladders, catwalk, conveyors) -> mission 3 (forest, chunk-46 bruiser, door room)
-> arena -> mission-4 cave -> door room -> hideout (rooms 6/5, stalactite corridor, stairs, slabs) -> door-2 room
-> door 2 -> shaft and maze -> sealed room (pair, both heavies, respawned pair) -> HALL ENTRY at step ~9900 ->
lock room 1 -> skull room -> lock room 2 (wall block) -> lap-3 corridor -> throne room lock phase -> the gunman
(`ending: the gunman is gone at step 12840`, HP 4, last life) -> throne doorway. The live process then stood at the
doorway because Marian read as a planner-blind enemy (v154f fixes it); its live state `fullgame13-ending-live.state`
was continued with the fixed runner: `*** GAME COMPLETE *** step 27: the doorway fade started (hp 4, lives 2)`,
saved as `ignored/fullgame13-GAME-COMPLETE.state`.

Honest accounting: 8 rewinds (1 mission-2 ladders, 3 forest second half, 3 hall entry room of which 2 wasted on a
doomed snapshot, 1 lock room 2) and 2 real deaths (forest, hall entry) along the way; the ending tail was run from
the live state with one code fix. This is a stitched teacher demonstration, suitable for DAgger labelling. The
unassisted live results remain: the hall from its entry (run 57w, Oct 5) and whatever fullgame14 - currently in
the hall at HP 7 with NO deaths and NO rewinds from power-on - achieves.

* **Ending video (09:35 UTC Oct 5):** `app/ignored/videos/fullgame13-ending-credits.mp4` - the fixed runner from
  `fullgame13-ending-live.state`: doorway fade at step 27, then the ending scene (Billy and Marian) and credits;
  600 frames.
* **v154g - the gunman from the arrival spot (10:45 UTC Oct 5):** fullgame14 (power-on, no real deaths, 2 rewinds)
  reached the gunman at HP 5 and three 900-s / beam-60 searches found no drain (427/411/394 s, candidates exhausted).
  Offline from its arrival state (x 62): beam 80 found a 60-step line in 577 s, a 3-tick allowance found nothing,
  walking RIGHT to x 127 found nothing, walking LEFT to x 10 found a 118-step drain in 337 s - the drain exists from
  the far-left spot (where run 57w stood). The boss search now uses beam 80, and after two failed searches at a spot
  the runner walks to x <= 16 before searching again. fullgame14's live state (`fg14-boss-live.state`) is being
  continued with this build (end14, with video).

## POWER-ON TO CREDITS, NO REAL DEATHS - fullgame14, 05 Oct 2026, 11:00 UTC

**fullgame14** (local, `DD_REWIND=1`, build c-af + v154a-c): power-on -> all four missions -> hideout -> door 2 ->
maze -> sealed room (pair, both heavies) -> hall entry at HP 7 with NO deaths and NO rewinds -> hall (two rewinds:
chunk 96 at step 9348 and lock room 2 at 9748) -> throne room -> the gunman on screen at step 10449 (bar 64, HP 5,
lives 4). The live process's beam-60 searches found no drain (427/411/394/386 s); its throne-room state
(`fg14-boss-live.state`, step 10400) was continued on the v154g build: `kill in 107 steps, hp left 5 in 449s`
(beam 80, from the arrival spot - the reposition rule was not needed), `ending: the gunman is gone at step 205`,
`*** GAME COMPLETE *** step 241: the doorway fade started (hp 5, lives 4)`.
States: `ignored/fullgame14-GAME-COMPLETE.state`; video of the gunman fight, the doorway and the credits:
`ignored/videos/fullgame14-gunman-and-credits.mp4`.

Accounting: zero real deaths from power-on to credits; two rewinds (both in the hall, each ~150 steps); one
continuation from the live state after a planner-budget fix. Still a stitched teacher demonstration by the rewind
definition, but the cleanest one: the remaining gap to a live, unassisted clear is two hall rewinds and the beam-80
budget being in the build from the start (it is now).

* **Live attempt (11:05 UTC Oct 5):** `fullgame17` launched from power-on WITHOUT rewind on the complete build
  (c-af + v154a-g: beam-80 boss search, ending fixes, reposition rule), with video, 16000 steps - the unassisted
  clear attempt. Cluster **v160** (20 workers, 16000 steps, same build) launched as the live sample.
* **fullgame16 stuck (11:10 UTC Oct 5):** after its slab-row rewind (step 7246 -> 7100) the canonical tracker
  regressed to mission 2's 5/6 occurrence, so the mission-2 navigation-first rule fired at the slab row (chunk 6,
  SCY 112) every 24 steps - 220 "no exit found" searches with the player parked at the left edge. **v154h:** the
  nav-first rule requires not-beyond, not-hideout, not-maze. fullgame16 stopped (5 rewinds, no real deaths, its
  demonstration value spent); fullgame17 (live, no rewind) continues.
* **fullgame17 (live) lost a life in the warehouse at step 1717 (11:25 UTC Oct 5)** at HP 7 after knockdown plans on
  planner-blind thugs; the replay from the step-1700 fixture does not reproduce it (it reaches the boss room
  unharmed) - a periodic save taken mid-plan is not a faithful fixture because the runner's in-flight plan is not
  part of the emulator state. No change; the run continues (mission-1 boss and second boss then passed at 7 -> 7).
* **v154i (11:55 UTC Oct 5):** fullgame17 (live) is down two lives by mission 2, both to unplanned walks between
  fights that no fixture reproduces. New option `DD_NAV_FIRST=1`: navigation-first (fall-checked, HP-strict nav
  walks whenever no thug is within 60 px) in missions 1-4 before the arena, not just mission 2; the hideout and
  the hall keep their own autopilots. `fullgame19` launched live (no rewind) with it; fullgame17 continues.
* **v154j (12:00 UTC Oct 5):** fullgame17's second death (step 2522, mission 2) followed two `exit in 12 steps
  (states 2)` nav plans on the 16/17 chunk pair - the same flip-as-exit flaw as the forest's 11/12. Nav goal areas
  now exclude any id seen in the last 120 steps (falls back to novelty mode). fullgame19 relaunched with it.
* **v154k - island escape (12:30 UTC Oct 5):** fullgame18 (rewind backup) stood 4000 steps on a one-tile lift between
  two gaps on a mission-2 belt (chunk 66, SCY 152): every nav macro died instantly (states 5), walking either way
  fell, idling was safe, and nothing else ever fired. Offline from that state a running kick LEFT lands on the
  catwalk (x 83, feet 76). New rule: with a pit edge within 20 px on both sides, no plan and no thug within 48 px,
  try verified hops right then left. Test island1 from the stuck state.
* **island1 (12:40 UTC Oct 5):** from fullgame18's lift state the island rule found an 18-step verified hop at once
  and the run went on through 32/33/4/5/106/73/74/13/26... at HP 6 with no death. fullgame18 stopped (stuck 4000
  steps on the lift); `fullgame20` launched as the rewind backup on the newest build (c-af + v154a-k).
* **v154l (12:50 UTC Oct 5):** fullgame17 (live, last life) stalled 300+ steps on the mission-2 balcony: a thug waits
  IN the doorway 60 px away, the kill searches fail from range ("no kill found" every 50 s), nav has no exit (the
  exit is the door itself), the model idles. New generic rule: after 60 stalled steps and two failed searches, walk
  toward a far enemy (48-110 px, level, no pit ahead) until the planner can reach it. Test balc1 from the stalled state.
* **v154m (13:10 UTC Oct 5):** the balcony "stall" of fullgame17 is a game soft-lock: from its state no input moves
  the player (right/up/down 30 steps: x 47 unchanged; a kick shifts 5 px) and the doorway thug is frozen too - the
  same class as the mission-1 warehouse lock (v129). The v129 checkpoint reload never fired because it required an
  empty radar; after 500 frozen steps an on-screen enemy now no longer blocks it. Test balc2 from the step-4700
  fixture (before the freeze).
* **Balcony soft-lock (13:30 UTC Oct 5) - fix UNVERIFIED, not blocking:** fullgame17's chunk-16 freeze is a rare bad
  state (a game soft-lock: no input moves the player, the doorway thug frozen too). The v154m/m2/m3 reload changes
  (fire in any chunk after 500 frozen steps; unconditional checkpoint ring) are in but cannot be verified - the only
  fixture (`fg17-balc-s4700.state`) is already frozen, so there is no pre-freeze checkpoint to reload. Four clean
  runs (fullgame13/14/19/21) passed chunk 16 without the freeze, so it is not on the critical path. Left as-is.
* **v154j regression fixed (13:35 UTC Oct 5):** the v154j nav-goal change did `tracker.goal_areas() - {...}` but
  `goal_areas()` is None past the arena, so `None - set` crashed BOTH live runs (fullgame19, fullgame21) at the cave
  upper walkway (chunk 246) deep in mission 4 - fullgame19 after the arena with 2 deaths, fullgame21 clean to that
  point. Guarded with `(goal_areas() or set())`. Relaunched: `fullgame22` (live, no rewind) and `fullgame23`
  (rewind backup). Cluster v160 predates v154j and was unaffected.
* **v154k4 (13:45 UTC Oct 5):** the island-escape hop fired 5x on the mission-1 street (sewer gratings read as
  two-sided gaps, not real drops); it is now gated to mission 2+ (the mission-2 belt lift and mission-4 cave are
  where real one-tile islands exist). fullgame22/23 (barely started, step 600) replaced by `fullgame24` (live) and
  `fullgame25` (rewind backup).
* **fullgame24 game over (14:45 UTC Oct 5) - the unassisted wall:** the no-rewind live run died at step 3661, all 3
  lives lost in the mission-2 vertical-conveyor/balcony/catwalk stretch (chunks 66/94/16). The deaths are TRAVERSAL,
  not fights - the model's steps between planned fights fall off moving lifts and ledge ends; the fight planner did
  not lose a fight. This is the wall for an unassisted power-on clear. Rewind (fullgame25) gets through it.
  Conclusion: per-room reflex patching has reached diminishing returns on mission-2 traversal; the principled next
  step is DAgger (distill the planner/rewind teacher into the student so it stops the traversal falls) - the
  teacher-clears-first plan ([[dagger-after-teacher-clear]]) is now unblocked since the teacher clears power-on->credits.
* **v160 final (complete build c-af + v154a-g, 15:36 UTC Oct 5):** 20 episodes, best distance 7761 px, 0 cleared.
  The cluster workers run without rewind, so like fullgame24 they hit the mission-2 traversal wall; the batch is a
  distribution sample (best worker reached the maze), not a clear. Confirms the live-clear gap is mission-2 traversal.
* **v154n (18:35 UTC Oct 5):** the full-video run died in a skull-room loop - it exhausted the 3-rewind per-chunk cap
  in the skull room (96), then each respawn at the hall entry re-entered with no rewinds left and died again. Fix: the
  hall chunks (96/97) get a 10-rewind cap (the skull timing needs many 8-step-shifted attempts), and a REAL death
  resets the per-chunk rewind budget so each fresh hall attempt starts over. `fullgamevid3` relaunched (video, 20000
  steps) - the clean power-on->hall footage is re-recorded but this time the hall should not dead-end.
* **v154o/o2 (21:40 UTC Oct 5):** the full-video run (vid3) stalled 500+ steps at the hideout slab row (chunk 6, SCY
  112, far-left x 16): the hop hook needs a gap within 40 px but the gaps were 90+ px right, so nav looped "no exit".
  Added a stalled-hop fallback (12-decision no-move + a right-side gap anywhere -> walk-and-hop; verified offline it
  crosses, 89/90/91). Keyed off the hideout/maze latches, not tracker.beyond, since a hideout respawn regresses the
  canonical position and flips beyond False (the actual vid3 trigger). `fullgamevid4` relaunched (video, 22000 steps).
* **v154p (06:00 UTC Oct 6):** `fullgamevid4` (power-on, 22000 steps) cleared missions 1-3, the arena, the full hideout
  interior (slab row included - the v154o2 fix held in a real run, first validation outside DD_FORCE flags) and door 2,
  then DEAD-LOOPED in the post-door-2 maze. Two findings: (1) the `skull-room:`/`hall:` marker prints fire in the shaft
  (chunk 72) and sealed room (chunk 7) too - they are NOT reliable position signals; trust the saved-state `aNN`/`scxNN`
  suffixes. vid4 never reached the real throne; it oscillated chunks 72/7/106/97 from step 11400-13700. (2) In the hall
  (chunk 97) `island-escape` (two-sided-gap hop LEFT) and `stalled-hop` (right-gap hop RIGHT) ping-ponged x=52<->x=107
  forever on the tile-354 holes (6+ identical cycles, 23 min). Fix: `island-escape` is now `not maze_entered` (it was
  built for mission-2 belts, has no business in the hall); `stalled-hop` STAYS on in the maze - the tile-354 holes must
  be hopped rightward and it is the only maze hop mechanism (the line-1286 hook is already gated not-maze). Tested from
  the exact stuck state (halltest-loop, step 13700, a97): both-off and island-off/stalled-on BOTH stay in a97 ("nav no
  exit found", novelty exhausted recent=[7,26,27,28,29,30,31,56]) - that re-entry state is a novelty dead-end, same class
  as the cave's flicker-id problem. So the island-escape fix removes the ping-pong but the deeper post-door-2 maze
  novelty-nav is the real single-run wall (maze route solved live only once, run 57w). Note: the RHDP cluster
  (rosa-rrvnz) EXPIRED mid-run (API NXDOMAIN) - vid4/vid5 run planner-only, no vLLM student. `fullgamevid5` relaunched
  (power-on, 22000 steps, video) with the island-escape fix; the game itself is already beaten as a stitched demo
  (fullgame13). DAgger on the maze is the principled next step once a cluster is back.
