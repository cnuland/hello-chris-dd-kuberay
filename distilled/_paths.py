"""Path helpers for the distilled approach (moved out of app/ in this directory).

* The shared PyBoy environment (app/ddenv.py) and the PPO model (app/custom_cnn_model.py) stay in app/ because the
  original PPO pipeline uses them too. Importing this module puts app/ on sys.path when running from the repo; on the
  Ray cluster the submit scripts stage everything into one flat working_dir, so the flat imports just work.
* The ROM and save states are NOT in git (app/ignored/ is git-ignored; the ROM is copyrighted). assets_dir() finds
  them: $DD_ASSETS_DIR, else ./ignored (the layout the cluster submit scripts stage), else <repo>/app/ignored.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.normpath(os.path.join(HERE, "..", "app"))
if os.path.isdir(APP) and APP not in sys.path:
    sys.path.append(APP)


def assets_dir() -> str:
    env = os.environ.get("DD_ASSETS_DIR")
    if env:
        return env
    if os.path.isdir("ignored"):
        return "ignored"
    return os.path.join(APP, "ignored")


def asset(name: str) -> str:
    return os.path.join(assets_dir(), name)
