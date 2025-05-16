
from os.path import exists
from pathlib import Path
import uuid
from ddenv import DDEnv
import warnings
warnings.filterwarnings("ignore")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray import tune
from ray.rllib.models import ModelCatalog
from custom_cnn_model import CustomCNNModel

ModelCatalog.register_custom_model("custom_cnn", CustomCNNModel)

# Define parallel rollout agents
num_agents = 10

# Set up session
ep_length = 2048 * 15
sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

env_config = {
    'headless': True,
    'save_final_state': True,
    'early_stop': False,
    'action_freq': 8,
    'init_state': 'ignored/dd.gb.state',
    'max_steps': ep_length,
    'print_rewards': True,
    'save_video': True,
    'fast_video': True,
    'session_path': str(sess_path),  # Ensure this is str, not PosixPath
    'gb_path': 'ignored/dd.gb',
    'debug': False,
    'sim_frame_dist': 2_000_000.0,
    'use_screen_explore': True,
    'extra_buttons': False
}

# Register custom environment
register_env("dd_env", lambda config: DDEnv(config))

ray.init()

config = (
    PPOConfig()
    .environment(env="dd_env", env_config=env_config)  # Correct env name here
    .framework("torch")
    .rollouts(
        num_rollout_workers=3,
        num_envs_per_worker=3,

        )
    .training(model={
        "custom_model": "custom_cnn",
    })
)

# Run training
tune.run(
    "PPO",
    name="PPO_DoubleDragon",
    stop={"timesteps_total": ep_length * 1000},
    checkpoint_freq=10,
    storage_path=str(Path("~/ray_results/dd").expanduser()),  # Correct usage of storage_path
    config=config.to_dict()
)
