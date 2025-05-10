from os.path import exists
from pathlib import Path
import uuid
import random
from ddenv import DDEnv
from typing import Callable, Union
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.tune.registry import register_env
from ray.train import RunConfig
from ray import tune
from torch import nn
from tensorboard_callback import TensorboardCallback

num_agents = 10

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = DDEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    return _init

if __name__ == '__main__':


    ep_length = 2048 * 30
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 8, 'init_state': 'ignored/dd.gb.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': False, 'session_path': sess_path,
                'gb_path': 'ignored/dd.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                'use_screen_explore': True, 'extra_buttons': False
            }

# Create a simple multi-agent version of the above Env by duplicating the single-agent
# env n (n=num agents) times and having the agents act independently, each one in a
# different corridor.



if __name__ == "__main__":

    # The `config` arg passed into our Env's constructor (see the class' __init__ method
    # above). Feel free to change these.
    env_options = {
        "corridor_length": 10,
        "max_steps": 100,
        "num_agents": num_agents,  # <- only used by the multu-agent version.
    }

    #tune.register_env("env", lambda config: DDEnv(env_config))
    custom_config = {
        "model": {
            "conv_filters": [
                [32, [8, 8], 4],   # 32 filters, 8x8 kernel, stride 4
                [64, [4, 4], 2],   # 64 filters, 4x4 kernel, stride 2
                [128, [3, 3], 1]   # 128 filters, 3x3 kernel, stride 1
            ],
            "conv_activation": "relu",  # Activation function
        },
    }
    # Example config switching on rendering.
    base_config = (
        PPOConfig()
        # Configure our env to be the above-registered one.
        .environment("dd")
        # Plugin our env-rendering (and logging) callback. This callback class allows
        # you to fully customize your rendering behavior (which workers should render,
        # which episodes, which (vector) env indices, etc..). We refer to this example
        # script here for further details:
        # https://github.com/ray-project/ray/blob/master/rllib/examples/envs/env_rendering_and_recording.py  # noqa
        .debugging(log_level="INFO")
        .framework(framework="torch")
        .rollouts(num_rollout_workers=3)
        .resources(num_cpus_per_worker=3)
    )

    base_config.training(model=custom_config["model"])

    ray.init()
    env_name = "dd"
    register_env(env_name, DDEnv)
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 2048 * 30 *12*1000 },
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=base_config.to_dict(),
    )
    #algo = PPO(env=DDEnv, config=base_config)
    #while True:
    #    print(algo.train())
#
#    if num_agents > 0:
#        print("GOT HERE!!!")
#        base_config.multi_agent(
##            policies={f"p{i}" for i in range(args.num_agents)},
#            policy_mapping_fn=lambda aid, eps, **kw: f"p{aid}",
#        )

#    run_rllib_example_script_experiment(base_config, args)