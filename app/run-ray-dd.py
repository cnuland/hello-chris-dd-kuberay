# run-ray-dd.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig # Import PPOConfig
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pathlib import Path
import warnings
import os

# Import your custom modules directly.
# Ensure dd_env.py and custom_cnn_model.py are in the same directory
# or correctly placed in your Docker image's PYTHONPATH / WORKDIR.
from ddenv import DDEnv
from custom_cnn_model import CustomCNNModel

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    """
    Main function to initialize Ray, configure, and run PPO training.
    """
    if not ray.is_initialized():
        print("Main Script: Initializing Ray...")
        ray.init(address='auto') # Connect to KubeRay cluster

    print(f"Main Script: Ray version: {ray.__version__}")
    print(f"Main Script: Ray cluster resources: {ray.cluster_resources()}")

    # --- Register custom components ---
    ModelCatalog.register_custom_model("custom_cnn", CustomCNNModel)
    print("Main Script: Custom model 'custom_cnn' registered.")
    
    # The lambda for env creation uses the config passed by RLlib at instantiation time.
    register_env("dd_env", lambda cfg_passed_by_rllib: DDEnv(cfg_passed_by_rllib))
    print("Main Script: Custom environment 'dd_env' registered.")
    # --- End Registrations ---

    # --- Configuration ---
    ep_length = 2048 * 15
    base_env_config = { # This will be part of the PPOConfig's env_config
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 8, 'init_state': 'ignored/dd.gb.state', 'max_steps': ep_length,
        'print_rewards': False, 'save_video': True, 'fast_video': True,
        'gb_path': 'ignored/dd.gb', 'debug': False,
        'sim_frame_dist': 2_000_000.0, 'use_screen_explore': True, 'extra_buttons': False
    }

    num_rollout_workers = 3
    num_envs_per_worker = 3
    # Assuming the driver (this script) runs on the head node,
    # which has a GPU as per your KubeRay setup.
    ppo_learner_num_gpus = 1

    # 1. Create the PPOConfig instance
    ppo_algo_config = PPOConfig()

    # 2. Set PPO-specific attributes directly on the instance.
    #    This is the fix for the TypeError related to __init__ and previous 'int not callable'.
    ppo_algo_config.sgd_minibatch_size = 500
    ppo_algo_config.num_sgd_iter = 10
    # If you have other PPO-specific parameters like clip_param, use_critic, use_gae, kl_coeff, etc.,
    # set them here as direct attributes if they are not constructor args or part of .training().
    # Example:
    # ppo_algo_config.clip_param = 0.2
    # ppo_algo_config.use_gae = True
    # ppo_algo_config.lambda_ = 0.95
    # ppo_algo_config.vf_loss_coeff = 0.5
    # ppo_algo_config.entropy_coeff = 0.01

    # 3. Apply general AlgorithmConfig builder methods to the modified config object
    ppo_algo_config = (
        ppo_algo_config # Start chaining from the config object with attributes set
        .environment(env="dd_env", env_config=base_env_config, disable_env_checking=True)
        .framework("torch")
        .env_runners(
            num_env_runners=num_rollout_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            num_cpus_per_env_runner=3,
        )
        .training( # General training parameters
            model={"custom_model": "custom_cnn"},
            gamma=0.99,
            lr=5e-5,
            train_batch_size=5000 # This is a general AlgorithmConfig training parameter
        )
        .resources(num_gpus=ppo_learner_num_gpus) # For the learner/driver
        .debugging(log_level="INFO")
    )
    # --- End Configuration ---

    full_ppo_config_dict = ppo_algo_config.to_dict()
    
    # Define storage path for results (ensure this path is writable on the head node)
    storage_path = str(Path("~/ray_results_kuberay/dd_ppo_minimal_corrected").expanduser())
    Path(storage_path).mkdir(parents=True, exist_ok=True) # Ensure directory exists
    
    total_timesteps_to_train = 300_000 # A smaller value for quicker testing

    print(f"Main Script: Launching PPO training. Storing results in: {storage_path}")

    # Run the training directly in this script
    results_analysis = tune.run(
        "PPO",
        name="PPO_DoubleDragon_KubeRay_Minimal", # Experiment name
        stop={"timesteps_total": total_timesteps_to_train},
        checkpoint_freq=20,
        checkpoint_at_end=True,
        storage_path=storage_path,
        config=full_ppo_config_dict,
        # verbose=1, # Adjust verbosity as needed
    )

    print("Main Script: Training completed.")

    if results_analysis:
        best_trial = results_analysis.get_best_trial("episode_reward_mean", mode="max", scope="last")
        if best_trial:
            print(f"Main Script: Best trial path: {best_trial.path}")
            best_checkpoint_obj = results_analysis.get_best_checkpoint(
                best_trial, metric="episode_reward_mean", mode="max"
            )
            if best_checkpoint_obj:
                best_checkpoint_path_str = str(best_checkpoint_obj) # Handles path string or Checkpoint object
                print(f"Main Script: Best checkpoint found at: {best_checkpoint_path_str}")
    else:
        print("Main Script: Tune run did not return a valid ExperimentAnalysis object.")

    print("Main Script: run-ray-dd.py finished.")

if __name__ == "__main__":
    main()