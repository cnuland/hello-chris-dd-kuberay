# run-ray-dd.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig # Import PPOConfig
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pathlib import Path
import warnings
import os
import torch # Import torch to check for CUDA availability

# Import your custom modules directly.
# Ensure dd_env.py and custom_cnn_model.py are in the same directory
# or correctly placed in your Docker image's PYTHONPATH / WORKDIR.
from ddenv import DDEnv # Assuming this is the Gymnasium 1.0.0 compatible version
from custom_cnn_model import CustomCNNModel

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning) # Keep this for now

def main():
    """
    Main function to initialize Ray, configure, and run PPO training.
    """
    # --- Check PyTorch CUDA availability ---
    pytorch_cuda_available = torch.cuda.is_available()
    print(f"Main Script: PyTorch CUDA available: {pytorch_cuda_available}")
    if pytorch_cuda_available:
        print(f"Main Script: PyTorch CUDA version: {torch.version.cuda}")
        # Attempt to get cuDNN version, handle potential errors if not fully available
        try:
            cudnn_version = torch.backends.cudnn.version()
            print(f"Main Script: PyTorch cuDNN version: {cudnn_version}")
        except Exception as e:
            print(f"Main Script: Could not retrieve PyTorch cuDNN version. Error: {e}")
        print(f"Main Script: PyTorch cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"Main Script: Number of GPUs PyTorch sees: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            try:
                print(f"Main Script: Current GPU name: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"Main Script: Could not retrieve GPU name. Error: {e}")
    else:
        print("Main Script: PyTorch cannot find CUDA. Training will be on CPU.")
    # --- End Check ---

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
    ep_length = 2048 * 15 # Original episode length
    # For faster testing during debugging, you might want to reduce this:
    # ep_length = 2048 

    base_env_config = { # This will be part of the PPOConfig's env_config
        'headless': True, 'save_final_state': True, 'early_stop': False,
        'action_freq': 8, 'init_state': 'ignored/dd.gb.state', 'max_steps': ep_length,
        'print_rewards': False, # Set to False for less log spam from workers; RLlib logs aggregated rewards
        'save_video': False,    # Set to True if you want videos from workers
        'fast_video': False,
        'gb_path': 'ignored/dd.gb', 'debug': False,
        'sim_frame_dist': 2_000_000.0, 'use_screen_explore': True, 'extra_buttons': False
    }

    num_rollout_workers = 3 # As per your original setup
    num_envs_per_worker = 3 # As per your original setup
    
    # Learner GPU (driver process on head node)
    # Your Ray cluster head has 1 GPU.
    detected_gpus_in_cluster = int(ray.cluster_resources().get("GPU", 0))
    ppo_learner_num_gpus = 1 if detected_gpus_in_cluster > 0 and pytorch_cuda_available else 0 # Only request GPU if PyTorch sees it
    print(f"Main Script: Detected GPUs in Ray cluster: {detected_gpus_in_cluster}")
    print(f"Main Script: Requesting {ppo_learner_num_gpus} GPU(s) for the PPO learner.")


    # 1. Create the PPOConfig instance
    ppo_algo_config = PPOConfig()

    # 2. Set PPO-specific attributes directly on the instance.
    ppo_algo_config.sgd_minibatch_size = 500
    ppo_algo_config.num_sgd_iter = 10
    # Add other PPO-specific parameters as needed, e.g.:
    # ppo_algo_config.clip_param = 0.2
    # ppo_algo_config.use_gae = True
    # ppo_algo_config.lambda_ = 0.95 # GAE lambda
    # ppo_algo_config.vf_loss_coeff = 0.5
    # ppo_algo_config.entropy_coeff = 0.01


    # 3. Apply general AlgorithmConfig builder methods
    ppo_algo_config = (
        ppo_algo_config # Start chaining from the config object with attributes set
        .environment(env="dd_env", env_config=base_env_config, disable_env_checking=True)
        .framework("torch") # Ensure PyTorch framework is used for GPU training
        .env_runners( 
            num_env_runners=num_rollout_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            num_cpus_per_env_runner=3, # CPUs per rollout worker actor
            # num_gpus_per_env_runner=0 # Correct for CPU-based environments like PyBoy
        )
        .training( 
            model={"custom_model": "custom_cnn"}, # This uses the ModelV2 API
            gamma=0.99,
            lr=5e-5,
            train_batch_size=5000 
        )
        .resources(
            num_gpus=ppo_learner_num_gpus # GPUs for the PPO learner process
        )
        .debugging(
            log_level="INFO" # Or "DEBUG" for more verbose RLlib logs
        )
        # --- CRITICAL FIX: Deactivate the new API stack ---
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        # --- END CRITICAL FIX ---
    )
    # --- End Configuration ---

    full_ppo_config_dict = ppo_algo_config.to_dict()
    
    storage_path = str(Path("~/ray_results_kuberay/dd_ppo_minimal_corrected").expanduser())
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    
    # Adjust total_timesteps_to_train for testing vs. full run
    # total_timesteps_to_train = ep_length * 100 # A larger number for a longer run
    total_timesteps_to_train = 300_000 # For initial testing

    print(f"Main Script: Launching PPO training. Storing results in: {storage_path}")
    # print(f"Main Script: PPO Configuration: {full_ppo_config_dict}") # Can be very verbose


    results_analysis = tune.run(
        "PPO",
        name="PPO_DoubleDragon_KubeRay_Minimal", 
        stop={"timesteps_total": total_timesteps_to_train},
        checkpoint_freq=20, # Save checkpoint every 20 training iterations
        checkpoint_at_end=True,
        storage_path=storage_path,
        config=full_ppo_config_dict,
        # verbose=1, # Tune verbosity: 0 (silent), 1 (table), 2 (trial detail), 3 (debug)
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
                best_checkpoint_path_str = str(best_checkpoint_obj) 
                print(f"Main Script: Best checkpoint found at: {best_checkpoint_path_str}")
    else:
        print("Main Script: Tune run did not return a valid ExperimentAnalysis object or no trials were run.")

    print("Main Script: run-ray-dd.py finished.")

if __name__ == "__main__":
    main()

