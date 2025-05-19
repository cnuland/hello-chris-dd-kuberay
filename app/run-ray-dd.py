# run-ray-dd.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from pathlib import Path
import warnings
import os
import torch

# Import your custom modules directly.
# These files (ddenv.py, custom_cnn_model.py) and the 'ignored/' directory
# must be in the working_dir of the Ray job.
from ddenv import DDEnv 
from custom_cnn_model import CustomCNNModel

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    """
    Main function to initialize Ray, configure, and run PPO training.
    """
    # --- Read custom S3 Env Vars and set standard AWS Env Vars for boto3/fsspec ---
    # This helps boto3 (used by Ray Tune and DDEnv) to automatically pick up credentials
    # and endpoint configurations if you are using S3_... prefixed env vars.
    s3_bucket_name_custom = os.environ.get('S3_BUCKET_NAME')
    s3_endpoint_url_custom = os.environ.get('S3_ENDPOINT_URL')
    s3_access_key_id_custom = os.environ.get('S3_ACCESS_KEY_ID')
    s3_secret_access_key_custom = os.environ.get('S3_SECRET_ACCESS_KEY')
    s3_region_name_custom = os.environ.get('S3_REGION_NAME')

    if s3_access_key_id_custom:
        os.environ['AWS_ACCESS_KEY_ID'] = s3_access_key_id_custom
        print("Main Script: Set AWS_ACCESS_KEY_ID from S3_ACCESS_KEY_ID.")
    if s3_secret_access_key_custom:
        os.environ['AWS_SECRET_ACCESS_KEY'] = s3_secret_access_key_custom
        print("Main Script: Set AWS_SECRET_ACCESS_KEY from S3_SECRET_ACCESS_KEY.")
    if s3_region_name_custom:
        os.environ['AWS_DEFAULT_REGION'] = s3_region_name_custom
        print(f"Main Script: Set AWS_DEFAULT_REGION to {s3_region_name_custom} from S3_REGION_NAME.")
    if s3_endpoint_url_custom:
        os.environ['AWS_S3_ENDPOINT_URL'] = s3_endpoint_url_custom 
        os.environ['AWS_ENDPOINT_URL_S3'] = s3_endpoint_url_custom 
        print(f"Main Script: Set AWS_S3_ENDPOINT_URL and AWS_ENDPOINT_URL_S3 to {s3_endpoint_url_custom} from S3_ENDPOINT_URL.")
    # --- End S3 Env Var Setup ---

    # --- Check PyTorch CUDA availability ---
    pytorch_cuda_available = torch.cuda.is_available()
    print(f"Main Script: PyTorch CUDA available: {pytorch_cuda_available}")
    if pytorch_cuda_available:
        print(f"Main Script: PyTorch CUDA version: {torch.version.cuda}")
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
        ray.init(address='auto') # Connects to the KubeRay cluster

    print(f"Main Script: Ray version: {ray.__version__}") # Should be 2.44.0
    print(f"Main Script: Ray cluster resources: {ray.cluster_resources()}")

    # --- Register custom components ---
    ModelCatalog.register_custom_model("custom_cnn", CustomCNNModel)
    print("Main Script: Custom model 'custom_cnn' registered.")
    
    register_env("dd_env", lambda cfg_passed_by_rllib: DDEnv(cfg_passed_by_rllib))
    print("Main Script: Custom environment 'dd_env' registered.")
    # --- End Registrations ---

    # --- Storage Path Configuration for Ray Tune ---
    # Use the S3 bucket name read at the start (which might be None)
    s3_bucket_name_for_tune = s3_bucket_name_custom 

    if s3_bucket_name_for_tune:
        s3_tune_results_prefix = "ray_tune_results/dd_ppo_experiment_final" # Unique prefix for this experiment
        storage_path = f"s3://{s3_bucket_name_for_tune}/{s3_tune_results_prefix}"
        print(f"Main Script: Using S3 storage path for Ray Tune: {storage_path}")
        if s3_endpoint_url_custom: 
            print(f"Main Script: S3 endpoint URL for Tune: {s3_endpoint_url_custom}")
    else:
        print("Main Script: S3_BUCKET_NAME not set. Using local storage path for Ray Tune.")
        storage_path = str(Path("~/ray_results_kuberay/dd_ppo_final").expanduser())
        Path(storage_path).mkdir(parents=True, exist_ok=True)
    # --- End Storage Path Configuration ---
    
    # --- Environment and PPO Configuration ---
    # Max steps per episode in DDEnv
    max_episode_steps = 2048 * 10 # Example: Reduced for potentially faster episode turnover with random starts
                                  # Adjust as needed. Original was 2048 * 30.

    base_env_config = { 
        'headless': True, 
        'save_final_state': True, # DDEnv will attempt to save state to S3 if configured
        'early_stop': False,
        'action_freq': 8, 
        'init_state_dir': 'ignored/', # Directory where .state files are located
        'available_init_states': [    # List of .state files to randomly choose from
            'boss.dd.gb.state', 'lvl2-dd.gb.state', 'lvl3.5-dd.gb.state',
            'dd.gb.state', 'lvl-1.5.dd.gb.state', 'lvl3-dd.gb.state'
        ],
        'max_steps': max_episode_steps, 
        'print_rewards': False, # DDEnv has its own print logic, RLlib also logs rewards
        'save_video': True,    # Enable video saving in DDEnv (will go to S3 if configured)
        'fast_video': True,    # DDEnv's frame saving logic was updated to save every step if save_video is True
        'gb_path': 'ignored/dd.gb', 
        'debug': False,
        'sim_frame_dist': 2_000_000.0, 
        'use_screen_explore': True, 
        'extra_buttons': False,
        # Pass S3 config to DDEnv instances using the initially read custom vars
        's3_bucket_name': s3_bucket_name_custom,
        's3_endpoint_url': s3_endpoint_url_custom,
        's3_access_key_id': s3_access_key_id_custom,
        's3_secret_access_key': s3_secret_access_key_custom,
        's3_region_name': s3_region_name_custom,
        's3_video_prefix': 'ddenv_videos/ppo_final/', # More specific prefix for videos from this run
        's3_state_prefix': 'ddenv_states/ppo_final/'  # More specific prefix for states from this run
    }

    num_rollout_workers = 3 
    num_envs_per_worker = 3 
    detected_gpus_in_cluster = int(ray.cluster_resources().get("GPU", 0))
    ppo_learner_num_gpus = 1 if detected_gpus_in_cluster > 0 and pytorch_cuda_available else 0
    print(f"Main Script: Detected GPUs in Ray cluster: {detected_gpus_in_cluster}")
    print(f"Main Script: Requesting {ppo_learner_num_gpus} GPU(s) for the PPO learner.")

    # Total timesteps for the entire training run
    total_training_timesteps = 300_000 # For initial testing; increase for a full run

    # Learning Rate Schedule
    initial_lr = 5e-5
    final_lr = 1e-6 
    lr_schedule = [
        [0, initial_lr],
        [total_training_timesteps, final_lr]
    ]
    print(f"Main Script: Using learning rate schedule: {lr_schedule}")

    ppo_algo_config = PPOConfig()
    ppo_algo_config.sgd_minibatch_size = 500
    ppo_algo_config.num_sgd_iter = 10
    # ppo_algo_config.clip_param = 0.2
    # ppo_algo_config.use_gae = True
    # ppo_algo_config.lambda_ = 0.95
    # ppo_algo_config.vf_loss_coeff = 0.5
    # ppo_algo_config.entropy_coeff = 0.01

    ppo_algo_config = (
        ppo_algo_config 
        .environment(env="dd_env", env_config=base_env_config, disable_env_checking=True)
        .framework("torch") 
        .env_runners( 
            num_env_runners=num_rollout_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            num_cpus_per_env_runner=3, 
        )
        .training( 
            model={"custom_model": "custom_cnn"}, 
            gamma=0.99,
            lr_schedule=lr_schedule, # Using the learning rate schedule
            train_batch_size=5000 
        )
        .resources(
            num_gpus=ppo_learner_num_gpus 
        )
        .debugging(
            log_level="INFO" 
        )
        .api_stack( # To use ModelV2
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
    )
    
    full_ppo_config_dict = ppo_algo_config.to_dict()
    
    print(f"Main Script: Launching PPO training. Storing results/checkpoints in: {storage_path}")
    
    results_analysis = tune.run(
        "PPO",
        name="PPO_DoubleDragon_KubeRay_Final", # Experiment name
        stop={"timesteps_total": total_training_timesteps}, 
        checkpoint_freq=20, 
        checkpoint_at_end=True,
        storage_path=storage_path, 
        config=full_ppo_config_dict,
        # verbose=1, # 0 (silent), 1 (table), 2 (trial detail), 3 (debug)
    )

    print("Main Script: Training completed.")

    if results_analysis:
        try:
            best_trial = results_analysis.get_best_trial("episode_reward_mean", mode="max", scope="last")
            if best_trial:
                print(f"Main Script: Best trial path (logical): {best_trial.path}") 
                print(f"Main Script: Best trial config: {best_trial.config}")
                print(f"Main Script: Best trial last result: {best_trial.last_result}")
                
                best_checkpoint_obj = results_analysis.get_best_checkpoint(
                    best_trial, metric="episode_reward_mean", mode="max"
                )
                if best_checkpoint_obj:
                    best_checkpoint_path_str = str(best_checkpoint_obj.path if hasattr(best_checkpoint_obj, 'path') else best_checkpoint_obj)
                    print(f"Main Script: Best checkpoint found at (S3 URI or local): {best_checkpoint_path_str}")
            else:
                print("Main Script: No best trial found (or metric not reported).")
        except Exception as e:
            print(f"Main Script: Error processing results: {e}")
            if hasattr(results_analysis, 'results'):
                 print(f"Main Script: All trial results: {results_analysis.results}")
            else:
                print("Main Script: No 'results' attribute in results_analysis object.")
    else:
        print("Main Script: Tune run did not return a valid ExperimentAnalysis object or no trials were run.")

    print("Main Script: run-ray-dd.py finished.")

if __name__ == "__main__":
    main()
