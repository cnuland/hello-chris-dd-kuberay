import random
import string
# import sys # sys.path modification is generally not recommended for Ray workers
import os
from math import floor
import uuid
import numpy as np
from einops import rearrange
from skimage.transform import resize # Ensure scikit-image is in your Pipfile/requirements
from pathlib import Path
import mediapy as media # Ensure mediapy is in your Pipfile/requirements
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from gymnasium import Env, spaces # Correct import for Gymnasium
from typing import Optional, Tuple, Dict, Any
import tempfile # For temporary video files
import boto3 # For S3 interaction

class DDEnv(Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config if config is not None else {}

        default_config = {
            'headless': True, 'save_final_state': False, 'early_stop': False,
            'action_freq': 8, 'init_state': 'ignored/dd.gb.state',
            'max_steps': 2048 * 30, 
            'print_rewards': True,
            'save_video': False, 
            'fast_video': True, 
            'gb_path': 'ignored/dd.gb', 'debug': False,
            'sim_frame_dist': 2_000_000.0, 'use_screen_explore': True,
            'extra_buttons': False,
            's3_bucket_name': os.environ.get('S3_BUCKET_NAME'),
            's3_endpoint_url': os.environ.get('S3_ENDPOINT_URL'),
            's3_access_key_id': os.environ.get('S3_ACCESS_KEY_ID'),
            's3_secret_access_key': os.environ.get('S3_SECRET_ACCESS_KEY'),
            's3_region_name': os.environ.get('S3_REGION_NAME'),
            's3_video_prefix': 'ddenv_videos/',
            's3_state_prefix': 'ddenv_states/' # New prefix for saved states
        }
        
        for key, val in default_config.items():
            config.setdefault(key, val)

        self.debug = config['debug']
        self.gb_path = Path(config['gb_path']) 
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.headless = config['headless']
        self.init_state = Path(config['init_state'])
        self.act_freq = int(config['action_freq'])
        self.max_steps = int(config['max_steps'])
        self.early_stopping = config['early_stop'] 
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.use_screen_explore = config['use_screen_explore']
        
        self.frame_stacks = 3
        self._single_frame_obs_shape = (36, 40, 3) 
        self.recent_frames_np_shape = (self.frame_stacks, *self._single_frame_obs_shape)

        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16 
        if self.col_steps <= 0: self.col_steps = 1

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B,
            97, 98, 99 
        ]
        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_UP
        ]
        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        screen_stacked_height = self.frame_stacks * self._single_frame_obs_shape[0]
        exploration_mem_height = self.memory_height 
        recent_mem_height = self.memory_height
        total_height = (
            exploration_mem_height + self.mem_padding +
            recent_mem_height + self.mem_padding +
            screen_stacked_height
        )
        final_obs_width = self._single_frame_obs_shape[1]
        final_obs_channels = self._single_frame_obs_shape[2]

        self.observation_space_shape = (total_height, final_obs_width, final_obs_channels)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_space_shape, dtype=np.uint8
        )

        # S3 Client Initialization
        self.s3_client = None
        self.s3_bucket_name = config.get('s3_bucket_name')
        self.s3_video_prefix = config.get('s3_video_prefix', 'ddenv_videos/')
        self.s3_state_prefix = config.get('s3_state_prefix', 'ddenv_states/') # For saved states

        # Initialize S3 client only if bucket name is provided and either video or state saving is enabled
        if (self.save_video or self.save_final_state) and self.s3_bucket_name:
            try:
                s3_params = {
                    'aws_access_key_id': config.get('s3_access_key_id'),
                    'aws_secret_access_key': config.get('s3_secret_access_key'),
                    'region_name': config.get('s3_region_name')
                }
                endpoint_url = config.get('s3_endpoint_url')
                if endpoint_url:
                    s3_params['endpoint_url'] = endpoint_url
                
                s3_params = {k: v for k, v in s3_params.items() if v is not None}

                self.s3_client = boto3.client('s3', **s3_params)
                print(f"DDEnv: S3 client initialized for bucket '{self.s3_bucket_name}'. Endpoint: {endpoint_url if endpoint_url else 'AWS Default'}")
            except Exception as e:
                print(f"DDEnv Warning: Failed to initialize S3 client. S3 features disabled. Error: {e}")
                self.s3_client = None 
                # self.save_video = False # Don't disable video saving yet, might save locally if s_path was configured
                # self.save_final_state = False 
        elif (self.save_video or self.save_final_state) and not self.s3_bucket_name:
            print("DDEnv Warning: S3 saving enabled but `s3_bucket_name` not configured. S3 features disabled.")
            # self.save_video = False
            # self.save_final_state = False


        if not self.gb_path.exists():
            script_dir = Path(__file__).parent.resolve()
            potential_path = script_dir / self.gb_path
            if potential_path.exists(): self.gb_path = potential_path
            else:
                potential_path_cwd = Path.cwd() / self.gb_path
                if potential_path_cwd.exists(): self.gb_path = potential_path_cwd
                else: raise FileNotFoundError(f"GB ROM not found at {self.gb_path}, {script_dir / self.gb_path}, or {potential_path_cwd}")
        
        window = 'headless' if self.headless else 'SDL2'
        self.pyboy = PyBoy(str(self.gb_path), debugging=self.debug, disable_input=False, window_type=window)
        if window == 'headless': self.pyboy.set_emulation_speed(0)

        self.screen = self.pyboy.botsupport_manager().screen()
        self.step_count = 0
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8] 
        self.full_frame_writer = None
        self.current_video_temp_file = None 
        self.game_state_reward_components = {}
        self.previous_total_game_state_reward = 0.0
        self.unique_levels_completed_in_episode = 0 

    def _get_initial_game_state(self):
        resolved_init_state = self.init_state
        if not resolved_init_state.is_file(): 
            script_dir = Path(__file__).parent.resolve()
            potential_path = script_dir / self.init_state
            if potential_path.is_file(): resolved_init_state = potential_path
            else:
                potential_path_cwd = Path.cwd() / self.init_state
                if potential_path_cwd.is_file(): resolved_init_state = potential_path_cwd
                else: raise FileNotFoundError(f"Initial game state file not found or not a file: {self.init_state}, {script_dir / self.init_state}, or {potential_path_cwd}")
        with open(resolved_init_state, "rb") as f:
            self.pyboy.load_state(f)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._get_initial_game_state()
        self.session = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))
        
        if self.full_frame_writer is not None:
            try: self.full_frame_writer.close()
            except Exception as e: print(f"DDEnv Warning: Error closing previous video writer: {e}")
            self.full_frame_writer = None
        if self.current_video_temp_file is not None:
            try: 
                self.current_video_temp_file.close() # This should delete the temp file if delete=True
                if Path(self.current_video_temp_file.name).exists(): # If delete=False, manually remove
                    os.remove(self.current_video_temp_file.name)
            except Exception as e: print(f"DDEnv Warning: Error closing/deleting previous temp video file: {e}")
            self.current_video_temp_file = None

        if self.save_video: 
            try:
                # delete=False so we can control deletion after upload
                self.current_video_temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) 
                temp_video_path = self.current_video_temp_file.name
                
                self.full_frame_writer = media.VideoWriter(
                    path=temp_video_path, shape=(144, 160), fps=60 
                )
                self.full_frame_writer.__enter__()
            except Exception as e:
                print(f"DDEnv Warning: Failed to start video recording. Error: {e}")
                self.save_video = False # Disable for this episode if setup fails
                if self.current_video_temp_file:
                    self.current_video_temp_file.close() # This will delete if delete=True
                    if Path(self.current_video_temp_file.name).exists() and not self.current_video_temp_file.delete:
                        os.remove(self.current_video_temp_file.name)
                    self.current_video_temp_file = None

        self.recent_frames = np.zeros(self.recent_frames_np_shape, dtype=np.uint8)
        self.recent_memory_flat = np.zeros((self.observation_space_shape[1] * self.memory_height, self.observation_space_shape[2]), dtype=np.uint8)

        self.step_count = 0
        self.kick_penalty = False
        self.last_score = self._get_current_score()
        self.last_level = self._get_current_level()
        self.last_lives = self._get_current_lives()
        self.old_x_pos = self._get_player_x_pos()
        self.old_y_pos = self._get_player_y_pos()
        
        self.locations = {i: False for i in range(1, 8)} 
        self.game_state_reward_components = self._get_current_game_state_potentials()
        self.previous_total_game_state_reward = sum(self.game_state_reward_components.values())
        self.unique_levels_completed_in_episode = 0 
        
        self.reset_count += 1
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _upload_file_to_s3(self, local_file_path: str, s3_object_key: str, content_type: Optional[str] = None):
        if not self.s3_client or not self.s3_bucket_name:
            print(f"DDEnv Warning: S3 client or bucket not configured. Cannot upload {local_file_path}.")
            return
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            print(f"DDEnv Info: Uploading {local_file_path} to S3 bucket '{self.s3_bucket_name}' as '{s3_object_key}'...")
            self.s3_client.upload_file(local_file_path, self.s3_bucket_name, s3_object_key, ExtraArgs=extra_args)
            print(f"DDEnv Info: File successfully uploaded to s3://{self.s3_bucket_name}/{s3_object_key}")
        except Exception as e:
            print(f"DDEnv Error: Failed to upload {local_file_path} to S3. Error: {e}")
        finally:
            if Path(local_file_path).exists():
                try:
                    os.remove(local_file_path)
                except Exception as e_del:
                    print(f"DDEnv Warning: Failed to delete temporary file {local_file_path}. Error: {e_del}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._run_action_on_emulator(action)
        obs = self._get_observation()
        reward = self._calculate_reward_from_potentials()
        
        self.step_count += 1
        terminated, truncated = self._check_episode_end()
        
        if self.save_video and self.full_frame_writer is not None:
            self.full_frame_writer.add_image(self.screen.screen_ndarray())
        
        if (terminated or truncated) and self.save_video and self.full_frame_writer is not None:
            temp_video_path_to_upload = None
            if self.current_video_temp_file:
                temp_video_path_to_upload = self.current_video_temp_file.name
            
            try:
                self.full_frame_writer.close()
            except Exception as e:
                print(f"DDEnv Warning: Failed to close video writer: {e}")
            self.full_frame_writer = None # Mark as closed
            
            if temp_video_path_to_upload and Path(temp_video_path_to_upload).exists() and self.s3_client:
                s3_object_key = f"{self.s3_video_prefix.strip('/')}/rollout_reset{self.reset_count-1}_session{self.session}_id{self.instance_id}_vid{str(uuid.uuid4())[:4]}.mp4"
                self._upload_file_to_s3(temp_video_path_to_upload, s3_object_key, content_type='video/mp4')
            
            if self.current_video_temp_file: 
                try: 
                    # If delete=False was used for NamedTemporaryFile, it's already removed by _upload_file_to_s3
                    # If delete=True, this close would delete it. Since we use delete=False and manual os.remove,
                    # just ensure the object is closed if not already.
                    if not self.current_video_temp_file.closed:
                         self.current_video_temp_file.close()
                except: pass
                self.current_video_temp_file = None

        if self.step_count % 100 == 0):
            print(f"DDEnv (Instance: {self.instance_id}, Session: {self.session}, Reset: {self.reset_count-1}), Step: {self.step_count}, Action: {action}, Reward: {reward:.4f}, Term: {terminated}, Trunc: {truncated}, Lives: {self.last_lives}, Score: {self.last_score}")

        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        game_frame = self.screen.screen_ndarray()
        h, w, c = self._single_frame_obs_shape
        processed_frame = (255 * resize(game_frame, (h, w, c))).astype(np.uint8)

        self.recent_frames = np.roll(self.recent_frames, shift=-1, axis=0)
        self.recent_frames[-1] = processed_frame

        exploration_mem = self._create_exploration_memory()
        recent_mem = self._create_recent_memory()
        pad = np.zeros((self.mem_padding, self.observation_space_shape[1], self.observation_space_shape[2]), dtype=np.uint8)
        screen_part = rearrange(self.recent_frames, 'f h w c -> (f h) w c')

        final_obs = np.concatenate([exploration_mem, pad, recent_mem, pad, screen_part], axis=0)
        return final_obs.astype(np.uint8)

    def _get_info(self) -> dict:
        return {"score": self.last_score, "lives": self.last_lives, "level": self.last_level, "steps": self.step_count, "instance_id": self.instance_id, "session": self.session, "levels_completed": self.unique_levels_completed_in_episode}

    def _check_episode_end(self) -> Tuple[bool, bool]:
        terminated = self._get_current_lives() == 0
        truncated = self.step_count >= self.max_steps
        return terminated, truncated

    def render(self) -> np.ndarray: 
        return self._get_observation() 

    def _run_action_on_emulator(self, action_idx: int):
        action_to_perform = self.valid_actions[action_idx]
        self.kick_penalty = False 

        if action_to_perform >= 97: 
            combo_map = {
                99: [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B], 
                97: [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B], 
                98: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B]
            }
            self._combo_action(combo_map[action_to_perform], is_kick=True)
        else: 
            self.pyboy.send_input(action_to_perform)
            for i in range(self.act_freq):
                if i == self.act_freq // 2:
                    if action_idx < 4: self.pyboy.send_input(self.release_arrow[action_idx])
                    elif 4 <= action_idx < 6: self.pyboy.send_input(self.release_button[action_idx - 4])
                self.pyboy.tick()

    def _combo_action(self, inputs: list, is_kick: bool = False):
        for event_press in inputs:
            self.pyboy.send_input(event_press)
        
        for _ in range(max(1, self.act_freq // 2)):
             self.pyboy.tick()
        
        release_map = {
            WindowEvent.PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
        }
        for event_press in reversed(inputs): 
            if event_press in release_map:
                 self.pyboy.send_input(release_map[event_press])
        
        for _ in range(max(1, self.act_freq - (self.act_freq // 2))):
            self.pyboy.tick()
        if is_kick: self.kick_penalty = True

    def _get_current_score(self) -> int:
        try:
            s_bytes = [self.pyboy.get_memory_value(addr) for addr in range(0xC640, 0xC645)] 
            s = 0
            for digit in s_bytes: s = s * 10 + digit
            return s
        except Exception: return getattr(self, 'last_score', 0)

    def _get_player_x_pos(self) -> int: 
        try: return self.pyboy.get_memory_value(0xE100) 
        except: return getattr(self, 'old_x_pos', 0)

    def _get_player_y_pos(self) -> int: 
        try: return self.pyboy.get_memory_value(0xE210) 
        except: return getattr(self, 'old_y_pos', 0)
        
    def _get_current_level(self) -> int:
        try: return self.pyboy.get_memory_value(0xE110) 
        except: return getattr(self, 'last_level', 0)

    def _get_current_lives(self) -> int:
        try: return self.pyboy.get_memory_value(0xC499) 
        except: return getattr(self, 'last_lives', 3)

    def _get_current_game_state_potentials(self) -> Dict[str, float]:
        score_potential = self._get_current_score() * 0.01 
        level_val = self._get_current_level()
        level_reward_map = {15: 0, 84: 500, 48: 600, 89: 700, 11: 800} 
        level_potential = float(level_reward_map.get(level_val, self.last_level * 10))
        lives_potential = self._get_current_lives() * 150.0 
        return {'score': score_potential, 'level': level_potential, 'lives': lives_potential}

    def _calculate_reward_from_potentials(self) -> float:
        current_potentials = self._get_current_game_state_potentials()
        score_reward = current_potentials['score'] - self.game_state_reward_components.get('score', current_potentials['score'])
        level_reward = 0.0 
        current_level_val = self._get_current_level()
        if current_level_val != self.last_level: 
             level_reward_map_direct = {15: 0, 84: 50, 48: 60, 89: 70, 11: 80} 
             if current_level_val in level_reward_map_direct and not self.locations.get(current_level_val, False):
                 level_reward = float(level_reward_map_direct[current_level_val])
                 self.locations[current_level_val] = True 
                 self.unique_levels_completed_in_episode += 1 
        
        lives_reward_penalty = 0.0
        if self._get_current_lives() < self.last_lives: 
            lives_reward_penalty = -100.0 

        self.game_state_reward_components = current_potentials
        self.last_score = self._get_current_score()
        self.last_lives = self._get_current_lives()
        self.last_level = current_level_val 

        pos_change_reward = 0.0
        current_x = self._get_player_x_pos()
        current_y = self._get_player_y_pos()
        if current_x != self.old_x_pos or current_y != self.old_y_pos:
            pos_change_reward = 0.05 
        self.old_x_pos = current_x
        self.old_y_pos = current_y

        kick_penalty_val = -1.0 if self.kick_penalty else 0.0
        if self.kick_penalty: self.kick_penalty = False

        total_step_reward = score_reward + level_reward + lives_reward_penalty + pos_change_reward + kick_penalty_val
        return total_step_reward * 0.1

    def _create_recent_memory(self) -> np.ndarray:
        obs_width = self.observation_space_shape[1]
        obs_channels = self.observation_space_shape[2]
        return np.zeros((self.memory_height, obs_width, obs_channels), dtype=np.uint8)

    def _create_exploration_memory(self) -> np.ndarray:
        def make_channel_original(val_to_encode):
            obs_width = self.observation_space_shape[1]
            current_col_steps = max(1, self.col_steps) 
            max_encodable_val = (obs_width - 1) * self.memory_height * current_col_steps
            val = max(0, min(val_to_encode, max_encodable_val if max_encodable_val > 0 else val_to_encode ))

            memory_channel = np.zeros((self.memory_height, obs_width), dtype=np.uint8)
            if current_col_steps == 0: return memory_channel

            full_cols_to_light = floor(val / (self.memory_height * current_col_steps))
            full_cols_to_light = min(full_cols_to_light, obs_width -1) 

            if full_cols_to_light >= 0 :
                 memory_channel[:, :int(full_cols_to_light)] = 255
            
            if full_cols_to_light < obs_width : 
                val_in_full_cols = full_cols_to_light * self.memory_height * current_col_steps
                remaining_val = val - val_in_full_cols
                
                cells_in_current_col = floor(remaining_val / current_col_steps)
                cells_in_current_col = min(cells_in_current_col, self.memory_height -1) 

                if cells_in_current_col >=0:
                    memory_channel[:int(cells_in_current_col), int(full_cols_to_light)] = 255
                
                if cells_in_current_col < self.memory_height: 
                    last_val_in_cell = remaining_val - cells_in_current_col * current_col_steps
                    intensity_step = (255 // current_col_steps)
                    if cells_in_current_col >=0: 
                         memory_channel[int(cells_in_current_col), int(full_cols_to_light)] = last_val_in_cell * intensity_step
            return memory_channel

        score_val = self._get_current_score() 
        level_val = self._get_current_level() 
        lives_val = self._get_current_lives() * 500 

        channel1 = make_channel_original(score_val / 100 if score_val > 0 else 0) 
        channel2 = make_channel_original(level_val * 100 if level_val > 0 else 0) 
        channel3 = make_channel_original(lives_val if lives_val > 0 else 0)      

        return np.stack([channel1, channel2, channel3], axis=-1).astype(np.uint8)

    def close(self):
        """Closes the environment, saves final state and video if configured, and stops PyBoy."""
        print(f"DDEnv Closing: Instance {self.instance_id}, Session {self.session}, Reset Count: {self.reset_count-1}")
        final_score = self._get_current_score() # Get score one last time
        final_lives = self._get_current_lives() # Get lives one last time
        print(f"DDEnv Final Stats: Score = {final_score}, Lives = {final_lives}, Unique Levels Completed in Episode = {self.unique_levels_completed_in_episode}")

        # Save final video frame if episode was ongoing and writer exists
        if self.save_video and self.full_frame_writer is not None and not self.full_frame_writer.closed:
            try:
                self.full_frame_writer.add_image(self.screen.screen_ndarray()) # Add one last frame
                print("DDEnv Info: Added final frame to video before closing.")
            except Exception as e:
                print(f"DDEnv Warning: Could not add final frame to video: {e}")
        
        # Close video writer and upload
        if self.full_frame_writer is not None:
            temp_video_path_to_upload = None
            if self.current_video_temp_file:
                temp_video_path_to_upload = self.current_video_temp_file.name
            
            try: 
                if not self.full_frame_writer.closed: # mediapy doesn't have .closed, check if writer is not None
                    self.full_frame_writer.close()
            except Exception as e: print(f"DDEnv Warning: Error closing video writer during env close: {e}")
            self.full_frame_writer = None

            if temp_video_path_to_upload and Path(temp_video_path_to_upload).exists() and self.s3_client:
                s3_object_key = f"{self.s3_video_prefix.strip('/')}/final_video_reset{self.reset_count-1}_session{self.session}_id{self.instance_id}_vid{str(uuid.uuid4())[:4]}.mp4"
                self._upload_file_to_s3(temp_video_path_to_upload, s3_object_key, content_type='video/mp4')
            
            if self.current_video_temp_file: # Ensure temp file object is closed and thus deleted
                try: 
                    if not self.current_video_temp_file.closed:
                        self.current_video_temp_file.close() 
                    # If delete=False, os.remove is handled by _upload_file_to_s3
                except: pass
                self.current_video_temp_file = None
        
        # Save final game state if configured
        if self.save_final_state and hasattr(self, 'pyboy') and self.pyboy is not None:
            state_temp_file = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as state_temp_file_obj:
                    state_temp_file = state_temp_file_obj.name
                    self.pyboy.save_state(state_temp_file_obj) # PyBoy saves to a file-like object
                
                print(f"DDEnv Info: Final game state saved to temporary file: {state_temp_file}")
                
                if self.s3_client and self.s3_bucket_name and state_temp_file:
                    s3_state_key = f"{self.s3_state_prefix.strip('/')}/final_state_reset{self.reset_count-1}_session{self.session}_id{self.instance_id}.state"
                    self._upload_file_to_s3(state_temp_file, s3_state_key, content_type='application/octet-stream')
                elif not self.s3_client: # Fallback to local save if S3 not configured but save_final_state is true
                    local_state_path = Path.cwd() / f"final_state_{self.instance_id}_reset{self.reset_count-1}.state"
                    Path(state_temp_file).rename(local_state_path)
                    print(f"DDEnv Info: S3 not configured. Final game state saved locally to: {local_state_path}")


            except Exception as e:
                print(f"DDEnv Warning: Failed to save final game state. Error: {e}")
            finally:
                # Ensure temp file is removed if it still exists and wasn't handled by _upload_file_to_s3
                if state_temp_file and Path(state_temp_file).exists():
                    try: os.remove(state_temp_file)
                    except: pass
        
        if hasattr(self, 'pyboy') and self.pyboy is not None:
            self.pyboy.stop() # PyBoy stop() method does not take arguments
            print(f"PyBoy instance for DDEnv {self.instance_id} stopped.")
