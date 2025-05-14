import random
import string
import sys
import os
from math import floor
import uuid
import numpy as np
from einops import rearrange
from skimage.transform import resize
from pathlib import Path
import mediapy as media
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from gymnasium import Env, spaces
from typing import Optional

# Set up relative path
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, file_path + "/../..")


class DDEnv(Env):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}
        default_config = {
            'headless': True, 'save_final_state': False, 'early_stop': False,
            'action_freq': 8, 'init_state': 'ignored/dd.gb.state',
            'max_steps': 2048 * 30 * 12 * 1000, 'print_rewards': True,
            'save_video': False, 'fast_video': False,
            'session_path': Path(f'session_{str(uuid.uuid4())[:8]}'),
            'gb_path': 'ignored/dd.gb', 'debug': False,
            'sim_frame_dist': 2_000_000.0, 'use_screen_explore': True,
            'extra_buttons': False
        }
        
        for key, val in default_config.items():
            config.setdefault(key, val)

        self.debug = config['debug']
        self.s_path = Path(config['session_path'])
        self.gb_path = config['gb_path']
        self.save_final_state = config['save_final_state']
        self.print_rewards = config['print_rewards']
        self.headless = config['headless']
        self.init_state = config['init_state']
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        self.early_stopping = config['early_stop']
        self.save_video = config['save_video']
        self.fast_video = config['fast_video']
        self.similar_frame_dist = config['sim_frame_dist']
        self.use_screen_explore = config['use_screen_explore']
        self.frame_stacks = 3
        self.downsample_factor = 2
        self.reset_count = 0
        self.instance_id = str(uuid.uuid4())[:8]
        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.session = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))


        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2]
        )

        Path(self.s_path).mkdir(parents=True, exist_ok=True)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            97,  # jump left
            98,  # jump right
            99   # jump kick
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'headless' if self.headless else 'SDL2'
        self.pyboy = PyBoy(
            self.gb_path,
            debugging=True,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv
        )

        self.screen = self.pyboy.botsupport_manager().screen()

    def reset(self, *, seed=None, options=None):
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        
        self.session = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(4))

        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.session}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()

        self.recent_frames = np.zeros((self.frame_stacks, *self.output_shape), dtype=np.uint8)
        self.recent_memory = np.zeros((self.output_shape[1] * self.memory_height, 3), dtype=np.uint8)

        self.old_x_pos = []
        self.old_y_pos = []
        self.step_count = 0
        self.kick_penalty = False
        self.last_score = 0
        self.last_level = 0
        self.last_lives = 3
        self.total_reward = 0
        self.total_score_rew = 0
        self.levels = 0
        self.total_lives_rew = 3
        self.locations = {i: False for i in range(1, 8)}
        self.progress_reward = self.get_game_state_reward()

        return self.render(), {}

    def step(self, action):
        self.run_action_on_emulator(action)
        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs = self.render()

        self.step_count += 1
        reward, _ = self.update_reward()

        done = self.check_if_done()
        if done and self.save_video:
            self.full_frame_writer.close()

        return obs, reward * 0.1, False, done, {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        frame = self.screen.screen_ndarray()
        if reduce_res:
            frame = (255 * resize(frame, self.output_shape)).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = frame
            if add_memory:
                pad = np.zeros((self.mem_padding, self.output_shape[1], 3), dtype=np.uint8)
                frame = np.concatenate([
                    self.create_exploration_memory(),
                    pad,
                    self.create_recent_memory(),
                    pad,
                    rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                ], axis=0)
        return frame

    def run_action_on_emulator(self, action):
        act = self.valid_actions[action]
        if act == 99:
            self.combo_action([WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B])
        elif act == 97:
            self.combo_action([WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B])
        elif act == 98:
            self.combo_action([WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B])
        else:
            self.pyboy.send_input(act)
            for i in range(self.act_freq):
                if i == 4:
                    if action < 4:
                        self.pyboy.send_input(self.release_arrow[action])
                    elif 4 <= action < 6:
                        self.pyboy.send_input(self.release_button[action - 4])
                if self.save_video and not self.fast_video:
                    self.add_video_frame()
                self.pyboy.tick()
            if self.save_video and self.fast_video:
                self.add_video_frame()

    def combo_action(self, inputs):
        for i in inputs:
            self.pyboy.send_input(i)
        self.pyboy.tick()
        for i in inputs:
            self.pyboy.send_input(i + 1)  # Release versions
        self.kick_penalty = True

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False))

    def create_recent_memory(self):
        return rearrange(self.recent_memory, '(w h) c -> h w c', h=self.memory_height)

    def create_exploration_memory(self):
        def make_channel(val):
            val = max(0, min(val, (self.output_shape[1] - 1) * self.memory_height * self.col_steps))
            row = floor(val / (self.memory_height * self.col_steps))
            memory = np.zeros((self.memory_height, self.output_shape[1]), dtype=np.uint8)
            memory[:, :row] = 255
            r_covered = row * self.memory_height * self.col_steps
            col = floor((val - r_covered) / self.col_steps)
            memory[:col, row] = 255
            last = floor(val - r_covered - col * self.col_steps)
            memory[col, row] = last * (255 // self.col_steps)
            return memory

        score, pos, level, lives = self.group_rewards()
        return np.stack([make_channel(level), make_channel(pos), make_channel(pos)], axis=-1)

    def group_rewards(self):
        r = self.progress_reward
        return (r['score'], r['pos'], r['level'], r['lives'])

    def get_game_state_reward(self):
        return {
            'score': self.get_score_reward() // 10,
            'pos': self.get_position_reward(),
            'level': self.get_level_reward(),
            'lives': self.get_lives_reward() * 15,
            'moves': self.get_moves_penalty()
        }

    def update_reward(self):
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum(self.progress_reward.values())
        delta = new_total - self.total_reward
        self.total_reward = new_total
        return delta, new_prog

    def check_if_done(self):
        return self.step_count >= self.max_steps or self.total_lives_rew == 0

    def get_score(self):
        vals = [PyBoy.get_memory_value(self.pyboy, addr) for addr in range(0xC640, 0xC646)]
        return int("".join(map(str, vals)))

    def get_score_reward(self):
        new_score = self.get_score()
        delta = new_score - self.last_score
        if delta:
            self.last_score = new_score
            self.total_score_rew += delta
        return delta

    def get_position_reward(self):
        x = [PyBoy.get_memory_value(self.pyboy, a) for a in range(0xE100, 0xE110)]
        y = [PyBoy.get_memory_value(self.pyboy, a) for a in range(0xE210, 0xE220)]
        reward = 0.5 if x != getattr(self, "old_x_pos", []) or y != getattr(self, "old_y_pos", []) else -0.5
        self.old_x_pos = x
        self.old_y_pos = y
        return reward

    def get_level(self):
        return PyBoy.get_memory_value(self.pyboy, 0xE110)

    def get_level_reward(self):
        level = self.get_level()
        reward_map = {15: 0, 84: 50, 48: 60, 89: 70, 11: 80}

        if level in reward_map:
            if level not in self.locations:
                self.locations[level] = False  # Initialize if missing

            if not self.locations[level]:
                self.locations[level] = True
                self.levels += 1
                self.last_level = level
                return reward_map[level]
        return 0

    def get_lives(self):
        return PyBoy.get_memory_value(self.pyboy, 0xC499)

    def get_lives_reward(self):
        lives = self.get_lives()
        delta = lives - self.last_lives
        self.last_lives = lives
        self.total_lives_rew = lives
        return -10 if lives == 0 else delta

    def get_moves_penalty(self):
        if self.kick_penalty:
            self.kick_penalty = False
            return -10
        return 0
