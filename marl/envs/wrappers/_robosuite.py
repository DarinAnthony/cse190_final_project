from robosuite.wrappers import Wrapper
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Callable
import os
import imageio

class RecordVideoWrapper(gym.Wrapper):
    """
    Records 200‐frame video snippets from each offscreen camera every time
    step_trigger(global_step) returns True.  After collecting 200 frames,
    the wrapper auto‐saves one .mp4 per camera and clears its buffers until
    the next trigger.

    Args:
        env (gym.Env): Any Gym/robosuite env with .camera_names and obs["<cam>_image"].
        video_folder (str): Directory where snippet files will be written.
        step_trigger (Callable[[int], bool]): Given a global step count (0,1,2,...),
                   returns True exactly on the steps where you want to start a new 200‐frame snippet.
        fps (int): Frames per second for the output videos (default=20).
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        step_trigger: Callable[[int], bool],
        fps: int = 20,
    ):
        super().__init__(env)

        # Ensure the wrapped env has camera_names
        if not hasattr(self.env.unwrapped, "camera_names"):
            raise AttributeError(
                "Wrapped env must have `camera_names` (list of strings)."
            )
        self.camera_names = list(self.env.unwrapped.camera_names)
        print(f'Recording videos with {self.camera_names} cameras')

        # Where to write the snippet files
        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        self.step_trigger = step_trigger
        self.fps = fps

        # Global step counter across all episodes
        self.global_step = -1

        # Are we currently recording a snippet? If True, we append frames until we hit 200.
        self.recording = False

        # How many frames have we collected in the current snippet?
        self._frame_count = 0
        self._max_frames = 200  # snippet length

        # Buffers: one list of frames per camera
        self.frames = {cam: [] for cam in self.camera_names}

        # How many snippets have we already saved?  Used to increment filenames.
        self._snippet_counter = 0

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.  Also clears any partial snippet (if in progress)
        so that each episode always begins with recording=False and 0 frames collected.
        We do NOT reset global_step, so step_trigger continues counting across episodes.
        """
        obs, info = super().reset(seed=seed, options=options)
        # Abort any ongoing snippet and clear buffers
        self.recording = False
        self._frame_count = 0
        self.frames = {cam: [] for cam in self.camera_names}
        return obs, info

    def step(self, action):
        """
        Step through the env, bump global_step, check step_trigger:
          • If step_trigger(global_step) is True and we are NOT already recording,
            we start a new 200‐frame snippet (recording=True, _frame_count=0, buffers cleared).
          • If recording=True, we append obs["<cam>_image"] for each camera. 
          • Once _frame_count hits 200, we auto‐save all camera buffers to disk
            (one .mp4 per camera, named <camera>_snippet_<idx>.mp4), clear buffers,
            increment _snippet_counter, and set recording=False until the next trigger.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        self.global_step += 1

        # If the trigger fires right now and we're not already collecting a snippet, start recording
        if not self.recording and self.step_trigger(self.global_step):
            self.recording = True
            self._frame_count = 0
            self.frames = {cam: [] for cam in self.camera_names}

        # If we're in the middle of a snippet, collect this frame
        if self.recording:
            for cam in self.camera_names:
                key = f"{cam}_image"
                if key not in obs:
                    raise KeyError(f"Expected obs to contain '{key}' but it's missing.")
                self.frames[cam].append(obs[key])
            self._frame_count += 1

            # If we've collected 200 frames, save and reset
            if self._frame_count >= self._max_frames:
                self._save_current_snippet()
                self.recording = False
                self._frame_count = 0
                # Leave self.frames cleared for the next snippet
                self.frames = {cam: [] for cam in self.camera_names}

        return obs, reward, terminated, truncated, info

    def _save_current_snippet(self):
        """
        Save the buffered frames (exactly 200 for each camera) as
        <video_folder>/<camera>_snippet_<snippet_idx>.mp4.  Uses imageio.
        Rotates frames 180 degrees for 'frontview' camera only.
        """
        snippet_idx = self._snippet_counter
        for cam, buffer in self.frames.items():
            if len(buffer) == 0:
                continue
            filename = os.path.join(
                self.video_folder, f"{cam}_snippet_{snippet_idx}.mp4"
            )
            writer = imageio.get_writer(filename, fps=self.fps)
            try:
                for frame in buffer:
                    # Only rotate if it's the frontview camera
                    if cam == 'frontview': #robosuite camera is flipped 180 degrees for some reason
                        frame = np.rot90(frame, k=2)
                    writer.append_data(frame)
            finally:
                writer.close()
        print(f"Saved snippet {snippet_idx} (200 frames) for cameras: {self.camera_names}")
        self._snippet_counter += 1

class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.
        flatten_obs (bool):
            Whether to flatten the observation dictionary into a 1d array. Defaults to True.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None, flatten_obs=True):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for key in list(self.env.reset().keys()):
                if key not in keys:
                    keys += [key]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()

        # Whether to flatten the observation space
        self.flatten_obs: bool = flatten_obs

        if self.flatten_obs:
            flat_ob = self._flatten_obs(obs)
            self.obs_dim = flat_ob.size
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        else:

            def get_box_space(sample):
                """Util fn to obtain the space of a single numpy sample data"""
                if np.issubdtype(sample.dtype, np.bool_):
                    return spaces.Box(low=0, high=1, shape=sample.shape, dtype=np.bool_)
                elif np.issubdtype(sample.dtype, np.integer):
                    low = np.iinfo(sample.dtype).min
                    high = np.iinfo(sample.dtype).max
                elif np.issubdtype(sample.dtype, np.inexact):
                    low = float("-inf")
                    high = float("inf")
                else:
                    raise ValueError()
                return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)

            self.observation_space = spaces.Dict({key: get_box_space(obs[key]) for key in self.keys})

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def _filter_obs(self, obs_dict) -> dict:
        """
        Filters keys of interest out of the observation dictionary, returning a filterd dictionary.
        """
        return {key: obs_dict[key] for key in self.keys if key in obs_dict}

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return observation instead of normal OrderedDict and optionally resets seed

        Returns:
            2-tuple:
                - (np.array) observations from the environment
                - (dict) an empty dictionary, as part of the standard return format
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return obs, {}

    def step(self, action):
        """
        Extends vanilla step() function call to return observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return obs, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def close(self):
        """
        wrapper for calling underlying env close function
        """
        self.env.close()
