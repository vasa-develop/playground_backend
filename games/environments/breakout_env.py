import gymnasium as gym
import numpy as np
import cv2
from .base_env import GameEnvironment

class BreakoutEnvironment(GameEnvironment):
    """Atari Breakout environment wrapper for MuZero."""

    def __init__(self):
        """Initialize Breakout environment."""
        self.env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state = None
        self.lives = 5  # Initial lives in Breakout
        self.frame_stack = []  # Store last 4 frames for MuZero
        self.frame_stack_size = 4

    def reset(self):
        """Reset the environment and return initial state."""
        self.state, info = self.env.reset()
        self.lives = info.get('lives', 5)
        self.frame_stack = []

        # Initialize frame stack with initial state
        processed_frame = self._preprocess_frame(self.state)
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(processed_frame)

        return self._get_stacked_state()

    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Handle life loss as a terminal state for training
        current_lives = info.get('lives', self.lives)
        life_lost = current_lives < self.lives
        self.lives = current_lives

        # Update state and frame stack
        self.state = next_state
        processed_frame = self._preprocess_frame(next_state)
        self.frame_stack.pop(0)
        self.frame_stack.append(processed_frame)

        return (
            self._get_stacked_state(),
            float(reward),
            done or life_lost,
            info
        )

    def get_state(self):
        """Get current stacked state for MuZero."""
        return self._get_stacked_state() if self.state is not None else None

    def get_valid_actions(self):
        """Get list of valid actions."""
        return list(range(self.action_space.n))

    def render(self):
        """Render the current state."""
        return self.env.render()

    def close(self):
        """Clean up resources."""
        self.env.close()

    def _preprocess_frame(self, frame):
        """Preprocess a single frame for MuZero."""
        if frame is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize to 84x84 (standard for DQN/MuZero Atari)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        normalized = resized / 255.0

        return normalized.astype(np.float32)

    def _get_stacked_state(self):
        """Stack last 4 frames for MuZero input."""
        if not self.frame_stack:
            return None

        # Stack frames along a new axis (channel dimension)
        return np.stack(self.frame_stack, axis=0)

    def get_observation_space(self):
        """Get the observation space for MuZero."""
        return (self.frame_stack_size, 84, 84)  # (channels, height, width)

    def get_action_space(self):
        """Get the action space size for MuZero."""
        return self.action_space.n
