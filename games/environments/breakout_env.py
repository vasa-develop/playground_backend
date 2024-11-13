import gymnasium as gym
import numpy as np
from .base_env import GameEnvironment

class BreakoutEnvironment(GameEnvironment):
    """Atari Breakout environment wrapper."""

    def __init__(self):
        """Initialize Breakout environment."""
        self.env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.state = None
        self.lives = 5  # Initial lives in Breakout

    def reset(self):
        """Reset the environment."""
        self.state, info = self.env.reset()
        self.lives = info.get('lives', 5)
        return self._preprocess_state(self.state)

    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Handle life loss as a terminal state for training
        current_lives = info.get('lives', self.lives)
        life_lost = current_lives < self.lives
        self.lives = current_lives

        # Update state
        self.state = next_state

        return (
            self._preprocess_state(next_state),
            float(reward),
            done or life_lost,
            info
        )

    def get_state(self):
        """Get current preprocessed state."""
        return self._preprocess_state(self.state) if self.state is not None else None

    def get_valid_actions(self):
        """Get list of valid actions."""
        return list(range(self.action_space.n))

    def render(self):
        """Render the current state."""
        return self.env.render()

    def close(self):
        """Clean up resources."""
        self.env.close()

    def _preprocess_state(self, state):
        """Preprocess state for MuZero."""
        # Convert to grayscale and normalize
        if state is None:
            return None

        # Convert RGB to grayscale using weighted sum
        grayscale = np.dot(state[..., :3], [0.299, 0.587, 0.114])

        # Normalize to [0, 1]
        normalized = grayscale / 255.0

        # Resize if needed (84x84 is common for Atari)
        # Note: Add resize logic here if needed

        return normalized.astype(np.float32)
