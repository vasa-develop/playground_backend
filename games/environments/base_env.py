from abc import ABC, abstractmethod
import numpy as np

class GameEnvironment(ABC):
    """Base class for all game environments."""

    @abstractmethod
    def reset(self):
        """Reset the environment to initial state."""
        pass

    @abstractmethod
    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        pass

    @abstractmethod
    def get_state(self):
        """Get current state of the environment."""
        pass

    @abstractmethod
    def get_valid_actions(self):
        """Get list of valid actions."""
        pass

    @abstractmethod
    def render(self):
        """Render the current state."""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
