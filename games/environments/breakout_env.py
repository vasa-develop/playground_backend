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
        self.ball_previous_y = None  # Track ball movement

    def reset(self):
        """Reset the environment and return initial state."""
        self.state, info = self.env.reset()
        self.lives = info.get('lives', 5)
        self.frame_stack = []
        self.ball_previous_y = None

        # Initialize frame stack with initial state
        processed_frame = self._preprocess_frame(self.state)
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(processed_frame)

        # Initialize info with required fields
        info = {
            'lives': int(self.lives),  # Convert to native Python int
            'episode_frame_number': 0,
            'frame_number': 0,
            'score': 0
        }

        # Get stacked state and ensure it's serializable
        stacked_state = self._get_stacked_state()
        if isinstance(stacked_state, np.ndarray):
            stacked_state = stacked_state.tolist()

        return stacked_state, info

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

        # Update info with frame numbers and ensure all values are native Python types
        info = {
            'lives': int(current_lives),
            'episode_frame_number': int(info.get('episode_frame_number', 0) + 1),
            'frame_number': int(info.get('frame_number', 0) + 1),
            'score': int(info.get('score', 0)),
            'termination_reason': 'life_lost' if life_lost else ('game_over' if terminated else None)
        }

        # Get stacked state and ensure it's serializable
        stacked_state = self._get_stacked_state()
        if isinstance(stacked_state, np.ndarray):
            stacked_state = stacked_state.tolist()

        return (
            stacked_state,
            float(reward),
            bool(done or life_lost),
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

        # Normalize to [0, 1] and convert to Python list
        normalized = (resized / 255.0).astype(np.float32)
        return normalized.tolist()  # Convert to Python list immediately after preprocessing

    def _get_stacked_state(self):
        """Stack last 4 frames for MuZero input."""
        if not self.frame_stack:
            return None

        # Since frames are already lists, we just need to return the frame stack
        return self.frame_stack  # Already a list of lists from _preprocess_frame

    def get_observation_space(self):
        """Get the observation space for MuZero."""
        return (self.frame_stack_size, 84, 84)  # (channels, height, width)

    def get_action_space(self):
        """Get the action space size for MuZero."""
        return self.action_space.n

    def get_ai_suggestion(self, state):
        """Get AI suggestion for the current state."""
        if state is None:
            return 1  # FIRE to start the game

        # Convert state back to numpy array for processing
        state_array = np.array(state)

        # Extract the most recent frame
        current_frame = state_array[-1]  # Shape: (84, 84)

        # Find paddle position (look at the bottom rows)
        paddle_row = current_frame[-5:]  # Last 5 rows
        paddle_col = np.argmax(np.mean(paddle_row, axis=0))

        # Find ball position (look for bright pixels in the play area)
        play_area = current_frame[10:-5]  # Exclude score area and paddle
        ball_positions = np.where(play_area > 0.5)

        if len(ball_positions[0]) > 0:
            ball_y = float(np.mean(ball_positions[0]))  # Convert to float
            ball_x = float(np.mean(ball_positions[1]))  # Convert to float

            # Initialize ball tracking if needed
            if not hasattr(self, 'ball_previous_y'):
                self.ball_previous_y = ball_y
                ball_moving_down = True
            else:
                # Only compare if we have a previous position
                ball_moving_down = True if self.ball_previous_y is None else ball_y > self.ball_previous_y

            # Update previous position
            self.ball_previous_y = ball_y

            # Only move if ball is moving down and getting close
            if ball_moving_down and ball_y > 40:  # Ball in lower half
                if ball_x < paddle_col - 2:
                    return 3  # LEFT
                elif ball_x > paddle_col + 2:
                    return 2  # RIGHT
                return 0  # NOOP

        # If we can't find the ball or it's not moving down
        self.ball_previous_y = None  # Reset tracking when ball is lost
        return np.random.choice([0, 2, 3])  # NOOP, RIGHT, LEFT
