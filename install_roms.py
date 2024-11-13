import os
import ale_py
from ale_py import ALEInterface
from importlib.resources import files

# Create ROMs directory if it doesn't exist
roms_path = os.path.expanduser('~/.local/roms')
os.makedirs(roms_path, exist_ok=True)

# Install ROMs
ale = ALEInterface()
roms = files('ale_py').joinpath('roms')
for rom in roms.iterdir():
    if rom.name.endswith('.bin'):
        target_path = os.path.join(roms_path, rom.name)
        if not os.path.exists(target_path):
            with open(target_path, 'wb') as f:
                f.write(rom.read_bytes())
            print(f'Installed ROM: {rom.name}')

print('\nROMs installation completed')

# Test environment creation
import gymnasium as gym
print('\nTesting Breakout environment...')
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
print('Environment created successfully')
print('Action space:', env.action_space)
print('Observation space:', env.observation_space)

state, info = env.reset()
print('\nInitial state shape:', state.shape)
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)
print('Next state shape:', next_state.shape)
print('Reward:', reward)
print('Info:', info)

env.close()
print('\nBasic functionality test completed')
