import gym
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer

class ZarrRecorder:
    def start(self):
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
    
    def add_state(self, state):
        self.state_buffer.append(state)

    def add_action(self, action):
        self.action_buffer.append(action)

    def add_reward(self, reward):
        self.reward_buffer.append(reward)
    
    def finish(self, file_path):
        buffer = ReplayBuffer.create_empty_numpy()
        assert len(self.state_buffer) - 1 == len(self.action_buffer) == len(self.reward_buffer)
        buffer.add_episode({
            'state': np.array(self.state_buffer[:-1]),
            'action': np.array(self.action_buffer),
            'reward': np.array(self.reward_buffer),
        })
        buffer.save_to_path(file_path, chunk_length=-1)
        self.start()


class ZarrRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            mode='state_action_reward',
            file_path=None,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.file_path = file_path

        self.recorder = ZarrRecorder()

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self.file_path:
            self.recorder.start()
            self.recorder.add_state(obs)
        return obs
    
    def step(self, action):
        result = super().step(action)
        obs, rew, done, info = result
        if self.file_path:
            self.recorder.add_action(action)
            self.recorder.add_state(obs)
            self.recorder.add_reward(rew)
            if done:
                self.recorder.finish(self.file_path)
        return result