import random
import gym
import numpy as np
from process import process_frame


class GameWrapper:
    def __init__(self, env_name, no_op_steps=10, history_length=4):
        self.env = gym.make(env_name)
        self.no_op_steps = no_op_steps
        self.history_length = history_length

        self.state = None
        self.last_lives = 0

    def reset(self, evaluation=False):
        self.frame = self.env.reset()
        self.last_lives = 0

        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

        # For the initial state we stack the first frame 4 times
        self.state = np.repeat(process_frame(self.frame), self.history_length, axis=2)
        return self.frame, process_frame(self.frame)

    def step(self, action, render_mode=None, gradcam=False):
        new_frame, reward, terminal, info = self.env.step(action)

        if info['ale.lives'] < self.last_lives:
            life_lost = True
        else:
            life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_frame = process_frame(new_frame)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=2)

        if render_mode == 'rgb_array':
            return processed_frame, reward, terminal, life_lost, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render()
        if gradcam:
            return new_frame, processed_frame, reward, terminal, life_lost

        return processed_frame, reward, terminal, life_lost
