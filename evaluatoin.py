from agent import Agent
from gamewrapper import GameWrapper
from prioretized_buffer import ReplayBuffer
from q_network import build_q_network
from config import *
import numpy as np


ENV_NAME = 'BreakoutDeterministic-v4'
game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))

MAIN_DQN = build_q_network(game_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = build_q_network(game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

print('Loading model...')
agent.load('breakout-saves/save-00054572/', load_replay_buffer=False)
print('Loaded')

terminal = True
eval_rewards = []
evaluate_frame_number = 0

for frame in range(EVAL_LENGTH):
    if terminal:
        game_wrapper.reset(evaluation=True)
        life_lost = True
        episode_reward_sum = 0
        terminal = False

    action = 1 if life_lost else agent.get_action(0, game_wrapper.state, evaluation=True)

    _, reward, terminal, life_lost = game_wrapper.step(action, render_mode='human')
    evaluate_frame_number += 1
    episode_reward_sum += reward

    if terminal:
        print(f'Game over, reward: {episode_reward_sum}, frame: {frame}/{EVAL_LENGTH}')
        eval_rewards.append(episode_reward_sum)

print('Average reward:', np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum)
