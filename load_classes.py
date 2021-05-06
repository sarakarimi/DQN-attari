import logging
from agent import Agent
from gamewrapper import GameWrapper
from config import *
from prioretized_buffer import ReplayBuffer
from q_network import build_q_network
import tensorflow as tf

logger = logging.getLogger('atari')

game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)
print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n,
                                                                game_wrapper.env.unwrapped.get_action_meanings()))
writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

MAIN_DQN = build_q_network(game_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = build_q_network(game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE,
              batch_size=BATCH_SIZE, use_per=USE_PER)

if LOAD_FROM is None:
    frame_number = 0
    rewards = []
    loss_list = []
else:
    logger.info('Loading from', LOAD_FROM)
    meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)
    frame_number = meta['frame_number']
    rewards = meta['rewards']
    loss_list = meta['loss_list']
    logger.info('Loaded!')
