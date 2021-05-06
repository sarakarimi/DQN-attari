import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

import time
import numpy as np
from load_classes import *
import logging


logger = logging.getLogger('atari')
logger.setLevel(logging.DEBUG)
hdlr = logging.FileHandler('logs/breakout.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)


# Main loop
# @tf.function
def main_loop(frame_number):
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                epoch_frame = 0
                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    game_wrapper.reset()
                    life_lost = True
                    episode_reward_sum = 0
                    gradient = None
                    for _ in range(MAX_EPISODE_LENGTH):
                        action = agent.get_action(frame_number, game_wrapper.state)

                        processed_frame, reward, terminal, life_lost = game_wrapper.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward

                        agent.add_experience(action=action,
                                             frame=processed_frame[:, :, 0],
                                             reward=reward, clip_reward=CLIP_REWARD,
                                             terminal=life_lost)

                        # Update agent
                        if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                            loss, _, gradient = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number,
                                                  priority_scale=PRIORITY_SCALE)

                            tf.summary.scalar('Gradient', gradient, frame_number)
                            loss_list.append(loss)

                        # Update target network
                        if frame_number % TARGET_UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                            agent.update_target_network()

                        if terminal:
                            terminal = False
                            break

                    rewards.append(episode_reward_sum)

                    if len(rewards) % 10 == 0:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                            tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)

                            writer.flush()

                        logger.info(
                            f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0

                for _ in range(EVAL_LENGTH):
                    if terminal:
                        game_wrapper.reset(evaluation=True)
                        life_lost = True
                        episode_reward_sum = 0
                        terminal = False

                    action = 1 if life_lost else agent.get_action(frame_number, game_wrapper.state, evaluation=True)
                    _, reward, terminal, life_lost = game_wrapper.step(action)
                    evaluate_frame_number += 1
                    episode_reward_sum += reward

                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    final_score = episode_reward_sum

                logger.info('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

                if len(rewards) > 300 and SAVE_PATH is not None:
                    agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards,
                               loss_list=loss_list)
    except KeyboardInterrupt:
        logger.info('\nTraining exited early.')
        writer.close()
        logger.info('Saving...')
        agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
        logger.info('Saved.')


main_loop(frame_number)