import numpy as np
import cv2
from PIL import Image
from agent import Agent
from config import *
from gamewrapper import GameWrapper
from prioretized_buffer import ReplayBuffer
from q_network import build_q_network
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.python.keras.backend as K
from gradcam_config import *
import os


def build_guided_model(action_space_n):
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)
    g = K.get_session().graph
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        return build_q_network(action_space_n)


def guided_backpropog(model, frame):
    grad_model = tf.keras.models.Model(inputs=[model.input],
                                       outputs=[model.output])
    input = tf.convert_to_tensor(frame, dtype=tf.float32)
    # print(model.input_shape)
    with tf.GradientTape() as tape:
        tape.watch(input)
        layer_output = grad_model(input)
    # print('output:', layer_output)
    # print('input', input)
    grads_val = tape.gradient(layer_output, input)
    # print('gradval', grads_val)
    # print('gradval shape', grads_val.get_shape())
    return grads_val[0]


def grad_cam(model, layer_name, frame):
    grad_model = tf.keras.models.Model(inputs=[model.input],
                                       outputs=[model.output, model.get_layer(layer_name).output])
    # print(model.input_shape)
    with tf.GradientTape() as tape:
        predictions, conv_outputs = grad_model(frame)
        loss = predictions[:, 1]
    # print('Prediction shape:', predictions.get_shape())
    output = conv_outputs
    grads_val = tape.gradient(loss, conv_outputs)[0]
    # print(grads_val.get_shape())
    # weights = np.mean(grads_val, axis=(2, 3))
    weights = tf.reduce_mean(grads_val, axis=(0, 1))

    # weights = weights[0, :]
    output = output[0, :, :, :]

    # weights = np.expand_dims(weights, axis=0)
    # cam = np.dot(output, weights)
    cam = tf.zeros([7,7])
    for i in range(weights.shape[0]):
        cam += weights[i] * output[:, :, i]
    cam = cam.numpy()
    cam = cv2.resize(cam, (84, 84), cv2.INTER_LINEAR)

    # cam = np.maximum(cam, 0)
    cam_max = cam.max()
    if cam_max != 0:
        cam = cam / cam_max
    cam[cam < 0.0] = 0
    return cam


def guided_gradcam(gradcam, guided_prop):
    guided_cam = np.dot(gradcam, guided_prop)
    return guided_cam


def load_model():
    loaded_model = build_q_network(NUMBER_OF_ACTIONS)
    loaded_model.load_weights(os.path.join(MODEL_DIR, SAVED_MODLE))
    return loaded_model


def init_saliency_map(history, first_frame=0, prefix='QF_', resolution=75):
    model = load_model()
    g_model = build_guided_model(4)

    total_frames = len(history['state'])
    fig_array1 = np.zeros((1, NUMBER_OF_FRAMES, INPUT_DIM, INPUT_DIM, 3))
    fig_array2 = np.zeros((1, NUMBER_OF_FRAMES, INPUT_DIM, INPUT_DIM, 3))
    fig_array3 = np.zeros((1, NUMBER_OF_FRAMES, INPUT_DIM, INPUT_DIM, 3))

    for i in range(NUMBER_OF_FRAMES):
        ix = first_frame + i
        if ix < total_frames:
            frame = history['state'][ix].copy()
            frame = np.expand_dims(frame, axis=0)
            if ix % 10 == 0:
                print(ix)
            # print(frame.shape)

            Cam_heatmap = grad_cam(model, LAYER_NAME, frame)
            gbp_heatmap = guided_backpropog(g_model, frame)
            ggc_heatmap = guided_gradcam(Cam_heatmap, gbp_heatmap)

            # Cam_heatmap = np.asarray(Cam_heatmap)
            history['gradCam'].append(Cam_heatmap)
            history['gbp'].append(gbp_heatmap)
            history['ggc'].append(ggc_heatmap)


    history_gradCam = history['gradCam'].copy()
    history_gbp = history['gbp'].copy()
    history_ggc = history['ggc'].copy()

    # fig_array1[0] = normalization(history_gradCam, history, visu='cam')
    fig_array2[0] = normalization(history_gbp, history, visu='gbp')
    # fig_array3[0] = normalization(history_ggc, history, visu='gbp')
    # make_movie(fig_array1, NUMBER_OF_FRAMES, resolution, 'cradcam', ENV_NAME)
    # make_movie(fig_array2, NUMBER_OF_FRAMES, resolution, 'gbp', ENV_NAME)
    # make_movie(fig_array3, NUMBER_OF_FRAMES, resolution, 'gcc', ENV_NAME)



def normalization(heatmap, history, visu):
    heatmap = np.asarray(heatmap)
    print('heat map shape', heatmap.shape)
    if visu == 'gbp':
        heatmap = heatmap[:, :, :]
        heatmap -= heatmap.mean()
        heatmap /= (heatmap.std() + 1e-5)

        # heatmap *= 50
        heatmap *= 0.1

        # clip to [0, 1]
        # gbp_heatmap += 0.5
        heatmap = np.clip(heatmap, -1, 1)
        # TODO fix later
        heatmap_pic1 = heatmap[:, :, :, 0]
        print("heatmapGdb", heatmap_pic1.shape)
    if visu == 'cam':
        heatmap *= 1
        heatmap = np.clip(heatmap, 0, 1)
        heatmap_pic1 = heatmap[:, :, :]
        print("heatmapCAM", heatmap_pic1.shape)

    all_unproc_frames = history['un_proc_state'].copy()
    frame = np.zeros((NUMBER_OF_FRAMES, INPUT_DIM, INPUT_DIM, 3))
    for i in range(NUMBER_OF_FRAMES):
        frame[i, :, :, :] = np.asarray(Image.fromarray(all_unproc_frames[i]).resize((84, 84), Image.BILINEAR)) / 255
    proc_frame1 = overlap(frame, heatmap_pic1)
    return proc_frame1


def overlap(frame, gbp_heatmap):
    print(gbp_heatmap)
    color_neg = [1.0, 0.0, 0.0]
    color_pos = [0.0, 1.0, 0.0]
    color_chan = np.ones((NUMBER_OF_FRAMES, INPUT_DIM, INPUT_DIM, 2), dtype=gbp_heatmap.dtype)
    alpha = 0.2
    # beta = 0.25
    # gbp_heatmap = np.expand_dims(gbp_heatmap, axis=4)
    _gbp_heatmap = [gbp_heatmap for _ in range(3)]
    _gbp_heatmap = np.stack(_gbp_heatmap, axis=3)
    gbp_heatmap = _gbp_heatmap
    # gbp_heatmap = np.concatenate((gbp_heatmap,color_chan),axis=3)
    gbp_heatmap_pos = np.asarray(gbp_heatmap.copy())
    gbp_heatmap_neg = np.asarray(gbp_heatmap.copy())
    gbp_heatmap_pos[gbp_heatmap_pos < 0.0] = 0
    gbp_heatmap_neg[gbp_heatmap_neg >= 0.0] = 0
    gbp_heatmap_neg = -gbp_heatmap_neg
    gbp_heatmap = color_pos * gbp_heatmap_pos[:, :, :, :] + color_neg * gbp_heatmap_neg[:, :, :, :]
    # gbp_heatmap = color_pos * gbp_heatmap_pos[:,:,:,:] + color_neg * gbp_heatmap_neg[:,:,:,:]
    mixed = alpha * gbp_heatmap + (1.0 - alpha) * frame
    mixed = np.clip(mixed, 0, 1)

    return mixed


def make_movie(fig_array, num_frames, resolution, prefix, env_name):
    movie_title = "{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower())
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='test', artist='mateus', comment='atari-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)
    fig = plt.figure(figsize=[6, 6 * 1.3], dpi=resolution)
    print("fig_array.shape: ", fig_array.shape)
    with writer.saving(fig, VIDEO_DIR + movie_title, resolution):
        for i in range(num_frames):
            img = fig_array[0, i, :, :, :]
            plt.imshow(img)
            writer.grab_frame()
            fig.clear()
            if i % 100 == 0:
                print(i)


def play():
    history = {'state': [], 'un_proc_state': [], 'action': [], 'gradCam': [], 'gbp': [], 'ggc': [], 'movie_frames': []}
    game_wrapper = GameWrapper(ENV_NAME, MAX_NOOP_STEPS)

    MAIN_DQN = build_q_network(game_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
    TARGET_DQN = build_q_network(game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, game_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

    print('Loading model...')
    agent.load(MODEL_DIR, load_replay_buffer=False)
    print('Loaded')

    terminal = True
    eval_rewards = []
    evaluate_frame_number = 0

    for frame in range(NUMBER_OF_FRAMES):
        if terminal:
            original_frame, state = game_wrapper.reset(evaluation=True)
            life_lost = True
            episode_reward_sum = 0
            terminal = False

        history['state'].append(game_wrapper.state)
        history['un_proc_state'].append(original_frame)
        action = 1 if life_lost else agent.get_action(0, game_wrapper.state, evaluation=True)

        original_frame, state, reward, terminal, life_lost = game_wrapper.step(action, gradcam=True)
        history['action'].append(action)
        evaluate_frame_number += 1
        episode_reward_sum += reward

        if terminal:
            print(f'Game over, reward: {episode_reward_sum}, frame: {frame}/{EVAL_LENGTH}')
            eval_rewards.append(episode_reward_sum)

    print('Average reward:', np.mean(eval_rewards) if len(eval_rewards) > 0 else episode_reward_sum)
    return history


if __name__ == '__main__':
    history = play()
    init_saliency_map(history)
