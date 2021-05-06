import json
import os
import numpy as np
import tensorflow as tf

class Agent(object):
    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 n_actions,
                 input_shape=(84, 84),
                 batch_size=32,
                 history_length=4,
                 eps_initial=1.0,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=1000000,
                 replay_buffer_start_size=50000,
                 max_frames=25000000,
                 use_per=True):

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length

        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.use_per = use_per

        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

       
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, frame_number, evaluation=False):
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2 * frame_number + self.intercept_2

    def get_action(self, frame_number, state, evaluation=False):
        eps = self.calc_epsilon(frame_number, evaluation)
        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        q_vals = self.DQN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0]
        return q_vals.argmax()

    def get_intermediate_representation(self, state, layer_names=None, stack_state=True):
        """
        Get the output of a hidden layer inside the model.  This will be/is used for visualizing model

        Arguments:
            state: The input to the model to get outputs for hidden layers from
            layer_names: Names of the layers to get outputs from.  This can be a list of multiple names, or a single name
            stack_state: Stack `state` four times so the model can take input on a single (84, 84, 1) frame

        Returns:
            Outputs to the hidden layers specified, in the order they were specified.
        """
        # Prepare list of layers
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.DQN.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.DQN.get_layer(name=layer_names).output

        # Model for getting intermediate output
        temp_model = tf.keras.Model(self.DQN.inputs, layers)

        # Stack state 4 times
        if stack_state:
            if len(state.shape) == 2:
                state = state[:, :, np.newaxis]
            state = np.repeat(state, self.history_length, axis=2)

        # Put it all together
        return temp_model.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))

    def update_target_network(self):
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        self.replay_buffer.add_experience(action, frame, reward, terminal, clip_reward)

    @tf.function
    def gradient(self, rewards, gamma, double_q, terminal_flags, states, actions, importance):
        flag = 0.0 if terminal_flags else 1.0
        target_q = rewards + (gamma * double_q * (1.0 - flag))
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)
            one_hot_actions = tf.one_hot(actions, self.n_actions, dtype=np.float32)
            Q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            td_error = Q - target_q
            loss = tf.keras.losses.Huber()(target_q, Q)

            if self.use_per:
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        gradient_norm = tf.linalg.global_norm(model_gradients)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))
        return gradient_norm, td_error, loss

    def learn(self, batch_size, gamma, frame_number, priority_scale=1.0):
        if self.use_per:
            (states, actions, rewards, new_states, terminal_flags), importance, indices = self.replay_buffer.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)
            importance = importance ** (1-self.calc_epsilon(frame_number))
        else:
            states, actions, rewards, new_states, terminal_flags = self.replay_buffer.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)

        arg_q_max = self.DQN.predict(new_states).argmax(axis=1)
        future_q_vals = self.target_dqn.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]

        # bellman equation

        gradient_norm, td_error, loss = self.gradient(rewards, gamma, double_q, terminal_flags, states, actions, importance)

        if self.use_per:
            self.replay_buffer.set_priorities(indices, td_error)

        return float(loss.numpy()), td_error, gradient_norm

    def save(self, folder_name, **kwargs):

        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        self.DQN.save(folder_name + '/dqn.h5')
        self.target_dqn.save(folder_name + '/target_dqn.h5')

        self.replay_buffer.save(folder_name + '/replay-buffer')

        with open(folder_name + '/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        self.DQN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.target_dqn = tf.keras.models.load_model(folder_name + '/target_dqn.h5')
        self.optimizer = self.DQN.optimizer

        if load_replay_buffer:
            self.replay_buffer.load(folder_name + '/replay-buffer')



        if load_replay_buffer:
            with open(folder_name + '/meta.json', 'r') as f:
                meta = json.load(f)
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

            del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        # return meta