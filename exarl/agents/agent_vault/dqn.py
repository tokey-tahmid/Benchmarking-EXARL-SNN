import time
import os
import random
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import gym
import pickle
import exarl as erl
from exarl.base.comm_base import ExaComm
from tensorflow import keras
from collections import deque
from datetime import datetime
import numpy as np
from exarl.agents.agent_vault._prioritized_replay import PrioritizedReplayBuffer
import exarl.utils.candleDriver as cd
from exarl.utils import log
from exarl.utils.introspect import introspectTrace
from tensorflow.compat.v1.keras.backend import set_session

import bindsnet.network as network
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import MSTDP
from bindsnet.encoding import poisson

if ExaComm.num_learners > 1:
    import horovod.tensorflow as hvd
    multiLearner = True
else:
    multiLearner = False

logger = log.setup_logger(__name__, cd.lookup_params('log_level', [3, 3]))

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))

class DQN(erl.ExaAgent):
    def __init__(self, env, is_learner):

        self.is_learner = is_learner
        self.model = None
        self.target_model = None
        self.target_weights = None
        self.device = None
        self.mirrored_strategy = None

        self.env = env
        self.agent_comm = ExaComm.agent_comm
        self.rank = self.agent_comm.rank
        self.size = self.agent_comm.size
        self.training_time = 0
        self.ntraining_time = 0
        self.dataprep_time = 0
        self.ndataprep_time = 0

        self.enable_xla = True if cd.run_params['xla'] == "True" else False
        if self.enable_xla:
            tf.config.optimizer.set_jit(True)
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
        self.results_dir = cd.run_params['output_dir']
        self.gamma = cd.run_params['gamma']
        self.epsilon = cd.run_params['epsilon']
        self.epsilon_min = cd.run_params['epsilon_min']
        self.epsilon_decay = cd.run_params['epsilon_decay']
        self.learning_rate = cd.run_params['learning_rate']
        self.batch_size = cd.run_params['batch_size']
        self.tau = cd.run_params['tau']
        self.model_type = cd.run_params['model_type']

        self.activation = cd.run_params['activation']
        self.out_activation = cd.run_params['out_activation']
        self.optimizer = cd.run_params['optimizer']
        self.loss = cd.run_params['loss']
        self.n_actions = cd.run_params['nactions']
        self.priority_scale = cd.run_params['priority_scale']
        self.is_discrete = (type(env.action_space) == gym.spaces.discrete.Discrete)
        if not self.is_discrete:
            env.action_space.n = self.n_actions
            self.actions = np.linspace(env.action_space.low, env.action_space.high, self.n_actions)
        self.dtype_action = np.array(self.env.action_space.sample()).dtype
        self.dtype_observation = self.env.observation_space.sample().dtype
        
        self.device = torch.device("cpu")

        if self.is_learner:
            self.model = self._build_model().to(self.device)
            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = True
            for name, param in self.model.named_parameters():
                print(name, param.requires_grad)
            self.loss_fn = nn.MSELoss()
            self.opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.target_model = self._build_model()
        self.target_model.to(self.device)
        self.target_weights = None

        if multiLearner and ExaComm.is_learner():
            hvd.init(comm=ExaComm.learner_comm.raw())
            self.first_batch = 1
            self.loss_fn = cd.candle.build_loss(self.loss, cd.kerasDefaults, reduction='none')
            self.opt = cd.candle.build_optimizer(self.optimizer, self.learning_rate * hvd.size(), cd.kerasDefaults)
        self.maxlen = cd.run_params['mem_length']
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=self.maxlen)


    def _build_model(self):
        if self.model_type == 'SNN':
            return self._build_snn_model()
        else:
            sys.exit("Oops! That was not a valid model type. Try again...")

    def _build_snn_model(self):
        net = Network(dt=1.0)
        input_layer = Input(n=self.env.observation_space.shape[0])
        middle_layer = LIFNodes(n=128)
        output_layer = LIFNodes(n=self.env.action_space.n)

        input_middle_conn = Connection(source=input_layer, target=middle_layer, rule=MSTDP, nu=(1e-4, 1e-2))
        middle_output_conn = Connection(source=middle_layer, target=output_layer, rule=MSTDP, nu=(1e-4, 1e-2))
        net.add_layer(input_layer, name="Input")
        net.add_layer(middle_layer, name="Middle")
        net.add_layer(output_layer, name="Output")
        net.add_connection(input_middle_conn, source="Input", target="Middle")
        net.add_connection(middle_output_conn, source="Middle", target="Output")

        output_monitor = Monitor(obj=output_layer, state_vars=("s",), time=125, device=self.device)
        net.add_monitor(output_monitor, name="Output")
        net.to(self.device)

        return net

    def set_learner(self):
        logger.debug(
            "Agent[{}] - Creating active model for the learner".format(self.rank)
        )

    def remember(self, state, action, reward, next_state, done):
        lost_data = self.replay_buffer.add((state, action, reward, next_state, done))
        if lost_data and self.priority_scale:
            print("Priority replay buffer size too small. Data loss negates replay effect!", flush=True)

    def get_action(self, state):
        random.seed(datetime.now())
        random_data = os.urandom(4)
        np.random.seed(int.from_bytes(random_data, byteorder="big"))
        rdm = np.random.rand()
        if rdm <= self.epsilon:
            self.epsilon_adj()
            action = random.randrange(self.env.action_space.n)
            return action, 0
        else:
            if self.model_type == 'SNN':
                encoded_state = poisson(datum=torch.tensor(state, dtype=torch.float).to(self.device), time=125)
                self.target_model.to(self.device)
                inputs = {"Input": encoded_state}
                self.target_model.run(inputs=inputs, time=125)

                output_spikes = self.target_model.monitors["Output"].get("s").to(self.device)
                action = torch.argmax(output_spikes.sum(dim=0)).item()

                return action, "SNN_policy"

    @introspectTrace()
    def action(self, state):
        action, policy = self.get_action(state)
        if not self.is_discrete:
            action = [self.actions[action]]
        return action, policy

    @introspectTrace()
    def calc_target_f(self, data):
        state, action, reward, next_state, done = data
        if self.model_type == 'SNN':

            encoded_next_state = poisson(datum=torch.tensor(next_state, dtype=torch.float).to(self.device), time=125)
            self.target_model.to(self.device) 
            inputs = {"Input": encoded_next_state}
            self.target_model.run(inputs=inputs, time=125)

            output_spikes = self.target_model.monitors["Output"].get("s").to(self.device)
            q_values = output_spikes.sum(dim=0).numpy()
    
            if done:
                expectedQs = reward + np.zeros_like(q_values)
            else:
                expectedQs = reward + self.gamma * q_values

        return expectedQs



    def has_data(self):
        return (self.replay_buffer.get_buffer_length() >= self.batch_size)

    @introspectTrace()
    def generate_data(self):
        if not self.has_data():
            batch_states = np.zeros((self.batch_size, 1, self.env.observation_space.shape[0]), dtype=self.dtype_observation)
            batch_target = np.zeros((self.batch_size, self.env.action_space.n), dtype=self.dtype_action)
            indices = -1 * np.ones(self.batch_size)
            importance = np.ones(self.batch_size)
        else:
            minibatch, importance, indices = self.replay_buffer.sample(self.batch_size, priority_scale=self.priority_scale)
            batch_target = list(map(self.calc_target_f, minibatch))
            batch_states = [np.array(exp[0], dtype=self.dtype_observation).reshape(1, 1, len(exp[0]))[0] for exp in minibatch]
            batch_states = np.reshape(batch_states, [len(minibatch), 1, len(minibatch[0][0])])
            batch_target = np.reshape(batch_target, [len(minibatch), self.env.action_space.n])

        if self.priority_scale > 0:
            yield batch_states, batch_target, indices, importance
        else:
            yield batch_states, batch_target

    @introspectTrace()
    def train(self, batch):
        ret = None
        if self.is_learner:
            start_time = time.time()
            if self.priority_scale > 0:
                if self.model_type == 'SNN':

                    encoded_state = poisson(datum=torch.tensor(batch[0], dtype=torch.float).to(self.device), time=125)
                    inputs = {"Input": encoded_state}
                    self.model.to(self.device)
                    self.model.run(inputs=inputs, time=125)
                    output_spikes = self.target_model.monitors["Output"].get("s").to(self.device)
                    if output_spikes.nelement() == 0:
                        logger.warning("No spikes recorded in the Output monitor.")
                        return

                    predictedQs = output_spikes.sum(dim=0).numpy()
                    targetQs = batch[1]
                
                    loss_value = self.training_step(batch)

                    if len(batch) == 4:
                        ret = batch[2], loss_value
                    else:
                        ret = None, loss_value
            end_time = time.time()
            self.training_time += (end_time - start_time)
            self.ntraining_time += 1
            logger.info('Agent[{}]- Training: {} '.format(self.rank, (end_time - start_time)))
            start_time_episode = time.time()
            logger.info('Agent[%s] - Target update time: %s ' % (str(self.rank), str(time.time() - start_time_episode)))
        else:
            logger.warning('Training will not be done because this instance is not set to learn.')
        return ret



    def training_step(self, batch):
        encoded_state = poisson(datum=torch.tensor(batch[0], dtype=torch.float).to(self.device), time=125)
        encoded_state = encoded_state.to(dtype=torch.float32)
        print(encoded_state.dtype)

        inputs = {"Input": encoded_state}
        self.model.run(inputs=inputs, time=125)
        output_spikes = self.model.monitors["Output"].get("s")
        predictedQs = output_spikes.sum(dim=0).numpy()
        targetQs = batch[1]
        mse_loss = np.mean((predictedQs - targetQs) ** 2)
        
        return mse_loss.item()
    def set_priorities(self, indices, loss):
        self.replay_buffer.set_priorities(indices, loss)

    def get_weights(self):
        logger.debug("Agent[%s] - get target weight." % str(self.rank))
        return self.target_model.get_weights()

    def set_weights(self, weights):
        logger.info("Agent[%s] - set target weight." % str(self.rank))
        logger.debug("Agent[%s] - set target weight: %s" % (str(self.rank), weights))
        self.target_model.set_weights(weights)

    @introspectTrace()
    def target_train(self):
        if self.is_learner:
            logger.info("Agent[%s] - update target weights." % str(self.rank))
        
            if self.model_type == 'SNN':
                input_middle_conn = self.model.connections[("Input", "Middle")]
                middle_output_conn = self.model.connections[("Middle", "Output")]
            
                model_weights_input_middle = input_middle_conn.w
                model_weights_middle_output = middle_output_conn.w
            
                target_weights_input_middle = self.target_model.connections[("Input", "Middle")].w
                target_weights_middle_output = self.target_model.connections[("Middle", "Output")].w
            
                target_weights_input_middle.data = self.tau * model_weights_input_middle + (1 - self.tau) * target_weights_input_middle
                target_weights_middle_output.data = self.tau * model_weights_middle_output + (1 - self.tau) * target_weights_middle_output
            
        else:
            logger.warning(
                "Weights will not be updated because this instance is not set to learn."
            )

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename):
        layers = self.target_model.layers
        with open(filename, 'rb') as f:
            pickle_list = pickle.load(f)

        for layerId in range(len(layers)):
            layers[layerId].set_weights(pickle_list[layerId][1])

    def save(self, filename):
        if self.model_type == 'SNN':
            torch.save(self.target_model.state_dict(), filename)

    def update(self):
        logger.info("Implement update method in dqn.py")

    def monitor(self):
        logger.info("Implement monitor method in dqn.py")
