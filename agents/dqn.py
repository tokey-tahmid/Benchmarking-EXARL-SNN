import random,sys
import numpy as np
from collections import deque
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Input,BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import csv
import json
import math

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

#The Deep Q-Network (DQN)
class DQN:
    def __init__(self, env,cfg='agents/agent_cfg/dqn_setup.json'):
        self.env = env
        self.memory = deque(maxlen = 2000)

        ## Implement the UCB approach
        self.sigma = 2 # confidence level
        self.total_actions_taken = 1
        self.individual_action_taken = np.ones(self.env.action_space.n)
            
        ## Setup GPU cfg
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)
        
        ## Get hyper-parameters from json cfg file
        data = []
        with open(cfg) as json_file:
            data = json.load(json_file)
            
        self.search_method =  (data['search_method']).lower() if 'search_method' in data.keys() else "epsilon"  # discount rate
        self.gamma =  float(data['gamma']) if 'gamma' in data.keys() else 0.95  # discount rate
        self.epsilon = float(data['epsilon']) if 'epsilon' in data.keys() else 1.0  # exploration rate
        self.epsilon_min = float(data['epsilon_min']) if 'epsilon_min' in data.keys() else 0.05
        self.epsilon_decay = float(data['epsilon_decay']) if 'epsilon_decay' in data.keys() else 0.995
        self.learning_rate =  float(data['learning_rate']) if 'learning_rate' in data.keys() else  0.001
        self.batch_size = int(data['batch_size']) if 'batch_size' in data.keys() else 32
        self.tau = float(data['tau']) if 'tau' in data.keys() else 0.5

        ##
        self.model = self._build_model()
        self.target_model = self._build_model()

        ## Save infomation ##
        train_file_name = "dqn_mse_cartpole_%s_lr%s__tau%s_v1.log" % (self.search_method, str(self.learning_rate) ,str(self.tau) )
        #train_file_name = "dqn_mse_tau_bcp_rewardv2_%s_lr%s_fixinit_v1.log" % (self.search_method, str(self.learning_rate) )
        self.train_file = open(train_file_name, 'w')
        self.train_writer = csv.writer(self.train_file, delimiter = " ")

    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1,axis=-1)

    def _build_model(self):
        ## Input: state ##       
        state_input = Input(self.env.observation_space.shape)
        #s1 = BatchNormalization()(state_input)
        h1 = Dense(24, activation='relu')(state_input)
        #b1 = BatchNormalization()(h1)
        h2 = Dense(48, activation='relu')(h1)
        #b2 = BatchNormalization()(h2)
        h3 = Dense(24, activation='relu')(h2)
        ## Output: action ##   
        output = Dense(self.env.action_space.n,activation='relu')(h3)
        model = Model(input=state_input, output=output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
#        model.compile(loss='mean_squared_logarithmic_error', optimizer=adam)
        return model       

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def action(self, state):
        action = -1
        ## TODO: Update greed-epsilon to something like UBC
        if np.random.rand() <= self.epsilon and self.search_method=="epsilon":
            print('Random action')
            action = random.randrange(self.env.action_space.n)        
            ## Update randomness
            if len(self.memory)>(self.batch_size):
                self.epsilon_adj()
        else:
            print('Policy action')
            np_state = np.array(state).reshape(1,len(state))
            act_values = self.target_model.predict(np_state)
            action = np.argmax(act_values[0])
            ## Adding the UCB 
            #if self.search_method=="ucb":
            #    print('START UCB')
            #    print( 'Default values')
            #    print( (act_values))
            #    print( (action))
            #    act_values +=  self.sigma*np.sqrt(math.log(self.total_actions_taken)/self.individual_action_taken)
            #    action = np.argmax(act_values[0])
            #    print( 'UCB values')
            #    print( (act_values))
            #    print( (action))
            #    ## Check if there are multiple candidates and select one randomly
            #    mask = [i for i in range(len(act_values[0])) if act_values[0][i] == act_values[0][action]]
            #    ncands=len(mask)
            #    print( 'Number of cands: %s' % str(ncands))
            #    if ncands>1:
            #        action = mask[random.randint(0,ncands-1)]
            #    print( (action))
            #    print('END UCB')
            print(act_values)
            print(action)
            mask = [i for i in range(len(act_values[0])) if act_values[0][i] == act_values[0][action]]
            ncands=len(mask)
            print( 'Number of cands: %s' % str(ncands))
            if ncands>1:
                action = mask[random.randint(0,ncands-1)]
        ## Capture the action statistics for the UBC methods
        print('total_actions_taken: %s' % self.total_actions_taken)
        print('individual_action_taken[%s]: %s' % (action,self.individual_action_taken[action]))
        self.total_actions_taken += 1
        self.individual_action_taken[action]+=1

        return action

    def play(self,state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self):
        if len(self.memory)<(self.batch_size):
            return
        print('### TRAINING ###')
        losses = []
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            np_state = np.array(state).reshape(1,len(state))
            np_next_state = np.array(next_state).reshape(1,len(next_state))
            expectedQ =0 
            if not done:
                expectedQ = self.gamma*np.amax(self.target_model.predict(np_next_state)[0])
            target = reward + expectedQ
            target_f = self.model.predict(np_state)
            target_f[0][action] = target
            history = self.model.fit(np_state, target_f, epochs = 1, verbose = 0)
            losses.append(history.history['loss'])
        self.target_train()
        self.train_writer.writerow([np.mean(losses)])
        self.train_file.flush()            

    def target_train(self):
        if len(self.memory)%(self.batch_size)!=0:
            return
        model_weights  = self.model.get_weights()
        target_weights =self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau*model_weights[i] + (1-self.tau)*target_weights[i]
        self.target_model.set_weights(target_weights)
        #self.target_model.set_weights(self.model.get_weights())

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)