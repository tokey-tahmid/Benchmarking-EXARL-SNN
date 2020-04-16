# (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and 
# to permit others to do so.


import gym
import exarl as erl

import os
import csv

from mpi4py import MPI
import json

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RL-Logger')
logger.setLevel(logging.INFO)

class ExaLearner():

    def __init__(self, agent_id, env_id, cfg='envs/env_vault/env_cfg/env_setup.json'):
        ## Default training 
        self.nepisodes=1
        self.nsteps=10
        self.results_dir='./results'
        self.do_render=False

        ## Setup agent and environents
        self.agent_id = agent_id
        self.env_id   = env_id
        self.agent, self.env = erl.make(agent_id, env_id)
        self.env._max_episode_steps=self.nsteps

        self.cfg = cfg
        with open(self.cfg) as json_file:
            data = json.load(json_file)
        self.mpi_children_per_parent = int(data['mpi_children_per_parent']) \
                                       if 'mpi_children_per_parent' in data.keys() \
                                       else 0

    def set_training(self,nepisodes,nsteps):
        self.nepisodes = nepisodes
        self.nsteps    = nsteps
        self.env._max_episode_steps=self.nsteps

    def set_config(self, params):
        self.nepisodes = int(params['n_episodes'])
        self.nsteps    = int(params['n_steps'])
        self.env._max_episode_steps = self.nsteps
        # set the agent up here also
        self.agent.set_config(params)

    def set_results_dir(self,results_dir):
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        ## General 
        self.results_dir=results_dir
        ## Set for agent
        self.agent.set_results_dir(self.results_dir)
        ## Set for env
        self.env.set_results_dir(self.results_dir)

    def render_env(self):
        self.do_render=True

    def run_exarl(self, comm):

        filename_prefix = 'ExaLearner_' + 'Episode%s_Steps%s_Rank%s_memory_v1' % ( str(self.nepisodes), str(self.nsteps), str(comm.rank))
        train_file = open(self.results_dir+'/'+filename_prefix + ".log", 'w')
        train_writer = csv.writer(train_file, delimiter = " ")
        for e in range(self.nepisodes):

            rank0_memories = 0
            target_weights = None
            current_state = self.env.reset()
            total_reward=0
            done = False

            while done!=True:
                ## All workers ##
                action = self.agent.action(current_state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward+=reward
                memory = (current_state, action, reward, next_state, done, total_reward)
                new_data = comm.gather(memory, root=0)
                logger.info('Rank[%s] - Memory length: %s ' % (str(comm.rank),len(self.agent.memory)))

                ## Learner ##
                if comm.rank==0:
                    ## Push memories to learner ##
                    for data in new_data:
                        self.agent.remember(data[0],data[1],data[2],data[3],data[4])
                        ## Train learner ##
                        self.agent.train()
                        rank0_memories = len(self.agent.memory)
                        target_weights = self.agent.get_weights()
                        if rank0_memories%(comm.size)==0:
                            self.agent.save(self.results_dir+filename_prefix+'.h5')

                ## Broadcast the memory size and the model weights to the workers  ##
                rank0_memories = comm.bcast(rank0_memories, root=0)
                current_weights = comm.bcast(target_weights, root=0)
                logger.info('Rank[%s] - rank0 memories: %s' % (str(comm.rank),str(rank0_memories)))

                ## Set the model weight for all the workers
                if comm.rank>0 and rank0_memories>30:# and rank0_memories%(size)==0:
                    logger.info('## Rank[%s] - Updating weights ##' % str(comm.rank))
                    self.agent.set_weights(current_weights)

                ## Update state
                current_state = next_state
                logger.info('Rank[%s] - Total Reward:%s' % (str(comm.rank),str(total_reward) ))

                ## Save memory for offline analysis
                train_writer.writerow([current_state,action,reward,next_state,total_reward,done])
                train_file.flush()

                ## Save Learning target model
                if comm.rank==0:
                    self.agent.save(self.results_dir+filename_prefix+'.h5')
        train_file.close()

    def run(self, run_type):
        self.agent.save_info()
        comm = MPI.COMM_WORLD
        if run_type == 'static':
            if comm.rank%(self.mpi_children_per_parent+1) == 0:
                self.run_exarl(comm)
        elif run_type == 'dynamic':
            self.run_exarl(comm)
