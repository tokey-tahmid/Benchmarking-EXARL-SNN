import os
import sys
import time
import math
import scipy
import pylab
import random
import argparse
import time
import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

import gym
from gym import spaces

sys.path.append('envs/env_vault/CahnHilliard2D/cpp/swig/') # CH-solver

from utils import *
import ch2d.aligned_vector as av
import ch2d.cahnhilliard as ch
import image_structure

############################# Environment class #############################

class CahnHilliardEnv(gym.Env):

  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

      def __init__(self, args: argparse.Namespace, comm=None):
        super(CahnHilliardEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Discrete( self.getActionSize() )
        self.observation_space = spaces.Box(
                low=np.zeros(self.args.size_struct_vec),
                high=np.ones(self.args.size_struct_vec)*10000, dtype=np.uint8)

        #self.observation_space = spaces.Box(low=0, high=255, shape=
        #                (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.args = args
        self.comm = comm
        if self.comm:
            self.comm_rank = self.comm.Get_rank()
        else:
            self.comm_rank = 0

        self.num_control_params = 1        # number of controlling parameter

        self.change_T = self.args.changeT  # 0.1   #
        self.initialT = self.args.initT    # 0.5   # initial temperature value for RL training
        self.targetT  = 0.8                # target temperature value to generate target

        self.currStructVec   = np.zeros(args.size_struct_vec)   # stores current structure vector
        self.targetStructVec = np.zeros(args.size_struct_vec)   # target  structure vector (loaded or generated at setTargetState() once)
        self.targetStructVecNorm = 1.
        self.reference_norm_function = lambda x: np.linalg.norm(x, ord=np.inf)

        self.vecWeight       = np.zeros(self.args.size_struct_vec)
        self.rangeStructVec  = np.zeros(self.args.size_struct_vec)

        self.maxStructVec    = [ -math.inf for _ in range(self.args.size_struct_vec)]
        self.minStructVec    = [  math.inf for _ in range(self.args.size_struct_vec)]

        self.hasBaseScore = False

        #timers
        self.time_getNextState    = 0
        self.time_getReward       = 0
        self.time_isTerminalState = 0

        self.startAnalyze         = False
        self.episode              = -1
        self.isTest               = True if self.args.notTrain else False
        self.isTarget = False

        self.setTargetState()

    ##################### get state space (mean, sd) #####################
    def getStateSize(self):  # state size is # of structured vector components and the current temperature
        return self.args.size_struct_vec+1

    ##################### get action space space (3^N) #####################
    def getActionSize(self):
        return 3  # 3^N

    ##################### set the initial temperature randomly #####################
    def setRandInitT(self):
        return np.random.uniform( self.chparams.T_min , self.chparams.T_max )

    ##################### reset to the initial state #####################
    def reset(self):

        self.episode += 1
        #print('episode', self.episode)
        if self.episode == self.args.episodes: self.isTest=True

        self.setInitSimParams()  # TODO: I do not have to initialze all parameter at each episode

        if self.args.randInitial: self.T = self.setRandInitT()
        else:                     self.T = self.initialT
                  # initial value, always start from the same initial value for now.

        print_status('Initial T: {}'.format(self.T), comm_rank=self.comm_rank)

        self.setControlParams(T=self.T)

        # TODO: Do I need to call this function here
        #       considering not calculating a structure vector
        #t0 = time.time()
        #ch.run_ch_solver(self.chparams, self.info)
        #print("Time ch_solver: ",time.time()-t0)

        # No need to calculate initial structure vector
        #self.currStructVec = self.getImgStruct("C_1.out", 1) # get the current structure vector
        #if self.args.debug>=-1: print("Init struct vec: ", np.around(self.currStructVec, 2))

        state = []
        state = np.append([self.T], self.currStructVec)
        return state

    ##################### set the target state #####################
    def setTargetState(self):

        if (self.args.genTarget): # generate target
            self.generateTargetState()

        else:  # given target
            if self.args.size_struct_vec==6:
                self.targetStructVec = [278.0, 0.0, 0.0, 69.0, 75.0, -4.0]
            elif self.args.size_struct_vec==2:
                self.targetStructVec = [9, 1] #[10.44, 1.6]  #[7.1, 1.6]
            elif self.args.size_struct_vec==200:
                self.targetStructVec = np.genfromtxt(self.args.target_dir + \
                                       self.args.target_file + '.out')[1]
            else:
                assert('linux' in sys.platform), \
                    "Error: The number of structure vector components should be 2, 6, or 200!"

        if self.args.debug>=10: print("Target Structured Vector: ", \
                                      np.around(self.targetStructVec,2))

    ########################### generate target state ###########################
    def generateTargetState(self):

        self.isTarget = True

        self.setInitSimParams()
        self.setControlParams(T=self.targetT)

        self.chparams.T_const = av.aligned_double_vector(self.targetT * np.ones(int(self.info.nx)**2))

        # create a target directory
        out_dir = os.path.join(self.args.target_dir)
        if not os.path.exists(out_dir): os.mkdir(out_dir)

        for i in range(self.args.steps):
            self.info.t0 = self.t[i]
            self.info.tf = self.t[i+1]
            if self.args.debug>=10:
                print( 't0 = ', self.t[i]/self.lin_dt, ' dt_lin , tf = ', \
                                self.t[i+1]/self.lin_dt, ' dt_lin')

            if self.args.debug>=1: t0 = time.time()
            residual = ch.run_ch_solver(self.chparams, self.info)
            residual /= np.linalg.norm(np.array(self.info.x), ord=2)
            if self.args.debug>=1:
                print_status("Time ch_solver: {}, solver relative residual: {:.6f}".format(time.time()-t0, residual), \
                             comm_rank=self.comm_rank)

            target_data = np.array(self.info.x)
            self.targetStructVec = self.getImgStruct(target_data, i)
            self.targetStructVecNorm = self.reference_norm_function(self.targetStructVec)

        if self.args.debug>=10:
            print_status("Target Structured Vector: ", np.around(self.targetStructVec,2), \
                         comm_rank=self.comm_rank, allranks=True)

        #self.targetStructVec = self.getImgStruct(target_data, self.args.steps)

        self.isTarget = False


    ################################ get the next setp info ################################
    # return the next state, reward, whether or not at the terimnal state
    def step(self, action_idx, t):

        #get the next state
        if self.args.debug>=1: time_tmp = time.time()
        self.currStructVec = self.getNextState(action_idx, t)
        state = []
        state = np.append([self.T], self.currStructVec)
        if self.args.debug>=1: self.time_getNextState += time.time() - time_tmp

        #get reward
        if self.args.debug>=1: time_tmp = time.time()
        reward             = self.getReward(t)
        if self.args.debug>=1: self.time_getReward += time.time() - time_tmp

        #check if done
        if self.args.debug>=1: time_tmp = time.time()
        deviation          = self.distanceFromTerminalState()
        if self.args.debug>=1: self.time_isTerminalState += time.time() - time_tmp

        return state, reward, deviation, None

    ########################################################
    # whether or not this state is the terminal state
    # if the difference of each component is less than 0.0001 of the range of each compoenent,
    # we consder that we reached the target
    def distanceFromTerminalState(self):
        return self.reference_norm_function(self.currStructVec-self.targetStructVec)

    def isTerminalState(self, normdiff=None):  # TODO: I need to give larger reward when it in the terminal state
        #use defined norm. Don't recompute if previously computed
        if normdiff==None:
            normdiff = self.distanceFromTerminalState()
        if normdiff < 0.0001 * self.targetStructVecNorm:
            print_status("!!!!!!!!!!!!!!!!!!!!!!!!! Terminal State Reached !!!!!!!!!!!!!!!!!!!!!!!!!", comm_rank=self.comm_rank, allranks=True)
            return True
        else:
            return False

    ############################ reward function ####################################
    # TODO: modify this reward function
    def getReward(self, t):

        #self.adjustWeight()

        if self.args.debug>=30: print_status("wt {}".format(self.vecWeight), comm_rank=self.comm_rank, allranks=True)

        reward= 0.0

        #if t==self.args.steps-1:

        if self.args.rewardOption==2 or self.args.rewardOption==3 or self.args.rewardOption==5:
            for i in range(self.args.size_struct_vec):
                reward -= 1.0/self.args.size_struct_vec * abs(self.currStructVec[i]-self.targetStructVec[i])
        elif self.args.rewardOption==0 or self.args.rewardOption==1 or self.args.rewardOption==4:
            for i in range(self.args.size_struct_vec):
                #reward -= self.vecWeight[i] * (self.currStructVec[i]-self.targetStructVec[i])**2
                reward -= 1.0/self.args.size_struct_vec * (self.currStructVec[i]-self.targetStructVec[i])**2

        if self.args.rewardOption==4 or self.args.rewardOption==5:

            if not self.hasBaseScore:
                 self.startAnalyze = True
                 self.base_score   = - reward * 1.2

            # reward += self.base_score
            # reward *= np.exp( 4 + 2.0 * t / self.args.steps )
            #print("rgeteward", reward)

        return reward

    ################## adjust weight ##################
    def adjustWeight(self):

        isAdjusted = False

        for i in range(self.args.size_struct_vec):
            if (self.maxStructVec[i] < self.currStructVec[i]):
                self.maxStructVec[i] = self.currStructVec[i]
                isAdjusted = True
            if (self.minStructVec[i] > self.currStructVec[i]):
                self.minStructVec[i] = self.currStructVec[i]
                isAdjusted = True

        if isAdjusted:
            total = 0.0
            for i in range(self.args.size_struct_vec):
                self.rangeStructVec[i] = self.maxStructVec[i] - self.minStructVec[i]
                total += self.rangeStructVec[i]
            if total==0:
                for i in range(self.args.size_struct_vec):
                    self.vecWeight[i] = 1 / self.args.size_struct_vec
            else:
                for i in range(self.args.size_struct_vec):
                    self.vecWeight[i] = self.rangeStructVec[i] / total

            if self.args.debug>=10: print_status('vecWeight: {}'.format(np.around(self.vecWeight,2)), comm_rank=self.comm_rank, allranks=True)
            if self.args.debug>=30: print_status('maxStructVec: {}'.format(np.around(self.maxStructVec,2)), comm_rank=self.comm_rank, allranks=True)
            if self.args.debug>=30: print_status('minStructVec:{}'.format(np.around(self.minStructVec,2)), comm_rank=self.comm_rank, allranks=True)
            if self.args.debug>=30: print_status('rangeStructVec: {}'.format(np.around(self.rangeStructVec,2)), comm_rank=self.comm_rank, allranks=True)

    ################################ get the next action ################################
    def setNextAction(self, action_idx):

        if self.args.debug>=30:
            print_status("action idex: {}".format(action_idx), \
                         comm_rank=self.comm_rank, allranks=True)

        if self.T <= self.chparams.T_min+.05 and action_idx==0:
            self.T = self.T
        elif self.T >= self.chparams.T_max-.05 and action_idx==2:
            self.T = self.T
        else:
            self.T += (action_idx-1) * self.change_T         # action_idx maps to action

        if self.args.debug>=30:
            print_status("getNextAction T param: {}".format(np.around(self.T,2)), \
                         comm_rank=self.comm_rank, allranks=True)
        # return self.T

    ########################################################
    # return the next state after taking the action at the current state
    def getNextState(self, action_idx, i):

        self.setNextAction(action_idx)     # self.T is updated to the next action
        self.setControlParams(T = self.T)  # set control parameter for the simulation code

        self.info.t0 = self.t[i]
        self.info.tf = self.t[i+1]
        if self.args.debug>=10:
            print( 't0 = ', self.t[i]/self.lin_dt, ' dt_lin , tf = ', \
                            self.t[i+1]/self.lin_dt, ' dt_lin')

        self.chparams.T_const = av.aligned_double_vector(self.T * np.ones(int(self.info.nx)**2))

        if self.args.debug>=1: t0 = time.time()
        residual = ch.run_ch_solver(self.chparams, self.info)
        residual /= np.linalg.norm(np.array(self.info.x), ord=2)

        if self.args.debug>=1:
            print_status("Time ch_solver: {}, solver relative residual: {:.6f}".format(time.time()-t0,residual), \
                         comm_rank=self.comm_rank)

        if self.args.debug>=10:
            print_status("exit ch solver", comm_rank=self.comm_rank, allranks=True)

        img_data = np.array(self.info.x)

        #only for debugging purposes
        #img_data_test = np.genfromtxt("results_dqn/episode_0/C_" + str(i+2) + ".out")

        #get image structure
        if self.args.debug>=1: t0 = time.time();
        img_struct = self.getImgStruct(img_data, i+2)
        if self.args.debug>=1:
            print_status("Time getImgStruct: {}".format(time.time()-t0), \
                          comm_rank=self.comm_rank)

        return img_struct

    ########################## initialize parameters for the simulation #########################
    # TODO: this function should be called once, not each episode
    def setInitSimParams(self):

        # ********* POLYMER PARAMETERS *********
        Xmin     = 0.055
        Xmax     = 0.5
        N        = np.mean([200, 2000])
        L_repeat = (10**-9) * np.mean([20, 80]) # meters
        n_repeat = 15
        L_omega  = n_repeat * L_repeat
        L_kuhn   = (10**-9) * np.mean([0.5, 3.0]) # meters
        Tmin     = 0.1
        Tmax     = 1
        T        = 1.0
        # **************************************

        # *********** INPUTS ***********
        self.info          = ch.SimInfo()

        if self.isTest:
            self.info.outdir = os.path.join(self.args.output_dir, "rank_"+str(self.comm_rank), 'test')
        else:
            self.info.outdir = os.path.join(self.args.output_dir, "rank_"+str(self.comm_rank), 'episode_' + str(self.episode))

        if self.isTarget:
            self.info.outdir = self.args.target_dir

        self.info.t0       = 0.0
        self.info.nx       = 128
        self.info.ny       = 128
        self.info.dx       = 1./self.info.nx
        self.info.dy       = 1./self.info.ny
        self.info.bc       = 'neumann'
        self.info.rhs_type = 'ch_thermal_no_diffusion'

        # Set up grid for spatial-field quantities
        nx    = int(self.info.nx)
        xx,yy = np.meshgrid( np.arange(0,1,1/self.info.nx), np.arange(0,1,1/self.info.nx) )

        self.chparams             = ch.CHparamsVector( self.info.nx , self.info.ny )

        self.chparams.b           = av.aligned_double_vector(1.0    * np.ones(nx**2))
        self.chparams.u           = av.aligned_double_vector(1.0    * np.ones(nx**2))
        self.chparams.m           = av.aligned_double_vector(0.15   * np.ones(nx**2))
        self.chparams.sigma_noise = 0.0
        self.chparams.eps2_min    = 0.0
        self.chparams.eps2_max    = 1.0
        self.chparams.sigma_min   = 0.0
        self.chparams.sigma_max   = 1.0e10
        self.chparams.T_min       = 0.1
        self.chparams.T_max       = 1.0
        self.chparams.T_const     = av.aligned_double_vector(0.5 * \
                               ( self.chparams.T_max + self.chparams.T_min ) * np.ones(nx**2))
        self.chparams.L_kuhn      = L_kuhn
        self.chparams.N           = N
        self.chparams.L_omega     = L_omega
        self.chparams.X_min       = Xmin
        self.chparams.X_max       = Xmax

        self.chparams.compute_and_set_eps2_and_sigma_from_polymer_params(
               0.5 * (self.chparams.T_max + self.chparams.T_min) , self.info )
        # ******************************

        # Define timescales
        self.biharm_dt    = (self.info.dx**4) / np.max(self.chparams.eps_2)
        self.diff_dt      = (self.info.dx**2) / np.max( [np.max(self.chparams.u) , \
                                                         np.max(self.chparams.b)] )
        self.lin_dt       = 1.0 / np.max(self.chparams.sigma)

        # Setup checkpointing in time
        n_dt              = self.args.length #2000
        n_tsteps          = self.args.steps  #100
        self.info.t0      = 0
        self.info.iter    = 0
        stiff_dt          = np.min([ self.biharm_dt , self.diff_dt , self.lin_dt ])
        self.t            = np.linspace(self.info.t0 , self.info.t0 + n_dt * stiff_dt , n_tsteps+1)
        dt_check          = self.t[1]-self.t[0]

        # Run solver
        if self.args.debug>=1:
            print( 'Biharmonic timescale dt_biharm = ' , self.biharm_dt )
            print( 'Diffusion timescale dt_diff = ' , self.diff_dt , ' = ' , \
                   self.diff_dt/self.biharm_dt , ' dt_biharm')
            print( 'Linear timescale dt_lin = ' , self.lin_dt , ' = ' , \
                   self.lin_dt/self.biharm_dt , ' dt_biharm')
            print( 'Sampling interval = ' , dt_check / stiff_dt , ' dt_stiff' )

    ############################# set the controlling parameters #############################
    def setControlParams(self, T):
        if self.args.debug>=20: print_status("setControlParam T: ".format(np.around(T,2)), comm_rank=self.comm_rank, allranks=True)
        T = self.chparams.T_min if T < self.chparams.T_min else T
        T = self.chparams.T_max if T > self.chparams.T_max else T
        self.chparams.T_const = av.aligned_double_vector(T * np.ones(int(self.info.nx)**2))

    ############################ select one structured vector generator ##########################
    def getImgStruct(self, data, t):
        if   self.args.size_struct_vec==6:   return self.getImgStruct6(data, t)
        elif self.args.size_struct_vec==2:   return self.getImgStruct2(data, t)
        elif self.args.size_struct_vec==200: return self.getFullCircAvgFFT(data, t , \
            interpolation_abscissa = np.linspace(0,2,self.args.size_struct_vec))

    ############################## get Anthony's structured vector ##############################
    def getImgStruct2(self, data, t):

        dimensions         = 2
        structure_function = 'fourier'

        # Named-tuple for handling input options
        Inputs       = namedtuple('Inputs', 'data dimensions structure_function') # output_file
        #datafiletype = datafile.split('.')[-1]
        #datafile     = os.path.join(self.args.output_dir , datafile)
        inputs       = Inputs(data, dimensions, structure_function) #, outfile

        # Compute structure function
        structure_analysis = ImageStructure(inputs)
        structure_metrics  = structure_analysis.compute_structure(
                               plot_metrics=True if not self.args.notPlotRL else False )
                             #, outdir=outdir, str_figure=""

        # Get structured vector
        try:    results = np.array([structure_metrics[0], structure_metrics[1]] )
        except: results = structure_metrics
        #if self.args.debug>=30: print("Struct Vec: ", results)

        return results

    ############################### get Kevin's structured vector ###############################
    def getImgStruct6(self, data, t):

        #datafile = os.path.join(self.args.output_dir , datafile)

        # Define expectations for simulations results
        w, h = self.info.nx, self.info.nx
        x_scale, y_scale = 1, 1 # Conversion of simulation units into realspace units

        # Load simulation output
        #frame   = load_result_file(datafile, w, h)
        frame = np.reshape(data, [w,h])
        results = structure_vector( frame, scale=[x_scale, y_scale],
                    plot=True if not self.args.notPlotRL else False,
                    output_condition=str(t) )

        return results

    ############################# get Kevin's full circ-avg'd fft ##############################
    def getFullCircAvgFFT( self, data, t , interpolation_abscissa = None ):

        N = self.info.nx
        M = self.info.ny

        # Named-tuple for handling input options
        Inputs   = namedtuple('Inputs' , 'data dimensions structure_function output_file nx ny')

        #datafile = os.path.join(self.args.output_dir , datafile)
        #w        = np.genfromtxt( datafile )

        #Ci     = w.reshape([N,M],order='C');
        inputs = Inputs(data, 2, 'fourier_yager_full',
                        os.path.join(self.args.output_dir, 'structure_metrics_2d.dat' ),
                        N , M)

        if self.args.debug>=50: print_status(data, comm_rank=self.comm_rank, allranks=True)
        structure_analysis = image_structure.src.ImageStructure.ImageStructure( inputs )

        if self.isTest:
            out_dir = os.path.join(self.args.output_dir, "rank_"+str(self.comm_rank), 'test/')
        else:
            out_dir = os.path.join(self.args.output_dir, "rank_"+str(self.comm_rank), 'episode_' + str(self.episode) + '/')

        if self.isTarget: out_dir = os.path.join(self.args.target_dir)

        x_ftt , y_fft , lm_result  = structure_analysis.compute_structure(
                                     plot_metrics=False,
                                     outdir=out_dir,
                                     str_figure='circfft_' + str(t) + '_',
                                     interpolation_abscissa=interpolation_abscissa )
                                      # plot_metrics= True if not self.args.notPlotRL else False,

        return y_fft

    ################################ plot target image ################################
    def plotTarget(self, datafile, M=128, N=128):

        data_file = os.path.join(self.args.output_dir , datafile)
        img = np.loadtxt(data_file)
        img = np.reshape(img, [M, N])

        # State plot, colored by time
        fig = plt.figure(1,figsize=(10,10))
        fig.clf()
        ax  = fig.add_subplot(111)
        ax.contourf(img, 30, vmin=-1,vmax=1)

        title = "Target"
        # if the number of structure vector components are too large, we won't print
        if self.args.size_struct_vec<=6:
            title += " structured vector: ["
            for i in range(self.args.size_struct_vec):
                if i<self.args.size_struct_vec-1:
                    title += str(np.around(self.targetStructVec[i],2)) + ", "
                else:
                    title += str(np.around(self.targetStructVec[i],2)) + "]"
        ax.set_title(title)

        fig.savefig(os.path.join(self.args.target_dir, "rank_"+str(self.comm_rank), \
                    self.args.target_file + str(self.args.size_struct_vec) \
                    + ".png"), bbox_inches='tight')

    ################################ plot target image ################################
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return None
