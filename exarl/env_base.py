# © (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
# Department of Energy/National Nuclear Security Administration. All rights in the program are
# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting on its behalf a
# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
# derivative works, distribute copies to the public, perform publicly and display publicly, and 
# to permit others to do so.


import json, os, sys
from abc import ABC, abstractmethod

class ExaEnv(ABC):
    def __init__(self, **kwargs):

        # Locationn to save results
        # Need to add MPI subdirectories
        self.results_dir = ''
        
        # Use relative path not absolute
        self.base_dir = os.path.dirname(__file__)
        print(self.base_dir)
        
    def set_results_dir(self,results_dir):
        # Default method to save environment specific information 
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # Top level directory 
        self.results_dir=results_dir

    def set_config(self, env_data):
        self.env_data = env_data
        # Add any MPI parameters                                                                                       
        self.mpi_children_per_parent = int(env_data['mpi_children_per_parent']) if 'mpi_children_per_parent' in env_data.keys() else 0    

        # Add any OMP parameters
        self.omp_thread = int(env_data['omp_thread']) if 'omp_thread' in env_data.keys() else 1                                           
        
        # Add any GPU parameters                                                                                                   

        # Executable                                                                                                               
        if(self.mpi_children_per_parent > 0):
            # Defaults to running toy example of computing PI                                                                             
            self.worker = (env_data['worker_app']).lower() if 'worker_app' in env_data.keys() else "envs/env_vault/cpi.py"
        else:
            self.worker = None

    def get_config(self):
        return self.env_data
        
    @abstractmethod
    def step(self, action):
        ''' 
        Required by all environment to be implemented by user 
        '''
        pass

    @abstractmethod
    def reset(self):
        ''' 
        Required by all environment to be implemented by user 
        '''
        pass

