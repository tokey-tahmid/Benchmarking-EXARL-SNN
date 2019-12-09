# Reinforcement Learning environments and agents/policies used for the Design and Control applications

## Software Requirement
* Python 3.7 
* The environemnt framework is built of [OpenAI Gym](https://gym.openai.com) 
* Additional python packages are defined in the setup.py 
* For now, we assumes you are running at the top directory 

## Directory Organization
```
├── setup.py                          : Python setup file with requirements files 
├── scripts                           : a folder contains RL steering scripts
├── agents                            : a folder contains agent codes
└── exa_gym                           : a folder containing the ExaLearn environments
    └── __init__.py                   : the list of environments registered with gym
    └── core                          : a folder containing core code (MPI, base class) for the ExaLean environments
    └── cfg                           : a folder that contains the default envinment configurations   
    └── envs                          : a folder containing the ExaLearn environments
├── cfg                               : a folder contains agent and environement configurations
├── utils                             : a folder contains utilities
          
```

## Installing 
* Pull code from repo
```
git clone https://github.com/exalearn/ExaRL.git
```
* Install ExaRL (via pip):
```
cd ExaRL
pip install -e . --user
```