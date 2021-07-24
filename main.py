# -*- coding: utf-8 -*-
"""
Created on Jan 21 10:10:05 2019

@author: Lab User
"""

from Environment import Environment
import numpy as np


N_EPISODE = 50000
MAX_STEP = 1000

NumberOfSheep = 30
NumberOfShepherds = 3
Obstacles = None
# Obstacles = np.load('Variables/Obstacles3.npy')

# Initialize environment
env = Environment(NumberOfSheep,NumberOfShepherds,Obstacles)

for episode_i in range(N_EPISODE):
    
    env.reset()
    for step in range(MAX_STEP):
        
        # update the position        
        terminate = env.step()
        
        # View environment
        env.view()
                                            
        if (np.abs(terminate)):
            break
