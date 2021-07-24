# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:34:10 2019

@author: Lab User
"""

import numpy as np
#import scipy.io



def create_Env(PaddockLength,NumberOfShepherds,NumberOfSheep,MaximumSheepDistanceToGlobalCentreOfMass):
        
    """Create the sheep matrix, shepherd matrix and target"""
    TargetCoordinate = np.array([0,0]) # Target coordinates
    # Spawn a random global center of mass near the center of the field    
    SheepMatrix = np.zeros([NumberOfSheep,5]) # initial population of sheep **A matrix of size OBJECTSx5
    
    #create the shepherd matrix
    ShepherdMatrix = np.zeros([NumberOfShepherds,5])  # initial population of shepherds **A matrix of size OBJECTSx5   
    
    SheepMatrix[:,[0,1]] = np.random.uniform(PaddockLength*1/3,PaddockLength*2/3,[NumberOfSheep,2])
    
    #Initialise Sheep Initial Directions Angle [-pi,pi]
    SheepMatrix[:,2] = np.pi - np.random.rand(len(SheepMatrix[:,2]))*2*np.pi #1 - because just having one column
    #Add the index of each sheep into the matrix
    SheepMatrix[:,4] = np.arange(0,len(SheepMatrix[:,4]),1)
    
    # Initialise Shepherd in the lower left corner
    ShepherdMatrix[:,0:2] = TargetCoordinate

    #Initialise Sheep Initial Directions Angle [-pi,pi]
    SheepMatrix[:,2] = np.pi - np.random.randn(len(SheepMatrix[:,2]))*2*np.pi #1 - because just having one column

    #Add the index of each sheep into the matrix
    SheepMatrix[:,4] = np.arange(0,len(SheepMatrix[:,4]),1)
    #Initialise Shepherds Initial Directions Angle [-pi,pi]
    ShepherdMatrix[:,2]= np.pi - np.random.randn(NumberOfShepherds)*2*np.pi

    #Add the index of each shepherd into the matrix
    ShepherdMatrix[:,4] = np.arange(0,len(ShepherdMatrix[:,4]),1)
      
    return(SheepMatrix,ShepherdMatrix,TargetCoordinate)