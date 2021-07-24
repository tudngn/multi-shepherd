# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:28:32 2019

@author: Lab User
"""

import numpy as np
from Sheep import Sheep
from Shepherds import Shepherds
import Sub_Env
from findFurthestSheep import findFurthestSheep
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Environment: 
    
    def __init__(
            self,
            NumberOfSheep,
            NumberOfShepherds,
            Obstacles = None
            ):
        
      self.NumberOfSheep = NumberOfSheep
      self.NumberOfShepherds = NumberOfShepherds
      # Constants
      self.PaddockLength=80
      self.SheepSensingOfShepherdRadius = 65      
      self.SheepSensingOfSheepRadius = 5      
      self.SheepSheepAvoidanceRadius = 0.5
      #self.ShepherdSensingOfShepherdRadius = 5
      self.ShepherdShepherdAvoidanceRadius = 2
      
      self.WeightRepellFromOtherSheep = 2 # p_0
      self.WeightAttractionToLCM = 1.05 # c
      self.WeightRepellFromShepherd = 1# p_s
      self.WeightShepherdRepellFromShepherd = 0.5# p_s
      self.WeightOfInertia = 0.5 # h
      self.NoiseLevel = 0.3 #e
      self.SheepStep = 1 # delta
      self.ShepherdStep = 3 # delta_s
      
      self.MinDistanceShepherdSheep = 4
      self.InfluenceDistance = 14
      
      self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs = 10
      # self.DogViolatingDistanceToSheep = 3
      # Because the original algorithm is designed for a single shepherd, we need
      # to add a parameter for repulsion among shepherds
      # ShepherdRadius            = Relative strength of repulsion from other shepherds
      # self.ShepherdRadius            = 10 # = 65 in the IJCNN paper with 300x300m environment
      ## Initialise Sheep in the top right quarter of the paddock
      self.Obstacles = Obstacles
      self.AvoidanceRadius = 3 #5
      self.SheepAvoidanceWeight = 2
      #self.ShepherdAvoidanceWeight = 3
      
      self.State = None
      self.CMAL_ALL = None        
      self.SheepGlobalCentreOfMass = None
      self.ShepherdGlobalCentreOfMass = None
      self.IndexOfFurthestSheep = None
      self.AreFurthestSheepCollected = None
      self.initial_dist = None
      self.Pre_dist_Target = None
      self.iReset = 0  
      self.SubGoal = None
      self.ViolationDistance = None
      
      self.MaximumSheepDistanceToGlobalCentreOfMass = self.SheepSheepAvoidanceRadius * (self.NumberOfSheep**(2/3)) # self.SheepSheepAvoidanceRadius * np.sqrt(2*self.NumberOfSheep) #  
      self.dist_driving = self.MaximumSheepDistanceToGlobalCentreOfMass + self.MinDistanceShepherdSheep
      
      # Initialize the computation class for shepherd:
      self.ShepherdUpdater = Shepherds(self.PaddockLength,
                                       self.NumberOfShepherds,
                                       self.ShepherdShepherdAvoidanceRadius,                                   
                                       self.ShepherdStep,
                                       self.NoiseLevel,
                                       self.Obstacles,
                                       self.WeightShepherdRepellFromShepherd)
      
      # Initialize the computation class for sheep:
      self.SheepUpdater = Sheep(self.PaddockLength,
                                self.NumberOfSheep,
                                self.NumberOfShepherds,
                                self.SheepSheepAvoidanceRadius,
                                self.SheepSensingOfSheepRadius,
                                self.InfluenceDistance,
                                self.MinDistanceShepherdSheep,
                                self.SheepStep,
                                self.ShepherdStep,
                                self.Obstacles,
                                self.AvoidanceRadius,
                                self.WeightOfInertia,
                                self.WeightRepellFromOtherSheep,
                                self.WeightRepellFromShepherd,
                                self.WeightAttractionToLCM,
                                self.SheepAvoidanceWeight,
                                self.NoiseLevel)
      

    def reset(self):
      self.SheepMatrix, self.ShepherdMatrix, self.TargetCoordinate = Sub_Env.create_Env(self.PaddockLength,
                                                                                         self.NumberOfShepherds,
                                                                                         self.NumberOfSheep,
                                                                                         self.MaximumSheepDistanceToGlobalCentreOfMass)
      self.CMAL_ALL = np.zeros([self.NumberOfSheep,8])

      #Calculating Initial State
      self.SheepGlobalCentreOfMass= np.array([np.mean(self.SheepMatrix[:,0]),np.mean(self.SheepMatrix[:,1])]) # GCM of sheep objects
      self.ShepherdGlobalCentreOfMass=np.array([np.mean(self.ShepherdMatrix[:,0]),np.mean(self.ShepherdMatrix[:,1])]) # GCM of shepherd objects
      self.IndexOfFurthestSheep, self.AreFurthestSheepCollected = findFurthestSheep(self.SheepMatrix,
                                                                          self.SheepGlobalCentreOfMass,
                                                                          self.TargetCoordinate,
                                                                          self.NumberOfShepherds,
                                                                          self.MaximumSheepDistanceToGlobalCentreOfMass)     
      
      print("Environment reset.")


    def step(self, Action=None):
    
      # Compute new Subgoal Point and the Violation Distance
      if self.AreFurthestSheepCollected == 0:
          self.SubGoal, Distance_GCM_FurthestSheep = self.collecting_behaviour()
          self.ViolationDistance = Distance_GCM_FurthestSheep + self.MinDistanceShepherdSheep
      else:
          #self.SubGoal = self.driving_behaviour(Action)
          self.SubGoal = np.tile(self.driving_behaviour_Strombom(), (self.NumberOfShepherds,1))
          self.ViolationDistance = [self.dist_driving]*self.NumberOfShepherds
          

      self.ShepherdMatrix = self.ShepherdUpdater.update(self.ShepherdMatrix, self.SheepGlobalCentreOfMass, self.SubGoal, self.ViolationDistance)

      #Performing action herding sheeps
      self.SheepMatrix, Collision = self.SheepUpdater.update(self.SheepMatrix, self.ShepherdMatrix) # Update position of sheep
            
      # Recompute Sheep Centre of mass
      self.SheepGlobalCentreOfMass = np.array([np.mean(self.SheepMatrix[:,0]),np.mean(self.SheepMatrix[:,1])])  # GCM of sheep objects
      self.ShepherdGlobalCentreOfMass = np.array([np.mean(self.ShepherdMatrix[:,0]),np.mean(self.ShepherdMatrix[:,1])]) # GCM of shepherd objects

      self.IndexOfFurthestSheep, self.AreFurthestSheepCollected = findFurthestSheep(self.SheepMatrix,
                                                                          self.SheepGlobalCentreOfMass,
                                                                          self.TargetCoordinate,
                                                                          self.NumberOfShepherds,
                                                                          self.MaximumSheepDistanceToGlobalCentreOfMass)
      #Checking terminate
      Terminate =  self.check_terminate(Collision)    
      
      #Saving data
      return Terminate   
             

    def check_terminate(self, Collision):
      ##pause(10)
      ### Termination Conditions
      terminate = 0 #Not achieving the target
            
      SheepGlobalCentreOfMassDistanceToTarget = np.sqrt((self.SheepGlobalCentreOfMass[0]-self.TargetCoordinate[0])**2+
                                                        (self.SheepGlobalCentreOfMass[1]-self.TargetCoordinate[1])**2) # Distance GCM sheep to target
      if Collision:
          terminate = -1
      
      if (SheepGlobalCentreOfMassDistanceToTarget < self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs) and self.AreFurthestSheepCollected:
          terminate = 1 #Achieving the target       
        
      return terminate
      
        
    def collecting_behaviour(self):
      CS_FS_x = self.SheepMatrix[self.IndexOfFurthestSheep,0] - self.SheepGlobalCentreOfMass[0] # CS to FS x
      CS_FS_y = self.SheepMatrix[self.IndexOfFurthestSheep,1] - self.SheepGlobalCentreOfMass[1] # CS to FS y
  
      DirectionFromGlobalCentreOfMassToFurthestSheep = np.array([CS_FS_x, CS_FS_y])
      NormOfDirectionFromGlobalCentreOfMassToFurthestSheep = np.sqrt(DirectionFromGlobalCentreOfMassToFurthestSheep[0]**2 + DirectionFromGlobalCentreOfMassToFurthestSheep[1]**2)
      NormalisedDirectionFromGlobalCentreOfMassToFurthestSheep = DirectionFromGlobalCentreOfMassToFurthestSheep / NormOfDirectionFromGlobalCentreOfMassToFurthestSheep
      DirectionFromGlobalCentreOfMassToPositionBehindFurthestSheep = NormalisedDirectionFromGlobalCentreOfMassToFurthestSheep*(NormOfDirectionFromGlobalCentreOfMassToFurthestSheep + self.MinDistanceShepherdSheep)
      PositionBehindFurthestSheep = self.SheepGlobalCentreOfMass + DirectionFromGlobalCentreOfMassToPositionBehindFurthestSheep.T
      
      return (PositionBehindFurthestSheep, NormOfDirectionFromGlobalCentreOfMassToFurthestSheep)
  
    
    def driving_behaviour_Strombom(self):
      DirectionFromTargetToGlobalCentreOfMass = np.array([self.SheepGlobalCentreOfMass[0]-self.TargetCoordinate[0], self.SheepGlobalCentreOfMass[1]-self.TargetCoordinate[1]])
      NormOfDirectionFromTargetToGlobalCentreOfMass = np.sqrt(DirectionFromTargetToGlobalCentreOfMass[0]**2 + DirectionFromTargetToGlobalCentreOfMass[1]**2)
      NormalisedDirectionFromTargetToGlobalCentreOfMass = DirectionFromTargetToGlobalCentreOfMass / NormOfDirectionFromTargetToGlobalCentreOfMass
      PositionBehindCenterFromTarget = self.TargetCoordinate + NormalisedDirectionFromTargetToGlobalCentreOfMass * (NormOfDirectionFromTargetToGlobalCentreOfMass + self.dist_driving)      
      return PositionBehindCenterFromTarget


    def view(self):
        # Plotting---------------------------------------------------------
       self.SheepGlobalCentreOfMass = np.array([np.mean(self.SheepMatrix[:,0]),np.mean(self.SheepMatrix[:,1])])  # GCM of sheep objects
       self.ShepherdGlobalCentreOfMass = np.array([np.mean(self.ShepherdMatrix[:,0]),np.mean(self.ShepherdMatrix[:,1])]) # GCM of shepherd objects

       fHandler = plt.figure(1)
       #fHandler.OuterPosition = [80 80 800 800];
       fHandler.Color = 'white'
       fHandler.MenuBar = 'none'
       fHandler.ToolBar = 'none'      
       fHandler.NumberTitle = 'off'
       ax = fHandler.gca()
       ax.cla() # clear things for fresh plot

        # change default range 
       ax.set_xlim((-self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs, self.PaddockLength+self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs))
       ax.set_ylim((-self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs, self.PaddockLength+self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs))
       #plt.hold(True)
       ax.plot(self.SheepMatrix[:,0],self.SheepMatrix[:,1],'k.',markersize=10) # plot sheeps 
       ax.plot(self.ShepherdMatrix[:,0],self.ShepherdMatrix[:,1],'b*',markersize=10)# plot shepherd
       ax.plot(self.SheepGlobalCentreOfMass[0],self.SheepGlobalCentreOfMass[1],'rs',markersize=10) # plot GCM of sheep
       ax.plot(self.ShepherdGlobalCentreOfMass[0],self.ShepherdGlobalCentreOfMass[1],'rd',markersize=10) # plot GCM of shepherds
       ax.plot(self.TargetCoordinate[0],self.TargetCoordinate[1],'gp',markersize=10) # plot target point
       circle=plt.Circle([self.TargetCoordinate[0],self.TargetCoordinate[1]],self.StopWhenSheepGlobalCentreOfMassDistanceToTargetIs,color='r')
       ax.add_artist(circle)
       if self.Obstacles is not None:
           for i in range(len(self.Obstacles)):
               rectangle = Rectangle((self.Obstacles[i,0] - self.Obstacles[i,2], self.Obstacles[i,1] - self.Obstacles[i,2]), 2*self.Obstacles[i,2], 2*self.Obstacles[i,2], color='k', fill=True)
               ax.add_patch(rectangle)
       ax.plot([0,0],[0,self.PaddockLength],'b-')
       ax.plot([0,self.PaddockLength],[0,0],'b-')
       ax.plot([self.PaddockLength,0],[self.PaddockLength,self.PaddockLength],'b-')
       ax.plot([self.PaddockLength,self.PaddockLength],[self.PaddockLength,0],'b-')
       ## Visualise Attraction Repulsion Directions
       ax.quiver(self.SheepMatrix[:,0],self.SheepMatrix[:,1],self.CMAL_ALL[:,0],self.CMAL_ALL[:,1],color = 'r')# Direction of repulsion from other sheep
       ax.quiver(self.SheepMatrix[:,0],self.SheepMatrix[:,1],self.CMAL_ALL[:,2],self.CMAL_ALL[:,3],color = 'b')# Direction of repulsion from shepherds
       ax.quiver(self.SheepMatrix[:,0],self.SheepMatrix[:,1],self.CMAL_ALL[:,4],self.CMAL_ALL[:,5], color = 'g')# Direction of attraction to other sheep.
       
       plt.xlabel('Paddock Length')
       plt.ylabel('Paddock Height')
       #plt.hold(False)
       plt.show()
       #Sys.sleep(0.05)
      #dev.off()


