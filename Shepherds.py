# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:04:50 2019

@author: Lab User
"""

import numpy as np
import copy


class Shepherds:
    
    ''' INPUTS
    
    % PaddockLength             = Width and Height of the environment 
    % NumberOfShepherds         = Number of shepherds
    % ShepherdShepherdAvoidanceRadius = the distance within which the repulsion forces between shepherds exist
    % ShepherdStep              = Displacement per time step
    % NoiseLevel                = Relative strength of angular noise
    
    % Obstacles                 = Matrix of positions and size of obstacles
    
    % OUTPUT
    % ShepherdUpdatedMatrix     = Updated shepherd object population matrix'''

    def __init__(
            self,
            PaddockLength,
            NumberOfShepherds,
            ShepherdShepherdAvoidanceRadius,            
            ShepherdStep,
            NoiseLevel,
            Obstacles,
            WeightShepherdRepellFromShepherd,
            AvoidanceRadius=None,
            AvoidanceWeight=None
          ):

        self.PaddockLength = PaddockLength
        self.NumberOfShepherds = NumberOfShepherds
        self.ShepherdShepherdAvoidanceRadius = ShepherdShepherdAvoidanceRadius
        self.ShepherdStep = ShepherdStep
        self.NoiseLevel = NoiseLevel
        self.Obstacles = Obstacles
        self.WeightShepherdRepellFromShepherd = WeightShepherdRepellFromShepherd
        self.AvoidanceRadius = AvoidanceRadius
        self.AvoidanceWeight = AvoidanceWeight          
        self.ShepherdMatrix = None
        
    def v_len(self, v):
        return np.sqrt(v[0]**2 + v[1]**2)
    
    
    def dist_p2p(self, p1,p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    
    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v/norm
        else:
            return np.array([0,0])

        
    def angle_vectors(self, u,v):
        return np.arccos(np.dot(u,v)/(self.v_len(u)*self.v_len(v)))
    
    
    def perpendicular_vector_pair(self, v):
        if v[1] == 0:
            nv1 = [0,-1]
            nv2 = [0,1]
        else:
            v1 = [1, -v[0]/v[1]]
            v2 = [-1, v[0]/v[1]]
            nv1 = self.normalize(v1)
            nv2 = self.normalize(v2)
        return (nv1, nv2)

    
    def computePointsOnCircle(self, currentPosition, SubGoal, v0, angle):    
        rotationMatrix1 = np.array(( (np.cos(angle), -np.sin(angle)),
                                   (np.sin(angle),  np.cos(angle)) ))
        rotationMatrix2 = np.array(( (np.cos(-angle), -np.sin(-angle)),
                                   (np.sin(-angle),  np.cos(-angle)) ))
        v1 = np.matmul(rotationMatrix1, v0)
        v2 = np.matmul(rotationMatrix2, v0)
        p1 = currentPosition + v1
        p2 = currentPosition + v2
        if self.dist_p2p(p1, SubGoal) < self.dist_p2p(p2, SubGoal):
            return p1
        else:
            return p2


    def compute_shepherd2shepherd_repulsion_regulator(self, distance):
        M_r = self.ShepherdStep*np.exp(-2*distance/self.ShepherdShepherdAvoidanceRadius)
        return M_r
            
        
    def compute_obstacle_avoidance_regulator(self, distance):
        M_a = self.ShepherdStep*np.exp(-2*distance/self.AvoidanceRadius)
        return M_a


    def compute_shepherd2shepherd_repulsion(self, ShepherdIndex, ShepherdPosition):
        
        shepherd2shepherd_repulsion = np.array([0,0])
        for j in range(self.NumberOfShepherds):        
            if (j != ShepherdIndex):
                distance = self.dist_p2p(ShepherdPosition, self.ShepherdMatrix[j,0:2])
                if distance <= self.ShepherdShepherdAvoidanceRadius:
                    force = self.normalize(ShepherdPosition - self.ShepherdMatrix[j,0:2])
                    M_r = self.compute_shepherd2shepherd_repulsion_regulator(distance)
                    shepherd2shepherd_repulsion = shepherd2shepherd_repulsion + M_r*force
            
        return self.normalize(shepherd2shepherd_repulsion)   
        

    def obstacle_avoidance(self, shepherdPosition):
        avoidance_force = np.array([0,0])
        for i in range(len(self.Obstacles)):
            distance = self.dist_p2p(shepherdPosition, self.Obstacles[i,0:2]) - self.Obstacles[i,2]*np.sqrt(2) # for square obstacle
            if distance <= self.AvoidanceRadius:
                v_shepherd2obstacle = [self.Obstacles[i,0] - shepherdPosition[0], self.Obstacles[i,1] - shepherdPosition[1]]
                nv1, nv2 = self.perpendicular_vector_pair(v_shepherd2obstacle)
                angle1 = self.angle_vectors(v_shepherd2obstacle, nv1)
                angle2 = self.angle_vectors(v_shepherd2obstacle, nv2)
                M_a = self.compute_obstacle_avoidance_regulator(distance)
                if angle1 < angle2:
                    avoidance_force = avoidance_force + M_a * np.array(nv1)
                else:
                    avoidance_force = avoidance_force + M_a * np.array(nv2)
        return self.normalize(avoidance_force)
    
    
    def obstacle_collision(self, shepherdPosition):
        collision = False
        for i in range(len(self.Obstacles)):
            if self.dist_p2p(shepherdPosition, self.Obstacles[i,0:2]) <= self.Obstacles[i,2]:
                collision = True
                break
        return collision
    
    
    def computeDetour(self, currentPosition, SheepGCM, SubGoal, ViolationDistance):
        d = self.dist_p2p(currentPosition, SheepGCM)
        cos_delta = (d**2 + self.ShepherdStep**2 - ViolationDistance**2)/(2*d*self.ShepherdStep)
        cos_delta = np.clip(cos_delta,-1,1)                                                                             
        angle = np.arccos(cos_delta)
        v = SheepGCM - currentPosition
        v0 = self.normalize(v) * self.ShepherdStep
        
        return self.computePointsOnCircle(currentPosition, SubGoal, v0, angle)


    def update(self, ShepherdMatrix, SheepGCM, SubGoal, ViolationDistance):
        self.ShepherdMatrix = ShepherdMatrix
        ShepherdUpdatedMatrix = copy.deepcopy(self.ShepherdMatrix)        
        for TheShepherd in range(self.NumberOfShepherds): # Go through every shepherd object.         
            currentPosition = self.ShepherdMatrix[TheShepherd,0:2]
            # Find direction from the shepherd to the subgoal point
            NormalisedDirection = self.normalize(SubGoal[TheShepherd] - currentPosition)
            
            # Find the shepherd-shepherd repulsion force:
            Shepherd2ShepherdForce = self.compute_shepherd2shepherd_repulsion(TheShepherd,currentPosition)
            
            ## Find the obstacle avoidance force
            if self.AvoidanceRadius is not None: # The case of ground shepherd 
                if self.Obstacles is None:
                    AvoidanceForce = np.array([0,0])
                else:
                    AvoidanceForce = self.obstacle_avoidance(currentPosition)
            # Find the total force
                CumulativeForce = NormalisedDirection \
                                + self.AvoidanceWeight*AvoidanceForce \
                                + self.WeightShepherdRepellFromShepherd * Shepherd2ShepherdForce \
                                + self.NoiseLevel*np.clip(np.random.randn(2),-1,1) 
                                
            else: # sky shepherd (UAV)
                CumulativeForce = NormalisedDirection \
                                + self.WeightShepherdRepellFromShepherd * Shepherd2ShepherdForce \
                                + self.NoiseLevel*np.clip(np.random.randn(2),-1,1) 

            # Find the normalised total force
            NormalisedForce = self.normalize(CumulativeForce)
            # Compute the next position of the shepherd
            NewPosition = currentPosition + NormalisedForce * self.ShepherdStep
            # Check if the next position violate the minimum distance to the sheep GCM
            # If YES, compute detour:
            if self.dist_p2p(NewPosition, SheepGCM) < ViolationDistance[TheShepherd]:
                if self.dist_p2p(currentPosition, SubGoal[TheShepherd]) <= self.ShepherdStep:
                    NewPosition = SubGoal[TheShepherd]
                else:
                    NewPosition = self.computeDetour(currentPosition, SheepGCM, SubGoal[TheShepherd], ViolationDistance[TheShepherd])
            
            if self.AvoidanceRadius is not None: # The case of ground shepherd            
                ## If shepherd next position collide with obstacles, then do not move:
                if self.Obstacles is not None:
                    if self.obstacle_collision(NewPosition):
                        ShepherdUpdatedMatrix[TheShepherd,:] = self.ShepherdMatrix[TheShepherd,:]
                    else:
                        ShepherdUpdatedMatrix[TheShepherd,0:2] = NewPosition
                else:
                    ShepherdUpdatedMatrix[TheShepherd,0:2] = NewPosition
            else: # sky sherpherd
                ShepherdUpdatedMatrix[TheShepherd,0:2] = NewPosition
            
            ## Limit the movement inside the paddock:                  
            if (ShepherdUpdatedMatrix[TheShepherd,0] < 0):
                ShepherdUpdatedMatrix[TheShepherd,0] = 0
                  
            if (ShepherdUpdatedMatrix[TheShepherd,1] < 0):
                ShepherdUpdatedMatrix[TheShepherd,1] = 0
                    
            if (ShepherdUpdatedMatrix[TheShepherd,0] > self.PaddockLength):
                ShepherdUpdatedMatrix[TheShepherd,0] = self.PaddockLength
                  
            if (ShepherdUpdatedMatrix[TheShepherd,1] > self.PaddockLength):
                ShepherdUpdatedMatrix[TheShepherd,1] = self.PaddockLength
            
        return ShepherdUpdatedMatrix

      

      
          