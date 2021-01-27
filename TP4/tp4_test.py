#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time
from marking_tools import *

from tp4_sujet import *

     
with open('tp4_hough.pkl', 'rb') as f:
    _points,\
    _scores_brute_force,\
    _scores_hough,\
    _peak_rhos,peak_thetas,\
    _scores_smoothed_brute_force,\
    _scores_hough_smooth,\
    _peak_rhos_smooth,\
    _peak_thetas_smooth= pickle.load(f, encoding='iso-8859-1')


    
def test():
    m=marking()

    n=100
    sigma=0.03
    random=xorshift()# use custom pseudo reandom to get repeatable results
    points1=random.rand(n,1)*np.array([0,4])+np.array([2,0])+random.normal(0,sigma,n,2)
    points2=random.rand(n,1)*np.array([4,4])+np.array([0,0])+random.normal(0,sigma,n,2)
    points3=random.rand(n,2)*np.array([4,4])
    
    points=np.row_stack((points1,points2,points3))

    min_theta=0
    max_theta=np.pi
    max_rho=np.max(np.linalg.norm(points,axis=1))
    min_rho=-max_rho
    
    nb_rhos=100
    nb_thetas=150  
    tau=10*sigma
 
    
    # Question 1

    d=distancePointsLine(points[1:5,:], 2,0.3)
    t=timer(lambda: distancePointsLine(points, 2,0.3),nb=100)
    m.add(1, d, [ 0.02298727,  0.60614286,  0.87389925,  0.04036134],duration=t,max_duration=0.05)  
   
       
    # Question 2
    c=countInliersLine(points,  2,0.3,tau) 
    t=timer(lambda: countInliersLine(points,  2,0.3,tau) ,nb=100)
    m.add(1,c,68,duration=t,max_duration=0.05)
   
    
    # Question 3
    v=h(np.linspace(0,1,11),0.5) 
    m.add(1,v,[ 1. , 0.884736,  0.592704,  0.262144, 0.046656,0, 0, 0, 0,0, 0])
   
    
    # Question 4
    s=smoothLineScore(points, 2,0.3,tau) 
    m.add(1,s,35.398607765521561)
    
    
    # Question 5
    r=getRhos(points[1:5,:],0.3)
    m.add(1,r,[ 2.02298727,  2.60614286,  2.87389925,  2.04036134])
    

    # Question 6
    scores_brute_force=scoresBruteForce(points,tau,nb_thetas,min_rho,max_rho,nb_rhos)
    m.add(1,scores_brute_force,_scores_brute_force)
    
           
    # Question 7
    scores_hough=scoresHough(points,tau,nb_thetas,min_rho,max_rho,nb_rhos)  
    m.add(2,scores_hough.astype(np. float16),_scores_hough)
        
    # Question 8
    scores_smoothed_brute_force=scoresSmoothBruteForce(points,tau,nb_thetas,min_rho,max_rho,nb_rhos)
    m.add(1,scores_smoothed_brute_force.astype(np.float16),_scores_smoothed_brute_force)
   
        
    # Question 9
    scores_hough_smooth=scoresHoughSmooth(points,tau,nb_thetas,min_rho,max_rho,nb_rhos)
    m.add(1,scores_hough_smooth.astype(np.float16),_scores_hough_smooth)
    
    
    
    
   
    m.print_note()
    
    
if __name__ == "__main__":
    test()
