
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time

class marking():
    def __init__(self):
        self.marks=[]  
        
    def add(self,coef,student_solution,exact_solution,duration=None,max_duration=None,relative_coef_duration=0.5,tol=1e-8,nbdiff_max=0,isimage=False):
        if student_solution is None:
            valid=False
            print('not answered')
        else:
            try:
                if isinstance(exact_solution,list):
                    exact_solution=np.array(exact_solution)
                if isinstance(student_solution,list):  
                    student_solution=np.array(student_solution)
                if isinstance(student_solution, np.ndarray):
                    student_solution=student_solution.astype(exact_solution.dtype) 
                diff=np.abs( student_solution - exact_solution)
                max_error=np.max(diff)
                nbdiff=np.sum(diff>tol)
                valid= (nbdiff<=nbdiff_max )
                if nbdiff>0 and valid:
                    print('WARNING : found '+str(nbdiff)+ ' differences , but still accepted')
                if not(valid):
                    print('found ' +str(nbdiff) +' values with difference with exact solution is  greater that '+str(tol))
                    print('the greatest error is '+str(max_error))
                    if nbdiff_max>0:
                        print('expect less than '+str(nbdiff_max)+' differences')
                    
                    cov_mat = np.cov([exact_solution.flatten(),student_solution.flatten()])
                    if np.min(np.linalg.eig(cov_mat)[0])<1e-4:
                        print('seem like there is a affine relationship between your data and the expected ones')
                        plt.plot(exact_solution.flatten(),student_solution.flatten())
                    
                    if isimage: 
                        plt.figure()
                        plt.subplot(1,3,1)
                        plt.imshow(student_solution)
                        plt.subplot(1,3,2)
                        plt.imshow(exact_solution)
                        plt.subplot(1,3,3)
                        plt.imshow(np.abs(exact_solution-student_solution))  
                        plt.show()
            except:
                valid=0
                print('Unexpected error:', sys.exc_info()[0],sys.exc_info()[1])

            if duration is not None and max_duration is not None:
                if duration>max_duration:
                    print('executation time ('+str(duration)+') expected to be less than ' + str(max_duration))
                    valid=valid*relative_coef_duration
        
        self.marks.append({'coef':coef,'valid':valid})
        print('question '+str(len(self.marks))+": "+str(valid*coef)+"/"+str(coef))
        
    def print_note(self):
        note=np.sum([m['coef']*m['valid'] for m in self.marks])
        total=np.sum([m['coef'] for m in self.marks])
        print('note= '+str(note)+'/'+str(total))
        return note
        
def timer(f,nb=1):    
    start = time.time()
    for i in range(nb):
        f()
    end = time.time() 
    return end-start 

class chrono():
    def __init__(self):
        pass
    def tic(self):
        self.start=time.clock() 
    def toc(self):
        return time.clock()-self.start 
