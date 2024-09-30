# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:38:39 2024

@author: kashy
"""

import pandas as pd
import numpy as np

def compute_metrics(confusionMatrix,list_of_N):
    '''
    Input:
    1. ConfusionMatrix: Rows are predicted and columns are true.
    2. List_of_N : For each class, the Total number of pixels.

    Returns
    -------
    Dictionary with all the metrics and values.
    '''
    allMetrics = {} #To be returned
    numberOfClasses = len(confusionMatrix[0])
    assert len(confusionMatrix[0]) == len(confusionMatrix[1]) ,"Confusion Matrix has to be a square."
    total_pixels = sum(list_of_N)
    allMetrics['Overall Accuracy'] = 0 #sum(confusionMatrix[i,i]) would be incorrect, since we want proportionality based on N values.
    accuracy_terms = []
    W = [list_of_N[i]/total_pixels for i in range(numberOfClasses)]
    
    assert sum(W) <1.001 and sum(W)>0.999 ,"Pixel counted area proportions aren't summing upto 1."    
    
    OA = sum(W[i] *confusionMatrix[i,i]/sum(confusionMatrix[i]) for i in range(numberOfClasses))
    #make an assertion that this OA must be equal to the one you get after the loop
    for i in range(numberOfClasses):
        tp = confusionMatrix[i,i]
        total_predicted  = sum(confusionMatrix[i,:])
        predicted_pixels = list_of_N[i]
        W1  = predicted_pixels / total_pixels
        tp_proportional = (tp / (total_predicted)) * W1
        allMetrics ['Overall Accuracy'] += tp_proportional
    #assert allMetrics['Overall Accuracy'] == OA , (allMetrics['Overall Accuracy'], OA)
    
    P_i_i = [W[i] *confusionMatrix[i,i]/sum(confusionMatrix[i]) for i in range(numberOfClasses)]
    assert sum(P_i_i) == OA
    assert OA == allMetrics['Overall Accuracy']
    P_i_dot =[]
    #The denominator term in Ui. Proportional total_predicted of class i.
    for i in range(numberOfClasses):
        P_i_dot.append(sum(W[i] * confusionMatrix[i,j]/sum(confusionMatrix[i])  for j in range(numberOfClasses)))
    U_i = [ P_i_i[i]/P_i_dot[i] for i in range(numberOfClasses)]
    allMetrics['Ui'] = U_i

    
    P_dot_i = []
    #Denominator terms in Producers. Estimated Proportional total_true of class i
    for i in range(numberOfClasses):
        P_dot_i.append(sum(W[j]*confusionMatrix[j,i]/sum(confusionMatrix[j]) for j in range(numberOfClasses)))
    Prod_i = [P_i_i[i]/P_dot_i[i] for i in range(numberOfClasses)]
    allMetrics['Prodi'] = Prod_i
    
    Pij = np.zeros((numberOfClasses,numberOfClasses))
    yh_ref = np.zeros((numberOfClasses,numberOfClasses))
    # Proportional confusion matrix. I would like to assert if users, producers created from this 
    # would be the same
    ni = [sum(confusionMatrix[i]) for i in range(numberOfClasses)]
    for i in range(numberOfClasses):
        for j in range(numberOfClasses):
            Pij[i,j] = W[i] *  confusionMatrix[i,j] / ni[i]
            yh_ref[i,j] = confusionMatrix[i,j] /ni[i]
    
    U_i_backup = [Pij[i,i]/sum(Pij[i,j] for j in range(numberOfClasses)) for i in range(numberOfClasses)]
    Prod_i_backup = [ Pij[i,i]/sum(Pij[j,i] for j in range(numberOfClasses)) for i in range(numberOfClasses)]
    
    allMetrics['PIJ'] = Pij
    allMetrics['Yh_ref for stehman check'] = yh_ref
    assert U_i_backup == U_i
    assert Prod_i == Prod_i_backup
    
    "VARIANCES"
    "1. User's Variance"
    
    VU_i = [ U_i[i] * (1-U_i[i])/sum(confusionMatrix[i]) for i in range(numberOfClasses)]
    allMetrics['User\'s Variance'] = VU_i
    allMetrics['User\'s Standard Error'] = np.sqrt(VU_i)
    allMetrics['User\'s 95% Error'] = 1.96*np.sqrt(VU_i)
    
    "2. Producer's Variance"
    VP_j = [] #I am using the notation that the paper followed.
    # brother ujhhhhhh
    #for producer's variance, you need:
    # Ni. , N.i , Pi, Ui, ni, nij, 
    N_i_dot = list_of_N.copy() #?
    N_dot_i = []
    for j in range(numberOfClasses):
        N_dot_i.append(sum( N_i_dot[i] * confusionMatrix[i,j] / sum(confusionMatrix[i]) for i in range(numberOfClasses)))
    
    exp1 = []
    for j in range(numberOfClasses):
        exp = (N_i_dot[j] **2 * (1 - Prod_i[j])**2 * U_i[j] * (1- U_i[j])) / (sum(confusionMatrix[j]) -1)
        exp1.append(exp)
    exp2 = []
    for j in range(numberOfClasses):
        exp = Prod_i[j] **2 
        subexp = []
        for i in range(numberOfClasses):
            if(i==j):
                continue
            subexp.append( (list_of_N[i]**2 * confusionMatrix[i,j] / sum(confusionMatrix[i])) * 
                            
                         (1 - (confusionMatrix[i,j]/sum(confusionMatrix[i]))) / (sum(confusionMatrix[i]) - 1)
                         )
        exp = exp * sum(subexp)
        exp2.append(exp)
    allMetrics['Prod exp N_dot_i']= N_dot_i
    allMetrics['Prod exp 1'] = exp1
    allMetrics['Prod exp 2'] = exp2
    VP_j = [(1/(N_dot_i[j]**2) )*(exp1[j] +exp2[j] ) for j in range(numberOfClasses)]
    allMetrics['Producer\'s Variance'] = VP_j
    allMetrics['Produser\'s Standard Error'] = np.sqrt(VP_j)
    allMetrics['Produser\'s 95% Error'] = 1.96*np.sqrt(VP_j)

    "3. Overall"

    VOA = sum(W[i]**2 * VU_i[i] for i in range(numberOfClasses))
    allMetrics['Overall accuracy Variance'] = VOA
    allMetrics['Accuracy std error'] = np.sqrt(VOA)
    allMetrics['Accuracy 95% Error'] = 1.96*np.sqrt(VOA)
    
    "4. Areas"
    Areas = [  sum(Pij[:,i]) * total_pixels for i in range(numberOfClasses)]
    allMetrics['Areas'] = Areas
    AreasSTDERR = []
    for k in range(numberOfClasses):
        AreasSTDERR.append(np.sqrt(sum( ((W[i]*Pij[i,k])-(Pij[i,k]**2))/(sum(confusionMatrix[i])-1) \
                                  for i in range(numberOfClasses)   )))
    
        
    allMetrics['Area Standard Error'] = AreasSTDERR
    Areasstdtot = [total_pixels*AreasSTDERR[i] for i in range(numberOfClasses)]
    allMetrics['Areas STD error to total Area'] = Areasstdtot
    
    return allMetrics
    
        
    
if __name__ == '__main__':
    cm = np.array([
    [309 , 1 , 5 , 0, 3],
    [7, 49 , 10 ,1 , 14],
    [6,  2 , 105, 1, 0 ],
    [5 , 2 , 2 , 141 , 2],
    [18 , 17, 11, 0, 189]
    ])
    
    cm = cm.T # Resembling Olofsson Paper
    all_pixels = [26029895.04 ,  968635.66 ,  6296973.80 ,  1339561.92,  14663367.14 ]
    list_of_N = all_pixels
    metrics = compute_metrics(cm,list_of_N)



