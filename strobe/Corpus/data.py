# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 09:48:07 2014

@author: Ruben Baetens
"""

import os
import numpy as np

import pathlib
strobepath = pathlib.Path(__file__).parent.parent.resolve()
datapath = os.path.join(strobepath,'Data')

def get_probability(rnd, prob, p_type='cum'):
    '''
    Find the x-value in a given comulative probability 'prob_cum' based on a
    given random y-value 'rnd'.
    '''
    if p_type != 'cum':
        prob = np.cumsum(prob)
        prob /= max(prob)
    idx = 1
    while rnd >= prob[idx-1]:
        idx += 1
    return idx

def get_clusters(employment, **kwargs):
    '''
    Find the clusters for weekdays, saturday and sunday for a household member
    of the given eployment type based on the Crosstables given at
    # http://homepages.vub.ac.be/~daerts/Occupancy.html
    '''
    #create an empty dictionary
    keys = ['wkdy', 'sat', 'son']
    cluDict = dict()
    ##########################################################################
    # we find the cluster for each of the daytypes for the given employment
    # in 'Crosstable_employment.txt'
    for key in keys:
        order = ['U12','FTE','PTE','Unemployed','Retired','School']
        emp_i = order.index(employment)
        data = np.loadtxt(datapath + '/Aerts_Occupancy/Crosstables/Crosstable_Employment_'+key+'.txt', float).T[emp_i]
        rnd = np.random.random()
        cluster = get_probability(rnd, data, p_type='prob')
        cluDict.update({key:cluster})
    ##########################################################################
    return cluDict

def get_occDict(cluster, **kwargs):
    '''
    Create the dictionary with occupancy data based on the files retrieved from
    Aerts et al. as given at http://homepages.vub.ac.be/~daerts/Occupancy.html
    and stored in 'StROBe/Data/Aerts_Occupancy'.
    '''
    #first go the the correct location
    path = datapath + '/Aerts_Occupancy/Pattern' + str(cluster) + '/'
    # create an empty dictionary
    occDict = dict()
    ##########################################################################
    # first we load the occupancy start states 'ss' from StartStates.txt
    ss = dict()
    data = np.loadtxt(path + 'StartStates.txt', float)
    for i in range(len(data)):
        ss.update({str(i+1):data[i]})
    # and add the 'ss' data to the occupancy dictionary
    occDict.update({'ss':ss})
    ##########################################################################
    # Second we load the occupancy transitions state probabilities 'os'
    # from TransitionProbability.txt
    data = np.loadtxt(path + 'TransitionProbability.txt', float)
    for i in range(3):
        os_i = dict()
        for j in range(48):
            os_i.update({str(j+1):data[i*48+j]})
        # and add the 'os_i' data to the occupancy dictionary
        occDict.update({'os_'+str(i+1):os_i})
    ##########################################################################
    # Third we load the Markov time density 'ol' from DurationProbability.txt
    data = np.loadtxt(path + 'DurationProbability.txt', float)
    for i in range(3):
        ol_i = dict()
        for j in range(48):
            ol_i.update({str(j+1):data[i*48+j]})
        # and add the 'osn_i' data to the occupancy dictionary
        occDict.update({'ol_'+str(i+1):ol_i})
    ##########################################################################
    # and return the final occDict
    return occDict

def get_actDict(cluster, **kwargs):
    '''
    Create the dictionary with activity data based on the files retrieved from
    Aerts et al. as given at http://homepages.vub.ac.be/~daerts/Activity.html
    and stored in 'StROBe/Data/Aerts_activity'.
    '''
    # create an empty dictionary
    actDict = dict()
    ##########################################################################
    # first we define the dictionary used as legend for the load file
    act = {0:'pc', 1:'food', 2:'vacuum', 3:'iron', 4:'tv', 5:'audio',
           6:'dishes', 7:'washing', 8:'drying', 9:'shower'}
    ##########################################################################
    # Second we load the activity proclivity functions
    # from Patter*cluster*.txt
    FILNAM = 'Pattern'+str(cluster)+'.txt'
    data = np.loadtxt(datapath + "/Aerts_Activities/" + FILNAM, float)
    for i in range(10):
        actDict.update({act[i]:data.T[i]})
    ##########################################################################
    # and return the final actDict
    actDict.update({'period':600, 'steps':144})
    return actDict
