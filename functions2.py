from asyncio.constants import DEBUG_STACK_DEPTH
from cProfile import label
import datetime
from email.mime import base
import json
import functions
import os
from sqlite3 import Row

import numpy as np
import pandas as pd
import pm4py
import csv
import networkx as nx
import string
import random

import causallearn.graph.Edge
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, mv_fisherz
from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge
from IPython.display import Image, display
from sklearn.metrics import auc, precision_recall_curve

def matrix_evaluation(matrix,nodes,edges,data_headers,datas):

    ola = {}
    new_nodes = []
    musterlösung = pd.read_csv('fixit/musterlösungmatrix.csv')
    musterlösung.set_index("Unnamed: 0",inplace=True)
    # mustKKCerlösung = musterlösung.drop(columns= ["Unnamed: 0"])

    i=0
    for x in nodes:
        popo ={x:str(data_headers[i])}
        new_nodes.append(x)
        ola.update(popo)
        i+=1

    df = pd.DataFrame(columns=new_nodes,index=nodes)
    df.fillna(0, inplace=True)


    for elem in edges:
        df.loc[df.index == elem.__dict__["node1"], [elem.__dict__["node2"]]] = df.loc[df.index == elem.__dict__["node1"], [elem.__dict__["node2"]]] + elem.__dict__["numerical_endpoint_1"]
        df.loc[df.index == elem.__dict__["node2"], [elem.__dict__["node1"]]] = df.loc[df.index == elem.__dict__["node2"], [elem.__dict__["node1"]]] + elem.__dict__["numerical_endpoint_2"]


    df.rename(index=ola, inplace=True)


    df.rename(ola, axis=1,inplace=True)
    # pd.DataFrame(df.values + matrix.values, columns=df.columns, index=df.index)

    #matrix ist die alte
    #importiere musterLösung, gehe beide ansetzt durch mit index  df[x,y]== musterlösung[x,y] , und df[y,x]== (musterlösung[x,y] oder 2)
    #seet index in loop
    df.to_csv('fixit/analysed_data/test/dontknow' + str(datas) +'.csv')
    
    loop_i = 0
    for idx,elem in df.iterrows():
        loop_i += 1
        for i in range(loop_i):
            #one way
            if musterlösung.loc[idx, data_headers[i]] == 1 or musterlösung.loc[idx, data_headers[i]] == -1:
                if(2 == df.loc[idx, data_headers[i]] and 2 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1
                elif(1 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1
                elif(2 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1                
                elif(1 == df.loc[idx, data_headers[i]] and 2 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1                
                elif(-1 == df.loc[idx, data_headers[i]] and -1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1 

                elif(1 == df.loc[idx, data_headers[i]] and -1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1         
                elif(-1 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[data_headers[i] , idx] = matrix.loc[data_headers[i] , idx] + 1
            
            elif musterlösung.loc[idx, data_headers[i]] ==  0 and 0 != df.loc[idx, data_headers[i]]:
                if(2 == df.loc[idx, data_headers[i]] and 2 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] -1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1
                elif(1 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1
                elif(2 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1                
                elif(1 == df.loc[idx, data_headers[i]] and 2 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1                
                elif(-1 == df.loc[idx, data_headers[i]] and -1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1 

                elif(1 == df.loc[idx, data_headers[i]] and -1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1         
                elif(-1 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[data_headers[i] , idx] = matrix.loc[data_headers[i] , idx] - 1


    # df.to_csv('fixit/examples/matrix_id.csv')
    # df.index = data_headers
    # df.columns.rename(data_headers,inplace=True)




    return matrix


def matrix_evaluation_outer(matrix,nodes,edges,data_headers,datas):

    ola = {}
    new_nodes = []
    musterlösung = pd.read_csv('fixit/musterlösungmatrix_outer.csv')
    musterlösung.set_index("Unnamed: 0",inplace=True)
    # mustKKCerlösung = musterlösung.drop(columns= ["Unnamed: 0"])

    i=0
    for x in nodes:
        popo ={x:str(data_headers[i])}
        new_nodes.append(x)
        ola.update(popo)
        i+=1

    df = pd.DataFrame(columns=new_nodes,index=nodes)
    df.fillna(0, inplace=True)
    
    for elem in edges:
        df.loc[df.index == elem.__dict__["node1"], [elem.__dict__["node2"]]] = df.loc[df.index == elem.__dict__["node1"], [elem.__dict__["node2"]]] + elem.__dict__["numerical_endpoint_1"]
        df.loc[df.index == elem.__dict__["node2"], [elem.__dict__["node1"]]] = df.loc[df.index == elem.__dict__["node2"], [elem.__dict__["node1"]]] + elem.__dict__["numerical_endpoint_2"]

    df.rename(index=ola, inplace=True)

    df.rename(ola, axis=1,inplace=True)
    # pd.DataFrame(df.values + matrix.values, columns=df.columns, index=df.index)

    #matrix ist die alte
    #importiere musterLösung, gehe beide ansetzt durch mit index  df[x,y]== musterlösung[x,y] , und df[y,x]== (musterlösung[x,y] oder 2)
    #seet index in loop

    df.to_csv('fixit/analysed_data/test/dontknow' + str(datas) +'.csv')

    loop_i = 0
    for idx,elem in df.iterrows():
        loop_i += 1
        for i in range(loop_i):
            #one way
            if musterlösung.loc[idx, data_headers[i]] == 1 or musterlösung.loc[idx, data_headers[i]] == -1:
                if(2 == df.loc[idx, data_headers[i]] and 2 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1
                elif(1 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1
                elif(2 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1                
                elif(1 == df.loc[idx, data_headers[i]] and 2 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1                
                elif(-1 == df.loc[idx, data_headers[i]] and -1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] + 1 

                elif(1 == df.loc[idx, data_headers[i]] and -1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] + 1         
                elif(-1 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[data_headers[i] , idx] = matrix.loc[data_headers[i] , idx] + 1
            
            elif musterlösung.loc[idx, data_headers[i]] ==  0 and 0 != df.loc[idx, data_headers[i]]:
                if(2 == df.loc[idx, data_headers[i]] and 2 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] -1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1
                elif(1 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1
                elif(2 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1                
                elif(1 == df.loc[idx, data_headers[i]] and 2 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1                
                elif(-1 == df.loc[idx, data_headers[i]] and -1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1
                    matrix.loc[data_headers[i],idx] = matrix.loc[data_headers[i],idx] - 1 

                elif(1 == df.loc[idx, data_headers[i]] and -1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[idx, data_headers[i]] = matrix.loc[idx, data_headers[i]] - 1         
                elif(-1 == df.loc[idx, data_headers[i]] and 1 == df.loc[data_headers[i] , idx]):
                    matrix.loc[data_headers[i] , idx] = matrix.loc[data_headers[i] , idx] - 1

    # df.to_csv('fixit/examples/matrix_id.csv')
    # df.index = data_headers
    # df.columns.rename(data_headers,inplace=True)




    return matrix


def add_missingedges(matrix,data_headers):
    musterlösung = pd.read_csv('fixit/musterlösungmatrix.csv')
    musterlösung.set_index("Unnamed: 0",inplace=True)

    loop_i = 0
    for idx,elem in matrix.iterrows():
        loop_i += 1
        for i in range(loop_i):
            # musterlösung != 0 matrix == 0 set matrix = -100
            same_edge = (matrix.loc[idx, data_headers[i]] == 0 and 0 !=  musterlösung.loc[idx, data_headers[i]])
            if same_edge:
                matrix.loc[idx, data_headers[i]] = 404
                matrix.loc[data_headers[i],idx] = 404
    
    return matrix

def add_missingedges_outer(matrix,data_headers):
    musterlösung = pd.read_csv('fixit/musterlösungmatrix_outer.csv')
    musterlösung.set_index("Unnamed: 0",inplace=True)

    loop_i = 0
    for idx,elem in matrix.iterrows():
        loop_i += 1
        for i in range(loop_i):
            # musterlösung != 0 matrix == 0 set matrix = -100
            same_edge = (matrix.loc[idx, data_headers[i]] == 0 and (1 ==  musterlösung.loc[idx, data_headers[i]] or -1 ==  musterlösung.loc[idx, data_headers[i]]))
            if same_edge:
                matrix.loc[idx, data_headers[i]] = 404
                matrix.loc[data_headers[i], idx] = 404
    
    return matrix


from cdt.metrics import precision_recall,SHD
import numpy as np
from numpy.random import randint

def eva(tar, pred):
    #tar, pred = np.random.randint(2, size=(10, 10)), np.random.randn(10, 10)
    # adjacency matrixes of size 10x10
    aupr, curve = precision_recall(tar, pred)
    # leave low_confidence_undirected to False as the predictions are continuous

    #tar = np.triu(randint(2, size=(10, 10)))
    #pred = np.triu(randint(2, size=(10, 10)))
    siderg = SHD(tar, pred, True)
    print(siderg , aupr, curve)

def zz():
    import re
    zah = re.findall("[-+]?(?:\d*\.\d+|\d+)", "action_desicion	0.0	1.0	0.0	-0.17	0.0	-0.99	-0.01	-0.14	1.0	0.0	0.0	0.0	1.0	0.0	0.0")
    erg = ""
    for i in zah: 
        erg = erg + "&" + str(i)
    print(erg)
