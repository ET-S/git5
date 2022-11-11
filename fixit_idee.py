from asyncio.constants import DEBUG_STACK_DEPTH
from cProfile import label
import datetime
from email.mime import base
import functions,functions2
import os
from sqlite3 import Row

import numpy as np
import pandas as pd
import pm4py
import csv
import networkx as nx
import string
import random

import causallearn.graph.GeneralGraph 
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, mv_fisherz
from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge
from IPython.display import Image, display


import math

import networkx as nx
import numpy as np
from networkx import *
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from PIL import Image


def build_datatable():

    #variante/inmvariante context person mit maxprice
    data = {
        # 'name': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #'name': np.random.randint(low=1000, high=9999, size=15),
        'name': random.sample(range(100, 130), 25),
        'money': np.random.randint(low=20, high=500, size=(25, 10)).tolist(),
        'preference': np.random.choice(range(131, 140), 25),
    }
    people = pd.DataFrame(data=data)
    people.to_csv('people.csv')

    #process variable produkt
    data = {
        # 'product_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #'product_id': np.random.randint(low=100, high=999, size=15),
        'product_id': random.sample(range(141, 180), 25),
        'value': np.random.randint(low=20, high=450, size=25),
        'liked': np.random.choice(range(181, 186), 25),
        'category': np.random.choice(range(131, 140), 25)
    }
    product = pd.DataFrame(data=data)
    product.to_csv('product.csv')

    #go through a process
    eventid = 1
    log = pd.DataFrame()
    session_id = 1

    for session_id in range(20000):
        if session_id % 5000 == 0:
            print("still Working")
        # action view website, you know the name of the aude doing this action
        #choose person who is going to view the website
        person_id = np.random.randint(low=0, high=24, size=1)[0]
        person_name = people.iloc[person_id]['name']

        #intervention_1
        if session_id > 11000 and session_id < 12500:
            person_preference = np.random.choice(range(131, 140), size=1)[0]
        else:
            person_preference = int(people.iloc[person_id]['preference'])

        #action view product
        #which product
        prod_id = np.random.randint(low=0, high=24, size=1)[0]
        product_id = int(product.iloc[prod_id]['product_id'])

        #intervention_1
        if session_id > 12500 and session_id < 15000:
            product_value = product.iloc[prod_id]['value']
        else:
            product_value = np.random.randint(low=20, high=450, size=25)[0]

        # intervention_1
        if session_id > 2000 and session_id < 3500:
            product_liked = np.random.choice(range(1,6), 1)[0]
        else:
            product_liked = product.iloc[prod_id]['liked'] - 180
        

        if session_id > 15000 and session_id < 16500:
            product_category = int(product.iloc[prod_id]['category'])
        else:
            product_category = np.random.choice(range(131, 140), 25)[0]

        #intervention_1
        if session_id > 4000 and session_id < 5500:
            person_max = np.random.randint(low=40, high=700, size=1)[0]
        else:
            person_max = people.iloc[person_id]['money'][product_category-131]

        action = 1
        timestamp = datetime.datetime.now()
        df2 = pd.DataFrame(data=[[session_id, person_name, person_max, person_preference,
                                  eventid, action, timestamp,
                                  0, 0, 0, 0, 0, 0]],
                           columns=['case ID', 'client_name', 'client_max_price', 'client_preference',
                                    'eventid', 'activity', 'timestamp',
                                    'product_id', 'product_category', 'product_liked', 'product_price', 'cart', 'cart_price'])
        log = pd.concat([log, df2])
        eventid += 1

        action = 2
        timestamp = datetime.datetime.now()
        df2 = pd.DataFrame(data=[[session_id, person_name, person_max, person_preference,
                                  eventid, action, timestamp,
                                  product_id, product_category, product_liked, product_value, 0, 0]],
                           columns=['case ID', 'client_name', 'client_max_price', 'client_preference',
                                    'eventid', 'activity', 'timestamp',
                                    'product_id', 'product_category', 'product_liked', 'product_price', 'cart', 'cart_price'])
        log = pd.concat([log, df2])
        eventid += 1

        #add to cart
        if (product_category == person_preference):
            rate = ((((product_liked-1) * 25) +
                    int(np.random.randint(-4, 4))) / 100) + 0.3
            # person_max = int(person_max * 1.3)
        else:
            rate = ((((product_liked-1) * 25) +
                    int(np.random.randint(-4, 4))) / 100)

        product_liked_reference = rate if 1 > rate else 1
        product_liked_reference = 0 if 0 > rate else product_liked_reference

        #intervention_1
        if session_id > 6000 and session_id < 7500:
            cart = product["product_id"][np.random.randint(
                low=0, high=15, size=1)[0]]
        else:
            cart = int(np.random.choice([product_id, 999999], 1, p=[
                       product_liked_reference, 1-product_liked_reference])[0])

        if (cart != 999999):
            action = 3
            timestamp = datetime.datetime.now()
            df2 = pd.DataFrame(data=[[session_id, person_name, person_max, person_preference,
                                      eventid, action, timestamp,
                                      product_id, product_category, product_liked, product_value, cart, 0]],
                               columns=['case ID', 'client_name', 'client_max_price', 'client_preference',
                                        'eventid', 'activity', 'timestamp',
                                        'product_id', 'product_category', 'product_liked', 'product_price', 'cart', 'cart_price'])
            log = pd.concat([log, df2])
            eventid += 1
        else:
            action = 7
            timestamp = datetime.datetime.now()
            df2 = pd.DataFrame(data=[[session_id, person_name, person_max, person_preference,
                                      eventid, action, timestamp,
                                      product_id, product_category, product_liked, product_value, cart, 0]],
                               columns=['case ID', 'client_name', 'client_max_price', 'client_preference',
                                        'eventid', 'activity', 'timestamp',
                                        'product_id', 'product_category', 'product_liked', 'product_price', 'cart', 'cart_price'])
            log = pd.concat([log, df2])
            eventid += 1

        #view cart
        action = 4
        #cart_price = product_value if cart != 0 else 0
        #intervention_1
        if session_id > 8000 and session_id < 9500:
            cart_price = int(np.random.randint(low=0, high=350, size=1)[0])
        else:
            cart_price = product.loc[product["product_id"] ==
                                     cart]["value"].values[0] if cart != 999999 else 999999

        timestamp = datetime.datetime.now()
        df2 = pd.DataFrame(data=[[session_id, person_name, person_max, person_preference,
                                  eventid, action, timestamp,
                                  product_id, product_category, product_liked, product_value, cart, int(cart_price)]],
                           columns=['case ID', 'client_name', 'client_max_price', 'client_preference',
                                    'eventid', 'activity', 'timestamp',
                                    'product_id', 'product_category', 'product_liked', 'product_price', 'cart', 'cart_price'])
        log = pd.concat([log, df2])
        eventid += 1

        #decision
        if cart == 999999:
            action = 6
        elif person_max > cart_price:
            action = np.random.choice([5, 6], 1, p=[0.95, 0.05])[0]
        else:
            action = np.random.choice([5, 6], 1, p=[0.05, 0.95])[0]

        timestamp = datetime.datetime.now()
        df2 = pd.DataFrame(data=[[session_id, person_name, person_max, person_preference,
                                  eventid, action, timestamp,
                                  product_id, product_category, product_liked, product_value, cart, int(cart_price)]],
                           columns=['case ID', 'client_name', 'client_max_price', 'client_preference',
                                    'eventid', 'activity', 'timestamp',
                                    'product_id', 'product_category', 'product_liked', 'product_price', 'cart', 'cart_price'])
        log = pd.concat([log, df2])
        eventid += 1

    return log








###############################################################
################Process mining and data allocation#############
###############################################################





def pm_allocatoion(i):
    dataframe = build_datatable()

    #case:clientID
    #dataframe = dataframe.rename(columns={'eventid': 'case:eventID'})
    dataframe = pm4py.format_dataframe(dataframe, case_id='case ID',
                     activity_key='activity', timestamp_key='timestamp')
    dataframe = dataframe.drop(columns=['case ID', 'activity', 'timestamp'])

    log = pm4py.convert_to_event_log(dataframe)
    
    pm4py.write_xes(log, 'extracting_causal_graphs_from_process_data/pseudo_data_2/data_' + str(i + 36) + '.xes')
    dataframe.to_csv('extracting_causal_graphs_from_process_data/pseudo_data_2/view_' + str(i + 36) + '.csv')


def inductive_miner():

    log = pm4py.read_xes(os.path.join("extracting_causal_graphs_from_process_data/pseudo_data_2/data_0.xes"))
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(
        log)
    # tree = pm4py.discover_process_tree_inductive(log)
    # pm4py.view_process_tree(tree)
    # net, initial_marking, final_marking = pm4py.convert_to_petri_net(tree)

    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    parameters = {
        pn_visualizer.Variants.FREQUENCY.value.Parameters.FORMAT: "png"}
    gviz = pn_visualizer.apply(net, initial_marking, final_marking,
                               parameters=parameters, variant=pn_visualizer.Variants.FREQUENCY, log=log)
                               
    pn_visualizer.save(gviz, "extracting_causal_graphs_from_process_data/pseudo_data_2/inductive_frequency.png")


# def cl_only(i):
#     df = pd.read_csv('view.csv')

#     df = df.drop(columns=['time:timestamp', '@@index',
#                  'Unnamed: 0', 'case:clientID', 'case:concept:name'])

#     df.rename(columns={'concept:name': 'action'}, inplace=True)

# #   zwischen
#     df.to_csv('extracting_causal_graphs_from_process_data/examples/cl_only_' + str(i) + '.csv')
#     data_headers = list(df.columns)
#     df.drop(columns=[list(df.columns)[0]])
#     data = df.to_numpy()

#     G, edges = fci(data, fisherz, 0.05, -1, -1, False)

#     # visualization
#     from causallearn.utils.GraphUtils import GraphUtils
#     pdy = GraphUtils.to_pydot(G, labels=data_headers)
#     pdy.write_png('extracting_causal_graphs_from_process_data/examples/cl_only_' + str(i) + '.png')

#     return G.get_nodes(),edges


###############################################################
#################Causal mining and prep########################
###############################################################

def cl_only_idmatrix(i):

    #learner
    df = pd.read_csv('extracting_causal_graphs_from_process_data/pseudo_data_2/view_' + str(i) + '.csv')

    #first attempt without desicion
    df.loc[df['concept:name'] == 1, 'p1_view_webside'] = 11
    df.loc[df['concept:name'] == 2, 'p2_view_product'] = 11
    df.loc[df['concept:name'] == 3, 'p3_add_product_decision'] = 11
    df.loc[df['concept:name'] == 7, 'p3_add_product_decision'] = 11
    df.loc[df['concept:name'] == 4, 'p4_view_card'] = 11
    df.loc[df['concept:name'] == 5, 'p5_buy_product_decision'] = 11
    df.loc[df['concept:name'] == 6, 'p5_buy_product_decision'] = 11

    df.fillna(10, inplace=True)

    df = df.astype({'p1_view_webside': int})
    df = df.astype({'p2_view_product': int})
    df = df.astype({'p3_add_product_decision': int})
    df = df.astype({'p4_view_card': int})
    df = df.astype({'p5_buy_product_decision': int})

    df = df.drop(df[df.p1_view_webside == 1].index)
    df = df.drop(columns=['concept:name', 'time:timestamp', '@@index',
                 'Unnamed: 0', 'eventid', 'p1_view_webside','case:concept:name'])



#   zwischen
    df.to_csv('extracting_causal_graphs_from_process_data/analysed_data/only_id/cl_only_idmatrix' + str(i) + '.csv')
    data_headers = list(df.columns)
    df.drop(columns=[list(df.columns)[0]])
    data = df.to_numpy()

    G, edges = fci(data, fisherz, 0.05, -1, -1, False)
    base_nodes = G.get_nodes()

    nodes = base_nodes[-4:]

    bk = BackgroundKnowledge()

    for x in nodes:
        sub_nodes = list(nodes)
        sub_nodes.remove(x)
        for y in sub_nodes:
            bk = bk.add_forbidden_by_node(x, y)

    G2, edges2 = fci(data, fisherz, 0.05, -1, -1, False, bk)

    # visualization
    from causallearn.utils.GraphUtils import GraphUtils
    pdy = GraphUtils.to_pydot(G2, labels=data_headers)
    pdy.write_png('extracting_causal_graphs_from_process_data/analysed_data/only_id/cl_only_idmatrix' + str(i) + '.png')

    G_pc = pc(data)
    base_nodes_pc = G_pc.G.get_nodes()
    nodes_pc = base_nodes_pc[-4:]
    bk_pc = BackgroundKnowledge()
    
    for x in nodes_pc:
        sub_nodes = list(nodes_pc)
        sub_nodes.remove(x)
        for y in sub_nodes:
            bk_pc = bk_pc.add_forbidden_by_node(x, y)
    G2_pc = pc(data, background_knowledge=bk_pc)

    pdy_pc = GraphUtils.to_pydot(G2_pc.G, labels=data_headers)
    pdy_pc.write_png('extracting_causal_graphs_from_process_data/analysed_data/only_id/cl_only_idmatrix_pc' + str(i) + '.png')

    return G2.get_nodes(),edges2, data_headers, G2_pc.G.__dict__["nodes"], G2_pc.G.get_graph_edges()


def cl_idmatrix_decision_within(i):
    #learner

    df = pd.read_csv('extracting_causal_graphs_from_process_data/pseudo_data_2/view_' + str(i) + '.csv')

    #first attempt without desicion
    df.loc[df['concept:name'] == 1, 'p1_view_webside'] = 11
    df.loc[df['concept:name'] == 2, 'p2_view_product'] = 11
    df.loc[df['concept:name'] == 3, 'p3_add_product_decision'] = 11
    df.loc[df['concept:name'] == 7, 'p3_add_product_decision'] = 12
    df.loc[df['concept:name'] == 4, 'p4_view_card'] = 11
    df.loc[df['concept:name'] == 5, 'p5_buy_product_decision'] = 11
    df.loc[df['concept:name'] == 6, 'p5_buy_product_decision'] = 12

    df.fillna(10, inplace=True)

    df = df.astype({'p1_view_webside': int})
    df = df.astype({'p2_view_product': int})
    df = df.astype({'p3_add_product_decision': int})
    df = df.astype({'p4_view_card': int})
    df = df.astype({'p5_buy_product_decision': int})

    df = df.drop(df[df.p1_view_webside == 1].index)
    df = df.drop(columns=['concept:name', 'time:timestamp', '@@index',
                 'Unnamed: 0', 'eventid', 'case:concept:name', 'p1_view_webside'])

#   zwischen
    df.to_csv('extracting_causal_graphs_from_process_data/analysed_data/id_within/cl_idmatrix_decision_within' + str(i) + '.csv')
    data_headers = list(df.columns)
    df.drop(columns=[list(df.columns)[0]])
    data = df.to_numpy()

    G, edges = fci(data, fisherz, 0.05, -1, -1, False)

    base_nodes = G.get_nodes()
    nodes = base_nodes[-4:]

    bk = BackgroundKnowledge()

    for x in nodes:
        sub_nodes = list(nodes)
        sub_nodes.remove(x)
        for y in sub_nodes:
            bk = bk.add_forbidden_by_node(x, y)

    G2, edges2 = fci(data, fisherz, 0.05, -1, -1, False, bk)

    # visualization
    from causallearn.utils.GraphUtils import GraphUtils
    pdy = GraphUtils.to_pydot(G2, labels=data_headers)
    pdy.write_png('extracting_causal_graphs_from_process_data/analysed_data/id_within/cl_idmatrix_decision_within' + str(i) + '.png')

    G_pc = pc(data)
    base_nodes_pc = G_pc.G.get_nodes()
    nodes_pc = base_nodes_pc[-4:]
    bk_pc = BackgroundKnowledge()
    
    for x in nodes_pc:
        sub_nodes = list(nodes_pc)
        sub_nodes.remove(x)
        for y in sub_nodes:
            bk_pc = bk_pc.add_forbidden_by_node(x, y)
    G2_pc = pc(data, background_knowledge=bk_pc)

    pdy_pc = GraphUtils.to_pydot(G2_pc.G, labels=data_headers)
    pdy_pc.write_png('extracting_causal_graphs_from_process_data/analysed_data/id_within/cl_idmatrix_decision_within_pc' + str(i) + '.png')

    return G2.get_nodes(),edges2, data_headers, G2_pc.G.__dict__["nodes"], G2_pc.G.get_graph_edges()

    return G2.get_nodes(),edges2, data_headers


def cl_idmatrix_decision_outer(i):
    #learner

    df = pd.read_csv('extracting_causal_graphs_from_process_data/pseudo_data_2/view_' + str(i) + '.csv')

    #first attempt without desicion
    df.loc[df['concept:name'] == 1, 'p1_view_webside'] = 11
    df.loc[df['concept:name'] == 2, 'p2_view_product'] = 11
    df.loc[df['concept:name'] == 3, 'p3_add_product_decision'] = 11
    df.loc[df['concept:name'] == 7, 'p3_add_product_decision'] = 11
    df.loc[df['concept:name'] == 4, 'p4_view_card'] = 11
    df.loc[df['concept:name'] == 5, 'p5_buy_product_decision'] = 11
    df.loc[df['concept:name'] == 6, 'p5_buy_product_decision'] = 11

    df.fillna(10, inplace=True)

    # the seperated desicion
    decision_concept = ['product_decision', 'action_desicion']
    df.loc[df['concept:name'] == 3, 'product_decision'] = 12
    df.loc[df['concept:name'] == 7, 'product_decision'] = 13
    df.loc[df['concept:name'] == 5, 'action_desicion'] = 14
    df.loc[df['concept:name'] == 6, 'action_desicion'] = 15

    df.fillna(16, inplace=True)

    df = df.astype({'p1_view_webside': int})
    df = df.astype({'p2_view_product': int})
    df = df.astype({'p3_add_product_decision': int})
    df = df.astype({'p4_view_card': int})
    df = df.astype({'p5_buy_product_decision': int})
    df = df.astype({'product_decision': int})
    df = df.astype({'action_desicion': int})

    df = df.drop(df[df.p1_view_webside == 1].index)
    df = df.drop(columns=['concept:name', 'time:timestamp', '@@index',
                 'Unnamed: 0', 'eventid', 'p1_view_webside','case:concept:name'])


#   zwischen
    df.to_csv('extracting_causal_graphs_from_process_data/analysed_data/id_outer/cl_idmatrix_decision_outer' + str(i) + '.csv')
    data_headers = list(df.columns)
    df.drop(columns=[list(df.columns)[0]])
    data = df.to_numpy()

    G, edges = fci(data, fisherz, 0.05, -1, -1, False)

    base_nodes = G.get_nodes()
    nodes = base_nodes[-6:][0:4]

    bk = BackgroundKnowledge()

    for x in nodes:
        sub_nodes = list(nodes)
        sub_nodes.remove(x)
        for y in sub_nodes:
            bk = bk.add_forbidden_by_node(x, y)

    G2, edges2 = fci(data, fisherz, 0.05, -1, -1, False, bk)

    # visualization
    from causallearn.utils.GraphUtils import GraphUtils
    pdy = GraphUtils.to_pydot(G2, labels=data_headers)
    pdy.write_png('extracting_causal_graphs_from_process_data/analysed_data/id_outer/cl_idmatrix_decision_outer' + str(i) + '.png')
    
    
    G_pc = pc(data)
    base_nodes_pc = G_pc.G.get_nodes()
    nodes_pc = base_nodes_pc[-4:]
    bk_pc = BackgroundKnowledge()
    
    for x in nodes_pc:
        sub_nodes = list(nodes_pc)
        sub_nodes.remove(x)
        for y in sub_nodes:
            bk_pc = bk_pc.add_forbidden_by_node(x, y)
    G2_pc = pc(data, background_knowledge=bk_pc)

    pdy_pc = GraphUtils.to_pydot(G2_pc.G, labels=data_headers)
    pdy_pc.write_png('extracting_causal_graphs_from_process_data/analysed_data/id_outer/cl_idmatrix_decision_outer_pc' + str(i) + '.png')

    return G2.get_nodes(),edges2, data_headers, G2_pc.G.__dict__["nodes"], G2_pc.G.get_graph_edges()

    return G2.get_nodes(),edges2, data_headers

















###############################################################
########################main###################################
###############################################################

def main():
        
    now = datetime.datetime.now()
    worked = 0
    worked_decision_within = 0
    worked_decision_outer = 0
    worked_id_only = 0
    something_went_wrong = 0

    column_headers_outer = ['client_name', 'client_max_price', 'client_preference', 'product_id', 'product_category', 'product_liked', 'product_price', 'cart', 'cart_price', 'p2_view_product', 'p3_add_product_decision', 'p4_view_card', 'p5_buy_product_decision','product_decision', 'action_desicion']
    column_headers = ['client_name', 'client_max_price', 'client_preference', 'product_id', 'product_category', 'product_liked', 'product_price', 'cart', 'cart_price', 'p2_view_product', 'p3_add_product_decision', 'p4_view_card', 'p5_buy_product_decision']

    causal_check_decision_within = pd.DataFrame(index=column_headers,columns=column_headers)
    causal_check_decision_within.fillna(value=0,inplace=True)
    causal_check_decision_within_pc = pd.DataFrame(index=column_headers,columns=column_headers)
    causal_check_decision_within_pc.fillna(value=0,inplace=True)

    causal_check_decision_outer = pd.DataFrame(index=column_headers_outer,columns=column_headers_outer)
    causal_check_decision_outer.fillna(value=0,inplace=True)
    causal_check_decision_outer_pc = pd.DataFrame(index=column_headers_outer,columns=column_headers_outer)
    causal_check_decision_outer_pc.fillna(value=0,inplace=True)

    causal_check_id_only = pd.DataFrame(index=column_headers,columns=column_headers)
    causal_check_id_only.fillna(value=0,inplace=True)
    causal_check_id_only_pc = pd.DataFrame(index=column_headers,columns=column_headers)
    causal_check_id_only_pc.fillna(value=0,inplace=True)


    for i in range(100):
        try:
            nodes,edges,data_headers, nodes_pc,edges_pc = cl_only_idmatrix(i)
            causal_check_id_only = functions2.matrix_evaluation(causal_check_id_only,nodes,edges,data_headers,1000+i)
            causal_check_id_only_pc = functions2.matrix_evaluation(causal_check_id_only_pc,nodes_pc,edges_pc,data_headers,2000 + i)
            worked_id_only += 1
        except Exception as e:
            something_went_wrong += 1
            print("_______________________cl_only_idmatrix_____________________________--")
            print(e)

        try:
            nodes,edges,data_headers, nodes_pc,edges_pc = cl_idmatrix_decision_within(i)
            causal_check_decision_within = functions2.matrix_evaluation(causal_check_decision_within,nodes,edges,data_headers,3000+i)
            causal_check_decision_within_pc = functions2.matrix_evaluation(causal_check_decision_within_pc,nodes_pc,edges_pc,data_headers,4000+i)
            worked_decision_within += 1
        except Exception as e:
            print("________________________cl_idmatrix_decision_within____________________________--")
            print(e)
            something_went_wrong += 1

        try:
            nodes,edges,data_headers,nodes_pc,edges_pc = cl_idmatrix_decision_outer(i)
            causal_check_decision_outer = functions2.matrix_evaluation_outer(causal_check_decision_outer,nodes,edges,data_headers,5000+i)
            causal_check_decision_outer_pc = functions2.matrix_evaluation_outer(causal_check_decision_outer_pc,nodes_pc,edges_pc,data_headers,6000+i)
            worked_decision_outer += 1
        except Exception as e:
            print("____________________cl_idmatrix_decision_outer________________________________")
            print(e)
            something_went_wrong += 1
        
        end = datetime.datetime.now()
        print(end-now)
        print(worked)



    causal_check_decision_within = causal_check_decision_within.div(worked_decision_within)  #* 100
    causal_check_id_only = causal_check_id_only.div(worked_id_only)  #* 100
    causal_check_decision_outer = causal_check_decision_outer.div(worked_decision_outer)  #* 100

    causal_check_decision_within_pc = causal_check_decision_within_pc.div(worked_decision_within)  #* 100
    causal_check_id_only_pc = causal_check_id_only_pc.div(worked_id_only) #* 100
    causal_check_decision_outer_pc = causal_check_decision_outer_pc.div(worked_decision_outer) #* 100

    causal_check_decision_within = functions2.add_missingedges(causal_check_decision_within,column_headers)
    causal_check_id_only = functions2.add_missingedges(causal_check_id_only,column_headers)
    causal_check_decision_outer = functions2.add_missingedges_outer(causal_check_decision_outer,column_headers_outer)

    causal_check_id_only_pc = functions2.add_missingedges(causal_check_id_only_pc,column_headers)
    causal_check_decision_within_pc = functions2.add_missingedges(causal_check_decision_within_pc,column_headers)
    causal_check_decision_outer_pc = functions2.add_missingedges_outer(causal_check_decision_outer_pc,column_headers_outer)

    causal_check_decision_within.to_csv('extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within.csv')
    causal_check_id_only.to_csv('extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only.csv')
    causal_check_decision_outer.to_csv('extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer.csv')

    causal_check_id_only_pc.to_csv('extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only_pc.csv')
    causal_check_decision_within_pc.to_csv('extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within_pc.csv')
    causal_check_decision_outer_pc.to_csv('extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer_pc.csv')



    end = datetime.datetime.now()
    print(end-now)
    print(worked)
    print(something_went_wrong)


#durchlauf der csv dateien mit i in range(100)
def create_pseudo_data():
    now = datetime.datetime.now()
    worked = 0
    time_list = []
    for i in range(100):
        try:
            now1 = datetime.datetime.now()
            pm_allocatoion(i)
            worked += 1
            end1 = datetime.datetime.now()
            time_list.append(end1-now1)

        except Exception as e:
            print("_____________________pm_allocatoion_______________________________--")
            print(e)
        print (worked)
        print(time_list)


    print (worked)
    print(time_list)
    end = datetime.datetime.now()
    print(end-now)
    return 


import matplotlib.pyplot as plt
from graph_plot import plot_from_adj_mat























node_list = ['client name', 
    'client max price', 
    'client preference', 
    'product id', 
    'product category', 
    'product liked', 
    'product value', 
    'cart', 
    'cart price', 
    'view product', 
    'add product', 
    'view card', 
    'buy/decline']

node_list_outer = ['client name', 
    'client max price', 
    'client preference', 
    'product id', 
    'product category', 
    'product liked', 
    'product value', 
    'cart', 
    'cart price', 
    'view product', 
    'add product', 
    'view card', 
    'buy/decline',
    'product decision',
    'action desicion']




#####################################################################
#################################create musterlösung#################
#####################################################################
def create_sample_solution():
    fig, ax = plt.subplots()

    df = pd.read_csv('extracting_causal_graphs_from_process_data/musterlösungmatrix.csv')
    df.set_index("Unnamed: 0",inplace=True)
    data_headers = df.head()

    df.multiply(-1)

    for elem in data_headers:
        df.loc[df[elem]==1,elem]=0
        df.loc[df[elem]==-1,elem]=1


    adj_mat = df.to_numpy()
    print(adj_mat)
    variable_names = list(range(len(df)))
    variable_names =[str(ssio) for ssio in variable_names]

    dataset_name = "causallearn"

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_aspect('equal')
    plt.sca(ax)
    plot_from_adj_mat(adj_mat, variable_names, dataset_name, ax=ax, abrev_vars=False, edge_mode='strength')

    extra = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    extra_list = []
    legend_list=[]
    for i,elemg in enumerate(variable_names):
        legend_list.append(str(elemg) + ":" + str(node_list_outer[i]))
        extra_list.append(extra)

    plt.legend(extra_list,legend_list, loc='upper left', title='Legend',bbox_to_anchor=(1.03, 1.0))
    plt.title("ground truth")
    plt.tight_layout()
    plt.savefig("extracting_causal_graphs_from_process_data/analysed_data/must_without2.jpg", bbox_inches='tight')







#####################################################################
#######################create other stuff ###########################
#####################################################################
#plot an average
def plot_new(matrixpath,ergpath,mod,name):

    fig, ax = plt.subplots()

    df = pd.read_csv(matrixpath)
    df.set_index("Unnamed: 0",inplace=True)
    data_headers = df.head()

    # df.multiply(-1)

    # for elem in data_headers:
    #     df.loc[df[elem]==1,elem]=0
    #     df.loc[df[elem]==-1,elem]=1

    df = df.copy()
    if mod == "found":
        # df = df * -1
        for elem in data_headers:
            df.loc[df[elem] < 0,elem] = 0
            df.loc[df[elem]==404,elem] = 2

    elif mod == "not_found":
        # df = df * -1
        for elem in data_headers:
            df.loc[df[elem] >= 0 ,elem] = 404
            #df.loc[df[elem] == 0 ,elem] = 404
            df.loc[(df[elem] > -0.2) & (df[elem] < 0),elem] = df[elem] * -1
            df.loc[df[elem] == 404 ,elem] = 0
            df.loc[df[elem] < -0.2 ,elem] = (df[elem] * -1) + 2

    elif mod =="dag":
        for elem in data_headers:
            df.loc[df[elem] < 0,elem] = 0
            df.loc[df[elem]==404,elem] = 0
        # for elem in data_headers:
    #     df.loc[df[elem]==1,elem]=0
    #     df.loc[df[elem]==-1,elem]=1

    adj_mat = df.to_numpy()
    variable_names = list(range(len(df)))
    variable_names =[str(ssio) for ssio in variable_names]

    dataset_name = "causallearn"

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_aspect('equal')
    plt.sca(ax)
    #ax.set_title(f'{dataset_name}', y=1)
    plot_from_adj_mat(adj_mat, variable_names, dataset_name, ax=ax, abrev_vars=False, edge_mode='strength')


    extra = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    extra_list = []
    legend_list=[]
    for i,elemg in enumerate(variable_names):
        legend_list.append(str(elemg) + ":" + str(node_list_outer[i]))
        extra_list.append(extra)

    # plt.legend([extra, extra],["123","456"], loc='upper left', title='Legend')
    # plt.savefig("extracting_causal_graphs_from_process_data/analysed_data/as.jpg")

    plt.legend(extra_list,legend_list, loc='upper left', title='Legend',bbox_to_anchor=(1.03, 1.0))
    plt.title(name)
    plt.tight_layout()
    plt.savefig(ergpath, bbox_inches='tight')


def regression_recall(matrixpath,name):

    df = pd.read_csv(matrixpath)
    df.set_index("Unnamed: 0",inplace=True)
    data_headers = df.head()
 
    erg = pd.DataFrame(data=[[0,0],[0,0]])
    l = len(df)
    data_headers = ['client_name', 'client_max_price','client_preference','product_id','product_category','product_liked', 
    'product_price','cart','cart_price','p2_view_product','p3_add_product_decision','p4_view_card','p5_buy_product_decision',
    'product_decision','action_desicion']

    for i in range(l):
        for j in range(l):
            if i != j and i > j:
                if df.loc[data_headers[i], data_headers[j]] == 404:
                    erg.loc[1,0] += 1
                elif df.loc[data_headers[i], data_headers[j]] == 0:
                    erg.loc[1,1] += 1
                elif df.loc[data_headers[i], data_headers[j]] > 0 and df.loc[data_headers[i], data_headers[j]] <=1 :
                    erg.loc[0,0] += 1
                elif df.loc[data_headers[i], data_headers[j]] < 0:
                    erg.loc[0,1] += 1
    
    erg.to_csv("extracting_causal_graphs_from_process_data/analysed_data/precision_recall/"+ name +".csv")
    print(erg)

    return




def create_dag(matrixpath,save_path):

    df = pd.read_csv(matrixpath)
    df.set_index("Unnamed: 0",inplace=True)
    data_headers = df.head()
 
    erg = df.copy()
    l = len(erg)
    data_headers = ['client_name', 'client_max_price','client_preference','product_id','product_category','product_liked', 
    'product_price','cart','cart_price','p2_view_product','p3_add_product_decision','p4_view_card','p5_buy_product_decision',
    'product_decision','action_desicion']

    for i in range(l):
        for j in range(l):
            if i != j and i > j and df.loc[ data_headers[i], data_headers[j]] != 404 and df.loc[ data_headers[i], data_headers[j]] != 0:
                if df.loc[data_headers[i], data_headers[j]] > df.loc[data_headers[j], data_headers[i]] and df.loc[data_headers[i], data_headers[j]] > 0.2:
                    erg.loc[data_headers[i], data_headers[j]] = 1
                    erg.loc[data_headers[j], data_headers[i]] = -1
                elif df.loc[data_headers[i], data_headers[j]] < df.loc[data_headers[j], data_headers[i]]and df.loc[data_headers[i], data_headers[j]] > 0.2:
                    erg.loc[data_headers[i], data_headers[j]] = -1
                    erg.loc[data_headers[j], data_headers[i]] = 1
                elif df.loc[data_headers[i], data_headers[j]] > df.loc[data_headers[j], data_headers[i]] and df.loc[data_headers[i], data_headers[j]] < -0.2:
                    erg.loc[data_headers[i], data_headers[j]] = -1
                    erg.loc[data_headers[j], data_headers[i]] = 1
                elif df.loc[data_headers[i], data_headers[j]] < df.loc[data_headers[j], data_headers[i]]and df.loc[data_headers[i], data_headers[j]] < -0.2:
                    erg.loc[data_headers[i], data_headers[j]] = 1
                    erg.loc[data_headers[j], data_headers[i]] = -1
                elif df.loc[data_headers[i], data_headers[j]] == df.loc[data_headers[j], data_headers[i]] and (df.loc[data_headers[i], data_headers[j]] < -0.2 or df.loc[data_headers[i], data_headers[j]] > 0.2):
                    erg.loc[data_headers[i], data_headers[j]] = -2
                    erg.loc[data_headers[j], data_headers[i]] = -2
                    print("unentschieden")
                else:
                    erg.loc[data_headers[i], data_headers[j]] = 0
                    erg.loc[data_headers[j], data_headers[i]] = 0
    erg = erg.multiply(-1)
    erg.to_csv(save_path)

def switch(read_path, save_path): 

    df = pd.read_csv(read_path)
    df.set_index("Unnamed: 0",inplace=True)
    data_headers = df.head()
 
    erg = df.copy()
    l = len(erg)
    data_headers = ['client_name', 'client_max_price','client_preference','product_id','product_category','product_liked', 
    'product_price','cart','cart_price','p2_view_product','p3_add_product_decision','p4_view_card','p5_buy_product_decision',
    'product_decision','action_desicion']

    for i in range(l):
        for j in range(l):
            if i != j and i > j and df.loc[ data_headers[i], data_headers[j]] != 404 and df.loc[ data_headers[i], data_headers[j]] != 0:
                erg.loc[ data_headers[i], data_headers[j]] = df.loc[ data_headers[j], data_headers[i]]
                erg.loc[ data_headers[j], data_headers[i]] = df.loc[ data_headers[i], data_headers[j]]
                


    erg.to_csv(save_path)




def create_avg_found_plots():
    
    switch("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only.csv","extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only2.csv")
    switch("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only_pc.csv","extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only_pc2.csv")
    switch("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer.csv","extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer2.csv")
    switch("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer_pc.csv","extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer_pc2.csv")
    switch("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within.csv","extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within2.csv")
    switch("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within_pc.csv","extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within_pc2.csv")

    regression_recall("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only.csv","var1_fci")
    regression_recall("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only_pc.csv","var1_pc")
    regression_recall("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer.csv","var3_fci")
    regression_recall("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer_pc.csv","var3_pc")
    regression_recall("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within.csv","var2_fci")
    regression_recall("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within_pc.csv","var2_pc")
    
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_fci.jpg","found","Variant1 FCI dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only_pc2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_pc.jpg","found","Variant1 PC dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_outer_fci.jpg","found","Variant3 FCI dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer_pc2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_outer_pc.jpg","found","Variant3 PC dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_fci.jpg","found","Variant2 FCI dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within_pc2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_pc.jpg","found","Variant2 PC dependencies")
    
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_not_found_fci.jpg","not_found","Variant1 FCI wrong dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only_pc2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_not_found_pc.jpg","not_found","Variant1 PC wrong dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_outer_not_found_fci.jpg","not_found","Variant3 FCI wrong dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer_pc2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_outer_not_found_pc.jpg","not_found","Variant3 PC wrong dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_not_found_fci.jpg","not_found","Variant2 FCI wrong dependencies")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within_pc2.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_not_found_pc.jpg","not_found","Variant2 PC wrong dependencies")

    create_dag("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_fci.csv")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_fci.csv", "extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_fci.jpg","dag","Variant1 FCI average graph")
    create_dag("extracting_causal_graphs_from_process_data/analysed_data/only_id/causal_check_id_only_pc.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_pc.csv")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_pc.csv", "extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_pc.jpg","dag","Variant1 PC average graph")
    create_dag("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_dag_fci.csv")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/test2/id_dag_fci.csv", "extracting_causal_graphs_from_process_data/analysed_data/test2/id_outer_dag_fci.jpg","dag","Variant3 FCI average graph")
    create_dag("extracting_causal_graphs_from_process_data/analysed_data/id_outer/causal_check_decision_outer_pc.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_dag_pc.csv")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/test2/id_dag_pc.csv", "extracting_causal_graphs_from_process_data/analysed_data/test2/id_outer_dag_pc.jpg","dag","Variant3 PC average graph")
    create_dag("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_dag_fci.csv")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_dag_fci.csv", "extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_dag_fci.jpg","dag","Variant2 FCI average graph")
    create_dag("extracting_causal_graphs_from_process_data/analysed_data/id_within/causal_check_decision_within_pc.csv","extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_dag_pc.csv")
    plot_new("extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_dag_pc.csv", "extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_dag_pc.jpg","dag","Variant2 PC average graph")

    do_eva(read_path = "extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_fci.csv", samplevariant = "")
    do_eva(read_path = "extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_pc.csv", samplevariant = "")

    do_eva(read_path = "extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_dag_fci.csv", samplevariant = "")
    do_eva(read_path = "extracting_causal_graphs_from_process_data/analysed_data/test2/id_within_dag_pc.csv", samplevariant = "")

    do_eva(read_path = "extracting_causal_graphs_from_process_data/analysed_data/test2/id_dag_fci.csv", samplevariant = "_outer")
    do_eva(read_path = "extracting_causal_graphs_from_process_data/analysed_data/test2/id_dag_pc.csv", samplevariant = "_outer")




def do_eva(read_path = "extracting_causal_graphs_from_process_data/analysed_data/test2/id_only_dag_fci.csv", samplevariant = ""):
    pred = pd.read_csv(read_path)
    pred.set_index("Unnamed: 0",inplace=True)

    tar = pd.read_csv("extracting_causal_graphs_from_process_data/musterlösungmatrix" + samplevariant + ".csv")
    tar.set_index("Unnamed: 0",inplace=True)

    data_headers = ['client_name', 'client_max_price','client_preference','product_id','product_category','product_liked', 
    'product_price','cart','cart_price','p2_view_product','p3_add_product_decision','p4_view_card','p5_buy_product_decision',
    'product_decision','action_desicion']

    pred = pred * -1
    for i in range(len(pred)):
        for j in range(len(pred)):
            if pred.loc[data_headers[i],data_headers[j]] == -2:
                pred.loc[data_headers[i],data_headers[j]] = 1                
            elif pred.loc[data_headers[i],data_headers[j]] <= 0:
                pred.loc[data_headers[i],data_headers[j]] = 0
            elif pred.loc[data_headers[i],data_headers[j]] == 404:
                pred.loc[data_headers[i],data_headers[j]] = 0


    for i in range(len(tar)):
        for j in range(len(tar)):
            if tar.loc[data_headers[i],data_headers[j]] <= 0:
                tar.loc[data_headers[i],data_headers[j]] = 0

    pred.to_csv("extracting_causal_graphs_from_process_data/isitit.csv")
    tar.to_csv("extracting_causal_graphs_from_process_data/isitit2.csv")

    tar = tar.to_numpy()
    pred = pred.to_numpy()

    functions2.eva(tar,pred)
    return

# create_avg_found_plots()
# main()

#create_avg_found_plots()
create_sample_solution()
#create_sample_solution()