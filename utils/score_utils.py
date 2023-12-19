import itertools

import collections

import numpy as np
import pandas as pd

import networkx as nx

from itertools import permutations


from scores.bge import BGEscore
from scores.marginal import MarginalLogLikelihood


from scipy.stats import entropy

import os
from math import comb


import zipfile
import matplotlib.pyplot as plt

import itertools


# This function produces a matrix with all the possible parents of a given node
# up to the maximum number of parents
def listpossibleparents(maxparents, elements):
    graphs = {}
    parents_list = {}
    for i in elements:
        remaining_elements = elements.copy()
        remaining_elements.remove(i)
        matrix_of_parents = []
        poss_graphs = []
        for r in range(1, maxparents + 1):
            poss_parents = list(itertools.combinations(remaining_elements, r))
            
            for elem in poss_parents:
                G_init = nx.DiGraph()
                G_init.add_nodes_from(elements)
                for x in elem:
                    G_init.add_edge(x, i)
                matrix_of_parents.append(elem)
                poss_graphs.append(G_init)

        G_init = nx.DiGraph()
        G_init.add_nodes_from(elements)
        poss_graphs.insert(0, G_init)
        graphs[i] = poss_graphs

        matrix_of_parents.insert(0, ())
        parents_list[i] = matrix_of_parents

    return parents_list, graphs

# This function scores all the possible parents
def scorePossibleParents(parentstable, data):
    score_dict = {}

    for k, v in parentstable.items():
        score_list = []
        for graph in v:
            score_list.append(BGEscore(data=data, graph=graph).compute()['parameters'][k]['score'])
        score_dict[k] = score_list

    return score_dict