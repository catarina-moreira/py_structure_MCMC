from abc import ABC, abstractmethod
import networkx as nx

import pandas as pd
import numpy as np

from scipy.stats import gamma


import statsmodels.api as sm

from scores.ScoreAbstract import ScoreStrategy

##############################################################################################
#
#  BDeu SCORE
#
##############################################################################################
class BDeu(ScoreStrategy):
    
    def __init__(self, data : pd.DataFrame, graph : nx.DiGraph = None):
        super().__init__(data)
        self.graph = graph
    
    
    # todo: for larger datasets, this function is very slow.
    def compute(self, alpha: float = 1.0):
        
        BDeu_score = 0.0
        graph = self.graph
        data = self.data

        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            r = data[parents].drop_duplicates().shape[0] if parents else 1
            q = data[node].nunique()

            # Calculate alpha values
            alpha_j = alpha / r
            alpha_ij = alpha / (r * q)

            # Compute counts
            if parents:
                counts = data.groupby(parents + [node]).size().reset_index(name='counts')
                parent_counts = data.groupby(parents).size().reset_index(name='parent_counts')
                counts = counts.merge(parent_counts, on=parents)
            else:
                counts = data[node].value_counts().reset_index()
                counts.columns = [node, 'counts']
                counts['parent_counts'] = len(data)

            for _, row in counts.iterrows():
                BDeu_score += (gamma(alpha_ij + row['counts']) - gamma(alpha_ij) + gamma(alpha_j) - gamma(alpha_j + row['parent_counts']))
                
        return BDeu_score
    
    # Getters and Setters
    #####################################################
    
    def getGraph(self):
        return self.graph
    
    def setGraph(self, graph):
        self.graph = graph
        
    def getData(self):
        return self.data

    def setData(self, data):
        self.data = data

        


