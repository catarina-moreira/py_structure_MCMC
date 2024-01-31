
from abc import ABC, abstractmethod

import numpy as np

import pandas as pd

from scores.ScoreAbstract import Score


class DummyScore(Score):

    def __init__(self, data : pd.DataFrame, incidence : np.ndarray, isLogSpace = False):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            graph (nx.DiGraph, optional): _description_. Defaults to None.
        """
        super().__init__(data, incidence)
        
        self.incidence = incidence
        self.data = data
        self.isLogSpace = isLogSpace
        
        self.node_labels = list(data.columns)
    
    def compute(self):
        
        parameters = {}
        total_ml = 0
        score_node = 1
        
        # Loop through each node in the graph
        for node in self.node_labels:
            
            parameters[node] = {
            'score' : score_node
            }
            
            total_ml = total_ml + score_node

        res = {
            'score': total_ml,
            'parameters': parameters
        }
        
        return res
    
    def compute_node(self, node):
        
        parameters = {}
        score_node = 1
        
        parameters[node] = {
            'score' : score_node}
        
        res = {
            'score': score_node,
            'parameters': parameters
        }
        return res
    
    
    # GETTERS
    #####################################################

    def to_string():
        return "Dummy Score"
    
    def get_to_string(self):
        return "Dummy Score"
    

    def get_incidence(self):
        return self.incidence
    
    def get_data(self):
        return self.data
    
    def get_node_labels(self):
        return self.node_labels
    
    def get_isLogSpace(self):
        return self.isLogSpace
    
    # SETTERS
    #####################################################

    def set_incidence(self, incidence):
        self.incidence = incidence
        
    def set_data(self, data):
        self.data = data
        
    def set_node_labels(self, node_labels):
        self.node_labels = node_labels
        
    def set_isLogSpace(self, isLogSpace):
        self.isLogSpace = isLogSpace
        