from abc import ABC, abstractmethod
import networkx as nx

import pandas as pd


class Score( ABC ):
    
    def __init__(self, data : pd.DataFrame, graph : nx.DiGraph = None, isLogSpace = True):
        """
        initialises the Score abstract class. All classes that inherit from this class must implement the compute method.

        Args:
            data (pd.DataFrame): dataset
            graph (nx.DiGraph, optional): graph structure. Defaults to None. The graph must be DAG.
        """
        self.data = data
        self.graph = graph
        self.isLogSpace = isLogSpace
    
    # abstract method to be implemented by subclasses 
    @abstractmethod
    def compute(self):
        """
        implements a score function (e.g. BGe, Marginal Likelihood, etc)
        """
        pass
    
    # GETTERS AND SETTERS
    ####################################################

    def get_data(self):
        return self.data
    
    def get_graph(self):
        return self.graph
    
    def set_graph(self, graph):
        self.graph = graph
        
    def set_data(self, data):
        self.data = data
        
    def get_isLogSpace(self):
        return self.isLogSpace
    
    def set_isLogSpace(self, isLogSpace):
        self.isLogSpace = isLogSpace
        
        
