
from abc import ABC, abstractmethod
import networkx as nx

import pandas as pd
import numpy as np

from scipy.special import gammaln
from scipy.stats import invgamma
from scipy.stats import multivariate_normal

import statsmodels.api as sm

from scores.ScoreAbstract import Score


class DummyScore(Score):

    ##
    ## Constructor
    ####################################################################
    def __init__(self, data : pd.DataFrame, graph : nx.DiGraph = None, isLogSpace = True):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            graph (nx.DiGraph, optional): _description_. Defaults to None.
        """
        super().__init__(data)
        self.graph = graph
        self.isLogSpace = isLogSpace
        self.am = 1
        self.aw = len(data.columns) + self.am + 1
        self.t = self.am * (self.aw - len(data.columns) - 1)/(self.am + 1)
        self.parameters = {}
        self.reg_coefficients = {}
        self.to_string = "BGe Score"
    
    ##
    ## Compute the marginal likelihood for the data
    ####################################################################
    def compute(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        
        res = {}
        res['score'] = 1
        
        # number of edges of graph
        #res['score']= 0#len(self.graph.edges)
        
        res['parameters'] = {}
        return res
    
    def compute_node(self, node):
        """_summary_

        Returns:
            _type_: _description_
        """
        
        res = {}
        res['score'] = 1
        
        # number of edges of graph
        #res['score']= 0#len(self.graph.edges)
        
        res['parameters'] = {}
        return res
    
    
    # GETTERS AND SETTERS
    #####################################################

    def get_graph(self):
        return self.graph
    
    def set_graph(self, graph):
        self.graph = graph
        
    def get_am(self):
        return self.am
    
    def set_am(self, am):
        self.am = am
        
    def get_aw(self):
        return self.aw
    
    def set_aw(self, aw):
        self.aw = aw
        
    def get_t(self):
        return self.t
    
    def set_t(self, t):
        self.t = t
        
    def get_parameters(self):
        return self.parameters
    
    def set_parameters(self, parameters):
        self.parameters = parameters
        
    def get_reg_coefficients(self):
        return self.reg_coefficients
    
    def set_reg_coefficients(self, reg_coefficients):
        self.reg_coefficients = reg_coefficients
