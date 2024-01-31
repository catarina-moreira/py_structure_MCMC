from abc import ABC, abstractmethod

import networkx as nx
import numpy as np

from itertools import combinations
from cache import Cache

import random

from collections import Counter


class StructureLearningProposal(ABC):
    
    def __init__(self, graph : nx.DiGraph, forbidden_arc_lst : list, mandatory_arc_lst : list):
        self.cache = Cache()
        
        self.cache.cache_init()
        
        self.G_curr = graph.copy()
        self.G_prop = None
        
        self.N = len(self.G_curr.nodes())
        
        self.prob_Gcurr_Gprop = None
        self.prob_Gprop_Gcurr = None
        
        self.den_Gcurr_Gprop = None
        self.den_Gprop_Gcurr = None
        
        self.forbidden_arc_lst = forbidden_arc_lst
        self.mandatory_arc_lst = mandatory_arc_lst
        
        self.graph_hash = self.cache.compute_graph_hash( graph )
        self.graph_id = self.cache.dict_hash_to_graph_id[ self.graph_hash ]
        
        self.possible_neighbors_addition = None
        self.possible_neighbors_deletion = None
        self.possible_neighbors_reversal = None
        
        self.operation = None
        
        self.cache.MCMC_proposed_graphs = []                      # dictionary to store all MCMC proposed graphs
        self.cache.MCMC_proposed_operations = []
        
    @abstractmethod
    def propose_DAG(self):
        pass
    
    @abstractmethod
    def prob_Gcurr_Gprop_f(self, G_prop : nx.DiGraph):
        pass
    
    @abstractmethod
    def prob_Gprop_Gcurr_f(self, G_prop : nx.DiGraph):
        pass


    
    
    def get_graph_hash(self):
        return self.graph_hash
    
    def get_graph_id(self):
        return self.graph_id
    
    def get_cache(self):
        return self.cache
    
    def get_current_graph(self):
        return self.G_curr
    
    def get_proposed_graph(self):
        return self.G_prop
    
    def get_forbidden_arc_list(self):
        return self.forbidden_arc_lst
    
    def get_mandatory_arc_list(self):
        return self.mandatory_arc_lst
    
    def get_possible_neighbors_addition(self):
        return self.possible_neighbors_addition
    
    def get_possible_neighbors_deletion(self):
        return self.possible_neighbors_deletion
    
    def get_possible_neighbors_reversal(self):
        return self.possible_neighbors_reversal
    
    def get_prob_Gcurr_Gprop(self):
        return self.prob_Gcurr_Gprop
    
    def get_prob_Gprop_Gcurr(self):
        return self.prob_Gprop_Gcurr
    
    def get_den_Gcurr_Gprop(self):
        return self.den_Gcurr_Gprop
    
    def get_den_Gprop_Gcurr(self):
        return self.den_Gprop_Gcurr
    
    def get_operation(self):
        return self.operation
    
    def set_graph_hash(self, graph_hash : str):
        self.graph_hash = graph_hash
        
    def set_graph_id(self, graph_id : int):
        self.graph_id = graph_id
        
    def set_cache(self, cache : Cache):
        self.cache = cache
        
    def set_current_graph(self, G_curr : nx.DiGraph):
        self.G_curr = G_curr
        
    def set_proposed_graph(self, G_prop : nx.DiGraph):
        self.G_prop = G_prop
        
    def set_forbidden_arc_list(self, forbidden_arc_lst : list):
        self.forbidden_arc_lst = forbidden_arc_lst
        
    def set_mandatory_arc_list(self, mandatory_arc_lst : list):
        self.mandatory_arc_lst = mandatory_arc_lst
        
    def set_possible_neighbors_addition(self, possible_neighbors_addition : list):
        self.possible_neighbors_addition = possible_neighbors_addition
        
    def set_possible_neighbors_deletion(self, possible_neighbors_deletion : list):
        self.possible_neighbors_deletion = possible_neighbors_deletion	
        
    def set_possible_neighbors_reversal(self, possible_neighbors_reversal : list):
        self.possible_neighbors_reversal = possible_neighbors_reversal
        
    def set_prob_Gcurr_Gprop(self, prob_Gcurr_Gprop : float):
        self.prob_Gcurr_Gprop = prob_Gcurr_Gprop
        
    def set_prob_Gprop_Gcurr(self, prob_Gprop_Gcurr : float):
        self.prob_Gprop_Gcurr = prob_Gprop_Gcurr
        
    def set_den_Gcurr_Gprop(self, den_Gcurr_Gprop : float):
        self.den_Gcurr_Gprop = den_Gcurr_Gprop
        
    def set_den_Gprop_Gcurr(self, den_Gprop_Gcurr : float):
        self.den_Gprop_Gcurr = den_Gprop_Gcurr
        
    def set_operation(self, operation : str):
        self.operation = operation
