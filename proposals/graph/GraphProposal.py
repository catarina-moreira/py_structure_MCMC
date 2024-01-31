
import networkx as nx
import numpy as np
import pandas as pd

from cache import Cache

import random

from collections import Counter

from utils.graph_utils import *

from proposals.graph.StructureLearning import StructureLearningProposal


##############################################################################################
#
#  BASE PROPOSAL
#
##############################################################################################
class GraphProposalUniform(StructureLearningProposal):
    
    # __init__
    ########################################################################################### 
    #
    #v the forbiden arc list should be represented ass an adjacency matrix
    # the mandatory arc list should be represented as an adjancency matrix
    def __init__(self, G_curr : nx.DiGraph, blacklist = None, whitelist = None):
        super().__init__(G_curr, blacklist, whitelist) # initialize the parent class
        
        
        self.node_labels = G_curr.nodes()
        self.num_nodes = len(G_curr.nodes())
        
        self.max_parents = self.num_nodes - 1
        
        self.node_label_to_indx = {node_label: indx for indx, node_label in enumerate(self.node_labels)}
        self.indx_to_node_label = {indx: node_label for indx, node_label in enumerate(self.node_labels)}
        
        self.G_curr = nx.adjacency_matrix(G_curr).toarray()
        self.G_prop = None
        
        self.G_curr_neigh = -1
        self.G_prop_neigh = -1
        
        # If forbidden_adj_mat is None, create a matrix of zeros with the dimensions of the graph's adjacency matrix
        if blacklist is None:
            self.blacklist = np.zeros((self.num_nodes, self.num_nodes))
        else:
            self.blacklist = blacklist
        
        if whitelist is None:
            self.whitelist = np.zeros((self.num_nodes, self.num_nodes))
        else:
            self.whitelist = whitelist
        
        self.operations = ["add_edge", "delete_edge", "reverse_edge", "stay_still"]
        

    def random_index_from_ones(self, matrix : np.ndarray ):
        """ Return a random index where the matrix element is 1. """
        ones_indices = np.argwhere(matrix == 1)
        if len(ones_indices) == 0:
            return None  # Return None if there are no 1s in the matrix.
        random_choice = np.random.choice(len(ones_indices))
        return ones_indices[random_choice]

    def propose_neighbor_by_addition(self, indx_mat : np.ndarray, incidence : np.ndarray ):
        
        # sample an edge from the indx_mat that has a value of 1
        try:
            r, c = self.random_index_from_ones( indx_mat )
        except:
            print("Incidence matrix")
            print(incidence)
            print("Index matrix")
            print( indx_mat)
            raise Exception("The incidence matrix is not valid!")
        
        # update the incidence matrix
        new_incidence = incidence.copy()
        
        new_incidence[r, c] = 1
        
        return new_incidence
        
    def propose_neighbor_by_reverse(self, indx_mat : np.ndarray, incidence : np.ndarray ):
        
        
        # sample an edge from the indx_mat that has a value of 1
        # depending if the indx matrix is addition, reversal or deletion,
        # this will pick up an edge to add, reverse or delete
        try:
            r, c = self.random_index_from_ones( indx_mat )
        except:
            print("Incidence matrix")
            print(incidence)
            print("Index matrix")
            print( indx_mat)
            raise Exception("The incidence matrix is not valid!")
        
        # update the incidence matrix
        new_incidence = incidence.copy()
        new_incidence[r, c] = 0
        new_incidence[c, r] = 1
        
        return new_incidence

    def propose_neighbor_by_deletion(self, indx_mat : np.ndarray, incidence : np.ndarray ):
        
        
        # sample an edge from the indx_mat that has a value of 1
        # depending if the indx matrix is addition, reversal or deletion,
        # this will pick up an edge to add, reverse or delete
        try:
            r, c = self.random_index_from_ones( indx_mat )
        except:
            print("Incidence matrix")
            print(incidence)
            print("Index matrix")
            print( indx_mat)
            raise Exception("The incidence matrix is not valid!")
        
    
        # update the incidence matrix
        new_incidence = incidence.copy()
        new_incidence[r, c] = 0
        
        return new_incidence
    
    def compute_nbhood(self, incidence : np.ndarray):    
            
            ancestor = compute_ancestor_matrix( incidence )
            
            # Matrices used in further computations
            fullmatrix = create_ones_matrix( self.num_nodes )
            Ematrix = create_identity_matrix( self.num_nodes )
            
            # 1.) Number of neighbour graphs obtained by edge deletions
            deletion = incidence.copy() - self.whitelist
            num_deletion = np.sum(deletion)
            
            
            # 2.) Number of neighbour graphs obtained by edge additions
            add = fullmatrix - Ematrix - incidence  - ancestor  - self.blacklist
            
            add[add < 0] = 0
            try:
                indx = np.where( np.sum(incidence, axis = 0) > self.max_parents - 1 )[0][0]
                add[:,indx] = 0  
                num_addition = np.sum(add)# eliminate cycles  
            except:                  
                num_addition = np.sum(add)
            
            # 3.) Number of neighbour graphs obtained by edge reversals
            reversal = (incidence - self.whitelist) - ((incidence - self.whitelist).T @ ancestor).T -  self.blacklist.T
            reversal[reversal < 0] = 0 # replace all negative values by zero
            
            try:
                reversal[indx, :] = 0
                num_reversal = np.sum(reversal)
            except:
                num_reversal = np.sum(reversal)

            # Total number of neighbour graphs
            currentnbhood =  num_deletion + num_addition + 1 + num_reversal
            
            return currentnbhood, deletion, add, reversal, num_deletion, num_addition, num_reversal

    # PROP_GCURR_GPROP
    ########################################################################################### 
    def prob_Gcurr_Gprop_f( self ):
        
        num_neigh, _, _, _ , _, _, _ = self.compute_nbhood( self.G_prop )
        self.G_prop_neigh = num_neigh
        
        # Q(G_curr -> G_prop) = 1 / (number of neighbors of G_prop)
        self.prob_Gcurr_Gprop =  1 / self.G_prop_neigh
        
        return self.prob_Gcurr_Gprop

    # PROB_GPROP_GCURR
    ########################################################################################### 
    def prob_Gprop_Gcurr_f( self ):

        # Q(G_prop -> G_curr) = 1 / (number of neighbors of G_curr)
        self.prob_Gprop_Gcurr = 1 / self.G_curr_neigh
        
        return self.prob_Gprop_Gcurr
	
    # PROPOSE_DAG
    ###########################################################################################    
    def propose_DAG(self):

        # add the whitelist to the current incidence matrix
        self.G_curr  = update_matrix(self.G_curr ,  self.G_curr  +  self.whitelist  ) 
        
        # compute all possible neighbours
        num_neighbours, del_indx_mat, add_indx_mat, rev_indx_mat, num_deletion, num_addition, num_reversal = self.compute_nbhood(self.G_curr)
        
        # set the number of neighbours
        self.G_curr_neigh = num_neighbours
        
        # is_zero_matrix = np.all(self.G_curr == 0)         # check if incidence is just zeros
        # if is_zero_matrix:
        #     operation = "add_edge"
        # else:
        # Randomly sample an operation
        operation = np.random.choice(self.operations, size=1, p = [num_addition/num_neighbours, num_deletion/num_neighbours, num_reversal/num_neighbours, 1/num_neighbours])
        
        # initialise new_incidence as zeros NxN
        self.G_prop  = np.zeros((self.num_nodes, self.num_nodes))
    
        if operation == "add_edge":
            self.G_prop  = self.propose_neighbor_by_addition( add_indx_mat, self.G_curr )
        elif operation == "delete_edge":
            self.G_prop  = self.propose_neighbor_by_deletion( del_indx_mat, self.G_curr )
        elif operation == "reverse_edge":
            self.G_prop  = self.propose_neighbor_by_reverse( rev_indx_mat, self.G_curr)
        elif operation == "stay_still":
            self.G_prop  = self.G_curr
        else:
            raise Exception(f"The operation '{operation}' is not valid!")
        
        self.operation = operation
        
        # compute the proppsal distribution
        self.prob_Gcurr_Gprop_f( )
        self.prob_Gprop_Gcurr_f( )
        
        return self.G_prop, operation
    


    # GET_GCURR
    ###########################################################################################
    
    def get_node_labels(self):
        return self.node_labels
    
    def set_node_labels(self, node_labels):
        self.node_labels = node_labels
        
    def get_num_nodes(self):
        return self.num_nodes
    
    def set_num_nodes(self, num_nodes):
        self.num_nodes = num_nodes
        
    def get_max_parents(self):
        return self.max_parents
    
    def set_max_parents(self, max_parents):
        self.max_parents = max_parents
        
    def get_node_label_to_indx(self):
        return self.node_label_to_indx
    
    def get_indx_to_node_label(self):
        return self.indx_to_node_label
    
    def set_indx_to_node_label(self, indx_to_node_label):
        self.indx_to_node_label = indx_to_node_label
    
    def set_node_label_to_indx(self, node_label_to_indx):
        self.node_label_to_indx = node_label_to_indx
    
    def get_G_curr_neigh(self):
        return self.G_curr_neigh

    def set_G_curr_neigh(self, G_curr_neigh):
        self.G_curr_neigh = G_curr_neigh
        
    def get_G_prop_neigh(self):
        return self.G_prop_neigh
    
    def set_G_prop_neigh(self, G_prop_neigh):
        self.G_prop_neigh = G_prop_neigh
    
    def get_G_curr(self):
        return self.G_curr
    
    def set_G_curr(self, G_curr):
        self.G_curr = G_curr
        
    def update_G_curr(self, G):
        self.set_G_curr( G )
        
    def get_G_prop(self):
            return self.G_curr
    
    def set_G_prop(self, G):
        self.G_prop = G

    # GET_OPERATION
    ###########################################################################################
    def get_operation(self):
        return self.operation   
    
    def set_operation(self, operation):
        self.operation = operation
    
    # GET_PROB_GCURR_GPROP
    ###########################################################################################
    def get_prob_Gcurr_Gprop(self):
        return self.prob_Gcurr_Gprop
    
    def set_get_prob_Gcurr_Gprop(self, value):
        self.prob_Gcurr_Gprop = value
    
    # GET_PROB_GPROP_GCURR
    ###########################################################################################
    def get_prob_Gprop_Gcurr(self):
        return self.prob_Gprop_Gcurr
    
    def set_prob_Gprop_Gcurr(self, value):
        self.prob_Gprop_Gcurr = value
    
    
    # GET_DEN_GCURR_GPROP
    ###########################################################################################
    def get_den_Gcurr_Gprop(self):
        return self.den_Gcurr_Gprop
    
    # GET_DEN_GPROP_GCURR
    ###########################################################################################
    def get_den_Gprop_Gcurr(self):
        return self.den_Gprop_Gcurr
    
    # GET_PROB_GCURR_GPROP
    ###########################################################################################
    def get_prob_Gcurr_Gprop(self):
        return self.prob_Gcurr_Gprop
    