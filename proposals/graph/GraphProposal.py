
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
        
        self.G_curr = G_curr.copy()
        self.G_prop = None
        
        self.G_curr_neigh = -1
        self.G_prop_neigh = -1
        
        num_nodes = len(G_curr.nodes())
        
        # If forbidden_adj_mat is None, create a matrix of zeros with the dimensions of the graph's adjacency matrix
        if blacklist is None:
            self.blacklist = np.zeros((num_nodes, num_nodes))
        else:
            self.blacklist = blacklist
        
        if whitelist is None:
            self.whitelist = np.zeros((num_nodes, num_nodes))
        else:
            self.whitelist = whitelist
        
        self.operations = ["add_edge", "delete_edge", "reverse_edge"]
        

    def random_index_from_ones(self, matrix):
        """ Return a random index where the matrix element is 1. """
        ones_indices = np.argwhere(matrix == 1)
        if len(ones_indices) == 0:
            return None  # Return None if there are no 1s in the matrix.
        random_choice = np.random.choice(len(ones_indices))
        return ones_indices[random_choice]

    def propose_neighbor_by_addition(self, indx_mat, incidence ):
        
        # sample an edge from the indx_mat that has a value of 1
        try:
            r, c = self.random_index_from_ones( indx_mat )
        except:
            print("Incidence matrix")
            print(incidence)
            print("Index matrix")
            print( indx_mat)
            raise Exception("The incidence matrix is not valid!")

        #incidence = incidence.values
        
        # update the incidence matrix
        new_incidence = incidence.copy()
        
        new_incidence[r, c] = 1
        
        return new_incidence
        
    def propose_neighbor_by_reverse(self, indx_mat, incidence ):
        
        incidence = incidence.values
        
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

    def propose_neighbor_by_deletion(self, indx_mat, incidence ):
        
        incidence = incidence.values
        
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
    
    def compute_nbhood(self, graph):    
            
            # if graph is a networkx type, then extract the adjacency matrix
            if isinstance(graph, nx.DiGraph):
                incidence = get_adjacency_matrix(graph)
                incidence = incidence.values
            else:
                raise Exception("The graph is not a networkx type!")
            
            # get information from the graph
            num_nodes = len( graph.nodes() )
            num_parents = num_nodes - 1
            ancestor = compute_ancestor_matrix( graph )
            
            # Matrices used in further computations
            fullmatrix = create_ones_matrix( num_nodes )
            Ematrix = create_identity_matrix( num_nodes )
            
            # 1.) Number of neighbour graphs obtained by edge deletions
            num_deletion = np.sum(incidence)
            deletion = incidence.copy() - self.whitelist
            
            # 2.) Number of neighbour graphs obtained by edge additions
            add = fullmatrix - Ematrix - incidence  - ancestor  - self.blacklist
            
            add[add < 0] = 0
            try:
                indx = np.where( np.sum(incidence, axis = 0) > num_parents - 1 )[0][0]
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
            
            return currentnbhood, deletion, add, reversal

    # PROP_GCURR_GPROP
    ########################################################################################### 
    def prob_Gcurr_Gprop_f( self ):
        
        num_neigh, _, _, _ = self.compute_nbhood( self.G_prop )
        self.G_prop_neigh = num_neigh
        
        # Q(G_curr -> G_prop) = 1 / (number of neighbors of G_prop)
        self.prob_Gcurr_Gprop =  1 / self.G_prop_neigh
        
        return self.prob_Gcurr_Gprop

    # PROB_GPROP_GCURR
    ########################################################################################### 
    def prob_Gprop_Gcurr_f(self):

        # Q(G_prop -> G_curr) = 1 / (number of neighbors of G_curr)
        self.prob_Gprop_Gcurr = 1 / self.G_curr_neigh
        
        return self.prob_Gprop_Gcurr
	
    # PROPOSE_DAG
    ###########################################################################################    
    def propose_DAG(self):
        
        # get the number of nodes
        num_nodes = len(self.G_curr.nodes())
        
        # TODO: improve this block: whitelists must be provided in networkx format
        # add the whitelist to the current incidence matrix
        incidence = get_adjacency_matrix( self.G_curr )
        incidence = update_matrix(incidence.values,  incidence.values +  self.whitelist ) 
        incidence = pd.DataFrame(incidence, columns = self.G_curr.nodes(), index = self.G_curr.nodes())
        self.G_curr = nx.DiGraph(incidence)
        
        # compute all possible neighbours
        num_neighbours, del_indx_mat, add_indx_mat, rev_indx_mat = self.compute_nbhood(self.G_curr)
        self.G_curr_neigh = num_neighbours

        # get the adjacency matrix of the current graph
        incidence = incidence.values
        
        is_zero_matrix = np.all(incidence == 0)         # check if incidence is just zeros
        is_add_zero_matrix = np.all(add_indx_mat == 0)  # check if add_indx_mat is just zeros
        is_rev_zero_matrix = np.all(rev_indx_mat == 0)  # check if rev_indx_mat is just zeros

        if len(self.G_curr.edges()) == 0 or is_zero_matrix:
            self.operations = ["add_edge"]
        elif is_add_zero_matrix:
            self.operations = ["delete_edge", "reverse_edge"]
        elif is_rev_zero_matrix and  (not is_zero_matrix):
            self.operations = ["add_edge", "delete_edge"]
        else:
            self.operations = ["add_edge", "delete_edge", "reverse_edge"]
        
        operation = random.choice(self.operations)
        
        # initialise new_incidence as zeros NxN
        new_incidence = np.zeros((num_nodes, num_nodes))
        
        if operation == "add_edge":
            new_incidence = self.propose_neighbor_by_addition( add_indx_mat, incidence )
        elif operation == "delete_edge":
            new_incidence = self.propose_neighbor_by_deletion( del_indx_mat, incidence )
        elif operation == "reverse_edge":
            new_incidence = self.propose_neighbor_by_reverse( rev_indx_mat, incidence)
        else:
            raise Exception(f"The operation '{operation}' is not valid!")
        
        # Select a random neighbor
        new_incidence = pd.DataFrame( new_incidence, columns = list(self.G_curr.nodes()) )
        self.G_prop = convert_adj_mat_to_graph( new_incidence )
        self.operation = operation
        
        # compute the proppsal distribution
        self.prob_Gcurr_Gprop_f( )
        self.prob_Gprop_Gcurr_f( )
        
        return self.G_prop, operation
    


    # GET_GCURR
    ###########################################################################################
    
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
    