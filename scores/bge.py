from abc import ABC, abstractmethod

import numpy as np

from scores.ScoreAbstract import Score

class BGeScore(Score):
    
    ##
    ## Constructor
    ####################################################################
    def __init__(self, data : pd.DataFrame, incidence : np.ndarray, isLogSpace = True):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
            graph (nx.DiGraph, optional): _description_. Defaults to None.
        """
        super().__init__(data, incidence)
        
        self.incidence = incidence
        self.node_labels = list(data.columns)
        
        # node_label to node_indx
        self.node_label_to_indx = {node_label: indx for indx, node_label in enumerate(self.node_labels)}
        
        self.num_cols = data.shape[1] # number of variables
        self.num_obvs = data.shape[0] # number of observations
        self.mu0 = np.zeros(self.num_cols) 

        # Scoring parameters.
        self.am = 1
        self.aw = self.num_cols + self.am + 1
        T0scale = self.am * (self.aw - self.num_cols - 1) / (self.am + 1)
        self.T0 = T0scale * np.eye(self.num_cols)
        self.TN = (
            self.T0 + (self.num_obvs - 1) * np.cov(data.T) + ((self.am * self.num_obvs) / (self.am + self.num_obvs))
            * np.outer(
                (self.mu0 - np.mean(data, axis=0)), (self.mu0 - np.mean(data, axis=0))
            )
        )
        self.awpN = self.aw + self.num_obvs
        self.constscorefact = - (self.num_obvs / 2) * np.log(np.pi) + 0.5 * np.log(self.am / (self.am + self.num_obvs))
        self.scoreconstvec = np.zeros(self.num_cols)
        for i in range(self.num_cols):
            awp = self.aw - self.num_cols + i + 1
            self.scoreconstvec[i] = (
                self.constscorefact
                - lgamma(awp / 2)
                + lgamma((awp + self.num_obvs) / 2)
                + (awp + i) / 2 * np.log(T0scale)
            )
            
        self.isLogSpace = isLogSpace
        self.t = T0scale
        self.parameters = {}
        self.reg_coefficients = {}
    
    ##
    ## Compute the marginal likelihood for the data
    ####################################################################
    def compute(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.compute_BGe_with_graph() #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    def compute_node(self, node : str):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.compute_BGe_with_node( node ) #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    
    def compute_BGe_with_graph(self):
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node
        
        # Loop through each node in the graph
        for node in self.node_labels:
            
            log_ml_node = self.compute_BGe_with_node( node )['score']
            
            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'parents': self.find_parents(self.incidence, self.node_label_to_indx[node])
            }
            
            total_log_ml += log_ml_node
        
        # save the parameters
        self.parameters = parameters
        
        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score
    
    def compute_BGe_with_node(self, node : str):
        
        parameters = {}  # Dictionary to store the parameters for each node
        
        node_indx = self.node_label_to_indx[node]
        parentnodes = self.find_parents(self.incidence, node_indx)
        num_parents = len(parentnodes) # number of parents
        
        awpNd2 = (self.awpN - self.num_cols + num_parents + 1) / 2
        
        A = self.TN[node_indx, node_indx]
        
        if num_parents == 0:  # just a single term if no parents
            corescore = self.scoreconstvec[num_parents] - awpNd2 * np.log(A)
        else:
            D = self.TN[np.ix_(parentnodes, parentnodes)]
            choltemp = np.linalg.cholesky(D)
            logdetD = 2 * np.sum(np.log(np.diag(choltemp)))

            B = self.TN[np.ix_([node_indx], parentnodes)]
            logdetpart2 = np.log( A - np.sum(np.linalg.solve(choltemp, B.T)**2) )
            corescore = self.scoreconstvec[num_parents] - awpNd2 * logdetpart2 - logdetD / 2
        
        # Save the parameters for the node
        parameters[node] = {
            'parents': parentnodes
        }
        self.parameters = parameters
        
        score = {
            'score': corescore,
            'parameters': parameters
            }

        return score
    
    def find_parents(self, adj_matrix: np.ndarray, node: int):
        """
    Find the indices of parent nodes for a given node in a directed graph.

    :param adj_matrix: numpy array representing the adjacency matrix of the graph
    :param node: index of the node to find parents for
    :return: list of indices of parent nodes of the specified node
    """
        if node < 0 or node >= adj_matrix.shape[0]:
            raise ValueError("Node index out of bounds")

        # Find the indices of non-zero entries in the node's column
        parent_indices = np.nonzero(adj_matrix[:, node])[0]
        
        return parent_indices.tolist()
    
    
    # GETTERS AND SETTERS
    #####################################################

    
    def to_string():
        return "BGe Score"
    
    def get_to_string(self):
        return "BGe Score"
    
    def get_incidence(self):
        return self.incidence
    
    def set_incidence(self, adj_matrix):
        self.incidence = adj_matrix
    
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
