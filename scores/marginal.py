from abc import ABC, abstractmethod
import networkx as nx

import pandas as pd
import numpy as np

from scipy.special import gammaln
from scipy.stats import invgamma
from scipy.stats import multivariate_normal

import statsmodels.api as sm

from scores.ScoreAbstract import Score


##############################################################################################
#
#  MARGLINAL LOG LIKELIHOOD SCORE
#
##############################################################################################
class MarginalLogLikelihood(Score):
    
    def __init__(self, data : pd.DataFrame, graph : nx.DiGraph = None, a : float = 1, b : float = 1, isLogSpace = True):
        super().__init__(data)
        self.graph = graph
        self.a = a
        self.b = b
        self.isLogSpace = isLogSpace
        self.parameters = {}
        self.reg_coefficients = {}
        self.to_string = "Marginal Log Likelihood"
    
    # Compute the marginal likelihood for the data
    def compute(self):
        return self.compute_marginal_log_likelihood_with_graph() if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    def compute_node(self, node):
        return self.compute_marginal_log_likelihood_with_node( node )
        
    
    def compute_marginal_log_likelihood_with_node(self, node):
        n, _ = self.data.shape  
        
        # For sigma^2
        a0 = self.a
        b0 = self.b
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node
        
        # Extract the data for the node
        y = self.data[node].values
        
        # If the node has parents
        if self.graph.in_degree(node) > 0:
            # Extract the data for the node's parents
            X = self.data[list(self.graph.predecessors(node))].values
        else:
            # For root nodes, X is just an intercept term
            X = np.ones((len(y), 1))
            
        if self.graph.in_degree(node) > 0:
            X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        
        # Setting up the Priors for beta    
        p_node = X.shape[1]             # Number of predictors for this node + intercept
        Lambda0 = np.eye(p_node)*0.1   	# Prior precision matrix
        m0 = np.zeros(p_node)       	# Prior mean vector
        
        # Bayesian Linear Regression
        # Compute the posterior precision matrix Lambda_n for beta
        Lambda_n = Lambda0 + X.T @ X
        
        # Compute the posterior mean m_n for beta
        # if the matrix is singular, then this will give an error
        # so shall we use regularization techniques to avoid this?
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        #beta_hat = np.linalg.inv(X.T @ X + regularizer * np.eye(X.shape[1])) @ X.T @ y
        m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ beta_hat + Lambda0 @ m0)
        
        # Compute a_n and b_n for sigma^2
        a_n = a0 + n/2
        b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)
        
        
        # Compute the Marginal Likelihood for this node and add to total
        log_ml_node = ( - (len(y)/2) * np.log(2*np.pi) 
                        + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1]) 
                        + a0 * np.log(b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(a0) )
        
        # Save the parameters for the node
        parameters[node] = {
            'score' : total_log_ml,
            'Lambda_n': Lambda_n,
            'm_n': m_n,
            'a_n': a_n,
            'b_n': b_n
        }
        
        total_log_ml += log_ml_node

        # compute the regression coefficients based on the parameters
        # self.reg_coefficients = self.recover_regr_paramns(parameters)
        
        # save the parameters
        self.parameters = parameters
        
        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score

    # Compute the marginal likelihood conditioned on a graph structure
    ####################################################################
    def compute_marginal_log_likelihood_with_graph(self):
        n, _ = self.data.shape  
        
        # For sigma^2
        a0 = self.a
        b0 = self.b
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node
        
        # Loop through each node in the graph
        for node in self.graph.nodes():
            
            # Extract the data for the node
            y = self.data[node].values
            
            # If the node has parents
            if self.graph.in_degree(node) > 0:
                # Extract the data for the node's parents
                X = self.data[list(self.graph.predecessors(node))].values
            else:
                # For root nodes, X is just an intercept term
                X = np.ones((len(y), 1))
                
            if self.graph.in_degree(node) > 0:
                X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
            
            # Setting up the Priors for beta    
            p_node = X.shape[1]             # Number of predictors for this node + intercept
            Lambda0 = np.eye(p_node)*0.1   	# Prior precision matrix
            m0 = np.zeros(p_node)       	# Prior mean vector
            
            # Bayesian Linear Regression
            # Compute the posterior precision matrix Lambda_n for beta
            Lambda_n = Lambda0 + X.T @ X
            
            # Compute the posterior mean m_n for beta
            # if the matrix is singular, then this will give an error
            # so shall we use regularization techniques to avoid this?
            beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
            #beta_hat = np.linalg.inv(X.T @ X + regularizer * np.eye(X.shape[1])) @ X.T @ y
            m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ beta_hat + Lambda0 @ m0)
            
            # Compute a_n and b_n for sigma^2
            a_n = a0 + n/2
            b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)
            
            
            # Compute the Marginal Likelihood for this node and add to total
            log_ml_node = ( - (len(y)/2) * np.log(2*np.pi) 
                            + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1]) 
                            + a0 * np.log(b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(a0) )
            
            
            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'Lambda_n': Lambda_n,
                'm_n': m_n,
                'a_n': a_n,
                'b_n': b_n
            }
            
            total_log_ml += log_ml_node

        # compute the regression coefficients based on the parameters
        # self.reg_coefficients = self.recover_regr_paramns(parameters)
        
        # save the parameters
        self.parameters = parameters
        
        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score
    
    
    # Compute the marginal likelihood directly from data
    ####################################################################
    def compute_marginal_log_likelihood_from_data( self):
        n, _ = self.data.shape  
        
        # For sigma^2
        a0 = self.a
        b0 = self.b
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each variable
        
        # Loop through each variable in the data
        for var in self.data.columns:
            
            # Extract the data for the variable
            y = self.data[var].values
            
            # Since we're not using a graph, we'll assume a simple model with just an intercept term for each variable
            X = np.ones((len(y), 1))
            
            # Setting up the Priors for beta    
            p_var = X.shape[1]             
            Lambda0 = np.eye(p_var) * 0.1  
            m0 = np.zeros(p_var)            
    
            # Bayesian Linear Regression
            Lambda_n = Lambda0 + X.T @ X
            
            # Compute the posterior mean m_n for beta
            m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ np.linalg.inv(X.T @ X) @ X.T @ y + Lambda0 @ m0)
            
            # Compute a_n and b_n for sigma^2
            a_n = a0 + n / 2
            b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)
            
            # Compute the Marginal Likelihood for this variable and add to total
            log_ml_var = ( - (len(y)/2) * np.log(2 * np.pi) 
                        + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1]) 
                        + a0 * np.log(b0) - a_n * np.log(b_n) 
                        + gammaln(a_n) - gammaln(a0) )
            
            
            # Save the parameters for the variable
            parameters[var] = {
                'score' : total_log_ml,
                'Lambda_n': Lambda_n,
                'm_n': m_n,
                'a_n': a_n,
                'b_n': b_n,
            }
            
            total_log_ml += log_ml_var

        # compute the regression coefficients based on the parameters
        #self.reg_coefficients = self.recover_regr_paramns(parameters)
        
        # Return the total marginal likelihood and the parameters
        score = {
            'score': total_log_ml,
            'parameters': parameters
        }
        return score

    # Recover the regression coefficients beta and sigma^2 from the parameters a_n and b_n, lambda_n and mu_n
    ####################################################################
    def recover_regr_paramns(self, parameters : dict):
        """Given the parameters of the marginal likelihood function a_n, b_n, lambda_n and mu_n, returns the regression coefficients beta and the regression sigma^2"""
        
        sampled_values = {}
        for node, param in parameters.items():
            # Sample sigma^2 from the inverse gamma distribution
            sigma2 = invgamma.rvs(a=param['a_n'], scale=  param['b_n'])
            
            # Sample beta from the multivariate normal distribution
            cov_matrix = sigma2 * np.linalg.inv(param['Lambda_n'])
            beta = multivariate_normal.rvs(mean=param['m_n'], cov=cov_matrix)

            sampled_values[node] = {
                'beta': beta,
                'sigma2': sigma2
            }
        
        self.reg_coefficients = sampled_values
        return sampled_values
    
    
    # GETTERS AND SETTERS
    #####################################################

    def get_a(self):
        return self.a
    
    def get_b(self):
        return self.b
    
    def set_a(self, a):
        self.a = a
        
    def set_b(self, b):
        self.b = b
        
    def get_parameters(self):
        return self.parameters
    
    def set_parameters(self, parameters):
        self.parameters = parameters
        
    def get_reg_coefficients(self):
        return self.reg_coefficients
    
    def set_reg_coefficients(self, reg_coefficients):
        self.reg_coefficients = reg_coefficients
    
    # getters and setters
    #####################################################
    
    def get_graph(self):
        return self.graph
    
    def set_graph(self, graph):
        self.graph = graph
        
    def get_data(self):
        return self.data
    
    def set_data(self, data):
        self.data = data
        
    def get_isLogSpace(self):
        return self.isLogSpace
    
    def set_isLogSpace(self, isLogSpace):
        self.isLogSpace = isLogSpace


