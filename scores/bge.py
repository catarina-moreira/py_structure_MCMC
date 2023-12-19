from abc import ABC, abstractmethod
import networkx as nx

import pandas as pd
import numpy as np

from scipy.special import gammaln
from scipy.stats import invgamma
from scipy.stats import multivariate_normal

import statsmodels.api as sm

from scores.ScoreAbstract import Score



class BGEscoreEfficient(Score):
    
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
        return self.compute_BGe_with_graph() #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    def compute_node(self, node : str):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.compute_BGe_with_node( node ) #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    ##
    ## Compute the marginal likelihood conditioned on a graph structure
    ####################################################################
    def compute_BGe_with_graph(self):
        n, cols = self.data.shape  
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node
        
        # Loop through each node in the graph
        for node in self.graph.nodes():
            
            parents = self.graph.in_degree(node)
            a0 = (self.aw - cols + parents + 1)/2
            b0 = self.t/2

            # Extract the data for the node
            y = self.data[node].values

            # If the node has parents
            if self.graph.in_degree(node) > 0:
                
                # Extract the data for the node's parents
                X = self.data[list(self.graph.predecessors(node))].values 
                X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)    # Add an intercept term
                
                p_node = X.shape[1]                 # Number of predictors for this node + intercept
                
                # when the node has parents, the precision matrix should have the form
                # [[am, 0, 0, 0],
                #  [0 , t, 0, 0],
                #  [0 , 0, ..., 0],
                #  [0 , 0, 0, t]]
                Lambda0 =  np.eye(p_node)*self.t 	# Prior precision matrix
                Lambda0[0,0] = self.am
            
            # If the node has not parents, then just keep the 
            # intercept term for X and adjust the precision matrix
            else:
                X = np.ones((len(y), 1))
                Lambda0 = np.eye(1)*self.am
                
            # Setting up the Priors for beta
            p_node = X.shape[1]    
            m0 = np.zeros(p_node)       	    # Prior mean vector
            
            # Bayesian Linear Regression
            # Compute the posterior precision matrix Lambda_n for beta
            Lambda_n = Lambda0 + X.T @ X
            
            # Compute the posterior mean m_n for beta
            # if the matrix is singular, then this will give an error
            # so shall we use regularization techniques to avoid this?
            beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
            m_n = np.linalg.pinv(Lambda_n) @ (X.T @ X @ beta_hat + Lambda0 @ m0)
            
            # Compute a_n and b_n for sigma^2
            a_n = a0 + n/2
            b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)
            
            
            
            # Compute the Marginal Likelihood for this node and add to total
            # log\_ml\_node = -\left(\frac{{\text{{len}}(y)}}{2}\right) \log(2\pi) + \frac{1}{2} \left( \log|\Lambda_0| - \log|\Lambda_n| \right) + a_0 \log(b_0) - a_n \log(b_n) + \Gamma(a_n) - \Gamma(a_0)            
            log_ml_node = ( - (len(y)/2) * np.log(2*np.pi) 
                            + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1]) 
                            + a0 * np.log(b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(a0) )
            
            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'Lambda_n': Lambda_n,
                'm_n': m_n,
                'a_0': a0,
                'b_0': b0,
                'a_n': a_n,
                'b_n': b_n, 
                'am': self.am,
                'aw': self.aw,
                't': self.t,
                'parents': parents
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
    
    
    def compute_BGe_with_node(self, node : str):
        n, cols = self.data.shape  
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node

        parents = self.graph.in_degree(node)
        a0 = (self.aw - cols + parents + 1)/2
        b0 = self.t/2

        # Extract the data for the node
        y = self.data[node].values

        # If the node has parents
        if self.graph.in_degree(node) > 0:
            
            # Extract the data for the node's parents
            X = self.data[list(self.graph.predecessors(node))].values 
            X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)    # Add an intercept term
            
            p_node = X.shape[1]                 # Number of predictors for this node + intercept
            
            # when the node has parents, the precision matrix should have the form
            # [[am, 0, 0, 0],
            #  [0 , t, 0, 0],
            #  [0 , 0, ..., 0],
            #  [0 , 0, 0, t]]
            Lambda0 =  np.eye(p_node)*self.t 	# Prior precision matrix
            Lambda0[0,0] = self.am
        
        # If the node has not parents, then just keep the 
        # intercept term for X and adjust the precision matrix
        else:
            X = np.ones((len(y), 1))
            Lambda0 = np.eye(1)*self.am
            
        # Setting up the Priors for beta
        p_node = X.shape[1]    
        m0 = np.zeros(p_node)       	    # Prior mean vector
        
        # Bayesian Linear Regression
        # Compute the posterior precision matrix Lambda_n for beta
        Lambda_n = Lambda0 + X.T @ X
        
        # Compute the posterior mean m_n for beta
        # if the matrix is singular, then this will give an error
        # so shall we use regularization techniques to avoid this?
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ beta_hat + Lambda0 @ m0)
        
        # Compute a_n and b_n for sigma^2
        a_n = a0 + n/2
        b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)
        
        
        
        # Compute the Marginal Likelihood for this node and add to total
        # log\_ml\_node = -\left(\frac{{\text{{len}}(y)}}{2}\right) \log(2\pi) + \frac{1}{2} \left( \log|\Lambda_0| - \log|\Lambda_n| \right) + a_0 \log(b_0) - a_n \log(b_n) + \Gamma(a_n) - \Gamma(a_0)            
        log_ml_node = ( - (len(y)/2) * np.log(2*np.pi) 
                        + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1]) 
                        + a0 * np.log(b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(a0) )
        
        # Save the parameters for the node
        parameters[node] = {
            'score' : log_ml_node,
            'Lambda_n': Lambda_n,
            'm_n': m_n,
            'a_0': a0,
            'b_0': b0,
            'a_n': a_n,
            'b_n': b_n, 
            'am': self.am,
            'aw': self.aw,
            't': self.t,
            'parents': parents
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





class BGEscore(Score):

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
        return self.compute_BGe_with_graph() #if self.graph is not None else self.compute_marginal_log_likelihood_from_data()
    
    ##
    ## Compute the marginal likelihood conditioned on a graph structure
    ####################################################################
    def compute_BGe_with_graph(self):
        n, cols = self.data.shape  
        
        total_log_ml = 0
        parameters = {}  # Dictionary to store the parameters for each node
        
        # Loop through each node in the graph
        for node in self.graph.nodes():
            
            parents = self.graph.in_degree(node)
            a0 = (self.aw - cols + parents + 1)/2
            b0 = self.t/2

            # Extract the data for the node
            y = self.data[node].values

            # If the node has parents
            if self.graph.in_degree(node) > 0:
                
                # Extract the data for the node's parents
                X = self.data[list(self.graph.predecessors(node))].values 
                X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)    # Add an intercept term
                
                p_node = X.shape[1]                 # Number of predictors for this node + intercept
                
                # when the node has parents, the precision matrix should have the form
                # [[am, 0, 0, 0],
                #  [0 , t, 0, 0],
                #  [0 , 0, ..., 0],
                #  [0 , 0, 0, t]]
                Lambda0 =  np.eye(p_node)*self.t 	# Prior precision matrix
                Lambda0[0,0] = self.am
            
            # If the node has not parents, then just keep the 
            # intercept term for X and adjust the precision matrix
            else:
                X = np.ones((len(y), 1))
                Lambda0 = np.eye(1)*self.am
                
            # Setting up the Priors for beta
            p_node = X.shape[1]    
            m0 = np.zeros(p_node)       	    # Prior mean vector
            
            # Bayesian Linear Regression
            # Compute the posterior precision matrix Lambda_n for beta
            Lambda_n = Lambda0 + X.T @ X
            
            # Compute the posterior mean m_n for beta
            # if the matrix is singular, then this will give an error
            # so shall we use regularization techniques to avoid this?
            beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
            m_n = np.linalg.inv(Lambda_n) @ (X.T @ X @ beta_hat + Lambda0 @ m0)
            
            # Compute a_n and b_n for sigma^2
            a_n = a0 + n/2
            b_n = b0 + 0.5 * (y.T @ y + m0.T @ Lambda0 @ m0 - m_n.T @ Lambda_n @ m_n)
            
            
            
            # Compute the Marginal Likelihood for this node and add to total
            # log\_ml\_node = -\left(\frac{{\text{{len}}(y)}}{2}\right) \log(2\pi) + \frac{1}{2} \left( \log|\Lambda_0| - \log|\Lambda_n| \right) + a_0 \log(b_0) - a_n \log(b_n) + \Gamma(a_n) - \Gamma(a_0)            
            log_ml_node = ( - (len(y)/2) * np.log(2*np.pi) 
                            + 0.5 * (np.linalg.slogdet(Lambda0)[1] - np.linalg.slogdet(Lambda_n)[1]) 
                            + a0 * np.log(b0) - a_n * np.log(b_n) + gammaln(a_n) - gammaln(a0) )
            
            # Save the parameters for the node
            parameters[node] = {
                'score' : log_ml_node,
                'Lambda_n': Lambda_n,
                'm_n': m_n,
                'a_0': a0,
                'b_0': b0,
                'a_n': a_n,
                'b_n': b_n, 
                'am': self.am,
                'aw': self.aw,
                't': self.t,
                'parents': parents
            }
            
            total_log_ml += log_ml_node

        # compute the regression coefficients based on the parameters
        self.reg_coefficients = self.recover_regr_paramns(parameters)
        
        # save the parameters
        self.parameters = parameters
        
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
        
    