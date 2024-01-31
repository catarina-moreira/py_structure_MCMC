from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import networkx as nx

from utils.graph_utils import *

from proposals.graph.StructureLearning import StructureLearningProposal

from scores.ScoreAbstract import Score

from mcmc.MCMCAbstract import MCMC

import random

class StructureMCMC(MCMC):

	def __init__(self, initial_graph : np.ndarray, max_iter : int, 
				proposal_object : StructureLearningProposal, 
				score_object : Score):
		
		self.score_object = score_object
		self.proposal_object = proposal_object
		self.data = score_object.get_data()
		self.num_nodes = len(self.data.columns)
		self.node_labels = list(self.data.columns)
		self.max_iter = max_iter

		# if the initial_graph is none, generate an empty graph with the same number of nodes as the data
		if initial_graph is None:
			self.initial_graph = np.zeros((self.num_nodes, self.num_nodes))
		else:
			self.initial_graph = initial_graph

		self.indx_to_node_label = self.proposal_object.get_indx_to_node_label()

	# main MCMC function that needs to be implemented
	def log_acceptance_ratio(self, posterior_Gcurr : float, posterior_Gprop : float, Q_Gcurr_Gprop : float, Q_Gprop_Gcurr : float):

		try:
			numerator = posterior_Gprop + np.log(Q_Gcurr_Gprop)
		except:
			print("RuntimeWarning: divide by zero encountered in log")
			print(f"\tposterior_Gcurr: {posterior_Gcurr}")
			print(f"\tQ_Gcurr_Gprop: {Q_Gcurr_Gprop}")
			Q_Gcurr_Gprop = 0.000001

		try:
			denominator = posterior_Gcurr + np.log(Q_Gprop_Gcurr)
		except:
			print("RuntimeWarning: divide by zero encountered in log")
			print(f"\tposterior_Gprop: {posterior_Gprop}")
			print(f"\tQ_Gprop_Gcurr: {Q_Gprop_Gcurr}")
			Q_Gprop_Gcurr = 0.000001
			
		return  min(0, numerator - denominator)

	def acceptance_ratio(self, posterior_Gcurr : float, posterior_Gprop : float, Q_Gcurr_Gprop : float, Q_Gprop_Gcurr : float):
		
		numerator = posterior_Gprop * Q_Gcurr_Gprop
		denominator = posterior_Gcurr * Q_Gprop_Gcurr
  
		return min(1, numerator/denominator)

	def run(self):

		mcmc_res = {}
		iter_indx = 0
		ACCEPT = 0

		# start with the initial current graph:
		# compute the score of the graph given the data
		G_curr = self.initial_graph.copy()
		G_curr_operation = "initial"

		# compute the score for the initial graph
		score_dict = self.score_object.compute()
		score_Gcurr = score_dict['score']

		mcmc_res[0] = {"graph": G_curr, 
						#"graph_matrix" : convert_graph_to_str(G_curr),
						"score": score_Gcurr, 
						"operation":G_curr_operation,
						"accepted" : 0,
						"Q_Gprop_Gcurr" : 1,
						"Q_Gcurr_Gprop" : 1,
						"score_Gprop" : 1,
						"acceptance_prob" : 0} 
		
		node_score_Gcurr = collect_node_scores(score_dict)

		for _ in range(self.max_iter):
			
			accept_indx = 0
			acceptance_prob = 0
			
			# propose a new graph and compute the proposal distribution Q
			self.proposal_object.update_G_curr(G_curr)
			G_prop, G_prop_operation = self.proposal_object.propose_DAG(  )

			# compute the proposal distribution Q
			Q_Gcurr_Gprop = self.proposal_object.get_prob_Gcurr_Gprop()
			Q_Gprop_Gcurr = self.proposal_object.get_prob_Gprop_Gcurr()

			if G_prop_operation == "stay_still":
				
				mcmc_res[iter_indx] = {"graph": self.proposal_object.get_G_curr(), 
									#"graph_id" : self.proposal_object.get_graph_id(),
									#"graph_matrix" : convert_graph_to_str( self.proposal_object.get_G_curr()),
									"score": score_Gcurr, 
									"operation": G_prop_operation,
									"accepted" : 0,
									"Q_Gprop_Gcurr" : Q_Gprop_Gcurr,
									"Q_Gcurr_Gprop" : Q_Gcurr_Gprop,
									"score_Gprop" : 0,
									"acceptance_prob" : 0}
					
				iter_indx += 1
				continue

			# we need to update the graph so we can extract the parents of the node
			self.score_object.set_incidence(G_prop)
			node_score_Gprop = node_score_Gcurr.copy() 

			rescored_nodes = compare_graphs(G_curr, G_prop, G_prop_operation, self.indx_to_node_label)
			
			if G_prop_operation == "add_edge" or G_prop_operation == "delete_edge":
				node_score_Gprop[rescored_nodes] = self.score_object.compute_node( rescored_nodes )['score']
			else:
				node_score_Gprop[rescored_nodes[0]] = self.score_object.compute_node( rescored_nodes[0] )['score']
				node_score_Gprop[rescored_nodes[1]] = self.score_object.compute_node( rescored_nodes[1] )['score']
	
			score_Gprop = sum( list( node_score_Gprop.values()))
			#score_dict = self.score_object.compute()
			#score_Gprop = score_dict['score']
			#params_Gprop = score_dict['parameters']


			if self.score_object.get_isLogSpace():
				acceptance_prob = self.log_acceptance_ratio(score_Gcurr, score_Gprop, Q_Gcurr_Gprop, Q_Gprop_Gcurr)
				u = np.log(np.random.uniform(0, 1))
			else:
				acceptance_prob = self.acceptance_ratio(score_Gcurr, score_Gprop, Q_Gcurr_Gprop, Q_Gprop_Gcurr)
				u =  random.uniform(0,1)


			if u < acceptance_prob:
				
				ACCEPT += 1
				G_curr = G_prop
				self.proposal_object.update_G_curr(G_prop)
				score_Gcurr = score_Gprop
				G_curr_operation = G_prop_operation
				node_score_Gcurr = node_score_Gprop.copy()
				accept_indx = 1

			
			mcmc_res[iter_indx] = {"graph": self.proposal_object.get_G_curr(), 
									#"graph_id" : self.proposal_object.get_graph_id(),
									#"graph_matrix" : convert_graph_to_str( self.proposal_object.get_G_curr()),
									"score": score_Gcurr, 
									"operation": G_curr_operation,
									"accepted" : accept_indx,
									"Q_Gprop_Gcurr" : Q_Gprop_Gcurr,
									"Q_Gcurr_Gprop" : Q_Gcurr_Gprop,
									"score_Gprop" : score_Gprop,
									"score_Gcurr" : score_Gcurr,
									"acceptance_prob" : acceptance_prob}
			# reset index
			accept_indx = 0
			iter_indx += 1

		return mcmc_res, np.round(ACCEPT / self.max_iter,4)
		

	def get_mcmc_res_graphs(self, results):
		mcmc_graph_lst = []
		for i in range(len(results)):
			mcmc_graph_lst.append( results[i]['graph'] )
		return mcmc_graph_lst

	def get_mcmc_res_operations(self, results):
		mcmc_operations_lst = []
		for i in range(len(results)):
			mcmc_operations_lst.append( results[i]['graph'] )
		return mcmc_operations_lst

	def get_mcmc_res_scores(self, results):
		mcmc_score_lst = []
		for i in range(len(results)):
			mcmc_score_lst.append( results[i]['score'] )
		return mcmc_score_lst

	def get_mcmc_res_accepted_graphs(self, results):
		mcmc_accepted_graph_lst = []
		mcmc_accepted_graph_indx = []
		for i in range(len(results)):
			mcmc_accepted_graph_indx.append( results[i]['accepted'] )	
			if results[i]['accepted'] == 1:
				mcmc_accepted_graph_lst.append( results[i]['graph'] )
		return mcmc_accepted_graph_lst, mcmc_accepted_graph_indx


	## GETTERS
	#####################################

	def get_score_object(self):
		return self.score_object

	def get_max_iter(self):
		return self.max_iter

	def get_proposal_object(self):
		return self.proposal_object

	def get_data(self):
		return self.data

	def get_num_nodes(self):
		return self.num_nodes

	def get_node_labels(self):
		return self.node_labels

	def get_sample_parameters(self):
		return self.sample_parameters

	def get_initial_graph(self):
		return self.initial_graph

	def get_indx_to_node_label(self):
		return self.indx_to_node_label

	## SETTERS
	#####################################
	def set_score_object(self, score_object):
		self.score_object = score_object

	def set_max_iter(self, max_iter : int):
		self.max_iter = max_iter

	def set_proposal_object(self, proposal_object):
		self.proposal_object = proposal_object

	def set_data(self, data):
		self.data = data

	def set_num_nodes(self, num_nodes : int):
		self.num_nodes = num_nodes

	def set_node_labels(self, node_labels):
		self.node_labels = node_labels

	def set_initial_graph(self, initial_graph):
		self.initial_graph = initial_graph

	def set_indx_to_node_label(self, indx_to_node_label):
		self.indx_to_node_label = indx_to_node_label