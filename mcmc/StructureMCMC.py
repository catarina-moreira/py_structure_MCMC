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

	def __init__(self, initial_graph : nx.DiGraph, max_iter : int, 
				proposal_object : StructureLearningProposal, 
				score_object : Score, burnIn : int = 0, sample_parameters: bool = False):
		
		self.score_object = score_object
		self.data = score_object.get_data()
		self.sample_parameters = sample_parameters
		self.n = len(self.data.columns)

		# if the initial_graph is none, generate an empty graph with the same number of nodes as the data
		if initial_graph is None:
			self.initial_graph = nx.DiGraph()
			self.initial_graph.add_nodes_from(self.data.columns)
		else:
			self.initial_graph = initial_graph

		self.max_iter = max_iter
		self.burnIt = burnIn
		self.proposal_object = proposal_object

	# main MCMC function that needs to be implemented
	def log_acceptance_ratio(self, posterior_Gcurr, posterior_Gprop, Q_Gcurr_Gprop, Q_Gprop_Gcurr):

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

	def acceptance_ratio(self, posterior_Gcurr, posterior_Gprop, Q_Gcurr_Gprop, Q_Gprop_Gcurr):
		
		numerator = posterior_Gprop * Q_Gcurr_Gprop
		denominator = posterior_Gcurr * Q_Gprop_Gcurr
		return min(1, numerator/denominator)

	def run(self):

		mcmc_res = {}
		iter_indx = 0
		ACCEPT = 0
		iter_indx = 1

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
						"score_Gcurr" : score_Gcurr,
						"acceptance_prob" : 0} 
		
		node_score_Gcurr = collect_node_scores(score_dict)

		for _ in range(self.max_iter):
			
			accept_indx = 0
			acceptance_prob = 0

			# with probability 0.01, stay at the current graph
			u = random.uniform(0,1)
			if u < 0.01:
				
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
									"acceptance_prob" : 0.01}
					
				iter_indx += 1
				continue

			# propose a new graph and compute the proposal distribution Q
			self.proposal_object.update_G_curr(G_curr)
			G_prop, G_prop_operation = self.proposal_object.propose_DAG(  )

			# compute the proposal distribution Q
			Q_Gcurr_Gprop = self.proposal_object.get_prob_Gcurr_Gprop()
			Q_Gprop_Gcurr = self.proposal_object.get_prob_Gprop_Gcurr()

			# we need to update the graph so we can extract the parents of the node
			self.score_object.set_graph(G_prop)
			node_score_Gprop = node_score_Gcurr.copy() 

			rescored_nodes = compare_graphs(G_curr, G_prop, G_prop_operation)
			
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


	## GETTERTS
	#####################################
	def getMax_iter(self):
		return self.max_iter

	def getInitial_graph(self):
		return self.initial_graph

	def getBurnIt(self):
		return self.burnIt

	def getRand_jump(self):
		return self.rand_jump

	def getRand_jump_prob(self):
		return self.rand_jump_prob

	def getProposal_function(self):
		return self.proposal_function
		
	## SETTERS
	##########################################
	def setInitial_graph(self, initial_graph : nx.DiGraph):
		self.initial_graph = initial_graph

	def setMax_iter(self, max_iter : int):
		self.max_iter = max_iter

	def setBurnIt(self, burnIt : int):
		self.burnIt = burnIt

	def setRand_jump(self, rand_jump : int):
		self.rand_jump = rand_jump

	def setRand_jump_prob(self, rand_jump_prob : float):
		self.rand_jump_prob = rand_jump_prob

	def setProposal_function(self, proposal_function):
		self.proposal_function = proposal_function
