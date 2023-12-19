import networkx as nx
import pandas as pd

import os

class Cache:
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super(Cache, cls).__new__(cls)
            
            cls.instance.dict_hash_to_graph = {}                        # dict to store the hash of the graph and the graph itself
            cls.instance.dict_hash_to_graph_id = {}                     # dict to store a graphID and the hash of the graph
            cls.instance.dict_graph_id_to_hash = {}                     # dict to store the graphID and the hash of the graph
            cls.instance.dict_graph_id_to_score = {}                        # dict to store a hash of the graph and the score of the graph
            cls.instance.curr_graph_id = 0                              # keeping track of the current graphID
            
            cls.instance.dict_graph_id_to_neighbors_by_addition = {}    # dictionary to store all neighbours of a graph by adding an edge
            cls.instance.dict_graph_id_to_edges_to_add = {}             # dictionary to store all possible edges that can be added
            
            cls.instance.dict_graph_id_to_neighbors_by_deletion = {}    # dictionary to store all neighbours of a graph by deleting an edge
            cls.instance.dict_graph_id_to_edges_to_delete = {}          # dictionary to store all possible edges that can be deleted
            
            cls.instance.dict_graph_id_to_neighbors_by_reversal = {}    # dictionary to store all neighbours of a graph by reversing an edge
            cls.instance.dict_graph_id_to_edges_to_reverse = {}         # dictionary to store all possible edges that can be reversed
            
            cls.instance.MCMC_proposed_graphs = []                      # list to store all MCMC proposed graphs
            cls.instance.MCMC_proposed_operations = []                  # list to store all MCMC proposed operations
            
            cls.instance.MCMC_accepted_graphs = []                      # list to store all MCMC accepted graphs
            cls.instance.MCMC_accepted_operations = []                  # list to store all MCMC accepted operations
            
            # to DEBUG the transition probabilities
            cls.instance.MCMC_G_curr_to_G_prop_den = []                 # list to store all MCMC G_curr to G_prop densities
            cls.instance.MCMC_G_prop_to_G_curr_den = []                 # list to store all MCMC G_prop to G_curr densities
            cls.instance.MCMC_G_curr_to_G_prop_prob = []                # list to store all MCMC G_curr to G_prop probabilities
            cls.instance.MCMC_G_prop_to_G_curr_prob = []                # list to store all MCMC G_prop to G_curr probabilities
            cls.instance.MCMC_G_curr_to_G_operation_prob = []           # list to store all MCMC G_curr to G_prop probabilities
            cls.instance.MCMC_G_prop_to_G_operation_prob = []           # list to store all MCMC G_prop to G_curr probabilities
            
            cls.instance.NODES_to_DAGs = {}                       # dict to store the number of nodes and the number of DAGs
            cls.instance.NODES_to_DAGs[2] = 2
            cls.instance.NODES_to_DAGs[3] = 25
            cls.instance.NODES_to_DAGs[4] = 543
            cls.instance.NODES_to_DAGs[5] = 29281
            cls.instance.NODES_to_DAGs[6] = 3781503
            cls.instance.NODES_to_DAGs[7] = 1171650245
            
            cls.instance.MCMC_proposal_scores = []                      # list to store all MCMC proposal scores
        return cls.instance
    
    def cache_init(self):

        
        
        self.dict_hash_to_graph = {}                            # dict to store the hash of the graph and the graph itself
        self.dict_hash_to_graph_id = {}                         # dict to store a graphID and the hash of the graph
        self.dict_graph_id_to_hash = {}                         # dict to store the graphID and the hash of the graph
        self.dict_hash_to_score = {}                            # dict to store a hash of the graph and the score of the graph
        self.curr_graph_id = 0                                  # keeping track of the current graphID
        
        self.dict_graph_id_to_neighbors_by_addition = {}        # dictionary to store all neighbours of a graph by adding an edge
        self.dict_graph_id_to_edges_to_add = {}                 # dictionary to store all possible edges that can be added
        
        self.dict_graph_id_to_neighbors_by_deletion = {}        # dictionary to store all neighbours of a graph by deleting an edge
        self.dict_graph_id_to_edges_to_delete = {}              # dictionary to store all possible edges that can be deleted
        
        self.dict_graph_id_to_neighbors_by_reversal = {}        # dictionary to store all neighbours of a graph by reversing an edge
        self.dict_graph_id_to_edges_to_reverse = {}             # dictionary to store all possible edges that can be reversed
        
        self.MCMC_proposed_graphs = []                          # list to store all MCMC proposed graphs
        self.MCMC_proposed_operations = []                      # list to store all MCMC proposed operations
        
        self.MCMC_accepted_graphs = []                          # list to store all MCMC accepted graphs
        self.MCMC_accepted_operations = []                      # list to store all MCMC accepted operations
        
        # to DEBUG the transition probabilities
        self.MCMC_G_curr_to_G_prop_den = []                 # list to store all MCMC G_curr to G_prop densities
        self.MCMC_G_prop_to_G_curr_den = []                 # list to store all MCMC G_prop to G_curr densities
        self.MCMC_G_curr_to_G_prop_prob = []                # list to store all MCMC G_curr to G_prop probabilities
        self.MCMC_G_prop_to_G_curr_prob = []                # list to store all MCMC G_prop to G_curr probabilities
    
    
    def compute_graph_hash(self, G: nx.DiGraph):
            
            # get the hash of the graph
            graph_hash = str(sorted(G.edges()))
            
            # check if the graph already exists in the cache
            if graph_hash in self.dict_hash_to_graph:
                return graph_hash
            
            # otherwise, store the graph in the cache
            self.dict_hash_to_graph[graph_hash] = G
            
            # store the graphID and the hash of the graph
            graph_id = self.curr_graph_id
            self.dict_hash_to_graph_id[graph_hash] = graph_id
            self.dict_graph_id_to_hash[graph_id] = graph_hash
            self.curr_graph_id = self.curr_graph_id + 1
            
            return graph_hash
        

    # given a graph, return the graphID
    def get_graph_id(self, G: nx.DiGraph):
        graph_hash = self.compute_graph_hash(G)
        return self.dict_hash_to_graph_id[graph_hash]
    
    # given a graphID, return the graph
    def get_graph(self, graph_id):
        graph_hash = self.dict_graph_id_to_hash[graph_id]
        return self.dict_hash_to_graph[graph_hash]
    
    # given a graphID, return the adjacency matrix of the graph
    def get_adjacency_matrix(self, graph_id):
        G = self.get_graph(graph_id)
        adj_matrix = nx.adjacency_matrix(G)
        dense_adj_matrix = adj_matrix.toarray()
    
        nodes = list(G.nodes())
        adj_df = pd.DataFrame(dense_adj_matrix, index=nodes, columns=nodes)
        return adj_df
    
    #   given a graphID, return the score of the graph  
    def compute_graph_frequency(self, graph_lst : list ):
        graph_ids = []
        graph_freq = {}
        for G in graph_lst:
            g_id = self.get_graph_id(G)
            graph_ids.append(g_id)
            
            # g_id not in graph_freq, then add it. Otherwise, increment it
            if g_id not in graph_freq:
                graph_freq[g_id] = 1
            else:
                graph_freq[g_id] += 1
                
        return graph_freq

    # given a list of graphs, return a dictionary of graphID and the number of edges in the graph
    def compute_graph_edge_frequency(self, graph_lst : list):
        graph_edges = {}
        for G in graph_lst:
            g_id = self.get_graph_id(G)
            graph_edges[g_id] = len(list(G.edges()))
            
        return graph_edges
    