
from cache import Cache

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import math

import seaborn as sns

from collections import Counter

class MCMCResultsEngine():
    
    #  empty constructor
    def __init__(self):
        self.cache = Cache()
        
    # given a graphID, return the score of the graph
    def plot_graph_from_id(self, graph_id : int, title="Graph", node_size = 2000, k = 5, figsize=(3, 3)):
        G = self.cache.get_graph(graph_id)
        self.plot_graph(G, title, node_size, k, figsize)
    
    def plot_graph(self, G : nx.DiGraph, title="Graph", node_size = 2000, k = 5, figsize=(3, 3)):
        pos = nx.spring_layout(G, k=k)

        nx.draw(G, with_labels=True, arrowsize=20, arrows=True, node_size=node_size, node_color="skyblue", pos=pos)
        ax = plt.gca()
        ax.margins(0.20)
        ax.set_title( title )
        plt.axis("off")
        
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        plt.show()


    def plot_neighbors(self, graph_id_current, graph_list, operation,  k = 5, node_size = 2000):
        N = len(graph_list)
        
        # get graph from graph_id
        G_current = self.cache.get_graph(graph_id_current)
        
        fig, axes = plt.subplots(1, N, figsize=(5*N, 4))  # Adjust the width based on the number of graphs
        
        for idx, G in enumerate(graph_list):
            if N == 1:  # Special case for only one graph
                ax = axes
            else:
                ax = axes[idx]
            
            edges = self.get_different_edges(G_current, G)
            
            # delete from edges the edges that already exist in G
            if not operation == "delete_edge":
                edges = [edge for edge in edges if edge not in G_current.edges()]
        
            plt.sca(ax)
            pos = nx.spring_layout(G, k=k)
            
            nx.draw(G, with_labels=True, arrowsize=20, node_size=node_size, node_color="skyblue", pos=pos)
            if operation == "add_edge":
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='g', arrowsize=20, node_size=node_size)
            if operation == "delete_edge":
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', arrowsize=20, node_size=node_size, style="dashed")
            if operation == "reverse_edge":
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='b', arrowsize=20, node_size=node_size)
                
            ax.margins(0.20)
            ax.set_title(f'Graph {idx+1}')
        
        plt.tight_layout()
        plt.show()
        
    def plot_mcmc_proposed_graphs(self, graph_list, operation_list, k=5, node_size=2000):
        
        N = len(graph_list)
        
        # Define the number of rows and columns
        num_rows = math.ceil(N / 4)
        num_cols = min(4, N) # 4 columns or fewer if there are fewer than 4 graphs
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
        
        # Handle the case where there's only one graph (axes would not be a 2D array)
        if N == 1:
            axes = np.array([[axes]])
        
        # Flatten the axes array for easy iteration
        flat_axes = axes.flatten()
        
        operation = operation_list[0]
        G_current = graph_list[0]
        
        for idx, (G, ax) in enumerate(zip(graph_list, flat_axes)):
            edges = []
            if idx > 0: # The first graph is the current one and doesn't need comparisons
                edges = self.get_different_edges(G_current, G)
                
            edges = self.get_different_edges(G_current, G)
            
            # delete from edges the edges that already exist in G
            operation = operation_list[idx]
            if not operation == "delete_edge":
                edges = [edge for edge in edges if edge not in G_current.edges()]
            
            plt.sca(ax)
            pos = nx.spring_layout(G, k=k)
            if operation == "random":
                nx.draw(G, with_labels=True, arrowsize=20, node_size=node_size, node_color="coral", pos=pos)
            else:
                nx.draw(G, with_labels=True, arrowsize=20, node_size=node_size, node_color="skyblue", pos=pos)

            if operation == "add_edge":
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='g', arrowsize=20, node_size=node_size)
            elif operation == "delete_edge":
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r', arrowsize=20, node_size=node_size, style="dashed")
            elif operation == "reverse_edge":
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='b', arrowsize=20, node_size=node_size)
                
            ax.margins(0.20)
            
            QGcurr_to_QGprop = self.cache.MCMC_G_curr_to_G_prop_den[idx]
            QGprop_to_QGcurr = self.cache.MCMC_G_prop_to_G_curr_den[idx]
            PGcurr_to_PGprop = self.cache.MCMC_G_curr_to_G_prop_prob[idx]
            PGprop_to_PGcurr = self.cache.MCMC_G_prop_to_G_curr_prob[idx]
            
            Poper_Gcurr_to_Gprop = self.cache.MCMC_G_curr_to_G_operation_prob[idx]
            Poper_Gprop_to_Gcurr = self.cache.MCMC_G_prop_to_G_operation_prob[idx]
            
            try:
                score = self.cache.MCMC_proposal_scores[idx]
            except:
                score = 0
            
            if operation == "random":
                ax.set_title(f"Graph {idx}: {operation} \n $Q(Gc->Gp) = 1/{QGcurr_to_QGprop} = {np.round(PGcurr_to_PGprop,2)} \n Q(Gp->Gc) = 1/{QGprop_to_QGcurr} = {np.round(PGprop_to_PGcurr,2)}"  )
            else:
                ax.set_title(f"Graph {idx}: {operation} : {edges} \n Q(Gc->Gp) = 1/{Poper_Gcurr_to_Gprop} x 1/{QGcurr_to_QGprop} = {np.round(PGcurr_to_PGprop,2)} \n Q(Gp->Gc) = 1/{Poper_Gprop_to_Gcurr} x 1/{QGprop_to_QGcurr} = {np.round(PGprop_to_PGcurr,2)}\nScore = {np.round(score,4)}" )
            
            #if score_list is not None:
                
            G_current = G

        plt.tight_layout()
        plt.show()

    # given two networkx graphs, return the edges that are different
    def get_different_edges(self, G1: nx.DiGraph, G2: nx.DiGraph):
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
    
        return edges1.symmetric_difference(edges2)


    def hash_graph(self, G: nx.DiGraph) :
        """Returns a unique hash for a graph based on its edge set."""
        return str(sorted(G.edges()))


    def compute_graph_frequencies(self, graph_list: list, title):
        """Plots the frequency of each graph in a list of graphs."""
        
        hash_to_graph = {}
        hash_to_graph_id = {}
        graph_id_to_graph = {}
        
        graph_hashes = []
        graph_ids = []
        
        id = 0
        
        # Create a list of graph hashes
        for G in graph_list:
            
            if self.hash_graph(G) not in graph_hashes:
                hash_to_graph[self.hash_graph(G)] = G
                graph_hashes.append(self.hash_graph(G))
                hash_to_graph_id[self.hash_graph(G)] = id
                graph_id_to_graph[id] = G
                graph_ids.append(id)
                id = id + 1
            
        # Count the frequency of each unique graph hash
        graph_freq = Counter(graph_hashes)
        
        return graph_freq
    
    
    
    def plot_graph_frequencies(self, graph_list : list, title : str  ):
        
        
        graph_freq = self.compute_graph_frequencies(graph_list, title)
        
        
        # Plot the frequencies
        plt.bar(graph_freq.keys(), graph_freq.values())
        plt.xlabel("Graphs")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.xticks([])  # Hide x-ticks as graph hashes can be long and messy
        plt.show()
        
    
        
    def plot_graph_frequencies_by_edge(self, graphs_lst : list, edges = True, title = "Graph Frequencies by Edge"):
        
        graph_freq = self.cache.compute_graph_frequency(graphs_lst)
        edge_freq = self.cache.compute_graph_edge_frequency(graphs_lst)

        # sorte the edge frequencies
        sorted_edge_freq = sorted(edge_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_edge_freq

        # transform the edge frequencies into a dictionary
        edge_freq_dict = dict(sorted_edge_freq)

        
        # plot the edge frequencies
        new_graph_lst = []
        for key in edge_freq_dict.keys():
            new_graph_lst.append(  graph_freq[key] )

        # make a bar plot
        plt.bar( list(edge_freq_dict.keys()), new_graph_lst )
        
        # set title
        plt.title(title)

        # add the edge frequencies as labels
        if edges:
            for i, v in enumerate(edge_freq_dict.values()):
                plt.text( i, v, str(v) )

        plt.xticks(rotation=90)
        # reduce font size of axis labels
        plt.tick_params(axis='x', labelsize=8)
        
        # set lim in y axis
        plt.ylim(0, max(graph_freq.values()) + 5)
        
        plt.show()
        
        return sorted_edge_freq, graph_freq
    
    def normalize_graph_freq(self, graph_freq):
        total = sum(graph_freq.values())
        for key in graph_freq:
            graph_freq[key] = graph_freq[key]/total
        return graph_freq
    
    def trace_plot(self, struct_mcmc, mcmc_res, burnIn, figsize=(12, 6)):
        
        scores = struct_mcmc.get_mcmc_res_scores(  mcmc_res )[burnIn:]
        
        lst_accepted_graph, lst_accepted_graph_indx = struct_mcmc.get_mcmc_res_accepted_graphs( mcmc_res )
        lst_accepted_graph_indx = lst_accepted_graph_indx[burnIn:]

        
        plt.figure(figsize=figsize)
        
        plt.plot(scores, color='blue', lw=0.2)
        
        # add a red dot for all graphs that were accepted
        for i in range(0, len(lst_accepted_graph_indx)):
            is_accepted = mcmc_res[i]['accepted']
            if is_accepted:
                plt.plot(i, scores[i], 'ro', markersize=1)
        
        plt.title("MCMC Trace Plot")
        plt.xlabel("Iteration")
        plt.ylabel("Score/Value")
        plt.tight_layout()
        plt.legend()
        plt.show()
    

    def density_plot(self, struct_mcmc, mcmc_res, burnIn, figsize=(12, 6)):
        
        scores = struct_mcmc.get_mcmc_res_scores(  mcmc_res )[burnIn:]

        plt.figure(figsize=figsize)
        
        sns.kdeplot(scores, fill=True)
        plt.title("Density Plot")
        plt.xlabel("Score/Value")
        plt.ylabel("Density")
        plt.tight_layout()
        
        plt.show()

    def mcmc_edge_frequency_heatmap(self, struct_mcmc, mcmc_res,  burnIn, figsize= (8,8), title = "Edge Occurrence Probabilities from Sampled DAGs" ):
        
        dags = struct_mcmc.get_mcmc_res_graphs(  mcmc_res )[burnIn:]
        
        nodes = list( dags[0].nodes())
        
        num_nodes = len(nodes)
        frequency_matrix = np.zeros((num_nodes, num_nodes))
        
        for G in dags:
            for edge in G.edges():
                source, target = edge
                frequency_matrix[nodes.index(source), nodes.index(target)] += 1
        
        # Normalize by the number of samples to get frequencies
        frequency_matrix /= len(dags)
        
        # Visualize as heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(frequency_matrix, annot=True, cmap="YlGnBu", xticklabels=nodes, yticklabels=nodes)
        # add a label to the yaxis
        plt.ylabel("Source")
        # add a label to the xaxis
        plt.xlabel("Target")
        plt.title(title)
        plt.show()


    def plot_true_posterior_vs_approx_posterior(self, true_posterior_distribution, approx_posterior_distribution, figsize = (10,7), title = 'True Posterior Graph Prob vs Estimated Posterior Graph Prob'):
        # Scatter plot
        plt.figure(figsize=figsize)
        plt.scatter(list(true_posterior_distribution.values()), list(approx_posterior_distribution.values()), color='blue', label='Predicted vs True')
        plt.plot([min(true_posterior_distribution.values()), max(true_posterior_distribution.values())], 
                [min(true_posterior_distribution.values()), max(true_posterior_distribution.values())], color='red', linestyle='--', label='Perfect Prediction Line')

        # Labels and Title
        plt.xlabel('True Posterior Graph Prob')
        plt.ylabel('Estimated Posterior Graph Prob')
        plt.title(title)
        plt.legend()
        plt.grid(True)

        plt.show()
