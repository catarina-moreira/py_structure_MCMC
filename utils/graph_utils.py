import itertools

import collections

import numpy as np
import pandas as pd

import networkx as nx

from itertools import permutations


from scores.bge import BGeScore

from scores.marginal import MarginalLogLikelihood
from scores.dummy import DummyScore
from scores.NIG import DAG_NIG
from scores.ScoreAbstract import Score

from collections import Counter

from scipy.stats import entropy
from scipy.stats import gamma, invgamma
from scipy.stats import multivariate_normal

import os
from math import comb


import zipfile
import matplotlib.pyplot as plt


## count_dags
##################################################################################
def count_dags(n : int):
    """Given $n$ nodes, the possible number of DAGs that can be built is given by 
        a(n) = \sum_{k=1}^{n} (-1)^{(k+1)} \binom{n}{k} 2^{k(n-k)} a(n-k) 
    Args:
        n (int): number of nodes
    Returns:
        long: number of possible directed graphs
    """
    if n == 0:
        return 1
    
    total = 0
    for k in range(1, n + 1):
        total += (-1)**(k+1) * comb(n, k) * (2**(k*(n-k))) * count_dags(n-k)
    return total


def update_matrix(matrix1, matrix2):
    """
    Update matrix1 with the values from matrix2 only where matrix1 is zero.

    :param matrix1: First matrix to update.
    :param matrix2: Second matrix with values to use for updating.
    :return: Updated matrix1.
    """
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    
    # Check if both matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same dimensions.")
    
    # Update matrix1 only where its elements are zero
    matrix1[matrix1 == 0] = matrix2[matrix1 == 0]
    
    return matrix1

## save_graphs
##################################################################################
def save_graphs( graphs: list, filename : str, num_nodes : int):
    """_summary_

    Args:
        graphs (list): a list of graphs nx.DiGraph
        filename (str): a string representing the filename to be saved
        num_nodes (int): number of nodes in graphs
    """

    # Save each graph to a separate GraphML file
    for idx, graph in enumerate(graphs):
        nx.write_graphml(graph, f"./results/graph_generation/{num_nodes}nodes/graph_{idx}.graphml")

    # Create a ZIP archive containing all the GraphML files
    with zipfile.ZipFile(f"./results/graph_generation/{num_nodes}nodes/{filename}.zip", "w") as zipf:
        for idx in range(len(graphs)):
            zipf.write(f"./results/graph_generation/{num_nodes}nodes/graph_{idx}.graphml")
            # os.remove(f"graph_{idx}.graphml")



def load_R_adj_matrices(folder_path: str):
    
    files = [f for f in os.listdir(folder_path) if f.startswith('matrix_') and f.endswith('.csv')]

    # sort files numericaly
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
    
    graphs = []

    for file in files:
        
        full_file_path = os.path.join(folder_path, file)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(full_file_path)
        
        df.index = df.columns
        
        # Convert the DataFrame into a directed graph and append to the list
        graph = nx.from_pandas_adjacency(df, create_using=nx.DiGraph())
        nx.set_node_attributes(graph, df.columns, 'label')
        graphs.append(graph)
        
    return graphs



def generate_all_dags_from_ordering(nodes : list):
    # Generate all permutations of edges based on the given topological order
    all_edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    unique_graphs = set()

    for edges in itertools.chain.from_iterable(itertools.combinations(all_edges, r) for r in range(len(all_edges)+1)):
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        # Check for cycles, since we only want DAGs
        if not nx.is_directed_acyclic_graph(G):
            continue

        # Generate a sorted adjacency matrix as a tuple and check if it's already in our set
        adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        matrix_tuple = tuple(map(tuple, adj_matrix))
        if matrix_tuple not in unique_graphs:
            unique_graphs.add(matrix_tuple)
            yield G

## gen_base_dag_dict
##################################################################################
def gen_base_dag_dict( N : int, node_labels : list = None ):
    """_summary_

    Args:
        N (int): _description_
        node_labels (list, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    if node_labels is None:
        # generate node labels for N nodes
        node_labels = [f"X{i}" for i in range(N)]
    
    nodes = node_labels
    
    # Dictionary to store all unique DAGs
    base_dag_dict = {}
    
    # Generate all possible directed edges among the nodes
    all_possible_edges = list(itertools.permutations(nodes, 2))
    
    # Iterate over all possible adjacency matrices
    # Iterate over the subsets of all possible edges to form directed graphs
    for r in range(len(all_possible_edges)+1):
        for subset in itertools.combinations(all_possible_edges, r):
            
            # Initialize an NxN matrix filled with zeros
            adj_matrix = np.zeros((N, N))
            adj_matrix = adj_matrix.astype(int)
            
            # Set entries corresponding to the edges in the current subset to 1
            for edge in subset:
                source, target = edge
                adj_matrix[nodes.index(source)][nodes.index(target)] = 1
            
            # convert adj_matrix to int
            adj_matrix = adj_matrix.astype(int)
            
            # cast the first row to int
            adj_matrix[0] = adj_matrix[0].astype(int)
            
            # Convert to DiGraph
            G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
            
            # Relabel nodes to original node names
            mapping = dict(zip(G.nodes(), nodes))
            G = nx.relabel_nodes(G, mapping)
            
            if nx.is_directed_acyclic_graph(G):
                
                G_str = convert_graph_to_str(G)
                
                # forcing the representation to be an int
                if "0." in G_str:
                    G_str = G_str.replace("0.", "0")
                
                # set graph frequency to 0
                base_dag_dict[G_str] = 0
                
    return base_dag_dict

def collect_node_scores( graph_score ):
    
    scores = {node: info['score'] for node, info in graph_score['parameters'].items()}
    return scores

def compare_graphs_old(g1, g2, op):
    
    if op == 'add_edge':
        return list(set(g2.edges()) - set(g1.edges()))[0][1]
        
    if op == 'delete_edge':
        return  list(set(g1.edges()) - set(g2.edges()) )[0][1]
    
    if op == 'reverse_edge':
        return list(set(g1.edges()) - set(g2.edges()))[0]
    
    return "[ERROR] Edge Operation Not Found"

def compare_graphs(g1: np.ndarray, g2: np.ndarray, op, indx_to_node_label):
    
    if g1.shape != g2.shape:
        return "[ERROR] Graphs are not the same size"
    
    if op == 'add_edge':
        diff = np.where((g1 == 0) & (g2 != 0))
        if len(diff[0]) > 0:
            return indx_to_node_label[diff[1][0]]
        else:
            return "[ERROR] No edge added"

    if op == 'delete_edge':
        diff = np.where((g1 != 0) & (g2 == 0))
        if len(diff[0]) > 0:
            return indx_to_node_label[diff[1][0]]
        else:
            return "[ERROR] No edge deleted"

    if op == 'reverse_edge':
        added_edges = np.where((g1 == 0) & (g2 != 0))
        deleted_edges = np.where((g1 != 0) & (g2 == 0))
        reversed_edges = list(set(zip(added_edges[1], added_edges[0])) & set(zip(deleted_edges[0], deleted_edges[1])))[0]

        if reversed_edges:
            return [indx_to_node_label[reversed_edges[0]], indx_to_node_label[reversed_edges[1]] ]
        else:
            return "[ERROR] No edge reversed"

    return "[ERROR] Edge Operation Not Found"
    
    
    
import numpy as np

def is_cyclic_util(v, visited, rec_stack, adj_matrix):
    visited[v] = True
    rec_stack[v] = True

    # Consider all adjacent vertices
    for i in range(len(adj_matrix)):
        if adj_matrix[v][i] != 0:
            if not visited[i]:
                if is_cyclic_util(i, visited, rec_stack, adj_matrix):
                    return True
            elif rec_stack[i]:
                return True

    rec_stack[v] = False
    return False

def has_cycle(adj_matrix):
    """
    Check if the graph represented by the adjacency matrix has a cycle.
    
    :param adj_matrix: numpy array representing the adjacency matrix of the graph
    :return: True if the graph has a cycle, False otherwise
    """
    num_vertices = len(adj_matrix)

    visited = [False] * num_vertices
    rec_stack = [False] * num_vertices

    for node in range(num_vertices):
        if not visited[node]:
            if is_cyclic_util(node, visited, rec_stack, adj_matrix):
                return True

    return False

# Example usage
# adj_matrix = np.array(...)  # Your adjacency matrix
# print(has_cycle(adj_matrix))


def generate_all_dags( data : pd.DataFrame, my_score : Score, with_aug_priors = False ):
    
    N = data.shape[1]
    nodes = list(data.columns)
    
    # Dictionary to store all unique DAGs
    base_dag_dict = {}
    
    total_score = 0
    total_score_ordering = 0
    total_orderings = 0
    norm_factor = 0
    
    # generate a graph index as a compact key for each graph
    graph_indx = 0
    
    # Generate all possible directed edges among the nodes
    all_possible_edges = list(itertools.permutations(nodes, 2))
    
    # Iterate over all possible adjacency matrices
    # Iterate over the subsets of all possible edges to form directed graphs
    for r in range(len(all_possible_edges)+1):
        for subset in itertools.combinations(all_possible_edges, r):
            
            # Initialize an NxN matrix filled with zeros
            adj_matrix = np.zeros((N, N))
            adj_matrix = adj_matrix.astype(int)
            
            # Set entries corresponding to the edges in the current subset to 1
            for edge in subset:
                source, target = edge
                adj_matrix[nodes.index(source)][nodes.index(target)] = 1
            
            # convert adj_matrix to int
            adj_matrix = adj_matrix.astype(int)
            adj_matrix[0] = adj_matrix[0].astype(int)
            
            if not has_cycle( adj_matrix ):
                
                adj_matrix_str = convert_graph_to_str( adj_matrix )
                
                # forcing the representation to be an int
                if "0." in adj_matrix_str:
                    adj_matrix_str = adj_matrix_str.replace("0.", "0")
                
                # start storing all information for DAG G
                base_dag_dict[adj_matrix_str] = {}
                
                G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
                
                base_dag_dict[adj_matrix_str]["DAG"] = G # store the DAG in networkx format
                base_dag_dict[adj_matrix_str]["DAG_indx"] = graph_indx # store the DAG index
                base_dag_dict[adj_matrix_str]['Freq'] = 1 # the true distrib generates 1 graph only
                
                # compute the total number of orderings compatible with the DAG
                base_dag_dict[adj_matrix_str]['num_orderings'] = len(all_valid_orderings(G)) 
                # keep track of the number of edges in the DAG
                base_dag_dict[adj_matrix_str]['num_edges'] = len(G.edges())
                
                # todo: later, we need to pass the score object as an argument instead of a string
                my_score_object = my_score(data=data, incidence=adj_matrix)
                
                
                #if my_score == "BGe Score":
                #    my_score_object = BGeScore(data=data, incidence = adj_matrix)
                #elif my_score == "Marginal Log Likelihood":
                #    my_score_object = MarginalLogLikelihood(data=data, incidence=adj_matrix)
                #elif my_score == "Dummy Score":
                #    my_score_object = DummyScore(data=data, incidence=adj_matrix)
                #else:
                #    raise ValueError("Invalid score function")

                # compute the score the DAG given the data
                #if my_score == "DAG NIG Score":
                #    params = my_score_object.sample_parameters()
                #    Betas = params['Betas'][0]
                #    Sigma2= params['Sigma2s'][0]
                #    score = my_score_object.compute(Betas,Sigma2)
                
                #else:
                #    score = my_score_object.compute()
                
                score = my_score_object.compute()
                base_dag_dict[adj_matrix_str]["log_score"] = score["score"]
                
                # blow the log_Score by the factor of the number of orderings that the DAG can be generated from
                base_dag_dict[adj_matrix_str]["log_score_ordering"] =  base_dag_dict[adj_matrix_str]["log_score"] 
                # base_dag_dict[G_str]["params"] = score["parameters"]
                
                graph_indx = graph_indx + 1
                
    # get the max / min scores of the log_scores
    max_score = max([ base_dag_dict[id]["log_score"] for id in base_dag_dict.keys() ])
    
    # get the max / min scores of the log_scores with the ordeing
    max_score_ordering = max([ base_dag_dict[id]["log_score_ordering"] for id in base_dag_dict.keys() ])
    
    # since the marginal likelihood grows very fast, we need to normalise the scores
    # we will subtract the max score from all scores and then divide by the total score
    for id in base_dag_dict.keys():
        base_dag_dict[id]["log_score_ordering_scaled"] = base_dag_dict[id]["log_score_ordering"]  - max_score_ordering
        base_dag_dict[id]["log_score_scaled"] = base_dag_dict[id]["log_score"]  - max_score
        
        # convert the logarithmic score to a scorem by making the exponential
        base_dag_dict[id]["score_ordering"] = np.exp( base_dag_dict[id]["log_score_ordering_scaled"] + np.log(base_dag_dict[id]['num_orderings']))   # convert the log score to a score
        base_dag_dict[id]["score"] = np.exp( base_dag_dict[id]["log_score_scaled"]  )                   # convert the log score to a score
        total_score = total_score + base_dag_dict[id]["score"]
        total_score_ordering = total_score_ordering + base_dag_dict[id]["score_ordering"]
        norm_factor = norm_factor + base_dag_dict[id]["score"]*base_dag_dict[id]['num_orderings']
    
    max_num_orders = max([ base_dag_dict[id]["num_orderings"] for id in base_dag_dict.keys() ])
    
    # iterate of the dags and normalise the scores
    for dag in base_dag_dict.keys():
        base_dag_dict[dag]["score_normalised"] = base_dag_dict[dag]["score"] / total_score
        base_dag_dict[dag]["score_ordering_normalised"] = (base_dag_dict[dag]["score"] * base_dag_dict[dag]["num_orderings"]) / norm_factor
        base_dag_dict[dag]['num_orderings_normalised'] = base_dag_dict[dag]['num_orderings'] / max_num_orders
    
    print(f"Total {N} node DAGs generated = {len(base_dag_dict.keys())}")
            
    return base_dag_dict, total_score
                

def generate_dag_from_ordering(ordered_nodes: list, edge_prob : float = 0.5, desired_order=None):
    
    G = nx.DiGraph()
    G.add_nodes_from(desired_order)
    
    for i in range(len(ordered_nodes)):
        for j in range(i+1, len(ordered_nodes)):
            if np.random.rand() < edge_prob:
                G.add_edge(ordered_nodes[i], ordered_nodes[j])
    return G



def convert_graph_to_str( adj_matrix ):
    
    # Build the string representation directly from the sparse matrix
    rows, cols = adj_matrix.shape
    matrix_string = ''
    for i in range(rows):
        for j in range(cols):
            matrix_string += str(int(adj_matrix[i, j])) if adj_matrix[i, j] != 0 else '0'
        matrix_string += ','  # Separator for rows
    
    return matrix_string.rstrip(',')


def get_adjacency_matrix(G: nx.DiGraph):
    
    adj_matrix = nx.adjacency_matrix(G)
    dense_adj_matrix = adj_matrix.toarray()
    
    nodes = list(G.nodes())
    adj_df = pd.DataFrame(dense_adj_matrix, index=nodes, columns=nodes)

    return adj_df

def adjacency_string_to_digraph(adj_matrix_str: str, node_labels : list):
    
    # Split the string into rows and then into individual characters
    adj_matrix = [list(map(int, row)) for row in adj_matrix_str.split(',')]

    # Validate the size of the adjacency matrix and the length of node labels
    if len(adj_matrix) != len(node_labels) or any(len(row) != len(node_labels) for row in adj_matrix):
        raise ValueError("Size of adjacency matrix must match the number of node labels")

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for label in node_labels:
        G.add_node(label)

    # Add edges
    for i, row in enumerate(adj_matrix):
        for j, val in enumerate(row):
            if val != 0:
                G.add_edge(node_labels[i], node_labels[j])

    return G

## update_graph_frequencies
##################################################################################
#def update_graph_frequencies(graph_list: list, num_nodes : int, node_labels : list = None):
# 
def update_graph_frequencies_old(graph_list: list, dag_dict : dict):
    """given a list of graphs, returns a dictionary with the number of times a graph occurs

    Args:
        graph_list (list): _description_
        data (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    
    # Generate the base DAG dictionary with zero frequencies
    # dag_dict = gen_base_dag_dict(num_nodes, node_labels)
    
    # Update frequencies based on the input graph list
    for graph in graph_list:
        graph_str = convert_graph_to_str(graph)
        if graph_str in dag_dict:
            dag_dict[graph_str] += 1
            
    # normalise the frequencies
    total = sum(dag_dict.values())
    for key in dag_dict.keys():
        dag_dict[key] = dag_dict[key] / total
    
    return dag_dict




def update_graph_frequencies_less_old(graph_list: list, dag_dict: dict):
    """Given a list of graphs, returns a dictionary with the number of times a graph occurs

    Args:
        graph_list (list): List of graphs.
        dag_dict (dict): Dictionary to be updated with graph frequencies.

    Returns:
        dict: Updated dictionary with normalized frequencies.
    """
    dag_dict_cp = dag_dict.copy()
    
    # Count occurrences of each graph string
    graph_str_counter = Counter(convert_graph_to_str(graph) for graph in graph_list)
    
    # sum all values in graph_str_counter
    total = sum(graph_str_counter.values())
    
    # Batch update dag_dict
    for graph_str, count in graph_str_counter.items():
        if graph_str in dag_dict_cp.keys():
            dag_dict_cp[graph_str] = count / total

    return dag_dict_cp

# intersetion of two dictionaries
def intersection(d1, d2):
    
    d2_cp = d2.copy()
    d1_keys = set(d1.keys())
    d2_keys = set(d2_cp.keys())
    shared_keys = d1_keys.intersection(d2_keys)
        
    # sum the values of d1 
    total = sum(d1.values())
    
    for key in shared_keys:
        d2_cp[key] = d1[key] / total
    
    return d2_cp

def update_graph_frequencies(graph_list: list, result_index: dict):
    """Given a list of graphs, returns a dictionary with the number of times a graph occurs

    Args:
        graph_list (list): List of graphs.
        dag_dict (dict): Dictionary to be updated with graph frequencies.

    Returns:
        dict: Updated dictionary with normalized frequencies.
    """
    # Count occurrences of each graph string
    graph_str_counter = Counter(convert_graph_to_str(graph) for graph in graph_list)
    
    # convert graph_str_counter to a dictionary
    graph_str_dict = dict(graph_str_counter)
    result = intersection(graph_str_dict, result_index)

    return result



def is_valid_ordering_OLD(graph, ordering):
    for i, node in enumerate(ordering):
        for successor in graph.successors(node):
            if successor in ordering[:i]:
                return False
    return True

def is_valid_ordering(graph, ordering):
    for i, node in enumerate(ordering):
        for node_after in ordering[i+1:]:
            if ((graph.in_degree[node_after] == 0) and (graph.in_degree[node] > 0)):
                return False
            else:
                continue
        for successor in graph.successors(node):
            if successor in ordering[:i]:
                return False
                
    return True

def all_valid_orderings(graph):
    # Generate all permutations of nodes
    all_orderings = permutations(graph.nodes())
    # Filter out permutations that do not satisfy the precedence constraints
    valid_orderings = [ordering for ordering in all_orderings if is_valid_ordering(graph, ordering)]
    return valid_orderings



def plot_true_posterior_distribution(all_dags_dict, score = 'score_normalised', fontsize = 10, ylabel = r'Groundtruth P($G | Data$)', prob_threshold = 0.001, figsize=(7,5), title="Groundtruth posterior distribution", my_color = 'skyblue', alpha = 1,  ax = None, label = None):
    
    # Filter all_dags_dict for scores greater than 0.0001
    filtered_dags = {k: v[score] for k, v in all_dags_dict.items() if v[score] >= prob_threshold}
    
    x_labels = filtered_dags.keys()
    scores = filtered_dags.values()
    
    
    # Plotting
    plt.figure(figsize=figsize)
    plt.bar(x_labels, scores, color='skyblue')
    
    # Show ticks only for x values greater than 0
    plt.xticks(range(len(x_labels)), rotation=90, fontsize=10)

    plt.xlabel('G = $g_j$')
    plt.ylabel(r'MCMC Approximate P($G | Data$)')
    plt.title(title)
    plt.grid(False)
    plt.show()
    

def plot_approx_posterior_distribution(all_dags_dict, prob_threshold=0.001, figsize=(7,5), title="MCMC Approximate Posterior Distribution"):
        

    # Filter all_dags_dict for scores greater than 0.0001
    filtered_dags = {k: v for k, v in all_dags_dict.items() if v >= prob_threshold}
    
    x_labels = filtered_dags.keys()
    scores = filtered_dags.values()
    
    
    # Plotting
    plt.figure(figsize=figsize)
    plt.bar(x_labels, scores, color='skyblue')
    
    # Show ticks only for x values greater than 0
    plt.xticks(range(len(x_labels)), rotation=90, fontsize=10)

    plt.xlabel('G = $g_j$')
    plt.ylabel(r'MCMC Approximate P($G | Data$)')
    plt.title(title)
    plt.grid(False)
    plt.show()

    
    
def plot_approx_posterior_distribution_back( all_dags_dict, figsize=(7,5), title = "Graph Posterior Distribution"):
    
    x_labels = [str(key) for key in range(0, len(all_dags_dict.keys()))]
    
    plt.figure(figsize=figsize)
    plt.bar(x_labels, all_dags_dict.values(), color='skyblue')
    plt.xlabel(r"Graph index $g_j$")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=10)
    plt.ylabel(r"P($G |$Data)")
    plt.title(title)
    plt.grid(False)
    plt.show()
    
def plot_approx_posterior_distribution_old(all_dags_dict, num_dags_threshold=50, prob_threshold=0.001, figsize=(7,5), title="MCMC Approximate Posterior Distribution"):
    
    # copy all_dags_dict and remove the first entry
    all_dags_dict_copy = all_dags_dict.copy()
    all_dags_dict_copy.pop(0, None) 

    # Create a mapper
    id_to_adjmat = {key: idx for idx, key in enumerate(all_dags_dict_copy.keys())}
    adjmat_to_id = {idx: key for idx, key in enumerate(all_dags_dict_copy.keys())}
    
    # Filter all_dags_dict for scores greater than 0.0001
    filtered_dags = {k: v for k, v in all_dags_dict_copy.items() if v['score_normalised'] >= prob_threshold}
    
    # Determine which entries to plot based on the length of filtered_dags and their score_normalised value
    if len(filtered_dags) >= num_dags_threshold:
        further_filtered_dags = {k: v for k, v in filtered_dags.items() if v['score_normalised'] >= prob_threshold}
        x_labels = [str(id_to_adjmat[key]) for key in further_filtered_dags.keys()]
        scores = [entry['score_normalised'] for entry in further_filtered_dags.values()]
    else:
        x_labels = [str(id_to_adjmat[key]) for key in filtered_dags.keys()]
        scores = [entry['score_normalised'] for entry in filtered_dags.values()]
    
    # Plotting
    plt.figure(figsize=figsize)
    plt.bar(x_labels, scores, color='skyblue')
    
    # Show ticks only for x values greater than 0
    plt.xticks(range(len(x_labels)), rotation=90, fontsize=10)

    plt.xlabel('G = $g_j$')
    plt.ylabel(r'MCMC Approximate P($G | Data$)')
    plt.title(title)
    plt.grid(False)
    plt.show()
    
    return adjmat_to_id



    


def plot_graph(G : nx.DiGraph, title="Graph", figsize = (5,3), node_size=2000, node_color="skyblue", k=5, ax=None):
    
    pos = nx.spring_layout(G, k=k)
    if ax:
        nx.draw(G, with_labels=True, arrowsize=20, arrows=True, node_size=node_size, node_color=node_color, pos=pos, ax=ax)
        ax.margins(0.20)
        ax.set_title(title)
        ax.axis("off")
    else:
        plt.figure(figsize=figsize)
        nx.draw(G, with_labels=True, arrowsize=20, arrows=True, node_size=node_size, node_color=node_color, pos=pos)
        plt.gca().margins(0.20)
        plt.title(title)
        plt.axis("off")
        plt.show()

def compute_ancestor_matrix( adj_matrix : np.ndarray):
        
        num_nodes = adj_matrix.shape[0]
        
        # Initialize the ancestor matrix as the adjacency matrix
        ancestor_matrix = np.copy(adj_matrix)
        
        # Compute powers of the adjacency matrix and update the ancestor matrix
        power_matrix = np.copy(adj_matrix)
        for _ in range(num_nodes - 1):
            power_matrix = np.dot(power_matrix, adj_matrix)
            # If a path exists (i.e., value > 0), set it to 1
            power_matrix[power_matrix > 0] = 1
            ancestor_matrix = np.logical_or(ancestor_matrix, power_matrix).astype(int)
            
            res = ancestor_matrix.tolist()
            res = np.array(res)
            res = res.T

        return res

# Function to convert edges to adjacency matrix
def edges_to_adjacency_matrix(edges, G):
    # Create a copy of the graph to avoid modifying the original
    G_temp = nx.DiGraph()
    
    G_temp.add_nodes_from(G.nodes())
    
    # Add edges to the temporary graph
    G_temp.add_edges_from(edges)
    
    # Check if the node sets match
    if set(G.nodes()) != set(G_temp.nodes()):
        raise ValueError("The node sets of the given graph and edge list do not match.")
    
    # Create an adjacency matrix with the order of nodes from G
    adj_matrix = nx.to_numpy_array(G_temp, nodelist=G.nodes())

    return adj_matrix.astype(int)

def compute_true_distribution( all_dags, with_aug_prior = False ):
    true_distr_score = {}
    for k in all_dags.keys():
        if with_aug_prior:
            #dag_orders = len(all_valid_orderings( all_dags[k]['DAG'] ))
            #print(dag_orders)
            true_distr_score[k] = all_dags[k]['score_ordering_normalised'] #*dag_orders 
        else:
            true_distr_score[k] = all_dags[k]['score_normalised']
    
    return true_distr_score

# make a function that creates an identity matrix of size N
def create_identity_matrix(N):
    return np.identity(N)

# create a function that creates a NxN matrix with all 1s
def create_ones_matrix(N):
    return np.ones((N,N))

# incidence - adjacency matrix of the current DAG
def convert_adj_mat_to_graph( incidence: pd.DataFrame ):
    
    labels = list( incidence.columns )

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(labels)

    # Add edges based on adjacency matrix
    for i in range(incidence.shape[0]):
        for j in range(incidence.shape[1]):
            if incidence.iloc[i, j] == 1:
                G.add_edge(labels[i], labels[j])
    
    return G

# n - how many nodes
# p - probability of an edge
# labels - list of node labels
def rDAG(n : int, p : float , labels : str):
    adjmat = np.zeros((n, n))
    adjmat[np.tril_indices_from(adjmat, k=-1)] = np.random.binomial(1, p, size=int(n * (n - 1) / 2))
    return pd.DataFrame(adjmat, columns=labels,index=labels)


def data_generation ( G: nx.DiGraph, num_obs: int, min_value: float = 0.5,  max_value: float = 2.5, max_noise: float = 1, random_seed: int = 42):

    np.random.seed(random_seed)

    # Number of variables
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    # Generate the adjacency matrix
    adj_mat = get_adjacency_matrix( G )

    # Random values between 0 and 1, scaled and shifted to the desired range
    random_matrix = np.random.rand(num_nodes, num_nodes) * (max_value - min_value) + min_value

    # Random sign of the weights
    sampled_values = np.random.choice([-1, 1], size=num_nodes*num_nodes, replace=True)

    # Generate the weights matrix, non zero values indicate betas
    W_mat = adj_mat.values * random_matrix * sampled_values.reshape(num_nodes,num_nodes) + np.eye(num_nodes)

    # Generate the diagonal conditional variance matrix, diagonal values indicate sigma^2_j
    D_mat = np.eye(num_nodes) * np.random.uniform(0.1, max_noise, num_nodes)

    # Covariance matrix 
    Sigma = np.linalg.pinv(W_mat.T) @ D_mat @ np.linalg.pinv(W_mat)

    # Generate data from MVN distribution
    data = multivariate_normal.rvs(cov=Sigma, size=num_obs)

    return pd.DataFrame(data, columns=nodes)

def add_edges_to_graph(edges, G_init):
    G = G_init.copy()
    G.add_edges_from(edges)
    return G


def remove_edges_from_graph(edges, G_init):
    G = G_init.copy()
    G.remove_edges_from(edges)
    return G


def count_parents(adj_matrix, node):
    """
    Count the number of parents for a given node in a directed graph.

    :param adj_matrix: numpy array representing the adjacency matrix of the graph
    :param node: index of the node to count parents for
    :return: number of parents (incoming edges) of the specified node
    """
    if node < 0 or node >= adj_matrix.shape[0]:
        raise ValueError("Node index out of bounds")
    
    # Count the non-zero entries in the node's column
    return np.count_nonzero(adj_matrix[:, node])

def find_parents(adj_matrix, node):
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