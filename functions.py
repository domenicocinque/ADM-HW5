from collections import defaultdict
import random
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import queue
import statistics 
from queue import PriorityQueue
from tqdm import tqdm 
from collections import deque
from time import time
import queue



def graphInfo(G):
    
    """ Gives all the requested infos about our graph
    
    Args:
        G : the initial graph we have built
        
    Returns:
        The function prints:
            -The number of articles V
            -The number of hyperlinks E
            -The average number of links per page
            -The graph density
    """
    
    V = G.number_of_nodes()
    E = G.number_of_edges()
    print('We have a graph containing',V,'articles and',E,'hyperlinks.')
    print('The average number of links per page is:', round(E/V,4))
    print('The graph density is',E/(V*(V-1)))
    return
    
    
    
def RQ2(page, nr_clicks, df_links):
    
    """ Returns the result required by RQ2
    
    Args:
        page: the starting page of the user
        nr_clicks: the maximum number of clicks a user can do
        df_links: the dataframe that contains all the edges in the graph
        
    Returns:
        the set of all pages that a user can reach within d clicks
    """
    
    reachable = []
    new_nodes = [page] # Starting page
    for i in range(nr_clicks):
         # Nodes that we can reach from the new_nodes in one click
        a = list(df_links.loc[df_links["origin"].isin(new_nodes)]["destination"]) 
        
        # Add those nodes to the reachable 
        reachable.extend(a)
        
        # Update new_nodes
        new_nodes = a
        
    return pd.DataFrame(list(set(reachable)))



def BFS(G,v):
    
    """ Implements the BFS algorithm
    
    Args:
        G: the graph on which the BFS is implemented
        v: the starting node for the BFS
        
    Returns:
        dist: a dictionary that associates to each reached node the number of edges necessary to reach that node from v. 
        paths: a dictionary that associates to each node the path (represented as a list of nodes) to reach it from v.
            This corresponds to the shortest path in graphs with all edge weights equal to 1.
    """
    
    q = deque()
    dist = dict()
    paths = defaultdict(list)
    
    visited = [v]
    dist[v] = 0
    paths[v].append(v)
    
    q.append(v)
    while len(q) != 0:
        v = q.popleft()
        for u in G.neighbors(v):
            if u not in visited:
                visited.append(u)
                dist[u] = dist[v] + 1
                paths[u] += paths[v] + [u]
                q.append(u)
                
    return dist, paths



def categoryGraph(G,C,nodes):
    
    """ 
    Args:
        G: the total graph
        C: the category for which we want to return the subgraph
        nodes: a dictionary where each node is a key and its value is in turn a dictionary containing the attributes "category"
            and "name" for that node
        
    Returns:
        out: the subgraph of all the nodes in C
    """
    
    out = nx.MultiDiGraph()
    
    for node in G.nodes():
        cat = G.nodes[node]['category']
        if cat == C:
            out.add_nodes_from([(int(node), nodes[node])])
            
    new_edges = []
    for edge in G.edges():
        if edge[0] in out.nodes() and edge[1] in out.nodes():
            new_edges.append(edge)
    out.add_edges_from(new_edges)
    
    return out



# Dummy function to initialize default dict with zeros
def zero():
    return 0



def most_central_article(subset, df_links):
    
    """ The function finds the most central article which is the one with the highest indegree
    
    Args:
        subset: the subset of nodes we are considering
        df_links: the dataframe that contains all the edges in the graph
        
    Returns:
        out: the most central article according to the indegree centrality
    """
    
    d = defaultdict(zero)
    n_nodes = len(df_links)
    for edge in subset.edges():
        d[edge[1]] += 1
    out = max(d, key=d.get)
    return out



def RQ3(G,C,df_links,nodes, S=None):
    
    """ The function prints the answer for RQ3
    
    Args:
        G: the total graph
        C: the category we are considering
        df_links: the dataframe that contains all the edges in the graph
        nodes: a dictionary where each node is a key and its value is in turn a dictionary containing the attributes "category"
            and "name" for that node
        S (optional): the subset of nodes inside C that we are considering.
        
    Returns:
        The function prints the approximate number of clicks to reach all nodes in S
    """
    
    # nG is the graph with all the nodes belonging to one category
    nG = categoryGraph(G,C,nodes)
    v = most_central_article(nG,df_links)
    
    # All shortest paths from v computed through BFS
    _, paths = BFS(nG,v)
    
    # If the set of pages is not given in input, we sample from the nodes that we can reach
    if S == None:
        S = set(random.sample(paths.keys(), 100))
        S.add(v)
    
    # Checking connectivity
    # if there is a node of S that we cannot reach from v 
    # return 'Not possible'
    if not set(paths.keys()).issuperset(S):
        return 'Not possible'
    
    # Creating the minimum tree that reaches all nodes in S from v
    pathGraph = nx.DiGraph()
    for node in paths.keys():
        if node in S:
            pathGraph.add_nodes_from([(int(node), nodes[node])]) 
    
    for node in paths.keys():
        if node in S:
            for i in range(len(paths[node])-1):
                pathGraph.add_edge(paths[node][i],paths[node][i+1])
    
    # Now we add edges until we have a path
    new_edges = []
    for node in pathGraph:
        
        # If the node is a dead end we add the first edge of nG that goes from that node to the tree
        if len(pathGraph.out_edges(node)) == 0:
            found = False
            for edge in nG.out_edges(node):
                if edge[1] in pathGraph.nodes:
                    new_edges.append((edge[0],edge[1]))
                    found = True
                    break
                    
            # If an edge from that node to the tree does not exists we add an edge that reconnects to v
            if found == False:
                new_edges.append((node,v))
               
    pathGraph.add_edges_from(new_edges)       
    
    print('---------------------------------------------')
    print('The approximated number of clicks is', pathGraph.number_of_edges())
    return
    
    

def subGraph(G,C1,C2,nodes):
    
    """ The function creates the subgraph induced by two selected categories C1 and C2
    Args:
        G: the total graph
        C1: the first category we are considering
        C2: the second category we are considering
        nodes: a dictionary where each node is a key and its value is in turn a dictionary containing the attributes "category"
            and "name" for that node
        
    Returns:
        out: the subgraph of all the nodes in C1 and C2
    """
    
    out = nx.MultiDiGraph()

    for node in G.nodes():
        cat = G.nodes[node]['category']
        if cat == C1 or cat == C2:
            out.add_nodes_from([(int(node), nodes[node])])
            
    new_edges = []
    for edge in G.edges():
        if edge[0] in out.nodes() and edge[1] in out.nodes():
            new_edges.append(edge)
            
    out.add_edges_from(new_edges)       
            
    return out



def contract(G, del_edge):
    
    """ The function performs the contraction of a given edge
    
    Args:
        G: the graph we are considering
        del_edge: the edge that will be deleted in the contraction procedure
        
    Returns:
        The function doesn't return anything but has a side effect on input graph G
    """
    
    to_keep = del_edge[1]
    to_delete =  del_edge[0]
    
    edges = G.edges(to_delete)
    to_add = []
    for edge in edges:
        if edge[1] != to_keep:
            to_add.append((edge[1], to_keep))
            
    G.remove_node(to_delete)
    G.add_edges_from(to_add)
    return



def Karger(G, N = None):
    
    """ The function uses Karger algorithm to estimats the minimum cut of a connected graph
    
    Args:
        G: the graph we are considering
        N(optional): the number of times the function will run Karger algorithm
        
    Returns:
        The function returns the minimum value found among all the simulations of the Karger algorithm
    """
    
    # number of simulations N, if not given in input uses the optimal number
    if N == None:
        n = len(G.nodes)
        N = n**2*np.log(n)
        
    results = []
    for i in range(int(N)):
        nG = G.to_undirected().copy()
    
        while nG.number_of_nodes() > 2:
            if nG.number_of_edges() > 1:
                deleted_edge = random.choice(list(nG.edges()))
                contract(nG, deleted_edge) 
            else:
                return 0

        results.append(len(list(nG.edges())))
        
    return min(results)



def RQ4(G,C1,C2,u,v,N,nodes):
    
    """ The function returns the result requested by RQ4
    
    Args:
        G: the total graph
        C1: the first category we are considering
        C2: the second category we are considering
        u: the first node we are considering
        v: the second node we are considering
        N: the number of times the Karger algorithm will run
        nodes: a dictionary where each node is a key and its value is in turn a dictionary containing the attributes "category"
            and "name" for that node
        
    Returns:
        The function returns the minimum value found among all the simulations of the Karger algorithm
    """
    
    # checking if the given nodes are in the choosen categories
    if not (nodes[u]['category'] == C1 or nodes[u]['category'] == C2):
        print("ERROR: the node ", u ,"is not in either of the two categories")
        return
    
    if not (nodes[v]['category'] == C1 or nodes[v]['category'] == C2):
        print("ERROR: the node ", v ,"is not in either of the two categories")
        return
    
    subG = subGraph(G,C1,C2,nodes)
    paths_u = BFS(subG, u)[1]
    paths_v = BFS(subG, v)[1]
    
    # We create the subgraph with all the possible paths from u and from v
    pathGraph = nx.DiGraph()
    for node in paths_u.keys():
        pathGraph.add_nodes_from([(int(node), nodes[node])]) 
        
    for node in paths_v.keys():
        pathGraph.add_nodes_from([(int(node), nodes[node])]) 
     
    for node in paths_u.keys():
        for i in range(len(paths_u[node])-1):
            pathGraph.add_edge(paths_u[node][i],paths_u[node][i+1])
            
    for node in paths_v.keys():
        for i in range(len(paths_v[node])-1):
            pathGraph.add_edge(paths_v[node][i],paths_v[node][i+1])  
    
    # Apply Karger to the resulting subgraph
    out = 'The minimum number of hyperlinks to remove in order to disconnect node u and node v is ' + str(Karger(pathGraph,N))
    return out



def RQ5(G,C,nodes,remaining):
    
    """ The function returns the result requested by RQ5
    
    Args:
        G: the total graph
        C: the category we are considering
        nodes: a dictionary where each node is a key and its value is in turn a dictionary containing the attributes "category"
            and "name" for that node
        remaining: the list of all the categories in the total graph
        
    Returns:
        The function returns, for each category in 'remaining', the distance between this category and 'C', with the distance
            between two categories defined as the median of the shortest paths from each pair of nodes in the two categories
    """
    
    t1 = time()
    
    dist = dict()
    categories = set(remaining).difference(set(C))
    scores = defaultdict(list)
    origins = [v for v in G if G.nodes[v]['category'] == C]
    
    i = 1
    print('Computing distances...')
    for cat in categories:
        print(i, 'out of',len(categories),'categories.')
        dist_from_cat = []
        nG = subGraph(G,C,cat,nodes)
        
        for node in origins:
            bfs_v = BFS(nG, node)[0] # All distances from each node in the main category C
            
            # We exclude in the computation the nodes that are not reachable 
            bfs_v = {k:v for k,v in bfs_v.items() if v != np.infty and nG.nodes[k]['category'] == cat} 
            dist_from_cat += list(bfs_v.values())
            
        scores[cat] += dist_from_cat    
        i += 1
    
    # At this point we have, for each category, a vector containing all the distances
    medians = dict()
    for k,v in scores.items():
        if len(v) > 0:
            medians[k] = statistics.median(v)
        else:
            # If the categories are disconnected we consider the distance as infinity
            medians[k] = np.infty
    
    # Sorting
    medians = {k: v for k, v in sorted(medians.items(), key=lambda item: item[1])}
    
    t2 = time()
    print('Reached solution in %f seconds' %(t2-t1))
    return medians



def RQ6(catG, alpha=0.15):
    
    """ The function returns the result requested by RQ6
    
    Args:
        catG: a graph in which each node represents a category
        alpha: the probability of going to a random node of the graph in the PageRank algorithm
        
    Returns:
        The function prints the PageRank score (multiplied by 100 to make it more readable) for each category/node in 'catG'
    """
    
    # Creating transition matrix from adjacecy matrix
    A = nx.adjacency_matrix(catG).toarray()
    sum_of_rows = A.sum(axis=1)
    A_norm = A / sum_of_rows[:, np.newaxis]  
    P = alpha * np.ones((len(A),len(A)))/len(A) + (1-alpha)*A_norm
    
    # Starting vector
    v = np.ones(len(A))/len(A)
    
    # Stop condition
    eps = 0.001
    while 1:
        v1 = v
        v = np.dot(v,P)
        if np.all(v1 - v < eps):
            break
            
    results = dict()
    
    print('|------------------|')
    print('| PageRank results |')
    print('|------------------|')
    print()
    
    for i in range(len(v)):
        # We multiplied the PR score by 100 to make it more readable but keep in mind that
        # the actual PR score is a probability that will sum up to one
        results[i] = round(v[i]*100,2)
       
    # Sorting
    for el in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print('Score:', el[1], 'Category:', catG.nodes[el[0]]['category'])
        
    return results

def plotPR(catG, results):
    
    """ Plots the results of the PageRank algorithm
    
    Args:
        catG: the reduced graph
        results: the output of the PageRank algorithm 
        
    Returns:
        The plot of the graph with the relative score label on each node
    """
    
    plt.figure(figsize=(20,10))
    pos = nx.spring_layout(catG, seed=42)
    d = dict(catG.degree)

    node_sizes = [v * 400 for v in results.values()]
    cmap = plt.cm.plasma

    nodes = nx.draw_networkx_nodes(catG, 
                                   pos, 
                                   node_size=node_sizes, 
                                   node_color="orange",
                                   alpha=0.85)

    edges = nx.draw_networkx_edges(catG,
                                   pos,
                                   node_size=node_sizes,
                                   arrowstyle= "->",
                                   arrowsize= 7,
                                   width= 1,
                                   alpha =0.5)

    labels = nx.draw_networkx_labels(catG, 
                                     pos, 
                                     font_family='DIN Alternate',
                                     labels = results,
                                     font_size=14)


    ax = plt.gca()
    ax.set_axis_off()
    plt.show()
