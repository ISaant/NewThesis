#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 06 18:44:45 2023

@author: isaac
"""
import time

import bct
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from netneurotools import networks
import bct
from sklearn.preprocessing import binarize
from tqdm import tqdm
import pickle
# features con be extracted using adj with nodos with zero connection.
# no need to use connectome_mod
def traditionalMetrics(connectome, Length, start_at):
    if start_at != 0:
        # print(f'start at: {start_at}')
        connectome = connectome[start_at:]
        Length = Length[start_at:]
        with open('Schaefer200_Sc_Features.pickle', 'rb') as f:
            features = pickle.load(f)
        local = features[0]
        glob = features[1]
    else:
        print('start at: 0')
        local = []
        glob = []
    for e,(adj,length) in tqdm(enumerate(zip(connectome,Length)), initial=start_at, total=len(connectome)):
        # print(e,adj.shape,length.shape)
        adj_bin = binarize(adj)
    #Modifica tu algiritmo para encontrar el minimum-spanning-tree para el treshold en Fc
    #Micro
        # Centrality:
        #bin
        degree = bct.degrees_und(adj_bin)
        page_rank_bin=bct.pagerank_centrality(adj_bin, d= 0.85) #the result does change between bin and wei
        eigen_cen_bin = bct.eigenvector_centrality_und(adj_bin) #the result does change between bin and wei
        betweenes_bin = bct.betweenness_bin(adj_bin)
        # edge_betweenes_bin = bct.edge_betweenness_bin(adj_bin)
        subG_cent_bin = bct.subgraph_centrality(adj_bin)
        clust_coeff_bin = bct.clustering_coef_bu(adj_bin)
        eff_bin = bct.efficiency_bin(adj_bin,local=True)



        #wei
        strength = bct.strengths_und(adj)
        page_rank_wei = bct.pagerank_centrality(adj, d=0.85)
        eigen_cen_wei = bct.eigenvector_centrality_und(adj)
        betweenes_wei = bct.betweenness_wei(adj)
        # edge_betweenes_wei = bct.edge_betweenness_wei(adj)
        subG_cent_wei = bct.subgraph_centrality(adj)
        clust_coeff_wei = bct.clustering_coef_wu(adj)
        eff_wei = bct.efficiency_wei(adj,local=True)

        local.append([degree, page_rank_bin, eigen_cen_bin, betweenes_bin, subG_cent_bin, clust_coeff_bin, eff_bin,
        strength, page_rank_wei, eigen_cen_wei, betweenes_wei, subG_cent_wei, clust_coeff_wei, eff_wei])
    #Meso?...


    #Macro
    # Global efficiency
        Eff_bin = bct.efficiency_bin(adj_bin, local=False)
        Eff_wei = bct.efficiency_wei(adj, local=False)

        #Characteristic Path Lenght
        lengthW = length * adj_bin
        pathl, _ = bct.distance_wei(lengthW)
        pathl = np.tril(pathl, k=1).flatten()
        CPL_wei = sum(pathl) / len(length)

        pathlr = bct.distance_bin(adj_bin)
        pathlr = np.tril(pathlr, k=1).flatten()
        CPL_bin = sum(pathlr) / len(length)
        # Fragmentation
        # Richclub ... How to define k? ... for now is difficult to interpret
        # WightedRichclub ...
        # Assortivity
        Assort_bin = bct.assortativity_bin(adj_bin)
        Assort_wei = bct.assortativity_wei(adj)
        # Characteristic path length/ For disconnected, the harmonic mean can be use

        # Shortest path probability this can be global too
        # mean first passage time
        # Diffusion efficiency
        # Weighted communicability
        # Clustering coefficient ... check if it can be used on weighted
        C_wei = sum(bct.clustering_coef_wu(adj)) / len(adj)
        C_bin = sum(bct.clustering_coef_bu(adj_bin)) / len(adj)
        # Small-worldness
        Sigma_wei, Sigma_bin=SmallWorld(adj, adj_bin, length)
        glob.append([Eff_bin, CPL_bin, Assort_bin, C_bin, Sigma_bin, Eff_wei, CPL_wei, Assort_wei, C_wei, Sigma_wei])
        if (e+start_at) % 10 == 0:
            print('Saving...')
            with open('Schaefer200_Sc_Features.pickle', 'wb') as f:
                pickle.dump([local, glob], f)

    with open('Schaefer200_Sc_Features.pickle', 'wb') as f:
        pickle.dump([local, glob], f)
        #Wights should be remaped to compute the shortest path, but not for "serach info"
        # Dijkstra’s algorithm can be used to compute the shortest paths in both
        # directed and undirected networks, as long as the edge weights are non-negative.
    return local, glob
#%%
def SmallWorld(adj,adj_bin,length):

    newB, newW, nr = networks.match_length_degree_distribution(adj,length,nbins=10, nswap=len(adj)*20)
    randomBinary,_ = networks.randmio_und(adj_bin, itr= 250)
    c = sum(bct.clustering_coef_wu(adj)) / len(adj)
    cr = sum(bct.clustering_coef_wu(newW)) / len(adj)

    lengthl = length * adj_bin
    pathl,_ = bct.distance_wei(lengthl)
    pathl = np.tril(pathl,k=1).flatten()
    l = sum(pathl)/len(length)
    lengthlr = length * newB
    pathlr,_ =  bct.distance_wei(lengthlr)
    pathlr = np.tril(pathlr, k=1).flatten()
    lr = sum(pathlr)/len(length)


    # print(c,cr,l,lr)
    gamma = c/cr
    lamb = l/lr
    sigmaW = gamma / lamb

    c = sum(bct.clustering_coef_bu(adj_bin)) / len(adj)
    cr = sum(bct.clustering_coef_bu(randomBinary)) / len(adj)

    pathl = bct.distance_bin(adj_bin)
    pathl = np.tril(pathl, k=1).flatten()
    l = sum(pathl) / len(length)

    pathlr = bct.distance_bin(randomBinary)
    pathlr = np.tril(pathlr, k=1).flatten()
    lr = sum(pathlr) / len(length)

    # print(c, cr, l, lr)
    gamma = c / cr
    lamb = l / lr
    sigmaB = gamma / lamb


    return sigmaW, sigmaB
#%%

def graphlet_degree_vector(G, k):
    graphlet_counts = np.zeros(k+1, dtype=int)

    for node in G.nodes():
        neighbors = set(G.neighbors(node))
        subgraph = G.subgraph(neighbors.union({node}))
        subgraph_order = len(subgraph)
        if 2 <= subgraph_order <= k:
            graphlet_counts[subgraph_order] += 1

    return graphlet_counts[2:]

# Ejemplo de uso
# G = nx.erdos_renyi_graph(20, 0.3)
# graphlet_vector = graphlet_degree_vector(G, 10)
# print("Graphlet Degree Vector:", graphlet_vector)
# nx.draw(G)
# plt.show()
#%%

import itertools

def generate_graphlets(graph, size):
    graphlets = []
    for nodes in itertools.combinations(graph.nodes(), size):
        subgraph = graph.subgraph(nodes)
        if nx.is_connected(subgraph):  # Puedes ajustar esta condición según tus necesidades
            graphlets.append(subgraph)
    return graphlets

def calculate_graphlet_degree_vector(graph, max_size):
    graphlet_degree_vector = [0] * max_size

    for size in range(2, max_size + 1):
        graphlets = generate_graphlets(graph, size)
        for g in graphlets:
            for node in graph.nodes():
                if nx.is_isomorphic(g, graph.subgraph(list(g.nodes()) + [node])):
                    graphlet_degree_vector[size - 2] += 1

    return graphlet_degree_vector

# Ejemplo de uso:
# G = nx.erdos_renyi_graph(20, 0.3)
# max_graphlet_size = 5
# graphlet_degree_vector = calculate_graphlet_degree_vector(G, max_graphlet_size)
# print("Graphlet Degree Vector:", graphlet_degree_vector)
# nx.draw(G)
# plt.show()
#%%

