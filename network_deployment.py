import numpy as np
import math
import networkx as nx
import scipy.spatial
import random

''' 
generates a network into a square shaped area,
using random placement, and unit disk graph comm. model

params:
num_nodes: number of sensors.
radius: communication radius of the sensors
side_len: length of the sides of the square
'''
def generateSquare(num_nodes, radius, side_len):
    # try 1000 times, to generate a random graph with these parameters
    for X in range(0,1000):
        # make graph, add nodes and random positions
        G = nx.Graph()
        G.add_nodes_from(range(0,num_nodes))
        positions = {}
        positions_ = []
        for i in range(0, num_nodes):
            positions[i] = [np.random.rand()*side_len, np.random.rand()*side_len]
            positions_.append(positions[i])
        
        # calculate pairwise distances
        Y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(Y)
        
        # add edges based on radius
        for i in range(0,num_nodes-1):
            for j in range(i,num_nodes):
                if dist_matrix[i,j] <= radius and i!=j:
                    G.add_edge(i, j)
        # if the generated network is not connected, generate a new one
        if nx.is_connected(G):
            return positions, G
            
    assert False, 'Error: the graph is not connected with these parameters!\n'    


''' 
generates a network into a square shaped area,
using random placement, and quasi unit disk graph comm. model:

params:
num_nodes: number of sensors. 
radius: communication radius of the sensors
side_len: length of the sides of the square
p: parameter for the quasi comm. model
'''
def generateSquareQuasi(num_nodes, radius, side_len, p=0.5):
    
    radius2 = radius
    radius1 = radius*p 
    # try 1000 times, to generate a random graph with these parameters
    for X in range(0,1000):
        # make graph, add nodes and random positions
        G = nx.Graph()
        G.add_nodes_from(range(0,num_nodes))
        positions = {}
        positions_ = []
        for i in range(0, num_nodes):
            positions[i] = [np.random.rand()*side_len, np.random.rand()*side_len]
            positions_.append(positions[i])
        
        # calculate pairwise distances
        Y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(Y)
        
            
        # add edges based on radius and param p
        for i in range(0,num_nodes-1):
            for j in range(i,num_nodes):
                if dist_matrix[i,j] <= radius1 and i!=j:
                    G.add_edge(i, j)
                if dist_matrix[i,j] > radius1 and dist_matrix[i,j] <=radius2:
                    asd = np.random.rand()
                    alpha = radius1 / radius2
                    prob = (alpha/(1 - alpha)) * ((radius2/dist_matrix[i,j]) - 1)
                    
                    if asd < prob:                        
                        G.add_edge(i, j)
                    
        # if the generated network is not connected, generate a new one
        if nx.is_connected(G):
            return positions, G
            
    assert False, 'Error: the graph is not connected with these parameters!\n'    



''' 
generates a network into a square shaped area,
using perturbed grid placement, and unit disk graph comm. model

params:
num_nodes: number of sensors. in this case a square number would be ideal
radius: communication radius of the sensors
side_len: length of the sides of the square
noise: noise used for perturbation
'''
def generateGrid(num_nodes, radius, side_len, noise=0.75):
    
    assert (math.sqrt(num_nodes).is_integer()), 'the number of sensors must be a square number'

    # try 1000 times, to generate a random graph with these parameters    
    for X in range(0,1000):    
        H = int(math.sqrt(num_nodes))
        node_dist = side_len / H
        max_noise = node_dist*noise

        G = nx.Graph()
        G.add_nodes_from(range(0,num_nodes))
        positions_ = []
        positions = {}

        # deploy network on grid
        for i in range(0,H):
            for j in range(0,H):
                rx = random.uniform(-1*max_noise, max_noise)
                ry = random.uniform(-1*max_noise, max_noise)
                positions[i + j*H] = [i* node_dist + rx, j* node_dist + ry]
                positions_.append(positions[i + j*H])
                
        Y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(Y)

        # add edges based on radius
        for i in range(0,num_nodes-1):
            for j in range(i,num_nodes):
                if dist_matrix[i,j] <= radius and i!=j:
                    G.add_edge(i, j)

        if nx.is_connected(G):
            return positions, G
            
    assert False, 'Error: the graph is not connected with these parameters!\n'    

        
''' 
generates a network into a square shaped area,
using perturbed grid placement, and quasi unit disk graph comm. model

params:
num_nodes: number of sensors. in this case a square number would be ideal
radius: communication radius of the sensors
side_len: length of the sides of the square
p: parameter for the quasi comm. model
noise: noise used for perturbation
'''    
def generateGridQuasi(num_nodes, radius, side_len, p=0.5, noise=0.75):
    
    assert (math.sqrt(num_nodes).is_integer()), 'Error: the number of sensors must be a square number\n'
    
    radius2 = radius
    radius1 = radius*p 
    
    # try 1000 times, to generate a random graph with these parameters    
    for i in range(0,1000):    
        H = int(math.sqrt(num_nodes))
        node_dist = side_len / H
        max_noise = node_dist*noise

        G = nx.Graph()
        G.add_nodes_from(range(0,num_nodes))
        positions_ = []
        positions = {}

        # deploy network on grid
        for i in range(0,H):
            for j in range(0,H):
                rx = random.uniform(-1*max_noise, max_noise) #max_noise*np.random.rand() - max_noise
                ry = random.uniform(-1*max_noise, max_noise) #max_noise*np.random.rand() - max_noise
                positions[i + j*H] = [i* node_dist + rx, j* node_dist + ry]
                positions_.append(positions[i + j*H])
                
        Y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(Y)

        # add edges based on radius
        for i in range(0,num_nodes-1):
            for j in range(i,num_nodes):
                if dist_matrix[i,j] <= radius1 and i!=j:
                    G.add_edge(i, j)
                if dist_matrix[i,j] > radius1 and dist_matrix[i,j] <=radius2:
                    asd = np.random.rand()
                    alpha = radius1 / radius2
                    prob = (alpha/(1 - alpha)) * ((radius2/dist_matrix[i,j]) - 1)
                    if asd < prob:                        
                        G.add_edge(i, j)

        if nx.is_connected(G):
            return positions, G
            
    assert False, 'Error: the graph is not connected with these parameters!\n'        
