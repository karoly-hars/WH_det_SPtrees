import scipy
import networkx as nx
import numpy as np
import random


'''
adds a wormhole to an input network

params:
positions: positions of the sensors in the network
G: connectivity graph
WH_radius: radius of the wormhole endpoints
WH_type:

1.:  x------------x  :: only the endpoints are affected by the wormhole edges

2.:  ___
    / x \--------
   |  x  |---------x  :: on one side only the endpoint is affected, on the other side, both the enpoint and the nodes in the wormhole radius are affected
    \  x/--------

3.:  ___           ___
    /  x\---------/  x\
   |  x  |-------|  x  | :: on both sides of the wormhole, the surrounding nodes are affected, based on the wormhole radius
    \  x/---------\ x /

min_dist: the minimum hop-distance of the wormhole endpoints.
if the min_dist is too large, the insertion might be unsuccessful
'''
def insert_wormhole(positions, G, WH_radius, WH_type, min_dist):
    
    npoints = len(positions)
    
    # try a 1000 times
    for i in range(0,1000):
        mod_G = G.copy()
        
        # randomly choose a point as an endpoint
        endpoint1 = np.random.randint(0, npoints)
        
        # choose a 2nd endpoint thats far enough
        distances = nx.shortest_path_length(G, source=endpoint1)
        possible_endpoint2s = [node for node in distances if distances[node] > min_dist]
        if len(possible_endpoint2s) == 0:
            continue
        endpoint2 = random.choice(possible_endpoint2s)
        
        # calculate distances
        Y = scipy.spatial.distance.pdist(np.asarray(list(positions.values())), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(Y)
        
        endpoints1 = [endpoint1]
        endpoints2 = [endpoint2]
        
        # find the other affected nodes, based on the WH_radius and WH_type    
        if WH_type >= 2:
            for i in range(0,npoints):
                if i != endpoint2 and dist_matrix[endpoint2, i] < WH_radius:
                    endpoints2.append(i)
        
        if WH_type == 3:
            for i in range(0,npoints):
                if i != endpoint1 and dist_matrix[endpoint1, i] < WH_radius:
                    endpoints1.append(i)
        
        # introduce the new wormhole edges
        for WH_point1 in endpoints1:
            for WH_point2 in endpoints2:
                mod_G.add_edge(WH_point1, WH_point2)        
        
        return mod_G, endpoints1, endpoints2
        
    assert len(possible_endpoint2s)>0, 'Error: unable to generate a wormhole with these parameters.\n'
