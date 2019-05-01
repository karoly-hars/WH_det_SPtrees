import scipy
import networkx as nx
import numpy as np
import random


def insert_wormhole(positions, connectivity_graph, wormhole_radius, wormhole_type, min_dist):
    """ Add wormhole to an input network graph

    :param positions: Positions of the sensors in the network
    :param connectivity_graph: The network connectivity graph
    :param wormhole_radius: Radius of the wormhole endpoints
    :param wormhole_type: 3 possible types of wormholes:
        1.:  x------------x  : only the endpoints are affected by the wormhole edges

        2.:  ___
            / x \
           | x  |---------x  : on one side only the endpoint is affected, on the other side,
           \  x/               both the endpoint and the nodes in the wormhole radius are affected

        3.:  ___          ___
            /  x\--------/  x\
           |  x |-------|  x | : on both sides of the wormhole,
           \ x /--------\ x /    the surrounding nodes are affected, based on the wormhole radius

    :param min_dist: The minimum hop-distance of the wormhole endpoints.
                     If the min_dist is too large, the insertion might be unsuccessful
    :return: The network graph with the added wormhole edges, and the list of wormhole nodes at the two endpoints
    """
    num_points = len(positions)
    
    # try a 1000 times
    for i in range(0, 1000):
        modified_graph = connectivity_graph.copy()
        
        # randomly choose a point as an endpoint
        endpoint1 = np.random.randint(0, num_points)
        
        # choose a 2nd endpoint that is far enough
        distances = nx.shortest_path_length(connectivity_graph, source=endpoint1)
        possible_endpoint2s = [node for node in distances if distances[node] > min_dist]
        if len(possible_endpoint2s) == 0:
            continue
        endpoint2 = random.choice(possible_endpoint2s)
        
        # calculate distances
        Y = scipy.spatial.distance.pdist(np.asarray(list(positions.values())), "euclidean")
        dist_matrix = scipy.spatial.distance.squareform(Y)
        
        endpoints1 = [endpoint1]
        endpoints2 = [endpoint2]
        
        # find the other affected nodes, based on the wormhole_radius and wormhole_type
        if wormhole_type >= 2:
            for p in range(0, num_points):
                if p != endpoint2 and dist_matrix[endpoint2, p] < wormhole_radius:
                    endpoints2.append(p)
        
        if wormhole_type == 3:
            for p in range(0, num_points):
                if p != endpoint1 and dist_matrix[endpoint1, p] < wormhole_radius:
                    endpoints1.append(p)
        
        # introduce the new wormhole edges
        for wormhole_point1 in endpoints1:
            for wormhole_point2 in endpoints2:
                modified_graph.add_edge(wormhole_point1, wormhole_point2)
        
        return modified_graph, endpoints1, endpoints2
        
    assert len(possible_endpoint2s) > 0, "Error: unable to generate a wormhole with these parameters.\n"
