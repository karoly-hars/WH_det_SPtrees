import network_deployment
import insert_wormhole
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import argparse


def get_root_nodes(connectivity_graph_with_wormhole, min_dist):
    """Choose root nodes randomly but evenly.

    :param connectivity_graph_with_wormhole: The graph with a wormhole inserted
    :param min_dist: The desired minimum distance between two root nodes
    :return: The list of root nodes
    """
    nodes = [i for i in range(0, len(connectivity_graph_with_wormhole))]  # set of all nodes
    root_nodes = []

    while len(nodes) != 0:
        # add a random point to the set of roots
        root_node = random.choice(nodes)
        root_nodes.append(root_node)

        # remove it from the
        distances = nx.shortest_path_length(connectivity_graph_with_wormhole, source=root_node)
        to_rem = [key for key in distances if distances[key] < min_dist]
        nodes = [x for x in nodes if x not in to_rem]

    assert len(root_nodes) > 1, 'the number of root nodes should be at least 2. choose a smaller k parameter'

    return root_nodes


def find_wormhole(connectivity_graph_with_wormhole, root_nodes, threshold, endpoints, make_plot=False):
    """Find the wormholes in the input graph.

    :param connectivity_graph_with_wormhole: The graph with a wormhole inserted
    :param root_nodes: IDs of the root nodes
    :param threshold: Above this threshold, nodes will be declared wormhole candidates
    :param endpoints: IDs of the wormhole endpoints, only used for visualization
    :param make_plot: -
    :return: The list of candidate nodes
    """
    var_matrix = np.zeros(shape=(len(root_nodes), len(connectivity_graph_with_wormhole)))

    if make_plot:
        plt.figure(1, figsize=(5, 10))

    # run detection from each root node
    for idx, root_node in enumerate(root_nodes):
        print('running detection from root {}/{}'.format(idx+1, len(root_nodes)))

        if make_plot:
            plt.subplot(len(root_nodes)+1, 1, (idx+1))
            plt.title('root {}:'.format(idx), loc='left')

        # we do not test the root and its direct neighbors in this round
        to_bypass = [root_node] + list(connectivity_graph_with_wormhole.neighbors(root_node))
        original_distances = nx.shortest_path_length(connectivity_graph_with_wormhole, root_node)

        for node in range(len(connectivity_graph_with_wormhole)):
            if node not in to_bypass:

                tmp_graph = connectivity_graph_with_wormhole.copy()
                nodes_to_delete = [node] + list(connectivity_graph_with_wormhole.neighbors(node))

                # delete the node and run a new BFS
                for node_to_delete in nodes_to_delete:
                    tmp_graph.remove_node(node_to_delete)
                new_distances = nx.shortest_path_length(tmp_graph, root_node)

                # calculate the distance differences
                distance_differences = [abs(original_distances[key] - new_distances[key]) for key in new_distances]

                # calculate the variance of the distance differences and store it
                variance_in_dd = np.var(distance_differences)
                var_matrix[idx, node] = variance_in_dd

                if make_plot:
                    plt.plot(node, variance_in_dd, color='#00ff00', marker='.')
                    if node in endpoints:
                        plt.plot(node, variance_in_dd, 'bo')

            # for the bypassed nodes store nan
            if node in to_bypass:
                var_matrix[idx, node] = np.nan

    # calculate averages
    avgs = list()
    for x in range(var_matrix.shape[1]):
        to_avg = var_matrix[:, x]
        avg = np.nanmean(to_avg)
        avgs.append(avg)

    final_threshold = np.mean(np.asarray(avgs))*threshold
    candidates = [x for x in range(len(avgs)) if avgs[x] > final_threshold]

    # remove candidates with no neighbors among the other candidates
    inducted_graph = connectivity_graph_with_wormhole.subgraph(candidates)
    for cc in nx.connected_components(inducted_graph):
        if len(cc) == 1:
            candidates.remove(list(cc)[0])

    if make_plot:
        plt.subplot(len(root_nodes)+1, 1, len(root_nodes)+1)
        for node in range(0, len(connectivity_graph_with_wormhole)):
            plt.plot(node, avgs[node], color='#00ff00', marker='.')
            if node in candidates:
                plt.plot(node, avgs[node], color='#ff99ff', marker='D', markersize=6)
            if node in endpoints:
                plt.plot(node, avgs[node], color='b', marker='o', markersize=3)
        plt.plot([0, len(connectivity_graph_with_wormhole)], [final_threshold, final_threshold], 'r-')

    return candidates


def main(args):
    print('Generating sensor network...')

    if args.deployment_type == 'random':
        if args.communication_model == 'UDG':
            positions, connectivity_graph = network_deployment.generate_square(args.num_nodes, args.comm_radius,
                                                                               args.side_len)
        else:
            positions, connectivity_graph = network_deployment.generate_square_quasi(args.num_nodes, args.comm_radius,
                                                                                     args.side_len)
    else:
        if args.communication_model == 'UDG':
            positions, connectivity_graph = network_deployment.generate_grid(args.num_nodes, args.comm_radius,
                                                                             args.side_len)
        else:
            positions, connectivity_graph = network_deployment.generate_grid_quasi(args.num_nodes, args.comm_radius,
                                                                                   args.side_len)

    print('Done.')

    degrees = nx.degree(connectivity_graph)
    avg_degree = np.mean([deg for node, deg in degrees])
    max_degree = np.max([deg for node, deg in degrees])

    print('\navg. # of neighbours', avg_degree)
    print('max. # of neighbours', max_degree)

    print('\nWormhole type: {}, wormhole radius: {}, endpoints min. distance: {}'.
          format(args.wormhole_type, args.wormhole_radius, args.wormhole_min_dist))
    print('Inserting wormhole...')
    connectivity_graph_with_wormhole, endpoints1, endpoints2 = insert_wormhole.insert_wormhole(positions,
                                                                                               connectivity_graph,
                                                                                               args.wormhole_radius,
                                                                                               args.wormhole_type,
                                                                                               args.wormhole_min_dist)
    print('Done.')

    print('\nMinimum root point distance(k): {}'.format(args.k))
    print('Choosing root points...')
    root_nodes = get_root_nodes(connectivity_graph_with_wormhole, args.k)
    print('Done.')

    print('\nLambda threshold for detection(lambda): {}'.format(args.th))
    print('Make plot: {}'.format(args.make_plot))

    print('\nRunning wormhole detection...')
    candidates = find_wormhole(connectivity_graph_with_wormhole, root_nodes,
                               args.th, endpoints1+endpoints2, args.make_plot)
    print('Done.')

    # calculate confusion matrix
    positives = set(candidates)
    negatives = set([x for x in range(0, len(connectivity_graph_with_wormhole))]).difference(positives)
    wormholes = set(endpoints1+endpoints2)
    not_wormholes = set([x for x in range(0, len(connectivity_graph_with_wormhole))]).difference(wormholes)

    # positives in WHs
    true_positives = positives.intersection(wormholes)
    # positives in notWHs
    false_positives = positives.intersection(not_wormholes)
    # negatives not in WHs
    true_negatives = negatives.intersection(not_wormholes)
    # negatives in WHs
    false_negatives = negatives.intersection(wormholes)

    print('\nConfusion matrix:')
    print(len(true_positives), len(false_negatives))
    print(len(false_positives), len(true_negatives))

    if args.make_plot:
        plt.figure(2)
        nx.draw_networkx(
            connectivity_graph, pos=positions, node_size=10, linewidths=0, edge_color='#00ff00', with_labels=False
        )

        for node in candidates:  # candidates = pink
            plt.plot(positions[node][0], positions[node][1], color='#ff99ff', marker='D', markersize=4)

        for endpoint in (endpoints1 + endpoints2):  # wormhole node = blue
            plt.plot(positions[endpoint][0], positions[endpoint][1],  color='b', marker='o', markersize=2)

        for idx, root_node in enumerate(root_nodes):  # root nodes = black
            plt.plot(positions[root_node][0], positions[root_node][1], marker='D', color='k', markersize=4)
            plt.annotate(str(idx), xy=(positions[root_node][0], positions[root_node][1]))

        plt.show()


def get_arguments():
    """Collect command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # args for generating the network
    parser.add_argument('--deployment_type', help='Deployment model. Possible choices: "grid" and "random"',
                        choices=['grid', 'random'], default='random')
    parser.add_argument('--communication_model',
                        help='Communication model. Possible choices: '
                             'unit-disk-graph->"UDG" and quasi-unit-disk-graph->"QUDG"',
                        choices=['UDG', 'QUDG'], default='UDG')
    parser.add_argument('--num_nodes', help='The number of sensors in the network.', type=int, default=900)
    parser.add_argument('--comm_radius', help='The communication radius of the sensors.', type=float, default=0.75)
    parser.add_argument('--side_len', help='The size of the observed area.', type=int, default=10)

    # args for wormhole insertion
    parser.add_argument('--wormhole_type', help='Wormhole_type. See insert_wormhole.py for examples.', type=int,
                        choices=[1, 2, 3], default=3)
    parser.add_argument('--wormhole_radius', help='Radius of the wormholes nodes.', type=float, default=0.5)
    parser.add_argument('--wormhole_min_dist', help='Minimum hop-distance between the wormholes endpoints.', type=int,
                        default=6)

    # args for the detection algorithm
    parser.add_argument('--k', help='Minimum hop-distance between the root nodes of the detection algorithm.', type=int,
                        default=7)
    parser.add_argument('--th', help='Classification threshold of the detection alg.', type=float, default=5)

    # args for visualization
    parser.add_argument('--make_plot', help='True/False for visualization.', type=bool, default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    print('Network type: {}, {}.'.format(args.deployment_type, args.communication_model))
    print('Number of sensors: {}, communication radius: {}, area of the observed region: {}.'.format(
        args.num_nodes, args.comm_radius, args.side_len**2))

    main(args)
