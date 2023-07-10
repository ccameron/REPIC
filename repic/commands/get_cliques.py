#!/usr/bin/env python3
#
# get_cliques.py
# author: Christopher JF Cameron
#
"""
    Finds cliques (potential consensus particles) of size k in each graph (micrograph)
"""

import itertools
import networkx as nx

from repic.utils.common import *
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree

name = "get_cliques"
"""str: module name (used by argparse subparser)"""


def add_arguments(parser):
    """
    Adds argparse command line arguments for get_cliques.py

    Args:
        parser (object): argparse parse_args() object

    Returns:
        None
    """
    parser.add_argument("in_dir",
                        help="path to input directory containing subdirectories of particle coordinate files ")
    parser.add_argument("out_dir",
                        help="path to output directory (WARNING - script will delete directory if it exists)")
    parser.add_argument("box_size", type=int,
                        help="particle detection box size (in int[pixels])")
    parser.add_argument("--multi_out", action="store_true",
                        help="set output of cliques to be members sorted by picker name")
    parser.add_argument("--get_cc", action="store_true",
                        help="filters cliques for those in the largest Connected Component (CC)")


def add_nodes_to_graph(graph, node_pairs, node_names, k=3):
    """
    Adds vertices and edges to the graph

    Args:
        graph (obj): NetworkX graph() object
        node_pairs (list): list of paired vertex (particle detection box) coordinates and their edge weight
        node_names (list): list of node (particle picking algorithm) names

    Keyword Args:
        k (int, default=3):  number of methods

    Returns:
        None
    """
    global node_id
    for x, y, key_1, weight_1, id_1, a, b, key_2, weight_2, id_2, jaccard in node_pairs:
        graph.add_node((x, y, id_1),
                       name=node_names[int(np.rint(key_1 * k))],
                       weight=weight_1)
        graph.add_node((a, b, id_2),
                       name=node_names[int(np.rint(key_2 * k))],
                       weight=weight_2)
        graph.add_edge((x, y, id_1), (a, b, id_2),
                       weight=jaccard)  # weight attribute used by nx


def calc_jaccard(x, y, a, b, box_size):
    """
    Calculates Jaccard Index (similarity) for particle detection boxes A (x,y) and B (a,b) with given box size

    Args:
        x (int): x-coodinate of particle dection box A
        y (int): y-coordinate of particle detection box A
        a (int): x-coordinate of particle detection box B
        b (int): y-coordinate of particle detection box B
        box_size (int): particle detection box height/width

    Returns:
        float: Jaccard Index of two particle detection boxes
    """
    x_overlap = np.max([(np.min([x, a]) + box_size - np.max([x, a])), 0])
    y_overlap = np.max([(np.min([y, b]) + box_size - np.max([y, b])), 0])
    jaccard = x_overlap * y_overlap

    return jaccard / ((2 * box_size ** 2) - jaccard)


def find_cliques(graph, k):
    """
    Finds all cliques in graph of size k

    Args:
        graph (obj): NetworkX graph() object
        k (int): clique size

    Returns:
        set: set of k-sized cliques in the graph
    """
    cliques = set()
    for clique in nx.find_cliques(graph):
        if len(clique) == k:
            cliques.add(tuple(sorted(clique)))

    return cliques


def main(args):
    """
    Builds NetworkX graph from file set and finds all k-sized cliques

    Args:
        args (obj): argparse command line argument object
    """
    # ensure input directory exists
    assert (os.path.exists(args.in_dir)
            ), "Error - input directory does not exist"

    # set up output directory
    del_dir(args.out_dir)
    exclude = ["box_size", "out_dir", "multi_out", "get_cc"]

    # get method subdirectories
    methods = sorted([os.path.basename(val) for val in glob.glob(os.path.join(args.in_dir, '*'))
                      if os.path.isdir(val)], key=str)
    create_dir(args.out_dir)

    # determine method with shortest naming convention
    start_method = None
    num_methods = len(methods)
    for method in methods:
        # collect example box file from each method subdirectory
        for box_file in glob.glob(os.path.join(args.in_dir, method, "*.box")):
            # identify basename of file and use it to find matching files in other subdirectories
            tmp = os.path.basename(box_file).replace(".box", '')
            tmp = f"*{tmp}*"
            n = len(sum([glob.glob(os.path.join(args.in_dir, method, tmp))
                    for method in methods], []))
            break
        # if the current method's naming convention can be used to identify pairs, keep it
        if n == num_methods:
            start_method = method
            break
    assert (not start_method ==
            None), "Error - particle file names cannot be paired across methods"
    del box_file, tmp, n

    print(f"Using {start_method} BOX files as starting point")

    # iterate over crYOLO files and parse matching DeepPicker & Topaz files
    k = len(methods)  # number of methods/clique size
    for in_file in glob.glob(os.path.join(args.in_dir, methods[0], "*.box")):

        start = time.time()
        # dertemine basename of particle file
        basename = os.path.basename(in_file).replace(".box", '')
        print(f"\n--- {basename} ---\n")
        basename = f"*{basename}*"

        print("Loading particle coordinates into memory ... ")
        try:
            # get coords for each provided picker
            # key = i/k for method i of k methods
            coords = np.array(get_box_coords(
                in_file, key=0., return_weights=True))
            for i, method in enumerate(methods[1:], 1):
                coords = np.concatenate((coords,
                                         np.asarray(get_box_coords(os.path.join(args.in_dir, method, basename),
                                                                   key=i / float(k),
                                                                   return_weights=True))))
            del i, method
        except (UnboundLocalError, IndexError) as e:
            # create empty BOX file if particles are not picked by all methods
            print("Skipping micrograph - not all methods have picked particles...")
            out_file = os.path.join(args.out_dir, ''.join(
                [basename[1:-1], ".box"]))
            with open(out_file, 'wt') as o:
                pass
            return

        print("Calculating Jaccard indices ... ")
        #   build k-d tree from x, y, and z coordinates with method key values
        kd_tree = KDTree(coords[:, :4])

        #   get pairs of particle detection boxes within distance threshold r
        #   k-d tree uses Minkowski distance (default p == 2 is Euclidean distance)
        r = args.box_size + 1.  # +1 for key column
        pairs = kd_tree.query_pairs(r)
        # calculate Jaccard indices between pairs
        data = []
        for i, j in pairs:
            x, y, z, key_1, weight_1, id_1 = coords[i]
            a, b, c, key_2, weight_2, id_2 = coords[j]
            if (not key_1 == key_2) and ((jaccard := calc_jaccard(x, y, a, b, args.box_size)) > 0.3):
                data.append(
                    tuple([x, y, key_1, weight_1, id_1, a, b, key_2, weight_2, id_2, jaccard]))
        del coords, kd_tree, x, y, z, key_1, weight_1, id_1, a, b, c, key_2, weight_2, id_2, jaccard

        print("Building graph ... ")
        # build graph weighted pairs
        graph = nx.Graph()
        # [add_nodes_to_graph(graph, vals, methods) for vals in data]
        add_nodes_to_graph(graph, data, methods, k=k)
        del data

        # list connected component stats
        components = [len(val) for val in nx.connected_components(graph)]
        print("\tNumber of CCs:", len(components))
        print("\tlargest CC length:", np.max(components))
        print("\tmean CC length:", np.mean(components))

        if args.get_cc:
            # replace graph of all particle detections with largest CC
            for cc in sorted(nx.connected_components(graph), key=len, reverse=True):
                graph = graph.subgraph(cc)
                break

        print("Finding cliques ... ")
        # find cliques
        all_cliques = find_cliques(graph, k)
        n = len(all_cliques)
        # sorted list of vertices in cliques
        v = sorted(set(sum(all_cliques, ())))
        print(f"\t{n} cliques found with {len(v)} unique vertices")

        print("Building ILP data structures ... ")
        # cliques confidences - median(clique confidences)
        confidence = np.zeros(n, dtype=np.float32)
        w = np.zeros(n, dtype=np.float32)  # weight vector of cliques
        # iterate over cliques, fill in w, retain vertex-to-clique assignments
        cliques, rows, cols = [], [], []
        for j, clique in enumerate(all_cliques):
            subgraph = graph.subgraph(clique)
            if args.multi_out:
                # return all nodes in clique sorted by picker name
                cliques.append(sorted(subgraph.nodes(),
                                      key=lambda x: subgraph.nodes[x]["name"]))
            else:
                # determine best particle identification in clique based on
                # overlap with other members
                cliques.append(max(subgraph.degree(weight="weight"),
                                   key=lambda x: x[1])[0])
            # calculate ILP weight for clique
            # median(Jaccard of members/edges) * median(members/nodes confidence)
            confidence[j] = np.median(
                list(nx.get_node_attributes(subgraph, "weight").values()))
            w[j] = confidence[j] * np.median(list(nx.get_edge_attributes(
                subgraph, "weight").values()))
            # retain row / col indices for sparse matrix
            cols.extend([j] * k)
            rows.extend([v.index(val) for val in clique])
        assert (len(cliques) == len(
            w)), "Error - concensus coordinates and ILP weight vector are not equal lengths"
        assert (len(w) == len(confidence)
                ), "Error - cliques weights and confidences are not equal lengths"
        assert (len(cols) == len(
            rows)), "Error - ILP sparse matrix indices (rows / cols) are not equal lengths"
        assert (len(cliques) * k == len(cols)
                ), "Error - consensus coordinates or ILP sparse matrix indices (rows / cols) missing"
        A = coo_matrix(([1] * len(cols), (rows, cols)), shape=(len(v), n))
        del n, v, rows, cols, j, clique, subgraph

        # write structures to storage for ILP optimization
        if args.multi_out:
            # add header
            cliques = [methods] + cliques
            if not args.get_cc:
                clique_set = set([val for clique in cliques for val in clique])
                for j in range(0, k, 1):
                    cliques.extend([get_box_vertex_entry(val, k, j)
                                    for val in set(coords[j]).difference(clique_set)])

        for label, val in zip(
                ["weight_vector", "consensus_coords",
                    "consensus_confidences", "constraint_matrix"],
                [w, cliques, confidence, A]):
            out_file = os.path.join(args.out_dir, ''.join(
                [basename[1:-1], '_', label, ".pickle"]))
            with open(out_file, 'wb') as o:
                pickle.dump(val, o, protocol=pickle.HIGHEST_PROTOCOL)

        out_file = os.path.join(args.out_dir, ''.join(
            [basename[1:-1], "_runtime.tsv"]))
        with open(out_file, 'wt') as o:
            # runtime (in seconds), largest CC, number of CC
            o.write('\t'.join([str(val) for val in [time.time() - start,
                    np.max(components), len(components)]]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ obj: argparse parse_args() object"""
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
