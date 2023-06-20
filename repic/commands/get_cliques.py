#!/usr/local/bin/python3
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


def add_nodes_to_graph(graph, node_pairs, node_names):
    """
    Adds vertices and edges to the graph

    Args:
        graph (obj): NetworkX graph() object
        node_pairs (list): list of paired vertex (particle detection box) coordinates and their edge weight
        node_names (list): list of node (particle picking algorithm) names

    Returns:
        None

    """
    global node_id
    for x, y, weight_1, id_1, a, b, weight_2, id_2, jaccard in node_pairs:
        graph.add_node((x, y, id_1), name=node_names[0], weight=weight_1)
        graph.add_node((a, b, id_2), name=node_names[1], weight=weight_2)
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


def get_jaccard(set_a, set_b, box_size, threshold):
    """
    Calculates Jaccard Index (JI) between two sets of particle detection boxes and returns set of paired boxes with JI greater than the given threshold

    Args:
        set_a (set): set of particle detection boxes A
        set_b (set): set of particle detection boxes B
        box_size (int): particle detection box height/width
        threshold (float): Jaccard Index threshold

    Returns:
        list: list of paired vertex (particle detection box) coordinates and their edge weight (JI)
    """
    vals = []
    for x, y, weight_1, id_1 in set_a:
        for a, b, weight_2, id_2 in set_b:
            if ((np.abs(x - a) <= box_size) and
                    ((jaccard := calc_jaccard(x, y, a, b, box_size)) > threshold)):
                vals.append(
                    tuple([x, y, weight_1, id_1, a, b, weight_2, id_2, jaccard]))

    return vals


def main(args):
    """
    Builds NetworkX graph from multiple picked particle sets and finds all k-sized cliques

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
    for i, box_file in enumerate(glob.glob(os.path.join(args.in_dir, methods[0], "*.box"))):

        start = time.time()
        # dertemine basename of particle file
        basename = os.path.basename(box_file).replace(".box", '')
        print(f"\n--- {basename} ---\n")
        basename = f"*{basename}*"

        print("Loading particle coordinates into memory ... ")
        try:
            # get coords for each provided picker
            coords = [get_box_coords(box_file, return_weights=True)]
            for method in methods[1:]:
                coords.append(get_box_coords(os.path.join(args.in_dir, method, basename),
                                             return_weights=True))
        except (UnboundLocalError, IndexError) as e:
            # create empty BOX file if particles are not picked by all methods
            print("Skipping micrograph - not all methods have picked particles...")
            out_file = os.path.join(args.out_dir, ''.join(
                [basename[1:-1], ".box"]))
            with open(out_file, 'wt') as o:
                pass
            continue

        print("Calculating Jaccard indices ... ")
        # calculate Jaccard indices between pairs
        label_pairs, jaccards = [], []
        for (j, k) in itertools.combinations(list(range(len(coords))), 2):
            label_pairs.append((methods[j], methods[k]))
            jaccards.append(get_jaccard(
                coords[j], coords[k], args.box_size, 0.3))

        print("Building graph ... ")
        # build graph weighted pairs
        graph = nx.Graph()
        [add_nodes_to_graph(graph, vals, methods) for vals in jaccards]

        # list connected component stats
        components = [len(val) for val in nx.connected_components(graph)]
        print("\tNumber of CCs:", len(components))
        print("\tlargest CC length:", np.max(components))
        print("\tmean CC length:", np.mean(components))

        if args.get_cc:
            # replace graph of all particle detections with largest CC
            for cc in sorted(nx.connected_components(graph), key=len, reverse=True):
                # if len(cc) == 12:
                graph = graph.subgraph(cc)
                break

        print("Finding cliques ... ")
        # find cliques
        clique_size = len(coords)
        all_cliques = find_cliques(graph, clique_size)
        n = len(all_cliques)
        # sorted list of vertices in cliques
        v = sorted(set(sum(all_cliques, ())))
        print('\t', n, "cliques found with", len(v), "unique vertices")

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
            w[j] = confidence[j] * \
                np.median(list(nx.get_edge_attributes(
                    subgraph, "weight").values()))
            # retain row / col indices for sparse matrix
            cols.extend([j] * clique_size)
            rows.extend([v.index(val) for val in clique])
        assert (len(cliques) == len(
            w)), "Error - concensus coordinates and ILP weight vector are not equal lengths"
        assert (len(w) == len(confidence)
                ), "Error - cliques weights and confidences are not equal lengths"
        assert (len(cols) == len(
            rows)), "Error - ILP sparse matrix indices (rows / cols) are not equal lengths"
        assert (len(cliques) * clique_size == len(cols)
                ), "Error - consensus coordinates or ILP sparse matrix indices (rows / cols) missing"
        A = coo_matrix(([1] * len(cols), (rows, cols)), shape=(len(v), n))
        del n, v, rows, cols, j, clique, subgraph

        # write structures to storage for ILP optimization
        if args.multi_out:
            # add header
            cliques = [methods] + cliques
            if not args.get_cc:
                clique_set = set([val for clique in cliques for val in clique])
                for j in range(0, clique_size, 1):
                    cliques.extend([get_box_vertex_entry(val, clique_size, j)
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
