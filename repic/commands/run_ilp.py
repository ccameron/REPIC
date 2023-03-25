#!/usr/local/bin/python3
#
#	run_ilp.py -  run Gurobi ILP optimizer to identify best particle cliques
#	author: Christopher JF Cameron
#

import gurobipy as gp

from repic.utils.common import *
from gurobipy import GRB

name = "run_ilp"


def add_arguments(parser):
    """adds parser arguments for script"""
    parser.add_argument(
        "in_dir", help="path to input directory containing get_cliques.py output")
    parser.add_argument("box_size", type=int,
                        help="particle detection box size (in int[pixels])")
    parser.add_argument("--num_particles", type=int,
                        help="filter for the number of expected particles (int)")


def main(args):

    assert(os.path.isdir(args.in_dir)), "Error - input directory is missing"

    for matrix_file in glob.glob(os.path.join(args.in_dir, "*_constraint_matrix.pickle")):

        start = time.time()
        basename = os.path.basename(
            matrix_file.replace("_constraint_matrix.pickle", ''))
        print(f"\n--- {basename} ---\n")

        #	load constraint matrix and weight vector
        with open(matrix_file, 'rb') as f:
            A = pickle.load(f)
        weight_file = matrix_file.replace(
            "_constraint_matrix", "_weight_vector")
        with open(weight_file, 'rb') as f:
            w = pickle.load(f)
        del weight_file

        ###
        #	set up Gurobi optimizer - https://www.gurobi.com/documentation/9.5/examples/mip1_py.html#subsubsection:mip1.py
        ###

        #	define model object
        model = gp.Model("model")

        #	set up constraint matrix
        #	src: https://www.gurobi.com/documentation/9.5/refman/py_model_addmconstr.html
        x = model.addMVar(A.shape[1], vtype=GRB.BINARY)
        b = np.full(A.shape[0], 1)
        model.addMConstr(A, x, '<', b)

        #	set objective function
        model.setObjective(gp.quicksum(
            [x_i * w_i for x_i, w_i in zip(x, w)]), GRB.MAXIMIZE)

        #	optimize model
        model.optimize()

        #	check that each vertex is only chosen once
        x = np.array([val.x for val in model.getVars()])
        assert(np.max(np.sum(A.toarray() * x, axis=1)) ==
               1), "Error - vertices are assigned to multiple cliques"
        del model, b, w

        #	load clique coordinates
        in_file = matrix_file.replace(
            "_constraint_matrix", "_consensus_coords")
        with open(in_file, 'rb') as f:
            coords = pickle.load(f)
        #	load clique confidences
        in_file = matrix_file.replace(
            "_constraint_matrix", "_consensus_confidences")
        with open(in_file, 'rb') as f:
            confidences = pickle.load(f)
        del in_file, f

        multi_out = True if type(coords[0][0]) == str else False
        if multi_out:
            labels = coords[0]
            coords = coords[1:]
        #	filter coords and clique weights for chosen cliques
        cliques, confidences = zip(*[(coords[i], confidences[i])
                                   for i in np.where(x == 1.)[0]])
        del x

        tmp = len(cliques)
        if multi_out:
            #	retain vertices not found in chosen cliques
            coord_sets = [set([val for val in coord_set if val])
                          for coord_set in zip(*coords[:])]
            clique_sets = [set([val for val in clique_set if val])
                           for clique_set in zip(*cliques[:])]
            cliques, w = list(cliques), list(confidences)
            n = len(coord_sets)
            for i in range(0, n, 1):
                v = coord_sets[i].difference(clique_sets[i])
                cliques.extend([get_box_vertex_entry(val, n, i) for val in v])
                confidences.extend([0.] * len(v))
            assert(len(cliques) == len(confidences)
                   ), "Error - missing vertices and / or weights"
            del coord_sets, clique_sets, n, v
        del coords

        box_size = str(args.box_size)
        out_file = matrix_file.replace("_constraint_matrix.pickle",
                                       ".tsv" if multi_out else ".box")
        with open(out_file, 'wt') as o:
            if multi_out:
                o.write('\t'.join(labels) + '\n')
                o.write('\n'.join(['\t'.join(['\t'.join([
                        str(int(np.rint(val[0]))), str(int(np.rint(val[1])))])
                    if val else "N/A\tN/A" for val in vals] + [str(weight)])
                    for (vals, weight) in zip(cliques, confidences)]))
            else:
                for i, (val, weight) in enumerate(sorted(zip(cliques, confidences),
                                                         key=lambda x: x[1], reverse=True)):
                    if (args.num_particles == None) or (i < args.num_particles):
                        o.write('\t'.join([
                                str(int(np.rint(val[0]))),
                                str(int(np.rint(val[1]))),
                                box_size,
                                box_size,
                                str(weight)]) + '\n')
        del out_file, basename, box_size

        out_file = matrix_file.replace(
            "_constraint_matrix.pickle", "_runtime.tsv")
        with open(out_file, 'a') as o:
            #	runtime (in seconds)
            o.write(str(time.time() - start) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
