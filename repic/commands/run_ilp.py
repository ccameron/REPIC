#!/usr/bin/env python3
#
# run_ilp.py
# author: Christopher JF Cameron
#
"""
Apply integer linear programming (ILP) optimizer (either Gurobi or SciPy supported) to identify best subset of k-sized cliques (consensus particles) in a globally optimal manner
"""


import matplotlib as mpl

from repic.utils.common import *
from matplotlib import ticker

mpl.rcParams['axes.unicode_minus'] = False

#   determine ILP optimizer package to use
use_gurobi = False
"""bool: Gurobi integer linear programming optimizer flag"""
try:
    import gurobipy as gp
    from gurobipy import GRB
    use_gurobi = True
    """bool: Gurobi integer linear programming optimizer flag"""
except ImportError:
    from scipy.optimize import LinearConstraint, Bounds, milp


name = "run_ilp"
"""str: module name (used by argparse subparser)"""


def add_arguments(parser):
    """
    Adds argparse command line arguments for run_ilp.py

    Args:
        parser (object): argparse parse_args() object

    Returns:
        None
    """
    parser.add_argument(
        "in_dir", help="path to input directory containing get_cliques.py output")
    parser.add_argument("box_size", type=int,
                        help="particle bounding box size (in int[pixels])")
    parser.add_argument("--num_particles", type=int,
                        help="filter for the number of expected particles (int)")


def plot_particle_weights(args, weights, num_mrc, out_dir):
    """
    Creates Matplotlib line plot of the expected number of particles per micrograph vs. clique weights

    Args:
        args (obj): argparse command line argument object
        weights (list): list of consensus particle weights
        num_mrc (int): number of micrographs analyzed
        out_dir (str): dirpath to output directory

    Return:
        None
    """
    out_file = os.path.join(out_dir, "particle_dist.png")

    #   sort weights: high->low
    weights = sorted(weights, key=float, reverse=True)

    #   find expected number of particles to retain 70% of consensus particles
    total = 0
    thres = sum(weights) * 0.7
    rec_num = next((i for i, val in enumerate(weights, 1)
                   if (total := total + val) >= thres), None)
    rec_num = int((rec_num / num_mrc) + 1)  # exp num of particles per mrc
    del thres, total

    #   plot distribution of consensus particle weights
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.plot(weights, color="#BF40BF", lw=2)
    y_max, y_min = max(weights), min(weights)
    ax.fill_between(range(0, len(weights), 1), weights, y_min,
                    color="#BF40BF", ec=None, alpha=0.32)
    if not args.num_particles == None:
        ax.vlines(num_mrc * args.num_particles, ymin=y_min, ymax=y_max,
                  colors='k', lw=3, linestyles="dashed",
                  label=f"USR $num\_particles = {args.num_particles}$")
    #   add recommended exp_num parameter value
    ax.vlines(rec_num * num_mrc, ymin=y_min, ymax=y_max, colors='k',
              lw=3, linestyle="solid", label=f"REC $num\_particles = {rec_num}$")
    plt.legend(ncol=2, bbox_to_anchor=(0.5, 1.1),
               frameon=False, fontsize=24, loc="upper center")
    adjust_plot_attributes(
        ax, "number of particles per micrograph", "particle weight")
    #   adjust x-tick labels
    fig.canvas.draw()
    xtick_labels = [item.get_text() for item in ax.get_xticklabels()]
    ax.xaxis.set_major_locator(ticker.FixedLocator(ax.get_xticks()))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(
        [(int(val) // num_mrc) if int(val) > 0 else val for val in xtick_labels]))
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    del weights, out_file, fig, ax, y_max, y_min, xtick_labels


def main(args):
    """
    Applies integer linear programming optimizer to output of get_cliques.py (clique weights, constraint matrix, linear constraints, etc.) and identifies the globally optimal subset of cliques

    Args:
        args (obj): argparse command line argument object
    """
    assert (os.path.isdir(args.in_dir)), "Error - input directory is missing"

    num_mrc = 0
    weights = []
    for matrix_file in glob.glob(os.path.join(args.in_dir, "*_constraint_matrix.pickle")):

        start = time.time()
        basename = os.path.basename(
            matrix_file.replace("_constraint_matrix.pickle", ''))
        print(f"\n--- {basename} ---\n")

        # load constraint matrix and weight vector
        with open(matrix_file, 'rb') as f:
            A = pickle.load(f)
        weight_file = matrix_file.replace(
            "_constraint_matrix", "_weight_vector")
        with open(weight_file, 'rb') as f:
            w = pickle.load(f)
        del weight_file

        if use_gurobi:
            # set up Gurobi optimizer - https://www.gurobi.com/documentation/9.5/examples/mip1_py.html#subsubsection:mip1.py

            # define model object
            model = gp.Model("model")

            # set up constraint matrix
            # src: https://www.gurobi.com/documentation/9.5/refman/py_model_addmconstr.html
            x = model.addMVar(A.shape[1], vtype=GRB.BINARY)
            b = np.full(A.shape[0], 1)
            model.addMConstr(A, x, '<', b)

            # set objective function
            model.setObjective(gp.quicksum(
                [x_i * w_i for x_i, w_i in zip(x, w)]), GRB.MAXIMIZE)

            # optimize model
            model.optimize()
            x = np.array([val.x for val in model.getVars()])

            del model, b, w
        else:  # fall back on SciPy optimizer

            #   SciPY only optimizes minimization problems
            w *= -1

            #   restrict clique selection to integers
            integrality = np.ones_like(w)
            #   binary selection of cliques
            b_u = np.full(len(w), 1.5)  # '1.5' incase bounds are not inclusive
            b_l = np.full(len(w), -0.5)
            bounds = Bounds(lb=b_l, ub=b_u)

            # set up constraint matrix
            b_u = np.full(A.shape[0], 1.5)
            b_l = np.full_like(b_u, -np.inf)
            constraint = LinearConstraint(A, b_l, b_u)

            #   optimize model
            res = milp(c=w, constraints=constraint,
                       integrality=integrality, bounds=bounds,
                       options={"disp": True})
            assert (res.success ==
                    True), "Error - optimal solution could not be found"
            x = res.x

            del w, b_u, b_l, constraint, res

        # check that each vertex is only chosen once
        assert (max(np.sum(A.toarray() * x, axis=1)) ==
                1), "Error - vertices are assigned to multiple cliques"

        # load clique coordinates
        in_file = matrix_file.replace(
            "_constraint_matrix", "_consensus_coords")
        with open(in_file, 'rb') as f:
            coords = pickle.load(f)
        # load clique confidences
        in_file = matrix_file.replace(
            "_constraint_matrix", "_consensus_confidences")
        with open(in_file, 'rb') as f:
            confidences = pickle.load(f)
        del in_file, f

        multi_out = True if type(coords[0][0]) == str else False
        if multi_out:
            labels = coords[0]
            coords = coords[1:]
        # filter coords and clique weights for chosen cliques
        cliques, confidences = zip(*sorted([(coords[i], confidences[i])
                                            for i in np.where(x == 1.)[0]], key=lambda x: float(x[-1]), reverse=True))
        del coords, x

        #   write consensus particles to storage
        box_size = str(args.box_size)
        out_file = matrix_file.replace("_constraint_matrix.pickle",
                                       ".tsv" if multi_out else ".box")
        with open(out_file, 'wt') as o:
            if multi_out:
                o.write(''.join(
                    ['\t'.join(
                        ['_'.join([label, dim]) for label in labels for dim, _ in zip(['x', 'y', 'z'], cliques[0][0][:-1])] +
                        ["clique_weight"]),
                     '\n']))
                o.write('\n'.join(
                        ['\t'.join(
                            [str(int(np.rint(val))) for vals in clique for val in vals[:-1]] +
                            [str(weight)])
                         for (clique, weight) in zip(cliques, confidences)]))
            else:
                for i, (vals, weight) in enumerate(zip(cliques, confidences)):

                    if (args.num_particles == None) or (i < args.num_particles):
                        o.write(''.join([
                            '\t'.join(
                                [str(int(np.rint(val))) for val in vals[:-1]] +
                                [box_size for val in vals[:-1]] +
                                [str(weight)]
                            ),
                            '\n']))
        del out_file, basename, box_size

        #   track ILP runtime
        out_file = matrix_file.replace(
            "_constraint_matrix.pickle", "_runtime.tsv")
        with open(out_file, 'a') as o:
            o.write(f"{str(time.time() - start)}\n")    # runtime (in seconds)

        num_mrc += 1
        weights += confidences

    #   plot consensus particle weights
    print("\nPlotting consensus particle weights ... ")
    plot_particle_weights(args, weights, num_mrc, os.path.dirname(matrix_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """obj: argparse parse_args() object"""
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
