#!/usr/bin/env python3
#
#   build_subsets.py
#   Author: Christopher JF Cameron
#
"""
    Creates cross-validation subsets for iterative ensemble particle picking
"""

import mrcfile
import warnings

from common import *
from bisect import bisect, bisect_right
from tqdm import tqdm

name = "build_subsets"
"""str: module name (used by argparse subparser)"""
rng = np.random.default_rng(0)
"""NumPy random generator (set to zero for reproducibility)"""


def add_arguments(parser):
    """
    Adds argparse command line arguments for build_subsets.py

    Args:
        parser (object): argparse parse_args() object

    Returns:
        None
    """
    parser.add_argument("defocus_file", type=str,
                        help="file path to RELION CTFFIND4 defocus values")
    parser.add_argument("box_dir", type=str,
                        help="file path to directory containing particle BOX files (*.box)")
    parser.add_argument("mrc_dir", type=str,
                        help="file path to directory containing micrograph MRC files (*.mrx)")
    parser.add_argument("out_dir", type=str,
                        help="file path to output directory")
    parser.add_argument("--train_set", type=str,
                        help="check if specific training subset is available after dataset splitting")
    parser.add_argument("--ignore_test", default=False, action="store_true",
                        help="only build train and val datasets (no train subsets)")


def calc_subsets(n, s=3):
    """
    Calculates subsets of examples (micrographs) for desired sampling percentages (1, 25, 50, and 100%)

    Args:
        n (int): total number of examples to sample from
        s (int): number of examples to sample each iteration (s = 3 represents the low, medium, and high defocus bins)

    Returns:
        dict: Python dictionary containing the number of examples (values) per subset (key)

    """
    subset_dict = {1: None, 25: None, 50: None, 100: None}
    tars = sorted(subset_dict.keys(), key=int)
    while s < n:

        i = bisect_right(tars, (s / n) * 100)
        subset_dict[tars[i]] = s
        s += 3
    # set 100% to full train / consensus set
    subset_dict[100] = n

    # remove subsets not found
    keys = [key for key in subset_dict.keys() if subset_dict[key] == None]
    for key in keys:
        del subset_dict[key]

    return subset_dict


def create_symlinks(args, files, label):
    """
    Creates symlinks for cross-validation files

    Args:
        args (obj): argparse command line argument object
        files (list): list of micrograph filenames to be symlinled
        label (str): name for created subdirectory that will contain linked files

    Returns:
        None

    """

    sub_dir = os.path.join(args.out_dir, label)
    del_dir(sub_dir)
    create_dir(sub_dir)
    for fname, defocus in files:

        basename = '.'.join(os.path.basename(fname).split('.')[:-1])
        # particle BOX file
        src = os.path.join(args.box_dir, '.'.join([basename, "box"]))
        if os.path.isfile(src):
            os.symlink(src, os.path.join(sub_dir, '.'.join([basename, "box"])))
        # micrograph MRC file
        os.symlink(os.path.join(args.mrc_dir, '.'.join([basename, "mrc"])),
                   os.path.join(sub_dir, '.'.join([basename, "mrc"])))
    del fname, defocus, sub_dir, basename


def plot_defocus(data, low, med, out_file):
    """
    Creates Matplotlib line plot of CTFFIND4 defocus values

    Args:
        data (list): list of paired micrograph filenames and CTFFIND4 defocus values
        low (float): low defocus bin upper threshold
        med (float): medium defocus bin upper threshold
        outfile (str): filepath of the produced line plot

    Returns:
        None
    """
    fnames, defocus = zip(*data)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    y_range, x_domain, _ = ax.hist(
        defocus, bins=32, facecolor="tab:blue", edgecolor='k')
    # bin_size = x_domain.ptp() // 32
    # add low, medium, and high bin lines
    ax.axvline(low[-1][1], color="tab:red", lw=2)
    if len(med) > 0:
        ax.axvline(med[-1][1], color="tab:red", lw=2)
    # add low, medium, and high text labels
    x = (x_domain.min() + low[-1][1]) / 2
    y = y_range.max() * 1.1
    ax.text(x, y, "Low", size=16, color='k', ha="center")
    if len(med) > 0:
        x = (low[-1][1] + med[-1][1]) / 2
        ax.text(x, y, "Medium", size=16, color='k', ha="center")
        x = (med[-1][1] + x_domain.max()) / 2
    else:
        x = (low[-1][1] + x_domain.max()) / 2
    ax.text(x, y, "High", size=16, color='k', ha="center")
    adjust_plot_attributes(ax, "Mean defocus value", "Frequency")
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    del ax, fig, data, fnames, defocus, y_range, x_domain, x, y


def sample_from_bin(bins, i):
    """
    Samples example from a random defocus bin (low, medium, and high) if the bin has items else randomly choose another bin to sample from

    Args:
        bins (list): list of defocus bins
        i (int): index of defocus bin to sample from

    Returns:
        tuple: filename (str) and CTFFIND4 defocus value (float) of sampled example
    """
    try:
        return bins[i].pop()
    except IndexError:
        # radomly sample from other bin with >1 example
        i = rng.choice([j for j, bin in enumerate(bins) if len(bin) > 0])

        return sample_from_bin(bins, i)


def main(args):
    """
    Builds training, validation, and testing subsets (cross-validation files) for machine learning algorithm training

    Args:
        args (obj): argparse command line argument object
    """
    use_defocus_values = True
    if not os.path.isfile(args.defocus_file):
        print(
            f"Error - defocus file '{args.defocus_file}' not found. Micrographs will be equally weighted")
        use_defocus_values = False
    assert (os.path.isdir(args.box_dir)
            ), f"Error - particle directory '{args.box_dir}' does not exist"
    assert (os.path.isdir(args.mrc_dir)
            ), f"Error - micrograph directory '{args.mrc_dir}' does not exist"

    # set absolute paths for all directories
    args.box_dir = os.path.abspath(args.box_dir)
    args.mrc_dir = os.path.abspath(args.mrc_dir)
    args.out_dir = os.path.abspath(args.out_dir)

    if not os.path.exists(args.out_dir):
        create_dir(args.out_dir)

    data = []
    if use_defocus_values:
        # parse defocus file
        with open(args.defocus_file, 'rt') as f:
            for line in f:
                fname, defocus_x, defocus_y = line.rstrip().split()
                defocus_x, defocus_y = (float(defocus_x), float(defocus_y))
                data.append((fname, (defocus_x + defocus_y) / 2))
        del line, f, fname, defocus_x, defocus_y
    else:
        # create list of valid MRC files with equal weights
        print(f"Checking for valid MRC files in {args.mrc_dir} ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore runtime warnings
            for file in tqdm(glob.glob(os.path.join(args.mrc_dir, '*'))):
                try:
                    with mrcfile.open(file, permissive=True) as mrc:
                        vals = mrc.data
                    if len(vals.shape) == 2:  # single-frame micrograph
                        data.append((file, 1.))
                except (AttributeError, ValueError, IsADirectoryError):
                    continue
        print(f"{len(data)} valid MRC files found")
        del file, vals

    ##
    # sort and split data by defocus value: low, medium, high
    ###

    n = len(data)
    data = sorted(data, key=lambda x: float(x[1]))
    fnames, defocus = zip(*data)
    low, med = [((defocus[-1] - defocus[0]) * val) + defocus[0]
                for val in [0.33, 0.66]]
    # build low subset
    i = bisect(defocus, low)
    low = data[:i + 1]
    # build medium and high
    j = bisect(defocus, med)
    med = data[i + 1:j + 1]
    high = data[j + 1:]
    del i, j
    assert (len(data) == (len(low) + len(med) + len(high))
            ), "Error - subset lengths do not equal original data"
    out_file = '.'.join(args.defocus_file.split('.')[:-1] + ["png"])
    plot_defocus(data, low, med, out_file)
    del out_file

    ###
    # build train, consensus, validation, and testing subsets
    ###

    # shuffle bins
    [rng.shuffle(val) for val in [low, med, high]]
    bins = [low, med, high]

    # build training set
    train = []
    curr_bin = 0
    rng.shuffle(bins)  # unbias sampling for last few examples
    if args.ignore_test:
        thres = len(data) - 6
    else:
        thres = int(np.rint(0.2 * len(data)))
    while len(train) < thres:

        train.append(sample_from_bin(bins, curr_bin))
        curr_bin = (curr_bin + 1) % 3
    subset_dict = calc_subsets(thres)
    del thres
    if args.ignore_test:
        subset_dict = {100: subset_dict[100]}

    # check training subset is available
    if not args.train_set == None:
        train_set = int(args.train_set.split('_')[-1])
        if not train_set in subset_dict.keys():
            print(
                f"Error - training subset '{args.train_set}' not available. Try a larger training subset or increase available data")
            sys.exit(-2)

    # build validation set
    val = []
    curr_bin = 0
    while len(val) < 6:

        val.append(sample_from_bin(bins, curr_bin))
        curr_bin = (curr_bin + 1) % 3

    if not args.ignore_test:
        # build test set (group together remaining examples)
        test = sum(bins, [])

        assert (len(train) + len(val) + len(test) ==
                n), "Error - examples lost while building subsets"
        del n, bins, curr_bin

    ###
    # create cross-validation files
    ###

    # setup directory hiearchy
    for key in subset_dict.keys():
        label = "train" if args.ignore_test else os.path.join(
            "train", ''.join(["train_", str(key)]))
        create_symlinks(args, train[:subset_dict[key]], label)
    create_symlinks(args, val, "val")
    del key, label, train, val
    if not args.ignore_test:
        create_symlinks(args, test, "test")
        del test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ obj: argparse parse_args() object"""
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
