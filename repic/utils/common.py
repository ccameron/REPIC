#!/usr/bin/env python3
#
# common.py
# author: Christopher JF Cameron
#
"""Common functions and libraries shared across REPIC scripts"""

from shutil import rmtree
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import sys
import pickle
import numpy as np
import argparse
import glob
import os
import json
import subprocess
import matplotlib.pyplot as plt
plt.switch_backend('agg')


box_id = 0
"""int: NetworkX initial vertex (particle bounding box) ID"""


def adjust_plot_attributes(ax, xlabel, ylabel, fontsize=32):
    """
    Adjusts general Matplotlib plot parameters

    Args:
        ax (obj): Matplotlib axis object
        xlabel (str): x-axis label
        ylabel (str): y-axis label

    Keyword Args:
        fontsize (int, default=32): font size of plot text

    Returns:
        None
    """
    for axis in ["bottom", "left"]:

        ax.spines[axis].set_visible(True)
        ax.spines[axis].set_color('k')
        ax.spines[axis].set_linewidth(1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(axis='x', which="both", bottom=True, top=False, direction="out",
                   length=4, width=1.0, color='k', labelsize=fontsize)
    ax.tick_params(axis='y', which="both", left=True, right=False, direction="out",
                   length=4, width=1.0, color='k', labelsize=fontsize)
    ax.grid(color="gray", ls=':', lw=0.5, zorder=-1.)


def is_float(val):
    """
    Checks if Python object can be converted to float datatype

    Args:
        val (obj): Python object

    Returns:
        bool: True if string can be converted (False otherwise)
    """
    try:
        float(val)
    except ValueError:
        return False

    return True


def create_dir(dir_path):
    """
    Creates directory at provided location

    Args:
        dir_path (str): file path to directory

    Returns:
        str: file path to directory
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    return dir_path


def del_dir(dir_path):
    """
    Deletes directory if it exists

    Args:
        dir_path (str): file path to directory

    Returns:
        str: file path to directory
    """
    directory = Path(dir_path)
    if directory.exists() and directory.is_dir():
        rmtree(directory)
    del directory

    return dir_path


def get_box_coords(pattern, key=1., size=None, return_weights=False, get_type=False):
    """
    Parses particle bounding box file and returns particle bounding box coordinates

    Args:
        pattern (str): filename RegEx pattern

    Keyword Args:
        key (float, default=1.): method key for k-d tree building
        size (int or None): restrict the number of coordinates returned to this value
        return_weight (bool, default=False): flag to include particle bounding box score or confidence in return
        get_type (bool, default=False): flag to return data type based on coordinates (cryo-EM or cryo-ET)

    Returns:
        list: list of particle bounding box coordinates
    """
    # BOX format description: https://blake.bcm.edu/emanwiki/Eman2OtherFiles
    global box_id

    data_type = "cryoem"
    for i, (in_file) in enumerate(glob.glob(pattern)):
        with open(in_file, 'rt') as f:

            # check for header
            if is_float(f.readline().rstrip().split()[0]):
                f.seek(0)
            try:    #   cryo-EM
                X, Y, H, W, weights = zip(*[val.strip().split() for val in f])
                Z, D = None, None
            except ValueError:  #   cryo-ET
                f.seek(0)
                data_type = "cryoet"
                X, Y, Z, H, W, D, weights = zip(*[val.strip().split() for val in f])
    assert (i == 0), ' '.join(["Error - multiple BOX files found using pattern:",
                               pattern])

    # is_float() handles header if present
    X = [float(x) for x in X if is_float(x)]
    Y = [float(y) for y in Y if is_float(y)]
    Z = [1.] * len(X) if Z == None else [float(z) for z in Z if is_float(z)]
    keys = [key] * len(X)
    weights = [float(val) for val in weights]

    # check that weights are probabilities (clique weights will be > 0)
    if min(weights) < 0. or max(weights) > 1.:
        # convert log-likelihood to probability
        weights = [1. / (1. + np.exp(-1. * val)) for val in weights]

    assert (len(X) == len(Y) == len(
        Z)), "Error - unequal number of 'x', 'y', and 'z' elements"

    if not size is None:
        X = X[:size]
        Y = Y[:size]
        Z = Z[:size]
        keys = keys[:size]
        try:
            weights = weights[:size]
        except UnboundLocalError:
            pass  # weights not defined

    # add unique box ID - required for optimal network X clique finding
    if return_weights:
        coords = [(x, y, z, key, weight, i) for i, (x, y, z, key, weight)
                  in enumerate(zip(X, Y, Z, keys, weights), box_id)]
    else:
        coords = [(x, y, z, key, i)
                  for i, (x, y, z) in enumerate(zip(X, Y, Z, keys), box_id)]
    box_id = coords[-1][-1] + 1

    if get_type:
        return coords,data_type
    else:
        return coords


def get_box_vertex_entry(coord, clique_size, index):
    """
    Returns particle bounding box coordinates of a clique as a vector

    Args:
        coord (list): particle bounding box coordinates
        clique_size (int): size of clique
        index (int): position of particle bounding boxes in clique (determined by order of particle picking algorithms)

    Returns:
        list: particle coordinates formatted for multi-out output

    """
    entry = [None] * clique_size
    entry[index] = coord

    return entry


def get_multi_in_coords(in_file):
    """
    Parses a particle bounding box file that contains multiple boxes per line (optimal cliques) and returns their coordinates, labels, and weights

    Args:
        in_file (str): filepath to particle bounding box file

    Returns:
        list, list, list: lists of particle bounding box coordinates, labels, and weights
    """
    coords, weights = [], []
    with open(in_file, 'rt') as f:
        # process header
        labels = f.readline().strip().split()
        n = len(labels)  # number of detections per line
        for line in f:
            line = line.strip().split()
            coords.append([tuple([float(val) for val in line[i:i + 2] if not val == "N/A"])
                           for i in range(0, n * 2, 2)])
            weights.append(float(line[-1]))

    return coords, labels, weights


def write_pickle(data, out_file):
    """
    Writes provided data to storage in Pickle format

    Args:
        data (obj): NumPy array object
        out_file (str): filepath for output file

    Returns:
        None
    """
    with open(out_file, 'wb') as o:
        pickle.dump(data, o, protocol=pickle.HIGHEST_PROTOCOL)
