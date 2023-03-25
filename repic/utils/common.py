#!/usr/bin/env python3
#
#	common.py - contains common Python functions and libraries shared across scripts
#	author: Christopher JF Cameron
#

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


def adjust_plot_attributes(ax, xlabel, ylabel, fontsize=32):
    """adjust common matplotlib plot parameters"""
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


def check_float(val):
    """return True if string can be converted to a float"""
    try:
        float(val)
    except ValueError:
        return False

    return True


def create_dir(dir_path):
    """creates directory at provided location"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    return dir_path


def del_dir(dir_path):
    """deletes directory if it exists"""
    directory = Path(dir_path)
    if directory.exists() and directory.is_dir():
        rmtree(directory)
    del directory

    return dir_path


def get_box_coords(pattern, size=None, return_weights=False):
    """parsed particle coordinates file in BOX format and returns coordinates"""
    #	BOX format description: https://blake.bcm.edu/emanwiki/Eman2OtherFiles
    global box_id
    # try:
    for i, (in_file) in enumerate(glob.glob(pattern)):
        with open(in_file, 'rt') as f:
            #	check for header
            if check_float(f.readline().rstrip().split()[0]):
                f.seek(0)
            X, Y, H, W, weights = zip(*[val.strip().split() for val in f])
    assert(i == 0), ' '.join(["Error - multiple BOX files found using pattern:",
                             pattern])
    # except UnboundLocalError:
    #     print("Error - no BOX files found at:", pattern)
    #     sys.exit(-2)
    #	check_float() handles header if present
    X = [float(x) for x in X if check_float(x)]
    Y = [float(y) for y in Y if check_float(y)]
    weights = [float(val) for val in weights]
    #	check that weights are probabilities (clique weights will be > 0)
    if np.min(weights) < 0:
        #	convert log-likelihood to probability
        weights = [1. / (1. + np.exp(-1. * val)) for val in weights]

    assert(len(X) == len(Y)), "Error - unequal number of 'x' and 'y' elements"

    if not size is None:
        X = X[:size]
        Y = Y[:size]
        try:
            weights = weights[:size]
        except UnboundLocalError:
            pass  # weights not defined

    #	add unique box ID - required for optimal network X clique finding
    if return_weights:
        coords = [(x, y, weight, i) for i, (x, y, weight)
                  in enumerate(zip(X, Y, weights), box_id)]
    else:
        coords = [(x, y, i) for i, (x, y) in enumerate(zip(X, Y), box_id)]
    box_id = coords[-1][-1] + 1

    return coords


def get_box_vertex_entry(coord, length, index):
    """returns vertex entry of coordinate (X,Y)"""
    entry = [None] * length
    entry[index] = coord

    return entry


def get_multi_in_coords(in_file, return_weights=False):
    """returns coordinates, labels, and weights for mult_in BOX file"""
    coords, weights = [], []
    with open(in_file, 'rt') as f:
        #	process header
        labels = f.readline().strip().split()
        n = len(labels)  # number of boxes per line
        for line in f:
            line = line.strip().split()
            coords.append([tuple([float(val) for val in line[i:i + 2] if not val == "N/A"])
                           for i in range(0, n * 2, 2)])
            weights.append(float(line[-1]))

    return coords, labels, weights


def write_pickle(data, out_file):
    """writes data to storage in Pickle format"""
    with open(out_file, 'wb') as o:
        pickle.dump(data, o, protocol=pickle.HIGHEST_PROTOCOL)
