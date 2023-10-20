#!/usr/bin/env python3
#
# score_detections.py
# original author: Sebastian JH Seager
# modified by: Christopher JF Cameron
"""
    Calaculates performance metrics of particle picking algorithm by comparing output to the normative (final particle sets from EMPIAR)
"""

import argparse
import numpy as np
import os
import sys

from coord_converter import process_conversion
from pathlib import Path
from tqdm import tqdm


def get_segmentation_scores(norm_boxes, pckr_boxes, conf_thresh=None, mrc_w=None,
                            mrc_h=None, mrc_d=None):
    """
    Creates segmanetation maps of particle picking algorithm and normative particle bounding boxes, then calculates performance metrics (precision, recall, F1-score, and positive fraction of particles)

    Args:
        norm_boxes (list): list of normative particle bounding boxes
        pckr_boxes (list): list of particle picking algorithm bounding boxes

    Keyword Args:
        conf_thresh (float or None): false positive filtering threshold of particle picking algorithm
        mrc_w (int or None): micrograph/tomogram width (pixels)
        mrc_h (int or None): micrograph/tomogram height (pixels)
        mrc_d (int or None): tomogram depth (pixels)

    Returns:
        float, float, float, float: precision, recall, F1-score, and positive fraction of pixels
    """
    # if micrograph width/height not set, calculate them from provided boxes
    if mrc_w is None:
        mrc_w = round(max([n.x + n.w for n in norm_boxes + pckr_boxes]))

    if mrc_h is None:
        mrc_h = round(max([n.y + n.h for n in norm_boxes + pckr_boxes]))

    if mrc_d is None:
        try:
            mrc_d = round(max([n.z + n.d for n in norm_boxes + pckr_boxes]))
        except AttributeError: #    2D coordinate data
            mrc_d = 1

    # make binary arrays masking out GT/picker boxes
    gt_arr = np.zeros((mrc_h, mrc_w, mrc_d), dtype=np.int16).squeeze()
    pckr_arr = np.zeros((mrc_h, mrc_w, mrc_d), dtype=np.int16).squeeze()
    for b in norm_boxes:
        try:
            z, x, y, w, h, d = round(b.z), round(b.x), round(b.y), round(b.w), round(b.h), round(b.d)
            gt_arr[y: y + h, x: x + w, z: z + d] = 1
        except AttributeError:
            x, y, w, h = round(b.x), round(b.y), round(b.w), round(b.h)
            gt_arr[y: y + h, x: x + w] = 1
    for b in pckr_boxes:
        if conf_thresh is not None and b.conf < conf_thresh:
            continue
        try:
            z, x, y, w, h, d = round(b.z), round(b.x), round(b.y), round(b.w), round(b.h), round(b.d)
            pckr_arr[y: y + h, x: x + w, z: z + d] = 1
        except:
            x, y, w, h = round(b.x), round(b.y), round(b.w), round(b.h)
            pckr_arr[y: y + h, x: x + w] = 1
    del b, x, y, w, h

    num_pos = np.sum(pckr_arr)
    pos_frac = num_pos / pckr_arr.size
    tp = np.sum(gt_arr * pckr_arr)
    prec = 0.0 if (tp == num_pos == 0.0) else (tp / num_pos)
    rec = tp / np.sum(gt_arr)
    f1 = 0.0 if (prec == rec == 0.0) else ((2 * prec * rec) / (prec + rec))
    del num_pos, tp

    return prec, rec, f1, pos_frac


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(
        description="Score particles between ground truth and particle picker "
        "coordinate sets, matching files by name. All coordinate files must be "
        "in BOX file format. Use coord_converter.py to perform any necessary "
        "conversion."
    )
    """obj: argparse parse_args() object"""

    parser.add_argument(
        "-g",
        help="Ground truth particle coordinate file(s)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-p",
        help="Particle picker coordinate file(s)",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-c",
        help="Confidence threshold",
        type=float,
    )
    parser.add_argument(
        "--height", help="Micrograph/tomogram height (pixels)", type=int, default=None
    )
    parser.add_argument(
        "--width", help="Micrograph/tomogram width (pixels)", type=int, default=None
    )
    parser.add_argument(
        "--depth", help="Tomogram depth (pixels)", type=int, default=None
    )
    parser.add_argument(
        "--verbose", help="Print individual boxfile pair scores", action="store_true"
    )
    parser.add_argument(
        "--out_dir", help="file path to output directory", type=str
    )
    parser.add_argument(
        "--tomo", help="BOX files contain 3D coordinates", action="store_true"
    )

    a = parser.parse_args()
    if not a.out_dir == None and not os.path.exists(a.out_dir):
        os.makedirs(a.out_dir)
    else:
        a.out_dir = os.path.dirname(a.p[0])

    a.g = np.atleast_1d(a.g)
    a.p = np.atleast_1d(a.p)

    gt_names = [Path(f).stem.lower() for f in a.g if f.endswith(".box")]
    pckr_names = [Path(f).stem.lower() for f in a.p if f.endswith(".box")]

    # do startswith in case pickers append suffixes
    gt_matches = [g for g in gt_names if sum(
        p.startswith(g) for p in pckr_names) > 0]

    if a.verbose:
        print(f"Found {len(gt_matches)} boxfile matches\n")

    assert len(
        gt_matches) > 0, "No paired ground truth and picker particle sets found"

    all_scores = []
    for match in tqdm(gt_matches):
        gt_path = next(f for f in a.g if Path(f).stem.lower() == match)
        pckr_path = next(f for f in a.p if Path(
            f).stem.lower().startswith(match))

        # process gt and pckr box files
        fmt = "box3d" if a.tomo else "box"
        gt_dfs = process_conversion(
            [gt_path], fmt, fmt, out_dir=None, quiet=True)
        p_dfs = process_conversion(
            [pckr_path], fmt, fmt, out_dir=None, quiet=True)

        gt_df = list(gt_dfs.values())[0]
        pckr_df = list(p_dfs.values())[0]

        for df in (gt_df, pckr_df):
            if "conf" not in df.columns:
                df["conf"] = 1

        norm_boxes = list(gt_df.itertuples(name="Box", index=False))
        pckr_boxes = list(pckr_df.itertuples(name="Box", index=False))

        precision, recall, f1, pos_frac = get_segmentation_scores(
            norm_boxes, pckr_boxes, conf_thresh=a.c, mrc_w=a.width, mrc_h=a.height, mrc_d=a.depth
        )

        if a.verbose:
            tqdm.write(
                f'{match} - precision: {precision:.5f} recall: {recall:.5f} F1-score: {f1:.5f}')

        all_scores.append(tuple([match, precision, recall, f1, pos_frac]))
        del match, precision, recall, f1, pos_frac

    # write scores to file
    out_file = os.path.join(a.out_dir, "particle_set_comp.tsv")
    with open(out_file, 'wt') as o:
        o.write('\t'.join(["filename", "precision",
                "recall", "f1", "pos_frac"]) + '\n')
        for entry in all_scores:
            o.write('\t'.join([str(val) for val in entry]) + '\n')
    del out_file, o, entry, all_scores
