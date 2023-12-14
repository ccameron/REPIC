#!/usr/bin/env python3
#
# coord_converter.py
# original author: Sebastian JH Seager
# modified by: Christopher JF Cameron
"""Converts particle bounding box coordinates between different formats (STAR, BOX, dat, coord, etc.)"""

import argparse
import numpy as np
import pandas as pd
import os
import re
import sys

from common import *
from collections import namedtuple
from pathlib import Path

# globals


Box = namedtuple("Box", ["x", "y", "w", "h", "conf"])
"""named tuple for script output in BOX format"""
# set defaults starting from rightmost arg (conf)
Box.__new__.__defaults__ = (0,)
STAR_COL_X = "_rlnCoordinateX"
"""STAR file column name for x-coordinate of particle bounding box"""
STAR_COL_Y = "_rlnCoordinateY"
"""STAR file column name for y-coordinate of particle bounding box"""
STAR_COL_C = "_rlnAutopickFigureOfMerit"
"""STAR file column name for figure of merit"""
STAR_COL_N = "_rlnMicrographName"
"""STAR file column name for micrograph name"""
DF_COL_NAMES_2D = ["x", "y", "w", "h", "conf", "name"]
"""default 2D column names for Pandas data frame"""
DF_COL_NAMES_3D = ['x', 'y', 'z', 'w', 'h', 'd', "conf", "name"]
"""default 3D column names for Pandas data frame"""
STAR_HEADER_MAP = {
    "x": STAR_COL_X,
    "y": STAR_COL_Y,
    "w": None,
    "h": None,
    "conf": STAR_COL_C,
    "name": STAR_COL_N,
}
"""dictionary of header mappings for STAR file"""
BOX_HEADER_MAP = {"x": 0, "y": 1, "w": 2, "h": 3, "conf": 4, "name": None}
"""dictionary of header mappings for BOX file"""
CBOX_HEADER_MAP = {"x": 0, "y": 1, "w": 3, "h": 4, "conf": 8, "name": None}
"""dictionary of header mappings for cBOX file"""
TSV_HEADER_MAP = {"x": 0, "y": 1, "w": None,
                  "h": None, "conf": 2, "name": None}
"""dictionary of header mappings for TSV file"""
HEADER_MAP_3D = {'x': 0, 'y': 1, 'z': 2, "w": 3, "h": 4, 'd':5, "conf": 6, "name": None}
"""dictionary of header mappings for POS file"""
CS_HEADER_MAP = {
    "mrc_dims": 9,
    "x": 10,
    "y": 11,
    "w": 3,
    "h": 3,
    "conf": None,
    "name": 8,
}
"""dictionary of header mappings for CryoSparc file"""

AUTO = "auto"
"""flag for automatically processing columns of certain file types"""


# utils


def _log(msg, lvl=0, quiet=False):
    """
    Formats and prints message to console with one of the following logging levels:
    0: info (print and continue execution; ignore if quiet=True),
    1: warning (print and continue execution),
    2: error (print and exit with code 1)

    Args:
        msg (str): message text

    Keyword Args:
        lvl (int, default=0): logging level
        quiet (bool, default=False): suppress log printing

    Returns:
        None
    """

    if lvl == 0 and quiet:
        return

    prefix = ""
    if lvl == 0:
        prefix = "INFO: "
    elif lvl == 1:
        prefix = "WARN: "
    elif lvl == 2:
        prefix = "CRITICAL: "

    print(f"{prefix}{msg}")
    if lvl == 2:
        sys.exit(1)


def _is_int(val):
    """
    Checks if Python object can be converted to int datatype

    Args:
        val (obj): Python object

    Returns:
        bool: True if string can be converted (False otherwise)
    """
    try:
        int(val)
    except ValueError:
        return False

    return True


def _row_is_all_nonnumeric(x):  # CJC change
    """
    Checks if all elements of Pandas dataframe row are numeric values

    Args:
        x (obj): Pandas dataframe row object

    Returns:
        bool: True if all elements are numeric values (False otherwise)
    """
    return all([not is_float(val) for val in x.dropna()])


def _has_numbers(s):
    """
    Checks if string s contains an integer

    Args:
        s (str): string

    Returns:
        bool: True if string contains integer (False otherwise)
    """

    res = re.search("[0-9]", str(s)) is not None
    return res


def _make_parent_dir(path_str):
    """
    Creates parent directory if it does not exist

    Args:
        path_str (str): filepath to subdirectory

    Returns:
        None
    """
    par_dir = Path(path_str).parent.resolve()
    if not par_dir.is_dir():
        par_dir.mkdir(parents=True, exist_ok=True)


def _path_occupied(path_str):
    """
    Checks if file path is a file

    Args:
        path_str (str): filepath

    Returns:
        bool: True if filepath is a file (False otherwise)
    """
    return Path(path_str).resolve().is_file()


# parsing


def pos_to_df(path):
    """
    Converts 3D particle coordinate file (in *.pos format) to Pandas dataframe

    Args:
        path (str): filepath to particle bounding pos file

    Returns:
        obj: Pandas dataframe of particle bounding box coordinates
    """
    df = pd.read_csv(path,
                     delim_whitespace=True,
                     header=None,
                     skip_blank_lines=True,
                     skiprows=None)
    # rename columns based on expected convention
    df = df.rename(columns={old: new for old,
                   new in zip(df.columns, HEADER_MAP_3D.keys())})

    return df

def xml_to_df(path):
    """
    Converts DeepFinder coordinate file (*.xml) to Pandas dataframe

    Args:
        path (str): filepath to coordinate file

    Returns:
        obj: Pandas dataframe of particle detection coordinates
    """
    df = pd.read_xml(path)
    #   drop extra columns
    df.drop(columns=['class_label', 'cluster_size'], axis=1, inplace=True)

    return df


def cs_to_df(path):
    """
    Converts particle bounding box coordinate file (in CryoSparc format) into a Pandas dataframe with the correct column headers

    Args:
        path (str): filepath to particle bounding box file

    Returns:
        obj: Pandas dataframe of particle bounding box coordinates
    """
    try:
        data = np.load(path, allow_pickle=True)
    except ValueError:
        _log(f"numpy could not load {path}", lvl=2)

    try:
        ncols = len(data[0])
    except IndexError:
        _log(f"no data found in file at {path}", lvl=2)

    df = pd.DataFrame(data.tolist())
    df = df[[v for v in CS_HEADER_MAP.values() if v is not None]]
    df.columns = [k for k, v in CS_HEADER_MAP.items() if v is not None]
    df["name"] = df["name"].apply(lambda x: x.decode("utf-8"))

    # convert from [rows, cols] shape
    df["w"] = df["w"].apply(lambda x: x[1])
    df["h"] = df["h"].apply(lambda x: x[0])

    # x and y are expressed as fractions of micrograph size, so convert to pixels
    df["x"] = df["x"] * df["mrc_dims"].apply(lambda x: x[1])
    df["y"] = df["y"] * df["mrc_dims"].apply(lambda x: x[0])

    # no longer need mrc_dims column
    df = df.drop(columns=["mrc_dims"])

    return df


def star_to_df(path):
    """
    Converts particle bounding box coordinate file (in STAR file format) into a Pandas dataframe with the correct column headers

    Args:
        path (str): filepath to particle bounding box file

    Returns:
        obj: Pandas dataframe of particle bounding box coordinates
    """
    header = {}
    header_line_count = 0  # file line index where data starts

    with open(path, mode="r") as f:
        # skip any data_ block with these names
        data_blocks_to_skip = ["data_optics"]
        skip_next_loop_block = False
        for i, line in enumerate(f):
            ln = line.strip()
            if not ln:
                continue  # skip blank lines
            if ln.startswith("data_"):
                skip_next_loop_block = any(
                    s in ln for s in data_blocks_to_skip)
                continue
            if skip_next_loop_block:
                continue
            if line.startswith("_") and line.count("#") == 1:
                header_entry = "".join(ln).split("#")
                try:
                    header[int(header_entry[1]) - 1] = header_entry[0].strip()
                except ValueError:
                    _log("STAR file not properly formatted", lvl=2)
                header_line_count = i + 1  # needed if empty STAR file
            elif header and _has_numbers(line):
                header_line_count = i
                break  # we've reached coordinate data
    try:
        df = pd.read_csv(
            path,
            delim_whitespace=True,
            header=None,
            skip_blank_lines=True,
            skiprows=header_line_count,
        )
        # rename columns according to STAR header
        df = df.rename(columns={df.columns[k]: v for k, v in header.items()})
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=[v for _, v in header.items()])

    return df


def tsv_to_df(path, header_mode=None):
    """
    Converts particle bounding box coordinate file (in TSV-like file format) into a Pandas dataframe, skipping any non-numeric header rows

    Args:
        path (str): filepath to particle bounding box file

    Keyword Args:
        header_mode (int, str, or None): One of None, "infer" or an int (row index). If None, any non-numeric rows at the top of the file are skipped and column names are not set. Otherwise, manual column skipping is not performed, and header_mode is passed directly to the header argument of pandas.read_csv

    Returns:
        obj: Pandas dataframe of particle bounding box coordinates
    """

    if header_mode is None:
        header_line_count = 0  # file line index where data starts
        with open(path, mode="r") as f:
            for i, line in enumerate(f):
                if (not line.startswith("_")) and _has_numbers(line):  # CJC change
                    header_line_count = i
                    break
        try:
            df = pd.read_csv(
                path,
                delim_whitespace=True,
                header=None,
                skip_blank_lines=True,
                skiprows=header_line_count,
            )
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
    else:
        df = pd.read_csv(
            path,
            delim_whitespace=True,
            header=header_mode,
            skip_blank_lines=True,
        )

    # 	drop rows that contain only NaNs and strings (ending CBOX lines) - CJC change
    df = df[~df.apply(_row_is_all_nonnumeric, axis=1)]

    return df


# writing


def df_to_star(df, out_path, force=False):
    """
    Writes Panda dataframe of particle bounding box coordinates (generated from one of the *_to_df methods) to storage in STAR file format

    Args:
        df (object): Pandas dataframe of particle bounding box coordinates
        out_path (str): filepath to output file

    Keyword Args:
        force (bool, default=False): overwrite output file if it exists

    Returns:
        None
    """

    if force:
        _make_parent_dir(out_path)
    else:
        if _path_occupied(out_path):
            _log("re-run with the force flag to replace existing files", lvl=2)

    df_cols = list(df.columns)
    star_loop = "data_\n\nloop_\n"
    for df_col, star_col in STAR_HEADER_MAP.items():
        if star_col is None:
            continue
        try:
            idx = df_cols.index(df_col)
            star_loop += f"{star_col} #{idx + 1}\n"
        except ValueError:
            pass

    with open(out_path, "w") as f:
        f.write(star_loop)

    df.to_csv(out_path, header=False, sep="\t", index=False, mode="a")


def df_to_tsv(df, col_order, out_path, include_header=False, force=False):
    """
    Writes Panda dataframe of particle bounding box coordinates (generated from one of the *_to_df methods) to storage, optionally writing out [x, y, w, h, conf] labels as a header

    Args:
        df (object): Pandas dataframe of particle bounding box coordinates
        col_order (list): list of ordered file columns
        out_path (str): filepath to output file

    Keyword Args:
        include_header (bool, default=False): include header in output file
        force (bool, default=False): overwrite output file if it exists

    Returns:
        None
    """
    if force:
        _make_parent_dir(out_path)
    else:
        if _path_occupied(out_path):
            _log("re-run with the force flag to replace existing files", lvl=2)

    out_cols = [c for c in col_order if c in df.columns]
    df[out_cols].to_csv(out_path, header=include_header, sep="\t", index=False)


# handler method


def process_conversion(
    paths,
    in_fmt,
    out_fmt,
    boxsize=None,
    out_dir=None,
    in_cols=["auto", "auto", "auto", "auto", "auto", "auto", "auto", "auto"],
    out_col_order=['x', 'y', 'z', 'w', 'h', 'd', "conf", "name"],
    suffix="",
    include_header=False,
    single_out=False,
    multi_out=False,
    round_to=None,
    norm_conf=None,
    require_conf=None,
    force=False,
    quiet=False,
):
    """
    Converts between different particle bounding box formats

    Args:
        paths (list): filepaths of particle bounding box coordinate files
        in_fmt (str): input file format
        out_fmt (str): output file format

    Keyword Args:
        boxsize (int or None): particle bounding box height/width
        out_dir (str or None): filepath to output file
        in_cols (list): tuple of column determination (default=auto)
        out_col_order (list): output column order
        suffix (str): additional suffix for output files (default='')
        include_header (bool, default=False): include header in output file
        single_out (bool, default=False): output particle bounding box coordinates in a single file
        multi_out (bool, default=False): output particle bounding box coordinates in multiple files (one per micrograph)
        round_to (int or None): round coordinates to the specified number of decimal places
        norm_conf (list or None): list of min and max confidence values to be used to normalize observed confidence scores
        require_conf (float or None): model confidence score to assign to particle bounding boxes without a score
        force (bool, default=False): overwrite output file if it exists
        quiet (bool, default=False): suppress log printing

    Returns:
        None
    """
    convert_2D = False if in_fmt in ["pos", "box3d", "xml"] else True

    # set default columns as needed
    cols = {}
    if convert_2D:
        #   delete z dim parameter
        del in_cols[2]
        del in_cols[4]
        del out_col_order[2]
        del out_col_order[4]
    for i, col in enumerate(DF_COL_NAMES_2D if convert_2D else DF_COL_NAMES_3D):
        cols[col] = in_cols[i] if in_cols[i] != "none" else None

    # read input files into dataframes
    dfs = {}
    try:
        if in_fmt == "star":
            default_cols = STAR_HEADER_MAP
            dfs = {p: star_to_df(p) for p in paths}
        elif in_fmt == "cs":
            default_cols = CS_HEADER_MAP
            dfs = {p: cs_to_df(p) for p in paths}
        elif in_fmt == "box":
            default_cols = BOX_HEADER_MAP
            dfs = {p: tsv_to_df(p) for p in paths}
        elif in_fmt == "cbox":  # CJC change
            default_cols = CBOX_HEADER_MAP
            dfs = {p: tsv_to_df(p).apply(pd.to_numeric) for p in paths}
        elif in_fmt == "tsv":
            default_cols = TSV_HEADER_MAP
            dfs = {p: tsv_to_df(p) for p in paths}
        elif in_fmt in ["pos", "box3d"]:
            default_cols = HEADER_MAP_3D
            dfs = {p: pos_to_df(p) for p in paths}
        elif in_fmt == "xml":
            default_cols = HEADER_MAP_3D
            dfs = {p: xml_to_df(p) for p in paths}
        else:
            _log("unknown format", lvl=2)
    except pd.errors.ParserError as e:
        _log(f"input '{in_fmt}' file not properly formatted")
        _log(f"{repr(e)}", lvl=2)

    # apply any default cols needed
    for k, v in default_cols.items():
        if k in cols:
            cols[k] = v if cols[k] == AUTO else cols[k]

    _log(f"using the following input column mapping:", quiet=quiet)
    _log(f"{cols}", 0, quiet=quiet)

    out_dfs = {}
    for name, df in dfs.items():
        # rename columns to make conversion logic easier
        rename_dict = {}
        for new_name, cur_name in cols.items():
            if cur_name is None:
                continue
            if _is_int(cur_name):
                cur_name = int(cur_name)
                if cur_name in range(len(df.columns)):
                    rename_dict[df.columns[cur_name]] = new_name
            else:
                if cur_name in df.columns:
                    rename_dict[cur_name] = new_name
        df = df.rename(columns=rename_dict)

        colum_names = ('x', 'y', 'w', 'h') if convert_2D else ('x', 'y', 'z', 'w', 'h')
        try:
            # shift coordinates from center to corner if needed
            if in_fmt in ("star", "tsv", "cs", "pos", "box3d") and out_fmt in ("box"):
                assert (
                    boxsize is not None), f"Expected integer boxsize but got {boxsize}"
                df['w'] = boxsize
                df['h'] = boxsize
                if not convert_2D:
                    df['d'] = boxsize
                for cl in colum_names:
                    df[cl] = df[cl].astype(float)
                df['x'] = df['x'] - df['w'].div(2)
                df['y'] = df['y'] - df['h'].div(2)
                if not convert_2D:
                    df['z'] = df['z'] - df['d'].div(2)
            # shift coordinates from corner to center if needed
            elif in_fmt in ("box") and out_fmt in ("star", "tsv"):
                for c in colum_names:
                    df[cl] = df[cl].astype(float)
                df['x'] = df['x'] + df['w'].div(2)
                df['y'] = df['y'] + df['h'].div(2)
            # no shift but add box size
            elif in_fmt in ("xml") and out_fmt in ("box"):
                assert (
                    boxsize is not None), f"Expected integer boxsize but got {boxsize}"
                df['w'] = boxsize
                df['h'] = boxsize
                df['d'] = boxsize

            if round_to is not None:
                for cl in colum_names:
                    if cl not in df.columns:
                        continue
                    df[cl] = df[cl].round(round_to)
                    if round_to == 0:
                        df[cl] = df[cl].round(round_to).astype(int)

        except KeyError as e:
            _log(
                f"didn't find column {e} in input columns ({list(df.columns)})", lvl=2)
        except TypeError as e:
            _log(f"unexpected type in input columns ({e})", lvl=2)
        except ValueError as e:
            _log(f"unexpected value in input columns ({e})", lvl=2)
        del colum_names

        if norm_conf is not None and "conf" in df.columns:
            old_max, old_min = df["conf"].max(), df["conf"].min()
            new_min, new_max = norm_conf
            old_range, new_range = old_max - old_min, new_max - new_min
            if old_min <= new_min or old_max > new_max:
                if old_range == 0:
                    # if the old range was 0, arbitrarily set everything to new_min
                    df["conf"] = new_min
                else:
                    # otherwise do linear normalization
                    df["conf"] = (
                        (df["conf"] - old_min) * new_range / old_range
                    ) + new_min

        if require_conf is not None and "conf" not in df.columns:
            df["conf"] = float(require_conf)

        out_cols = ['x', 'y', 'z', 'w', 'h', 'd', "conf", "name"]
        if convert_2D:
            del out_cols[2]     # ["x", "y", "w", "h", "conf", "name"]
            del out_cols[4]
        if out_fmt in ("star", "tsv"):
            del out_cols[2:6]   # ["x", "y", "conf", "name"]

        out_dfs[name] = df[[x for x in out_cols if x in df.columns]]

    if single_out:
        out_dfs = {"all": pd.concat(out_dfs, ignore_index=True)}

    if multi_out:
        if all("name" in df.columns for df in out_dfs.values()):
            grouped_by_mrc = pd.concat(
                out_dfs, ignore_index=True).groupby("name")
            out_dfs = {k: df.drop("name", axis=1) for k, df in grouped_by_mrc}
        else:
            _log("cannot fulfill multi_out without micrograph name information", lvl=1)

    if out_dir is None:
        return out_dfs

    os.chdir(out_dir)

    for name, df in out_dfs.items():
        full_path = Path(name).resolve()
        parent = full_path.parents[0]
        if full_path in [p.resolve() for p in paths]:
            # if this path is found in input exactly, replace parents with out_dir
            # since there are no other subdirectories to worry about
            parent = out_dir
        else:
            # this either means `name` was a relative path (in which case it's now
            # already in out_dir because it was resolved after os.chdir) or a different
            # absolute path; in either case, we must ensure the output goes to out_dir
            if os.path.commonpath([parent, out_dir]) != out_dir:
                parent = out_dir / parent

        parent.mkdir(parents=True, exist_ok=True)
        out_file = f"{full_path.stem}{suffix}.{out_fmt}"
        out_path = parent / out_file

        if out_fmt == "star":
            df_to_star(df, out_path, force=force)
        elif out_fmt in ("box", "tsv"):
            _log(f"using the following output column order:", quiet=quiet)
            _log(f"{out_col_order}", quiet=quiet)
            df_to_tsv(
                df,
                out_col_order,
                out_path,
                include_header=include_header,
                force=force,
            )

        _log(f"wrote to {out_path}", quiet=quiet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts particle coordinate file data between "
        "several different formats. The -f (input format) and -t (output format) "
        "parameters define the conversion. The -c argument can be used if more "
        "granular control over column indices is required."
    )
    """obj: argparse parse_args() object"""
    parser.add_argument(
        "input", help="Path(s) to input particle coordinates", nargs="+"
    )
    parser.add_argument(
        "out_dir",
        help="Output directory in which to store generated coordinate files (will be "
        "created if it does not exist)",
    )
    parser.add_argument(
        "-f",
        choices=["star", "box", "cbox", "tsv", "cs", "pos", "box3d", "xml"],
        help="Format FROM which to convert the input",
    )
    parser.add_argument(
        "-t",
        choices=["star", "box", "tsv"],
        default="box",
        help="Format TO which to convert the input. Only 'box' supports 3D coordinates.",
    )
    parser.add_argument(
        "-b",
        type=int,
        help="Specifies or overrides the box size to be used "
        "(required if input does not include a box size)",
    )
    parser.add_argument(
        "-c",
        nargs=8,
        metavar=("X_COL", "Y_COL", "Z_COL", "W_COL",
                 "H_COL", "D_COL", "CONF_COL", "NAME_COL"),
        default=["auto", "auto", "auto", "auto",
                 "auto", "auto", "auto", "auto"],
        help="""Manually specify input column names (STAR) or zero-based indices
        (BOX/TSV). This can be useful if input file does not follow default column
        indices of the specified input (-f) format. Expects seven positional arguments,
        corresponding to: [x, y, z, h, w, d, g, conf, mrc_name]. Set a column to 'none' to
        exclude it from conversion and 'auto' to keep its default value. Z_COL AND D_COL
        are dropped for 2D coordinate conversion. """,
    )
    parser.add_argument(
        "-d",
        nargs=8,
        default=['x', 'y', 'z', 'w', 'h', 'd', "conf", "name"],
        help="""Manually specify the order of columns in output files (only applies to
        BOX/TSV output formats). Expects six positional arguments, which should be
        some ordering of the strings ['x', 'y', 'z', 'w', 'h', 'd', 'conf', 'name'].
        Use any other string (like 'none') at any of the six positions to exclude the
        missing column from the output. 'z' and 'd' are dropped for 2D conversion. """,
    )
    parser.add_argument(
        "-s",
        default="",
        type=str,
        help="Suffix to append to generated output (default: no suffix)",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="If output format is BOX or TSV, include column headers (has no effect "
        "with STAR output)",
    )
    parser.add_argument(
        "--single_out",
        action="store_true",
        help="If possible, make output a single file, with column for micrograph name",
    )
    parser.add_argument(
        "--multi_out",
        action="store_true",
        help="If possible, split output into multiple files by micrograph name",
    )
    parser.add_argument(
        "--round",
        default=None,
        type=int,
        help="Round coordinates to the specified number of decimal places. "
        "Don't round by default.",
    )
    parser.add_argument(
        "--require_conf",
        default=None,
        type=float,
        help="Supply a decimal confidence value to fill any missing confidences in "
        "the output. Don't fill in missing confidences by default.",
    )
    parser.add_argument(
        "--norm_conf",
        default=None,
        type=float,
        nargs=2,
        help="If confidence values exceed the provided range, normalize them to that "
        "range (e.g. [0, 1]). No normalization is done by default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow files in output directory to be overwritten and make output "
        "directory if it does not exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence info-level output",
    )

    a = parser.parse_args()

    # validation
    if a.f in ("star", "tsv", "xml") and a.t != "star" and a.b is None:
        _log(f"box size required for '{a.f}' input", lvl=2)
    if a.single_out and a.multi_out:
        _log(f"cannot fulfill both single_out and multi_out flags", lvl=2)

    a.input = [Path(p).resolve() for p in np.atleast_1d(a.input)]
    if not all(p.is_file() for p in a.input):
        _log(f"bad input paths", lvl=2)

    a.out_dir = Path(a.out_dir).resolve()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    process_conversion(
        paths=a.input,
        in_fmt=a.f,
        out_fmt=a.t,
        boxsize=a.b,
        out_dir=a.out_dir,
        in_cols=a.c,
        out_col_order=a.d,
        suffix=a.s,
        include_header=a.header,
        single_out=a.single_out,
        multi_out=a.multi_out,
        round_to=a.round,
        norm_conf=a.norm_conf,
        require_conf=a.require_conf,
        force=a.force,
        quiet=a.quiet,
    )

    _log("done.", quiet=a.quiet)
