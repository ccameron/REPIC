#!/usr/bin/env python3
#
# iter_pick.py
# author: Christopher JF Cameron
#
"""Python wrapper script for repic/iterative_particle_picking/run.sh"""
import pathlib
from repic.utils.common import *

name = "iter_pick"
"""str: module name (used by argparse subparser)"""


def add_arguments(parser):
    """
    Adds argparse command line arguments for iter_pick.py

    Args:
        parser (object): argparse parse_args() object

    Returns:
        None
    """
    parser.add_argument("config_file", type=str,
                        help="path to REPIC config file")
    parser.add_argument("num_iter", type=int,
                        help="number of iterations (int)")
    parser.add_argument("train_size", type=int,
                        help="training subset size (int)")
    parser.add_argument("--semi_auto", action="store_true",
                        help="initialize training labels with known particles (semi-automatic)")
    parser.add_argument("--score", action="store_true",
                        help="evaluate picked particle sets")
    parser.add_argument("--out_file_path", type=str,
                        help="path for picking log file (default:<data_dir>/iter_pick.log)")


def main(args):
    """
    Loads config file (output of iter_config.py) and runs repic/iterative_particle_picking/run.sh with loaded parameters

    Args:
        args (obj): argparse command line argument object
    """
    #   load JSON config file
    with open(args.config_file, 'rt') as f:
        params_dict = json.load(f)

    #   determine full file path for iterative ensemble particle picking  'run.sh' Bash script
    script_dir = pathlib.Path(__file__).parent.resolve()
    file_path = os.path.join(
        script_dir, "..", "iterative_particle_picking", "run.sh")

    #   build subprocess Bash command
    cmd = ['bash', file_path]
    #   add command line arguments
    cmd += [
        params_dict["data_dir"],
        str(args.num_iter),
        str(params_dict["box_size"]),
        str(params_dict["exp_particles"]),
        ''.join(["train_", str(args.train_size)]),
        "semi" if args.semi_auto else "auto",
        '1' if args.score else '0',
        params_dict["cryolo_env"],
        params_dict["cryolo_model"],
        params_dict["deep_env"],
        params_dict["deep_dir"],
        params_dict["topaz_env"],
        str(params_dict["topaz_scale"]),
        str(params_dict["topaz_rad"]),
    ]

    #   run iterative ensemble particle picking
    out_file = os.path.join(
        params_dict["data_dir"], "iter_pick.log") if args.out_file_path is None else args.out_file_path
    print(f"""Note - stderr and stdout are being written to: {out_file}
Please review this file for iterative ensemble particle picking progress""")
    with open(out_file, 'wt') as o:
        subprocess.run(cmd, text=True, stderr=subprocess.STDOUT, stdout=o)
    del params_dict, script_dir, file_path, cmd, out_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ obj: argparse parse_args() object"""
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
