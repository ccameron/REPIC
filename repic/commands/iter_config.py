#!/usr/bin/env python3
#
# iter_config.py
# author: Christopher JF Cameron
#
"""
    Creates config file (JSON format) of the general iterative ensemble particle picking parameters
"""

import copy
from repic.utils.common import *

name = "iter_config"
"""str: module name (used by argparse subparser)"""
env_dict = {
    "cryolo": "cryolo",
    "deep": "deep",
    "topaz": "topaz"
}
"""dict: dictionary of default Conda enviroment names for SPHIRE-crYOLO, DeepPicker, and Topaz"""
exp_deep_files = set([
    "analysis_pick_results.py",
    "autoPicker.py",
    "autoPick.py",
    "dataLoader.py",
    "deepModel.py",
    "display.py",
    "extractData.py",
    "Makefile",
    "README.md",
    "starReader.py",
    "train.py",
    os.path.join("trained_model", "checkpoint"),
    os.path.join("trained_model", "model_demo_type3"),
    os.path.join("trained_model", "model_demo_type3.meta")
])
"""set: set of expected DeepPicker files (check for proper installation)"""


def add_arguments(parser):
    """
    Adds argparse command line arguments for iter_config.py

    Args:
        parser (object): argparse parse_args() object

    Returns:
        None
    """
    parser.add_argument(
        "data_dir", type=str, help="path to directory containing training data")
    parser.add_argument("box_size", type=int,
                        help="particle bounding box size (in int[pixels])")
    parser.add_argument("exp_particles", type=int,
                        help="number of expected particles (int)")
    parser.add_argument(
        "cryolo_model", type=str, help="path to LOWPASS SPHIRE-crYOLO model")
    parser.add_argument("deep_dir", type=str,
                        help="path to DeepPicker scripts")
    parser.add_argument("topaz_scale", type=int,
                        help="Topaz scale value (int)")
    parser.add_argument("topaz_rad", type=int,
                        help="Topaz particle radius size (in int[pixels])")
    parser.add_argument("--cryolo_env", type=str, default=env_dict['cryolo'],
                        help=f"Conda environment name or prefix for SPHIRE-crYOLO installation (default:{env_dict['cryolo']})")
    parser.add_argument("--deep_env", type=str, default=env_dict['deep'],
                        help=f"Conda environment name or prefix for DeepPicker installation (default:{env_dict['deep']})")
    parser.add_argument(
        "--deep_model", help="path to pre-trained DeepPicker model (default:out-of-the-box model)")
    parser.add_argument("--topaz_env", type=str, default=env_dict['topaz'],
                        help=f"Conda environment name or prefix for Topaz installation (default:{env_dict['topaz']})")
    parser.add_argument(
        "--topaz_model", help="path to pre-trained Topaz model (default:out-of-the-box model)")
    parser.add_argument("--out_file_path", type=str, default="iter_config.json",
                        help="path for created config file (default:./iter_config.json)")


def main(args):
    """
    Creates config file of general iterative ensemble particle picking parameters

    Args:
        args (obj): argparse command line argument object
    """
    print("Validating config parameters")
    # check that the training data directory and SPHIRE-crYOLO model exist
    assert (os.path.exists(args.data_dir)
            ), f"Error - training data directory does not exist: {args.data_dir}"
    assert (os.path.exists(args.cryolo_model)
            ), f"Error - provided SPHIRE-crYOLO model not found: {args.cryolo_model}"

    #   check that DeepPicker path is valid
    assert (os.path.exists(args.deep_dir)
            ), f"Error - DeepPicker directory does not exist: {args.deep_dir}"
    #   check that DeepPicker folder contains expected files
    prefix = os.path.join(args.deep_dir, '')
    deep_files = set([file.replace(os.path.join(prefix), '')
                      for file in glob.glob(os.path.join(prefix, "**", '*'), recursive=True)])
    deep_files = exp_deep_files - deep_files
    assert (len(deep_files) ==
            0), f"Error - DeepPicker file(s) are missing: {', '.join(deep_files)}"
    del prefix, deep_files

    #   check that provided Conda environment names can be found
    stdout = subprocess.check_output(
        "conda env list", shell=True).decode(sys.stdout.encoding).strip()
    envs = [line.strip().split()[0] for line in stdout.split('\n')]
    envs = set([args.cryolo_env, args.deep_env, args.topaz_env]) - set(envs)
    assert (len(envs) ==
            0), f"Error - Conda environment(s) not found: {', '.join(envs)}"
    del envs, stdout

    #   check if optional models provided and are valid
    if not args.deep_model is None:
        assert (os.path.exists(os.path.dirname(args.deep_model))
                ), f"Error - provided DeepPicker model not found: {args.deep_model}"
    if not args.topaz_model is None:
        assert (os.path.exists(args.topaz_model)
                ), f"Error - provided Topaz model not found: {args.topaz_model}"

    #   write JSON file of iter_pick parameters
    print(f"Writing config file to {args.out_file_path}")
    params_dict = vars(copy.deepcopy(args))
    del params_dict["command"]
    del params_dict["out_file_path"]
    del params_dict["func"]
    with open(args.out_file_path, 'wt') as o:
        json.dump(params_dict, o, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """obj: argparse parse_args() object"""
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
