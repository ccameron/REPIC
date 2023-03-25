#!/usr/bin/env python3

import argparse
import repic
import repic.commands.get_cliques
import repic.commands.run_ilp
import repic.commands.iter_config
import repic.commands.iter_pick


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version",
                        version=f"REPIC {repic.__version__}")

    module_list = [
        repic.commands.get_cliques,
        repic.commands.run_ilp,
        repic.commands.iter_config,
        repic.commands.iter_pick
    ]

    subparser = parser.add_subparsers(
        title="commands", dest="command", required=True)
    for module in module_list:
        tmp = subparser.add_parser(module.name)
        module.add_arguments(tmp)
        tmp.set_defaults(func=module.main)
    del tmp, module_list

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
