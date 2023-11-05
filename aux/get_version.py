#!/usr/bin/env python3

import argparse
import pathlib
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--version-file", required=True)
    return parser.parse_args(argv)


def main(args):
    with open(args.version_file, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("__version__ = "):
                version = line.split("=", 1)[-1].strip()
                version = version[1:-1]  # Remove quotation charactersi
                print(version)
                return

    raise RuntimeError("Unable to find __version__ string")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
