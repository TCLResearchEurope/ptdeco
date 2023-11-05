#!/usr/bin/env python3

import argparse
import pathlib
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--version-file", required=True)
    parser.add_argument("--version-segment", type=int, required=True)
    return parser.parse_args(argv)


def main(args):
    lines = []

    with open(args.version_file, "rt") as f:
        for l in f:
            if l.strip().startswith("__version__ ="):
                line_segments = l.split("=")
                assert len(line_segments) == 2
                version_segments = line_segments[1].strip()[1:-1].split(".")
                version_new = int(version_segments[args.version_segment]) + 1
                version_segments[args.version_segment] = str(version_new)
                l_new = line_segments[0] + '= "' + ".".join(version_segments) + '"\n'
                lines.append(l_new)
            else:
                lines.append(l)

    with open(args.version_file, "wt") as f:
        for l in lines:
            f.write(l)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
