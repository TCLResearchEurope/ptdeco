import os
import pathlib
from typing import List

import setuptools  # type: ignore


def is_comment(s: str) -> bool:
    return not s.strip() or s.strip().startswith("#")


def read_requirements() -> List[str]:
    lib_folder = os.path.dirname(os.path.realpath(__file__))
    requirements_path = lib_folder + "/requirements.txt"

    requirements = []

    if os.path.isfile(requirements_path):
        with open(requirements_path) as f:
            requirements = [r for r in f.read().splitlines() if not is_comment(r)]

        print("Found the following requirements:")
        for r in requirements:
            print("* " + r)
    return requirements


def read_version() -> str:
    project_path = pathlib.Path(__file__).parent
    version_path = project_path / project_path.name / "_version.py"

    with version_path.open("rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("__version__ = "):
                version = line.split("=", 1)[-1].strip()
                version = version[1:-1]  # Remove quotation characters
                return version

    raise RuntimeError("Unable to find __version__ string")


setuptools.setup(
    name="ptdeco",
    version=read_version(),
    packages=["ptdeco"],
    install_requires=read_requirements(),
    package_data={
        "ptdeco": ["py.typed"],
    },
)
