import setuptools  # type: ignore

PACKAGE_NAME = "ptdeco"


def read_version() -> str:
    with open(f"src/{PACKAGE_NAME}/_version.py", "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("__version__ = "):
                version = line.split("=", 1)[-1].strip()
                version = version[1:-1]  # Remove quotation characters
                return version

    raise RuntimeError("Unable to find __version__ string")


setuptools.setup(
    version=read_version(),
)
