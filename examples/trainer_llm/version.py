# This is parsed by setup.py, so we need to stick to str -> int parsing
__version__ = "0.0.28"

_ver_major = int(__version__.split(".")[0])
_ver_minor = int(__version__.split(".")[1])
_ver_patch = int(__version__.split(".")[2])

__version_info__ = _ver_major, _ver_minor, _ver_patch
