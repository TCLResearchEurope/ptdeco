MODULE_NAME=ptdeco

PY_DIRS=src/ptdeco tests setup.py

PY_MYPY_FLAKE8=src/ptdeco tests setup.py

FILES_TO_CLEAN=src/ptdeco.egg-info dist

include Makefile.inc
