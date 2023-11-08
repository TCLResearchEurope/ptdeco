SHELL=bash

MODULE_NAME=ptdeco
VER_FILE=version.py

PYTHON_LIB_FILES= configurator.py datasets_dali.py builder.py version.py \
			      run_decompose_falor.py run_decompose_lockd.py run_finetune.py

PYTHON_RUN_FILE=run.py

PYTHON_FILES=$(PYTHON_LIB_FILES) $(PYTHON_RUN_FILE)


check:
	-isort --check --profile=black $(PYTHON_FILES)
	-black --check  $(PYTHON_FILES)
	-flake8 --max-line-length 88  $(PYTHON_FILES)

mypy:
	-mypy --color-output --disallow-incomplete-defs --disallow-untyped-defs  $(PYTHON_FILES)


commit:
	git commit -m "[trainer] " -e


vercommit:
	git restore --staged $(VER_FILE)
	git checkout $(VER_FILE)
	../../aux/update_version.py --version-segment=2 --version-file=$(VER_FILE)
	git add $(VER_FILE)
	VER=$$(../../aux/get_version.py --version-file=$(VER_FILE)) && git commit -m "[trainer] $${VER} " -e

manualvercommit:
	git add $(VER_FILE)
	VER=$$(../../aux/get_version.py --version-file=$(VER_FILE)) && git commit -m "[trainer] $${VER}" -e


build:
	@echo "Build - started"
	@mkdir -p build
	@cp -rp ../../ptdeco .
	@cp $(PYTHON_RUN_FILE)  __main__.py
		ZIP_NAME="build/ptdeco-$$(git rev-list --count HEAD)-$$(git rev-parse --short HEAD).zip"; \
		zip -r $${ZIP_NAME} $(PYTHON_LIB_FILES) __main__.py ptdeco; \
		rm -rf ptdeco __main__.py; \
		echo "Build - created $${ZIP_NAME}"


.PHONY: check build
