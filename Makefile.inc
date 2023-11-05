
VER_FILE=$(MODULE_NAME)/_version.py

F=set_it_to_reformated_file_from_cli

all: check test

check:
	@-black --color --check $(PY_DIRS) > make.black.log 2>&1 && \
        	echo "PASSED black" || \
        	(echo "FAILED black"; cat make.black.log)
	@-isort --check --profile=black $(ISORT_FLAGS) $(PY_DIRS) && echo "PASSED isort"
	@-(flake8  --color=always --ignore=E203,W503 --max-line-length 88 $(PY_MYPY_FLAKE8) > make.flake8.log 2>&1) && \
        	echo "PASSED flake8" || \
        	(echo "FAILED flake8"; cat make.flake8.log)
	@-MYPY_FORCE_COLOR=1 mypy --color-output --disallow-incomplete-defs --disallow-untyped-defs $(PY_MYPY_FLAKE8) > make.mypy.log 2>&1 && \
        	echo "PASSED mypy" || \
        	(echo "FAILED mypy"; cat make.mypy.log)


test:
	pytest -vv


clean:
	rm -rf make.black.log make.flake8.log make.mypy.log $(FILES_TO_CLEAN)

fix:
	ISORT_FLAGS="$(ISORT_FLAGS)" ./aux/fix.sh $(F)

fixall:
	ISORT_FLAGS="$(ISORT_FLAGS)" ./aux/fix.sh $(PY_MYPY_FLAKE8)

commit:
	git commit -e


vercommit:
	git restore --staged $(VER_FILE)
	git checkout $(VER_FILE)
	./aux/update_version.py --version-segment=2 --version-file=$(VER_FILE)
	git add $(VER_FILE)
	VER=$$(./aux/get_version.py --version-file=$(VER_FILE)) && git commit -m "$${VER}" -e


manualvercommit:
	git add $(VER_FILE)
	VER=$$(./aux/get_version.py --version-file=$(VER_FILE)) && git commit -m "$${VER}" -e


.PHONY: all check test clean fix fixall
