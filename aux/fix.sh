#!/bin/bash


INPUTS=${*}


function fix() {
    isort --profile="black" ${ISORT_FLAGS} $1
    black $1
}


for I in ${INPUTS}; do
    if [ -f ${I} ]; then
        if [[ "${I}" == *.py ]]; then
            fix ${I}
        else
            echo "Not Python file, skipping ${I}"
        fi
    elif [ -d ${I} ]; then
        PY_FILES=$(find ${I} -name "*.py")
        for F in ${PY_FILES}; do
            fix ${F}
        done
    fi
done
