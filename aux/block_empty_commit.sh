#!/bin/bash

if git diff --cached --quiet; then
    echo "No changes staged in git"
    exit 1
fi
