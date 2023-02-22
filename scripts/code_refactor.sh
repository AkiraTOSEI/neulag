#!/bin/bash

python -m isort $1
python -m black $1
python -m flake8 $1 --show-source
python -m mypy $1
