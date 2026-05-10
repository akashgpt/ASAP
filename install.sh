#!/bin/bash
# Lightweight installer: install ASAP into the *currently active*
# Python / conda environment using pip.
set -e
python -m pip install .
