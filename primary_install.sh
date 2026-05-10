#!/bin/bash
###############################################################################
# primary_install.sh -- one-shot ASAP installer into a fresh conda env.
#
# Use this script when you want a clean, isolated conda environment for ASAP
# from scratch. For a lighter "install ASAP into the env I already have
# active" path, use install.sh (or `python -m pip install .`) instead.
#
# What this script does:
#   1. Bootstraps `conda` into the current non-interactive shell (necessary
#      because `bash primary_install.sh` doesn't inherit shell functions like
#      `conda activate`).
#   2. Creates the conda environment named in $ENV_NAME (default: "asap"),
#      reusing it if it already exists. Pass `-f` for a forced fresh rebuild.
#   3. Installs ASAP's runtime dependencies via conda-forge (dscribe>=2.0,
#      scipy, scikit-learn, ase, umap-learn, pyyaml, tqdm, pandas, click) so
#      they can pull in modern versions cleanly.
#   4. Installs ASAP itself via `pip install --no-deps .` -- since every
#      runtime dep is already satisfied above, --no-deps avoids any second
#      resolver pass.
#
# Why this exists in the repo:
#   `setup.py` only handles step (4). On a brand-new machine you also need
#   steps (1)-(3); this script wraps all of that into one command so a new
#   user can go from "fresh login" to "working `asap` CLI" without reading
#   the README.
#
# Assumptions:
#   - You're running this from the ASAP project root (the directory holding
#     setup.py and this script).
#   - The machine can reach the conda-forge channel.
#   - `conda` is installed somewhere standard (PATH, $HOME/miniforge3,
#     $HOME/miniconda3, /opt/conda) -- if not, set CONDA_BASE manually or
#     load your cluster's conda module before running.
#
# Usage:
#   bash primary_install.sh         # create env if missing, otherwise reuse
#   bash primary_install.sh -f      # remove existing env first, then create
#
# Output:
#   All stdout+stderr is captured to log.primary_install in the cwd, so you
#   can re-read the install transcript without scrolling back through the
#   terminal.
###############################################################################

# Mirror everything to a log file. Final terminal output is just whatever
# `tail log.primary_install` shows, but the log preserves the full transcript.
exec > log.primary_install 2>&1

# Fail fast on any error. Note: NOT using `set -u` because some conda
# activate.d hooks reference variables (e.g. INCLUDE) before defining them.
set -e

echo "ASAP primary install"
echo "  cwd            : $(pwd)"
echo "  hostname       : $(hostname)"
echo "  date           : $(date)"
echo

#-----------------------------------------------------------------------------
# Step 1: bootstrap conda into this shell.
#
# A non-interactive `bash <script>` shell does not have the `conda` shell
# function loaded, so `conda activate` would fail with "Run 'conda init'
# before 'conda activate'". We explicitly source conda's profile.d hook to
# make activation work.
#-----------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    for cand in \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        /opt/conda/etc/profile.d/conda.sh; do
        if [ -f "$cand" ]; then
            echo "Sourcing conda from: $cand"
            # shellcheck disable=SC1090
            source "$cand"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH and no standard install location matched."
    echo "       Either install conda, set CONDA_BASE, or run your cluster's"
    echo "       conda module load before invoking this script."
    exit 1
fi
# Source the hook from whatever conda we just found. Idempotent if the
# previous loop already sourced one.
source "$(conda info --base)/etc/profile.d/conda.sh"

#-----------------------------------------------------------------------------
# Step 2: create / reuse the target conda environment.
#-----------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-asap}"   # override with `ENV_NAME=foo bash primary_install.sh`

fresh_install=0
if [ "${1:-}" = "-f" ]; then
    fresh_install=1
fi

if [ "$fresh_install" -eq 1 ]; then
    echo "Fresh install requested (-f). Removing existing '${ENV_NAME}' env if present."
    conda env remove -n "${ENV_NAME}" -y || true
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists; reusing."
else
    echo "Creating the '${ENV_NAME}' conda environment."
    conda create -n "${ENV_NAME}" -y -c conda-forge
fi

echo "Activating '${ENV_NAME}' ..."
conda activate "${ENV_NAME}"

#-----------------------------------------------------------------------------
# Step 3: install ASAP's runtime dependencies via conda-forge.
#
# We let conda's solver pick the latest mutually-compatible versions instead
# of pinning -- the codebase is now NumPy-2 / dscribe-2 ready (see commit
# "Modernize for NumPy 2.x / Python 3.10+ / dscribe 2.x"), so old version
# ceilings are no longer needed.
#-----------------------------------------------------------------------------
echo "Installing ASAP runtime dependencies (conda-forge) ..."
conda install -y -c conda-forge "dscribe>=2.0,<3"
conda install -y -c conda-forge "click>=7.0"
conda install -y -c conda-forge scipy scikit-learn ase umap-learn pyyaml tqdm pandas

#-----------------------------------------------------------------------------
# Step 4: install ASAP itself.
#
# --no-deps because every runtime dep is already installed via conda above.
# Skipping pip's resolver here avoids the resolver doing a redundant pass and
# potentially upgrading/downgrading conda-installed packages.
#-----------------------------------------------------------------------------
echo "Installing ASAP into '${ENV_NAME}' ..."
python -m pip install --no-deps .

echo
echo "Installation finished. To use this environment in a new shell:"
echo "    conda activate ${ENV_NAME}"
echo "    asap --help"
