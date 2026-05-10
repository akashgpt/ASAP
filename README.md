<p align="left">
  <img src="ASAP-logo.png" width="200" title="logo">
</p>

# ASAP
Automatic Selection And Prediction tools for materials and molecules

[![DOI](https://zenodo.org/badge/201763628.svg)](https://zenodo.org/badge/latestdoi/201763628)

> **`ALCHEMY` branch — modernized for NumPy 2.x / Python 3.10+ / dscribe 2.x.**
>
> This branch lives at <https://github.com/akashgpt/ASAP/tree/ALCHEMY>
> and is a near-drop-in replacement for upstream
> [`BingqingCheng/ASAP@master`](https://github.com/BingqingCheng/ASAP/tree/master).
>
> Key differences vs. upstream:
> - `np.complex_` &rarr; `np.complex128` (removed in NumPy 2.0).
> - `collections.Iterable` &rarr; `collections.abc.Iterable` (removed in Python 3.10).
> - `np.hstack(generator)` &rarr; `np.hstack([list])` (NumPy 1.20+ deprecation).
> - `ASAPXYZ.get_descriptors`: typed `except` handlers instead of bare `except:`.
> - SOAP: maps the legacy `crossover=` boolean to dscribe 2.x's
>   `compression={"mode": ...}` API. Works against `dscribe 2.0.x` *and*
>   `dscribe >= 2.1` (where the original `crossover=` kwarg was removed).
> - LMBTR_K2 / LMBTR_K3 currently raise `NotImplementedError` &mdash; the
>   underlying `k2=`/`k3=` API was removed in dscribe 2.x and the wrappers
>   need a port to the new `geometry`/`grid`/`weighting` schema. **Use SOAP
>   or ACSF**, or pin `dscribe<2` if LMBTR is required.
> - `setup.py`: dropped hard upper-bounds on `numpy`/`scipy`/`scikit-learn`/
>   `ase`/`matplotlib`; the floor is now `dscribe>=2.0,<3`.
> - `install.sh`: switched from the deprecated `python3 setup.py install --user`
>   to `python -m pip install .`.
> - Added `primary_install.sh`: one-shot conda env + ASAP installer for
>   green-field setups (see "Installation &amp; requirements" below).
>
> Verified end-to-end on a 4001-frame, 360-atom He/MgSiO3 trajectory: SOAP
> descriptors agree with the legacy `dscribe 1.2.2` path to ~1e-13 and FPS
> frame selection is bit-identical.

### [Documentation](https://bingqingcheng.github.io/index.html) (in progress)

### Mapping Materials and Molecules [(Paper link)](https://pubs.acs.org/doi/full/10.1021/acs.accounts.0c00403)

Cheng B, Griffiths RR, Wengert S, Kunkel C, Stenczel T, Zhu B, Deringer VL, Bernstein N, Margraf JT, Reuter K, Csanyi G. Mapping Materials and Molecules. Accounts of Chemical Research. 2020 Aug 14:12697-705.

This tutorial style paper covers:

* A concise summary of the theory of representing chemical environments, an a simple yet practical conceptual approach for generating structure maps in a generic and automated manner. 

* Several illustrative examples on mapping material and chemical datasets, including crystalline and amorphous materials, interfaces, and organic molecules. The datasets of these examples are in this [repo](https://github.com/BingqingCheng/Mapping-the-space-of-materials-and-molecules).

* Snippets of `asap` commands that were used to analyze the examples and make figures. 

### Basic usage

Type `asap` and use the sub-commands for various tasks.

To get help string:

`asap --help` .or. `asap subcommand --help` .or. `asap subcommand subcommand --help` depending which level of help you are interested in.

* `asap gen_desc`: generate global or atomic descriptors based on the input [ASE](https://wiki.fysik.dtu.dk/ase/ase/atoms.html)) xyze file. 

* `asap map`: make 2D plots using the specified design matrix. Currently PCA `pca`, sparsified kernel PCA `skpca`, UMAP `umap`, and t-SNE `tsne` are implemented. 

* `asap cluster`: perform density based clustering. Currently supports DBSCAN `dbscan` and [Fast search of density peaks](https://science.sciencemag.org/content/344/6191/1492) `fdb`.

* `asap fit`: fast fit ridge regression `ridge` or sparsified kernel ridge regression model `kernelridge` based on the input design matrix and labels.

* `asap kde`: quick kernel density estimation on the design matrix. Several versions of kde available.

* `asap select`: select a subset of frames using sparsification algorithms.

### Quick & basic example

#### Step 1: generate a design matrix

The first step for a machine-learning analysis or visualization is to generate a "design matrix" made from either global descriptors or atomic descriptors. To do this, we supply `asap gen_desc` with an input file that contains the atomic coordintes. Many formats are supported; anything can be read using [ase.io](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) is supported. You can use a wildcard to specify the list of input files that matches the pattern (e.g. `POSCAR*`, `H*`, or `*.cif`). However, it is most robust if you use an extended xyz file format (units in angstrom, additional info and cell size in the comment line).

As a quick example, in the folder ./tests/

to generate SOAP descriptors:

```bash
asap gen_desc --fxyz small_molecules-1000.xyz soap
```

for columb matrix:

```bash
asap gen_desc -f small_molecules-1000.xyz --no-periodic cm
```

#### Step 2: generate a low-dimensional map

After generating the descriptors, one can make a two-dimensional map (`asap map`), or regression model (`asap fit`), or clustering (`asap cluster`), or select a subset of frames (`asap select`), or do a clustering analysis (`asap cluster`), or estimate the probablity of observing each sample (`asap kde`).

For instance, to make a pca map:

```bash
asap map -f small_molecules-SOAP.xyz -dm '[SOAP-n4-l3-c1.9-g0.23]' -c dft_formation_energy_per_atom_in_eV pca
```

You can specify a list of descriptor vectors to include in the design matrix, e.g. `'[SOAP-n4-l3-c1.9-g0.23, SOAP-n8-l3-c5.0-g0.3]'`

one can use a wildcard to specify the name of all the descriptors to use for the design matrix, e.g.

```bash
asap map -f small_molecules-SOAP.xyz -dm '[SOAP*]' -c dft_formation_energy_per_atom_in_eV pca
```

or even

```bash
asap map -f small_molecules-SOAP.xyz -dm '[*]' -c dft_formation_energy_per_atom_in_eV pca
```

#### Step 2+: interactive visualization

Using `asap map`, a png figure is generated. In addition, the code also output the low-dimensional coordinates of the structures and/or atomic environments. The default output is extended xyz file. One can also specify a different output format using `--output` or `-o` flag. and the available options are `xyz`, `matrix` and `chemiscope`. 

* If one select `chemiscope` format, a `*.json.gz` file will be writen, which can be directly used as the input of [chemiscope](https://github.com/cosmo-epfl/chemiscope)

* If the output is in `xyz` format, it can be visualized interactively using [projection_viewer](https://github.com/chkunkel/projection_viewer).

### Installation & requirements

This branch supports **Python 3.10+** and is verified against **NumPy 2.x**
and **dscribe 2.0–2.1**. Older Python 3.7–3.9 may still work but is no
longer the target.

#### Option A — clone this branch (recommended for the modernized stack)

```bash
git clone -b ALCHEMY https://github.com/akashgpt/ASAP.git
cd ASAP
```

Then choose **one** of the two install paths below.

##### A1. Lightweight install into an environment you already have

If you already have an active conda or virtualenv with the runtime deps
installed, just:

```bash
bash install.sh        # equivalent to `python -m pip install .`
```

This is the right choice when you're iterating on the source and only
need to re-install ASAP itself.

##### A2. One-shot conda env + ASAP installer (green-field)

For a clean conda environment from scratch:

```bash
bash primary_install.sh        # creates env "asap", installs deps + ASAP
bash primary_install.sh -f     # same, but force-rebuild if env exists
ENV_NAME=foo bash primary_install.sh   # or pick a different env name
```

This script:
1. Bootstraps `conda` into the non-interactive shell.
2. Creates / reuses the target conda environment.
3. Installs ASAP's runtime dependencies via `conda-forge`.
4. Installs ASAP itself with `pip install --no-deps .`.

All output is mirrored to `log.primary_install`.

#### Option B — install from the upstream PyPI release

The `asaplib` package on PyPI is from the upstream
[`BingqingCheng/ASAP`](https://github.com/BingqingCheng/ASAP) repository
and **does not include the modernization patches in this branch**. It pins
old versions of NumPy/SciPy/scikit-learn/dscribe; use it only if you need
the original behaviour.

```bash
pip install asaplib
```

#### Runtime dependencies

`setup.py` declares these and resolves the latest mutually-compatible
versions when installed via `pip` or `conda`:

`dscribe>=2.0,<3`, `click>=7.0`, `numpy`, `scipy`, `scikit-learn`,
`ase`, `umap-learn`, `PyYAML`, `tqdm`, `pandas`.

The lower bound on `dscribe` is hard:

- `dscribe 1.x` is incompatible with NumPy 2.x.
- `dscribe 2.x` renamed SOAP/ACSF kwargs (`rcut`/`nmax`/`lmax` &rarr;
  `r_cut`/`n_max`/`l_max`) and replaced the SOAP `crossover=` boolean with
  `compression={"mode": ...}`. This branch maps the old API internally so
  existing user configs keep working.

#### Add-Ons (optional)

- (for finding symmetries of crystals) `spglib`
- (for annotation without overlaps) `adjustText`
- The FCHL19 representation requires the development branch of the QML
  package — see <https://www.qmlcode.org/installation.html>.

#### Known limitation in this branch

The LMBTR wrappers (`Atomic_Descriptor_LMBTR_K2`, `_K3`) raise
`NotImplementedError` on instantiation: the dscribe 2.x rewrite of the
LMBTR API (`geometry` / `grid` / `weighting` dicts in place of `k2=`/`k3=`)
has not been ported. **Use SOAP or ACSF**, or pin `dscribe<2` and use
upstream master, if LMBTR is required.

### Additional tools
In the directory ./scripts/ you can find a selection of other python tools.

### Tab completion
Tab completion can be enabled by sourcing the `asap_completion.sh` script in the ./scripts/ directory. 
If a conda environment is used, you can copy this file to `$CONDA_PREFIX/etc/conda/activate.d/` to automatically load the completion upon environment activation.
