"""Lossless, uncompressed NumPy handoff for global frame descriptors."""

import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Sequence
from zipfile import BadZipFile

import numpy as np

if TYPE_CHECKING:
	from .xyz import ASAPXYZ


def read_descriptor_bundle(filename: str | Path) -> dict[str, np.ndarray]:
	"""Read and validate a global-descriptor archive without allowing pickles.

	Args:
		filename: NPZ archive produced by write_descriptor_bundle.

	Returns:
		Validated arrays, detached from the closed archive.

	Raises:
		ValueError: The archive is malformed.
	"""
	try:
		with np.load(filename, allow_pickle=False) as archive:
			bundle = {name: archive[name] for name in archive.files}
		_validate_descriptor_bundle(bundle)
		return bundle
	except (OSError, KeyError, TypeError, ValueError, EOFError, BadZipFile) as exc:
		raise ValueError(f"Cannot read descriptor bundle {filename}: {exc}") from exc


def _validate_descriptor_bundle(bundle: dict[str, np.ndarray]) -> None:
	"""Check the version-one array schema used by the writer and reader."""
	required = {"format_version", "descriptors", "descriptor_keys", "descriptor_offsets",
		"species", "cells", "frame_indices", "source_stride", "default_descriptor_keys"}
	if not required.issubset(bundle) or bundle["format_version"].shape != () or bundle["format_version"].dtype.kind not in "iu" or bundle["format_version"].item() != 1:
		raise ValueError("missing fields or unsupported format version")
	matrix = bundle["descriptors"]
	keys = bundle["descriptor_keys"]
	offsets = bundle["descriptor_offsets"]
	if matrix.ndim != 2:
		raise ValueError("descriptors must be a matrix")
	nframes = len(matrix)
	if matrix.dtype != np.dtype("float64") or not nframes or not matrix.shape[1] or not np.isfinite(matrix).all():
		raise ValueError("descriptors must be a nonempty finite float64 matrix")
	if keys.ndim != 1 or keys.dtype.kind != "U" or len(set(keys)) != len(keys) or not len(keys):
		raise ValueError("descriptor keys must be unique strings")
	default_keys = bundle["default_descriptor_keys"]
	if default_keys.ndim != 1 or default_keys.dtype.kind != "U" or not len(default_keys) or any(key not in keys for key in default_keys):
		raise ValueError("invalid default descriptor keys")
	if offsets.dtype.kind not in "iu" or offsets.shape != (len(keys) + 1,) or offsets[0] != 0 or offsets[-1] != matrix.shape[1] or np.any(offsets[1:] <= offsets[:-1]):
		raise ValueError("invalid descriptor column offsets")
	if bundle["cells"].shape != (nframes, 3, 3) or bundle["cells"].dtype != np.dtype("float64") or not np.isfinite(bundle["cells"]).all():
		raise ValueError("invalid cell matrices")
	indices = bundle["frame_indices"]
	if indices.shape != (nframes,) or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
		raise ValueError("invalid frame order")
	species = bundle["species"]
	if species.ndim != 1 or species.dtype.kind not in "iu" or not len(species) or np.any(species < 1) or np.any(species > 118) or np.any(species[1:] <= species[:-1]):
		raise ValueError("invalid atomic species")
	if bundle["source_stride"].shape != () or bundle["source_stride"].dtype.kind not in "iu" or bundle["source_stride"].item() < 1:
		raise ValueError("invalid source stride")


def descriptor_bundle_matrix(bundle: dict[str, np.ndarray], descriptor_names: Sequence[str]) -> np.ndarray:
	"""Select descriptor columns with the existing ASAP wildcard convention.

	Args:
		bundle: Validated result of read_descriptor_bundle.
		descriptor_names: Keys, or wildcard patterns such as ['*'].

	Returns:
		The requested global descriptor matrix in key request order.

	Raises:
		ValueError: A requested descriptor is missing.
	"""
	keys = bundle["descriptor_keys"].tolist()
	selected = []
	for name in descriptor_names:
		if name == "*":
			matches = bundle["default_descriptor_keys"].tolist()
		else:
			matches = [key for key in keys if re.search(name.replace("*", ".+"), key)] if "*" in name else [name]
		if not matches or any(key not in keys for key in matches):
			raise ValueError(f"No descriptor matching {name!r}")
		selected.extend(matches)
	offsets = bundle["descriptor_offsets"]
	return np.hstack([bundle["descriptors"][:, offsets[keys.index(key)]:offsets[keys.index(key) + 1]] for key in selected])


def write_descriptor_bundle(asapxyz: "ASAPXYZ", filename: str) -> None:
	"""Write lossless global descriptors and frame identity to one NPZ file.

	Args:
		asapxyz: Structures and their computed global descriptors.
		filename: Output prefix; '.npz' is appended.

	Raises:
		ValueError: Frame metadata cannot form consistent numeric columns.
	"""
	import tempfile

	# Use the same field insertion and extraction as the XYZ path.
	# Numeric metadata (such as Time) stays available by explicit key; '*' uses
	# the existing descriptor-only matching and ordering recorded below.
	for i, frame in enumerate(asapxyz.frames):
		asapxyz._write_computed_descriptors_to_xyz(asapxyz.global_desc[i], frame)
	keys = list(asapxyz.frames[0].info)
	if not keys or any(list(frame.info) != keys for frame in asapxyz.frames):
		raise ValueError("Binary output requires consistent numeric frame info keys")
	for frame in asapxyz.frames:
		for key in keys:
			values = np.asarray(frame.info[key])
			if values.dtype.kind not in "biuf":
				raise ValueError(f"Binary output requires real numeric frame info: {key}")
			if values.dtype.kind in "iu" and any(int(value) != int(float(value)) for value in values.flat):
				raise ValueError(f"Integer frame info is not exactly representable as float64: {key}")
	matrix, _ = asapxyz.get_descriptors(keys)
	matrix = np.asarray(matrix, dtype=np.float64)
	offsets = np.cumsum([0] + [np.asarray(asapxyz.frames[0].info[key]).size for key in keys], dtype=np.int64)
	payload = {
		"format_version": np.asarray(1, dtype=np.int64),
		"descriptors": matrix,
		"descriptor_keys": np.asarray(keys),
		"descriptor_offsets": offsets,
		"default_descriptor_keys": np.asarray(asapxyz._desc_name_with_wild_card(["*"])),
		"species": np.asarray(asapxyz.global_species, dtype=np.int64),
		"cells": np.asarray([frame.cell.array for frame in asapxyz.frames], dtype=np.float64),
		"frame_indices": np.arange(asapxyz.nframes, dtype=np.int64) * asapxyz.stride,
		"source_stride": np.asarray(asapxyz.stride, dtype=np.int64),
	}
	_validate_descriptor_bundle(payload)
	# Publish only a complete archive. The caller must check generation success
	# and remove previous outputs before starting a replacement calculation.
	destination = str(filename) + ".npz"
	staged = None
	try:
		with tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(destination)), suffix=".npz", delete=False) as handle:
			staged = handle.name
			np.savez(handle, **payload)
		os.replace(staged, destination)
	finally:
		if staged is not None and os.path.exists(staged):
			os.unlink(staged)
