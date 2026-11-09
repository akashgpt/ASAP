"""Check lossless handoff, CLI compatibility, and malformed archive rejection."""

from pathlib import Path
import random
from unittest.mock import patch

from ase import Atoms
from ase.io import write
from click.testing import CliRunner
import numpy as np
import pytest

from asaplib.cli.cmd_asap import asap
from asaplib.data import ASAPXYZ, descriptor_bundle_matrix, read_descriptor_bundle, write_descriptor_bundle


def make_structures(path: Path) -> ASAPXYZ:
	"""Build two tiny frames with numeric time metadata and a triclinic cell."""
	frames = []
	for i in range(2):
		frame = Atoms("OH2", positions=[[0, 0, 0], [0.9, 0.1, 0], [0, 0.9, 0.1]],
			cell=[[4.123456789123456, 0, 0], [0.1, 4, 0], [0.2, 0.3, 4]], pbc=True)
		frame.info["Time"] = 25 * i
		frames.append(frame)
	write(path, frames)
	return ASAPXYZ(str(path))


def make_bundle(tmp_path: Path) -> tuple[dict[str, np.ndarray], ASAPXYZ]:
	"""Write synthetic descriptors whose last binary digits must survive."""
	structures = make_structures(tmp_path / "frames.xyz")
	for i in range(2):
		structures.global_desc[i] = {"soap": {"acronym": "SOAP-test", "descriptors":
			np.array([np.nextafter(1.0, 2.0) + i, -0.0, 1e-200, 1e200])}}
	write_descriptor_bundle(structures, str(tmp_path / "binary"))
	return read_descriptor_bundle(tmp_path / "binary.npz"), structures


def test_exact_roundtrip_and_legacy_keys(tmp_path: Path) -> None:
	"""Preserve exact float64 bits, metadata/key order, cells and species."""
	bundle, structures = make_bundle(tmp_path)
	matrix = descriptor_bundle_matrix(bundle, ["*"])
	expected, _ = structures.get_descriptors(["*"])
	assert np.array_equal(matrix.view(np.uint64), expected.view(np.uint64))
	assert bundle["descriptor_keys"].tolist() == ["Time", "SOAP-test"]
	assert descriptor_bundle_matrix(bundle, ["Time"]).ravel().tolist() == [0, 25]
	assert bundle["descriptor_offsets"].tolist() == [0, 1, 5]
	assert bundle["species"].tolist() == [1, 8]
	assert bundle["frame_indices"].tolist() == [0, 1]
	assert np.array_equal(bundle["cells"], [frame.cell.array for frame in structures.frames])
	assert np.array_equal(descriptor_bundle_matrix(bundle, ["SOAP*"]), expected)
	with np.load(tmp_path / "binary.npz", allow_pickle=False) as archive:
		assert all(archive[name].dtype.kind != "O" for name in archive.files)
	with pytest.raises(ValueError, match="No descriptor"):
		descriptor_bundle_matrix(bundle, ["missing"])


@pytest.mark.parametrize("field,value", [
	("format_version", np.asarray(2)),
	("descriptors", np.ones((2, 5), dtype=np.float32)),
	("descriptors", np.asarray([1.0])),
	("descriptors", np.full((2, 5), np.nan)),
	("descriptor_offsets", np.asarray([0, 1, 4])),
	("descriptor_offsets", np.asarray([0, 6, 5], dtype=np.uint64)),
	("cells", np.zeros((1, 3, 3))),
	("frame_indices", np.asarray([1, 0])),
	("frame_indices", np.asarray([1, 0], dtype=np.uint64)),
	("species", np.asarray([8, 1])),
	("descriptor_keys", np.asarray(["Time", "Time"])),
])
def test_reject_malformed_schema(tmp_path: Path, field: str, value: np.ndarray) -> None:
	"""Malformed shape/type/identity must fail instead of changing selection."""
	bundle, _ = make_bundle(tmp_path)
	bundle[field] = value
	np.savez(tmp_path / "malformed.npz", **bundle)
	with pytest.raises(ValueError, match="Cannot read descriptor bundle"):
		read_descriptor_bundle(tmp_path / "malformed.npz")


def test_missing_corrupt_and_object_archives(tmp_path: Path) -> None:
	"""Absent, truncated and pickled payloads cannot be mistaken for valid data."""
	with pytest.raises(ValueError):
		read_descriptor_bundle(tmp_path / "missing.npz")
	(tmp_path / "corrupt.npz").write_bytes(b"PK\x03\x04truncated")
	with pytest.raises(ValueError):
		read_descriptor_bundle(tmp_path / "corrupt.npz")
	np.savez(tmp_path / "object.npz", format_version=np.asarray({"x": 1}))
	with pytest.raises(ValueError):
		read_descriptor_bundle(tmp_path / "object.npz")
	np.savez(tmp_path / "empty.npz", descriptors=np.ones((1, 1)))
	with pytest.raises(ValueError):
		read_descriptor_bundle(tmp_path / "empty.npz")


def test_failed_publication_preserves_previous_archive(tmp_path: Path) -> None:
	"""A failed atomic replacement leaves the previous complete file intact."""
	_, structures = make_bundle(tmp_path)
	before = (tmp_path / "binary.npz").read_bytes()
	with patch("asaplib.data.descriptor_bundle.os.replace", side_effect=OSError("test failure")):
		with pytest.raises(OSError):
			write_descriptor_bundle(structures, str(tmp_path / "binary"))
	assert (tmp_path / "binary.npz").read_bytes() == before
	assert sorted(path.name for path in tmp_path.iterdir()) == ["binary.npz", "frames.xyz"]


@pytest.mark.parametrize("time", [2**53 + 1, 2**70])
def test_reject_lossy_integer_metadata(tmp_path: Path, time: int) -> None:
	"""Large integer time metadata must fail instead of silently rounding."""
	_, structures = make_bundle(tmp_path)
	structures.frames[0].info["Time"] = time
	with pytest.raises(ValueError, match="frame info"):
		write_descriptor_bundle(structures, str(tmp_path / "lossy"))
	assert not (tmp_path / "lossy.npz").exists()


def test_cli_binary_matches_default_text(tmp_path: Path) -> None:
	"""Tiny real SOAP uses identical arrays while the default still writes XYZ."""
	make_structures(tmp_path / "frames.xyz")
	runner = CliRunner()
	common = ["gen_desc", "--fxyz", str(tmp_path / "frames.xyz")]
	soap = ["soap", "--crossover", "-e", "-c", "2", "-n", "2", "-l", "2", "-g", "0.44"]
	random.seed(17)
	text = runner.invoke(asap, common + ["--prefix", str(tmp_path / "text")] + soap)
	assert text.exit_code == 0, text.exception
	random.seed(17)
	binary = runner.invoke(asap, common + ["--prefix", str(tmp_path / "binary"), "--output-format", "npz"] + soap)
	assert binary.exit_code == 0, binary.exception
	legacy = ASAPXYZ(str(tmp_path / "text.xyz"))
	matrix, _ = legacy.get_descriptors(["*"])
	bundle = read_descriptor_bundle(tmp_path / "binary.npz")
	assert np.array_equal(matrix, descriptor_bundle_matrix(bundle, ["*"]))
	assert np.array_equal(bundle["cells"], [frame.cell.array for frame in legacy.frames])
	assert (tmp_path / "text-state.yaml").read_bytes() == (tmp_path / "binary-state.yaml").read_bytes()
	assert not (tmp_path / "binary.xyz").exists()
	assert not (tmp_path / "text.npz").exists()
	peratom = runner.invoke(asap, common + ["--prefix", str(tmp_path / "peratom"), "--output-format", "npz"] + soap + ["--peratom"])
	assert peratom.exit_code != 0
	assert not (tmp_path / "peratom.npz").exists()
