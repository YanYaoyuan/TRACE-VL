"""Mirror mapping must change URLs only, retaining all versions, hashes and markers."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("setup_mirror", Path(__file__).parents[1] / "scripts/setup_mirror.py")
mirror = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mirror)


def lock():
    return '''version = 1
[[package]]
name = "peft"
version = "0.21.2"
source = { registry = "https://pypi.org/simple" }
wheels = [{ url = "https://files.pythonhosted.org/packages/example.whl", hash = "sha256:abcd" }]
resolution-markers = ["python_full_version < '3.11'"]
[[package]]
name = "torch"
version = "2.8.0+cu128"
source = { registry = "https://download.pytorch.org/whl/cu128" }
wheels = [{ url = "https://download-r2.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl", hash = "sha256:1234" }]
[[package]]
name = "torchvision"
version = "0.23.0+cu128"
source = { registry = "https://download.pytorch.org/whl/cu128" }
wheels = [{ url = "https://download-r2.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl", hash = "sha256:5678" }]
'''


def test_mirror_retains_versions_hashes_and_markers():
    original = lock()
    transformed = mirror.remap(original)
    versions = mirror.verify_lock(original, transformed)
    assert versions["torch"] == ["2.8.0+cu128"]
    assert versions["peft"] == ["0.21.2"]
    assert 'hash = "sha256:abcd"' in transformed
    assert "python_full_version < '3.11'" in transformed
    assert mirror.CUDA_WHEELS["torch"] in transformed
    assert "files.pythonhosted.org" not in transformed and "download.pytorch.org" not in transformed
    assert mirror.PYPI in original


@pytest.mark.parametrize("before,after", [
    ("0.21.2", "0.22.0"),
    ("sha256:abcd", "sha256:changed"),
    ("python_full_version < '3.11'", "python_full_version < '3.12'"),
    (mirror.CUDA, mirror.CANONICAL_CUDA),
])
def test_mirror_rejects_non_url_changes_and_official_fallback(before, after):
    original = lock()
    with pytest.raises(ValueError, match="URLs only"):
        mirror.verify_lock(original, mirror.remap(original).replace(before, after))


def test_unmapped_artifact_host_is_rejected():
    original = lock().replace("https://files.pythonhosted.org/packages/", "https://unexpected.invalid/")
    with pytest.raises(ValueError, match="Unmapped"):
        mirror.verify_lock(original, mirror.remap(original))


def test_real_canonical_lock_maps_to_aliyun():
    original = (Path(__file__).parents[1] / "uv.lock").read_text()
    transformed = mirror.remap(original)
    versions = mirror.verify_lock(original, transformed)
    assert versions["transformers"] == ["5.20.2", "5.22.1"]
    assert len(mirror.inventory(original)) == len(mirror.inventory(transformed))


def test_project_python_range_and_explicit_sources_remain():
    project = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    transformed = mirror.remap(project)
    assert 'requires-python = ">=3.10,<3.13"' in transformed
    assert 'explicit = true' in transformed
    assert mirror.CUDA in transformed and mirror.ALIYUN in transformed
